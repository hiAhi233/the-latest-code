"""ResNet-101 视觉 patch 编码器（方案 §3.3、§5.3）。

``ImageNet`` 预训练 ResNet-101，删除 ``avgpool`` 与 ``fc``，输出 C5
``[B,2048,7,7]``（224 输入）。随后：

    C5 → AdaptiveAvgPool2d(3,3) → [B,2048,3,3]
       → flatten spatial          → [B,9,2048]
       → Linear(2048,256)+LayerNorm → [B,9,256]

每张切片得到 9 个 visual patch tokens。导出脚本只缓存这 256 维 tokens，
报告训练阶段不再运行 ResNet（方案 §7B）。
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "ResNetPatchEncoder",
           "slice_to_tensor", "state_dict_sha256"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_backbone(backbone: str, pretrained: bool, weights: str) -> tuple[nn.Module, int]:
    if backbone != "resnet101":
        raise ValueError(f"首版只支持 resnet101，收到 {backbone!r}")
    from torchvision.models import ResNet101_Weights, resnet101

    if pretrained:
        if not hasattr(ResNet101_Weights, weights):
            raise ValueError(f"未知 torchvision 权重 {weights!r}")
        model = resnet101(weights=getattr(ResNet101_Weights, weights))
    else:
        model = resnet101(weights=None)
    channels = int(model.fc.in_features)
    body = nn.Sequential(*list(model.children())[:-2])   # conv1 .. layer4
    return body, channels


class ResNetPatchEncoder(nn.Module):
    """冻结的 ResNet-101 + 3×3 patch 池化 + 2048→d 投影。

    ``proj_seed`` 固定 2048→d 投影层与 LayerNorm 的随机初始化。缓存里保存的是
    投影后的 256 维 tokens，因此**不同导出进程必须得到完全相同的投影**，否则
    B0/B1 缓存和 ``visual_tokens_null`` 之间不可比。同一 seed 下投影权重是确定的，
    其 SHA256 写入视觉缓存 meta 与身份链。
    """

    def __init__(self, backbone: str = "resnet101", pretrained: bool = True,
                 patch_grid: int = 3, output_dim: int = 256,
                 frozen: bool = True, weights: str = "IMAGENET1K_V1",
                 proj_seed: int = 0) -> None:
        super().__init__()
        if patch_grid < 1:
            raise ValueError("patch_grid 必须 >= 1")
        self.backbone_name = backbone
        self.patch_grid = int(patch_grid)
        self.output_dim = int(output_dim)
        self.frozen = bool(frozen)
        self.proj_seed = int(proj_seed)
        self.body, channels = _build_backbone(backbone, pretrained, weights)
        self.feature_dim = channels
        self.pool = nn.AdaptiveAvgPool2d((self.patch_grid, self.patch_grid))
        # 确定性初始化：固定 seed，保证任何进程导出的 tokens 完全一致
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.proj_seed)
            self.proj = nn.Linear(channels, self.output_dim)
            self.norm = nn.LayerNorm(self.output_dim)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
                             persistent=False)
        if self.frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            self.eval()

    @property
    def num_patches(self) -> int:
        return self.patch_grid ** 2

    # -- 前向 ------------------------------------------------------------- #
    def forward_c5(self, images: torch.Tensor) -> torch.Tensor:
        """``(B,3,H,W)`` (0..1) -> C5 feature map ``(B,2048,h,w)``。"""
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"期望 (B,3,H,W)，收到 {tuple(images.shape)}")
        x = (images.float() - self.mean) / self.std
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            return self.body(x)

    def pool_and_project(self, c5: torch.Tensor, project: bool = True) -> torch.Tensor:
        pooled = self.pool(c5)                       # (B,2048,p,p)
        flat = pooled.flatten(2).transpose(1, 2)     # (B,p²,2048)
        if not project:
            return flat
        return self.norm(self.proj(flat))            # (B,p²,d)

    def forward(self, images: torch.Tensor, project: bool = True) -> torch.Tensor:
        return self.pool_and_project(self.forward_c5(images), project=project)

    def train(self, mode: bool = True) -> "ResNetPatchEncoder":
        if self.frozen:
            # 冻结编码器永远保持 eval（BN 统计量固定）；直接走父类避免递归
            return super().train(False)
        return super().train(mode)

    def extra_repr(self) -> str:
        return (f"backbone={self.backbone_name}, patch_grid={self.patch_grid}, "
                f"output_dim={self.output_dim}, proj_seed={self.proj_seed}, "
                f"frozen={self.frozen}")


def slice_to_tensor(slice01: np.ndarray) -> torch.Tensor:
    """0..1 单通道 ``[H,W]`` -> ``[3,H,W]``（灰度复制三通道）。"""
    array = np.clip(np.asarray(slice01, dtype=np.float32), 0.0, 1.0)
    if array.ndim != 2:
        raise ValueError(f"切片必须是二维，收到 {array.shape}")
    tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
    return tensor


def state_dict_sha256(module: nn.Module) -> str:
    """按参数名排序后的确定性权重哈希（用于缓存身份，不依赖文件路径）。"""
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        array = state[key].detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def encoder_summary(encoder: ResNetPatchEncoder) -> dict[str, Any]:
    return {"backbone": encoder.backbone_name, "patch_grid": encoder.patch_grid,
            "num_patches": encoder.num_patches, "feature_dim": encoder.feature_dim,
            "output_dim": encoder.output_dim, "frozen": encoder.frozen,
            "proj_seed": encoder.proj_seed,
            "weights_sha256": state_dict_sha256(encoder.body),
            "projection_sha256": state_dict_sha256(
                nn.ModuleDict({"proj": encoder.proj, "norm": encoder.norm}))}
