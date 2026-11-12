"""ADM 式异常注意力增强（方案 §3.2、§5.2）。

只做一件事：把定位器的预测热图变成一张 0..1 的注意力图 ``A``，再用

    Iadm = I × [γ + (1-γ) × A]

增强原图。高异常区域基本保留（系数→1），正常背景缩放到 γ；不做硬裁剪，
因此脑室、中线、侧别和整体解剖仍然可见。异常置信度低于正常阈值时直接
返回原图（``normal_fallback: original``），避免把正常图整体压暗。

本模块是纯 NumPy 实现，不依赖 torch，便于在导出脚本和单元测试中复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "AdmParams", "normalize_heatmap", "resize_heatmap", "select_top_slices",
    "enhance_image", "render_adm_grid",
]


@dataclass(frozen=True)
class AdmParams:
    gamma: float = 0.4
    threshold: float = 0.6
    mode: str = "soft"
    normal_fallback: str = "original"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError(f"gamma 必须在 [0,1]，收到 {self.gamma}")
        if not 0.0 < float(self.threshold) <= 1.0:
            raise ValueError(f"threshold 必须在 (0,1]，收到 {self.threshold}")
        if self.mode not in ("soft", "hard"):
            raise ValueError(f"未知 ADM 模式 {self.mode!r}，只允许 soft/hard")
        if self.normal_fallback not in ("original", "none"):
            raise ValueError(f"未知 normal_fallback {self.normal_fallback!r}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_heatmap(heatmap: np.ndarray, method: str = "minmax",
                      eps: float = 1e-6) -> np.ndarray:
    """把任意范围的热图确定性地映射到 [0,1]。

    ``minmax``：``(h-min)/(max-min)``；``percentile``：先按 p1/p99 截断再归一化，
    对个别尖峰更稳健。热图全平（max-min < eps）时返回全 0，表示“没有注意力证据”。
    """
    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.ndim != 2:
        raise ValueError(f"热图必须是二维 [H,W]，收到 {heat.shape}")
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    if method == "percentile":
        low, high = np.percentile(heat, [1.0, 99.0])
    elif method == "minmax":
        low, high = float(heat.min()), float(heat.max())
    else:
        raise ValueError(f"未知归一化方法 {method!r}")
    if not np.isfinite(high - low) or (high - low) < eps:
        return np.zeros_like(heat, dtype=np.float32)
    return np.clip((heat - low) / max(float(high - low), eps), 0.0, 1.0).astype(np.float32)


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """双线性重采样到 ``(H, W)``；``size`` 与图像 ``shape[:2]`` 一致。"""
    heat = np.asarray(heatmap, dtype=np.float32)
    height, width = int(size[0]), int(size[1])
    if heat.shape != (height, width):
        heat = np.asarray(
            Image.fromarray(heat, mode="F").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32)
    return heat


def select_top_slices(score_per_slice: Sequence[float], k: int = 3,
                      min_score: float = 0.0) -> list[int]:
    """按切片异常分数选 top-k，返回升序 z，便于按解剖顺序排列 token。"""
    scores = np.asarray(list(score_per_slice), dtype=np.float32)
    if scores.size == 0:
        return []
    k = max(1, int(k))
    order = np.argsort(-scores)[:k]
    chosen = [int(z) for z in order if float(scores[z]) >= float(min_score)]
    if not chosen:
        chosen = [int(order[0])]
    return sorted(set(chosen))


def enhance_image(image: np.ndarray, heatmap: np.ndarray, params: AdmParams,
                  abnormal_score: float | None = None,
                  heat_norm: str = "minmax",
                  oracle_mask: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """返回 ``(Iadm, meta)``；``image`` 与返回值都是 0..1 的单通道数组。

    ``oracle_mask=True`` 时 ``heatmap`` 是 GT 二值 mask：

    * 不再做 min-max 归一化，直接按 ``> 0.5`` 二值化；
    * **不做** ``abnormal_score < threshold`` 的正常回退——Oracle 是上限实验，
      小病灶的面积占比本来就可能远低于 0.6，按预测置信度回退会把 O0 退化成 B0；
    * 只有当前切片完全没有病灶（mask 全 0）时保留原图（这一层没有可增强的 GT）。
    """
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if image.ndim != 2:
        raise ValueError(f"图像必须是二维 [H,W]，收到 {image.shape}")
    if oracle_mask:
        attention = (resize_heatmap(heatmap, image.shape) >= 0.5).astype(np.float32)
    else:
        attention = normalize_heatmap(resize_heatmap(heatmap, image.shape), heat_norm)
    meta: dict[str, Any] = {
        "gamma": float(params.gamma),
        "threshold": float(params.threshold),
        "mode": params.mode,
        "normal_fallback": params.normal_fallback,
        "oracle_mask": bool(oracle_mask),
        "abnormal_score": None if abnormal_score is None else float(abnormal_score),
        "attention_max": float(attention.max()) if attention.size else 0.0,
        "attention_mean": float(attention.mean()) if attention.size else 0.0,
        "attention_positive_fraction": float((attention > 0.05).mean())
        if attention.size else 0.0,
        "fallback": False,
        "reason": "",
    }
    if oracle_mask:
        if float(attention.max()) <= 1e-6:
            meta["fallback"] = True
            meta["reason"] = "oracle_empty_mask"
            return image.copy(), meta
        factor = float(params.gamma) + (1.0 - float(params.gamma)) * attention
        meta["factor_min"] = float(factor.min())
        meta["factor_max"] = float(factor.max())
        meta["reason"] = "oracle_gt_mask"
        return (image * factor).astype(np.float32), meta
    if params.normal_fallback == "original" and abnormal_score is not None \
            and float(abnormal_score) < float(params.threshold):
        meta["fallback"] = True
        meta["reason"] = "abnormal_score_below_threshold"
        return image.copy(), meta
    if float(attention.max()) <= 1e-6:
        meta["fallback"] = True
        meta["reason"] = "flat_heatmap"
        return image.copy(), meta

    if params.mode == "soft":
        factor = float(params.gamma) + (1.0 - float(params.gamma)) * attention
    else:  # hard：只保留阈值以上区域
        factor = np.where(attention >= float(params.threshold), 1.0,
                          float(params.gamma)).astype(np.float32)
    meta["factor_min"] = float(factor.min())
    meta["factor_max"] = float(factor.max())
    return (image * factor).astype(np.float32), meta


def render_adm_grid(original: np.ndarray, heatmap: np.ndarray,
                    enhanced: np.ndarray, params: AdmParams,
                    abnormal_score: float | None = None) -> Image.Image:
    """三联图：原图 / 归一化热图 / 增强图，用于人工验收。"""
    original = np.clip(np.asarray(original, dtype=np.float32), 0.0, 1.0)
    attention = normalize_heatmap(resize_heatmap(heatmap, original.shape))
    heat = np.clip(attention, 0.0, 1.0)
    enhanced = np.clip(np.asarray(enhanced, dtype=np.float32), 0.0, 1.0)
    panels = [
        (original * 255.0).astype(np.uint8),
        (heat * 255.0).astype(np.uint8),
        (enhanced * 255.0).astype(np.uint8),
    ]
    gap = 4
    height, width = original.shape
    canvas = np.full((height, width * 3 + gap * 2), 255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        start = index * (width + gap)
        canvas[:, start:start + width] = panel
    return Image.fromarray(canvas, mode="L")
