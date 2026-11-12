"""语义投影与 ADM 证据 token 装配（方案 §3.4、§3.5）。

只有两类 token：

* 视觉：108 个 patch tokens，加 modality / normalized-z / view / type embedding；
* 语义：13 个概念 tokens，每个 = ``Linear(512→256)(text_prototype)`` +
  ``MLP([prob, confidence, margin])``，再按 confidence gate 与 learnable null
  token 插值。

没有 geometry token、没有 ROI token、没有坐标编码。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..errors import ContractError

__all__ = ["SemanticProjector", "AdmEvidenceAssembler"]


class SemanticProjector(nn.Module):
    """``prototype + [prob, confidence, margin]``；置信度门控 + null token。"""

    def __init__(self, d_proto: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.prototype_proj = nn.Linear(d_proto, d_model)
        self.stats_mlp = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model))
        self.gate = nn.Sequential(nn.Linear(3, max(1, d_model // 2)), nn.GELU(),
                                  nn.Linear(max(1, d_model // 2), 1))
        self.null_token = nn.Parameter(torch.zeros(d_model))

    def forward(self, prototypes: torch.Tensor, scores: torch.Tensor,
                confidence: torch.Tensor, margin: torch.Tensor | None,
                mask: torch.Tensor, available: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 ``tokens (B,K,d)``、``valid (B,K)``、``gate (B,K)``。"""
        if prototypes.dim() != 3:
            raise ContractError(f"prototypes 必须是 (B,K,dp)，收到 {tuple(prototypes.shape)}")
        if margin is None:
            margin = torch.zeros_like(scores)
        stats = torch.stack([scores.float(), confidence.float(), margin.float()], dim=-1)
        tokens = self.prototype_proj(prototypes.float()) + self.stats_mlp(stats)
        gate = torch.sigmoid(self.gate(stats)).squeeze(-1)
        tokens = gate.unsqueeze(-1) * tokens + (1.0 - gate.unsqueeze(-1)) * self.null_token
        valid = mask.bool() & available.bool().unsqueeze(1)
        return tokens, valid, gate


class AdmEvidenceAssembler(nn.Module):
    """给视觉/语义 token 叠加类型、模态、z、视图与 **2D patch 位置** embedding 并拼接。

    ResNet 的 3×3 patch 是行优先展平的；如果只加 modality/z/view，同一张切片内的
    9 个 patch 在序列里没有任何空间区分，打乱后模型看到的是同一个集合，部位/侧别
    描述会退化。因此额外加入 patch 行、列两个 embedding（与 ViT 的 2D 位置编码
    同一思路）。
    """

    TYPES = {"visual": 0, "semantic": 1}

    def __init__(self, d_model: int, num_modalities: int = 4, num_views: int = 2,
                 z_buckets: int = 8, patch_grid: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_modalities = int(num_modalities)
        self.num_views = max(1, int(num_views))
        self.z_buckets = max(2, int(z_buckets))
        self.patch_grid = max(1, int(patch_grid))
        self.type_embedding = nn.Embedding(2, d_model)
        self.modality_embedding = nn.Embedding(self.num_modalities, d_model)
        self.z_embedding = nn.Embedding(self.z_buckets, d_model)
        self.view_embedding = nn.Embedding(self.num_views, d_model)
        self.patch_row_embedding = nn.Embedding(self.patch_grid, d_model)
        self.patch_col_embedding = nn.Embedding(self.patch_grid, d_model)
        self.null_visual = nn.Parameter(torch.zeros(d_model))
        self.null_semantic = nn.Parameter(torch.zeros(d_model))
        for embedding in (self.type_embedding, self.modality_embedding,
                          self.z_embedding, self.view_embedding,
                          self.patch_row_embedding, self.patch_col_embedding):
            nn.init.normal_(embedding.weight, std=0.02)

    def bucketize_z(self, z_value: torch.Tensor) -> torch.Tensor:
        scaled = (z_value.float().clamp(0.0, 1.0) * (self.z_buckets - 1)).round().long()
        return scaled.clamp(0, self.z_buckets - 1)

    def embed_visual(self, tokens: torch.Tensor, modality: torch.Tensor,
                     z_value: torch.Tensor, view: torch.Tensor,
                     mask: torch.Tensor, available: torch.Tensor,
                     patch_row: torch.Tensor | None = None,
                     patch_col: torch.Tensor | None = None) -> torch.Tensor:
        batch, count, _ = tokens.shape
        modality = modality.long().clamp(0, self.num_modalities - 1)
        view = view.long().clamp(0, self.num_views - 1)
        if patch_row is None or patch_col is None:
            # 缓存未提供时按 token 顺序推断行优先 2D 位置
            index = torch.arange(count, device=tokens.device)
            patch_row = (index // self.patch_grid) % self.patch_grid
            patch_col = index % self.patch_grid
            patch_row = patch_row.unsqueeze(0).expand(batch, -1)
            patch_col = patch_col.unsqueeze(0).expand(batch, -1)
        patch_row = patch_row.long().clamp(0, self.patch_grid - 1)
        patch_col = patch_col.long().clamp(0, self.patch_grid - 1)
        type_ids = torch.full((batch, count), self.TYPES["visual"], dtype=torch.long,
                              device=tokens.device)
        hidden = (tokens
                  + self.modality_embedding(modality)
                  + self.z_embedding(self.bucketize_z(z_value))
                  + self.view_embedding(view)
                  + self.patch_row_embedding(patch_row)
                  + self.patch_col_embedding(patch_col)
                  + self.type_embedding(type_ids))
        valid = (mask.bool() & available.bool().unsqueeze(1)).unsqueeze(-1)
        return torch.where(valid, hidden, self.null_visual.to(hidden.dtype))

    def embed_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, _ = tokens.shape
        type_ids = torch.full((batch, count), self.TYPES["semantic"], dtype=torch.long,
                              device=tokens.device)
        return tokens + self.type_embedding(type_ids)

    def forward(self, visual_tokens: torch.Tensor, visual_mask: torch.Tensor,
                visual_modality: torch.Tensor, visual_z: torch.Tensor,
                visual_view: torch.Tensor, visual_available: torch.Tensor,
                visual_patch_row: torch.Tensor | None = None,
                visual_patch_col: torch.Tensor | None = None,
                semantic_tokens: torch.Tensor | None = None,
                semantic_mask: torch.Tensor | None = None,
                semantic_available: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = visual_tokens.shape[0]
        device = visual_tokens.device
        if visual_mask is None:
            visual_mask = torch.ones(visual_tokens.shape[:2], dtype=torch.bool,
                                     device=device)
        if visual_modality is None:
            visual_modality = torch.zeros(visual_tokens.shape[:2], dtype=torch.long,
                                          device=device)
        if visual_z is None:
            visual_z = torch.zeros(visual_tokens.shape[:2], device=device)
        if visual_view is None:
            visual_view = torch.zeros(visual_tokens.shape[:2], dtype=torch.long,
                                      device=device)
        if visual_available is None:
            visual_available = torch.ones(batch, dtype=torch.bool, device=device)

        visual = self.embed_visual(visual_tokens, visual_modality, visual_z,
                                   visual_view, visual_mask, visual_available,
                                   visual_patch_row, visual_patch_col)
        visual_mask = torch.where(
            visual_available.bool().unsqueeze(1),
            visual_mask.bool(),
            torch.ones_like(visual_mask, dtype=torch.bool))
        parts = [visual]
        masks = [visual_mask]

        if semantic_tokens is not None:
            if semantic_tokens.dim() != 3:
                raise ContractError(
                    f"semantic_tokens 必须是 (B,K,d)，收到 {tuple(semantic_tokens.shape)}")
            available = (semantic_available.bool() if semantic_available is not None
                         else torch.ones(batch, dtype=torch.bool, device=device))
            if semantic_mask is None:
                semantic_mask = torch.ones(semantic_tokens.shape[:2], dtype=torch.bool,
                                           device=device)
            valid = semantic_mask.bool() & available.unsqueeze(1)
            semantic_tokens = torch.where(valid.unsqueeze(-1), semantic_tokens,
                                          self.null_semantic.to(semantic_tokens.dtype))
            mask = torch.where(available.unsqueeze(1),
                               semantic_mask.bool(),
                               torch.ones_like(semantic_mask, dtype=torch.bool))
            parts.append(self.embed_semantic(semantic_tokens))
            masks.append(mask)

        return torch.cat(parts, dim=1), torch.cat(masks, dim=1)
