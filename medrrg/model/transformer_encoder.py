"""轻量证据 Encoder（简化版 §3.8）：2 层 Pre-Norm Transformer。"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["EvidenceEncoder"]


class EvidenceEncoder(nn.Module):
    def __init__(self, d_model: int = 256, heads: int = 8, layers: int = 2,
                 ffn: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} 不能被 heads={heads} 整除")
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ffn, dropout=dropout,
            activation="gelu", norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens, src_key_padding_mask=~mask.bool())
