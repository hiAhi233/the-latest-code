"""从零训练的小型报告 Decoder（简化版 §3.8）。

4 层 decoder、d_model=256、8 heads；teacher forcing 训练，greedy/beam 生成。
生成只依赖 memory（证据），不读取参考报告；遇到 EOS 停止，并受 max_len 约束。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DecoderConfig", "ReportDecoder"]


@dataclass
class DecoderConfig:
    vocab_size: int = 12000
    d_model: int = 256
    heads: int = 8
    layers: int = 4
    ffn: int = 1024
    dropout: float = 0.1
    max_len: int = 128
    pad_id: int = 0


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype)


class ReportDecoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model,
                                      padding_idx=config.pad_id)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.position = _PositionalEncoding(config.d_model, config.max_len)
        self.dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model, nhead=config.heads,
            dim_feedforward=config.ffn, dropout=config.dropout,
            activation="gelu", norm_first=True, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.layers,
                                             norm=nn.LayerNorm(config.d_model))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._causal: torch.Tensor | None = None

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        if self._causal is None or self._causal.shape[0] < size or self._causal.device != device:
            self._causal = torch.triu(
                torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
        return self._causal[:size, :size]

    def forward(self, decoder_input_ids: torch.Tensor, memory: torch.Tensor,
                memory_mask: torch.Tensor) -> torch.Tensor:
        target = self.position(self.embedding(decoder_input_ids))
        target = self.dropout(target)
        padding = decoder_input_ids == self.config.pad_id
        hidden = self.decoder(
            target, memory,
            tgt_mask=self._causal_mask(decoder_input_ids.shape[1],
                                       decoder_input_ids.device),
            tgt_key_padding_mask=padding,
            memory_key_padding_mask=~memory_mask.bool())
        return self.lm_head(hidden)

    # -- 生成 ------------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, memory: torch.Tensor, memory_mask: torch.Tensor,
                 bos_id: int, eos_id: int, max_len: int | None = None,
                 mode: str = "greedy", beam: int = 1) -> list[list[int]]:
        limit = int(max_len or self.config.max_len)
        if mode == "beam" and beam > 1:
            return [self._beam_one(memory[i:i + 1], memory_mask[i:i + 1], bos_id,
                                   eos_id, limit, beam)[0]
                    for i in range(memory.shape[0])]
        return self._greedy(memory, memory_mask, bos_id, eos_id, limit)

    def _greedy(self, memory: torch.Tensor, memory_mask: torch.Tensor,
                bos_id: int, eos_id: int, max_len: int) -> list[list[int]]:
        batch = memory.shape[0]
        sequences = torch.full((batch, 1), bos_id, dtype=torch.long,
                               device=memory.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        for _ in range(max_len - 1):
            logits = self.forward(sequences, memory, memory_mask)[:, -1]
            next_ids = logits.argmax(dim=-1)
            # 已结束样本持续输出 EOS，避免在 EOS 后继续追加正文
            next_ids = torch.where(finished,
                                   torch.full_like(next_ids, eos_id), next_ids)
            sequences = torch.cat([sequences, next_ids.unsqueeze(1)], dim=1)
            finished |= next_ids == eos_id
            if bool(finished.all()):
                break
        return sequences[:, 1:].tolist()

    def _beam_one(self, memory: torch.Tensor, memory_mask: torch.Tensor,
                  bos_id: int, eos_id: int, max_len: int, beam: int) -> list[list[int]]:
        beams: list[tuple[list[int], float]] = [([bos_id], 0.0)]
        for _ in range(max_len - 1):
            candidates: list[tuple[list[int], float]] = []
            for sequence, score in beams:
                if sequence[-1] == eos_id:
                    candidates.append((sequence, score))
                    continue
                ids = torch.tensor([sequence], dtype=torch.long, device=memory.device)
                logp = F.log_softmax(self.forward(ids, memory, memory_mask)[0, -1], dim=-1)
                values, indices = torch.topk(logp, k=beam)
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidates.append((sequence + [int(index)], score + float(value)))
            candidates.sort(key=lambda item: item[1], reverse=True)
            beams = candidates[:beam]
            if all(sequence[-1] == eos_id for sequence, _ in beams):
                break
        best, _ = max(beams, key=lambda item: item[1])
        return [best[1:]]
