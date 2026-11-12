"""ADM 证据报告 Transformer（方案 §3.5、§5.5）。

唯一装配路径：训练、验证、生成都调用 :meth:`encode_evidence`。

* ``visual_tokens`` → ``Linear(d_visual,d_model)+LayerNorm`` → modality/z/view/type
  embedding；
* ``semantic_*`` → :class:`SemanticProjector` → 13 个语义 tokens；
* 拼接后 108(+13)=121 个 token → 2 层 Encoder → 4 层 Decoder → Findings/Impression。

``use_semantic: false``（B0/B1）时 memory 只有 108 个视觉 token。
没有 global/roi resampler、没有 GeometryProjector、没有 ROI null token。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts import AdmEvidenceBatch
from ..errors import ConfigError
from .token_projectors import AdmEvidenceAssembler, SemanticProjector
from .transformer_decoder import DecoderConfig, ReportDecoder
from .transformer_encoder import EvidenceEncoder

__all__ = ["AdmReportModelConfig", "AdmEvidenceReportTransformer"]


@dataclass
class AdmReportModelConfig:
    evidence_mode: str = "adm_simple"
    d_model: int = 256
    heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 4
    ffn: int = 1024
    dropout: float = 0.1
    max_report_tokens: int = 256
    visual_tokens: int = 108
    semantic_tokens: int = 13
    d_visual: int = 256
    d_proto: int = 512
    use_semantic: bool = True
    num_modalities: int = 4
    num_views: int = 2
    z_buckets: int = 8
    patch_grid: int = 3
    evidence_dropout: float = 0.1
    use_geometry: bool = False
    use_roi_branch: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_mode != "adm_simple":
            raise ConfigError(
                f"只支持 evidence_mode='adm_simple'，收到 {self.evidence_mode!r}")
        if self.use_geometry or self.use_roi_branch:
            raise ConfigError("ADM 正式模型不允许 geometry / ROI 支路")
        if self.d_model % self.heads != 0:
            raise ConfigError(f"d_model={self.d_model} 不能被 heads={self.heads} 整除")
        if self.visual_tokens < 1:
            raise ConfigError("visual_tokens 必须 >= 1")
        if self.patch_grid < 1:
            raise ConfigError("patch_grid 必须 >= 1")
        if self.use_semantic and self.semantic_tokens < 1:
            raise ConfigError("use_semantic=True 时 semantic_tokens 必须 >= 1")

    @classmethod
    def from_config(cls, raw: dict[str, Any], d_visual: int, d_proto: int,
                    visual_tokens: int = 108, semantic_tokens: int = 13,
                    patch_grid: int = 3) -> "AdmReportModelConfig":
        raw = dict(raw or {})
        known = {item.name for item in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = {key: value for key, value in raw.items() if key not in known}
        return cls(
            evidence_mode=str(raw.get("evidence_mode", "adm_simple")),
            d_model=int(raw.get("d_model", 256)),
            heads=int(raw.get("heads", 8)),
            encoder_layers=int(raw.get("encoder_layers", 2)),
            decoder_layers=int(raw.get("decoder_layers", 4)),
            ffn=int(raw.get("ffn", 1024)),
            dropout=float(raw.get("dropout", 0.1)),
            max_report_tokens=int(raw.get("max_report_tokens", 256)),
            visual_tokens=int(raw.get("visual_tokens", visual_tokens)),
            semantic_tokens=int(raw.get("semantic_tokens", semantic_tokens)),
            d_visual=int(d_visual),
            d_proto=int(d_proto),
            use_semantic=bool(raw.get("use_semantic", True)),
            num_modalities=int(raw.get("num_modalities", 4)),
            num_views=int(raw.get("num_views", 2)),
            z_buckets=int(raw.get("z_buckets", 8)),
            patch_grid=int(patch_grid),
            evidence_dropout=float(raw.get("evidence_dropout", 0.1)),
            use_geometry=bool(raw.get("use_geometry", False)),
            use_roi_branch=bool(raw.get("use_roi_branch", False)),
            extra=extra)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdmEvidenceReportTransformer(nn.Module):
    def __init__(self, config: AdmReportModelConfig | None = None, *,
                 vocab_size: int, pad_id: int = 0,
                 model_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            if model_config is None:
                raise ConfigError("必须提供 config 或 model_config")
            config = AdmReportModelConfig(**model_config)
        self.config = config
        self.visual_proj = nn.Linear(config.d_visual, config.d_model)
        self.visual_norm = nn.LayerNorm(config.d_model)
        self.semantic_projector: SemanticProjector | None = None
        if config.use_semantic:
            self.semantic_projector = SemanticProjector(
                d_proto=config.d_proto, d_model=config.d_model,
                dropout=config.dropout)
        self.assembler = AdmEvidenceAssembler(
            config.d_model, num_modalities=config.num_modalities,
            num_views=config.num_views, z_buckets=config.z_buckets,
            patch_grid=config.patch_grid)
        self.encoder = EvidenceEncoder(config.d_model, config.heads,
                                       config.encoder_layers, config.ffn,
                                       config.dropout)
        self.decoder = ReportDecoder(DecoderConfig(
            vocab_size=vocab_size, d_model=config.d_model, heads=config.heads,
            layers=config.decoder_layers, ffn=config.ffn, dropout=config.dropout,
            max_len=config.max_report_tokens, pad_id=pad_id))

    # -- 证据编码 --------------------------------------------------------- #
    def encode_evidence(self, evidence: AdmEvidenceBatch) -> dict[str, Any]:
        if evidence.visual_tokens is None:
            raise ConfigError("证据批缺少 visual_tokens")
        device = evidence.visual_tokens.device
        batch = evidence.visual_tokens.shape[0]
        visual_available = (evidence.visual_available.bool()
                            if evidence.visual_available is not None
                            else torch.ones(batch, dtype=torch.bool, device=device))
        semantic_available = (evidence.semantic_available.bool()
                              if evidence.semantic_available is not None
                              else torch.zeros(batch, dtype=torch.bool, device=device))
        # 只在训练期做证据 dropout；生成/验证必须使用真实证据
        if self.training and self.config.evidence_dropout > 0:
            keep_visual = torch.rand(batch, device=device) >= self.config.evidence_dropout
            visual_available = visual_available & keep_visual
            if self.config.use_semantic:
                keep_semantic = (torch.rand(batch, device=device)
                                 >= self.config.evidence_dropout)
                semantic_available = semantic_available & keep_semantic

        visual = self.visual_norm(self.visual_proj(evidence.visual_tokens.float()))
        semantic_tokens = None
        semantic_mask = None
        semantic_gate = None
        if self.config.use_semantic and self.semantic_projector is not None:
            if evidence.semantic_prototypes is None:
                raise ConfigError("use_semantic=True 但证据批缺少 semantic_prototypes")
            semantic_tokens, semantic_mask, semantic_gate = self.semantic_projector(
                evidence.semantic_prototypes, evidence.semantic_scores,
                evidence.semantic_confidence, evidence.semantic_margin,
                evidence.semantic_mask.bool(), semantic_available)
            semantic_available = semantic_mask.any(dim=1) & semantic_available

        tokens, mask = self.assembler(
            visual, evidence.visual_mask, evidence.visual_modality,
            evidence.visual_z, evidence.visual_view, visual_available,
            evidence.visual_patch_row, evidence.visual_patch_col,
            semantic_tokens, semantic_mask, semantic_available)
        memory = self.encoder(tokens, mask)
        return {"memory": memory, "memory_mask": mask,
                "evidence_tokens": tokens, "evidence_mask": mask,
                "visual_available": visual_available,
                "semantic_available": semantic_available,
                "semantic_gate": semantic_gate}

    # -- 训练 / 验证 ------------------------------------------------------ #
    def forward(self, evidence: AdmEvidenceBatch, decoder_input_ids: torch.Tensor,
                labels: torch.Tensor | None = None, label_smoothing: float = 0.0
                ) -> dict[str, Any]:
        encoded = self.encode_evidence(evidence)
        logits = self.decoder(decoder_input_ids, encoded["memory"],
                              encoded["memory_mask"])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                ignore_index=-100, label_smoothing=label_smoothing)
        encoded.update({"logits": logits, "loss": loss})
        return encoded

    # -- 生成 ------------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, evidence: AdmEvidenceBatch, max_len: int | None = None,
                 mode: str = "greedy", beam: int = 1, bos_id: int = 2,
                 eos_id: int = 3) -> list[list[int]]:
        encoded = self.encode_evidence(evidence)
        return self.decoder.generate(
            encoded["memory"], encoded["memory_mask"],
            bos_id=int(bos_id), eos_id=int(eos_id),
            max_len=max_len or self.config.max_report_tokens,
            mode=mode, beam=beam)

    # -- 训练阶段 --------------------------------------------------------- #
    def freeze_decoder(self, freeze: bool = True) -> None:
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(not freeze)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
