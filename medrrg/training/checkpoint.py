"""checkpoint 保存/加载：模型配置、词表、身份与证据引用一体。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..data.adm_dataset import WordTokenizer
from ..model.report_model import AdmEvidenceReportTransformer, AdmReportModelConfig
from ..utils import now_iso, sha256_file

__all__ = ["CHECKPOINT_VERSION", "save_checkpoint", "load_checkpoint"]

CHECKPOINT_VERSION = "medadm.checkpoint.v1"


def save_checkpoint(path: str | Path, model: AdmEvidenceReportTransformer,
                    tokenizer: WordTokenizer, model_config: AdmReportModelConfig,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_config": model_config.to_dict(),
        "vocab": list(tokenizer.vocab),
        "tokenizer_max_len": tokenizer.max_len,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "extra": dict(extra or {}),
        "created_at": now_iso(),
    }
    torch.save(payload, p)
    return {"checkpoint": str(p), "checkpoint_sha256": sha256_file(p)}


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint 版本 {payload.get('checkpoint_version')!r} 不兼容 "
            f"（期望 {CHECKPOINT_VERSION}）")
    config = AdmReportModelConfig(**payload["model_config"])
    tokenizer = WordTokenizer(vocab=list(payload["vocab"]),
                              max_len=int(payload.get("tokenizer_max_len", 256)))
    model = AdmEvidenceReportTransformer(
        config, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return {"model": model, "tokenizer": tokenizer, "config": config,
            "extra": dict(payload.get("extra") or {}),
            "checkpoint": str(p), "checkpoint_sha256": sha256_file(p)}
