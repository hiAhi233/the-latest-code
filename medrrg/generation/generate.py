"""生成：只读 ADM 证据缓存，不读取参考报告（方案 §3.5、§12）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import DataLoader

from ..data.adm_dataset import (AdmDatasetConfig, AdmEvidenceDataset,
                                collate_adm_samples)
from ..data.manifest import load_manifest
from ..errors import DataError
from ..evidence import load_sources_from_extra
from ..training.checkpoint import load_checkpoint
from ..utils import resolve_device, write_json, write_jsonl

__all__ = ["GenerationConfig", "postprocess_report", "generate", "truncate_at_eos"]

SECTION_RE = re.compile(r"<\s*(FINDINGS|IMPRESSION)\s*>(.*?)(?:<\s*/\s*\1\s*>|$)",
                        re.IGNORECASE | re.DOTALL)


@dataclass
class GenerationConfig:
    max_len: int = 256
    mode: str = "greedy"
    beam: int = 1
    batch_size: int = 8
    device: str = "auto"


def truncate_at_eos(ids: Sequence[int], eos_id: int) -> list[int]:
    """解码前截断到首个 EOS（不含 EOS），防止 EOS 后的多余 token 进入报告。"""
    output = []
    for value in ids:
        if int(value) == int(eos_id):
            break
        output.append(int(value))
    return output


def postprocess_report(text: str, has_impression: bool = True) -> dict[str, Any]:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    cleaned = re.sub(r"<\s*/?\s*(BOS|EOS|PAD|UNK)\s*>", " ", cleaned,
                     flags=re.IGNORECASE)
    matches = list(SECTION_RE.finditer(cleaned))
    findings, impression = "", ""
    warnings: list[str] = []
    if matches:
        for match in matches:
            name = match.group(1).lower()
            body = re.sub(r"</?\s*(FINDINGS|IMPRESSION)\s*>", " ", match.group(2),
                          flags=re.IGNORECASE).strip()
            if name == "findings":
                findings = (findings + " " + body).strip()
            else:
                impression = (impression + " " + body).strip()
    else:
        findings = cleaned
        warnings.append("missing_control_tokens")
    findings = re.sub(r"\s+", " ", findings).strip()
    impression = re.sub(r"\s+", " ", impression).strip()
    if not has_impression:
        impression = ""
    if findings and not impression and has_impression:
        warnings.append("missing_impression_section")
    formatted = ""
    if findings:
        formatted += f"<FINDINGS> {findings} </FINDINGS>"
    if impression:
        formatted += (" " if formatted else "") + f"<IMPRESSION> {impression} </IMPRESSION>"
    return {"findings": findings, "impression": impression,
            "formatted": formatted, "warnings": warnings}


def build_dataset_from_checkpoint(loaded: dict[str, Any], split: str
                                 ) -> tuple[AdmEvidenceDataset, Any]:
    """按 checkpoint 记录的证据源与模型配置构建数据集。"""
    extra = loaded["extra"]
    model_config = loaded["config"]
    manifest_path = extra.get("manifest")
    if not manifest_path:
        raise DataError("checkpoint 缺少 manifest 引用，无法生成")
    manifest = load_manifest(manifest_path)
    sources = load_sources_from_extra(extra, manifest)
    dataset_config = AdmDatasetConfig(
        max_report_tokens=model_config.max_report_tokens,
        use_semantic=model_config.use_semantic,
        d_visual=model_config.d_visual, d_proto=model_config.d_proto)
    dataset = AdmEvidenceDataset(manifest, split, loaded["tokenizer"],
                                 sources.visual, dataset_config,
                                 semantic=sources.semantic,
                                 localization=sources.localization)
    return dataset, manifest


def generate(checkpoint: str | Path, split: str = "test",
             out: str | Path | None = None,
             config: GenerationConfig | None = None,
             limit: int | None = None, verbose: bool = True) -> dict[str, Any]:
    cfg = config or GenerationConfig()
    device = resolve_device(cfg.device)
    loaded = load_checkpoint(checkpoint, device=device)
    extra = loaded["extra"]
    dataset, manifest = build_dataset_from_checkpoint(loaded, split)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_adm_samples)
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    rows: list[dict[str, Any]] = []
    for batch in loader:
        if limit is not None and len(rows) >= limit:
            break
        evidence = batch["evidence"].to(device)
        sequences = model.generate(evidence, max_len=cfg.max_len, mode=cfg.mode,
                                   beam=cfg.beam, bos_id=tokenizer.bos_id,
                                   eos_id=tokenizer.eos_id)
        for index, sequence in enumerate(sequences):
            study_id = batch["study_ids"][index]
            study = manifest.get(study_id)
            sequence = truncate_at_eos(sequence, tokenizer.eos_id)
            raw = tokenizer.decode(sequence, skip_special=False)
            decoded = postprocess_report(
                raw, has_impression=bool(study.report.impression.strip()))
            rows.append({
                "study_id": study_id,
                "patient_id": batch["patient_ids"][index],
                "split": study.split,
                "prediction": decoded["formatted"] or decoded["findings"],
                "findings": decoded["findings"],
                "impression": decoded["impression"],
                "raw_output": raw,
                "num_tokens": len(sequence),
                "warnings": decoded["warnings"],
                "adm_applied": bool(batch["adm_applied"][index]),
                "reference": study.report.text(),
                "reference_findings": study.report.findings,
                "reference_impression": study.report.impression,
            })
            if limit is not None and len(rows) >= limit:
                break
        if verbose and len(rows) % 10 == 0:
            print(f"[generate] {len(rows)} 例", flush=True)
    if out is None:
        from ..config import project_root

        results_root = Path(str(extra.get("results_root", "results")))
        if not results_root.is_absolute():
            results_root = project_root() / results_root
        out_path = (results_root / str(extra.get("experiment", "run")) / split
                    / f"predictions_{split}.jsonl")
    else:
        out_path = Path(out)
        if out_path.suffix != ".jsonl":
            out_path = out_path / f"predictions_{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows)
    meta = {
        "split": split, "count": len(rows), "checkpoint": loaded["checkpoint"],
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "manifest": manifest.path, "manifest_sha256": manifest.sha256,
        "model_config": loaded["config"].to_dict(),
        "cache_identity": extra.get("cache_identity", {}),
        "run_meta": str(out_path.parent / "run_meta.json"),
    }
    write_json(out_path.parent / "run_meta.json", meta)
    if verbose:
        print(f"[generate] 写入 {len(rows)} 例 -> {out_path}", flush=True)
    return {**meta, "out": str(out_path)}
