"""评测编排：predictions.jsonl -> NLG / 概念 / 幻觉 / 分段完整率 -> 可选证据干预。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import Study
from ..data.manifest import load_manifest
from ..generation.generate import build_dataset_from_checkpoint, postprocess_report
from ..ontology import Ontology
from ..training.checkpoint import load_checkpoint
from ..utils import read_json, read_jsonl, resolve_device, write_json
from .concept_metrics import compute_concept_metrics
from .evidence_intervention import run_interventions
from .nlg_metrics import compute_nlg_metrics, strip_control_tags

__all__ = ["evaluate_run", "evaluate_predictions"]


def _text_of(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value) if value is not None else ""


def _hallucination(rows: list[dict[str, Any]], ontology: Ontology) -> dict[str, Any]:
    """概念级幻觉率：预测提到但参考报告没有的概念占比。"""
    rates: list[float] = []
    unsupported_total = predicted_total = 0
    for row in rows:
        prediction = strip_control_tags(_text_of(row, "prediction"))
        reference = strip_control_tags(_text_of(row, "reference"))
        predicted = set(ontology.match(prediction))
        supported = set(ontology.match(reference))
        unsupported = predicted - supported
        predicted_total += len(predicted)
        unsupported_total += len(unsupported)
        rates.append(len(unsupported) / max(1, len(predicted)))
    return {
        "concept_hallucination_rate": float(sum(rates) / max(1, len(rates))),
        "unsupported_concepts": int(unsupported_total),
        "predicted_concepts": int(predicted_total),
    }


def _section_completeness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = findings = impression = both = 0
    for row in rows:
        total += 1
        has_findings = bool(_text_of(row, "findings").strip())
        has_impression = bool(_text_of(row, "impression").strip())
        findings += int(has_findings)
        impression += int(has_impression)
        both += int(has_findings and has_impression)
    return {
        "findings_present_rate": findings / max(1, total),
        "impression_present_rate": impression / max(1, total),
        "both_sections_rate": both / max(1, total),
        "num_examples": total,
    }


def evaluate_predictions(rows: list[dict[str, Any]], ontology: Ontology,
                         train_texts: list[str] | None = None) -> dict[str, Any]:
    return {
        "num_examples": len(rows),
        "nlg": compute_nlg_metrics(rows, train_texts=train_texts),
        "concepts": compute_concept_metrics(rows, ontology),
        "hallucination": _hallucination(rows, ontology),
        "sections": _section_completeness(rows),
        "empty_predictions": sum(1 for row in rows
                                 if not _text_of(row, "prediction").strip()),
    }


def evaluate_run(run_dir: str | Path, interventions: bool = True,
                 n_resamples: int = 1000, concept_deltas: bool = True,
                 verbose: bool = True) -> dict[str, Any]:
    run_path = Path(run_dir)
    meta_path = run_path / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{run_dir} 缺少 run_meta.json；请先运行 generate")
    run_meta = read_json(meta_path)
    split = str(run_meta.get("split", "test"))
    predictions_path = run_path / f"predictions_{split}.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"预测文件不存在：{predictions_path}")
    rows = read_jsonl(predictions_path)

    loaded = load_checkpoint(run_meta["checkpoint"], device="cpu")
    extra = loaded["extra"]
    ontology = Ontology.from_config(extra["ontology"])
    manifest = load_manifest(extra["manifest"])
    train_texts = [study.report.text() for study in manifest.split("train")]
    metrics = evaluate_predictions(rows, ontology, train_texts)
    metrics.update({
        "run_dir": str(run_path), "split": split,
        "checkpoint": loaded["checkpoint"],
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "manifest_sha256": manifest.sha256,
        "model_config": loaded["config"].to_dict(),
        "cache_identity": extra.get("cache_identity", {}),
    })

    if interventions:
        device = resolve_device("auto")
        model = loaded["model"].to(device)
        dataset, _ = build_dataset_from_checkpoint(loaded, split)
        metrics["interventions"] = run_interventions(
            model, dataset, device, batch_size=8, n_resamples=n_resamples,
            concept_deltas=concept_deltas, ontology=ontology,
            tokenizer=loaded["tokenizer"], verbose=verbose)
    out_path = run_path / "metrics.json"
    write_json(out_path, metrics)
    metrics["metrics_path"] = str(out_path)
    return metrics


def load_reference_findings(manifest: Any) -> dict[str, str]:
    """调试辅助：study_id -> reference findings。"""
    return {study.study_id: study.report.findings for study in manifest}
