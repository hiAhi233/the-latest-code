"""上游证据质量（方案 §9）：定位 Dice/召回/注意力覆盖 + UniMed 概念 AUROC/F1。

这些指标评价的是**缓存里的上游证据本身**，与报告文本指标分开报告。

定位指标在缓存热图网格上计算（默认 64×64×Z），GT 来自训练切片的 seg NIfTI；
预测掩码用缓存行里记录的定位阈值二值化，不再改动阈值。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..contracts import Study
from ..errors import DataError
from ..ontology import Ontology
from .calibration_utils import auroc, binary_metrics

__all__ = ["localization_metrics", "semantic_metrics"]


def _seg_zyx(adapter: Any, study: Study) -> np.ndarray | None:
    seg_path = study.meta.get("seg_path")
    if not seg_path:
        return None
    path = adapter.resolve(seg_path)
    if not Path(path).is_file():
        return None
    import nibabel as nib

    volume = np.asanyarray(nib.as_closest_canonical(nib.load(str(path))).dataobj)
    # NIfTI 为 volume[x,y,z]；模型看到的是 volume[:, :, z] -> [z,x,y]
    return np.transpose(volume > 0, (2, 0, 1))


def _resize_volume(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(shape):
        return mask.astype(bool)
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    out = F.interpolate(tensor, size=shape, mode="nearest")[0, 0]
    return out.numpy() > 0.5


def localization_metrics(manifest: Any, adapter: Any, localization: Any,
                         splits: Sequence[str] = ("val", "test"),
                         limit: int | None = None,
                         verbose: bool = False) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for split in splits:
        for study in manifest.split(split):
            if limit is not None and len(rows) >= limit:
                break
            item = localization.get(study.study_id)
            gt = _seg_zyx(adapter, study)
            if gt is None:
                continue
            heat = item.heatmap
            gt = _resize_volume(gt, heat.shape)
            pred = heat >= float(item.threshold)
            pred_volume, gt_volume = float(pred.sum()), float(gt.sum())
            if pred_volume == 0 and gt_volume == 0:
                dice = 1.0
            else:
                dice = float(2.0 * (pred & gt).sum() / max(1e-9, pred_volume + gt_volume))
            from scipy import ndimage

            labels, count = ndimage.label(gt, structure=ndimage.generate_binary_structure(3, 1))
            if count:
                hits = sum(1 for c in range(1, count + 1)
                           if (pred & (labels == c)).sum()
                           / max(1, (labels == c).sum()) >= 0.1)
                recall = hits / count
            else:
                recall = 1.0
            total_heat = float(heat.sum())
            coverage = float((heat * gt).sum() / total_heat) if total_heat > 0 else 0.0
            lesion_mean = float(heat[gt].mean()) if gt.any() else 0.0
            background_mean = float(heat[~gt].mean()) if (~gt).any() else 0.0
            ratio = lesion_mean / max(1e-9, background_mean)
            top_flat = int(np.argmax(heat))
            top_index = np.unravel_index(top_flat, heat.shape)
            rows.append({
                "dice": dice,
                "lesion_recall": float(recall),
                "attention_coverage": coverage,
                "lesion_background_ratio": ratio,
                "false_positive_volume": float((pred & ~gt).sum()) / max(1.0, gt_volume),
                "pixel_auroc": float(auroc(heat.reshape(-1), gt.reshape(-1)) or 0.0),
                "top1_inside_lesion": float(gt[top_index]),
            })
            if verbose and len(rows) % 20 == 0:
                print(f"[evidence-quality] {len(rows)}", flush=True)
    if not rows:
        return {"num_studies": 0}
    result: dict[str, Any] = {"num_studies": len(rows),
                              "grid": "adm localization cache heatmap",
                              "threshold": "cached (checkpoint choice)"}
    for key in ("dice", "lesion_recall", "attention_coverage",
                "lesion_background_ratio", "false_positive_volume",
                "pixel_auroc", "top1_inside_lesion"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[key] = float(values.mean()) if values.size else None
        result[f"{key}_median"] = float(np.median(values)) if values.size else None
    return result


def _study_labels(study: Study, ontology: Ontology) -> set[str]:
    labels = set(ontology.match(study.report.text()))
    raw = list(study.meta.get("raw_disease") or [])
    if raw:
        labels |= set(ontology.match(" ".join(str(item) for item in raw)))
    return labels


def semantic_metrics(manifest: Any, ontology: Ontology, semantic: Any,
                     splits: Sequence[str] = ("val", "test"),
                     exclusive_groups: Sequence[dict[str, Any]] | None = None,
                     limit: int | None = None) -> dict[str, Any]:
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    concept_ids: list[str] = []
    for split in splits:
        for study in manifest.split(split):
            if limit is not None and len(scores) >= limit:
                break
            item = semantic.get(study.study_id)
            ids = list(item.concept_ids)
            values = np.asarray(item.probabilities, dtype=np.float32)
            if not ids or values.size != len(ids):
                continue
            concept_ids = ids
            labels = _study_labels(study, ontology)
            scores.append(values)
            targets.append(np.array([1.0 if cid in labels else 0.0 for cid in ids]))
    if not scores:
        return {"num_examples": 0}
    probabilities = np.stack(scores)
    labels = np.stack(targets)
    per_concept: dict[str, Any] = {}
    aurocs, f1s = [], []
    for index, concept_id in enumerate(concept_ids):
        metrics = binary_metrics(probabilities[:, index], labels[:, index])
        per_concept[concept_id] = {
            key: (round(value, 4) if isinstance(value, float)
                  and not np.isnan(value) else value)
            for key, value in metrics.items()}
        if metrics["auroc"] is not None:
            aurocs.append(metrics["auroc"])
            f1s.append(metrics["f1"])
    result: dict[str, Any] = {
        "num_examples": len(scores),
        "per_concept": per_concept,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "num_concepts": len(concept_ids),
        "note": "UniMed 相似度未校准，probability 不是临床诊断概率",
    }
    if exclusive_groups:
        index_of = {cid: i for i, cid in enumerate(concept_ids)}
        group_results: dict[str, Any] = {}
        for group in exclusive_groups:
            members = [cid for cid in group.get("concept_ids", []) if cid in index_of]
            if len(members) < 2:
                continue
            columns = [index_of[cid] for cid in members]
            member_rows = labels[:, columns]     # (N, len(members))
            correct = total = 0
            for row_index in range(len(scores)):
                truth = member_rows[row_index]
                if truth.sum() != 1:
                    continue
                total += 1
                predicted = int(np.argmax(scores[row_index][columns]))
                if truth[predicted] == 1.0:
                    correct += 1
            group_results[str(group.get("id", "group"))] = {
                "concepts": members,
                "accuracy": (correct / total) if total else None,
                "num_examples": total}
        result["exclusive_group_accuracy"] = group_results
    return result
