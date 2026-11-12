"""医学概念指标（简化版 §9.2）：概念 P/R/F1 + 左右侧/脑叶属性。"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

from ..ontology import Ontology

__all__ = ["extract_attributes", "compute_concept_metrics"]

SIDE_RE = re.compile(r"\b(left|right|bilateral|both)\b", re.IGNORECASE)
LOBE_RE = re.compile(
    r"\b(frontal|parietal|temporal|occipital|frontoparietal|frontotemporal|"
    r"parieto-occipital|basal ganglia|thalam|ventricle|ventricular|midline)\b",
    re.IGNORECASE)


def _normalise(values: set[str]) -> set[str]:
    mapping = {"both": "bilateral", "thalam": "thalamic"}
    return {mapping.get(value.lower(), value.lower()) for value in values}


def extract_attributes(text: str) -> dict[str, set[str]]:
    sides = {match.group(1).lower() for match in SIDE_RE.finditer(text or "")}
    lobes = {match.group(1).lower() for match in LOBE_RE.finditer(text or "")}
    return {"side": _normalise(sides), "lobe": _normalise(lobes)}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compute_concept_metrics(rows: Sequence[Mapping[str, Any]],
                            ontology: Ontology) -> dict[str, Any]:
    from .nlg_metrics import strip_control_tags

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    attribute_hits = {"side": [0, 0], "lobe": [0, 0]}   # correct, support
    for row in rows:
        prediction = strip_control_tags(str(row.get("prediction", "")))
        reference = strip_control_tags(str(row.get("reference", "")))
        pred_concepts = set(ontology.match(prediction))
        ref_concepts = set(ontology.match(reference))
        for concept_id in pred_concepts | ref_concepts:
            key = ("tp" if concept_id in pred_concepts and concept_id in ref_concepts
                   else "fp" if concept_id in pred_concepts else "fn")
            counts[concept_id][key] += 1
        pred_attr = extract_attributes(prediction)
        ref_attr = extract_attributes(reference)
        for kind in ("side", "lobe"):
            # 参考未提及该属性时不计入准确率，避免“空对空”抬高指标
            if not ref_attr[kind]:
                continue
            attribute_hits[kind][1] += 1
            if pred_attr[kind] == ref_attr[kind]:
                attribute_hits[kind][0] += 1

    per_concept = {}
    f1_values = []
    tp_total = fp_total = fn_total = 0
    for concept_id in ontology.concept_ids:
        stat = counts[concept_id]
        if stat["tp"] + stat["fp"] + stat["fn"] == 0:
            continue
        precision, recall, f1 = _prf(stat["tp"], stat["fp"], stat["fn"])
        per_concept[concept_id] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), **stat}
        f1_values.append(f1)
        tp_total += stat["tp"]
        fp_total += stat["fp"]
        fn_total += stat["fn"]
    micro = _prf(tp_total, fp_total, fn_total)
    return {
        "num_examples": len(rows),
        "micro_precision": round(micro[0], 4),
        "micro_recall": round(micro[1], 4),
        "micro_f1": round(micro[2], 4),
        "macro_f1": round(sum(f1_values) / max(1, len(f1_values)), 4),
        "per_concept": per_concept,
        "num_concepts_evaluated": len(f1_values),
        "attribute_exact_match": {
            kind: ({"accuracy": round(values[0] / values[1], 4),
                    "num_with_reference": values[1]}
                   if values[1] else {"accuracy": None, "num_with_reference": 0})
            for kind, values in attribute_hits.items()},
    }
