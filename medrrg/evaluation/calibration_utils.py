"""小工具：二分类/多标签指标的确定性实现（不依赖 sklearn）。

AUROC 用平均秩（并列值取平均秩）；AUPRC 用排序后的精度-召回积分；F1 在 0.5
阈值；Brier 与 ECE 用于校准。单类别时 ``auroc=None``，不伪造 0.5。
"""

from __future__ import annotations

import numpy as np

__all__ = ["binary_metrics", "auroc", "average_precision", "ece"]


def auroc(probabilities: np.ndarray, targets: np.ndarray) -> float | None:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64) > 0.5
    if labels.sum() == 0 or (~labels).sum() == 0:
        return None
    order = np.argsort(probs)
    ranks = np.empty(probs.size, dtype=np.float64)
    ranks[order] = np.arange(1, probs.size + 1)
    sorted_probs = probs[order]
    start = 0
    while start < sorted_probs.size:
        end = start
        while end + 1 < sorted_probs.size and sorted_probs[end + 1] == sorted_probs[start]:
            end += 1
        if end > start:
            ranks[order[start:end + 1]] = np.mean(np.arange(start + 1, end + 2))
        start = end + 1
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    rank_sum = ranks[labels].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(probabilities: np.ndarray, targets: np.ndarray) -> float | None:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64) > 0.5
    if labels.sum() == 0:
        return None
    order = np.argsort(-probs)
    ordered = labels[order]
    tp = np.cumsum(ordered)
    precision = tp / np.arange(1, ordered.size + 1)
    return float((precision * ordered).sum() / labels.sum())


def ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probs >= low) & (probs < high)
        if mask.any():
            value += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(value)


def binary_metrics(probabilities: np.ndarray, targets: np.ndarray,
                   threshold: float = 0.5) -> dict[str, float | None]:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64) > 0.5
    pred = probs >= threshold
    tp = float((pred & labels).sum())
    fp = float((pred & ~labels).sum())
    fn = float((~pred & labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "auroc": auroc(probs, labels),
        "auprc": average_precision(probs, labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": float(((probs - labels) ** 2).mean()),
        "ece": ece(probs, labels),
        "prevalence": float(labels.mean()) if labels.size else 0.0,
        "num_positive": int(labels.sum()),
        "num_total": int(labels.size),
    }
