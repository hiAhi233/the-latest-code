"""Compare predicted heatmaps on the same RadGenome patient partition."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medrrg.config import load_config
from medrrg.data.adapters import build_adapter
from medrrg.data.manifest import load_manifest
from medrrg.evaluation.upstream import _resize_volume, _seg_zyx
from medrrg.features.adm_cache import AdmLocalizationCache
from medrrg.utils import write_json


def case_metrics(item, mask, labels, lesion_sizes, slice_areas):
    heat = item.heatmap
    positive = heat >= item.threshold
    intersection = int((positive & mask).sum())
    overlap = np.bincount(labels[positive], minlength=len(lesion_sizes))[1:]
    foreground = int(mask.sum())
    background = mask.size - foreground
    ranks = stats.rankdata(heat.ravel(), method="average")
    auc = (ranks[mask.ravel()].sum() - foreground * (foreground + 1) / 2) / (foreground * background)
    areas = slice_areas[item.selected_z]
    return {
        "dice": 2 * intersection / (int(positive.sum()) + foreground),
        "lesion_recall": float(np.mean(overlap / lesion_sizes[1:] >= 0.1)),
        "pixel_auroc": float(auc),
        "peak_in_lesion": float(mask.ravel()[np.argmax(heat)]),
        "top3_case_hit": float(np.any(areas > 0)),
        "top3_slice_hit": float(np.mean(areas > 0)),
        "top3_relative_tumor_area": float(np.mean(areas / slice_areas.max())),
        "false_positive_volume_ratio": int((positive & ~mask).sum()) / foreground,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adm_radgenome_f0.yaml")
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--candidate", action="append", required=True, help="name=cache directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_manifest(config.path("manifest"))
    manifest.verify_patients()
    adapter = build_adapter(config)
    old = AdmLocalizationCache.load(config.path("evidence.localization_dir"))
    caches = {"existing": old}
    for specification in args.candidate:
        name, path = specification.split("=", 1)
        if name in caches:
            raise ValueError(f"Duplicate name: {name}")
        caches[name] = AdmLocalizationCache.load(path)
    rows = {name: [] for name in caches}
    for study in manifest.split(args.split):
        sid = study.study_id
        gt = _seg_zyx(adapter, study)
        baseline = old.get(sid)
        mask = _resize_volume(gt, baseline.heatmap.shape)
        labels, _ = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        sizes = np.bincount(labels.ravel())
        areas = gt.sum(axis=(1, 2))
        for name, cache in caches.items():
            item = cache.get(sid)
            if item.heatmap.shape != mask.shape or item.threshold != baseline.threshold:
                raise ValueError(f"Heatmap grid or threshold differs: {name} {sid}")
            if item.source != "predicted" or len(item.selected_z) != 3:
                raise ValueError(f"Expected predicted heatmap with three selected slices: {name} {sid}")
            if cache.index[sid]["input_sha256"] != old.index[sid]["input_sha256"]:
                raise ValueError(f"Source image differs: {name} {sid}")
            rows[name].append({"study_id": sid, **case_metrics(item, mask, labels, sizes, areas)})
    means = {name: {key: float(np.mean([row[key] for row in cases]))
                    for key in cases[0] if key != "study_id"} for name, cases in rows.items()}
    delta = {name: {key: value - means["existing"][key] for key, value in scores.items()}
             for name, scores in means.items() if name != "existing"}
    winner = max(delta, key=lambda name: means[name]["dice"]) if args.split == "val" else None
    result = {"split": args.split, "case_count": len(rows["existing"]), "means": means,
              "delta_vs_existing": delta, "best_candidate_by_validation_dice": winner,
              "source_caches": {name: str(cache.root) for name, cache in caches.items()}, "cases": rows}
    write_json(args.out, result)
    for name, scores in means.items():
        print(name, " ".join(f"{key}={value:.4f}" for key, value in scores.items()))
    if winner:
        print(f"Selected by validation Dice: {winner}")


if __name__ == "__main__":
    main()
