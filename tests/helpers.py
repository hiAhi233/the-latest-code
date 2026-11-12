"""测试用合成缓存构造器（不读真实医学数据）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from medrrg.contracts import ImageRef, Report, Study
from medrrg.features.adm_cache import (LOCALIZATION_FORMAT, SEMANTIC_FORMAT,
                                       VISUAL_FORMAT)
from medrrg.features.cache_identity import (adm_localization_identity,
                                            resnet_visual_identity,
                                            unimed_semantic_identity)

CONCEPTS = ["brain_tumor", "enhancement", "edema"]


def make_study(study_id: str, split: str, findings: str, impression: str,
               dataset_id: str = "synthetic") -> Study:
    return Study(
        study_id=study_id,
        patient_id=study_id.rsplit("-", 1)[0],
        dataset_id=dataset_id,
        split=split,
        modality="MRI",
        language="en",
        images=[ImageRef(path=f"images/{study_id}/{group}.nii.gz", group=group)
                for group in ("t1n", "t1c", "t2w", "t2f")],
        report=Report(findings=findings, impression=impression),
        meta={"seg_path": f"images/{study_id}/seg.nii.gz"},
    )


def make_studies(count: int = 4) -> list[Study]:
    studies = []
    splits = ["train", "train", "val", "test"]
    for index in range(count):
        study_id = f"synthetic-{index:03d}-000"
        studies.append(make_study(
            study_id, splits[index % len(splits)],
            "A lesion with high signal is seen in the left frontal lobe.",
            "A lesion in the left frontal lobe. Consider a glioma."))
    return studies


def _write_index(root: Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(root / "index.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_localization_cache(root: Path, studies: list[Study], heat_size: int = 8,
                             slices: int = 4, selected_z: list[int] | None = None,
                             source: str = "predicted", threshold: float = 0.6,
                             gamma: float = 0.4) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "arrays").mkdir(exist_ok=True)
    selected = selected_z or [1, 2]
    rows = []
    image_hashes: dict[str, dict[str, str]] = {}
    for index, study in enumerate(studies):
        rng = np.random.default_rng(index)
        heat = np.zeros((slices, heat_size, heat_size), dtype=np.float16)
        heat[:, 2:5, 2:5] = 0.8
        heat += rng.random(heat.shape, dtype=np.float32).astype(np.float16) * 0.05
        scores = np.asarray([float(heat[z].max()) for z in range(slices)],
                            dtype=np.float32)
        abnormal = float(0.8)
        np.savez_compressed(root / "arrays" / f"{study.study_id}.npz",
                            heatmap_3d=heat, score_per_slice=scores,
                            abnormal_score=np.float32(abnormal))
        rows.append({"study_id": study.study_id, "source_sequence": "t2f",
                     "array_path": f"arrays/{study.study_id}.npz",
                     "shape_zyx": [slices, 16, 16], "heat_size": heat_size,
                     "threshold": threshold, "selected_z": list(selected),
                     "abnormal_score": abnormal, "source": source,
                     "checkpoint_sha256": "0" * 64,
                     "input_sha256": {"t2f": str(index)}})
        image_hashes[study.study_id] = {"t2f": str(index)}
    _write_index(root, rows)
    identity = adm_localization_identity(
        "code", "0" * 64,
        {"source_sequence": "t2f", "heat_size": heat_size},
        {"heat_threshold": threshold, "heat_size": heat_size}, image_hashes)
    meta = {"format": LOCALIZATION_FORMAT, "identity": identity,
            "source": source, "checkpoint_sha256": "0" * 64,
            "preprocessor": "synthetic", "thresholds": {"heat_threshold": threshold},
            "preprocessing": {"heat_size": heat_size, "source_sequence": "t2f"},
            "adm": {"gamma": gamma, "threshold": threshold, "mode": "soft"},
            "studies": [study.study_id for study in studies]}
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                    encoding="utf-8")
    return meta


def write_visual_cache(root: Path, studies: list[Study], localization_identity: str,
                       tokens_per_slice: int = 4, modalities: int = 2,
                       selected_z: list[int] | None = None, d_visual: int = 8,
                       adm_enabled: bool = True, gamma: float = 0.4,
                       threshold: float = 0.6) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "visual").mkdir(exist_ok=True)
    selected = selected_z or [1, 2]
    count = len(selected) * modalities * tokens_per_slice
    rows = []
    image_hashes: dict[str, dict[str, str]] = {}
    for index, study in enumerate(studies):
        rng = np.random.default_rng(100 + index)
        tokens = rng.normal(size=(count, d_visual)).astype(np.float32)
        tokens_null = rng.normal(size=(count, d_visual)).astype(np.float32)
        modality = np.repeat(np.arange(modalities), tokens_per_slice)
        modality = np.tile(modality, len(selected)).astype(np.int64)
        z_values = np.repeat([z / 3.0 for z in selected],
                             modalities * tokens_per_slice).astype(np.float32)
        patch_grid = 2
        patch = np.arange(tokens_per_slice)
        patch_rows = np.tile(patch // patch_grid, len(selected) * modalities)
        patch_cols = np.tile(patch % patch_grid, len(selected) * modalities)
        np.savez_compressed(root / "visual" / f"{study.study_id}.npz",
                            visual_tokens=tokens, visual_tokens_null=tokens_null,
                            visual_mask=np.ones(count, dtype=bool),
                            visual_modality=modality, visual_z=z_values,
                            visual_view=np.zeros(count, dtype=np.int64),
                            visual_patch_row=patch_rows.astype(np.int64),
                            visual_patch_col=patch_cols.astype(np.int64),
                            selected_z=np.asarray(selected, dtype=np.int64),
                            adm_applied=np.asarray(bool(adm_enabled)))
        rows.append({"study_id": study.study_id,
                     "array_path": f"visual/{study.study_id}.npz",
                     "num_tokens": count, "d_visual": d_visual,
                     "selected_z": list(selected), "adm_applied": bool(adm_enabled),
                     "input_sha256": {"t2f": str(index)}})
        image_hashes[study.study_id] = {"t2f": str(index)}
    _write_index(root, rows)
    adm = {"enabled": adm_enabled, "gamma": gamma, "threshold": threshold,
           "mode": "soft", "normal_fallback": "original", "oracle_mask": False}
    sampling = {"topk_slices": len(selected), "source_sequence": "t2f"}
    visual = {"backbone": "resnet101", "patch_grid": 2, "output_dim": d_visual,
              "modalities": ["t1n", "t1c"]}
    identity = resnet_visual_identity(localization_identity, "resnet-sha", adm,
                                      sampling, visual, image_hashes)
    meta = {"format": VISUAL_FORMAT, "identity": identity,
            "localization_identity": localization_identity,
            "resnet_weights_sha256": "resnet-sha", "adm": adm,
            "sampling": sampling, "visual": visual, "d_visual": d_visual,
            "tokens_per_study": count,
            "studies": [study.study_id for study in studies],
            "image_sha256": image_hashes}
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                    encoding="utf-8")
    return meta


def write_semantic_cache(root: Path, studies: list[Study],
                         localization_identity: str, d_proto: int = 4,
                         concept_ids: list[str] | None = None) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "semantic").mkdir(exist_ok=True)
    concepts = list(concept_ids or CONCEPTS)
    rows = []
    image_hashes: dict[str, dict[str, str]] = {}
    for index, study in enumerate(studies):
        rng = np.random.default_rng(200 + index)
        prototypes = rng.normal(size=(len(concepts), d_proto)).astype(np.float32)
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
        probabilities = rng.random(len(concepts)).astype(np.float32)
        np.savez_compressed(root / "semantic" / f"{study.study_id}.npz",
                            concept_ids=np.asarray(concepts),
                            text_prototypes=prototypes,
                            probabilities=probabilities,
                            confidence=np.abs(2 * probabilities - 1).astype(np.float32),
                            margins=(probabilities - 0.5).astype(np.float32))
        rows.append({"study_id": study.study_id,
                     "array_path": f"semantic/{study.study_id}.npz",
                     "directories": concepts,
                     "input_sha256": {"t2f": str(index)}})
        image_hashes[study.study_id] = {"t2f": str(index)}
    _write_index(root, rows)
    sampling = {"topk_slices": 2, "views_per_study": 4, "top_fraction": 0.1}
    identity = unimed_semantic_identity("code", "weights-sha", "text-sha",
                                        "prompt-sha", localization_identity,
                                        sampling, image_hashes)
    meta = {"format": SEMANTIC_FORMAT, "identity": identity,
            "localization_identity": localization_identity,
            "unimed_checkpoint_sha256": "weights-sha",
            "text_encoder_sha256": "text-sha", "prompt_bank_sha256": "prompt-sha",
            "classification_mode": "zero_shot", "d_proto": d_proto,
            "concept_ids": concepts, "sampling": sampling,
            "studies": [study.study_id for study in studies],
            "image_sha256": image_hashes}
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                    encoding="utf-8")
    return meta
