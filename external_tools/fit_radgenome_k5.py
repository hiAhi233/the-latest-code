"""Adapt the existing MRI lesion head using five patient-disjoint training cases."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_tools.export_adm_localization import _normalize_slice, _resize
from medrrg.config import load_config
from medrrg.data.adapters import build_adapter
from medrrg.data.manifest import load_manifest
from medrrg.utils import sha256_file, write_json


def spaced(indices: np.ndarray, count: int) -> list[int]:
    positions = np.linspace(0, len(indices) - 1, min(count, len(indices))).round().astype(int)
    return [int(z) for z in indices[positions]]


def support_cases(manifest, rng):
    by_patient = {}
    for study in sorted(manifest.split("train"), key=lambda row: row.study_id):
        by_patient.setdefault(study.patient_id, study)
    choices = rng.choice(sorted(by_patient), 5, replace=False)
    return [by_patient[patient] for patient in sorted(choices)]


def support_features(studies, adapter, encoder, make_views, extract_views, crop):
    features, targets, metadata, cases = [], [], [], []
    for study in studies:
        flair_path = adapter.resolve(next(ref.path for ref in study.images if ref.group == "t2f"))
        mask_path = adapter.resolve(study.meta["seg_path"])
        flair = nib.as_closest_canonical(nib.load(str(flair_path)))
        mask = nib.as_closest_canonical(nib.load(str(mask_path)))
        if flair.shape != mask.shape or not np.allclose(flair.affine, mask.affine):
            raise ValueError(f"Image and mask geometry differ: {study.study_id}")
        volume = np.asanyarray(flair.dataobj, dtype=np.float32)
        gt = np.asanyarray(mask.dataobj) > 0
        area = gt.sum(axis=(0, 1))
        brain = (volume > 0).sum(axis=(0, 1))
        positive = spaced(np.flatnonzero(area > 0), 12)
        negative = spaced(np.flatnonzero((area == 0) & (brain >= 0.15 * brain.max())), 6)
        if not positive:
            raise ValueError(f"No tumor slices in support case {study.study_id}")
        cases.append({"study_id": study.study_id, "patient_id": study.patient_id,
                      "positive_z": positive, "negative_z": negative,
                      "image_sha256": sha256_file(flair_path),
                      "seg_sha256": sha256_file(mask_path)})
        for z in sorted(positive + negative):
            gray = _resize(_normalize_slice(volume[:, :, z]), (224, 224))
            hard = _resize(gt[:, :, z], (224, 224), nearest=True)
            soft = _resize(gt[:, :, z], (224, 224))
            roi = (gray > 0) | (hard > 0) | (soft > 0)
            images, labels = make_views(gray, soft, hard, roi, crop=crop)
            values = extract_views(encoder, images, "cuda")
            for view, (feature, target) in enumerate(zip(values, labels)):
                if target["roi"].any():
                    features.append(feature)
                    targets.append(target)
                    metadata.append({"study_id": study.study_id, "z": z, "view": view,
                                     "positive": bool(target["hard"].any())})
        print(f"[support] {study.study_id}: {len(positive)} lesion, {len(negative)} negative slices",
              flush=True)
    return torch.stack(features), {key: torch.stack([row[key] for row in targets])
                                   for key in targets[0]}, metadata, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adm_radgenome_f0.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", nargs="+", type=int, default=[200, 600])
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if min(args.steps) < 1:
        parser.error("--steps values must be positive")
    config = load_config(args.config)
    manifest = load_manifest(config.path("manifest"))
    manifest.verify_patients()
    adapter = build_adapter(config)
    source = config.path("localization.checkpoint")
    args.out.mkdir(parents=True, exist_ok=False)
    source_sha = sha256_file(source)
    sys.path.insert(0, str(config.path("localization.detector_root")))
    import run_localization  # Establish upstream paclip package.
    from paclip.localization_v3.foreground_experiment import (
        FrozenEncoder, extract_views, initialize_environment, make_views)
    from paclip.localization_v3.loss import recall_loss
    from paclip.localization_v3.model import RecallDenseHead

    initialize_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    chosen = support_cases(manifest, rng)
    old = torch.load(source, map_location="cpu", weights_only=False)
    if old.get("format") != "foreground_separated_v1":
        raise ValueError("Expected existing foreground_separated_v1 checkpoint")
    started = time.monotonic()
    encoder = FrozenEncoder().cuda().eval()
    features, targets, meta, support = support_features(
        chosen, adapter, encoder, make_views, extract_views, old["config"]["crop_size"])
    del encoder
    torch.cuda.empty_cache()
    positives = [i for i, row in enumerate(meta) if row["positive"]]
    negatives = [i for i, row in enumerate(meta) if not row["positive"]]
    if not positives or not negatives:
        raise ValueError("Both lesion and negative views are needed")
    protocol = {"k_cases": 5, "seed": args.seed, "support": support,
                "base_checkpoint_sha256": source_sha, "manifest_sha256": manifest.sha256,
                "steps": sorted(set(args.steps)), "lr": args.lr,
                "trainable": "RecallDenseHead", "frozen": ["BiomedCLIP", "ForegroundHead"],
                "threshold": float(old["choice"]["threshold"]),
                "positive_views": len(positives), "negative_views": len(negatives),
                "validation_selection": "macro Dice at original threshold",
                "support_split": "train"}
    write_json(args.out / "protocol.json", protocol)
    print(f"[fit] {len(positives)} positive and {len(negatives)} negative views", flush=True)
    head = RecallDenseHead().cuda().train()
    head.load_state_dict(old["head"], strict=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)
    history = []
    for step in range(1, max(args.steps) + 1):
        indices = np.r_[rng.choice(positives, 4), rng.choice(negatives, 4)]
        logits = head(features[indices].cuda())
        loss = recall_loss(logits, {key: value[indices].cuda() for key, value in targets.items()})["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        if step % 50 == 0 or step in args.steps:
            history.append({"step": step, "loss": float(loss.detach())})
            print(f"[fit] {step}/{max(args.steps)} loss={history[-1]['loss']:.5f}", flush=True)
        if step in args.steps:
            path = args.out / f"step{step}.pt"
            state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
            torch.save({"format": old["format"], "head": state, "foreground": old["foreground"],
                        "config": old["config"], "choice": {"threshold": protocol["threshold"]},
                        "step": step, "radgenome_adaptation": protocol}, path)
            write_json(args.out / f"step{step}.yaml", {
                "defaults": ["configs/adm_radgenome_f0.yaml"],
                "localization": {"checkpoint": str(path.resolve())},
            })
    write_json(args.out / "training.json", {"history": history,
                                           "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
