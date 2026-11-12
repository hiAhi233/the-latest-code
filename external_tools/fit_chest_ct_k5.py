"""Fit and calibrate a CT lesion localization head on the packaged MSD-Lung split."""
import argparse
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from external_tools.chest_ct_locator import (
    CROP_SIZE, FORMAT, HEAT_SIZE, hottest_mean, import_localizer, predict_volume, resize,
)
from medrrg.config import load_config
from medrrg.features.ct_windows import CT_ORIENTATION, CT_WINDOWS, ct_window, radiological_axial, select_ct_slices
from medrrg.utils import sha256_file, write_json


def pairs(root: Path) -> dict[str, list[tuple[Path, Path]]]:
    records = json.loads((root / "split.json").read_text(encoding="utf-8"))
    if set(records) != {"support", "val", "test"} or any(len(items) != 5 for items in records.values()):
        raise ValueError("Expected five separate cases for support, calibration and test")
    result = {}
    seen = set()
    for split, rows in records.items():
        result[split] = []
        for row in rows:
            image, mask = root / row["image"], root / row["label"]
            if image.name != mask.name or image.name in seen:
                raise ValueError(f"Unmatched CT/mask or repeated patient: {image.name}")
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"MSD pair missing: {image}, {mask}")
            seen.add(image.name)
            result[split].append((image, mask))
    return result


def load_pair(image_path: Path, mask_path: Path):
    image = nib.as_closest_canonical(nib.load(image_path))
    mask = nib.as_closest_canonical(nib.load(mask_path))
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-4):
        raise ValueError(f"Unaligned CT and annotation: {image_path.name}")
    return np.asarray(image.dataobj, dtype=np.float32), np.asarray(mask.dataobj) > 0


def evenly_spaced(indices: np.ndarray, limit: int) -> list[int]:
    return indices[np.linspace(0, len(indices) - 1, min(len(indices), limit))
                   .round().astype(int)].tolist()


def encode_support(items, encoder, amp):
    from paclip.localization_v3.data import extract_views, make_views

    features, targets, support = [], [], []
    for image, label in items:
        volume, mask = load_pair(image, label)
        occupied = mask.any(axis=(0, 1))
        body_area = (volume > -950).sum(axis=(0, 1))
        positive = evenly_spaced(np.flatnonzero(occupied), 12)
        negative = evenly_spaced(np.flatnonzero(~occupied & (body_area > 0.3 * body_area.max())), 8)
        if not positive or not negative:
            raise ValueError(f"Missing lesion or background slices: {image.name}")
        support.append({"case": image.name, "positive_z": positive, "negative_z": negative,
                        "image_sha256": sha256_file(image), "mask_sha256": sha256_file(label)})
        for z in positive + negative:
            gray = resize(ct_window(volume[:, :, z], "lung"), 224)
            native = torch.from_numpy(radiological_axial(mask[:, :, z]).astype(np.float32))[None, None]
            soft = F.adaptive_avg_pool2d(native, (224, 224))[0, 0].numpy()
            hard = F.adaptive_max_pool2d(native, (224, 224))[0, 0].numpy()
            views, labels = make_views(gray, soft, hard, np.ones_like(gray, dtype=bool), crop=CROP_SIZE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                features.append(extract_views(encoder, views, "cuda"))
            targets.extend(labels)
        print(f"[support] {image.name}: {len(positive)} lesion and {len(negative)} background slices",
              flush=True)
    return torch.cat(features), {key: torch.stack([row[key] for row in targets])
                                 for key in targets[0]}, support


def measure(items, encoder, head, thresholds, slice_batch, amp, split):
    records = {str(threshold): [] for threshold in thresholds}
    for image, label in items:
        volume, mask = load_pair(image, label)
        heat = predict_volume(volume, encoder, head, slice_batch, amp)
        occupied = mask.any(axis=(0, 1))
        gt = np.zeros(heat.shape, dtype=bool)
        for z in np.flatnonzero(occupied):
            axial = torch.from_numpy(radiological_axial(mask[:, :, z]).astype(np.float32))
            gt[z] = F.adaptive_max_pool2d(axial[None, None], (HEAT_SIZE, HEAT_SIZE))[0, 0].numpy() > 0
        truth = int(gt.sum())
        if not truth:
            raise ValueError(f"No lesion voxels in {image.name}")
        scores = np.asarray([hottest_mean(row, .01) for row in heat])
        top_six = select_ct_slices(scores, 6)
        peak = np.unravel_index(int(heat.argmax()), heat.shape)
        for threshold in thresholds:
            positive = heat >= threshold
            intersection = int((positive & gt).sum())
            predicted = int(positive.sum())
            records[str(threshold)].append({
                "case": image.name, "dice": 2 * intersection / (truth + predicted),
                "voxel_recall": intersection / truth, "fp_to_gt": (predicted - intersection) / truth,
                "peak_in_lesion": bool(gt[peak]),
                "six_slice_hit": bool(occupied[top_six].any()),
            })
        print(f"[{split}] {image.name} depth={len(heat)}", flush=True)
    keys = ("dice", "voxel_recall", "fp_to_gt", "peak_in_lesion", "six_slice_hit")
    return {threshold: {"macro": {key: float(np.mean([row[key] for row in rows]))
                                 for key in keys}, "cases": rows}
            for threshold, rows in records.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/chest_ct/f0.yaml")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slice-batch", type=int, default=8)
    parser.add_argument("--encoder-amp", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.slice_batch < 1:
        parser.error("Positive steps and slice batch required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    groups = pairs(config.path("localization.msd_root"))
    checkpoint = config.path("localization.checkpoint")
    if checkpoint.exists():
        raise FileExistsError(f"Existing localization checkpoint: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    import_localizer(config.path("localization.detector_root"))
    from paclip.localization_v3.loss import recall_loss
    from paclip.localization_v3.model import FrozenEncoder, RecallDenseHead

    encoder = FrozenEncoder().cuda().eval()
    features, targets, support = encode_support(groups["support"], encoder, args.encoder_amp)
    lesion_view = targets["hard"].flatten(1).any(dim=1).numpy()
    positives, negatives = np.flatnonzero(lesion_view), np.flatnonzero(~lesion_view)
    if not len(positives) or not len(negatives):
        raise ValueError("K=5 requires both lesion and background views")
    head = RecallDenseHead().cuda().train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-5)
    history = []
    for step in range(1, args.steps + 1):
        indices = np.r_[rng.choice(positives, 4), rng.choice(negatives, 4)]
        logits = head(features[indices].cuda().float())
        loss = recall_loss(logits, {key: value[indices].cuda() for key, value in targets.items()})["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            history.append({"step": step, "loss": float(loss.detach())})
            print(f"[fit] {step}/{args.steps}: {history[-1]['loss']:.5f}", flush=True)
    del features, targets
    head.eval()
    thresholds = tuple(round(i / 20, 2) for i in range(2, 19))
    val = measure(groups["val"], encoder, head, thresholds, args.slice_batch, args.encoder_amp, "val")
    best = max(thresholds, key=lambda t: (val[str(t)]["macro"]["dice"], t))
    test = measure(groups["test"], encoder, head, (best,), args.slice_batch, args.encoder_amp, "test")
    protocol = {
        "k": 5, "dataset": "MSD Task06 Lung", "seed": args.seed, "steps": args.steps,
        "orientation": CT_ORIENTATION, "lung_window_hu": CT_WINDOWS["lung"],
        "encoder_amp": args.encoder_amp, "support": support, "calibration": "macro Dice",
        "threshold": best, "validation": val, "test": test[str(best)],
        "elapsed_seconds": time.monotonic() - started, "history": history,
    }
    write_json(checkpoint.parent / "msd_localization_metrics.json", protocol)
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"format": FORMAT, "head": {key: value.cpu() for key, value in head.state_dict().items()},
                "choice": {"threshold": best}, "protocol": protocol}, temporary)
    temporary.replace(checkpoint)
    print(f"[selected] {checkpoint} threshold={best} test={test[str(best)]['macro']}", flush=True)


if __name__ == "__main__":
    main()
