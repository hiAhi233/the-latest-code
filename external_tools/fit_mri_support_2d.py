"""Adapt the MedADM MRI lesion head using 2D normal/abnormal support slices."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# This is the localization module used by MedADM_RRG, not the PA-clip training entry.
DETECTOR_ROOT = Path("/root/autodl-tmp/medadm_k5/detector")
sys.path.insert(0, str(DETECTOR_ROOT))
import run_localization  # noqa: F401,E402  establishes paclip package isolation
from paclip.localization_v3.foreground_experiment import (  # noqa: E402
    FrozenEncoder, extract_views, initialize_environment, make_views,
)
from paclip.localization_v3.loss import recall_loss  # noqa: E402
from paclip.localization_v3.model import RecallDenseHead  # noqa: E402

ROOT = Path("/root/autodl-tmp/medadm_k5/MedADM_RRG")
DATA = Path("/root/autodl-tmp/paclip_k32/brain_mri/train")
MASKS = Path("/root/autodl-tmp/paclip_k32/brain_mri_masks/train")
BASE = ROOT / "localization_adaptation/checkpoint_selected.pt"
OUTPUT = ROOT / "localization_adaptation"

K5_NORMAL = ["00033_99.png", "00321_97.png"]
K5_ABNORMAL = ["00154_90.png", "01597_80.png", "01597_90.png"]


def read_image(path: Path, size: int = 224) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def read_mask(path: Path, size: int = 224) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(image) > 0


def choose_support(mode: str):
    if mode == "k5":
        normal = [DATA / "normal" / name for name in K5_NORMAL]
        abnormal = [DATA / "abnormal" / name for name in K5_ABNORMAL]
    else:
        normal_files = sorted((DATA / "normal").glob("*.png"))
        by_case = {}
        for path in normal_files:
            case = path.name.split("_")[0]
            by_case.setdefault(case, path)
        normal = [by_case[case] for case in sorted(by_case)[:16]]
        abnormal = sorted((DATA / "abnormal").glob("*.png"))[:16]
    for path in normal + abnormal:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in abnormal:
        if not (MASKS / "abnormal" / path.name).is_file():
            raise FileNotFoundError(MASKS / "abnormal" / path.name)
    return normal, abnormal


def extract_support(encoder, normal, abnormal, crop: int):
    features, targets, metadata = [], [], []
    for label, paths in ((0, normal), (1, abnormal)):
        for path in paths:
            gray = read_image(path)
            if label:
                hard = read_mask(MASKS / "abnormal" / path.name)
                soft = hard.astype(np.float32)
            else:
                hard = np.zeros((224, 224), dtype=bool)
                soft = np.zeros((224, 224), dtype=np.float32)
            roi = (gray > 0) | hard
            images, labels = make_views(gray, soft, hard, roi, crop=crop)
            values = extract_views(encoder, images, "cuda")
            kept = 0
            for view, (feature, target) in enumerate(zip(values, labels)):
                if target["roi"].any():
                    features.append(feature)
                    targets.append(target)
                    metadata.append({"file": path.name, "label": label, "view": view,
                                     "positive": bool(target["hard"].any())})
                    kept += 1
            print(f"[support] {path.name} label={label} views={kept}", flush=True)
    return (torch.stack(features),
            {key: torch.stack([row[key] for row in targets]) for key in targets[0]},
            metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("k32", "k5"), required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    initialize_environment()
    normal, abnormal = choose_support(args.mode)
    out = OUTPUT / args.mode
    if out.exists():
        raise FileExistsError(f"Output exists: {out}")
    out.mkdir(parents=True)
    support_dir = out / "support_2d"
    for kind, paths in (("normal", normal), ("abnormal", abnormal)):
        dest = support_dir / kind
        dest.mkdir(parents=True)
        for path in paths:
            shutil.copy2(path, dest / path.name)
            if kind == "abnormal":
                shutil.copy2(MASKS / "abnormal" / path.name, dest / path.name)

    old = torch.load(BASE, map_location="cpu", weights_only=False)
    if old.get("format") != "foreground_separated_v1":
        raise ValueError("Unexpected base checkpoint format")
    crop = int(old["config"]["crop_size"])
    threshold = float(old["choice"]["threshold"])
    encoder = FrozenEncoder().cuda().eval()
    features, targets, metadata = extract_support(encoder, normal, abnormal, crop)
    del encoder
    torch.cuda.empty_cache()
    positives = [i for i, row in enumerate(metadata) if row["positive"]]
    negatives = [i for i, row in enumerate(metadata) if not row["positive"]]
    if not positives or not negatives:
        raise RuntimeError(f"Both classes required: positives={len(positives)}, negatives={len(negatives)}")
    print(f"[fit] support={len(normal)} normal + {len(abnormal)} abnormal, "
          f"positive_views={len(positives)}, negative_views={len(negatives)}", flush=True)

    head = RecallDenseHead().cuda().train()
    head.load_state_dict(old["head"], strict=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    history = []
    for step in range(1, args.steps + 1):
        idx = np.r_[rng.choice(positives, 4), rng.choice(negatives, 4)]
        ix = torch.as_tensor(idx)
        result = recall_loss(head(features[ix].cuda()),
                             {key: value[ix].cuda() for key, value in targets.items()})
        loss = result["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            history.append({"step": step, "loss": float(loss.detach())})
            print(f"[fit] {step}/{args.steps}: loss={history[-1]['loss']:.5f}", flush=True)
        if step in (200, 600) or step == args.steps:
            state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
            checkpoint = {
                "format": old["format"],
                "head": state,
                "foreground": old["foreground"],
                "config": old["config"],
                "choice": {"threshold": threshold},
                "step": step,
                "support_2d": {
                    "mode": args.mode,
                    "normal": [p.name for p in normal],
                    "abnormal": [p.name for p in abnormal],
                    "normal_count": len(normal),
                    "abnormal_count": len(abnormal),
                    "positive_views": len(positives),
                    "negative_views": len(negatives),
                    "seed": args.seed,
                    "steps": args.steps,
                    "base_checkpoint": str(BASE),
                },
            }
            torch.save(checkpoint, out / f"step{step}.pt")
            (out / f"step{step}.json").write_text(json.dumps(checkpoint["support_2d"], indent=2), encoding="utf-8")
    (out / "training.json").write_text(json.dumps({
        "mode": args.mode,
        "normal": [p.name for p in normal],
        "abnormal": [p.name for p in abnormal],
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2), encoding="utf-8")
    torch.save({"format": old["format"], "head": head.state_dict(),
                "foreground": old["foreground"], "config": old["config"],
                "choice": {"threshold": threshold}, "step": args.steps,
                "support_2d": {"mode": args.mode}}, out / "best.pt")
    print(f"[done] {out}/best.pt", flush=True)


if __name__ == "__main__":
    main()
