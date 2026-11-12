"""Portable cached-evidence training. Run from any extraction location."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("MEDADM_DATA_ROOT", str(ROOT / "data/RadGenome-Brain_MRI/BraTS_GLI"))


def config_for(name: str, seed: int = 0, epochs: int = 100):
    from medrrg.config import load_config
    from medrrg.errors import ConfigError
    from medrrg.runtime_guard import validate_formal
    config = load_config(f"configs/cloud/{name}.yaml")
    run_name = f"cloud_{name}_seed{seed}"
    config["experiment"]["name"] = run_name
    config["training"].update(seed=seed, epochs=epochs)
    if name == "f0":
        config["training"]["init_checkpoint"] = f"checkpoints/cloud_b1_seed{seed}/best.pt"
    if not config.path("dataset.root").is_dir():
        raise ConfigError("Dataset missing; extract the complete bundle or set MEDADM_DATA_ROOT")
    validate_formal(config)
    return config


def prepare():
    from medrrg.data.adapters import build_adapter
    from medrrg.data.manifest import build_manifest
    from medrrg.evidence import load_sources
    config = config_for("f0")
    manifest, _ = build_manifest(build_adapter(config), config.path("manifest"))
    for name in ("b0", "b1", "f0"):
        sources = load_sources(config_for(name), manifest)
        print(f"[verified] {name}: {len(manifest)} cases; "
              f"{sources.visual_tokens} visual + {sources.semantic_tokens} semantic tokens", flush=True)
    print("Patient split:", manifest.counts(), flush=True)


def benchmark(epochs: int):
    import torch
    from medrrg.training.trainer import train
    from medrrg.utils import write_json
    config = config_for("f0", epochs=epochs)
    # Time a fresh complete F0, without requiring an already trained B1.
    config["training"].update(init_checkpoint="", warmup_new_epochs=0,
                              early_stopping_patience=0)
    out = ROOT / "results/cloud_benchmark" / time.strftime("%Y%m%d_%H%M%S")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    summary = train(config, out_dir=out)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    report = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
              "epochs": epochs, "elapsed_seconds": seconds,
              "seconds_per_epoch_including_startup_and_checkpoint_io": seconds / epochs,
              "estimated_100_epoch_minutes": seconds / epochs * 100 / 60,
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
              if torch.cuda.is_available() else 0,
              "note": "Includes startup/validation/checkpoint writes. Early stopping may reduce time.",
              "best_val_loss": summary["best_val_loss"]}
    write_json(out / "timing.json", report)
    print(report, flush=True)


def train_runs(names: list[str], seed: int, epochs: int, evaluate: bool,
               overwrite: bool):
    from medrrg.training.trainer import train
    from medrrg.generation.generate import GenerationConfig, generate
    from medrrg.evaluation.run import evaluate_run
    for name in names:
        config = config_for(name, seed, epochs)
        if name == "f0" and not (ROOT / config["training"]["init_checkpoint"]).is_file():
            raise RuntimeError("F0 requires the same-seed B1 checkpoint; run --experiments b1 f0 first")
        target = ROOT / "checkpoints" / config["experiment"]["name"]
        if target.exists() and any(target.iterdir()) and not overwrite:
            raise RuntimeError(f"Output already exists: {target}; use another seed or --overwrite")
        start = time.perf_counter()
        train(config)
        print(f"[time] {name}: {(time.perf_counter()-start)/60:.2f} minutes", flush=True)
        if evaluate:
            out = ROOT / "results" / config["experiment"]["name"] / "test"
            generate(target / "best.pt", out=out, config=GenerationConfig())
            evaluate_run(out, interventions=False, concept_deltas=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "benchmark", "train"])
    parser.add_argument("--experiments", nargs="+", choices=["b0", "b1", "f0"],
                        default=["b0", "b1", "f0"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >= 1")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    os.chdir(ROOT)
    if args.mode == "prepare":
        prepare()
    elif args.mode == "benchmark":
        benchmark(args.epochs or 3)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select a working PyTorch GPU environment")
        train_runs(args.experiments, args.seed, args.epochs or 100,
                   args.evaluate, args.overwrite)


if __name__ == "__main__":
    main()
