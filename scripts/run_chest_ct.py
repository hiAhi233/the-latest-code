"""Run K=5 CT localization, report training and paired report evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
F0 = "configs/chest_ct/f0.yaml"


def run(script: str, *args: str) -> None:
    print(f"[run] {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-u", script, *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "localize", "report", "evaluate", "all"),
                        default="all")
    parser.add_argument("--locator-steps", type=int, default=600)
    parser.add_argument("--slice-batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--experiments", nargs="+", choices=("b0", "b1", "f0"), default=["b0", "f0"])
    args = parser.parse_args()
    if min(args.locator_steps, args.slice_batch, args.epochs) < 1:
        parser.error("Steps, epochs and slice batch must be positive")

    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", str(ROOT / "vendor/hf"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / "vendor/torch"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    sys.path.insert(0, str(ROOT))

    if args.stage in ("check", "localize", "all"):
        run("scripts/step01_build_manifest.py", "--config", F0)
        from external_tools.fit_chest_ct_k5 import pairs

        groups = pairs(ROOT / "data/msd_lung")
        print(f"[MSD] {[(key, len(value)) for key, value in groups.items()]}", flush=True)
    if args.stage == "check":
        return

    checkpoint = ROOT / "checkpoints/chestct_locator/checkpoint_selected.pt"
    if args.stage in ("localize", "all"):
        if checkpoint.is_file():
            record = json.loads(checkpoint.with_name("msd_localization_metrics.json").read_text(encoding="utf-8"))
            if record["steps"] != args.locator_steps:
                raise ValueError("Existing locator used a different step count; use a fresh extraction")
            print(f"[reuse] {checkpoint}", flush=True)
        else:
            run("external_tools/fit_chest_ct_k5.py", "--config", F0,
                "--steps", str(args.locator_steps), "--slice-batch", str(args.slice_batch), "--encoder-amp")
        run("external_tools/export_adm_localization.py", "--config", F0,
            "--slice-batch", str(args.slice_batch), "--encoder-amp", "--visualize", "3")

    if args.stage in ("report", "all"):
        if not checkpoint.is_file():
            raise FileNotFoundError("Run --stage localize before report training")
        if "f0" in args.experiments:
            run("external_tools/export_unimed_semantic.py", "--config", F0)
        for variant in args.experiments:
            config = f"configs/chest_ct/{variant}.yaml"
            name = f"chestct_{variant}"
            model = ROOT / "checkpoints" / name
            report_dir = ROOT / "results" / name / "test"
            run("external_tools/export_resnet_visual.py", "--config", config)
            summary = model / "train_summary.json"
            if not summary.is_file():
                run("scripts/step03_train.py", "--config", config, "--epochs", str(args.epochs))
            elif json.loads(summary.read_text(encoding="utf-8"))["epochs"] != args.epochs:
                raise ValueError(f"Existing {name} used a different epoch limit")
            if not (report_dir / "predictions_test.jsonl").is_file():
                run("scripts/step04_generate.py", "--checkpoint", str(model / "best.pt"),
                    "--out", str(report_dir), "--max-len", "512")

    if args.stage in ("evaluate", "all"):
        output = ROOT / "results" / ("chestct_eval_" + "_".join(args.experiments))
        if (output / "metrics.json").is_file():
            print(f"[metrics] {output / 'comparison.md'}", flush=True)
        else:
            if output.exists():
                from datetime import datetime
                output = output.with_name(output.name + datetime.now().strftime("_%Y%m%d_%H%M%S"))
            inputs = []
            for variant in args.experiments:
                inputs.extend(["--input", f"{variant}=results/chestct_{variant}/test/predictions_test.jsonl"])
            run("scripts/evaluate_standard_reports.py", *inputs,
                "--scorer-root", "vendor/pycocoevalcap", "--section", "findings", "--out", str(output))


if __name__ == "__main__":
    main()
