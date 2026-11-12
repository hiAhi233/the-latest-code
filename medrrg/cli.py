"""统一命令行入口（方案 §12）。

prepare -> export(step02a/b/c) -> validate -> train -> generate -> evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

USAGE = __doc__ or ""


def _config(path: str) -> Any:
    from .config import load_config

    return load_config(path)


def _adapter(config: Any) -> Any:
    from .data.adapters import build_adapter

    return build_adapter(config)


def _manifest(config: Any) -> Any:
    from .data.manifest import load_manifest

    return load_manifest(config.path("manifest"))


def cmd_prepare(args: argparse.Namespace) -> int:
    from .data.adm_dataset import WordTokenizer
    from .data.manifest import build_manifest

    config = _config(args.config)
    adapter = _adapter(config)
    manifest_path = config.path("manifest")
    manifest, report = build_manifest(adapter, manifest_path)
    max_len = int(config.get_path("tokenizer.max_len", 256))
    tokenizer = WordTokenizer.build(
        [study.report.target_text(control_tokens=True)
         for study in manifest.split("train")],
        min_freq=int(config.get_path("tokenizer.min_freq", 1)),
        max_vocab=int(config.get_path("tokenizer.max_vocab", 12000)),
        max_len=max_len)
    token_lengths: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        lengths = [len(tokenizer.encode(study.report.target_text(control_tokens=True),
                                        max_len=10 ** 6))
                   for study in manifest.split(split)]
        if lengths:
            ordered = sorted(lengths)
            token_lengths[split] = {
                "count": len(lengths),
                "p50": ordered[len(ordered) // 2],
                "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
                "max": ordered[-1],
                "truncated_at_max_len": sum(1 for value in lengths if value > max_len),
                "max_len": max_len,
            }
    summary = {"manifest": str(manifest_path), **manifest.summary(),
               "validation": report, "token_lengths": token_lengths}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .runtime_guard import validate_formal
    from .training.trainer import train

    config = _config(args.config)
    validate_formal(config)
    if args.overfit_cases:
        config["training"] = dict(config.get("training") or {})
        config["training"]["epochs"] = int(config.get_path("overfit.epochs", 200))
        config["training"]["batch_size"] = int(config.get_path("overfit.batch_size", 2))
        config["training"]["lr"] = float(config.get_path("overfit.lr", 1e-3))
        config["training"]["grad_accum"] = int(config.get_path("overfit.grad_accum", 1))
    # 显式命令行参数优先于 overfit 预设
    if getattr(args, "epochs", None):
        config["training"] = dict(config.get("training") or {})
        config["training"]["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None):
        config["training"] = dict(config.get("training") or {})
        config["training"]["batch_size"] = int(args.batch_size)
    if getattr(args, "lr", None):
        config["training"] = dict(config.get("training") or {})
        config["training"]["lr"] = float(args.lr)
    summary = train(config, overfit_cases=args.overfit_cases, out_dir=args.out)
    printable = {key: value for key, value in summary.items() if key != "history"}
    print(json.dumps(printable, ensure_ascii=False, indent=1, default=str))
    history = summary.get("history", [])
    if history:
        print(f"[train] best val_loss={summary['best_val_loss']:.4f} "
              f"@ epoch {summary['best_epoch']}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .generation.generate import GenerationConfig, generate

    summary = generate(
        args.checkpoint, split=args.split, out=args.out,
        config=GenerationConfig(max_len=args.max_len, mode=args.mode,
                                beam=args.beam, batch_size=args.batch_size,
                                device=args.device),
        limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation.run import evaluate_run

    metrics = evaluate_run(args.run, interventions=bool(args.interventions),
                           n_resamples=args.boot,
                           concept_deltas=bool(args.concept_deltas))
    summary = {"num_examples": metrics["num_examples"],
               "nlg": metrics["nlg"], "concepts": metrics["concepts"],
               "hallucination": metrics["hallucination"],
               "sections": metrics["sections"]}
    if "interventions" in metrics:
        summary["interventions"] = metrics["interventions"]
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(f"[evaluate] 完整指标 -> {metrics['metrics_path']}")
    return 0


def cmd_evaluate_evidence(args: argparse.Namespace) -> int:
    from .evidence import load_sources
    from .evaluation.upstream import localization_metrics, semantic_metrics
    from .utils import read_json, write_json

    config = _config(args.config)
    manifest = _manifest(config)
    adapter = _adapter(config)
    sources = load_sources(config, manifest,
                           use_semantic=bool(config.get_path("model.use_semantic", True)),
                           require_predicted=not bool(config.get_path("adm.oracle_mask",
                                                                     False)))
    splits = tuple(args.splits.split(",")) if args.splits else ("val", "test")
    prompts = {}
    prompt_path = config.get_path("semantic.prompts", "")
    if prompt_path:
        from pathlib import Path

        from .config import project_root

        path = Path(str(prompt_path))
        if not path.is_absolute():
            path = project_root() / path
        if path.is_file():
            prompts = read_json(path)
    metrics = {
        "localization": localization_metrics(manifest, adapter, sources.localization,
                                             splits=splits, limit=args.limit),
        "semantic": (semantic_metrics(manifest, adapter.ontology, sources.semantic,
                                      splits=splits,
                                      exclusive_groups=prompts.get("exclusive_groups"),
                                      limit=args.limit)
                     if sources.semantic is not None else {"num_examples": 0}),
        "manifest_sha256": manifest.sha256,
        "ontology_sha256": adapter.ontology.sha256,
        "cache_identity": sources.cache_identity,
    }
    out = config.path("results_root", "results") / "evidence_quality.json"
    write_json(out, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=1, default=str))
    print(f"[evidence-quality] -> {out}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import torch

    from .utils import resolve_device

    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "devices": [torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())],
        "device": resolve_device("auto"),
    }
    if args.config:
        config = _config(args.config)
        adapter = _adapter(config)
        payload.update({
            "config_sources": config.sources,
            "dataset_root": str(adapter.root),
            "dataset_exists": adapter.root.is_dir(),
            "ontology": adapter.ontology.to_dict(),
            "manifest": str(config.path("manifest")),
            "manifest_exists": config.path("manifest").is_file(),
            "evidence_dirs": {
                "localization": str(config.get_path("evidence.localization_dir", "")),
                "visual": str(config.get_path("evidence.visual_dir", "")),
                "semantic": str(config.get_path("evidence.semantic_dir", "")),
            },
        })
    print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """合成 batch 冒烟：验证安装与 token 装配，不读医学数据。"""
    import torch

    from .contracts import AdmEvidenceBatch
    from .data.adm_dataset import WordTokenizer, collate_adm_samples
    from .model.report_model import AdmEvidenceReportTransformer, AdmReportModelConfig

    torch.manual_seed(0)
    tokenizer = WordTokenizer.build([
        "<FINDINGS> a lesion with edema </FINDINGS> "
        "<IMPRESSION> glioma </IMPRESSION>"])
    config = AdmReportModelConfig(
        d_model=64, heads=4, encoder_layers=1, decoder_layers=2, ffn=128,
        visual_tokens=9, semantic_tokens=4, d_visual=16, d_proto=16,
        dropout=0.0, use_semantic=True, evidence_dropout=0.0, max_report_tokens=32)
    model = AdmEvidenceReportTransformer(config, vocab_size=tokenizer.vocab_size,
                                        pad_id=tokenizer.pad_id)
    batch = 2
    count, concepts = 9, 4
    evidence = AdmEvidenceBatch(
        study_ids=["smoke-a", "smoke-b"], patient_ids=["p1", "p2"],
        visual_tokens=torch.randn(batch, count, 16),
        visual_mask=torch.ones(batch, count, dtype=torch.bool),
        visual_modality=torch.arange(count).remainder(4).unsqueeze(0).repeat(batch, 1),
        visual_z=torch.rand(batch, count),
        visual_view=torch.zeros(batch, count, dtype=torch.long),
        visual_available=torch.ones(batch, dtype=torch.bool),
        semantic_prototypes=torch.randn(batch, concepts, 16),
        semantic_scores=torch.rand(batch, concepts),
        semantic_confidence=torch.rand(batch, concepts),
        semantic_margin=torch.rand(batch, concepts),
        semantic_mask=torch.ones(batch, concepts, dtype=torch.bool),
        semantic_available=torch.ones(batch, dtype=torch.bool))
    evidence.assert_adm_contract()
    ids = tokenizer.encode("<FINDINGS> lesion </FINDINGS> <IMPRESSION> glioma </IMPRESSION>",
                           max_len=12)
    output = model(evidence, torch.tensor([ids[:-1]] * batch),
                   torch.tensor([ids[1:]] * batch))
    output["loss"].backward()
    generated = model.generate(evidence, max_len=6, bos_id=tokenizer.bos_id,
                               eos_id=tokenizer.eos_id)
    print(json.dumps({
        "ok": True,
        "memory_shape": list(output["memory"].shape),
        "logits_shape": list(output["logits"].shape),
        "loss": float(output["loss"].detach()),
        "generated_lengths": [len(item) for item in generated],
        "note": "合成数据，仅验证安装与链路",
    }, ensure_ascii=False, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medadm", description=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="构建并验证 manifest")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="训练 B0/B1/F0")
    p.add_argument("--config", required=True)
    p.add_argument("--overfit-cases", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="覆盖 training.epochs（快速自检）")
    p.add_argument("--batch-size", type=int, default=None, help="覆盖 training.batch_size")
    p.add_argument("--lr", type=float, default=None, help="覆盖 training.lr")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("generate", help="生成报告")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--out", default=None)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--mode", default="greedy", choices=("greedy", "beam"))
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("evaluate", help="文本/概念/幻觉/证据干预评价")
    p.add_argument("--run", required=True)
    p.add_argument("--interventions", action="store_true")
    p.add_argument("--concept-deltas", action="store_true")
    p.add_argument("--boot", type=int, default=1000)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("evaluate-evidence", help="定位 Dice/召回/注意力覆盖与 UniMed AUROC")
    p.add_argument("--config", required=True)
    p.add_argument("--splits", default="val,test")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_evaluate_evidence)

    p = sub.add_parser("doctor", help="环境/路径自检")
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("smoke", help="合成 batch 冒烟")
    p.set_defaults(func=cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
