"""端到端链路自检：合成缓存 → 训练 → 生成 → 指标/干预（CPU，不需要真实数据）。

覆盖 trainer / checkpoint / evidence 身份链 / generate / evaluate / interventions
之间的接口，避免单元测试都过但拼不起来。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medrrg.config import Config
from medrrg.evidence import load_sources
from medrrg.features.adm_cache import AdmVisualCache
from medrrg.training.checkpoint import load_checkpoint


def _config(tmp: Path, manifest_path: Path, evidence_dirs, epochs: int = 2) -> Config:
    return Config({
        "manifest": str(manifest_path),
        "checkpoint_root": str(tmp / "checkpoints"),
        "results_root": str(tmp / "results"),
        "experiment": {"name": "smoke"},
        "evidence": {"localization_dir": str(evidence_dirs["localization"]),
                     "visual_dir": str(evidence_dirs["visual"]),
                     "semantic_dir": str(evidence_dirs["semantic"])},
        "visual": {"backbone": "resnet101", "output_dim": 8, "patch_grid": 2},
        "model": {"evidence_mode": "adm_simple", "d_model": 32, "heads": 4,
                  "encoder_layers": 1, "decoder_layers": 1, "ffn": 64,
                  "dropout": 0.0, "visual_tokens": 16, "semantic_tokens": 3,
                  "d_visual": 8, "d_proto": 4, "num_modalities": 4,
                  "num_views": 2, "z_buckets": 4, "evidence_dropout": 0.0,
                  "max_report_tokens": 48, "use_semantic": True,
                  "use_geometry": False, "use_roi_branch": False},
        "training": {"epochs": epochs, "batch_size": 2, "grad_accum": 1, "lr": 1e-3,
                     "weight_decay": 0.0, "warmup_ratio": 0.0, "label_smoothing": 0.0,
                     "early_stopping_patience": 0, "amp": False, "device": "cpu",
                     "num_workers": 0, "seed": 0},
        "tokenizer": {"max_len": 48, "min_freq": 1, "max_vocab": 500},
        "runtime": {"require_zero_shot": True},
        "semantic": {"num_concepts": 3, "prompts": ""},
        "adm": {"enabled": True, "gamma": 0.4, "threshold": 0.6, "mode": "soft",
                "normal_fallback": "original", "oracle_mask": False},
        "ontology": {"name": "smoke", "language": "en", "concepts": {
            "brain_tumor": ["lesion", "tumor", "mass"],
            "enhancement": ["enhancement", "enhancing"],
            "edema": ["edema", "oedema"]}},
    }, ["<test>"])


def test_train_generate_evaluate_intervene(tmp_path, manifest, evidence_dirs):
    from medrrg.data.manifest import load_manifest
    from medrrg.evaluation.run import evaluate_run
    from medrrg.evaluation.evidence_intervention import run_interventions
    from medrrg.generation.generate import GenerationConfig, generate
    from medrrg.training.trainer import train

    manifest_path = Path(manifest.path)
    config = _config(tmp_path, manifest_path, evidence_dirs)
    sources = load_sources(config, load_manifest(manifest_path))
    assert sources.visual.tokens_per_study == 16

    summary = train(config, out_dir=tmp_path / "checkpoints" / "smoke", verbose=False)
    assert summary["trainable_parameters"] > 0
    checkpoint = Path(summary["best"]["checkpoint"])
    assert checkpoint.is_file()
    loaded = load_checkpoint(checkpoint, device="cpu")
    assert loaded["config"].use_semantic is True

    generation = generate(checkpoint, split="train",
                          out=tmp_path / "results" / "smoke" / "train",
                          config=GenerationConfig(max_len=24, batch_size=2,
                                                  device="cpu"),
                          verbose=False)
    assert generation["count"] == 2
    run_dir = Path(generation["out"]).parent
    metrics = evaluate_run(run_dir, interventions=False)
    assert metrics["num_examples"] == 2
    assert "bleu_4" in metrics["nlg"]
    assert "micro_f1" in metrics["concepts"]
    assert 0.0 <= metrics["sections"]["both_sections_rate"] <= 1.0

    # 干预：null_adm / swap_adm / null_semantic / swap_semantic 都应可执行
    from medrrg.data.adm_dataset import (AdmDatasetConfig, AdmEvidenceDataset)
    from medrrg.features.adm_cache import AdmSemanticCache
    dataset = AdmEvidenceDataset(
        load_manifest(manifest_path), "train", loaded["tokenizer"],
        AdmVisualCache.load(evidence_dirs["visual"]),
        AdmDatasetConfig(max_report_tokens=48, use_semantic=True,
                         d_visual=8, d_proto=4),
        semantic=AdmSemanticCache.load(evidence_dirs["semantic"]),
        localization=sources.localization)
    report = run_interventions(loaded["model"], dataset, "cpu", batch_size=2,
                               n_resamples=20, concept_deltas=False)
    assert set(report["conditions"]) == {"null_adm", "swap_adm",
                                        "null_semantic", "swap_semantic"}
    for condition in report["conditions"].values():
        assert condition["num_patients"] == 2
