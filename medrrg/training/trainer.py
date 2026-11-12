"""ADM 报告训练器：只做报告交叉熵（方案 §7D）。

冻结上游（定位器、ResNet-101、UniMed-CLIP）；训练内容：

* visual patch projection（Linear+LayerNorm）
* modality / z / view / type embedding
* SemanticProjector
* 2 层 Encoder、4 层 Decoder

B0/B1/F0 共用本训练器；F0 可用 ``training.init_checkpoint`` 从 B1 初始化，
前 ``warmup_new_epochs`` 个 epoch 冻结 decoder 只训练语义投影器。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..config import Config, project_root
from ..data.adm_dataset import (
    AdmDatasetConfig, AdmEvidenceDataset, WordTokenizer, collate_adm_samples,
)
from ..data.manifest import load_manifest
from ..errors import CacheError, ConfigError, DataError
from ..evidence import load_sources, source_paths
from ..features.cache_identity import report_checkpoint_identity
from ..model.report_model import AdmEvidenceReportTransformer, AdmReportModelConfig
from ..utils import (append_jsonl, now_iso, resolve_device, seed_everything,
                     sha256_json, write_json)
from .checkpoint import load_checkpoint, save_checkpoint

__all__ = ["TrainSettings", "train"]


def _source_tree_hash() -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted((project_root() / "medrrg").rglob("*.py")):
        digest.update(str(path.relative_to(project_root())).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    return {"created_at": now_iso(), "python_torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "source_tree_hash": _source_tree_hash()}


@dataclass
class TrainSettings:
    epochs: int = 100
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    warmup_new_epochs: int = 0
    init_checkpoint: str = ""
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0
    grad_accum: int = 2
    warmup_ratio: float = 0.05
    early_stopping_patience: int = 12
    amp: bool = True
    freeze_visual_projection: bool = False

    @classmethod
    def from_config(cls, config: Config) -> "TrainSettings":
        raw = dict(config.get("training") or {})
        return cls(
            epochs=int(raw.get("epochs", 100)),
            batch_size=int(raw.get("batch_size", 8)),
            lr=float(raw.get("lr", 2e-4)),
            weight_decay=float(raw.get("weight_decay", 0.01)),
            grad_clip=float(raw.get("grad_clip", 1.0)),
            label_smoothing=float(raw.get("label_smoothing", 0.05)),
            warmup_new_epochs=int(raw.get("warmup_new_epochs", 0)),
            init_checkpoint=str(raw.get("init_checkpoint", "")),
            device=str(raw.get("device", config.get_path("device", "auto"))),
            seed=int(raw.get("seed", config.get_path("seed", 0))),
            num_workers=int(raw.get("num_workers", 0)),
            grad_accum=max(1, int(raw.get("grad_accum", 1))),
            warmup_ratio=max(0.0, min(1.0, float(raw.get("warmup_ratio", 0.0)))),
            early_stopping_patience=max(0, int(raw.get("early_stopping_patience", 0))),
            amp=bool(raw.get("amp", True)),
            freeze_visual_projection=bool(raw.get("freeze_visual_projection", False)))


@torch.no_grad()
def evaluate_loss(model: AdmEvidenceReportTransformer, loader: DataLoader,
                  device: str) -> dict[str, float]:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        evidence = batch["evidence"].to(device)
        output = model(evidence, batch["decoder_input_ids"].to(device),
                       batch["labels"].to(device))
        total += float(output["loss"]) * len(batch["study_ids"])
        count += len(batch["study_ids"])
    model.train()
    return {"val_loss": total / max(1, count)}


def train(config: Config, overfit_cases: int | None = None,
          out_dir: str | Path | None = None, verbose: bool = True) -> dict[str, Any]:
    settings = TrainSettings.from_config(config)
    seed_everything(settings.seed)
    device = resolve_device(settings.device)
    manifest = load_manifest(config.path("manifest"))
    manifest.verify_patients()
    use_semantic = bool(config.get_path("model.use_semantic", True))
    allow_oracle = bool(config.get_path("adm.oracle_mask", False))
    sources = load_sources(config, manifest, use_semantic=use_semantic,
                           require_predicted=not allow_oracle)

    train_studies = manifest.split("train")
    val_studies = manifest.split("val")
    if overfit_cases:
        train_studies = train_studies[: int(overfit_cases)]
        val_studies = train_studies
    if not train_studies:
        raise DataError("没有训练病例")
    if not val_studies:
        val_studies = train_studies

    tokenizer_texts = [study.report.target_text(control_tokens=True)
                       for study in manifest.split("train")]
    tokenizer = WordTokenizer.build(
        tokenizer_texts,
        min_freq=int(config.get_path("tokenizer.min_freq", 1)),
        max_vocab=int(config.get_path("tokenizer.max_vocab", 12000)),
        max_len=int(config.get_path("tokenizer.max_len", 256)))

    dataset_config = AdmDatasetConfig(
        max_report_tokens=tokenizer.max_len, use_semantic=use_semantic,
        d_visual=sources.d_visual, d_proto=sources.d_proto)
    configured_grid = int(config.get_path("model.patch_grid", sources.visual.patch_grid))
    if configured_grid != sources.visual.patch_grid:
        raise ConfigError(
            f"model.patch_grid={configured_grid} 与视觉缓存 patch_grid="
            f"{sources.visual.patch_grid} 不一致；二维位置编码会错位。")
    model_config = AdmReportModelConfig.from_config(
        dict(config.get("model") or {}), d_visual=sources.d_visual,
        d_proto=sources.d_proto, visual_tokens=sources.visual_tokens,
        semantic_tokens=max(1, sources.semantic_tokens or
                            int(config.get_path("model.semantic_tokens", 13))),
        patch_grid=sources.visual.patch_grid)
    if model_config.use_semantic and sources.semantic is None:
        raise ConfigError("use_semantic=True 但没有语义缓存")

    full_train = AdmEvidenceDataset(manifest, "train", tokenizer, sources.visual,
                                    dataset_config, semantic=sources.semantic,
                                    localization=sources.localization)
    full_val = AdmEvidenceDataset(manifest, "val", tokenizer, sources.visual,
                                  dataset_config, semantic=sources.semantic,
                                  localization=sources.localization)
    if overfit_cases:
        index = {study.study_id: i for i, study in enumerate(manifest.split("train"))}
        indices = [index[study.study_id] for study in train_studies]
        train_dataset: Any = Subset(full_train, indices)
        val_dataset: Any = Subset(full_train, indices)
    else:
        train_dataset, val_dataset = full_train, full_val
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,
                              shuffle=True, collate_fn=collate_adm_samples,
                              num_workers=settings.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size,
                            shuffle=False, collate_fn=collate_adm_samples,
                            num_workers=settings.num_workers)

    model = AdmEvidenceReportTransformer(model_config, vocab_size=tokenizer.vocab_size,
                                        pad_id=tokenizer.pad_id)
    init_sha = ""
    if settings.init_checkpoint:
        init_path = Path(settings.init_checkpoint)
        if not init_path.is_absolute():
            init_path = project_root() / init_path
        init = load_checkpoint(init_path, device="cpu")
        missing, unexpected = model.load_state_dict(init["model"].state_dict(),
                                                   strict=False)
        if unexpected:
            raise ConfigError(f"初始化 checkpoint 有未使用参数：{list(unexpected)[:5]}")
        allowed = {"semantic_projector"}
        bad_missing = [key for key in missing
                       if key.split(".")[0] not in allowed]
        if bad_missing:
            raise ConfigError(f"初始化 checkpoint 缺少关键参数：{bad_missing[:5]}")
        init_extra = dict(init.get("extra") or {})
        if init_extra.get("manifest_sha256") and \
                init_extra["manifest_sha256"] != manifest.sha256:
            raise CacheError("初始化 checkpoint 绑定的 manifest 与当前不一致")
        previous = dict(init_extra.get("cache_identity") or {})
        for kind, current in sources.cache_identity.items():
            old = previous.get(kind)
            if old and current and old != current:
                raise CacheError(
                    f"初始化 checkpoint 的 {kind} 缓存身份与当前缓存不一致；"
                    "请重建缓存或使用对应 checkpoint。")
        init_sha = init["checkpoint_sha256"]
        if verbose:
            print(f"[train] 从 {init_path} 初始化；新增参数 {len(missing)} 个", flush=True)
    if settings.freeze_visual_projection:
        for parameter in (list(model.visual_proj.parameters())
                          + list(model.visual_norm.parameters())):
            parameter.requires_grad_(False)
    model.to(device)
    model.train()
    if settings.warmup_new_epochs > 0:
        model.freeze_decoder(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr,
                                  weight_decay=settings.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / settings.grad_accum))
    total_steps = max(1, steps_per_epoch * settings.epochs)
    warmup_steps = int(total_steps * settings.warmup_ratio)

    def lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_enabled = bool(settings.amp and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    run_dir = Path(out_dir) if out_dir else config.path(
        "checkpoint_root", "checkpoints") / str(config.get_path("experiment.name", "run"))
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(run_dir / "tokenizer.json")
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_plain(), allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    write_json(run_dir / "environment.json", _environment())

    source_tree_sha = _source_tree_hash()
    tokenizer_sha = sha256_json({"vocab": list(tokenizer.vocab),
                                 "max_len": tokenizer.max_len})
    evidence_identity_value = sha256_json({
        "manifest_sha256": manifest.sha256,
        "cache_identity": sources.cache_identity})
    report_identity = report_checkpoint_identity(
        evidence_identity_value, tokenizer_sha, model_config.to_dict(),
        dict(config.get("training") or {}), source_tree_sha)
    paths = source_paths(config)
    base_extra = {
        "manifest": manifest.path, "manifest_sha256": manifest.sha256,
        "config_path": str(config.sources[-2] if len(config.sources) > 1
                           else config.sources[0]),
        "resolved_config": config.to_plain(),
        "ontology": config.get("ontology") or {},
        "evidence_mode": "adm_simple",
        "evidence_paths": paths,
        "cache_identity": sources.cache_identity,
        "d_visual": sources.d_visual, "d_proto": sources.d_proto,
        "visual_tokens": sources.visual_tokens,
        "semantic_tokens": sources.semantic_tokens,
        "use_semantic": model_config.use_semantic,
        "allow_oracle": allow_oracle,
        "experiment": str(config.get_path("experiment.name", "run")),
        "results_root": str(config.get_path("results_root", "results")),
        "init_checkpoint_sha256": init_sha,
        "seed": settings.seed,
        "evidence_identity": evidence_identity_value,
        "tokenizer_sha256": tokenizer_sha,
        "source_tree_sha256": source_tree_sha,
        "report_checkpoint_identity": report_identity,
        "adm": dict(config.get("adm") or {}),
        "sampling": paths["sampling"],
    }

    best_loss = float("inf")
    best_epoch = 0
    best_payload: dict[str, Any] | None = None
    curve_path = run_dir / "training_curve.jsonl"
    history: list[dict[str, Any]] = []
    global_step = 0
    epochs_without_improvement = 0
    for epoch in range(1, settings.epochs + 1):
        if settings.warmup_new_epochs and epoch == settings.warmup_new_epochs + 1:
            model.freeze_decoder(False)
            if verbose:
                print(f"[train] epoch {epoch}: 解冻 decoder，进入联合训练", flush=True)
        model.train()
        epoch_loss, batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            evidence = batch["evidence"].to(device)
            decoder_input = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=amp_enabled):
                output = model(evidence, decoder_input, labels,
                               label_smoothing=settings.label_smoothing)
                loss = output["loss"]
            group_start = (batch_index // settings.grad_accum) * settings.grad_accum
            group_size = min(settings.grad_accum, len(train_loader) - group_start)
            scaler.scale(loss / max(1, group_size)).backward()
            should_step = ((batch_index + 1) % settings.grad_accum == 0
                           or batch_index + 1 == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                               settings.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            epoch_loss += float(loss.detach())
            batches += 1
        train_loss = epoch_loss / max(1, batches)
        metrics = evaluate_loss(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": round(train_loss, 6),
                  "val_loss": round(metrics["val_loss"], 6),
                  "optimizer_steps": global_step,
                  "lr": float(optimizer.param_groups[0]["lr"]),
                  "decoder_frozen": bool(settings.warmup_new_epochs
                                         and epoch <= settings.warmup_new_epochs)}
        history.append(record)
        append_jsonl(curve_path, record)
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            best_epoch = epoch
            best_payload = save_checkpoint(
                run_dir / "best.pt", model, tokenizer, model_config,
                {**base_extra, "best_epoch": epoch, "val_loss": metrics["val_loss"]})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if verbose:
            print(f"[train] epoch {epoch}/{settings.epochs} "
                  f"train={train_loss:.4f} val={metrics['val_loss']:.4f}", flush=True)
        if settings.early_stopping_patience and \
                epochs_without_improvement >= settings.early_stopping_patience:
            if verbose:
                print(f"[train] early stop：连续 {epochs_without_improvement} "
                      "个 epoch 无改善", flush=True)
            break

    last_payload = save_checkpoint(
        run_dir / "last.pt", model, tokenizer, model_config,
        {**base_extra, "best_epoch": best_epoch, "best_val_loss": best_loss,
         "last_epoch": history[-1]["epoch"] if history else 0,
         "last_val_loss": history[-1]["val_loss"] if history else None})
    summary = {
        "run_dir": str(run_dir),
        "experiment": str(config.get_path("experiment.name", "run")),
        "device": device,
        "epochs": settings.epochs,
        "overfit_cases": overfit_cases,
        "trainable_parameters": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        "manifest_sha256": manifest.sha256,
        "cache_identity": sources.cache_identity,
        "visual_tokens": sources.visual_tokens,
        "semantic_tokens": sources.semantic_tokens,
        "best": best_payload,
        "last": last_payload,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "history": history,
    }
    write_json(run_dir / "train_summary.json", summary)
    return summary
