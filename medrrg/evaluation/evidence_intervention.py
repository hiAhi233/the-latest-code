"""推理期证据干预（方案 §9 末段）。

不重新训练，直接在 checkpoint 上做四种干预：

* ``null_adm``：把增强视觉 tokens 换成同一批 ResNet 对**原图**编码的 tokens，
  等价于 Iadm=I；
* ``swap_adm``：batch 内交换增强视觉 tokens（同时交换 modality/z/view）；
* ``null_semantic``：屏蔽 UniMed 语义 tokens；
* ``swap_semantic``：batch 内交换语义 tokens。

报告真实/干预后的 NLL、患者级 bootstrap 差值与相对变化。判读：置空/交换后
NLL 应上升；若几乎不变，说明 Transformer 在背模板，证据没被真正使用。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..contracts import AdmEvidenceBatch
from ..data.adm_dataset import collate_adm_samples
from .nlg_metrics import tokenize_words  # noqa: F401  (概念差值复用同一 tokenizer)

__all__ = ["CONDITIONS", "run_interventions", "paired_bootstrap", "perturb_batch"]

CONDITIONS = ("null_adm", "swap_adm", "null_semantic", "swap_semantic")


def paired_bootstrap(differences: Sequence[float], n_resamples: int = 1000,
                     seed: int = 0) -> dict[str, float]:
    arr = np.asarray(list(differences), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(int(n_resamples), arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "p_value": float(min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean()))),
        "n": int(arr.size),
    }


def _copy(evidence: AdmEvidenceBatch, **changes: Any) -> AdmEvidenceBatch:
    data = dict(evidence.__dict__)
    data.update(changes)
    return AdmEvidenceBatch(**data)


def _permutation(batch: int, seed: int) -> torch.Tensor | None:
    if batch < 2:
        return None
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(batch, generator=generator)
    for _ in range(10):
        if all(int(perm[i]) != i for i in range(batch)):
            return perm
        perm = torch.roll(perm, shifts=1)
    return None


def perturb_batch(batch: dict[str, Any], condition: str,
                  seed: int = 0) -> AdmEvidenceBatch | None:
    evidence: AdmEvidenceBatch = batch["evidence"]
    batch_size = len(evidence.study_ids)
    if condition == "null_adm":
        null_tokens = batch.get("visual_tokens_null")
        if null_tokens is None:
            raise ValueError("batch 缺少 visual_tokens_null，无法执行 null_adm")
        return _copy(evidence, visual_tokens=null_tokens.to(evidence.visual_tokens.device))
    if condition == "null_semantic":
        if evidence.semantic_scores is None:
            return None
        return _copy(
            evidence,
            semantic_available=torch.zeros(batch_size, dtype=torch.bool,
                                           device=evidence.semantic_scores.device),
            semantic_mask=torch.zeros_like(evidence.semantic_mask))
    perm = _permutation(batch_size, seed)
    if perm is None:
        return None
    device = evidence.visual_tokens.device
    perm = perm.to(device)
    if condition == "swap_adm":
        return _copy(
            evidence,
            visual_tokens=evidence.visual_tokens[perm],
            visual_mask=evidence.visual_mask[perm],
            visual_modality=evidence.visual_modality[perm],
            visual_z=evidence.visual_z[perm],
            visual_view=evidence.visual_view[perm],
            visual_available=evidence.visual_available[perm])
    if condition == "swap_semantic":
        if evidence.semantic_scores is None:
            return None
        return _copy(
            evidence,
            semantic_prototypes=evidence.semantic_prototypes[perm],
            semantic_scores=evidence.semantic_scores[perm],
            semantic_confidence=evidence.semantic_confidence[perm],
            semantic_margin=evidence.semantic_margin[perm],
            semantic_mask=evidence.semantic_mask[perm],
            semantic_available=evidence.semantic_available[perm])
    raise ValueError(f"未知干预 {condition!r}")


def _per_study_nll(model: Any, batch: dict[str, Any], evidence: AdmEvidenceBatch,
                   device: str) -> list[float]:
    output = model(evidence.to(device),
                   batch["decoder_input_ids"].to(device),
                   batch["labels"].to(device))
    logits = output["logits"]
    labels = batch["labels"].to(device)
    vocab = logits.shape[-1]
    log_probs = F.log_softmax(logits.reshape(-1, vocab).float(), dim=-1)
    targets = labels.reshape(-1)
    valid = targets != -100
    gathered = log_probs.gather(1, targets.clamp_min(0).unsqueeze(1)).squeeze(1)
    gathered = gathered * valid
    lengths = valid.reshape(labels.shape[0], -1).sum(dim=1).clamp_min(1)
    values = gathered.reshape(labels.shape[0], -1).sum(dim=1) / lengths
    return (-values).tolist()


@torch.no_grad()
def run_interventions(model: Any, dataset: Any, device: str,
                      batch_size: int = 8, conditions: Sequence[str] = CONDITIONS,
                      n_resamples: int = 1000, seed: int = 0,
                      concept_deltas: bool = False, delta_cases: int = 8,
                      ontology: Any = None, tokenizer: Any = None,
                      verbose: bool = False) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_adm_samples)
    true_by_study: dict[str, float] = {}
    patient_of: dict[str, str] = {}
    values: dict[str, dict[str, float]] = {condition: {} for condition in conditions}
    skipped: dict[str, int] = {condition: 0 for condition in conditions}
    for batch_index, batch in enumerate(loader):
        evidence = batch["evidence"].to(device)
        base = _per_study_nll(model, batch, evidence, device)
        for index, study_id in enumerate(batch["study_ids"]):
            true_by_study[study_id] = base[index]
            patient_of[study_id] = batch["patient_ids"][index]
        for condition in conditions:
            variant = perturb_batch(batch, condition, seed=seed + batch_index)
            if variant is None:
                skipped[condition] += len(batch["study_ids"])
                continue
            scores = _per_study_nll(model, batch, variant, device)
            for index, study_id in enumerate(batch["study_ids"]):
                values[condition][study_id] = scores[index]
        if verbose:
            print(f"[intervention] batch {batch_index + 1}", flush=True)

    true_mean = (float(np.mean(list(true_by_study.values())))
                 if true_by_study else float("nan"))
    report: dict[str, Any] = {
        "num_studies": len(true_by_study),
        "true_nll": true_mean,
        "conditions": {},
        "note": "NLL 为 nats/token；干预后 NLL 上升才说明证据被使用",
    }
    for condition in conditions:
        diffs_by_patient: dict[str, list[float]] = {}
        for study_id, true_value in true_by_study.items():
            if study_id not in values[condition]:
                continue
            diffs_by_patient.setdefault(patient_of[study_id], []).append(
                values[condition][study_id] - true_value)
        patient_diffs = [float(np.mean(items)) for items in diffs_by_patient.values()]
        stats = paired_bootstrap(patient_diffs, n_resamples=n_resamples, seed=seed)
        perturbed_mean = (float(np.mean(list(values[condition].values())))
                          if values[condition] else float("nan"))
        report["conditions"][condition] = {
            "perturbed_nll": perturbed_mean,
            "nll_increase": stats["mean"],
            "relative_increase": stats["mean"] / max(1e-9, true_mean),
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "p_value": stats["p_value"],
            "num_patients": stats["n"],
            "num_skipped": skipped.get(condition, 0),
        }

    if concept_deltas and ontology is not None and tokenizer is not None:
        from ..generation.generate import postprocess_report

        report["concept_deltas"] = {}
        subset = list(dataset.studies[:delta_cases])
        for condition in ("null_adm", "null_semantic"):
            changed = []
            for study in subset:
                index = dataset.studies.index(study)
                batch = collate_adm_samples([dataset[index]])
                evidence = batch["evidence"].to(device)
                device_batch = dict(batch)
                device_batch["evidence"] = evidence
                true_ids = model.generate(evidence, max_len=model.config.max_report_tokens,
                                          bos_id=tokenizer.bos_id,
                                          eos_id=tokenizer.eos_id)
                variant = perturb_batch(device_batch, condition, seed=seed)
                if variant is None:
                    continue
                perturbed_ids = model.generate(
                    variant, max_len=model.config.max_report_tokens,
                    bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
                true_text = postprocess_report(
                    tokenizer.decode(true_ids[0], skip_special=False))["formatted"]
                other_text = postprocess_report(
                    tokenizer.decode(perturbed_ids[0], skip_special=False))["formatted"]
                true_concepts = set(ontology.match(true_text))
                other_concepts = set(ontology.match(other_text))
                changed.append(len(true_concepts.symmetric_difference(other_concepts)))
            report["concept_deltas"][condition] = {
                "mean_changed_concepts": float(np.mean(changed)) if changed else 0.0,
                "num_cases": len(changed)}
    return report
