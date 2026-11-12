"""方案 §10：ADM 数据合同——只有 visual/semantic 两类 token。"""

from __future__ import annotations

import torch

from medrrg.contracts import AdmEvidenceBatch
from medrrg.data.adm_dataset import (AdmDatasetConfig, AdmEvidenceDataset,
                                     collate_adm_samples)
from medrrg.features.adm_cache import (AdmLocalizationCache, AdmSemanticCache,
                                       AdmVisualCache)

FORBIDDEN = ("roi", "geometry", "coord")


def _dataset(manifest, tokenizer, evidence_dirs, use_semantic: bool = True):
    visual = AdmVisualCache.load(evidence_dirs["visual"])
    semantic = AdmSemanticCache.load(evidence_dirs["semantic"]) if use_semantic else None
    localization = AdmLocalizationCache.load(evidence_dirs["localization"])
    config = AdmDatasetConfig(max_report_tokens=64, use_semantic=use_semantic,
                              d_visual=visual.d_visual,
                              d_proto=semantic.d_proto if semantic else 4)
    return AdmEvidenceDataset(manifest, "train", tokenizer, visual, config,
                              semantic=semantic, localization=localization)


def test_sample_has_only_visual_and_semantic_evidence(manifest, tokenizer, evidence_dirs):
    dataset = _dataset(manifest, tokenizer, evidence_dirs)
    sample = dataset[0]
    for key in sample:
        for banned in FORBIDDEN:
            assert banned not in key, f"ADM 正式批出现禁用字段 {key}"


def test_collate_contract_and_shapes(manifest, tokenizer, evidence_dirs):
    dataset = _dataset(manifest, tokenizer, evidence_dirs)
    batch = collate_adm_samples([dataset[0], dataset[1]])
    evidence = batch["evidence"]
    evidence.assert_adm_contract()
    assert isinstance(evidence, AdmEvidenceBatch)
    count = evidence.visual_tokens.shape[1]
    concepts = evidence.semantic_scores.shape[1]
    assert count == 16            # 2 z × 2 模态 × 4 patch
    assert concepts == 3
    assert evidence.visual_mask.all()
    assert evidence.semantic_mask.all()
    # token 顺序：z 优先，模态其次
    assert evidence.visual_modality[0, 0].item() == 0
    assert evidence.visual_modality[0, 4].item() == 1
    assert evidence.visual_z[0, 0].item() == evidence.visual_z[0, 7].item()
    assert evidence.visual_z[0, 8].item() > evidence.visual_z[0, 0].item()


def test_teacher_forcing_shift(manifest, tokenizer, evidence_dirs):
    dataset = _dataset(manifest, tokenizer, evidence_dirs)
    sample = dataset[0]
    batch = collate_adm_samples([sample])
    ids = sample["target_ids"]
    assert len(ids) > 2
    assert torch.equal(batch["decoder_input_ids"][0, :len(ids) - 1], ids[:-1])
    assert torch.equal(batch["labels"][0, :len(ids) - 1], ids[1:])
    assert batch["labels"][0, len(ids) - 1:].eq(-100).all()


def test_padding_and_mask_for_unequal_lengths(tokenizer):
    samples = []
    for count in (6, 10):
        samples.append({
            "study_id": f"s{count}", "patient_id": f"p{count}",
            "visual_tokens": torch.randn(count, 8),
            "visual_tokens_null": torch.randn(count, 8),
            "visual_mask": torch.arange(count) < (count - 2),   # 末尾 2 个 padding
            "visual_modality": torch.zeros(count, dtype=torch.long),
            "visual_z": torch.zeros(count),
            "visual_view": torch.zeros(count, dtype=torch.long),
            "visual_patch_row": (torch.arange(count) // 2).remainder(2),
            "visual_patch_col": torch.arange(count).remainder(2),
            "visual_available": torch.tensor(True),
            "semantic_prototypes": torch.randn(3, 4),
            "semantic_scores": torch.rand(3),
            "semantic_confidence": torch.rand(3),
            "semantic_margin": torch.rand(3),
            "semantic_mask": torch.ones(3, dtype=torch.bool),
            "semantic_available": torch.tensor(True),
            "adm_applied": torch.tensor(True),
            "abnormal_score": torch.tensor(0.5),
            "target_ids": torch.tensor(tokenizer.encode("a lesion in the left lobe",
                                                        max_len=16)),
            "reference": "a lesion in the left lobe",
            "target_text": "a lesion in the left lobe",
        })
    batch = collate_adm_samples(samples)
    evidence = batch["evidence"]
    assert evidence.visual_tokens.shape[1] == 10
    assert evidence.visual_mask[0].sum().item() == 4
    assert evidence.visual_mask[1].sum().item() == 8
    assert batch["visual_tokens_null"].shape == evidence.visual_tokens.shape


def test_patch_grid_positions_are_exported(manifest, tokenizer, evidence_dirs):
    dataset = _dataset(manifest, tokenizer, evidence_dirs)
    sample = dataset[0]
    rows = sample["visual_patch_row"].tolist()
    cols = sample["visual_patch_col"].tolist()
    assert len(rows) == len(cols) == 16
    assert set(rows) == {0, 1}
    assert set(cols) == {0, 1}
    # 每个切片内按行优先： (0,0) (0,1) (1,0) (1,1)
    assert rows[:4] == [0, 0, 1, 1]
    assert cols[:4] == [0, 1, 0, 1]
    batch = collate_adm_samples([sample])
    evidence = batch["evidence"]
    assert evidence.visual_patch_row.shape == evidence.visual_tokens.shape[:2]


def test_semantic_disabled_has_no_semantic_tokens(manifest, tokenizer, evidence_dirs):
    dataset = _dataset(manifest, tokenizer, evidence_dirs, use_semantic=False)
    batch = collate_adm_samples([dataset[0]])
    evidence = batch["evidence"]
    assert evidence.semantic_mask.sum().item() == 0
    assert evidence.semantic_available.item() is False
