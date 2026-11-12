"""方案 §10：报告模型的 memory 长度、B0/B1/F0 差异、teacher forcing 与 EOS。"""

from __future__ import annotations

import torch

from medrrg.data.adm_dataset import (AdmDatasetConfig, AdmEvidenceDataset,
                                     collate_adm_samples)
from medrrg.features.adm_cache import (AdmLocalizationCache, AdmSemanticCache,
                                       AdmVisualCache)
from medrrg.generation.generate import truncate_at_eos
from medrrg.model.report_model import (AdmEvidenceReportTransformer,
                                       AdmReportModelConfig)
from medrrg.model.token_projectors import AdmEvidenceAssembler


def _config(use_semantic: bool) -> AdmReportModelConfig:
    return AdmReportModelConfig(
        d_model=32, heads=4, encoder_layers=1, decoder_layers=2, ffn=64,
        dropout=0.0, max_report_tokens=24, visual_tokens=16, semantic_tokens=3,
        d_visual=8, d_proto=4, use_semantic=use_semantic, evidence_dropout=0.0,
        patch_grid=2)


def _batch(manifest, tokenizer, evidence_dirs, use_semantic: bool):
    visual = AdmVisualCache.load(evidence_dirs["visual"])
    semantic = AdmSemanticCache.load(evidence_dirs["semantic"]) if use_semantic else None
    localization = AdmLocalizationCache.load(evidence_dirs["localization"])
    dataset = AdmEvidenceDataset(
        manifest, "train", tokenizer, visual,
        AdmDatasetConfig(max_report_tokens=24, use_semantic=use_semantic,
                         d_visual=8, d_proto=4),
        semantic=semantic, localization=localization)
    return collate_adm_samples([dataset[0], dataset[1]])


def test_f0_memory_is_visual_plus_semantic(manifest, tokenizer, evidence_dirs):
    batch = _batch(manifest, tokenizer, evidence_dirs, use_semantic=True)
    model = AdmEvidenceReportTransformer(_config(True),
                                         vocab_size=tokenizer.vocab_size,
                                         pad_id=tokenizer.pad_id)
    output = model(batch["evidence"], batch["decoder_input_ids"], batch["labels"])
    assert tuple(output["memory"].shape) == (2, 16 + 3, 32)
    assert output["logits"].shape[-1] == tokenizer.vocab_size


def test_b0_b1_memory_is_visual_only(manifest, tokenizer, evidence_dirs):
    batch = _batch(manifest, tokenizer, evidence_dirs, use_semantic=False)
    model = AdmEvidenceReportTransformer(_config(False),
                                         vocab_size=tokenizer.vocab_size,
                                         pad_id=tokenizer.pad_id)
    output = model(batch["evidence"], batch["decoder_input_ids"], batch["labels"])
    assert tuple(output["memory"].shape) == (2, 16, 32)


def test_thirteen_semantic_tokens_in_memory(manifest, tokenizer, evidence_dirs):
    batch = _batch(manifest, tokenizer, evidence_dirs, use_semantic=True)
    evidence = batch["evidence"]
    k = evidence.semantic_scores.shape[1]
    config = _config(True)
    config.semantic_tokens = k
    model = AdmEvidenceReportTransformer(config, vocab_size=tokenizer.vocab_size,
                                        pad_id=tokenizer.pad_id)
    output = model(evidence, batch["decoder_input_ids"], batch["labels"])
    assert output["memory"].shape[1] == evidence.visual_tokens.shape[1] + k


def test_teacher_forcing_loss_backward(manifest, tokenizer, evidence_dirs):
    batch = _batch(manifest, tokenizer, evidence_dirs, use_semantic=True)
    model = AdmEvidenceReportTransformer(_config(True),
                                         vocab_size=tokenizer.vocab_size,
                                         pad_id=tokenizer.pad_id)
    output = model(batch["evidence"], batch["decoder_input_ids"], batch["labels"])
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_generation_stops_at_max_len_and_marks_eos(manifest, tokenizer, evidence_dirs):
    batch = _batch(manifest, tokenizer, evidence_dirs, use_semantic=True)
    model = AdmEvidenceReportTransformer(_config(True),
                                         vocab_size=tokenizer.vocab_size,
                                         pad_id=tokenizer.pad_id)
    model.eval()
    sequences = model.generate(batch["evidence"], max_len=8,
                               bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
    for sequence in sequences:
        assert 1 <= len(sequence) <= 7
        trimmed = truncate_at_eos(sequence, tokenizer.eos_id)
        assert tokenizer.eos_id not in trimmed
        # EOS 之后只能是 EOS（greedy 用 EOS 填充已结束样本）
        if tokenizer.eos_id in sequence:
            tail = sequence[sequence.index(tokenizer.eos_id):]
            assert all(int(item) == tokenizer.eos_id for item in tail)


def test_roi_and_geometry_are_rejected():
    try:
        AdmReportModelConfig(use_geometry=True)
    except Exception as error:      # noqa: BLE001
        assert "geometry" in str(error) or "ROI" in str(error)
    else:
        raise AssertionError("use_geometry=True 必须被拒绝")


def test_2d_patch_positions_distinguish_grid_cells():
    """同内容、同模态/z 的 patch 必须因网格位置不同而得到不同表示。"""
    assembler = AdmEvidenceAssembler(16, patch_grid=3)
    tokens = torch.zeros(1, 9, 16)
    mask = torch.ones(1, 9, dtype=torch.bool)
    available = torch.ones(1, dtype=torch.bool)
    modality = torch.zeros(1, 9, dtype=torch.long)
    z_value = torch.zeros(1, 9)
    view = torch.zeros(1, 9, dtype=torch.long)
    rows = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2]])
    cols = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2]])
    hidden = assembler.embed_visual(tokens, modality, z_value, view, mask,
                                    available, rows, cols)
    assert not torch.allclose(hidden[0, 0], hidden[0, 4])
    assert not torch.allclose(hidden[0, 1], hidden[0, 3])
    assert not torch.allclose(hidden[0, 0], hidden[0, 8])


def test_swapping_patch_tokens_changes_evidence_set(manifest, tokenizer, evidence_dirs):
    """打乱 patch 内容（位置不变）会改变 token 集合，否则位置编码形同虚设。"""
    batch = _batch(manifest, tokenizer, evidence_dirs, use_semantic=True)
    evidence = batch["evidence"]
    model = AdmEvidenceReportTransformer(_config(True),
                                        vocab_size=tokenizer.vocab_size,
                                        pad_id=tokenizer.pad_id)
    model.eval()
    with torch.no_grad():
        visual = model.visual_norm(model.visual_proj(evidence.visual_tokens.float()))
        count = visual.shape[1]
        perm = torch.arange(count)
        perm[0], perm[1] = 1, 0
        base = model.assembler.embed_visual(
            visual, evidence.visual_modality, evidence.visual_z,
            evidence.visual_view, evidence.visual_mask, evidence.visual_available,
            evidence.visual_patch_row, evidence.visual_patch_col)
        swapped = model.assembler.embed_visual(
            visual[:, perm], evidence.visual_modality, evidence.visual_z,
            evidence.visual_view, evidence.visual_mask, evidence.visual_available,
            evidence.visual_patch_row, evidence.visual_patch_col)
    # 行范数是排列不变量；内容与位置绑定改变后必须变化
    base_norms = torch.sort((base ** 2).sum(-1), dim=-1).values
    swapped_norms = torch.sort((swapped ** 2).sum(-1), dim=-1).values
    assert not torch.allclose(base_norms, swapped_norms, atol=1e-6)
