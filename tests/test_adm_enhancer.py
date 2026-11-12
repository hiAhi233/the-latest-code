"""方案 §10：ADM 增强器的边界行为。"""

from __future__ import annotations

import numpy as np
import pytest

from medrrg.features.adm_enhancer import (AdmParams, enhance_image, normalize_heatmap,
                                          resize_heatmap, select_top_slices)


def _image(size: int = 16) -> np.ndarray:
    image = np.linspace(0.0, 1.0, size * size, dtype=np.float32).reshape(size, size)
    return image


def _heat(size: int = 16) -> np.ndarray:
    heat = np.zeros((size, size), dtype=np.float32)
    heat[4:8, 4:8] = 1.0
    return heat


def test_normalize_heatmap_range():
    rng = np.random.default_rng(0)
    heat = rng.normal(0.5, 2.0, (16, 16))
    attention = normalize_heatmap(heat)
    assert attention.min() == pytest.approx(0.0, abs=1e-6)
    assert attention.max() == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(attention).all()


def test_flat_heatmap_is_zero_and_falls_back():
    flat = np.full((8, 8), 0.3, dtype=np.float32)
    assert normalize_heatmap(flat).max() == 0.0
    enhanced, meta = enhance_image(_image(8), flat, AdmParams(), abnormal_score=0.9)
    assert meta["fallback"] is True
    assert meta["reason"] in ("flat_heatmap", "abnormal_score_below_threshold")


def test_gamma_one_is_identity():
    image = _image()
    enhanced, meta = enhance_image(image, _heat(), AdmParams(gamma=1.0),
                                   abnormal_score=0.9)
    assert meta["fallback"] is False
    np.testing.assert_allclose(enhanced, image, atol=1e-6)
    # gamma=1 时背景系数也是 1，不会把正常组织压暗
    assert meta["factor_min"] == pytest.approx(1.0)
    assert meta["factor_max"] == pytest.approx(1.0)


def test_gamma_zero_zeros_background_but_keeps_lesion():
    image = np.ones((16, 16), dtype=np.float32)
    heat = _heat()
    enhanced, _ = enhance_image(image, heat, AdmParams(gamma=0.0), abnormal_score=0.9)
    # 病灶内部 attention=1 -> 系数 1，保留原强度；背景 attention=0 -> 系数 0
    assert enhanced[5, 5] == pytest.approx(1.0)
    assert enhanced[0, 0] == pytest.approx(0.0)


def test_default_gamma_keeps_background_above_zero():
    image = np.ones((16, 16), dtype=np.float32)
    enhanced, meta = enhance_image(image, _heat(), AdmParams(gamma=0.4),
                                   abnormal_score=0.9)
    assert enhanced[0, 0] == pytest.approx(0.4)
    assert enhanced[5, 5] == pytest.approx(1.0)
    assert enhanced.min() > 0.0          # 背景不完全清零
    assert meta["factor_min"] == pytest.approx(0.4)


def test_hard_mode_uses_threshold():
    image = np.ones((16, 16), dtype=np.float32)
    params = AdmParams(mode="hard", gamma=0.2, threshold=0.6)
    enhanced, meta = enhance_image(image, _heat(), params, abnormal_score=0.9)
    assert enhanced[5, 5] == pytest.approx(1.0)
    assert enhanced[0, 0] == pytest.approx(0.2)
    assert meta["mode"] == "hard"


def test_normal_fallback_below_threshold():
    image = _image()
    enhanced, meta = enhance_image(image, _heat(), AdmParams(threshold=0.6),
                                   abnormal_score=0.1)
    assert meta["fallback"] is True
    assert meta["reason"] == "abnormal_score_below_threshold"
    np.testing.assert_allclose(enhanced, image)


@pytest.mark.parametrize("gamma", [-0.1, 1.5])
def test_gamma_bounds(gamma):
    with pytest.raises(ValueError):
        AdmParams(gamma=gamma)


def test_threshold_bounds():
    with pytest.raises(ValueError):
        AdmParams(threshold=0.0)
    with pytest.raises(ValueError):
        AdmParams(threshold=1.5)


def test_select_top_slices_sorted_and_limited():
    scores = [0.1, 0.9, 0.5, 0.8, 0.2]
    assert select_top_slices(scores, k=3) == [1, 2, 3]   # 升序 z
    assert select_top_slices(scores, k=1) == [1]
    assert select_top_slices([0.0, 0.0], k=3) == [0, 1]


def test_resize_heatmap_matches_image_shape():
    resized = resize_heatmap(_heat(8), (32, 24))
    assert resized.shape == (32, 24)
    assert resized.max() <= 1.0


def test_oracle_small_lesion_is_enhanced_without_fallback():
    """GT 小病灶面积占比远低于 threshold，也不能回退成原图（O0 上限失真）。"""
    image = np.ones((32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[15:17, 15:17] = 1.0          # 4/1024 ≈ 0.4%，远小于 threshold=0.6
    enhanced, meta = enhance_image(image, mask, AdmParams(threshold=0.6),
                                   abnormal_score=0.004, oracle_mask=True)
    assert meta["fallback"] is False
    assert meta["reason"] == "oracle_gt_mask"
    assert meta["oracle_mask"] is True
    assert enhanced[15, 15] == pytest.approx(1.0)
    assert enhanced[0, 0] == pytest.approx(0.4)


def test_oracle_empty_mask_keeps_original_slice():
    image = np.ones((16, 16), dtype=np.float32)
    enhanced, meta = enhance_image(image, np.zeros((16, 16), dtype=np.float32),
                                   AdmParams(), abnormal_score=0.0,
                                   oracle_mask=True)
    assert meta["fallback"] is True
    assert meta["reason"] == "oracle_empty_mask"
    np.testing.assert_allclose(enhanced, image)


def test_oracle_does_not_use_predicted_normal_fallback():
    image = _image()
    enhanced, meta = enhance_image(image, _heat(), AdmParams(),
                                   abnormal_score=0.0, oracle_mask=True)
    assert meta["reason"] == "oracle_gt_mask"
    assert not np.allclose(enhanced, image)
