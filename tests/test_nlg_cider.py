"""CIDEr-D：官方 cider_scorer 公式的回归测试。"""

from __future__ import annotations

import pytest

from medrrg.evaluation.nlg_metrics import cider_d

REFERENCES = ["a lesion in the left frontal lobe", "there is no edema"]


def test_perfect_match_scores_ten():
    assert cider_d(list(REFERENCES), REFERENCES) == pytest.approx(10.0, abs=1e-6)


def test_partial_match_is_between_zero_and_ten():
    score = cider_d(["a lesion in the right frontal lobe", "there is edema"],
                    REFERENCES)
    assert 0.0 < score < 10.0


def test_zero_overlap_scores_zero():
    assert cider_d(["completely unrelated words here",
                    "another unrelated sentence"], REFERENCES) == pytest.approx(0.0)


def test_single_document_degenerates_like_official():
    # N=1 时 idf 全为 0，官方实现同样返回 0；评测应使用完整测试集
    assert cider_d(["a b"], ["a b"]) == pytest.approx(0.0)


def test_length_penalty_reduces_score():
    long_hypothesis = "a lesion in the left frontal lobe " * 3
    short = cider_d(["a lesion in the left frontal lobe", "there is no edema"],
                    REFERENCES)
    long = cider_d([long_hypothesis, "there is no edema"], REFERENCES)
    assert long < short
