"""方案 §10：缓存身份链——权重/γ/threshold/prompt 变化必须使缓存失效。"""

from __future__ import annotations

import json
import shutil

import pytest

from medrrg.config import Config
from medrrg.errors import CacheError
from medrrg.evidence import load_sources
from medrrg.features.adm_cache import (AdmLocalizationCache, AdmSemanticCache,
                                       AdmVisualCache)
from medrrg.features.cache_identity import (adm_localization_identity,
                                            resnet_visual_identity,
                                            unimed_semantic_identity)


def test_localization_identity_depends_on_checkpoint_and_images():
    base = dict(detector_code_sha256="code", checkpoint_sha256="ckpt",
                preprocessing={"heat_size": 64}, thresholds={"heat_threshold": 0.6},
                image_sha256={"a": {"t2f": "1"}})
    reference = adm_localization_identity(**base)
    changed_checkpoint = dict(base, checkpoint_sha256="other")
    changed_images = dict(base, image_sha256={"a": {"t2f": "2"}})
    assert reference != adm_localization_identity(**changed_checkpoint)
    assert reference != adm_localization_identity(**changed_images)


def test_visual_identity_depends_on_gamma_threshold_and_weights():
    kwargs = dict(localization_identity_value="loc", resnet_weights_sha256="resnet",
                  adm={"gamma": 0.4, "threshold": 0.6, "mode": "soft"},
                  sampling={"topk_slices": 3}, visual={"patch_grid": 3},
                  image_sha256={"a": "1"})
    reference = resnet_visual_identity(**kwargs)
    for key, value in (("gamma", 0.3), ("threshold", 0.7), ("mode", "hard")):
        changed = dict(kwargs, adm=dict(kwargs["adm"], **{key: value}))
        assert reference != resnet_visual_identity(**changed), key
    assert reference != resnet_visual_identity(
        **dict(kwargs, resnet_weights_sha256="other-resnet"))
    assert reference != resnet_visual_identity(
        **dict(kwargs, localization_identity_value="other-loc"))


def test_semantic_identity_depends_on_prompt_bank():
    kwargs = dict(unimed_code_sha256="code", checkpoint_sha256="weights",
                  text_encoder_sha256="text", prompt_bank_sha256="prompt",
                  localization_identity_value="loc",
                  sampling={"topk_slices": 3}, image_sha256={"a": "1"})
    reference = unimed_semantic_identity(**kwargs)
    assert reference != unimed_semantic_identity(
        **dict(kwargs, prompt_bank_sha256="new-prompt"))
    assert reference != unimed_semantic_identity(
        **dict(kwargs, checkpoint_sha256="new-weights"))


def _config(evidence_dirs, **adm) -> Config:
    payload = {
        "evidence": {"localization_dir": str(evidence_dirs["localization"]),
                     "visual_dir": str(evidence_dirs["visual"]),
                     "semantic_dir": str(evidence_dirs["semantic"])},
        "visual": {"output_dim": 8},
        "model": {"use_semantic": True},
        "runtime": {"require_zero_shot": True},
        "semantic": {"num_concepts": 3},
        "adm": {"enabled": True, "gamma": 0.4, "threshold": 0.6, "mode": "soft",
                "oracle_mask": False},
    }
    payload["adm"].update(adm)
    return Config(payload, ["test"])


def test_load_sources_accepts_matching_cache(manifest, evidence_dirs):
    sources = load_sources(_config(evidence_dirs), manifest)
    assert sources.d_visual == 8
    assert sources.visual_tokens == 16
    assert sources.semantic_tokens == 3


def test_load_sources_rejects_gamma_mismatch(manifest, evidence_dirs):
    with pytest.raises(CacheError):
        load_sources(_config(evidence_dirs, gamma=0.3), manifest)


def test_load_sources_rejects_disabled_mismatch(manifest, evidence_dirs):
    with pytest.raises(CacheError):
        load_sources(_config(evidence_dirs, enabled=False), manifest)


def test_visual_cache_rejects_wrong_localization_identity(evidence_dirs):
    cache = AdmVisualCache.load(evidence_dirs["visual"])
    with pytest.raises(CacheError):
        cache.verify([f"synthetic-{i:03d}-000" for i in range(4)],
                     localization_identity="not-the-same")


def test_cache_rejects_missing_study(evidence_dirs):
    cache = AdmLocalizationCache.load(evidence_dirs["localization"])
    with pytest.raises(CacheError):
        cache.verify(["only-one-study"])


def test_semantic_cache_requires_zero_shot(evidence_dirs, tmp_path):
    root = tmp_path / "semantic_copy"
    shutil.copytree(evidence_dirs["semantic"], root)
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["classification_mode"] = "prior"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    cache = AdmSemanticCache.load(root)
    with pytest.raises(CacheError):
        cache.verify([f"synthetic-{i:03d}-000" for i in range(4)],
                     require_zero_shot=True)
