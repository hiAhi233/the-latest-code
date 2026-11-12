"""ADM 证据源装配：配置/checkpoint -> 三类缓存 + 身份校验。

训练、生成、评价都从这里拿证据，保证三者读的是同一批缓存和同一条身份链。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config, project_root
from .errors import CacheError
from .features.adm_cache import (AdmLocalizationCache, AdmSemanticCache,
                                 AdmVisualCache)
from .data.manifest import Manifest

__all__ = ["AdmEvidenceSources", "load_sources", "source_paths", "load_sources_from_extra"]


def _resolve(directory: str | Path) -> Path:
    path = Path(directory)
    return path if path.is_absolute() else project_root() / path


@dataclass
class AdmEvidenceSources:
    localization: AdmLocalizationCache
    visual: AdmVisualCache
    semantic: AdmSemanticCache | None
    localization_dir: str
    visual_dir: str
    semantic_dir: str

    @property
    def use_semantic(self) -> bool:
        return self.semantic is not None

    @property
    def d_visual(self) -> int:
        return self.visual.d_visual

    @property
    def visual_tokens(self) -> int:
        return self.visual.tokens_per_study

    @property
    def semantic_tokens(self) -> int:
        return len(self.semantic.concept_ids) if self.semantic is not None else 0

    @property
    def d_proto(self) -> int:
        return self.semantic.d_proto if self.semantic is not None else 512

    @property
    def cache_identity(self) -> dict[str, str]:
        identity = {"localization": self.localization.identity,
                    "visual": self.visual.identity}
        if self.semantic is not None:
            identity["semantic"] = self.semantic.identity
        return identity

    def admission(self) -> dict[str, Any]:
        return {"adm": dict(self.visual.meta.get("adm") or {}),
                "sampling": dict(self.visual.meta.get("sampling") or {}),
                "resnet_weights_sha256": self.visual.meta.get("resnet_weights_sha256")}


def source_paths(config: Config) -> dict[str, Any]:
    return {
        "localization_dir": str(config.get_path("evidence.localization_dir", "")),
        "visual_dir": str(config.get_path("evidence.visual_dir", "")),
        "semantic_dir": str(config.get_path("evidence.semantic_dir", "")),
        "adm": dict(config.get("adm") or {}),
        "sampling": {
            "topk_slices": int(config.get_path("localization.topk_slices", 3)),
            "source_sequence": str(config.get_path("localization.source_sequence", "t2f")),
        },
        "visual": {
            "backbone": str(config.get_path("visual.backbone", "resnet101")),
            "patch_grid": int(config.get_path("visual.patch_grid", 3)),
            "output_dim": int(config.get_path("visual.output_dim", 256)),
            "modalities": list(config.get_path("visual.modalities",
                                               ["t1n", "t1c", "t2w", "t2f"])),
        },
    }


def load_sources(config: Config, manifest: Manifest,
                 use_semantic: bool | None = None,
                 require_predicted: bool = True) -> AdmEvidenceSources:
    studies = [study.study_id for study in manifest]
    localization = AdmLocalizationCache.load(_resolve(config.require("evidence.localization_dir")))
    localization.verify(studies, require_predicted=require_predicted)
    visual = AdmVisualCache.load(_resolve(config.require("evidence.visual_dir")))
    output_dim = int(config.get_path("visual.output_dim", visual.d_visual))
    visual.verify(studies, localization_identity=localization.identity,
                  d_visual=output_dim)
    if use_semantic is None:
        use_semantic = bool(config.get_path("model.use_semantic", True))
    semantic = None
    semantic_dir = str(config.get_path("evidence.semantic_dir", ""))
    if use_semantic:
        if not semantic_dir:
            raise CacheError("use_semantic=True 但没有 evidence.semantic_dir")
        semantic = AdmSemanticCache.load(_resolve(semantic_dir))
        semantic.verify(studies, localization_identity=localization.identity,
                        require_zero_shot=bool(config.get_path("runtime.require_zero_shot",
                                                              True)))
        expected = int(config.get_path("semantic.num_concepts", 13))
        if len(semantic.concept_ids) != expected:
            raise CacheError(
                f"语义缓存概念数 {len(semantic.concept_ids)} != 配置 {expected}")
    # ADM 参数必须与导出缓存一致，否则视觉 token 与配置描述的不是同一个变换
    adm_meta = dict(visual.meta.get("adm") or {})
    for key in ("enabled", "oracle_mask"):
        if key in adm_meta and config.get_path(f"adm.{key}") is not None:
            if bool(adm_meta[key]) != bool(config.get_path(f"adm.{key}")):
                raise CacheError(
                    f"视觉缓存的 adm.{key}={adm_meta[key]!r} 与配置 "
                    f"{config.get_path(f'adm.{key}')!r} 不一致；"
                    "请用当前配置重新导出视觉缓存。")
    for key in ("gamma", "threshold", "mode"):
        if key in adm_meta and config.get_path(f"adm.{key}") is not None:
            configured = config.get_path(f"adm.{key}")
            if isinstance(configured, float):
                same = abs(float(configured) - float(adm_meta[key])) < 1e-9
            else:
                same = str(configured) == str(adm_meta[key])
            if not same:
                raise CacheError(
                    f"视觉缓存的 adm.{key}={adm_meta[key]!r} 与配置 {configured!r} 不一致；"
                    "请用当前 ADM 参数重新导出视觉缓存。")
    return AdmEvidenceSources(
        localization=localization, visual=visual, semantic=semantic,
        localization_dir=str(config.get_path("evidence.localization_dir")),
        visual_dir=str(config.get_path("evidence.visual_dir")),
        semantic_dir=semantic_dir)


def load_sources_from_extra(extra: dict[str, Any], manifest: Manifest
                            ) -> AdmEvidenceSources:
    """从 checkpoint 的 extra 重建证据源，并核对身份（生成/评价用）。"""
    paths = dict(extra.get("evidence_paths") or {})
    required = ("localization_dir", "visual_dir")
    missing = [key for key in required if not paths.get(key)]
    if missing:
        raise CacheError(f"checkpoint 缺少 evidence_paths.{missing}")
    studies = [study.study_id for study in manifest]
    localization = AdmLocalizationCache.load(_resolve(paths["localization_dir"]))
    localization.verify(studies, require_predicted=not bool(extra.get("allow_oracle", False)))
    visual = AdmVisualCache.load(_resolve(paths["visual_dir"]))
    visual.verify(studies, localization_identity=localization.identity,
                  d_visual=int(extra.get("d_visual", 0)) or None)
    semantic = None
    identities = dict(extra.get("cache_identity") or {})
    if bool(extra.get("use_semantic", True)):
        if not paths.get("semantic_dir"):
            raise CacheError("checkpoint 记录 use_semantic=True 但缺少 semantic_dir")
        semantic = AdmSemanticCache.load(_resolve(paths["semantic_dir"]))
        semantic.verify(studies, localization_identity=localization.identity,
                        require_zero_shot=True)
    for kind, actual in (("localization", localization.identity),
                         ("visual", visual.identity),
                         ("semantic", semantic.identity if semantic else None)):
        expected = identities.get(kind)
        if expected and actual and str(expected) != str(actual):
            raise CacheError(
                f"{kind} 证据 identity 与 checkpoint 不一致：{str(actual)[:12]} != "
                f"{str(expected)[:12]}；请重建缓存或换回对应 checkpoint。")
    if bool(extra.get("manifest_sha256")) and \
            str(extra["manifest_sha256"]) != str(manifest.sha256):
        raise CacheError("checkpoint 绑定的 manifest 与当前不一致")
    return AdmEvidenceSources(
        localization=localization, visual=visual, semantic=semantic,
        localization_dir=str(paths["localization_dir"]),
        visual_dir=str(paths["visual_dir"]),
        semantic_dir=str(paths.get("semantic_dir", "")))
