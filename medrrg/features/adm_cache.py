"""三类 ADM 证据缓存的读取与严格校验（方案 §4）。

缓存目录（``<dir>`` 由配置的 ``evidence.*_dir`` 指定）::

    caches/adm/localization/<run>/
      meta.json  index.jsonl  arrays/<study_id>.npz
    caches/adm/resnet/<run>/
      meta.json  index.jsonl  visual/<study_id>.npz
    caches/adm/unimed/<run>/
      meta.json  index.jsonl  semantic/<study_id>.npz  prompts.json

所有读取都做身份校验：病例集合、格式版本、上游 identity、``source=predicted``
（Oracle 需显式放行）、字段形状与数值有限性。任何不一致立即失败，不静默复用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..errors import CacheError, DataError
from ..utils import read_json

__all__ = [
    "LOCALIZATION_FORMAT", "VISUAL_FORMAT", "SEMANTIC_FORMAT",
    "AdmLocalizationItem", "AdmVisualItem", "AdmSemanticItem",
    "AdmLocalizationCache", "AdmVisualCache", "AdmSemanticCache",
    "missing_npz_keys",
]

LOCALIZATION_FORMAT = "medadm.adm.localization.v1"
VISUAL_FORMAT = "medadm.adm.resnet.v2"
SEMANTIC_FORMAT = "medadm.adm.unimed.v1"

LOCALIZATION_ARRAYS = ("heatmap_3d", "score_per_slice", "abnormal_score")
VISUAL_ARRAYS = ("visual_tokens", "visual_tokens_null", "visual_mask",
                 "visual_modality", "visual_z", "visual_view",
                 "visual_patch_row", "visual_patch_col")
SEMANTIC_ARRAYS = ("concept_ids", "text_prototypes", "probabilities",
                   "confidence", "margins")


def missing_npz_keys(path: str | Path, keys: Iterable[str]) -> list[str]:
    """返回 NPZ 中缺失的字段名；用于导出脚本复用旧缓存前的完整性检查。"""
    with np.load(Path(path), allow_pickle=False) as payload:
        return [str(key) for key in keys if key not in payload]


def _read_index(root: Path, kind: str) -> dict[str, dict[str, Any]]:
    path = root / "index.jsonl"
    if not path.is_file():
        raise CacheError(f"{kind} 缓存缺少 index.jsonl：{root}")
    index: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[str(row["study_id"])] = row
    if not index:
        raise CacheError(f"{kind} 缓存 index.jsonl 为空：{root}")
    return index


class _BaseCache:
    kind = "base"
    format = ""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        if not meta_path.is_file():
            raise CacheError(f"{self.kind} 缓存缺少 meta.json：{self.root}")
        self.meta = read_json(meta_path)
        if self.meta.get("format") != self.format:
            raise CacheError(
                f"{self.kind} 缓存格式 {self.meta.get('format')!r} != {self.format!r}")
        self.index = _read_index(self.root, self.kind)
        self.identity = str(self.meta.get("identity", ""))
        if not self.identity:
            raise CacheError(f"{self.kind} 缓存 meta.json 缺少 identity")

    def _verify_studies(self, expected_study_ids: Iterable[str]) -> None:
        expected = sorted({str(item) for item in expected_study_ids})
        actual = sorted(self.index)
        if actual != expected:
            raise CacheError(
                f"{self.kind} 缓存病例集合与 manifest 不一致："
                f"缺少 {sorted(set(expected) - set(actual))[:5]}，"
                f"多出 {sorted(set(actual) - set(expected))[:5]}")
        declared = sorted(str(item) for item in self.meta.get("studies", []))
        if declared and declared != expected:
            raise CacheError(f"{self.kind} 缓存 meta.studies 与 index.jsonl 不一致")

    def _row(self, study_id: str) -> dict[str, Any]:
        if study_id not in self.index:
            raise KeyError(f"{self.kind} 缓存没有病例 {study_id}")
        return self.index[study_id]

    def _array_path(self, row: dict[str, Any], key: str = "array_path") -> Path:
        path = self.root / str(row[key])
        if not path.is_file():
            raise CacheError(f"{self.kind} 缓存数组不存在：{path}")
        return path

    def __len__(self) -> int:
        return len(self.index)

    @classmethod
    def load(cls, root: str | Path) -> "_BaseCache":
        return cls(root)


@dataclass
class AdmLocalizationItem:
    study_id: str
    heatmap: np.ndarray              # [Z, heat, heat] float32
    score_per_slice: np.ndarray      # [Z]
    abnormal_score: float
    shape_zyx: tuple[int, int, int]
    selected_z: list[int]
    heat_size: int
    threshold: float
    source: str
    source_sequence: str
    checkpoint_sha256: str

    @property
    def num_slices(self) -> int:
        return int(self.heatmap.shape[0])

    def heatmap_at(self, z: int) -> np.ndarray:
        if not 0 <= int(z) < self.num_slices:
            raise DataError(f"{self.study_id}: z={z} 超出 [0,{self.num_slices})")
        return np.asarray(self.heatmap[int(z)], dtype=np.float32)


class AdmLocalizationCache(_BaseCache):
    kind = "localization"
    format = LOCALIZATION_FORMAT

    def verify(self, expected_study_ids: Iterable[str], require_predicted: bool = True
               ) -> dict[str, Any]:
        self._verify_studies(expected_study_ids)
        allowed = {"predicted"} if require_predicted else {"predicted", "ground_truth_oracle"}
        if str(self.meta.get("source", "predicted")) not in allowed:
            raise CacheError(
                f"定位缓存 source={self.meta.get('source')!r} 不在 {allowed}；"
                "Oracle 必须使用独立目录与独立配置。")
        for study_id, row in self.index.items():
            row_source = str(row.get("source", self.meta.get("source", "predicted")))
            if row_source not in allowed:
                raise CacheError(f"{study_id}: source={row_source!r} 不被当前配置允许")
            self._array_path(row)
            shape = row.get("shape_zyx")
            if not shape or len(shape) != 3:
                raise CacheError(f"{study_id}: shape_zyx 必须是 3 个整数")
            if int(row.get("heat_size", 0)) <= 0:
                raise CacheError(f"{study_id}: heat_size 非法")
        return self.meta

    def get(self, study_id: str) -> AdmLocalizationItem:
        row = self._row(study_id)
        path = self._array_path(row)
        with np.load(path, allow_pickle=False) as payload:
            heatmap = np.asarray(payload["heatmap_3d"], dtype=np.float32)
            scores = np.asarray(payload["score_per_slice"], dtype=np.float32)
            abnormal = float(payload["abnormal_score"]) if "abnormal_score" in payload \
                else float(row.get("abnormal_score", 0.0))
        if heatmap.ndim != 3:
            raise DataError(f"{study_id}: heatmap_3d 必须是 [Z,H,W]")
        if scores.shape[0] != heatmap.shape[0]:
            raise DataError(f"{study_id}: score_per_slice 与 Z 不一致")
        if not np.isfinite(heatmap).all() or not np.isfinite(scores).all():
            raise DataError(f"{study_id}: 定位缓存包含 NaN/Inf")
        selected = [int(z) for z in row.get("selected_z", [])]
        if not selected:
            selected = [int(z) for z in np.argsort(-scores)[:3]]
        return AdmLocalizationItem(
            study_id=study_id, heatmap=heatmap, score_per_slice=scores,
            abnormal_score=abnormal,
            shape_zyx=tuple(int(v) for v in row["shape_zyx"]),
            selected_z=selected, heat_size=int(row["heat_size"]),
            threshold=float(row.get("threshold", 0.5)),
            source=str(row.get("source", "predicted")),
            source_sequence=str(row.get("source_sequence", "")),
            checkpoint_sha256=str(row.get("checkpoint_sha256", "")))


@dataclass
class AdmVisualItem:
    tokens: np.ndarray               # [Nv, D]
    tokens_null: np.ndarray          # [Nv, D]（未增强原图的 tokens，用于 null-adm）
    mask: np.ndarray                 # [Nv] bool
    modality: np.ndarray             # [Nv] int
    z: np.ndarray                    # [Nv] float in [0,1]
    view: np.ndarray                 # [Nv] int
    patch_row: np.ndarray            # [Nv] int，3×3 patch 网格行号
    patch_col: np.ndarray            # [Nv] int，3×3 patch 网格列号
    selected_z: list[int]
    adm_applied: bool


class AdmVisualCache(_BaseCache):
    kind = "visual"
    format = VISUAL_FORMAT

    @property
    def d_visual(self) -> int:
        return int(self.meta.get("d_visual", 0))

    @property
    def tokens_per_study(self) -> int:
        return int(self.meta.get("tokens_per_study", 0))

    @property
    def patch_grid(self) -> int:
        return int((self.meta.get("visual") or {}).get("patch_grid", 3))

    @property
    def localization_identity(self) -> str:
        return str(self.meta.get("localization_identity", ""))

    def verify(self, expected_study_ids: Iterable[str],
               localization_identity: str | None = None,
               d_visual: int | None = None) -> dict[str, Any]:
        self._verify_studies(expected_study_ids)
        if localization_identity is not None and \
                self.localization_identity != str(localization_identity):
            raise CacheError(
                "视觉缓存绑定的定位 identity 与当前定位缓存不一致；"
                "请用最新定位缓存重新导出视觉特征。")
        if d_visual is not None and self.d_visual != int(d_visual):
            raise CacheError(f"视觉缓存 d_visual={self.d_visual} != 期望 {d_visual}")
        grid = self.patch_grid
        if grid < 1:
            raise CacheError("视觉缓存 meta.visual.patch_grid 非法")
        for study_id, row in self.index.items():
            self._array_path(row, "array_path")
            if int(row.get("num_tokens", 0)) <= 0:
                raise CacheError(f"{study_id}: num_tokens 非法")
        return self.meta

    def get(self, study_id: str) -> AdmVisualItem:
        row = self._row(study_id)
        with np.load(self._array_path(row, "array_path"), allow_pickle=False) as payload:
            missing = [key for key in ("visual_patch_row", "visual_patch_col")
                       if key not in payload]
            if missing:
                raise CacheError(
                    f"{study_id}: 视觉缓存缺少 {missing}（v2 起需要二维 patch 位置）；"
                    "请用 step02b 重新导出视觉缓存。")
            tokens = np.asarray(payload["visual_tokens"], dtype=np.float32)
            tokens_null = (np.asarray(payload["visual_tokens_null"], dtype=np.float32)
                           if "visual_tokens_null" in payload else tokens.copy())
            mask = np.asarray(payload["visual_mask"]) > 0
            modality = np.asarray(payload["visual_modality"], dtype=np.int64)
            z_value = np.asarray(payload["visual_z"], dtype=np.float32)
            view = (np.asarray(payload["visual_view"], dtype=np.int64)
                    if "visual_view" in payload else np.zeros(tokens.shape[0], dtype=np.int64))
            patch_row = np.asarray(payload["visual_patch_row"], dtype=np.int64)
            patch_col = np.asarray(payload["visual_patch_col"], dtype=np.int64)
            selected = [int(item) for item in payload["selected_z"]] \
                if "selected_z" in payload else [int(item) for item in row.get("selected_z", [])]
            adm_applied = bool(payload["adm_applied"]) if "adm_applied" in payload \
                else bool(row.get("adm_applied", True))
        if tokens.ndim != 2 or tokens_null.shape != tokens.shape:
            raise DataError(f"{study_id}: visual tokens 必须是 [Nv,D] 且两者同形")
        if self.d_visual and tokens.shape[1] != self.d_visual:
            raise CacheError(f"{study_id}: tokens 维度与 meta.d_visual 不一致")
        count = tokens.shape[0]
        for name, value in (("mask", mask), ("modality", modality), ("z", z_value),
                            ("view", view), ("patch_row", patch_row),
                            ("patch_col", patch_col)):
            if value.shape[0] != count:
                raise CacheError(f"{study_id}: {name} 长度与 tokens 不一致")
        grid = max(1, self.patch_grid)
        if ((patch_row < 0) | (patch_row >= grid)).any() or \
                ((patch_col < 0) | (patch_col >= grid)).any():
            raise DataError(f"{study_id}: patch_row/patch_col 超出 [0,{grid})")
        if not np.isfinite(tokens).all() or not np.isfinite(tokens_null).all():
            raise DataError(f"{study_id}: visual tokens 包含 NaN/Inf")
        if z_value.size and ((z_value < 0).any() or (z_value > 1).any()):
            raise DataError(f"{study_id}: visual_z 必须归一化到 [0,1]")
        return AdmVisualItem(tokens=tokens, tokens_null=tokens_null, mask=mask,
                             modality=modality, z=z_value, view=view,
                             patch_row=patch_row, patch_col=patch_col,
                             selected_z=selected, adm_applied=adm_applied)


@dataclass
class AdmSemanticItem:
    concept_ids: list[str]
    text_prototypes: np.ndarray      # [K, d_proto]
    probabilities: np.ndarray        # [K]（未校准，不是诊断概率）
    confidence: np.ndarray           # [K]
    margins: np.ndarray              # [K]


class AdmSemanticCache(_BaseCache):
    kind = "semantic"
    format = SEMANTIC_FORMAT

    @property
    def concept_ids(self) -> list[str]:
        return [str(item) for item in self.meta.get("concept_ids", [])]

    @property
    def d_proto(self) -> int:
        return int(self.meta.get("d_proto", 0))

    @property
    def localization_identity(self) -> str:
        return str(self.meta.get("localization_identity", ""))

    def verify(self, expected_study_ids: Iterable[str],
               localization_identity: str | None = None,
               require_zero_shot: bool = True) -> dict[str, Any]:
        self._verify_studies(expected_study_ids)
        if localization_identity is not None and \
                self.localization_identity != str(localization_identity):
            raise CacheError(
                "语义缓存绑定的定位 identity 与当前定位缓存不一致；"
                "请用最新定位缓存重新导出语义特征。")
        if require_zero_shot and self.meta.get("classification_mode") != "zero_shot":
            raise CacheError(
                f"正式配置要求 UniMed zero_shot 语义，缓存记录为 "
                f"{self.meta.get('classification_mode')!r}")
        if not self.concept_ids:
            raise CacheError("语义缓存 meta 缺少 concept_ids")
        for study_id, row in self.index.items():
            self._array_path(row, "array_path")
        return self.meta

    def get(self, study_id: str) -> AdmSemanticItem:
        row = self._row(study_id)
        with np.load(self._array_path(row, "array_path"), allow_pickle=False) as payload:
            concept_ids = [str(item) for item in payload["concept_ids"].tolist()]
            prototypes = np.asarray(payload["text_prototypes"], dtype=np.float32)
            probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
            confidence = np.asarray(payload["confidence"], dtype=np.float32)
            margins = np.asarray(payload["margins"], dtype=np.float32)
        if prototypes.shape[0] != len(concept_ids):
            raise DataError(f"{study_id}: 原型数量与概念数不一致")
        for name, value in (("probabilities", probabilities),
                            ("confidence", confidence), ("margins", margins)):
            if value.shape[0] != len(concept_ids):
                raise DataError(f"{study_id}: {name} 数量与概念数不一致")
            if not np.isfinite(value).all():
                raise DataError(f"{study_id}: {name} 包含 NaN/Inf")
        if concept_ids != self.concept_ids:
            raise CacheError(f"{study_id}: 概念顺序与 meta 不一致")
        return AdmSemanticItem(concept_ids=concept_ids, text_prototypes=prototypes,
                               probabilities=probabilities, confidence=confidence,
                               margins=margins)
