"""ADM 简化方案的数据合同（方案 §3、§4、§5.4）。

正式证据只有两类 token：

* ``visual_tokens``：异常增强（或原图）切片经 ResNet-101 得到的 patch tokens；
* ``semantic_tokens``：UniMed-CLIP 疾病语义投影得到的概念 tokens。

正式路径**没有** ROI token 支路、没有位置/几何 token；本文件用
:meth:`AdmEvidenceBatch.assert_adm_contract` 把这条约束写成可执行断言，
``test_adm_dataset.py`` 会调用它。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import ContractError, LeakageError

SCHEMA_STUDY = "medadm.study.v1"
SPLITS = ("train", "val", "test")
PREDICTED = "predicted"
ORACLE = "ground_truth_oracle"
EVIDENCE_SOURCES = (PREDICTED, ORACLE)
ADM_EVIDENCE_MODE = "adm_simple"
TOKEN_TYPES = ("visual", "semantic")
FORBIDDEN_BATCH_FIELDS = ("roi", "geometry", "coord", "box")


@dataclass(frozen=True)
class ImageRef:
    """一个序列/期相。``group`` 用于 modality embedding。"""

    path: str
    group: str

    def __post_init__(self) -> None:
        if not self.path or not self.group:
            raise ContractError("ImageRef.path/group 不能为空")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "group": self.group}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ImageRef":
        return cls(path=str(raw["path"]), group=str(raw["group"]))


@dataclass
class Report:
    findings: str = ""
    impression: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.findings.strip() or self.impression.strip())

    def text(self) -> str:
        return " ".join(part for part in (self.findings.strip(),
                                          self.impression.strip()) if part)

    def target_text(self, control_tokens: bool = True) -> str:
        parts = []
        if self.findings.strip():
            body = self.findings.strip()
            parts.append(f"<FINDINGS> {body} </FINDINGS>" if control_tokens else body)
        if self.impression.strip():
            body = self.impression.strip()
            parts.append(f"<IMPRESSION> {body} </IMPRESSION>" if control_tokens else body)
        return " ".join(parts)

    def to_dict(self) -> dict[str, str]:
        return {"findings": self.findings, "impression": self.impression}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "Report":
        raw = raw or {}
        return cls(findings=str(raw.get("findings", "") or ""),
                   impression=str(raw.get("impression", "") or ""))


@dataclass
class Study:
    """manifest 的一行（方案 §4）。"""

    study_id: str
    patient_id: str
    dataset_id: str
    split: str
    modality: str
    language: str
    images: list[ImageRef] = field(default_factory=list)
    report: Report = field(default_factory=Report)
    localization_cache: str = ""
    semantic_cache: str = ""
    visual_cache: str = ""
    schema_version: str = SCHEMA_STUDY
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_STUDY:
            raise ContractError(f"不支持的 study schema {self.schema_version!r}")
        for name in ("study_id", "patient_id", "dataset_id"):
            if not str(getattr(self, name)).strip():
                raise ContractError(f"Study.{name} 不能为空")
        if self.split not in SPLITS:
            raise ContractError(f"未知 split={self.split!r}，只允许 {SPLITS}")
        if not self.images:
            raise ContractError(f"{self.study_id}: 至少需要一个图像")

    @property
    def groups(self) -> list[str]:
        return [image.group for image in self.images]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "patient_id": self.patient_id,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "modality": self.modality,
            "language": self.language,
            "images": [image.to_dict() for image in self.images],
            "localization_cache": self.localization_cache,
            "semantic_cache": self.semantic_cache,
            "visual_cache": self.visual_cache,
            "report": self.report.to_dict(),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Study":
        return cls(
            schema_version=str(raw.get("schema_version", SCHEMA_STUDY)),
            study_id=str(raw["study_id"]),
            patient_id=str(raw["patient_id"]),
            dataset_id=str(raw["dataset_id"]),
            split=str(raw["split"]),
            modality=str(raw.get("modality", "OTHER")),
            language=str(raw.get("language", "en")),
            images=[ImageRef.from_dict(item) for item in raw.get("images", [])],
            report=Report.from_dict(raw.get("report")),
            localization_cache=str(raw.get("localization_cache", "")),
            semantic_cache=str(raw.get("semantic_cache", "")),
            visual_cache=str(raw.get("visual_cache", "")),
            meta=dict(raw.get("meta") or {}),
        )


@dataclass
class AdmEvidenceBatch:
    """报告模型的唯一输入（方案 §5.4）。

    形状约定::

        visual_tokens        (B, Nv, d_visual)   Nv = topk_z × modalities × patch_grid²
        visual_mask          (B, Nv)             True=有效
        visual_modality      (B, Nv)             0..3 -> T1/T1C/T2/FLAIR
        visual_z             (B, Nv)             归一化 z ∈ [0,1]
        visual_view          (B, Nv)             视图 id（首版恒为 0）
        visual_patch_row     (B, Nv)             3×3 patch 网格行号（0..2）
        visual_patch_col     (B, Nv)             3×3 patch 网格列号（0..2）
        visual_available     (B,)                ADM 证据是否可用
        semantic_prototypes  (B, K, d_proto)
        semantic_scores      (B, K)              UniMed 概率（未校准，不是诊断概率）
        semantic_confidence  (B, K)
        semantic_margin      (B, K)
        semantic_mask        (B, K)
        semantic_available   (B,)
    """

    study_ids: list[str] = field(default_factory=list)
    patient_ids: list[str] = field(default_factory=list)
    visual_tokens: Any = None
    visual_mask: Any = None
    visual_modality: Any = None
    visual_z: Any = None
    visual_view: Any = None
    visual_patch_row: Any = None
    visual_patch_col: Any = None
    visual_available: Any = None
    semantic_prototypes: Any = None
    semantic_scores: Any = None
    semantic_confidence: Any = None
    semantic_margin: Any = None
    semantic_mask: Any = None
    semantic_available: Any = None

    def __len__(self) -> int:
        return len(self.study_ids)

    def to(self, device: Any) -> "AdmEvidenceBatch":
        import torch

        data = dict(self.__dict__)
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(device)
        return AdmEvidenceBatch(**data)

    def assert_adm_contract(self) -> None:
        """正式路径不允许出现 ROI / geometry / 坐标字段。"""
        names = [item.name for item in fields(self)]
        bad = [name for name in names
               if any(token in name for token in FORBIDDEN_BATCH_FIELDS)]
        if bad:
            raise ContractError(f"ADM 正式批不允许 ROI/geometry 字段：{bad}")
        required = ("visual_tokens", "visual_mask", "visual_modality", "visual_z",
                    "visual_view", "visual_patch_row", "visual_patch_col",
                    "semantic_prototypes", "semantic_scores",
                    "semantic_confidence", "semantic_margin", "semantic_mask")
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ContractError(f"ADM 正式批缺少字段：{missing}")


def assert_source(source: str, allow_oracle: bool = False) -> None:
    if source not in EVIDENCE_SOURCES:
        raise ContractError(f"未知证据来源 {source!r}，只允许 {EVIDENCE_SOURCES}")
    if source == ORACLE and not allow_oracle:
        raise LeakageError(
            "默认实验收到 ground_truth_oracle 证据；Oracle 只能用于单独命名的实验配置。")
