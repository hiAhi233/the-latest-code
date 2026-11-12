"""证据身份链（方案 §4.4）。

三级组合，任何上游变化都会产生新的缓存身份：

1. **定位**：定位权重、预处理、热图网格、阈值、影像哈希；
2. **视觉**：定位 identity + ResNet 权重 + ADM 参数（gamma/threshold/mode）+
   切片策略 + patch 网格 + 输出维度；
3. **语义**：UniMed 权重/文本塔/提示词 + 定位 identity + 切片策略 + 影像哈希。

报告 checkpoint 身份再加 tokenizer、模型配置、训练配置与源码树哈希。
"""

from __future__ import annotations

from typing import Any, Mapping

from ..utils import sha256_json

__all__ = ["adm_localization_identity", "resnet_visual_identity",
           "unimed_semantic_identity", "report_checkpoint_identity"]


def adm_localization_identity(detector_code_sha256: str, checkpoint_sha256: str,
                              preprocessing: Mapping[str, Any],
                              thresholds: Mapping[str, Any],
                              image_sha256: Mapping[str, Any]) -> str:
    return sha256_json({
        "kind": "adm_localization",
        "detector_code_sha256": detector_code_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing": dict(preprocessing),
        "thresholds": dict(thresholds),
        "image_sha256": dict(image_sha256),
    })


def resnet_visual_identity(localization_identity_value: str,
                           resnet_weights_sha256: str,
                           adm: Mapping[str, Any],
                           sampling: Mapping[str, Any],
                           visual: Mapping[str, Any],
                           image_sha256: Mapping[str, Any]) -> str:
    return sha256_json({
        "kind": "adm_resnet_visual",
        "localization_identity": localization_identity_value,
        "resnet_weights_sha256": resnet_weights_sha256,
        "adm": dict(adm),
        "sampling": dict(sampling),
        "visual": dict(visual),
        "image_sha256": dict(image_sha256),
    })


def unimed_semantic_identity(unimed_code_sha256: str, checkpoint_sha256: str,
                             text_encoder_sha256: str,
                             prompt_bank_sha256: str,
                             localization_identity_value: str,
                             sampling: Mapping[str, Any],
                             image_sha256: Mapping[str, Any]) -> str:
    return sha256_json({
        "kind": "adm_unimed_semantic",
        "unimed_code_sha256": unimed_code_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "text_encoder_sha256": text_encoder_sha256,
        "prompt_bank_sha256": prompt_bank_sha256,
        "localization_identity": localization_identity_value,
        "sampling": dict(sampling),
        "image_sha256": dict(image_sha256),
    })


def report_checkpoint_identity(evidence_identity_value: str, tokenizer_sha256: str,
                               model_config: Mapping[str, Any],
                               train_config: Mapping[str, Any],
                               source_code_sha256: str) -> str:
    return sha256_json({
        "kind": "report_checkpoint",
        "evidence_identity": evidence_identity_value,
        "tokenizer_sha256": tokenizer_sha256,
        "model_config": dict(model_config),
        "train_config": dict(train_config),
        "source_code_sha256": source_code_sha256,
    })
