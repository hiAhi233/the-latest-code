"""MedADM-RRG：少样本异常检测 + UniMed-CLIP 语义 + ADM 增强 + 报告 Transformer。

正式主线（方案 §1）::

    图像 → 冻结异常检测器（少样本） → 预测热图 A
         → ADM 增强 Iadm = I×(γ+(1-γ)A)
         → 冻结 ResNet-101 → 108 个 visual patch tokens
    原图 → 冻结 UniMed-CLIP 零样本疾病语义 → 13 个 semantic tokens
    [108 visual; 13 semantic] → 2 层 Encoder → 4 层 Decoder → Findings + Impression

本包只提供工程实现；所有上游证据都是**预测**，不用于临床诊断。
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
