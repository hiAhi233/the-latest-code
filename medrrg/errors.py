"""项目异常类型。"""

from __future__ import annotations


class MedRRGError(Exception):
    """所有项目异常的基类。"""


class ConfigError(MedRRGError):
    """配置缺字段、类型错误或互相矛盾。"""


class DataError(MedRRGError):
    """manifest、报告、NIfTI 等数据问题。"""


class CacheError(MedRRGError):
    """缓存身份（病例集合、权重哈希、ontology）不一致。"""


class LeakageError(MedRRGError):
    """GT/Oracle 证据进入默认实验，或患者跨 split。"""


class ContractError(MedRRGError):
    """张量形状/合同不满足约定。"""
