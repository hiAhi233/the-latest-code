"""正式配置守卫（方案 §5、§10）。

formal=true 时：

* 只允许 ``evidence.mode: adm_simple``；
* 禁止 tiny / heuristic / gt_oracle / hash / probe / prior 占位后端；
* 禁止 ROI 支路与几何/位置 token（``model.use_roi_branch`` / ``use_geometry``）；
* 禁止 ``adm.oracle_mask: true``（Oracle 只能单独命名并 ``runtime.formal: false``）；
* 必须提供三类证据缓存目录与真实权重路径。
"""

from __future__ import annotations

from .config import Config
from .errors import ConfigError

__all__ = ["PLACEHOLDER_BACKENDS", "is_formal", "validate_formal"]

PLACEHOLDER_BACKENDS = {"tiny", "tiny_cnn", "heuristic", "gt_oracle", "hash",
                        "hash_fallback", "prior", "probe", "random"}


def is_formal(config: Config) -> bool:
    return bool(config.get_path("runtime.formal", False))


def validate_formal(config: Config) -> None:
    if not is_formal(config):
        return
    forbidden = set(config.get_path("runtime.forbid_backends", PLACEHOLDER_BACKENDS))
    backends = {
        "visual.backbone": config.get_path("visual.backbone", ""),
        "localization.backend": config.get_path("localization.backend", ""),
        "semantic.backend": config.get_path("semantic.backend", ""),
    }
    bad = {key: value for key, value in backends.items() if value in forbidden}
    if bad:
        raise ConfigError(
            f"formal 配置检测到占位 backend：{bad}；正式路径必须使用真实定位/ResNet/UniMed。")

    mode = str(config.get_path("evidence.mode", "builtin"))
    if mode != "adm_simple":
        raise ConfigError(f"formal 配置必须设置 evidence.mode: adm_simple，当前 {mode!r}")
    if bool(config.get_path("model.use_roi_branch", False)):
        raise ConfigError("正式 ADM 路径不允许 model.use_roi_branch: true")
    if bool(config.get_path("model.use_geometry", False)):
        raise ConfigError("正式 ADM 路径不允许 model.use_geometry: true")
    if bool(config.get_path("adm.oracle_mask", False)):
        raise ConfigError("formal 配置不允许 adm.oracle_mask: true（Oracle 只能用于 O0 上限）")
    if str(config.get_path("adm.mode", "soft")) not in ("soft", "hard"):
        raise ConfigError("adm.mode 只允许 soft / hard")
    gamma = float(config.get_path("adm.gamma", 0.4))
    threshold = float(config.get_path("adm.threshold", 0.6))
    if not 0.0 <= gamma <= 1.0:
        raise ConfigError(f"adm.gamma 必须在 [0,1]，当前 {gamma}")
    if not 0.0 < threshold <= 1.0:
        raise ConfigError(f"adm.threshold 必须在 (0,1]，当前 {threshold}")
    if str(config.get_path("visual.backbone", "")) != "resnet101":
        raise ConfigError("首版 visual.backbone 必须是 resnet101")
    if not bool(config.get_path("visual.frozen", True)):
        raise ConfigError("首版 ResNet-101 必须冻结（visual.frozen: true）")
    if not bool(config.get_path("semantic.frozen", True)):
        raise ConfigError("首版 UniMed-CLIP 必须冻结（semantic.frozen: true）")
    if not bool(config.get_path("semantic.use_original_image", True)):
        raise ConfigError("语义支路必须使用原图（semantic.use_original_image: true）")
    for key in ("evidence.localization_dir", "evidence.visual_dir",
                "evidence.semantic_dir"):
        if not config.get_path(key):
            raise ConfigError(f"formal 配置缺少 {key}")
    for key in ("localization.checkpoint", "semantic.weights",
                "semantic.text_encoder", "semantic.prompts"):
        if not config.get_path(key):
            raise ConfigError(f"formal 配置缺少 {key}")
