"""配置加载：YAML + defaults 合并 + 环境变量展开 + 点路径访问。

实验配置形如::

    defaults:
      - configs/radgenome_data.yaml
      - configs/transformer_small.yaml
    experiment:
      name: f0_full
    model:
      use_roi: true
      use_semantic: true

约定：所有相对路径相对于**项目根目录**解析，而不是当前工作目录。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError
from .utils import expand_env

__all__ = ["Config", "load_config", "merge_dicts", "project_root"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射：{path}")
    return data


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return expand_env(value)
        except KeyError as exc:
            raise ConfigError(str(exc)) from exc
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


class Config(dict):
    def __init__(self, data: dict[str, Any], sources: list[str] | None = None) -> None:
        super().__init__(data)
        self.sources = list(sources or [])

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get_path(dotted, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"配置缺少必需字段 {dotted!r}（来源：{self.sources}）")
        return value

    def path(self, dotted: str, default: str | None = None) -> Path:
        value = self.get_path(dotted, default)
        if value is None:
            raise ConfigError(f"配置缺少路径字段 {dotted!r}")
        raw = Path(str(value))
        return raw if raw.is_absolute() else (project_root() / raw)

    def to_plain(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self))


_MISSING = object()


def load_config(path: str | os.PathLike[str], overrides: Mapping[str, Any] | None = None
                ) -> Config:
    root = project_root()
    seen: set[Path] = set()

    def _load(p: Path) -> tuple[dict[str, Any], list[str]]:
        p = p.resolve()
        if p in seen:
            raise ConfigError(f"defaults 循环引用：{p}")
        seen.add(p)
        raw = _load_raw(p)
        data: dict[str, Any] = {}
        sources: list[str] = []
        defaults = raw.pop("defaults", [])
        if defaults:
            if not isinstance(defaults, list):
                raise ConfigError(f"defaults 必须是列表：{p}")
            for entry in defaults:
                dep = Path(str(entry))
                dep = dep if dep.is_absolute() else (root / dep)
                dep_data, dep_sources = _load(dep)
                data = merge_dicts(data, dep_data)
                sources.extend(dep_sources)
        data = merge_dicts(data, raw)
        sources.append(str(p))
        return data, sources

    data, sources = _load(Path(path) if Path(path).is_absolute() else root / str(path))
    if overrides:
        data = merge_dicts(data, dict(overrides))
        sources.append("<overrides>")
    data = _expand(data)
    cfg = Config(data, sources)
    cfg.setdefault("project_root", str(root))
    return cfg
