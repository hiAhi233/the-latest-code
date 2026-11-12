"""无业务含义的小工具。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

__all__ = [
    "now_iso", "sha256_bytes", "sha256_file", "sha256_json", "stable_json_dumps",
    "read_json", "write_json", "read_jsonl", "write_jsonl", "append_jsonl",
    "seed_everything", "resolve_device", "Timer", "chunked", "expand_env",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(stable_json_dumps(obj).encode("utf-8"))


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], obj: Any, indent: int = 1) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    os.replace(tmp, p)


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    count = 0
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, p)
    return count


def append_jsonl(path: str | os.PathLike[str], row: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(spec: str | None = None) -> str:
    import torch

    if spec and spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Timer:
    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start


def chunked(seq: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: str) -> str:
    """展开 ``${VAR}``；未定义变量立即报错，避免静默使用错误路径。"""

    def repl(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in os.environ:
            raise KeyError(f"环境变量 {name} 未定义（配置值：{value!r}）")
        return os.environ[name]

    return _ENV_RE.sub(repl, os.path.expanduser(value))
