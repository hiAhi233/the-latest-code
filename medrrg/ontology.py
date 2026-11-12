"""版本化概念表：MVP 只保留少量明确概念（方案 §4.4）。

否定判断按**句内作用域**处理：

* 句子切分在 ``.;!?。；！？`` 处（``.`` 只在后接空白/结尾时切，避免切碎小数）；
* 否定只从**同一句**匹配位置之前寻找前置否定线索（no / without / absence of …），
  不会跨句泄漏；
* 后置否定必须**紧贴该概念自己的谓语**（"is absent" / "was not seen" / "阴性"），
  并且只看当前分句（逗号/分号/冒号/对比连词处截断）；因此
  "edema is present and enhancement is absent" 不会把 edema 判成否定；
* 并列主语共享后置谓语（"edema and hemorrhage are absent"）由单独的并列规则处理；
* 出现对比连词（but/however/whereas/although/yet）时，前置线索只从连词之后算起，
  避免 "no edema but a glioma" 把 glioma 误判为否定。
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .errors import ConfigError
from .utils import sha256_json

__all__ = ["Ontology", "NEGATION_CUES", "POST_NEGATION_CUES"]

# 同一句内、匹配位置之前出现的否定线索
NEGATION_CUES = ("no evidence of", "no evidence for", "no ", "without",
                 "negative for", "absence of", "free of", "lack of", "not ",
                 "never ", "neither ", "ruled out", "excluded ", "未见", "未发现",
                 "未显示", "无", "排除", "否认")
# 同一句内、紧邻匹配位置之后出现的后置否定（必须紧贴该概念的谓语）
POST_NEGATION_CUES = ("absent", "negative", "not seen", "not observed",
                      "not identified", "not detected", "not present",
                      "not evident", "not found", "not demonstrated", "阴性")
_CONTRASTIVE = (" but ", " however ", " whereas ", " although ", " though ", " yet ")
_SENTENCE_RE = re.compile(r"[^.;!?。；！？\n]+(?:[.;!?。；！？](?=\s|$)|$)")
# 后置否定只在**本分句**内生效：遇到逗号/分号/冒号或对比连词即截断
_CLAUSE_END_RE = re.compile(
    r"[,;:，；：]|\b(?:" + "|".join(
        re.escape(marker.strip()) for marker in _CONTRASTIVE) + r")\b")
_PRE_NEGATION_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(
        re.escape(cue.strip()) for cue in NEGATION_CUES if cue.strip().isascii())
    + r")(?![a-z0-9])")
# 后置否定必须**紧贴概念本身**：可选逗号/冒号 + 可选系动词 + 否定谓语。
# 这样 "Edema is present and enhancement is absent" 不会把 edema 判成否定。
_POST_NEGATION_ANCHOR = re.compile(
    r"^\s*(?:[,:，：]\s*)?"
    r"(?:(?:is|are|was|were|has|have|had|been)\s+)?"
    r"(?:"
    r"absent\b|negative\b|"
    r"not\s+(?:seen|observed|identified|detected|present|evident|found|demonstrated)\b"
    r")", re.IGNORECASE)
# 并列主语共享后置谓语："edema and hemorrhage are absent"
_POST_COORD_ANCHOR = re.compile(
    r"^\s*(?:[,;，；]|and|or)\s+[A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,4}?\s+"
    r"(?:(?:is|are|was|were|has|have|had|been)\s+)?"
    r"(?:absent\b|negative\b|"
    r"not\s+(?:seen|observed|identified|detected|present|evident|found|demonstrated)\b)",
    re.IGNORECASE)
# 中文后置否定（无系动词，紧贴概念）
_POST_CN_ANCHOR = re.compile(r"^\s*[，,：:]?\s*(?:阴性|未见|未发现|未显示|消失)")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _SENTENCE_RE.finditer(text)]


def _negation_window(prefix: str) -> str:
    """前置线索只看最后 80 个字符，并在对比连词处重置。"""
    lowered = prefix.lower()
    for marker in _CONTRASTIVE:
        index = lowered.rfind(marker)
        if index >= 0:
            lowered = lowered[index + len(marker):]
            break
    return lowered[-80:]


def _clause_suffix(suffix: str) -> str:
    """后置否定只看当前分句：在下一个逗号/分号/冒号/对比连词处截断。"""
    match = _CLAUSE_END_RE.search(suffix)
    return suffix[:match.start()] if match else suffix


def _is_negated(prefix: str, suffix: str) -> bool:
    window = _negation_window(prefix)
    if _PRE_NEGATION_RE.search(window):
        return True
    for cue in NEGATION_CUES:
        if not cue.strip().isascii() and cue in window:
            return True
    tail = _clause_suffix(suffix)[:80]
    if _POST_NEGATION_ANCHOR.match(tail):
        return True
    if _POST_COORD_ANCHOR.match(tail):
        return True
    if _POST_CN_ANCHOR.match(tail):
        return True
    return False


@dataclass
class Ontology:
    name: str
    concepts: dict[str, list[str]] = field(default_factory=dict)
    language: str = "en"
    version: str = ""
    _alias_map: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.concepts:
            raise ConfigError(f"ontology {self.name!r} 为空")
        alias_map: dict[str, str] = {}
        for concept_id, aliases in self.concepts.items():
            if not concept_id or not aliases:
                raise ConfigError(f"概念 {concept_id!r} 缺少 aliases")
            for alias in aliases:
                key = str(alias).strip().lower()
                if not key:
                    continue
                owner = alias_map.get(key)
                if owner is not None and owner != concept_id:
                    raise ConfigError(f"alias {alias!r} 同时属于 {owner} 与 {concept_id}")
                alias_map[key] = concept_id
        self._alias_map = alias_map
        if not self.version:
            self.version = f"{self.name}:{self.sha256[:12]}"

    @property
    def concept_ids(self) -> list[str]:
        return list(self.concepts)

    @property
    def sha256(self) -> str:
        return sha256_json({"name": self.name, "concepts": self.concepts})

    def match(self, text: str, include_negated: bool = False) -> list[str]:
        """最长 alias 优先的确定性匹配；简单否定规则避免 ``no edema`` 被算作 edema。"""
        cleaned = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
        candidates: list[tuple[int, int, str]] = []
        for alias, concept_id in self._alias_map.items():
            pattern = re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])")
            for match in pattern.finditer(cleaned):
                candidates.append((match.start(), match.end(), concept_id))
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
        spans = _sentence_spans(cleaned)
        starts = [span[0] for span in spans]
        occupied: list[tuple[int, int]] = []
        found: list[str] = []
        for start, end, concept_id in candidates:
            if any(start < e and s < end for s, e in occupied):
                continue
            index = bisect.bisect_right(starts, start) - 1
            if index < 0:
                sentence_start, sentence_end = 0, len(cleaned)
            else:
                sentence_start, sentence_end = spans[index]
            prefix = cleaned[max(sentence_start, start - 80):start]
            suffix = cleaned[end:min(sentence_end, end + 80)]
            if not include_negated and _is_negated(prefix, suffix):
                occupied.append((start, end))   # 否定提及占用区间，避免重复计数
                continue
            occupied.append((start, end))
            if concept_id not in found:
                found.append(concept_id)
        order = {concept_id: index for index, concept_id in enumerate(self.concepts)}
        return sorted(found, key=lambda concept_id: order[concept_id])

    def vector(self, text: str) -> list[float]:
        positives = set(self.match(text))
        return [1.0 if concept_id in positives else 0.0
                for concept_id in self.concepts]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "concepts": self.concepts,
                "language": self.language, "version": self.version,
                "sha256": self.sha256}

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "Ontology":
        concepts_raw = raw.get("concepts")
        if not concepts_raw:
            raise ConfigError("ontology 配置缺少 concepts")
        concepts: dict[str, list[str]] = {}
        if isinstance(concepts_raw, Mapping):
            for concept_id, aliases in concepts_raw.items():
                concepts[str(concept_id)] = [str(a) for a in aliases]
        else:
            for item in concepts_raw:
                concepts[str(item["id"])] = [str(a) for a in item.get("aliases", [])]
        return cls(name=str(raw.get("name", "ontology")), concepts=concepts,
                   language=str(raw.get("language", "en")),
                   version=str(raw.get("version", "")))
