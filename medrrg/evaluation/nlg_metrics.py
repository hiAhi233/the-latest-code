"""文本指标（自包含实现）。

* BLEU-1..4：标准 brevity penalty + 无平滑 n-gram 精度；
* ROUGE-L：LCS 的 F1；
* **CIDEr-D**：按 Vedantam et al. (CVPR 2015) 的官方 `cider_scorer` 公式实现
  （IDF 取 ``log(N) - log(df)``，分子为 clipped tf-idf 内积、分母为两个范数之积，
  带长度高斯惩罚并乘 10）；不是自造的余弦版本；
* METEOR-lite：明确不是官方 METEOR。

评测前两边都去掉 ``<FINDINGS>/<IMPRESSION>`` 控制标签，保证同一文本口径。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

__all__ = ["tokenize_words", "corpus_bleu", "rouge_l", "cider_d", "meteor_lite",
           "repetition_ratio", "copy_rate", "compute_nlg_metrics", "strip_control_tags"]

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.*/-][a-z0-9]+)*")
_CONTROL_RE = re.compile(r"</?\s*(FINDINGS|IMPRESSION)\s*>", re.IGNORECASE)


def strip_control_tags(text: str) -> str:
    """评测前去掉章节控制标签，保证预测与参考同一文本口径。"""
    return re.sub(r"\s+", " ", _CONTROL_RE.sub(" ", text or "")).strip()


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str],
                max_n: int = 4) -> dict[str, float]:
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses/references 数量不一致")
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        h, r = tokenize_words(hyp), tokenize_words(ref)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            h_grams = _ngrams(h, n)
            r_grams = _ngrams(r, n)
            matches[n - 1] += sum(min(count, r_grams[gram])
                                  for gram, count in h_grams.items())
            totals[n - 1] += max(0, len(h) - n + 1)
    precisions = [matches[i] / totals[i] if totals[i] else 0.0
                  for i in range(max_n)]
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(1, hyp_len))
    out = {}
    for n in range(1, max_n + 1):
        values = precisions[:n]
        out[f"bleu_{n}"] = (0.0 if any(v <= 0 for v in values)
                            else bp * math.exp(sum(math.log(v) for v in values) / n))
    return out


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if token_a == token_b else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    scores = []
    for hyp, ref in zip(hypotheses, references):
        h, r = tokenize_words(hyp), tokenize_words(ref)
        lcs = _lcs(h, r)
        if not lcs:
            scores.append(0.0)
            continue
        precision, recall = lcs / len(h), lcs / len(r)
        scores.append(2 * precision * recall / (precision + recall))
    return float(sum(scores) / max(1, len(scores)))


def cider_d(hypotheses: Sequence[str], references: Sequence[str],
            sigma: float = 6.0, max_n: int = 4) -> float:
    """CIDEr-D（Vedantam et al., CVPR 2015）的官方评分公式。

    对每个 n∈[1,max_n] 与每个样本：

    * ``df(gram)`` = 包含该 n-gram 的参考文档数（文档总数为 ``N``）；
    * ``idf(gram) = log(N) - log(max(1, df))``（与官方 ``cider_scorer`` 一致，
      出现在所有参考中的 n-gram 权重为 0）；
    * ``v_h[gram] = tf_h(gram) * idf``，``v_r`` 同理；
    * ``sim_n = Σ min(v_h, v_r) · v_r / (||v_h|| · ||v_r||)``；
    * 长度惩罚 ``exp(-(len_h - len_r)² / (2σ²))``（len 为 unigram 数）；
    * 样本得分 ``= 10 · mean_n(sim_n · penalty) / num_references``。

    本工程每个样本只有一条参考，故 ``num_references = 1``。若整个参考集只有 1 个
    文档，``idf`` 全为 0、CIDEr-D 退化为 0（与官方实现一致），评测应使用完整测试集。
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses/references 数量不一致")
    documents = len(references)
    if documents == 0:
        return 0.0
    hyp_tokens = [tokenize_words(h) for h in hypotheses]
    ref_tokens = [tokenize_words(r) for r in references]

    document_frequency: defaultdict[tuple[str, ...], float] = defaultdict(float)
    for tokens in ref_tokens:
        seen: set[tuple[str, ...]] = set()
        for n in range(1, max_n + 1):
            seen.update(_ngrams(tokens, n))
        for gram in seen:
            document_frequency[gram] += 1.0
    log_documents = math.log(float(documents))

    def vector(tokens: Sequence[str]) -> tuple[list[defaultdict], list[float], int]:
        vec: list[defaultdict] = [defaultdict(float) for _ in range(max_n)]
        norm = [0.0] * max_n
        for n in range(1, max_n + 1):
            for gram, count in _ngrams(tokens, n).items():
                df = math.log(max(1.0, document_frequency.get(gram, 0.0)))
                weight = float(count) * (log_documents - df)
                vec[n - 1][gram] = weight
                norm[n - 1] += weight * weight
        length = sum(_ngrams(tokens, 1).values())
        return vec, [math.sqrt(value) for value in norm], length

    scores: list[float] = []
    for hyp, ref in zip(hyp_tokens, ref_tokens):
        vec_h, norm_h, length_h = vector(hyp)
        vec_r, norm_r, length_r = vector(ref)
        penalty = math.exp(-((length_h - length_r) ** 2) / (2.0 * sigma ** 2))
        per_n: list[float] = []
        for n in range(max_n):
            if norm_h[n] == 0.0 or norm_r[n] == 0.0:
                per_n.append(0.0)
                continue
            numerator = sum(min(value, vec_r[n].get(gram, 0.0)) * vec_r[n].get(gram, 0.0)
                            for gram, value in vec_h[n].items())
            per_n.append(numerator / (norm_h[n] * norm_r[n]) * penalty)
        scores.append(10.0 * sum(per_n) / max_n)
    return float(sum(scores) / len(scores))


def meteor_lite(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    scores = []
    for hyp, ref in zip(hypotheses, references):
        h, r = tokenize_words(hyp), tokenize_words(ref)
        if not h or not r:
            scores.append(0.0)
            continue
        used = [False] * len(r)
        matched, flags = 0, []
        for token in h:
            hit = False
            for j, candidate in enumerate(r):
                if used[j]:
                    continue
                if token == candidate or token[:4] == candidate[:4]:
                    used[j] = True
                    matched += 1
                    hit = True
                    break
            flags.append(hit)
        if matched == 0:
            scores.append(0.0)
            continue
        precision, recall = matched / len(h), matched / len(r)
        f_mean = 2 * precision * recall / (precision + recall)
        chunks = sum(1 for i, flag in enumerate(flags)
                     if flag and (i == 0 or not flags[i - 1]))
        scores.append(f_mean * (1 - 0.5 * (chunks / matched) ** 3))
    return float(sum(scores) / max(1, len(scores)))


def repetition_ratio(text: str, n: int = 4) -> float:
    tokens = tokenize_words(text)
    if len(tokens) < n:
        return 0.0
    grams = _ngrams(tokens, n)
    repeated = sum(count - 1 for count in grams.values() if count > 1)
    return float(repeated / max(1, len(tokens) - n + 1))


def copy_rate(hypotheses: Sequence[str], train_texts: Sequence[str],
              n: int = 5) -> float:
    train_grams = set()
    for text in train_texts:
        train_grams.update(_ngrams(tokenize_words(text), n))
    hits = total = 0
    for hyp in hypotheses:
        grams = _ngrams(tokenize_words(hyp), n)
        total += sum(grams.values())
        hits += sum(count for gram, count in grams.items() if gram in train_grams)
    return float(hits / max(1, total))


def compute_nlg_metrics(rows: Sequence[Mapping[str, Any]],
                        train_texts: Sequence[str] | None = None) -> dict[str, Any]:
    hypotheses = [strip_control_tags(str(row.get("prediction", ""))) for row in rows]
    references = [strip_control_tags(str(row.get("reference", ""))) for row in rows]
    if not hypotheses:
        return {"num_examples": 0}
    metrics = {"num_examples": len(hypotheses)}
    metrics.update({key: round(value, 6)
                    for key, value in corpus_bleu(hypotheses, references).items()})
    metrics.update({
        "rouge_l": round(rouge_l(hypotheses, references), 6),
        "cider_d": round(cider_d(hypotheses, references), 6),
        "meteor_lite": round(meteor_lite(hypotheses, references), 6),
        "mean_hypothesis_words": round(
            sum(len(tokenize_words(h)) for h in hypotheses) / len(hypotheses), 2),
        "mean_reference_words": round(
            sum(len(tokenize_words(r)) for r in references) / len(references), 2),
        "repetition_4gram": round(
            sum(repetition_ratio(h) for h in hypotheses) / len(hypotheses), 6),
        "empty_predictions": sum(1 for h in hypotheses if not h.strip()),
        "metric_notes": {
            "text_normalization": "control tokens stripped from both sides",
            "meteor": "meteor_lite, not official METEOR",
            "cider": "CIDEr-D per Vedantam et al. official cider_scorer formula "
                     "(idf over reference set, clipped tf-idf, length penalty, x10)",
        },
    })
    if train_texts:
        clean_train = [strip_control_tags(text) for text in train_texts]
        metrics["training_copy_rate_5gram"] = round(
            copy_rate(hypotheses, clean_train), 6)
    return metrics
