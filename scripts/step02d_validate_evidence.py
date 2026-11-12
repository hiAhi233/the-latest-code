"""step02d：校验三类 ADM 证据缓存与身份链，可选评价上游质量。

    & $py scripts\\step02d_validate_evidence.py --config configs\\adm_radgenome_f0.yaml
    & $py scripts\\step02d_validate_evidence.py --config ... --evaluate --splits val,test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.config import load_config, project_root  # noqa: E402
from medrrg.data.adapters import build_adapter  # noqa: E402
from medrrg.data.manifest import load_manifest  # noqa: E402
from medrrg.evaluation.upstream import (localization_metrics,  # noqa: E402
                                        semantic_metrics)
from medrrg.evidence import load_sources  # noqa: E402
from medrrg.utils import read_json, sha256_json, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluate", action="store_true",
                        help="同时计算定位与语义上游指标")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = load_manifest(config.path("manifest"))
    adapter = build_adapter(config)
    use_semantic = bool(config.get_path("model.use_semantic", True))
    sources = load_sources(config, manifest, use_semantic=use_semantic,
                           require_predicted=not bool(config.get_path("adm.oracle_mask",
                                                                     False)))
    studies = [study.study_id for study in manifest]
    visual = sources.visual
    expected_tokens = int(visual.tokens_per_study)
    configured_tokens = int(config.get_path("model.visual_tokens", expected_tokens))
    if expected_tokens != configured_tokens:
        raise SystemExit(
            f"visual tokens 数 {expected_tokens} != model.visual_tokens {configured_tokens}")
    report: dict = {
        "manifest": {"path": manifest.path, "sha256": manifest.sha256,
                     "counts": manifest.counts(), "num_patients":
                     len({study.patient_id for study in manifest})},
        "caches": {
            "localization": {"dir": sources.localization_dir,
                             "identity": sources.localization.identity,
                             "num_studies": len(sources.localization),
                             "source": sources.localization.meta.get("source"),
                             "checkpoint_sha256":
                             sources.localization.meta.get("checkpoint_sha256")},
            "visual": {"dir": sources.visual_dir, "identity": visual.identity,
                       "num_studies": len(visual), "d_visual": visual.d_visual,
                       "tokens_per_study": expected_tokens,
                       "resnet_weights_sha256": visual.meta.get("resnet_weights_sha256"),
                       "adm": visual.meta.get("adm"),
                       "sampling": visual.meta.get("sampling")},
        },
        "checks": {
            "manifest_matches_localization": len(sources.localization) == len(studies),
            "manifest_matches_visual": len(visual) == len(studies),
            "visual_bound_to_localization":
                visual.localization_identity == sources.localization.identity,
            "tokens_per_study": expected_tokens,
            "identity_chain_sha256": sha256_json(sources.cache_identity),
        },
    }
    if use_semantic:
        semantic = sources.semantic
        report["caches"]["semantic"] = {
            "dir": sources.semantic_dir,
            "identity": semantic.identity,
            "num_studies": len(semantic),
            "d_proto": semantic.d_proto,
            "concept_ids": semantic.concept_ids,
            "classification_mode": semantic.meta.get("classification_mode"),
            "prompt_bank_sha256": semantic.meta.get("prompt_bank_sha256"),
        }
        report["checks"]["manifest_matches_semantic"] = len(semantic) == len(studies)
        report["checks"]["semantic_bound_to_localization"] = \
            semantic.localization_identity == sources.localization.identity
        report["checks"]["semantic_tokens"] = len(semantic.concept_ids)
    if args.evaluate:
        splits = tuple(args.splits.split(","))
        prompts = {}
        prompt_path = config.get_path("semantic.prompts", "")
        if prompt_path:
            path = Path(str(prompt_path))
            if not path.is_absolute():
                path = project_root() / path
            if path.is_file():
                prompts = read_json(path)
        report["upstream"] = {
            "localization": localization_metrics(
                manifest, adapter, sources.localization, splits=splits,
                limit=args.limit),
            "semantic": (semantic_metrics(
                manifest, adapter.ontology, sources.semantic, splits=splits,
                exclusive_groups=prompts.get("exclusive_groups"),
                limit=args.limit) if sources.semantic is not None else {}),
        }
    out = project_root() / "caches" / "adm" / "evidence_validation.json"
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(f"[validate] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
