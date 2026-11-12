"""Shared offline report evaluation: vendored COCO scorers + Java METEOR 1.5.

This module also runs standalone, with no model/checkpoint/torch dependency.
Never substitutes METEOR-lite when Java or the scorer resources are unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

TOKENIZER_VERSION = "ekagen_regex_symmetric_v1"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\w\s]")
TAG_RE = re.compile(r"</?\s*(?:FINDINGS|IMPRESSION|BOS|EOS|PAD)\s*>", re.I)


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(TAG_RE.sub(" ", text).lower()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path, section: str = "findings") -> dict[str, dict]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty list of reports: {path}")
    result = {}
    for row in rows:
        case_id = row.get("study_id") or row.get("id")
        if not case_id or case_id in result:
            raise ValueError(f"Missing or duplicate case ID in {path}: {case_id}")
        if section == "findings":
            if "study_id" in row:
                prediction, reference = row.get("findings"), row.get("reference_findings")
            else:
                prediction, reference = row.get("prediction"), row.get("reference")
        else:
            if "study_id" not in row:
                raise ValueError("EKAGen MRI records only contain Findings; cannot compare full reports")
            prediction, reference = row.get("prediction"), row.get("reference")
        if not isinstance(prediction, str) or not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Missing report fields for {case_id}")
        result[case_id] = {"prediction": normalize(prediction), "reference": normalize(reference)}
    return result


def align_records(models: dict[str, dict]) -> list[str]:
    first = next(iter(models.values()))
    case_ids = sorted(first)
    for name, records in models.items():
        if set(records) != set(case_ids):
            raise ValueError(f"Case ID set mismatch: {name}")
        for case_id in case_ids:
            if records[case_id]["reference"] != first[case_id]["reference"]:
                raise ValueError(f"Reference mismatch: {name}, {case_id}")
    return case_ids


def import_scorers(root: Path):
    root = root.resolve()
    if root.name != "pycocoevalcap":
        root = root / "pycocoevalcap"
    if not root.is_dir():
        raise FileNotFoundError(f"pycocoevalcap not found: {root}")
    sys.path.insert(0, str(root.parent))
    modules = [importlib.import_module(f"pycocoevalcap.{name}.{name}")
               for name in ("bleu", "rouge")]
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("A different pycocoevalcap checkout was already imported")
    return root, modules[0].Bleu, modules[1].Rouge


def meteor_score(hypotheses: list[str], references: list[str], root: Path,
                 output: Path, java: str | None = None) -> float:
    executable = java or shutil.which("java")
    if not executable:
        raise RuntimeError("Java is required for METEOR 1.5; no METEOR-lite fallback is allowed")
    jar, data = root / "meteor/meteor-1.5.jar", root / "meteor/data"
    if not jar.is_file() or not (data / "paraphrase-en.gz").is_file():
        raise FileNotFoundError("METEOR jar or English paraphrase data is missing")
    with tempfile.TemporaryDirectory(prefix="medadm_meteor_") as folder:
        work = Path(folder)
        shutil.copy2(jar, work / "meteor.jar")
        shutil.copytree(data, work / "data")
        (work / "hyp.txt").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
        (work / "ref.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [executable, "-Xmx1G", "-jar", "meteor.jar", "hyp.txt", "ref.txt", "-l", "en", "-norm"],
            cwd=work, capture_output=True, text=True, timeout=300)
    output.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    completed.check_returncode()
    match = re.search(r"Final score:\s*([0-9.eE+-]+)", completed.stdout)
    if not match:
        raise RuntimeError(f"METEOR did not return a final score; see {output}")
    return float(match.group(1))


def evaluate_models(inputs: dict[str, Path], scorer_root: Path, output: Path,
                    section: str = "findings", java: str | None = None) -> dict:
    models = {name: load_records(path, section) for name, path in inputs.items()}
    ids = align_records(models)
    root, Bleu, Rouge = import_scorers(scorer_root)
    if output.exists():
        raise FileExistsError(f"Use a new output directory; preserving existing results: {output}")
    output.mkdir(parents=True)
    first = next(iter(models.values()))
    refs = {i: [first[case_id]["reference"]] for i, case_id in enumerate(ids)}
    reference_sha = hashlib.sha256(json.dumps(
        {case_id: first[case_id]["reference"] for case_id in ids},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    files = [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    files.extend([root / "meteor/meteor-1.5.jar", root / "meteor/data/paraphrase-en.gz"])
    resource_hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(files)}
    result = {"section": section, "cases": len(ids), "case_ids": ids,
              "tokenizer": TOKENIZER_VERSION, "reference_sha256": reference_sha,
              "scorer_root": str(root), "scorer_sha256": resource_hashes,
              "protocol": "COCO Bleu(4), Rouge, METEOR-1.5 Java CLI -l en -norm; "
                          "same normalized references for all models",
              "models": {}}
    for name, records in models.items():
        hypotheses = [records[case_id]["prediction"] for case_id in ids]
        hyps = {i: [text] for i, text in enumerate(hypotheses)}
        bleu, _ = Bleu(4).compute_score(refs, hyps, verbose=0)
        rouge, _ = Rouge().compute_score(refs, hyps)
        meteor = meteor_score(hypotheses, [refs[i][0] for i in refs], root,
                              output / f"{name}_meteor.log", java)
        metrics = {f"BLEU_{i+1}": float(value) for i, value in enumerate(bleu)}
        metrics.update(METEOR=meteor, ROUGE_L=float(rouge))
        result["models"][name] = {"metrics": metrics, "input": str(inputs[name]),
                                  "input_sha256": sha256_file(inputs[name])}
        (output / f"{name}_normalized.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, json.dumps(metrics), flush=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    headings = ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L"]
    lines = [f"# Unified report evaluation ({len(ids)} cases, {section})", "",
             "| Model | " + " | ".join(headings) + " |",
             "|---|" + "---:|" * len(headings)]
    for name, value in result["models"].items():
        lines.append("| " + name + " | " + " | ".join(f"{value['metrics'][key]:.6f}" for key in headings) + " |")
    lines += ["", result["protocol"], "", f"Tokenizer: {TOKENIZER_VERSION}",
              "", f"Reference SHA256: {reference_sha}"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="NAME=reports.json/jsonl (repeatable)")
    parser.add_argument("--scorer-root", required=True, type=Path, help="EKAGen/original or pycocoevalcap path")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--section", choices=["findings", "full"], default="findings")
    parser.add_argument("--java", default=None)
    args = parser.parse_args()
    inputs = {}
    for value in args.input:
        name, separator, path = value.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in inputs:
            parser.error("Each input needs a unique filesystem-safe NAME=PATH")
        inputs[name] = Path(path)
    evaluate_models(inputs, args.scorer_root, args.out, args.section, args.java)


if __name__ == "__main__":
    main()
