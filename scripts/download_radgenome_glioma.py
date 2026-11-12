"""Download only RadGenome's report-matched BraTS GLI cases, with SHA256 checks.

Public image files come from a pinned MedOtter mirror; reports come from the
original RadGenome authors. No Hugging Face credentials are read or required.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path
import threading
import time
from urllib.parse import quote

import requests

IMAGE_REPO = "MedOtter/brats2023-gli-dataset"
IMAGE_REVISION = "b032d353a3e80911a5f850bc54e6fb575298a354"
REPORT_REPO = "JiayuLei/RadGenome-Brain_MRI"
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers.update({"User-Agent": "RadGenomeGLI-research-download/1.0", "Accept-Encoding": "identity"})
    return LOCAL.session


def fetch_json(url):
    response = session().get(url, timeout=(20, 90))
    response.raise_for_status()
    return response.json()


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(row, root):
    target = root / row["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size, expected_hash = row["size"], row["sha256"]
    if target.exists():
        if target.stat().st_size == expected_size and sha256(target) == expected_hash:
            return {**row, "status": "verified_existing"}
        raise RuntimeError(f"Existing destination fails checksum; preserved: {target}")
    partial = target.with_name(target.name + ".part")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                raise RuntimeError(f"Partial file is larger than expected: {partial}")
            if offset < expected_size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with session().get(row["url"], headers=headers, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("Unexpected Content-Range")
                        mode = "ab"
                    else:
                        mode = "wb"
                    with partial.open(mode) as output:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                output.write(block)
            if partial.stat().st_size != expected_size:
                raise RuntimeError(f"Size mismatch: {partial.name}")
            if sha256(partial) != expected_hash:
                # Keep corrupt bytes for inspection and start a fresh partial.
                partial.rename(partial.with_name(partial.name + f".bad-{time.time_ns()}"))
                raise RuntimeError(f"SHA256 mismatch: {partial.name}")
            os.replace(partial, target)
            return {**row, "status": "downloaded_verified"}
        except Exception as exc:
            if attempt == 4:
                raise RuntimeError(f"{row['local_path']}: {type(exc).__name__}: {exc}") from exc
            time.sleep(min(2 ** attempt, 12))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_info = fetch_json(f"https://huggingface.co/api/datasets/{REPORT_REPO}")
    report_revision = report_info["sha"]
    annotation_files = ["BraTS_GLI/global_finding.json", "BraTS_GLI/modal_wise_finding.json", "BraTS_GLI/impression.json", "train_val_test_split.json"]
    loaded, report_checksums = {}, []
    for name in annotation_files:
        url = f"https://huggingface.co/datasets/{REPORT_REPO}/resolve/{report_revision}/{name}"
        response = session().get(url, timeout=(20, 60))
        response.raise_for_status()
        raw = response.content
        loaded[name] = json.loads(raw)
        target = root / "reports" / Path(name).name
        if target.exists() and target.read_bytes() != raw:
            raise RuntimeError(f"Existing annotation differs; preserved: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        report_checksums.append({"local_path": target.relative_to(root).as_posix(), "source_path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "url": url})
    reports = loaded["BraTS_GLI/global_finding.json"]
    splits = loaded["train_val_test_split.json"]
    split_cases = {s: list(dict.fromkeys(x.rsplit("-", 1)[0] for x in entries if x.startswith("BraTS-GLI-"))) for s, entries in splits.items()}
    pilot = {s: cases[:2] for s, cases in split_cases.items()}
    dump_json(root / "metadata" / "pilot_cases.json", pilot)
    dump_json(root / "metadata" / "glioma_case_splits.json", split_cases)
    info = fetch_json(f"https://huggingface.co/api/datasets/{IMAGE_REPO}/revision/{IMAGE_REVISION}?blobs=true")
    pilot_cases = {x for values in pilot.values() for x in values}
    split_by_case = {x: s for s, values in split_cases.items() for x in values}
    manifest = []
    for item in info["siblings"]:
        path = item["rfilename"]
        case = next((part for part in path.split("/") if part in reports), None)
        if not case or not path.endswith(".nii.gz"):
            continue
        lfs = item.get("lfs") or {}
        if not lfs.get("sha256"):
            raise RuntimeError(f"No SHA256 in upstream manifest: {path}")
        manifest.append({"case_id": case, "subject_id_prefix": case.rsplit("-", 1)[0], "split": split_by_case[case], "local_path": f"images/{case}/{Path(path).name}", "source_path": path, "size": item["size"], "sha256": lfs["sha256"], "url": f"https://huggingface.co/datasets/{IMAGE_REPO}/resolve/{IMAGE_REVISION}/{quote(path, safe='/')}"})
    counts = {c: sum(row["case_id"] == c for row in manifest) for c in reports}
    if len(reports) != 230 or len(manifest) != 1150 or set(counts.values()) != {5}:
        raise RuntimeError(f"Unexpected image coverage: cases={len(reports)}, files={len(manifest)}")
    manifest.sort(key=lambda x: (x["case_id"] not in pilot_cases, x["case_id"], x["local_path"]))
    provenance = {"image_repo": IMAGE_REPO, "image_revision": IMAGE_REVISION, "report_repo": REPORT_REPO, "report_revision": report_revision, "case_count": len(reports), "expected_image_files": len(manifest), "expected_image_bytes": sum(x["size"] for x in manifest), "report_files": report_checksums, "image_files": manifest, "official_image_source": "https://www.synapse.org/Synapse:syn51514105", "scope": "report-matched glioma cases only; four MRI sequences and segmentation mask per case"}
    dump_json(root / "metadata" / "download_manifest.json", provenance)
    print(json.dumps({"event": "download_start", "root": str(root), "cases": len(reports), "files": len(manifest), "GB": provenance["expected_image_bytes"] / 1e9, "pilot_cases_first": pilot}, ensure_ascii=False), flush=True)
    start, last_update, done, verified_bytes = time.monotonic(), 0.0, 0, 0
    failures = []
    with (root / "metadata" / "verified_files.jsonl").open("a", encoding="utf-8") as log:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download, row, root): row for row in manifest}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                    log.write(json.dumps({k: v for k, v in result.items() if k != "url"}, ensure_ascii=False) + "\n")
                    log.flush()
                    done += 1
                    verified_bytes += row["size"]
                except Exception as exc:
                    failures.append({"local_path": row["local_path"], "error": str(exc)})
                    print(json.dumps({"event": "download_error", **failures[-1]}, ensure_ascii=False), flush=True)
                now = time.monotonic()
                if now - last_update >= 10 or done == len(manifest):
                    state = {"status": "downloading", "verified_files": done, "total_files": len(manifest), "verified_GB": round(verified_bytes / 1e9, 4), "total_GB": round(provenance["expected_image_bytes"] / 1e9, 4), "errors": len(failures), "elapsed_seconds": round(now - start, 1)}
                    dump_json(root / "metadata" / "download_status.json", state)
                    print(json.dumps(state, ensure_ascii=False), flush=True)
                    last_update = now
    state = {"status": "complete" if not failures else "incomplete", "verified_files": done, "total_files": len(manifest), "verified_image_bytes": verified_bytes, "case_count": len(reports), "elapsed_seconds": round(time.monotonic() - start, 1), "failures": failures}
    dump_json(root / "metadata" / "download_status.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
