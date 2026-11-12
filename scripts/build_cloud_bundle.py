"""Create a verified ZIP with project, full GLI data and real predicted evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CODE_DIRS = ("configs", "docs", "external_tools", "medrrg", "scripts", "tests")
CACHE_DIRS = ("caches/adm/localization/run1", "caches/adm/unimed/run1",
              "caches/adm/resnet/run_b0_original", "caches/adm/resnet/run_b1_adm")
TOP_FILES = ("README.md", "CLOUD_README.md", "pyproject.toml", "requirements-cloud.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--b0-evidence-config", help="Source config for original-slice visual features")
    parser.add_argument("--f0-evidence-config", help="Source config for ADM visual and semantic features")
    parser.add_argument("--experiment-record", type=Path,
                        help="Include localization experiment JSON records and selected checkpoint")
    args = parser.parse_args()
    if bool(args.b0_evidence_config) != bool(args.f0_evidence_config):
        parser.error("Provide both evidence configs when changing the packaged cache")
    if args.experiment_record and not args.f0_evidence_config:
        parser.error("--experiment-record requires the evidence configs")
    provenance = json.loads((args.dataset / "metadata/download_manifest.json").read_text(encoding="utf-8"))
    if provenance["case_count"] != 230:
        raise RuntimeError("Expected the complete 230-case GLI subset")
    for row in provenance["image_files"] + provenance["report_files"]:
        path = args.dataset / row["local_path"]
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise RuntimeError(f"Dataset file missing or size mismatch: {path}")
    cache_sources = {name: ROOT / name for name in CACHE_DIRS}
    if args.f0_evidence_config:
        from medrrg.config import load_config
        from medrrg.data.manifest import load_manifest
        from medrrg.evidence import load_sources
        b0 = load_config(args.b0_evidence_config)
        f0 = load_config(args.f0_evidence_config)
        manifest = load_manifest(f0.path("manifest"))
        manifest.verify_patients()
        if load_manifest(b0.path("manifest")).sha256 != manifest.sha256:
            raise RuntimeError("B0 and F0 use different manifests")
        if any(cfg.path("dataset.root").resolve() != args.dataset.resolve() for cfg in (b0, f0)):
            raise RuntimeError("Evidence configs and packaged dataset differ")
        original = load_sources(b0, manifest, use_semantic=False)
        complete = load_sources(f0, manifest, use_semantic=True)
        if original.localization.identity != complete.localization.identity:
            raise RuntimeError("B0 and F0 must use the same localization cache")
        cache_sources = dict(zip(CACHE_DIRS, (complete.localization.root, complete.semantic.root,
                                             original.visual.root, complete.visual.root)))
    from medrrg.features.adm_cache import AdmVisualCache
    for cache in CACHE_DIRS[2:]:
        visual = AdmVisualCache.load(cache_sources[cache])
        visual.verify(visual.index.keys(), d_visual=256)
        if len(visual) != 230:
            raise RuntimeError(f"Incomplete visual cache: {cache}")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for directory in CODE_DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append((path, path.relative_to(ROOT)))
    for destination, source in cache_sources.items():
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append((path, Path(destination) / path.relative_to(source)))
    if args.experiment_record:
        for name in ("protocol.json", "training.json", "validation_metrics.json", "test_metrics.json"):
            path = args.experiment_record / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append((path, Path("localization_adaptation") / name))
        files.append((f0.path("localization.checkpoint"),
                      Path("localization_adaptation/checkpoint_selected.pt")))
    for name in TOP_FILES:
        files.append((ROOT / name, Path(name)))
    # Keep all images, masks, reports and provenance/verification metadata.
    for path in sorted(args.dataset.rglob("*")):
        if path.is_file():
            files.append((path, Path("data/RadGenome-Brain_MRI/BraTS_GLI") / path.relative_to(args.dataset)))
    records = []
    total = sum(path.stat().st_size for path, _ in files)
    print(f"Packaging {len(files)} files, {total/1e9:.3f} GB", flush=True)
    with zipfile.ZipFile(args.out, "x", allowZip64=True) as archive:
        for i, (path, relative) in enumerate(files):
            digest = hashlib.sha256()
            compression = zipfile.ZIP_STORED if path.suffix in {".gz", ".npz", ".pth", ".pt"} else zipfile.ZIP_DEFLATED
            name = "MedADM_RRG/" + relative.as_posix()
            info = zipfile.ZipInfo(name)
            info.compress_type = compression
            info.external_attr = 0o644 << 16
            with path.open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                for block in iter(lambda: src.read(1024*1024), b""):
                    digest.update(block)
                    dst.write(block)
            records.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                            "sha256": digest.hexdigest()})
            if i % 200 == 0:
                print(f"[pack] {i}/{len(files)}", flush=True)
        manifest = {"dataset": "RadGenome-Brain_MRI GLI subset only", "cases": 230,
                    "includes": "code, four MRI sequences + segmentation, reports, predicted localization/semantic and v2 visual caches",
                    "upstream_weights_included": args.experiment_record is not None,
                    "upstream_backbone_weights_included": False,
                    "localization_adaptation_record_included": args.experiment_record is not None,
                    "files": records}
        archive.writestr("MedADM_RRG/BUNDLE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(args.out) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Archive CRC failed: {bad}")
    digest = hashlib.sha256()
    with args.out.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    checksum = args.out.with_suffix(args.out.suffix + ".sha256")
    checksum.write_text(f"{digest.hexdigest()}  {args.out.name}\n", encoding="utf-8")
    print(f"DONE {args.out} ({args.out.stat().st_size/1e9:.3f} GB) SHA256 {digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
