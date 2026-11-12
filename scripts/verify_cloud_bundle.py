"""Check extracted bundle against its SHA256 file manifest (standard library only)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for i, row in enumerate(manifest["files"]):
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            failures.append(row["path"] + ": missing or size mismatch")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            failures.append(row["path"] + ": SHA256 mismatch")
        if i % 300 == 0:
            print(f"[verify] {i}/{len(manifest['files'])}", flush=True)
    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Verified {len(manifest['files'])} files, dataset={manifest['cases']} cases")


if __name__ == "__main__":
    main()
