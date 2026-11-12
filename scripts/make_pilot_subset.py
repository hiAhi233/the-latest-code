"""从正式 manifest 抽取一个小型 pilot 子集，用于端到端链路自检。

    & $py scripts\\make_pilot_subset.py --manifest caches\\manifests\\radgenome_brain_mri.jsonl `
        --out caches\\manifests\\pilot.jsonl --train 2 --val 1 --test 1

只复制 manifest 行（影像、报告、split 不变），不复制影像数据；pilot 配置把
``evidence.*_dir`` 指向独立目录，避免污染正式缓存。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.data.manifest import load_manifest, write_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="caches/manifests/radgenome_brain_mri.jsonl")
    parser.add_argument("--out", default="caches/manifests/pilot.jsonl")
    parser.add_argument("--train", type=int, default=2)
    parser.add_argument("--val", type=int, default=1)
    parser.add_argument("--test", type=int, default=1)
    args = parser.parse_args()

    source = load_manifest(args.manifest)
    wanted = {"train": args.train, "val": args.val, "test": args.test}
    selected = []
    for split, count in wanted.items():
        studies = source.split(split)
        if len(studies) < count:
            raise SystemExit(f"{split} 只有 {len(studies)} 例，少于请求的 {count}")
        selected.extend(studies[:count])
    out = Path(args.out)
    manifest = write_manifest(selected, out, {
        "adapter": "pilot_subset",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": source.sha256,
        "note": "pilot 链路自检用；不是论文实验划分"})
    print(f"[pilot] {len(selected)} 例 -> {out} sha={manifest.sha256[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
