"""step04：生成报告。

    & $py scripts\\step04_generate.py --checkpoint checkpoints\\adm_f0_resnet_unimed\\best.pt --split test --max-len 256 --out results\\adm_f0\\test
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.cli import main as cli_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli_main(["generate"] + sys.argv[1:]))
