"""step05：评价报告（NLG / 概念 / 幻觉 / 分段完整率 / null-swap 干预）。

    & $py scripts\\step05_evaluate.py --run results\\adm_f0\\test --interventions --concept-deltas
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.cli import main as cli_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli_main(["evaluate"] + sys.argv[1:]))
