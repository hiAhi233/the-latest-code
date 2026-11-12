"""step03：训练 B0/B1/F0。

    & $py scripts\\step03_train.py --config configs\\adm_radgenome_b0.yaml
    & $py scripts\\step03_train.py --config configs\\adm_radgenome_f0.yaml --overfit-cases 8
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.cli import main as cli_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli_main(["train"] + sys.argv[1:]))
