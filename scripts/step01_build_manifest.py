"""step01：构建并验证 manifest。

    & $py scripts\\step01_build_manifest.py --config configs\\adm_radgenome_f0.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.cli import main as cli_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli_main(["prepare"] + sys.argv[1:]))
