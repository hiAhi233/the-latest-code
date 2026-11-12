"""step02a：导出 ADM 定位热图缓存（真实少样本异常检测器，需要 CUDA）。

    & $py scripts\\step02a_export_adm.py --config configs\\adm_radgenome_f0.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from external_tools.export_adm_localization import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
