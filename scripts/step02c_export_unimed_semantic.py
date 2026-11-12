"""step02c：UniMed-CLIP 零样本疾病语义导出（需要 CUDA + 2.35GB 权重）。

    & $py scripts\\step02c_export_unimed_semantic.py --config configs\\adm_radgenome_f0.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from external_tools.export_unimed_semantic import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
