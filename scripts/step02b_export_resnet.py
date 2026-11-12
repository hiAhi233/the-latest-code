"""step02b：ADM 增强 + ResNet-101 导出视觉 patch tokens（需要 CUDA）。

    & $py scripts\\step02b_export_resnet.py --config configs\\adm_radgenome_f0.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from external_tools.export_resnet_visual import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
