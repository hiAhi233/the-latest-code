"""pytest 共享夹具：合成 manifest + 三类 ADM 缓存。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.data.adm_dataset import WordTokenizer  # noqa: E402
from medrrg.data.manifest import write_manifest  # noqa: E402
from tests.helpers import make_studies, write_localization_cache  # noqa: E402
from tests.helpers import write_semantic_cache, write_visual_cache  # noqa: E402


@pytest.fixture(scope="session")
def studies():
    return make_studies(4)


@pytest.fixture(scope="session")
def manifest(tmp_path_factory, studies):
    path = tmp_path_factory.mktemp("manifest") / "synthetic.jsonl"
    return write_manifest(studies, path)


@pytest.fixture(scope="session")
def tokenizer(studies):
    texts = [study.report.target_text(control_tokens=True) for study in studies]
    return WordTokenizer.build(texts, max_len=64)


@pytest.fixture(scope="session")
def evidence_dirs(tmp_path_factory, studies):
    root = tmp_path_factory.mktemp("caches")
    localization_dir = root / "localization" / "run1"
    visual_dir = root / "visual" / "run1"
    semantic_dir = root / "semantic" / "run1"
    loc_meta = write_localization_cache(localization_dir, studies)
    write_visual_cache(visual_dir, studies, loc_meta["identity"])
    write_semantic_cache(semantic_dir, studies, loc_meta["identity"])
    return {"root": root, "localization": localization_dir,
            "visual": visual_dir, "semantic": semantic_dir,
            "localization_meta": loc_meta}
