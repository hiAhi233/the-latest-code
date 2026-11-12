"""导出脚本的缓存复用守卫：格式版本 + v2 必需字段。"""

from __future__ import annotations

import numpy as np
import pytest

from medrrg.features.adm_cache import (SEMANTIC_ARRAYS, VISUAL_ARRAYS,
                                       VISUAL_FORMAT, missing_npz_keys)
from external_tools.export_resnet_visual import (validate_existing_file,
                                                 validate_existing_meta)


def test_v1_meta_is_rejected_and_mentions_format():
    previous = {"format": "medadm.adm.resnet.v1", "d_visual": 256}
    invariants = {"format": VISUAL_FORMAT, "d_visual": 256}
    with pytest.raises(RuntimeError) as error:
        validate_existing_meta(previous, invariants)
    message = str(error.value)
    assert "v1" in message and "v2" in message
    assert "--force" in message


def test_matching_meta_passes():
    meta = {"format": VISUAL_FORMAT, "d_visual": 256, "adm": {"gamma": 0.4}}
    validate_existing_meta(meta, {"format": VISUAL_FORMAT, "d_visual": 256,
                                 "adm": {"gamma": 0.4}})


def test_parameter_change_is_rejected():
    meta = {"format": VISUAL_FORMAT, "adm": {"gamma": 0.4}}
    with pytest.raises(RuntimeError, match="不一致"):
        validate_existing_meta(meta, {"format": VISUAL_FORMAT,
                                      "adm": {"gamma": 0.3}})


def test_old_npz_missing_patch_positions_is_rejected(tmp_path):
    path = tmp_path / "old.npz"
    np.savez_compressed(path, visual_tokens=np.zeros((2, 4), dtype=np.float32),
                        visual_mask=np.ones(2, dtype=bool),
                        visual_modality=np.zeros(2, dtype=np.int64),
                        visual_z=np.zeros(2, dtype=np.float32))
    missing = missing_npz_keys(path, VISUAL_ARRAYS)
    assert "visual_patch_row" in missing and "visual_patch_col" in missing
    with pytest.raises(RuntimeError, match="缺少字段"):
        validate_existing_file(path, "study-x")


def test_complete_visual_npz_passes(tmp_path):
    path = tmp_path / "new.npz"
    np.savez_compressed(
        path,
        visual_tokens=np.zeros((2, 4), dtype=np.float32),
        visual_tokens_null=np.zeros((2, 4), dtype=np.float32),
        visual_mask=np.ones(2, dtype=bool),
        visual_modality=np.zeros(2, dtype=np.int64),
        visual_z=np.zeros(2, dtype=np.float32),
        visual_view=np.zeros(2, dtype=np.int64),
        visual_patch_row=np.zeros(2, dtype=np.int64),
        visual_patch_col=np.ones(2, dtype=np.int64))
    assert missing_npz_keys(path, VISUAL_ARRAYS) == []
    validate_existing_file(path, "study-x")


def test_semantic_arrays_list_is_complete():
    assert "text_prototypes" in SEMANTIC_ARRAYS
    assert "concept_ids" in SEMANTIC_ARRAYS
