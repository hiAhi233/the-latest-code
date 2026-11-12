"""manifest 完整性与患者独立 split。"""

from __future__ import annotations

import json

import pytest

from medrrg.data.manifest import load_manifest, write_manifest
from medrrg.errors import DataError, LeakageError
from tests.helpers import make_study, make_studies


def test_manifest_roundtrip(tmp_path, studies):
    path = tmp_path / "manifest.jsonl"
    written = write_manifest(studies, path)
    loaded = load_manifest(path)
    assert loaded.sha256 == written.sha256
    assert len(loaded) == len(studies)
    assert loaded.split("train")
    assert loaded.get(studies[0].study_id).report.findings


def test_manifest_detects_tampering(tmp_path, studies):
    path = tmp_path / "manifest.jsonl"
    write_manifest(studies, path)
    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["report"]["findings"] = "tampered"
    rows[0] = json.dumps(first, ensure_ascii=False)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(DataError):
        load_manifest(path)


def test_patient_leakage_is_rejected():
    first = make_study("study-001-000", "train", "a lesion", "glioma")
    second = make_study("study-001-001", "test", "a lesion", "glioma")
    assert first.patient_id == second.patient_id == "study-001"
    path_holder = {}
    from pathlib import Path

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "leak.jsonl"
        path_holder["path"] = path
        manifest = write_manifest([first, second], path)
    with pytest.raises(LeakageError):
        manifest.verify_patients()
