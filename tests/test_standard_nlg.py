import json

import pytest

from medrrg.evaluation.standard_nlg import align_records, load_records, normalize, meteor_score


def test_symmetric_tokenization():
    assert normalize("T1W: 51*70*41 mm.") == normalize("t 1 w : 51 * 70 * 41 mm .")
    assert normalize("<FINDINGS> Left frontal lobe. </FINDINGS>") == "left frontal lobe ."


def test_findings_only_never_uses_impression(tmp_path):
    path = tmp_path / "ours.jsonl"
    row = {"study_id": "a", "findings": "left frontal", "reference_findings": "right frontal",
           "prediction": "left frontal impression glioma", "reference": "right frontal glioma"}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = load_records(path)
    assert result["a"] == {"prediction": "left frontal", "reference": "right frontal"}


def test_case_order_is_aligned_but_mismatched_references_fail():
    a = {"a": {"reference": "one"}, "b": {"reference": "two"}}
    assert align_records({"ours": a, "other": dict(reversed(list(a.items())))}) == ["a", "b"]
    with pytest.raises(ValueError, match="Reference mismatch"):
        align_records({"ours": a, "other": {"a": {"reference": "wrong"}, "b": a["b"]}})
    with pytest.raises(ValueError, match="Case ID set mismatch"):
        align_records({"ours": a, "other": {"a": a["a"]}})


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "eka.json"
    row = {"id": "a", "prediction": "one", "reference": "two"}
    path.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_records(path)


def test_meteor_never_falls_back_to_lite(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="no METEOR-lite fallback"):
        meteor_score(["one"], ["one"], tmp_path, tmp_path / "log")
