import pytest
from helpers import write_local_dataset

from cwe_qlora.cli import main
from cwe_qlora.data import data_fingerprint
from cwe_qlora.io_utils import read_jsonl, write_jsonl
from cwe_qlora.modeling import resolve_model_revision
from cwe_qlora.report import build_report


def _prepare(tmp_path, mode):
    raw = write_local_dataset(tmp_path / "raw", train=40, validation=10, test=10)
    work = tmp_path / "work"
    assert main([
        "prepare", "--source", "local", "--local-dir", str(raw), "--out", str(work),
        "--train-size", "20", "--val-size", "5", "--test-size", "5", "--run-mode", mode,
    ]) == 0
    return work


def test_prepare_records_run_mode(tmp_path):
    work = _prepare(tmp_path, "dry-run")
    import json
    meta = json.loads((work / "prepare_meta.json").read_text())
    assert meta["run_mode"] == "dry-run"


def test_report_rejects_dry_run_by_default(tmp_path):
    work = _prepare(tmp_path, "dry-run")
    results = tmp_path / "results"
    rows = read_jsonl(work / "test.jsonl")
    write_jsonl(results / "pred_majority.jsonl", [
        {"id": row["id"], "gold": row["label"], "raw": row["label"], "pred": row["label"]}
        for row in rows
    ])
    with pytest.raises(ValueError, match="dry-run"):
        build_report(work, results)
    assert build_report(work, results, allow_dry_run=True)["test_sample_size"] == 5


def test_data_fingerprint_is_order_independent_but_content_sensitive():
    rows = {"train": [{"id": "1", "description": "a", "label": "CWE-1"}]}
    reordered = {"train": list(reversed(rows["train"]))}
    assert data_fingerprint(rows) == data_fingerprint(reordered)
    changed = {"train": [{"id": "1", "description": "b", "label": "CWE-1"}]}
    assert data_fingerprint(rows) != data_fingerprint(changed)


def test_local_model_revision_does_not_require_hub_access(tmp_path):
    assert resolve_model_revision(str(tmp_path)) is None
    assert resolve_model_revision(str(tmp_path), "local-test-revision") == "local-test-revision"


def test_report_rejects_duplicate_prediction_ids(tmp_path):
    work = _prepare(tmp_path, "full")
    results = tmp_path / "results"
    rows = read_jsonl(work / "test.jsonl")
    write_jsonl(results / "pred_majority.jsonl", [
        {"id": rows[0]["id"], "gold": rows[0]["label"], "raw": rows[0]["label"], "pred": rows[0]["label"]},
        {"id": rows[0]["id"], "gold": rows[0]["label"], "raw": rows[0]["label"], "pred": rows[0]["label"]},
    ])
    with pytest.raises(ValueError, match="Duplicate prediction ids"):
        build_report(work, results)
