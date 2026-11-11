import pytest
from helpers import write_local_dataset

from cwe_qlora.cli import main
from cwe_qlora.io_utils import read_json, read_jsonl, write_jsonl


@pytest.fixture
def prepared(tmp_path):
    raw = write_local_dataset(tmp_path / "raw")
    work = tmp_path / "work"
    code = main(
        ["prepare", "--source", "local", "--local-dir", str(raw), "--out", str(work),
         "--train-size", "200", "--val-size", "30", "--test-size", "60", "--seed", "5"]
    )
    assert code == 0
    return tmp_path, work


def test_prepare_writes_all_splits_and_metadata(prepared):
    _, work = prepared
    assert len(read_jsonl(work / "train.jsonl")) == 200
    assert len(read_jsonl(work / "val.jsonl")) == 30
    assert len(read_jsonl(work / "test.jsonl")) == 60
    assert len(read_jsonl(work / "full_train.jsonl")) == 600
    meta = read_json(work / "prepare_meta.json")
    assert meta["sizes"]["test_sample"] == 60
    assert meta["dataset"].startswith("local:")


def test_prepare_requires_local_dir_for_local_source(tmp_path):
    assert main(["prepare", "--source", "local", "--out", str(tmp_path / "w")]) == 2


def test_baselines_and_report_end_to_end(prepared):
    tmp_path, work = prepared
    results = tmp_path / "results"
    assert main(["baselines", "--work", str(work), "--out", str(results)]) == 0
    for name in ("pred_majority", "pred_tfidf_svm_matched", "pred_tfidf_svm_full", "similarity"):
        assert (results / f"{name}.jsonl").exists()

    test = read_jsonl(work / "test.jsonl")
    cutoff = int(len(test) * 0.8)
    write_jsonl(
        results / "pred_base_zero_shot.jsonl",
        [{"id": row["id"], "gold": row["label"], "raw": "CWE-79", "pred": "CWE-79"} for row in test],
    )
    write_jsonl(
        results / "pred_qlora_tuned.jsonl",
        [
            {"id": row["id"], "gold": row["label"], "raw": "x", "pred": row["label"] if index < cutoff else "CWE-1"}
            for index, row in enumerate(test)
        ],
    )
    assert main(["report", "--work", str(work), "--out", str(results)]) == 0

    report = read_json(results / "results.json")
    assert set(report["systems"]) == {"majority", "tfidf_svm_matched", "tfidf_svm_full", "base_zero_shot", "qlora_tuned"}
    assert report["systems"]["qlora_tuned"]["accuracy"] == pytest.approx(cutoff / len(test))
    assert report["systems"]["qlora_tuned"]["invalid_rate"] == 0.0
    assert report["systems"]["tfidf_svm_full"]["accuracy"] > 0.9
    assert len(report["paired_comparisons"]) == 3
    assert "novel_subset" in report["systems"]["majority"]
    markdown = (results / "results.md").read_text(encoding="utf-8")
    assert "How to read these numbers" in markdown
    assert "agreement-filtered" in markdown
    assert "Base model + QLoRA adapter" in markdown


def test_report_counts_missing_predictions_as_invalid(prepared):
    tmp_path, work = prepared
    results = tmp_path / "results"
    test = read_jsonl(work / "test.jsonl")
    write_jsonl(
        results / "pred_qlora_tuned.jsonl",
        [{"id": row["id"], "gold": row["label"], "raw": row["label"], "pred": row["label"]} for row in test[:30]],
    )
    assert main(["report", "--work", str(work), "--out", str(results)]) == 0
    item = read_json(results / "results.json")["systems"]["qlora_tuned"]
    assert item["invalid_rate"] == pytest.approx(0.5)
    assert item["accuracy"] == pytest.approx(0.5)


def test_report_fails_clearly_without_predictions(prepared):
    tmp_path, work = prepared
    with pytest.raises(FileNotFoundError):
        main(["report", "--work", str(work), "--out", str(tmp_path / "empty")])
