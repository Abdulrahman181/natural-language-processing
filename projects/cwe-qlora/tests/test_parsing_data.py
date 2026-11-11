import pytest
from helpers import make_rows

from cwe_qlora.data import (
    canonical_rows,
    cap_per_class,
    class_counts,
    exact_overlap,
    prepare_splits,
    sample_rows,
)
from cwe_qlora.io_utils import read_jsonl, write_jsonl
from cwe_qlora.parsing import INVALID, canonical, parse_prediction
from cwe_qlora.prompts import SYSTEM_PROMPT, build_messages, clean_description


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("CWE-79", "CWE-79"),
        ("cwe 79", "CWE-79"),
        ("CWE-079", "CWE-79"),
        ("CWE_89", "CWE-89"),
        ("The answer is CWE-89: SQL injection", "CWE-89"),
        ("CWE-119 or CWE-120", "CWE-119"),
        ("79", "CWE-79"),
        (" 79. ", "CWE-79"),
        ("CWE - 22", "CWE-22"),
        ("NVD-CWE-Other", None),
        ("", None),
        (None, None),
        ("no identifier here", None),
    ],
)
def test_canonical(raw, expected):
    assert canonical(raw) == expected


def test_parse_prediction_marks_unparseable_output_as_invalid():
    assert parse_prediction("CWE-416") == "CWE-416"
    assert parse_prediction("I cannot tell") == INVALID


def test_clean_description_collapses_whitespace_and_truncates():
    assert clean_description("  a \n\t b  ") == "a b"
    assert len(clean_description("x" * 5000)) == 1500
    assert clean_description(None) == ""


def test_build_messages_uses_system_prompt_and_cleaned_user_text():
    messages = build_messages("  desc   text ")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "desc text"}


def test_canonical_rows_normalizes_and_drops_invalid_rows():
    rows = [
        {"cve_id": "CVE-1", "description": "  A   bug \n here ", "cwe_id": "CWE-079"},
        {"cve_id": "CVE-2", "description": "ok", "cwe_id": "NVD-CWE-Other"},
        {"cve_id": "CVE-3", "description": "   ", "cwe_id": "CWE-89"},
        {"cve_id": "CVE-4", "description": "fine", "cwe_id": None},
    ]
    cleaned, dropped = canonical_rows(rows)
    assert cleaned == [{"id": "CVE-1", "description": "A bug here", "label": "CWE-79"}]
    assert dropped == 3


def test_sample_rows_is_deterministic_and_handles_oversized_requests():
    rows = [{"id": str(number), "label": "A", "description": str(number)} for number in range(50)]
    first = sample_rows(rows, 10, seed=4)
    assert first == sample_rows(rows, 10, seed=4)
    assert len({row["id"] for row in first}) == 10
    assert first != sample_rows(rows, 10, seed=5)
    assert {row["id"] for row in sample_rows(rows, 500, seed=1)} == {row["id"] for row in rows}


def test_cap_per_class_limits_head_classes_only():
    rows = [{"id": f"a{n}", "label": "A", "description": f"a{n}"} for n in range(10)]
    rows += [{"id": f"b{n}", "label": "B", "description": f"b{n}"} for n in range(3)]
    capped = cap_per_class(rows, 4, seed=1)
    assert class_counts(capped) == {"A": 4, "B": 3}


def test_exact_overlap_counts_repeated_descriptions():
    reference = [{"description": "same"}, {"description": "other"}]
    candidates = [{"description": "same"}, {"description": "new"}, {"description": "same"}]
    assert exact_overlap(reference, candidates) == 2


def raw_splits():
    return {
        "train": make_rows(300, 1, 0),
        "validation": make_rows(40, 2, 10000),
        "test": make_rows(60, 3, 20000),
    }


def test_prepare_splits_reports_sizes_and_leakage_statistics():
    splits, meta = prepare_splits(raw_splits(), train_size=100, val_size=20, test_size=30, seed=7)
    assert len(splits["train"]) == 100 and len(splits["val"]) == 20 and len(splits["test"]) == 30
    assert len(splits["full_train"]) == 300
    assert meta["sizes"] == {"full_train": 300, "train_subset": 100, "validation_sample": 20, "test_sample": 30}
    assert meta["classes"]["full_train"] == 5
    assert meta["test_rows_with_label_unseen_in_train_subset"] == 0
    assert meta["test_exact_description_duplicates_in_full_train"] >= 0


def test_prepare_splits_applies_class_cap_before_sampling():
    _, meta = prepare_splits(raw_splits(), train_size=500, val_size=10, test_size=10, seed=7, max_per_class=20)
    assert meta["sizes"]["train_subset"] <= 100


def test_prepare_splits_rejects_id_overlap_between_train_and_test():
    raw = raw_splits()
    raw["test"][0]["cve_id"] = raw["train"][0]["cve_id"]
    with pytest.raises(ValueError, match="both train"):
        prepare_splits(raw, 50, 10, 60, seed=1)


def test_jsonl_roundtrip_keeps_unicode(tmp_path):
    rows = [{"id": "1", "description": "café مرحبا"}]
    write_jsonl(tmp_path / "x.jsonl", rows)
    assert read_jsonl(tmp_path / "x.jsonl") == rows
