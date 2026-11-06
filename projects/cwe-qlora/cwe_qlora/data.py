import json
import hashlib
import random
from collections import Counter
from pathlib import Path

from .io_utils import read_jsonl
from .parsing import canonical
from .prompts import clean_description

DEFAULT_DATASET = "xamxte/cve-to-cwe"
SPLIT_FILES = {"train": "train", "validation": "validation", "test": "test"}


def canonical_rows(rows, id_key="cve_id", text_key="description", label_key="cwe_id"):
    cleaned = []
    dropped = 0
    for row in rows:
        label = canonical(row.get(label_key))
        text = clean_description(row.get(text_key))
        if label is None or not text:
            dropped += 1
            continue
        cleaned.append({"id": str(row[id_key]), "description": text, "label": label})
    return cleaned, dropped


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(rows):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(rows, n)


def cap_per_class(rows: list[dict], cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    counts: Counter = Counter()
    kept = []
    for row in shuffled:
        if counts[row["label"]] < cap:
            counts[row["label"]] += 1
            kept.append(row)
    return kept


def class_counts(rows: list[dict]) -> Counter:
    return Counter(row["label"] for row in rows)


def exact_overlap(reference: list[dict], candidates: list[dict]) -> int:
    seen = {row["description"] for row in reference}
    return sum(1 for row in candidates if row["description"] in seen)


def data_fingerprint(splits: dict[str, list[dict]]) -> str:
    """Return a stable digest of the exact canonical rows used by an experiment."""
    payload = []
    for split_name in sorted(splits):
        for row in sorted(splits[split_name], key=lambda item: item["id"]):
            payload.append({"split": split_name, **row})
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_unique_ids(rows: list[dict], split_name: str) -> None:
    ids = [row["id"] for row in rows]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise ValueError(f"{duplicates} duplicate ids found within {split_name} split")


def load_local_splits(directory) -> dict[str, list[dict]]:
    base = Path(directory)
    return {name: read_jsonl(base / f"{stem}.jsonl") for name, stem in SPLIT_FILES.items()}


def load_hub_splits(dataset_id: str, revision: str | None = None) -> dict[str, list[dict]]:
    from datasets import load_dataset

    loaded = load_dataset(dataset_id, revision=revision)
    columns = ["cve_id", "description", "cwe_id"]
    return {name: [dict(row) for row in loaded[name].select_columns(columns)] for name in SPLIT_FILES}


def dataset_revision(dataset_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset_id).sha
    except Exception:
        return None


def prepare_splits(
    raw: dict[str, list[dict]],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    max_per_class: int | None = None,
):
    train_all, dropped_train = canonical_rows(raw["train"])
    val_all, dropped_val = canonical_rows(raw["validation"])
    test_all, dropped_test = canonical_rows(raw["test"])
    for name, rows in (("train", train_all), ("validation", val_all), ("test", test_all)):
        _assert_unique_ids(rows, name)
    pool = cap_per_class(train_all, max_per_class, seed) if max_per_class else train_all
    train = sample_rows(pool, train_size, seed)
    val = sample_rows(val_all, val_size, seed + 1)
    test = sample_rows(test_all, test_size, seed + 2)
    ids_by_split = {
        "train": {row["id"] for row in train_all},
        "validation": {row["id"] for row in val_all},
        "test": {row["id"] for row in test_all},
    }
    overlaps = {
        "train_validation": len(ids_by_split["train"] & ids_by_split["validation"]),
        "train_test": len(ids_by_split["train"] & ids_by_split["test"]),
        "validation_test": len(ids_by_split["validation"] & ids_by_split["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"IDs appear in both train/validation/test splits: {overlaps}")
    train_labels = set(class_counts(train))
    test_counts = class_counts(test)
    meta = {
        "seed": seed,
        "max_per_class": max_per_class,
        "rows_dropped_invalid": {"train": dropped_train, "validation": dropped_val, "test": dropped_test},
        "sizes": {
            "full_train": len(train_all),
            "train_subset": len(train),
            "validation_sample": len(val),
            "test_sample": len(test),
        },
        "classes": {
            "full_train": len(class_counts(train_all)),
            "train_subset": len(train_labels),
            "test_sample": len(test_counts),
        },
        "top_train_labels": class_counts(train).most_common(10),
        "test_rows_with_label_unseen_in_train_subset": sum(
            count for label, count in test_counts.items() if label not in train_labels
        ),
        "test_exact_description_duplicates_in_full_train": exact_overlap(train_all, test),
        "validation_exact_description_duplicates_in_full_train": exact_overlap(train_all, val),
        "split_id_overlaps": overlaps,
    }
    splits = {"train": train, "val": val, "test": test, "full_train": train_all}
    meta["data_fingerprint"] = data_fingerprint(splits)
    return splits, meta


def meta_to_json(meta: dict) -> str:
    return json.dumps(meta, ensure_ascii=False, indent=2, default=str)
