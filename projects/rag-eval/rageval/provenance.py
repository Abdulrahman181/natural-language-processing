"""Reproducibility and artifact-integrity helpers."""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def corpus_fingerprint(documents) -> str:
    return stable_digest([{"title": item.title, "text": item.text} for item in documents])


def chunks_fingerprint(chunks) -> str:
    return stable_digest([{"id": item.id, "title": item.title, "text": item.text} for item in chunks])


def questions_fingerprint(questions) -> str:
    return stable_digest(
        [
            {"id": item.id, "question": item.question, "evidence": item.evidence, "doc_title": item.doc_title}
            for item in questions
        ]
    )


def run_manifest(index_meta: dict, questions_in_file: int, questions_evaluated: int, verified_only: bool) -> dict:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "index_fingerprint": index_meta.get("index_fingerprint"),
        "corpus_fingerprint": index_meta.get("corpus_fingerprint"),
        "benchmark_fingerprint": index_meta.get("benchmark_fingerprint"),
        "questions_in_file": questions_in_file,
        "questions_evaluated": questions_evaluated,
        "verified_only": verified_only,
    }


def assert_index_integrity(directory: Path, index) -> None:
    expected = index.meta.get("chunks_fingerprint")
    if expected and expected != chunks_fingerprint(index.chunks):
        raise ValueError("Index chunks do not match chunks_fingerprint in meta.json; rebuild the index")
    if index.embeddings is not None and index.embeddings.shape[0] != len(index.chunks):
        raise ValueError("Embeddings row count does not match index chunks")
    if index.meta.get("dense") and index.embeddings is None:
        raise ValueError("Index metadata says dense=true but embeddings.npy is missing")
