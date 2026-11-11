import csv
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .index import Chunk
from .text import contains_evidence, raw_tokens, word_count

FIELDNAMES = ["id", "question", "evidence", "chunk_id", "doc_title", "chunk_text", "verified", "note"]

PROMPT_PATH = Path(__file__).parent / "prompts" / "question_ar.txt"

_PASSAGE_WORDS = {"\u0627\u0644\u0645\u0642\u0637\u0639", "\u0627\u0644\u0646\u0635", "\u0627\u0644\u0641\u0642\u0631\u0647"}


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    evidence: str
    doc_title: str


def build_prompt(passage: str) -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace("__PASSAGE__", passage)


def parse_response(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question = data.get("question")
    evidence = data.get("evidence")
    if not isinstance(question, str) or not isinstance(evidence, str):
        return None
    return {"question": question.strip(), "evidence": evidence.strip()}


def max_shared_run(first: list[str], second: list[str]) -> int:
    best = 0
    previous = [0] * (len(second) + 1)
    for token in first:
        current = [0] * (len(second) + 1)
        for position, other in enumerate(second, start=1):
            if token == other:
                current[position] = previous[position - 1] + 1
                best = max(best, current[position])
        previous = current
    return best


def validate(
    candidate: dict,
    chunk_text: str,
    max_run: int = 3,
    min_question_tokens: int = 4,
    min_evidence_tokens: int = 6,
) -> str | None:
    question_tokens = raw_tokens(candidate["question"])
    if len(question_tokens) < min_question_tokens:
        return "question_too_short"
    if len(raw_tokens(candidate["evidence"])) < min_evidence_tokens:
        return "evidence_too_short"
    if not contains_evidence(chunk_text, candidate["evidence"]):
        return "evidence_not_in_chunk"
    if _PASSAGE_WORDS.intersection(question_tokens):
        return "references_passage"
    if max_shared_run(question_tokens, raw_tokens(chunk_text)) > max_run:
        return "copies_passage"
    return None


def sample_chunks(chunks: list[Chunk], per_doc: int, min_words: int, seed: int) -> list[Chunk]:
    import numpy as np

    rng = np.random.default_rng(seed)
    eligible = [chunk for chunk in chunks if word_count(chunk.text) >= min_words]
    eligible = [eligible[index] for index in rng.permutation(len(eligible))]
    taken: Counter = Counter()
    picked = []
    for chunk in eligible:
        if taken[chunk.title] >= per_doc:
            continue
        taken[chunk.title] += 1
        picked.append(chunk)
    return picked


def generate(
    chunks: list[Chunk],
    llm,
    target: int,
    per_doc: int = 2,
    min_words: int = 60,
    seed: int = 13,
    max_run: int = 3,
    pause: float = 0.0,
    max_consecutive_errors: int = 5,
    max_attempts: int | None = None,
    sleep=time.sleep,
) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    stats: Counter = Counter()
    seen_questions: set[str] = set()
    seen_evidence: set[str] = set()
    consecutive_errors = 0
    limit = max_attempts if max_attempts is not None else target * 4
    for chunk in sample_chunks(chunks, per_doc, min_words, seed):
        if len(rows) >= target or stats["attempts"] >= limit:
            break
        stats["attempts"] += 1
        try:
            raw = llm(build_prompt(chunk.text))
            consecutive_errors = 0
        except Exception as error:
            stats["llm_error"] += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(f"Stopped after {consecutive_errors} consecutive LLM errors: {error}") from error
            continue
        parsed = parse_response(raw)
        if parsed is None:
            stats["rejected_unparseable"] += 1
            continue
        reason = validate(parsed, chunk.text, max_run=max_run)
        if reason is not None:
            stats[f"rejected_{reason}"] += 1
            continue
        question_key = " ".join(raw_tokens(parsed["question"]))
        evidence_key = " ".join(raw_tokens(parsed["evidence"]))
        if question_key in seen_questions or evidence_key in seen_evidence:
            stats["rejected_duplicate"] += 1
            continue
        seen_questions.add(question_key)
        seen_evidence.add(evidence_key)
        rows.append(
            {
                "id": f"q{len(rows) + 1:04d}",
                "question": parsed["question"],
                "evidence": parsed["evidence"],
                "chunk_id": chunk.id,
                "doc_title": chunk.title,
                "chunk_text": chunk.text,
                "verified": "0",
                "note": "",
            }
        )
        stats["accepted"] += 1
        if pause > 0:
            sleep(pause)
    return rows, stats


def write_questions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_questions(path: Path, verified_only: bool = True) -> tuple[list[Question], int]:
    questions = []
    total = 0
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not (row.get("question") or "").strip():
                continue
            total += 1
            question_id = (row.get("id") or "").strip()
            if not question_id:
                raise ValueError(f"Question row {total} has no id")
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question id: {question_id}")
            seen_ids.add(question_id)
            verified = (row.get("verified") or "").strip().lower() in {"1", "true", "yes", "y"}
            if verified_only and not verified:
                continue
            questions.append(
                Question(
                    id=question_id,
                    question=row["question"].strip(),
                    evidence=row["evidence"].strip(),
                    doc_title=row.get("doc_title", ""),
                )
            )
    return questions, total
