
import pytest
from helpers import FakeEmbedder

from rageval.corpus import Document, save_corpus
from rageval.index import build_index, load_index
from rageval.provenance import chunks_fingerprint, corpus_fingerprint, questions_fingerprint
from rageval.questions import Question


def test_corpus_and_chunk_fingerprints_are_stable(tmp_path):
    docs = [Document("A", "alpha beta gamma " * 20)]
    path = tmp_path / "corpus.jsonl"
    save_corpus(docs, path)
    assert corpus_fingerprint(docs) == corpus_fingerprint(docs)
    index = build_index(docs, tmp_path / "idx", 20, 5, 5, FakeEmbedder())
    assert index.meta["corpus_fingerprint"] == corpus_fingerprint(docs)
    assert index.meta["chunks_fingerprint"] == chunks_fingerprint(index.chunks)
    assert load_index(tmp_path / "idx").meta["chunks_fingerprint"]


def test_load_index_rejects_tampered_chunks(tmp_path):
    docs = [Document("A", "alpha beta gamma " * 20)]
    index_dir = tmp_path / "idx"
    build_index(docs, index_dir, 20, 5, 5)
    chunks = (index_dir / "chunks.jsonl").read_text()
    (index_dir / "chunks.jsonl").write_text(chunks.replace("alpha", "tampered", 1))
    with pytest.raises(ValueError, match="chunks_fingerprint"):
        load_index(index_dir)


def test_duplicate_question_ids_are_rejected(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(
        "id,question,evidence,chunk_id,doc_title,chunk_text,verified,note\n"
        "q1,What is this?,alpha words here,c1,A,,1,\n"
        "q1,What else?,beta words here,c2,A,,1,\n",
        encoding="utf-8",
    )
    from rageval.questions import read_questions

    with pytest.raises(ValueError, match="Duplicate question id"):
        read_questions(path)


def test_question_fingerprint_changes_with_content():
    first = [Question("q1", "سؤال", "دليل", "وثيقة")]
    second = [Question("q1", "سؤال آخر", "دليل", "وثيقة")]
    assert questions_fingerprint(first) != questions_fingerprint(second)
