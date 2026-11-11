import json

import pytest
from helpers import UNIQUE_WORDS, FakeEmbedder, FakeLLM, fact_sentence, make_documents, question_text

from rageval import cli
from rageval.cli import main
from rageval.corpus import save_corpus
from rageval.evaluate import evaluate_retriever
from rageval.index import build_index, load_index
from rageval.questions import FIELDNAMES, Question, read_questions, write_questions
from rageval.retrievers import Bm25Retriever

BUILD_ARGS = ["--chunk-size", "60", "--overlap", "10", "--min-words", "20"]


def question_rows(verified_flags=None):
    rows = []
    for index, word in enumerate(UNIQUE_WORDS, start=1):
        flag = "1" if verified_flags is None else verified_flags[index - 1]
        rows.append(
            {
                "id": f"q{index:04d}",
                "question": question_text(word),
                "evidence": fact_sentence(index, word),
                "chunk_id": "",
                "doc_title": f"وثيقة {index}",
                "chunk_text": "",
                "verified": flag,
                "note": "",
            }
        )
    return rows


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    save_corpus(make_documents(), corpus)
    monkeypatch.setattr(cli, "_make_embedder", lambda *args, **kwargs: FakeEmbedder())
    return tmp_path, corpus


def build(workspace, extra=()):
    tmp_path, corpus = workspace
    code = main(["build-index", "--corpus", str(corpus), "--out", str(tmp_path / "idx"), *BUILD_ARGS, *extra])
    assert code == 0
    return tmp_path / "idx"


def test_build_index_writes_chunks_embeddings_and_metadata(workspace):
    index_dir = build(workspace)
    index = load_index(index_dir)
    assert index.meta["documents"] == 8
    assert index.meta["chunks"] == len(index.chunks) == 24
    assert index.meta["dense"] is True
    assert index.embeddings.shape == (24, 1024)
    assert index.meta["embedding_model"] == "fake-hash-embedder"


def test_build_index_without_dense_has_no_embeddings(workspace):
    index = load_index(build(workspace, ["--no-dense"]))
    assert index.embeddings is None and index.meta["dense"] is False


def test_evaluate_compares_bm25_dense_and_hybrid(workspace):
    tmp_path, _ = workspace
    index_dir = build(workspace)
    write_questions(tmp_path / "questions.csv", question_rows())
    out = tmp_path / "results"
    code = main(["evaluate", "--index", str(index_dir), "--questions", str(tmp_path / "questions.csv"), "--out", str(out)])
    assert code == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    by_name = {item["retriever"]: item for item in results["results"]}
    assert set(by_name) == {"bm25", "dense", "hybrid_rrf"}
    assert by_name["bm25"]["recall@1"] == 1.0
    assert by_name["dense"]["recall@5"] == 1.0
    assert by_name["hybrid_rrf"]["recall@5"] == 1.0
    assert by_name["bm25"]["questions"] == 8
    assert results["meta"]["questions_evaluated"] == 8
    assert (out / "per_query.csv").exists()
    report = (out / "results.md").read_text(encoding="utf-8")
    assert "Recall@5" in report and "fake-hash-embedder" in report


def test_evaluate_uses_only_verified_questions_by_default(workspace):
    tmp_path, _ = workspace
    index_dir = build(workspace, ["--no-dense"])
    flags = ["1", "1", "1", "0", "0", "0", "0", "0"]
    write_questions(tmp_path / "questions.csv", question_rows(flags))
    base = ["evaluate", "--index", str(index_dir), "--questions", str(tmp_path / "questions.csv")]
    assert main(base + ["--out", str(tmp_path / "a")]) == 0
    assert json.loads((tmp_path / "a" / "results.json").read_text(encoding="utf-8"))["meta"]["questions_evaluated"] == 3
    assert main(base + ["--out", str(tmp_path / "b"), "--include-unverified"]) == 0
    assert json.loads((tmp_path / "b" / "results.json").read_text(encoding="utf-8"))["meta"]["questions_evaluated"] == 8


def test_evaluate_refuses_when_nothing_is_verified(workspace):
    tmp_path, _ = workspace
    index_dir = build(workspace, ["--no-dense"])
    write_questions(tmp_path / "questions.csv", question_rows(["0"] * 8))
    code = main(["evaluate", "--index", str(index_dir), "--questions", str(tmp_path / "questions.csv"), "--out", str(tmp_path / "r")])
    assert code == 2


def test_adapter_retriever_is_scored_and_a_wrong_adapter_scores_zero(workspace):
    tmp_path, _ = workspace
    index_dir = build(workspace, ["--no-dense"])
    write_questions(tmp_path / "questions.csv", question_rows())
    adapter = tmp_path / "wrong.py"
    adapter.write_text('NAME = "always-wrong"\n\ndef search(query, k):\n    return ["unrelated words only"] * k\n', encoding="utf-8")
    out = tmp_path / "r"
    code = main(
        ["evaluate", "--index", str(index_dir), "--questions", str(tmp_path / "questions.csv"), "--out", str(out), "--adapter", str(adapter)]
    )
    assert code == 0
    by_name = {item["retriever"]: item for item in json.loads((out / "results.json").read_text(encoding="utf-8"))["results"]}
    assert by_name["always-wrong"]["recall@10"] == 0.0
    assert by_name["always-wrong"]["mrr@10"] == 0.0
    assert by_name["bm25"]["recall@1"] == 1.0


def test_adapter_without_search_function_is_rejected(tmp_path):
    from rageval.retrievers import load_adapter

    broken = tmp_path / "broken.py"
    broken.write_text("NAME = 'x'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="search"):
        load_adapter(broken)


def test_questions_with_mismatched_evidence_score_zero(workspace):
    index_dir = build(workspace, ["--no-dense"])
    index = load_index(index_dir)
    wrong = [Question("q1", question_text(UNIQUE_WORDS[0]), "عبارة لا وجود لها في أي وثيقة من الوثائق", "")]
    summary, ranks, _ = evaluate_retriever(Bm25Retriever(index.chunks), wrong)
    assert ranks == [None] and summary["recall@10"] == 0.0


def test_evaluate_retriever_rejects_empty_question_list(workspace):
    index = load_index(build(workspace, ["--no-dense"]))
    with pytest.raises(ValueError):
        evaluate_retriever(Bm25Retriever(index.chunks), [])


def test_evidence_is_independent_of_the_chunking_used_by_the_retriever(workspace):
    docs = make_documents()
    for size, overlap in [(30, 5), (60, 10), (200, 20)]:
        index = build_index(docs, workspace[0] / f"idx_{size}", size, overlap, 5)
        questions = [Question(f"q{n}", question_text(word), fact_sentence(n, word), "") for n, word in enumerate(UNIQUE_WORDS, start=1)]
        summary, _, _ = evaluate_retriever(Bm25Retriever(index.chunks), questions)
        assert summary["recall@10"] == 1.0


def test_gen_questions_command_end_to_end(workspace, monkeypatch):
    tmp_path, _ = workspace
    index_dir = build(workspace, ["--no-dense"])
    monkeypatch.setattr(cli, "ChatClient", lambda *args, **kwargs: FakeLLM())
    out = tmp_path / "generated.csv"
    base = ["gen-questions", "--index", str(index_dir), "--out", str(out), "--n", "8", "--per-doc", "3", "--min-words", "40", "--pause", "0", "--model", "m"]
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert main(base) == 2
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert main(base) == 0
    rows, total = read_questions(out, verified_only=False)
    assert total == 8 and len(rows) == 8
    assert read_questions(out, verified_only=True)[0] == []
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")


def test_gen_questions_allows_local_endpoints_without_a_key(workspace, monkeypatch):
    tmp_path, _ = workspace
    index_dir = build(workspace, ["--no-dense"])
    monkeypatch.setattr(cli, "ChatClient", lambda *args, **kwargs: FakeLLM())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    code = main(
        ["gen-questions", "--index", str(index_dir), "--out", str(tmp_path / "g.csv"), "--n", "2", "--per-doc", "3", "--min-words", "40", "--pause", "0", "--model", "m", "--base-url", "http://localhost:11434/v1"]
    )
    assert code == 0


def test_fetch_and_export_commands(tmp_path, monkeypatch):
    titles = tmp_path / "titles.txt"
    titles.write_text("A\nB\n", encoding="utf-8")
    monkeypatch.setattr(cli, "fetch_wikipedia", lambda *args, **kwargs: (make_documents()[:2], [("B", "not_found")]))
    corpus = tmp_path / "corpus.jsonl"
    assert main(["fetch", "--titles", str(titles), "--out", str(corpus)]) == 0
    assert main(["export-txt", "--corpus", str(corpus), "--out-dir", str(tmp_path / "txt")]) == 0
    assert len(list((tmp_path / "txt").glob("*.txt"))) == 2
    monkeypatch.setattr(cli, "fetch_wikipedia", lambda *args, **kwargs: ([], [("A", "not_found")]))
    assert main(["fetch", "--titles", str(titles), "--out", str(tmp_path / "empty.jsonl")]) == 1


def test_question_csv_columns_are_stable():
    assert FIELDNAMES == ["id", "question", "evidence", "chunk_id", "doc_title", "chunk_text", "verified", "note"]
