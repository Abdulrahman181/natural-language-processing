import json

import pytest
from helpers import FakeLLM, make_documents

from rageval.index import Chunk, chunk_corpus
from rageval.questions import (
    build_prompt,
    generate,
    max_shared_run,
    parse_response,
    read_questions,
    sample_chunks,
    validate,
    write_questions,
)

CHUNK = "يشتهر الإقليم رقم 1 بإنتاج زيتون بكميات كبيرة منذ قرون عديدة وتتنوع الأنشطة"
GOOD = {
    "question": "أي إقليم اشتهر قديما بصناعة زيتون بالتحديد",
    "evidence": "يشتهر الإقليم رقم 1 بإنتاج زيتون بكميات كبيرة",
}


def corpus_chunks():
    return chunk_corpus(make_documents(), size=60, overlap=10, min_words=20)


def test_build_prompt_embeds_the_passage_between_markers():
    prompt = build_prompt("نص تجريبي")
    assert "<<<\nنص تجريبي\n>>>" in prompt
    assert "__PASSAGE__" not in prompt


def test_parse_response_accepts_json_inside_prose_and_rejects_bad_payloads():
    assert parse_response('Here:\n```json\n{"question": " س؟ ", "evidence": "ج"}\n```') == {
        "question": "س؟",
        "evidence": "ج",
    }
    assert parse_response("no json") is None
    assert parse_response('{"question": "x"}') is None
    assert parse_response('{"question": 1, "evidence": 2}') is None
    assert parse_response("{broken json}") is None


def test_max_shared_run():
    assert max_shared_run(["a", "b", "c", "d"], ["x", "b", "c", "y"]) == 2
    assert max_shared_run(["a", "b", "c"], ["a", "b", "c"]) == 3
    assert max_shared_run([], ["a"]) == 0


def test_validate_accepts_a_good_candidate():
    assert validate(GOOD, CHUNK) is None


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"question": "ما هو"}, "question_too_short"),
        ({"evidence": "يشتهر الإقليم"}, "evidence_too_short"),
        ({"evidence": "جملة غير موجودة في المقطع الأصلي إطلاقا"}, "evidence_not_in_chunk"),
        ({"question": "ما هي الفكرة الرئيسية التي يذكرها المقطع هنا"}, "references_passage"),
        ({"question": "يشتهر الإقليم رقم 1 بإنتاج ماذا"}, "copies_passage"),
    ],
)
def test_validate_rejects_bad_candidates(override, reason):
    assert validate({**GOOD, **override}, CHUNK) == reason


def test_sample_chunks_caps_per_document_filters_short_chunks_and_is_seeded():
    chunks = corpus_chunks()
    picked = sample_chunks(chunks, per_doc=1, min_words=50, seed=3)
    assert len({chunk.title for chunk in picked}) == len(picked) == 8
    assert all(len(chunk.text.split()) >= 50 for chunk in picked)
    assert [chunk.id for chunk in picked] == [chunk.id for chunk in sample_chunks(chunks, 1, 50, 3)]


def test_generate_accepts_only_validated_unique_questions():
    llm = FakeLLM()
    rows, stats = generate(corpus_chunks(), llm, target=8, per_doc=3, min_words=40, seed=1, max_attempts=100)
    assert len(rows) == 8 == stats["accepted"]
    assert [row["id"] for row in rows] == [f"q{number:04d}" for number in range(1, 9)]
    assert len({row["doc_title"] for row in rows}) == 8
    assert all(row["verified"] == "0" for row in rows)
    assert all(row["evidence"] in row["chunk_text"] for row in rows)


def test_generate_counts_unparseable_responses():
    rows, stats = generate(corpus_chunks(), lambda prompt: "nothing", target=3, per_doc=3, min_words=40, max_attempts=10)
    assert rows == []
    assert stats["rejected_unparseable"] == 10


def test_generate_rejects_duplicates():
    twin = [Chunk("a", "T", CHUNK + " " + "كلمة " * 60), Chunk("b", "U", CHUNK + " " + "كلمة " * 60)]
    def llm(prompt):
        return json.dumps(GOOD, ensure_ascii=False)

    rows, stats = generate(twin, llm, target=5, per_doc=1, min_words=10, max_attempts=10)
    assert len(rows) == 1
    assert stats["rejected_duplicate"] == 1


def test_generate_stops_after_consecutive_llm_errors():
    def failing(prompt):
        raise ConnectionError("down")

    with pytest.raises(RuntimeError, match="consecutive"):
        generate(corpus_chunks(), failing, target=3, per_doc=3, min_words=40, max_consecutive_errors=3)


def test_generate_survives_an_isolated_llm_error():
    inner = FakeLLM()
    state = {"first": True}

    def flaky(prompt):
        if state["first"]:
            state["first"] = False
            raise ConnectionError("blip")
        return inner(prompt)

    rows, stats = generate(corpus_chunks(), flaky, target=8, per_doc=3, min_words=40, seed=1, max_attempts=100)
    assert stats["llm_error"] == 1
    assert len(rows) >= 7


def test_question_file_roundtrip_and_verification_flag(tmp_path):
    rows, _ = generate(corpus_chunks(), FakeLLM(), target=4, per_doc=3, min_words=40, seed=1, max_attempts=100)
    rows[0]["verified"] = "1"
    rows[1]["verified"] = "TRUE"
    rows[2]["verified"] = "yes"
    target = tmp_path / "questions.csv"
    write_questions(target, rows)
    verified, total = read_questions(target, verified_only=True)
    assert total == 4 and len(verified) == 3
    everything, _ = read_questions(target, verified_only=False)
    assert len(everything) == 4
    assert verified[0].question == rows[0]["question"]
