import pytest

from rageval.corpus import (
    Document,
    clean_extract,
    export_text_files,
    fetch_wikipedia,
    load_corpus,
    load_titles,
    save_corpus,
)

LONG_TEXT = "كلمة " * 800


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, params, headers, timeout):
        title = params["titles"]
        self.calls.append((title, headers))
        queue = self.responses[title]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def page(title, extract):
    return FakeResponse(200, {"query": {"pages": [{"title": title, "extract": extract}]}})


def missing():
    return FakeResponse(200, {"query": {"pages": [{"title": "x", "missing": True}]}})


def no_sleep(_seconds):
    return None


def test_fetch_collects_documents_and_reports_skips():
    session = FakeSession(
        {
            "A": [page("A", LONG_TEXT)],
            "B": [missing()],
            "C": [page("C", "قصير")],
            "D": [page("A", LONG_TEXT)],
        }
    )
    documents, skipped = fetch_wikipedia(
        ["A", "B", "C", "D"], user_agent="test-agent", session=session, sleep=no_sleep
    )
    assert [document.title for document in documents] == ["A"]
    assert dict(skipped) == {"B": "not_found", "C": "too_short", "D": "duplicate_redirect"}
    assert all(headers["User-Agent"] == "test-agent" for _, headers in session.calls)


def test_fetch_retries_transient_errors():
    delays = []
    session = FakeSession({"A": [FakeResponse(503), FakeResponse(429), page("A", LONG_TEXT)]})
    documents, skipped = fetch_wikipedia(["A"], session=session, sleep=delays.append, pause=0)
    assert len(documents) == 1 and skipped == []
    assert delays[:2] == [1, 2]


def test_fetch_raises_after_exhausting_retries_or_on_forbidden():
    with pytest.raises(RuntimeError, match="503"):
        fetch_wikipedia(["A"], session=FakeSession({"A": [FakeResponse(503)]}), sleep=no_sleep)
    session = FakeSession({"A": [FakeResponse(403)]})
    with pytest.raises(RuntimeError, match="403"):
        fetch_wikipedia(["A"], session=session, sleep=no_sleep)
    assert len(session.calls) == 1


def test_clean_extract_cuts_reference_sections():
    assert clean_extract("مقدمة طويلة\n\nمراجع\nالمصدر الأول") == "مقدمة طويلة"
    assert clean_extract("مقدمة\n\n== وصلات خارجية ==\nرابط") == "مقدمة"
    assert clean_extract("فقرة\n\n\n\nفقرة ثانية") == "فقرة\n\nفقرة ثانية"


def test_corpus_roundtrip_keeps_arabic_readable(tmp_path):
    documents = [Document("مصر", "نص عربي"), Document("القاهرة", "نص آخر")]
    target = tmp_path / "nested" / "corpus.jsonl"
    save_corpus(documents, target)
    assert "مصر" in target.read_text(encoding="utf-8")
    assert load_corpus(target) == documents


def test_load_titles_skips_blank_lines_comments_and_bom(tmp_path):
    target = tmp_path / "titles.txt"
    target.write_text("﻿# comment\nالحاسوب\n\nمصر\n", encoding="utf-8")
    assert load_titles(target) == ["الحاسوب", "مصر"]


def test_export_text_files_sanitizes_names(tmp_path):
    documents = [Document('a/b: c?', "محتوى"), Document("مصر", "محتوى ثان")]
    written = export_text_files(documents, tmp_path / "txt")
    assert len(written) == 2
    assert all(not any(char in path.name for char in '/:?') for path in written)
    assert written[1].read_text(encoding="utf-8") == "محتوى ثان"
