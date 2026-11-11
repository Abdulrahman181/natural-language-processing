import pytest

from rageval.text import chunk_words, contains_evidence, normalize, strip_prefix, tokenize


def test_normalize_removes_diacritics_and_unifies_letters():
    assert normalize("أَحْمَد") == normalize("احمد")
    assert normalize("مدرسة") == normalize("مدرسه")
    assert normalize("إسلام") == normalize("اسلام")
    assert normalize("على") == normalize("علي")


def test_normalize_maps_indic_digits_and_removes_tatweel():
    assert normalize("٢٠٢٥") == "2025"
    assert normalize("۱۲۳") == "123"
    assert normalize("العـــربية") == normalize("العربية")


def test_strip_prefix_requires_a_long_enough_stem():
    assert strip_prefix("الكتاب") == "كتاب"
    assert strip_prefix("الله") == "الله"


def test_tokenize_strips_prefixes_after_normalization():
    assert tokenize("والمدينة") == ["مدينه"]
    assert tokenize("للمدرسة") == ["مدرسه"]
    assert tokenize("الأنشطة") == ["انشطه"]


def test_tokenize_handles_punctuation_and_latin_text():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]
    assert tokenize("") == []


def test_chunk_words_overlap_and_coverage():
    text = " ".join(str(number) for number in range(10))
    assert chunk_words(text, size=4, overlap=1, min_words=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_chunk_words_drops_short_tail():
    text = " ".join(str(number) for number in range(10))
    assert chunk_words(text, size=4, overlap=0, min_words=3) == ["0 1 2 3", "4 5 6 7"]


def test_chunk_words_validates_arguments_and_empty_text():
    with pytest.raises(ValueError):
        chunk_words("a b", 2, 2, 1)
    with pytest.raises(ValueError):
        chunk_words("a b", 0, 0, 1)
    assert chunk_words("", 4, 1, 1) == []


def test_contains_evidence_ignores_diacritics_and_punctuation():
    chunk = "قال المؤلِّف: إن الشمسَ، كما نعلم، مصدر الطاقة الرئيسي."
    assert contains_evidence(chunk, "الشمس كما نعلم مصدر الطاقة")


def test_contains_evidence_respects_token_boundaries():
    assert not contains_evidence("تتحدث الجملة عن الكتابة", "الكتاب")
    assert not contains_evidence("abc def", "")
    assert not contains_evidence("abc def", "def abc")
