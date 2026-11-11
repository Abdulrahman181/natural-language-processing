import re
import unicodedata
from functools import lru_cache

_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_LETTER_MAP = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0671": "\u0627",
        "\u0649": "\u064a",
        "\u0629": "\u0647",
        "\u0640": None,
    }
)
_DIGIT_MAP = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9",
    "01234567890123456789",
)
_PREFIXES = (
    "\u0648\u0627\u0644",
    "\u0628\u0627\u0644",
    "\u0643\u0627\u0644",
    "\u0641\u0627\u0644",
    "\u0644\u0644",
    "\u0627\u0644",
)
_MIN_STEM_LENGTH = 3


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_LETTER_MAP).translate(_DIGIT_MAP)
    return text.lower()


def raw_tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def strip_prefix(token: str) -> str:
    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= _MIN_STEM_LENGTH:
            return token[len(prefix):]
    return token


def tokenize(text: str) -> list[str]:
    return [strip_prefix(token) for token in raw_tokens(text)]


def word_count(text: str) -> int:
    return len(text.split())


def chunk_words(text: str, size: int, overlap: int, min_words: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= min_words:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


@lru_cache(maxsize=8192)
def _haystack(text: str) -> str:
    return " " + " ".join(raw_tokens(text)) + " "


def contains_evidence(chunk_text: str, evidence: str) -> bool:
    tokens = raw_tokens(evidence)
    if not tokens:
        return False
    needle = " " + " ".join(tokens) + " "
    return needle in _haystack(chunk_text)
