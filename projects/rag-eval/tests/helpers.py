import hashlib
import json

import numpy as np

from rageval.corpus import Document
from rageval.text import tokenize

UNIQUE_WORDS = ["زيتون", "نحاس", "مرجان", "لؤلؤ", "حرير", "عنبر", "فيروز", "بلور"]
FILLER = "تتنوع الأنشطة الاقتصادية والاجتماعية في المنطقة عبر العصور المختلفة ويتبادل السكان الخبرات"


def fact_sentence(index: int, word: str) -> str:
    return f"يشتهر الإقليم رقم {index} بإنتاج {word} بكميات كبيرة منذ قرون عديدة"


def question_text(word: str) -> str:
    return f"أي إقليم اشتهر قديما بصناعة {word} بالتحديد"


def make_documents() -> list[Document]:
    documents = []
    for index, word in enumerate(UNIQUE_WORDS, start=1):
        body = " ".join([FILLER] * 12)
        documents.append(Document(f"وثيقة {index}", fact_sentence(index, word) + " " + body))
    return documents


class FakeEmbedder:
    model_name = "fake-hash-embedder"
    query_prefix = ""
    passage_prefix = ""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            slot = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[slot] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        passage = prompt.split("<<<", 1)[1].rsplit(">>>", 1)[0]
        for index, word in enumerate(UNIQUE_WORDS, start=1):
            if f"بإنتاج {word}" in passage:
                return json.dumps(
                    {"question": question_text(word), "evidence": fact_sentence(index, word)},
                    ensure_ascii=False,
                )
        return "no json here"
