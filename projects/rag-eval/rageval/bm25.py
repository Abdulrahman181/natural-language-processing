import math
from collections import Counter

import numpy as np

from .ranking import top_indices


class BM25:
    def __init__(self, tokenized_documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.size = len(tokenized_documents)
        self.lengths = np.array([len(tokens) for tokens in tokenized_documents], dtype=np.float64)
        mean_length = float(self.lengths.mean()) if self.size else 0.0
        self.avg_length = max(mean_length, 1e-9)
        collected: dict[str, tuple[list[int], list[int]]] = {}
        for index, tokens in enumerate(tokenized_documents):
            for term, count in Counter(tokens).items():
                indices, counts = collected.setdefault(term, ([], []))
                indices.append(index)
                counts.append(count)
        self.postings = {
            term: (np.array(indices, dtype=np.int64), np.array(counts, dtype=np.float64))
            for term, (indices, counts) in collected.items()
        }
        self.idf = {
            term: math.log(1.0 + (self.size - len(indices) + 0.5) / (len(indices) + 0.5))
            for term, (indices, _) in self.postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        result = np.zeros(self.size, dtype=np.float64)
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if posting is None:
                continue
            indices, counts = posting
            length_norm = self.k1 * (1.0 - self.b + self.b * self.lengths[indices] / self.avg_length)
            result[indices] += self.idf[term] * counts * (self.k1 + 1.0) / (counts + length_norm)
        return result

    def top_k(self, query_tokens: list[str], k: int) -> list[int]:
        scores = self.scores(query_tokens)
        return [index for index in top_indices(scores, k) if scores[index] > 0.0]
