import importlib.util
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .index import Chunk
from .ranking import reciprocal_rank_fusion, top_indices
from .text import tokenize


class Bm25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.texts = [chunk.text for chunk in chunks]
        self.engine = BM25([tokenize(text) for text in self.texts])

    def rank(self, query: str, depth: int) -> list[int]:
        return self.engine.top_k(tokenize(query), depth)

    def search(self, query: str, k: int) -> list[str]:
        return [self.texts[index] for index in self.rank(query, k)]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder):
        self.texts = [chunk.text for chunk in chunks]
        self.embeddings = embeddings
        self.embedder = embedder

    def rank(self, query: str, depth: int) -> list[int]:
        vector = self.embedder.encode_queries([query])[0]
        return top_indices(self.embeddings @ vector, depth)

    def search(self, query: str, k: int) -> list[str]:
        return [self.texts[index] for index in self.rank(query, k)]


class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(self, sparse: Bm25Retriever, dense: DenseRetriever, pool: int = 50, rrf_k: int = 60):
        self.sparse = sparse
        self.dense = dense
        self.pool = pool
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[str]:
        fused = reciprocal_rank_fusion(
            [self.sparse.rank(query, self.pool), self.dense.rank(query, self.pool)],
            k=self.rrf_k,
            top=k,
        )
        return [self.sparse.texts[index] for index in fused]


class AdapterRetriever:
    def __init__(self, name: str, search_function):
        self.name = name
        self._search = search_function

    def search(self, query: str, k: int) -> list[str]:
        return list(self._search(query, k))[:k]


def load_adapter(path: Path) -> AdapterRetriever:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "search"):
        raise ValueError(f"Adapter {path} must define search(query, k) -> list[str]")
    return AdapterRetriever(getattr(module, "NAME", path.stem), module.search)
