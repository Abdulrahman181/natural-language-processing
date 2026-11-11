import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .corpus import Document
from .provenance import chunks_fingerprint, corpus_fingerprint
from .text import chunk_words


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray | None
    meta: dict


def chunk_corpus(documents: list[Document], size: int, overlap: int, min_words: int) -> list[Chunk]:
    chunks = []
    for doc_index, document in enumerate(documents, start=1):
        for chunk_index, piece in enumerate(chunk_words(document.text, size, overlap, min_words)):
            chunks.append(Chunk(f"d{doc_index:04d}-c{chunk_index:04d}", document.title, piece))
    return chunks


def build_index(
    documents: list[Document],
    out_dir: Path,
    size: int,
    overlap: int,
    min_words: int,
    embedder=None,
) -> Index:
    if size <= 0 or overlap < 0 or overlap >= size or min_words <= 0:
        raise ValueError("chunk size must be positive, overlap must be non-negative and smaller than size")
    if not documents:
        raise ValueError("Cannot build an index from an empty corpus")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_corpus(documents, size, overlap, min_words)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps({"id": chunk.id, "title": chunk.title, "text": chunk.text}, ensure_ascii=False) + "\n")
    embeddings = None
    if embedder is not None:
        embeddings = embedder.encode_passages([chunk.text for chunk in chunks])
        np.save(out_dir / "embeddings.npy", embeddings)
    meta = {
        "chunk_size_words": size,
        "overlap_words": overlap,
        "min_chunk_words": min_words,
        "documents": len(documents),
        "chunks": len(chunks),
        "dense": embedder is not None,
        "embedding_model": getattr(embedder, "model_name", None),
        "query_prefix": getattr(embedder, "query_prefix", ""),
        "passage_prefix": getattr(embedder, "passage_prefix", ""),
        "max_seq_length": getattr(getattr(embedder, "model", None), "max_seq_length", None),
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_fingerprint": corpus_fingerprint(documents),
        "chunks_fingerprint": chunks_fingerprint(chunks),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return Index(chunks, embeddings, meta)


def load_index(directory: Path) -> Index:
    chunks = []
    with (directory / "chunks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                chunks.append(Chunk(record["id"], record["title"], record["text"]))
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    embeddings_path = directory / "embeddings.npy"
    embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
    index = Index(chunks, embeddings, meta)
    expected = meta.get("chunks_fingerprint")
    if expected and expected != chunks_fingerprint(chunks):
        raise ValueError("Index chunks do not match chunks_fingerprint in meta.json; rebuild the index")
    if meta.get("dense") and embeddings is None:
        raise ValueError("Index metadata says dense=true but embeddings.npy is missing")
    if embeddings is not None and embeddings.shape[0] != len(chunks):
        raise ValueError("Embeddings row count does not match index chunks")
    return index
