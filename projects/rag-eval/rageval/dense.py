import numpy as np


class SentenceEmbedder:
    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        max_seq_length: int | None = None,
        batch_size: int = 16,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + text for text in texts], show_progress=True)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + text for text in texts], show_progress=False)
