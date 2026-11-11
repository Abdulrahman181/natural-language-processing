import math

import numpy as np

from .text import contains_evidence


def first_relevant_rank(texts: list[str], evidence: str) -> int | None:
    for position, text in enumerate(texts, start=1):
        if contains_evidence(text, evidence):
            return position
    return None


def hits(ranks: list[int | None], k: int) -> np.ndarray:
    return np.array([1.0 if rank is not None and rank <= k else 0.0 for rank in ranks])


def reciprocal_ranks(ranks: list[int | None], k: int) -> np.ndarray:
    return np.array([1.0 / rank if rank is not None and rank <= k else 0.0 for rank in ranks])


def ndcg_values(ranks: list[int | None], k: int) -> np.ndarray:
    return np.array([1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0 for rank in ranks])


def bootstrap_ci(values: np.ndarray, resamples: int = 1000, seed: int = 0, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    lower = (1.0 - confidence) / 2.0 * 100.0
    return (float(np.percentile(means, lower)), float(np.percentile(means, 100.0 - lower)))


def latency_summary(milliseconds: list[float]) -> dict:
    data = np.asarray(milliseconds, dtype=np.float64)
    if data.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
    }
