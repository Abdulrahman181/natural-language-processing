import numpy as np


def top_indices(scores: np.ndarray, k: int) -> list[int]:
    size = scores.shape[0]
    if size == 0 or k <= 0:
        return []
    k = min(k, size)
    candidates = np.argpartition(-scores, k - 1)[:k]
    ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
    return ordered.tolist()


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60, top: int = 10) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [item for item, _ in ordered[:top]]
