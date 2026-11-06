from collections import Counter

import numpy as np

from .parsing import INVALID


def correct_vector(gold: list[str], pred: list[str]) -> np.ndarray:
    return np.array([1.0 if g == p else 0.0 for g, p in zip(gold, pred)])


def accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    return float(correct_vector(gold, pred).mean())


def invalid_rate(pred: list[str]) -> float:
    if not pred:
        return 0.0
    return sum(1 for item in pred if item == INVALID) / len(pred)


def per_class_scores(gold: list[str], pred: list[str]) -> list[dict]:
    gold_count = Counter(gold)
    pred_count = Counter(pred)
    true_positive: Counter = Counter()
    for g, p in zip(gold, pred):
        if g == p:
            true_positive[g] += 1
    rows = []
    for label in sorted(gold_count):
        hits = true_positive[label]
        precision = hits / pred_count[label] if pred_count[label] else 0.0
        recall = hits / gold_count[label]
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "support": gold_count[label],
                "predicted": pred_count[label],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def macro_f1(gold: list[str], pred: list[str]) -> float:
    rows = per_class_scores(gold, pred)
    return float(np.mean([row["f1"] for row in rows])) if rows else 0.0


def bootstrap_ci(values, resamples: int = 2000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    means = data[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def paired_bootstrap_diff(correct_a, correct_b, resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    difference = np.asarray(correct_a, dtype=np.float64) - np.asarray(correct_b, dtype=np.float64)
    low, high = bootstrap_ci(difference, resamples=resamples, seed=seed)
    return float(difference.mean()) if difference.size else 0.0, low, high


def top_confusions(gold: list[str], pred: list[str], limit: int = 15) -> list[dict]:
    counts = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return [{"gold": g, "pred": p, "count": count} for (g, p), count in counts.most_common(limit)]


def select(values: list, mask: list[bool]) -> list:
    return [value for value, keep in zip(values, mask) if keep]
