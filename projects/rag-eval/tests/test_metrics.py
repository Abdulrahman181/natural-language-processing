import math

import numpy as np
import pytest

from rageval.metrics import (
    bootstrap_ci,
    first_relevant_rank,
    hits,
    latency_summary,
    ndcg_values,
    reciprocal_ranks,
)


def test_first_relevant_rank_returns_position_or_none():
    texts = ["alpha beta", "gamma delta epsilon", "zeta"]
    assert first_relevant_rank(texts, "gamma delta") == 2
    assert first_relevant_rank(texts, "omega") is None
    assert first_relevant_rank([], "gamma") is None


def test_hit_and_rank_metrics():
    ranks = [1, 3, None, 12]
    assert hits(ranks, 1).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert hits(ranks, 5).mean() == 0.5
    assert reciprocal_ranks(ranks, 10).tolist() == pytest.approx([1.0, 1 / 3, 0.0, 0.0])
    ndcg = ndcg_values(ranks, 10)
    assert ndcg[0] == pytest.approx(1.0)
    assert ndcg[1] == pytest.approx(1.0 / math.log2(4))
    assert ndcg[2] == 0.0
    assert ndcg[3] == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = np.array([1.0] * 30 + [0.0] * 70)
    low, high = bootstrap_ci(values)
    assert low < 0.3 < high
    assert (low, high) == bootstrap_ci(values)
    assert bootstrap_ci(np.array([])) == (0.0, 0.0)
    assert bootstrap_ci(np.ones(10)) == (1.0, 1.0)


def test_latency_summary():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["p50"] == pytest.approx(3.0)
    assert summary["p95"] == pytest.approx(4.8)
    assert latency_summary([]) == {"mean": 0.0, "p50": 0.0, "p95": 0.0}
