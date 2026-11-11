import numpy as np

from rageval.bm25 import BM25
from rageval.ranking import reciprocal_rank_fusion, top_indices


def test_bm25_ranks_matching_documents_first():
    bm25 = BM25([["apple", "banana"], ["banana", "cherry"], ["cherry", "date", "date"]])
    assert bm25.top_k(["apple"], 3) == [0]
    assert bm25.top_k(["date"], 3) == [2]
    assert bm25.top_k(["banana", "cherry"], 3)[0] == 1


def test_bm25_idf_stays_positive_for_terms_in_every_document():
    bm25 = BM25([["common", "a"], ["common", "b"], ["common", "c"], ["common", "d"]])
    assert all(value >= 0 for value in bm25.idf.values())
    assert bm25.scores(["common"]).min() > 0


def test_bm25_unknown_terms_and_empty_corpus():
    assert BM25([["a"]]).top_k(["zzz"], 5) == []
    empty = BM25([])
    assert empty.scores(["a"]).shape == (0,)
    assert empty.top_k(["a"], 3) == []


def test_bm25_length_normalization_prefers_the_shorter_document():
    scores = BM25([["x"], ["x"] + ["y"] * 20]).scores(["x"])
    assert scores[0] > scores[1]


def test_bm25_repeated_query_terms_do_not_change_scores():
    bm25 = BM25([["a", "b"], ["b", "c"]])
    assert np.allclose(bm25.scores(["a"]), bm25.scores(["a", "a", "a"]))


def test_top_indices_orders_by_score_and_handles_small_inputs():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_indices(scores, 2) == [1, 3]
    assert top_indices(scores, 10) == [1, 3, 2, 0]
    assert top_indices(np.array([]), 3) == []
    assert top_indices(scores, 0) == []


def test_reciprocal_rank_fusion_rewards_agreement():
    assert reciprocal_rank_fusion([[1, 2], [2, 3]], k=60, top=3) == [2, 1, 3]


def test_reciprocal_rank_fusion_is_deterministic_on_ties_and_respects_top():
    assert reciprocal_rank_fusion([[5], [4]], top=2) == [4, 5]
    assert reciprocal_rank_fusion([[1, 2, 3]], top=2) == [1, 2]
