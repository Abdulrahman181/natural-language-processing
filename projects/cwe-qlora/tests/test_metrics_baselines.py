import numpy as np
import pytest

from cwe_qlora.baselines import fit_tfidf_svm, majority_label
from cwe_qlora.metrics import (
    accuracy,
    bootstrap_ci,
    correct_vector,
    invalid_rate,
    macro_f1,
    paired_bootstrap_diff,
    per_class_scores,
    select,
    top_confusions,
)
from cwe_qlora.parsing import INVALID
from cwe_qlora.similarity import max_train_similarity, near_duplicate_mask

from helpers import make_rows


def test_accuracy_and_invalid_rate():
    gold = ["a", "b", "c", "a"]
    pred = ["a", "b", INVALID, "c"]
    assert accuracy(gold, pred) == 0.5
    assert invalid_rate(pred) == 0.25
    assert accuracy([], []) == 0.0 and invalid_rate([]) == 0.0


def test_macro_f1_matches_hand_computation():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "b"]
    rows = {row["label"]: row for row in per_class_scores(gold, pred)}
    assert rows["a"]["precision"] == 1.0 and rows["a"]["recall"] == 0.5
    assert rows["b"]["precision"] == pytest.approx(2 / 3) and rows["b"]["recall"] == 1.0
    assert macro_f1(gold, pred) == pytest.approx((2 / 3 + 0.8) / 2)


def test_invalid_predictions_count_as_wrong_without_adding_a_class():
    gold = ["a", "b"]
    pred = [INVALID, "b"]
    assert [row["label"] for row in per_class_scores(gold, pred)] == ["a", "b"]
    assert macro_f1(gold, pred) == pytest.approx(0.5)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = np.array([1.0] * 80 + [0.0] * 20)
    low, high = bootstrap_ci(values)
    assert low < 0.8 < high
    assert (low, high) == bootstrap_ci(values)
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_paired_bootstrap_diff_detects_a_clear_gap():
    gold = ["a"] * 200
    strong = ["a"] * 190 + ["b"] * 10
    weak = ["a"] * 100 + ["b"] * 100
    mean, low, high = paired_bootstrap_diff(correct_vector(gold, strong), correct_vector(gold, weak))
    assert mean == pytest.approx(0.45)
    assert low > 0.3 and high < 0.6
    flipped, _, _ = paired_bootstrap_diff(correct_vector(gold, weak), correct_vector(gold, strong))
    assert flipped == pytest.approx(-0.45)


def test_top_confusions_and_select():
    gold = ["a", "a", "b", "c"]
    pred = ["b", "b", "b", "a"]
    assert top_confusions(gold, pred)[0] == {"gold": "a", "pred": "b", "count": 2}
    assert select([1, 2, 3], [True, False, True]) == [1, 3]


def test_tfidf_svm_beats_majority_on_templated_data():
    train = make_rows(400, 1, 0)
    test = make_rows(100, 2, 10000)
    model = fit_tfidf_svm([row["description"] for row in train], [row["cwe_id"] for row in train])
    gold = [row["cwe_id"] for row in test]
    predicted = list(model.predict([row["description"] for row in test]))
    assert accuracy(gold, predicted) > 0.95
    prior = majority_label([row["cwe_id"] for row in train])
    assert accuracy(gold, [prior] * len(gold)) < 0.5


def test_max_similarity_flags_duplicates_and_not_unrelated_text():
    train = [f"vendor{n} product{n} flaw{n} allows attackers to misbehave" for n in range(30)]
    similarities = max_train_similarity([train[0], "completely different astronomy telescope galaxy lens"], train)
    assert similarities[0] > 0.99
    assert similarities[1] < 0.2


def test_max_similarity_is_chunk_size_independent():
    train = [f"vendor{n} product{n} flaw{n} component{n % 7} issue" for n in range(40)]
    queries = [train[3], train[17] + " extra{n}".format(n=1), "unrelated words entirely"]
    assert np.allclose(max_train_similarity(queries, train, chunk=1), max_train_similarity(queries, train, chunk=100))


def test_near_duplicate_mask_uses_inclusive_threshold():
    assert near_duplicate_mask([0.9, 0.5, 0.85], 0.85) == [True, False, True]
