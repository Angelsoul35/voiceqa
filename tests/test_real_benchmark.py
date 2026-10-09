"""Functional tests for the grader-validation benchmark.

These prove two things: the discrimination metrics are implemented correctly
(perfectly separable -> AUC 1.0, inverted -> 0.0, agreement with scikit-learn),
and VoiceQA's own grader really does separate correct from incorrect answers on
the real data while both negative controls collapse to chance. Each test fails if
the property it checks is broken.
"""

from __future__ import annotations

import random

import pytest
from sklearn.metrics import roc_auc_score

from voiceqa import real_benchmark as rb
from voiceqa.real_data import build_examples, load_records

# A modest but real slice keeps the suite fast while staying statistically clear.
_LIMIT = 120


# ---------------------------------------------------------------------------
# Metric correctness: the measurement must be right before any verdict means
# anything.
# ---------------------------------------------------------------------------


def test_roc_auc_perfect_separation_is_one():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1, 1]
    assert rb.roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_inverted_is_zero():
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.0]
    labels = [0, 0, 0, 1, 1, 1]
    assert rb.roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_all_ties_is_half():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert rb.roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_matches_sklearn_on_random_data():
    rng = random.Random(7)
    scores = [round(rng.random(), 3) for _ in range(200)]  # ties included
    labels = [rng.randint(0, 1) for _ in range(200)]
    assert rb.roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-9)


def test_roc_auc_requires_both_classes():
    with pytest.raises(ValueError):
        rb.roc_auc([0.1, 0.2], [1, 1])


def test_operating_point_metrics_are_correct():
    #            TP    FP    FN     TN
    decisions = [True, True, False, False]
    labels = [1, 0, 1, 0]
    m = rb.operating_point_metrics(decisions, labels)
    assert m["accuracy"] == pytest.approx(0.5)  # 1 TP + 1 TN of 4
    assert m["precision"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FP)
    assert m["recall"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FN)


def test_spearman_monotonic_and_antitonic():
    assert rb.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert rb.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# The grader actually discriminates on real held-out data.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored():
    examples = build_examples(load_records("default", limit=_LIMIT))
    return examples, rb.score_examples(examples)


def test_grader_discriminates_above_chance_on_heldout(scored):
    _, s = scored
    labels = s.labels
    held = list(range(len(labels) // 2, len(labels)))
    auc = rb.roc_auc([s.scores[i] for i in held], [labels[i] for i in held])
    assert auc > 0.75  # well above the 0.5 chance line

    pos = [s.scores[i] for i in range(len(labels)) if labels[i] == 1]
    neg = [s.scores[i] for i in range(len(labels)) if labels[i] == 0]
    # Correct answers score clearly higher than incorrect ones on average.
    assert sum(pos) / len(pos) - sum(neg) / len(neg) > 0.3


def test_shuffled_label_control_collapses(scored):
    _, s = scored
    real = rb.roc_auc(s.scores, s.labels)
    aucs = rb.shuffled_label_auc(s.scores, s.labels, trials=12, seed=1)
    mean = sum(aucs) / len(aucs)
    assert 0.4 < mean < 0.6  # chance
    assert real - mean > 0.25  # real signal is far above the shuffled control


def test_random_reference_control_collapses(scored):
    examples, s = scored
    real = rb.roc_auc(s.scores, s.labels)
    aucs = rb.random_reference_auc(examples, trials=8, seed=1)
    mean = sum(aucs) / len(aucs)
    assert 0.4 < mean < 0.6  # grounding an answer in an unrelated passage is chance
    assert real - mean > 0.25


def test_run_reports_a_validated_verdict():
    report = rb.run(config="default", limit=_LIMIT, trials=10)
    assert report.auc > 0.9
    assert report.n_pos == _LIMIT
    assert 0.4 < report.shuffle_mean < 0.6
    assert 0.4 < report.random_ref_mean < 0.6
    assert report.validated is True
    text = rb.format_report(report)
    assert "VALIDATED" in text and "ROC-AUC" in text
