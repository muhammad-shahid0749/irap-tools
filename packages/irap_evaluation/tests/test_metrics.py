import dataclasses as dc

import numpy as np
import pytest

from irap_data.metadata import IGNORE_LABEL_INDEX
from irap_evaluation.metric_names import (
    MANUAL_REVIEW_PROPORTION_METRIC,
    add_per_attribute_metric_names,
)
from irap_evaluation.metrics import (
    NUM_CONFIDENCE_BINS,
    ClassificationStatistics,
    ConfidenceKind,
    compute_classification_metrics,
    compute_classification_statistics,
    compute_manual_review_proportion,
    compute_multi_attribute_metrics,
    compute_review_curve,
    select_metric_attributes,
    sum_statistics,
)


def test_confusion_matrix_metrics_by_hand():
    # Class 2 is neither in the ground truth nor predicted, so the means leave it out.
    confusion_matrix = np.array([[3, 1, 0], [2, 4, 0], [0, 0, 0]])
    metrics = compute_classification_metrics(
        ClassificationStatistics(confusion_matrix), ["P", "R", "mF1", "mP", "A", "n", "MCC"])
    np.testing.assert_allclose(metrics["P"], [3 / 5, 4 / 5, 0])
    np.testing.assert_allclose(metrics["R"], [3 / 4, 4 / 6, 0])
    f1 = [2 * 0.6 * 0.75 / 1.35, 2 * 0.8 * (4 / 6) / (0.8 + 4 / 6)]
    assert metrics["mF1"] == pytest.approx(np.mean(f1))
    assert metrics["mP"] == pytest.approx(0.7)
    assert metrics["A"] == pytest.approx(0.7)
    assert metrics["n"] == 10
    # MCC = (c n - sum_k p_k t_k) / sqrt((n^2 - sum p^2) (n^2 - sum t^2))
    assert metrics["MCC"] == pytest.approx((7 * 10 - (5 * 4 + 5 * 6))
                                           / np.sqrt((100 - 50) * (100 - 52)))
    all_classes = compute_classification_metrics(ClassificationStatistics(confusion_matrix),
                                                 ["mP"], ignore_missing_classes=False)
    assert all_classes["mP"] == pytest.approx(1.4 / 3)


def test_support_restricted_metrics():
    confusion_matrix = np.array([[6, 0, 0], [0, 4, 1], [0, 0, 1]])
    metrics = compute_classification_metrics(ClassificationStatistics(confusion_matrix),
                                             ["F1", "mF1_supp5", "nc_supp5", "mF1_supp9"])
    assert metrics["nc_supp5"] == 2
    assert metrics["mF1_supp5"] == pytest.approx(metrics["F1"][:2].mean())
    assert np.isnan(metrics["mF1_supp9"])


def test_statistics_ignore_unlabeled_examples_and_sum_over_groups():
    rng = np.random.default_rng(0)
    targets = rng.integers(-1, 3, size=200)
    probs = rng.dirichlet(np.ones(3), size=200).astype(np.float32)
    groups = rng.integers(0, 7, size=200)
    grouped = compute_classification_statistics(targets, probs.argmax(1), 3, probs=probs,
                                                group_indices=groups, num_groups=7)
    whole = compute_classification_statistics(targets, probs.argmax(1), 3, probs=probs)
    summed = sum_statistics(grouped)
    np.testing.assert_array_equal(summed.confusion_matrix, whole.confusion_matrix)
    np.testing.assert_allclose(summed.nll_sums, whole.nll_sums)
    np.testing.assert_array_equal(summed.confidence_histogram, whole.confidence_histogram)
    assert whole.confusion_matrix.sum() == (targets != IGNORE_LABEL_INDEX).sum()
    # Every labeled example is a correct or a missed example of its class.
    np.testing.assert_array_equal(
        whole.confidence_histogram[[ConfidenceKind.CORRECT, ConfidenceKind.MISSED]].sum((0, 2)),
        whole.support)
    # Weights of 2 for every group double every count.
    doubled = sum_statistics(grouped, np.full((1, 7), 2))
    np.testing.assert_array_equal(doubled.confusion_matrix[0], 2 * whole.confusion_matrix)
    np.testing.assert_array_equal(doubled.confidence_histogram[0],
                                  2 * whole.confidence_histogram)


def test_weighted_sums_leave_out_infinite_sums_of_undrawn_groups():
    # Group 1 gives its label probability 0, so its NLL sum is infinite.
    targets, groups = np.array([0, 1, 0]), np.array([0, 1, 2])
    probs = np.array([[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]], dtype=np.float32)
    grouped = compute_classification_statistics(targets, probs.argmax(1), 2, probs=probs,
                                                group_indices=groups, num_groups=3)
    weighted = sum_statistics(grouped, np.array([[1, 0, 2], [0, 3, 0]]))
    np.testing.assert_allclose(weighted.nll_sums[0], [np.log(2) - 2 * np.log(0.25), 0])
    np.testing.assert_equal(weighted.nll_sums[1], [0, np.inf])


def test_probabilistic_metrics():
    targets = np.array([0, 0, 1])
    probs = np.array([[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]], dtype=np.float32)
    metrics = compute_classification_metrics(
        compute_classification_statistics(targets, probs.argmax(1), 2, probs=probs),
        ["NLL", "Brier", "mNLL"])
    nll = [np.log(2), 0, -np.log(0.75)]
    assert metrics["NLL"] == pytest.approx(np.mean(nll))
    assert metrics["mNLL"] == pytest.approx(np.mean([np.mean(nll[:2]), nll[2]]))
    assert metrics["Brier"] == pytest.approx(np.mean([0.5, 0, 2 * 0.25 ** 2]))


def test_hard_statistics_refuse_metrics_of_distributions():
    stats = compute_classification_statistics(np.array([0, 1]), np.array([0, 0]), 2)
    for name in ["NLL", MANUAL_REVIEW_PROPORTION_METRIC]:
        with pytest.raises(ValueError, match="hard"):
            compute_classification_metrics(stats, [name])
    with pytest.raises(ValueError, match="hard"):
        compute_review_curve(stats)


def _compute_statistics_of_confidences(targets, predicted_indices, confidences, num_classes=2):
    """Computes the statistics of predictions with the given confidences."""
    targets, predicted_indices = np.array(targets), np.array(predicted_indices)
    probs = np.zeros((len(targets), num_classes))
    probs[np.arange(len(targets)), predicted_indices] = confidences
    return compute_classification_statistics(targets, predicted_indices, num_classes,
                                             probs=probs)


def _compute_manual_review_proportion_metric(statistics):
    name = MANUAL_REVIEW_PROPORTION_METRIC
    return compute_classification_metrics(statistics, [name])[name]


def test_review_curve_by_hand():
    # Both examples of class 1 are predicted as class 0, with the second and the fourth lowest
    # confidence of the 10 examples.
    statistics = _compute_statistics_of_confidences(
        targets=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0], predicted_indices=[0] * 10,
        confidences=[0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.99])
    curve = compute_review_curve(statistics)
    assert curve.review_proportions.shape == (NUM_CONFIDENCE_BINS + 1,)
    # The points of the curve, without those of empty bins.
    proportions, indices = np.unique(curve.review_proportions, return_index=True)
    np.testing.assert_allclose(proportions, np.arange(11) / 10)
    # With one example of class 1 corrected, its F1 is 2 / 3, and that of class 0 is 16 / 17.
    f1_after_one_correction = (2 / 3 + 16 / 17) / 2
    np.testing.assert_allclose(
        curve.macro_f1[indices],
        [(16 / 18) / 2] * 2 + [f1_after_one_correction] * 2 + [1] * 7)
    np.testing.assert_allclose(curve.accuracy[indices], [0.8] * 2 + [0.9] * 2 + [1] * 7)
    # mF1 reaches 0.9 between the reviewed proportions 0.3 and 0.4.
    expected = 0.3 + 0.1 * (0.9 - f1_after_one_correction) / (1 - f1_after_one_correction)
    assert compute_manual_review_proportion(curve, 0.9) == pytest.approx(expected)
    assert _compute_manual_review_proportion_metric(statistics) == pytest.approx(expected)
    assert compute_manual_review_proportion(curve, 0.4) == 0
    assert compute_manual_review_proportion(curve, 1) == pytest.approx(0.4)


def test_manual_review_proportion_of_equal_confidences_is_that_of_a_review_in_random_order():
    # 2 of 10 examples are wrong: mF1 is ((12 / 14) + (4 / 6)) / 2 before the review.
    statistics = _compute_statistics_of_confidences(
        targets=[0] * 6 + [1] * 4, predicted_indices=[0] * 6 + [1, 1, 0, 0],
        confidences=[1.0] * 10)
    macro_f1 = (12 / 14 + 4 / 6) / 2
    metrics = compute_classification_metrics(statistics,
                                             ["mF1", MANUAL_REVIEW_PROPORTION_METRIC])
    assert metrics["mF1"] == pytest.approx(macro_f1)
    assert metrics[MANUAL_REVIEW_PROPORTION_METRIC] == pytest.approx(
        (0.9 - macro_f1) / (1 - macro_f1))


def test_review_curve_ends_at_1_with_a_class_that_is_only_predicted():
    statistics = _compute_statistics_of_confidences(
        targets=[0, 0, 0, 0], predicted_indices=[0, 1, 0, 2], confidences=[0.9, 0.5, 0.8, 0.6],
        num_classes=3)
    curve = compute_review_curve(statistics)
    # Class 0 has F1 2 * 2 / (2 * 2 + 2), and the predicted classes 1 and 2 have F1 0.
    assert curve.macro_f1[0] == pytest.approx((4 / 6) / 3)
    assert curve.macro_f1[-1] == 1
    assert (np.diff(curve.macro_f1) >= 0).all()
    # All classes count without `ignore_missing_classes`, so mF1 does not reach 0.9.
    all_classes = compute_review_curve(statistics, ignore_missing_classes=False)
    assert all_classes.macro_f1[-1] == pytest.approx(1 / 3)
    assert np.isnan(compute_manual_review_proportion(all_classes, 0.9))


def test_manual_review_proportion_without_review_and_without_examples():
    correct = _compute_statistics_of_confidences([0, 1, 1], [0, 1, 1], [0.7, 0.8, 0.9])
    assert _compute_manual_review_proportion_metric(correct) == 0
    unlabeled = _compute_statistics_of_confidences([IGNORE_LABEL_INDEX] * 2, [0, 1], [0.7, 0.8])
    assert np.isnan(_compute_manual_review_proportion_metric(unlabeled))


def test_batched_review_proportions_equal_separate_computations():
    rng = np.random.default_rng(2)
    targets = rng.integers(0, 3, size=300)
    probs = rng.dirichlet(np.ones(3), size=300).astype(np.float32)
    groups = rng.integers(0, 9, size=300)
    grouped = compute_classification_statistics(targets, probs.argmax(1), 3, probs=probs,
                                                group_indices=groups, num_groups=9)
    weights = rng.multinomial(9, np.full(9, 1 / 9), size=5)
    batched = _compute_manual_review_proportion_metric(sum_statistics(grouped, weights))
    assert batched.shape == (5,)
    for i, group_weights in enumerate(weights):
        single = _compute_manual_review_proportion_metric(sum_statistics(grouped, group_weights))
        assert batched[i] == pytest.approx(single)
    assert 0 < batched.min() and batched.max() < 1


def test_attribute_average_leaves_out_undefined_values():
    defined = ClassificationStatistics(np.array([[3, 1], [1, 3]]))
    # A single ground-truth and predicted class makes MCC undefined (0/0).
    undefined = ClassificationStatistics(np.array([[4, 0], [0, 0]]))
    metrics = compute_multi_attribute_metrics({"x": defined, "y": undefined}, ["aMCC", "MCC"])
    assert np.isnan(metrics.per_attribute["MCC"]["y"])
    assert metrics.averages["aMCC"] == pytest.approx(metrics.per_attribute["MCC"]["x"])
    with pytest.raises(ValueError, match="per-class"):
        compute_multi_attribute_metrics({"x": defined}, ["aF1"])


def _to_float_values(metrics):
    return type(metrics)(
        averages={n: float(v) for n, v in metrics.averages.items()},
        per_attribute={n: {a: float(v) for a, v in values.items()}
                       for n, values in metrics.per_attribute.items()})


def test_select_metric_attributes_equals_computing_on_the_subset():
    # x has an undefined MCC (one class), y an infinite NLL (label probability 0).
    targets = {"x": np.array([0, 0, 0]), "y": np.array([0, 1, 1]), "z": np.array([1, 0, 1])}
    probs = {"x": np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]),
             "y": np.array([[1.0, 0.0], [1.0, 0.0], [0.4, 0.6]]),
             "z": np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]])}
    statistics = {a: compute_classification_statistics(
                      targets[a], probs[a].argmax(1), 2, probs=probs[a].astype(np.float32))
                  for a in targets}
    names = add_per_attribute_metric_names(["amF1", "aMCC", "aNLL"])
    metrics = _to_float_values(compute_multi_attribute_metrics(statistics, names))
    for subset in (["x", "z"], ["y", "z"], ["x"], ["z", "x", "y"]):
        expected = _to_float_values(compute_multi_attribute_metrics(
            {a: s for a, s in statistics.items() if a in subset}, names))
        # Equal values, NaN where undefined.
        np.testing.assert_equal(dc.asdict(select_metric_attributes(metrics, subset)),
                                dc.asdict(expected))
    assert np.isnan(select_metric_attributes(metrics, ["x"]).averages["aMCC"])
    assert select_metric_attributes(metrics, ["y", "z"]).averages["aNLL"] == np.inf

    with pytest.raises(ValueError, match="At least one"):
        select_metric_attributes(metrics, [])
    with pytest.raises(ValueError, match=r"\['w'\]"):
        select_metric_attributes(metrics, ["x", "w"])
    averages_only = _to_float_values(compute_multi_attribute_metrics(statistics, ["amF1"]))
    with pytest.raises(ValueError, match="add_per_attribute_metric_names"):
        select_metric_attributes(averages_only, ["x"])


def test_batched_metrics_equal_separate_computations():
    rng = np.random.default_rng(1)
    matrices = rng.integers(0, 5, size=(4, 3, 3))
    batched = compute_classification_metrics(ClassificationStatistics(matrices), ["mF1", "MCC"])
    for i, matrix in enumerate(matrices):
        single = compute_classification_metrics(ClassificationStatistics(matrix), ["mF1", "MCC"])
        assert batched["mF1"][i] == pytest.approx(single["mF1"])
        assert batched["MCC"][i] == pytest.approx(single["MCC"], nan_ok=True)
