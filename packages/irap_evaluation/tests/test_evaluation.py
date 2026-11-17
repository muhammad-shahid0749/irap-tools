import dataclasses as dc

import numpy as np
import pytest

from irap_data.metadata import IGNORE_LABEL_INDEX
from irap_evaluation.bootstrap import compute_bootstrap_intervals
from irap_evaluation.evaluation import (
    align_to_evaluation_set,
    compute_class_metrics,
    evaluate_predictions,
    get_default_metric_names,
    select_evaluated_attributes,
    select_result_attributes,
)
from irap_evaluation.metrics import (
    compute_classification_metrics,
    compute_classification_statistics,
    compute_multi_attribute_metrics,
    select_metric_attributes,
    sum_statistics,
)
from irap_evaluation.predictions import (
    PredictionFormatError,
    select_segments,
)
from irap_evaluation.reports.evaluation_report import to_class_table, to_json_dict

from synthetic_vietnam import (
    SPLIT,
    invalidate_attribute_predictions,
    make_predictions,
    to_hard_predictions,
)


def test_evaluated_attributes_are_the_labeled_canonical_ones(reference_set, predictions):
    assert select_evaluated_attributes(reference_set) == ("Carriageway label", "Number of lanes",
                                                          "Lane width", "Curvature")
    with pytest.raises(ValueError, match=r"not in the dataset: \['Lane widht'\]"):
        select_evaluated_attributes(reference_set, ["Lane width", "Lane widht"])
    with pytest.raises(ValueError, match="not in the dataset"):
        evaluate_predictions(predictions, reference_set, attributes=["Lane width", "Speed limit"])


def test_evaluation_matches_direct_metric_computation(metadata, reference_set):
    # In reverse segment order, so that the rows have to be aligned with the evaluation set.
    predictions = make_predictions(metadata.vocabulary, list(reversed(metadata.splits[SPLIT])),
                                   name="m", seed=0)
    result = evaluate_predictions(predictions, reference_set)
    selected = select_segments(predictions, reference_set.segment_ids)
    expected = compute_multi_attribute_metrics(
        {a: compute_classification_statistics(
            reference_set.class_indices[a], selected.probs[a].argmax(1),
            selected.probs[a].shape[1], probs=selected.probs[a])
         for a in result.attributes}, result.metrics.names)
    for name in ["amF1", "aMCC", "aNLL", "amF1_supp5"]:
        assert result.metrics.averages[name] == pytest.approx(float(expected.averages[name]),
                                                              nan_ok=True)
    assert result.metrics.per_attribute["n"]["Curvature"] == int(
        (reference_set.class_indices["Curvature"] != IGNORE_LABEL_INDEX).sum())


def test_evaluation_is_independent_of_the_predicted_class_order(metadata, reference_set,
                                                                predictions):
    reversed_ = make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]), name="m",
                                 seed=0, reverse_classes=True)
    # Same random logits, so reversing the class order changes which class gets which logit.
    # Re-align the reversed file's probabilities to check that alignment is by code.
    realigned = dc.replace(reversed_,
                           probs={a: p[:, ::-1].copy() for a, p in reversed_.probs.items()})
    np.testing.assert_equal(dc.asdict(evaluate_predictions(predictions, reference_set).metrics),
                            dc.asdict(evaluate_predictions(realigned, reference_set).metrics))


def test_align_to_evaluation_set(metadata, reference_set):
    segments = list(metadata.splits[SPLIT])
    reversed_ = make_predictions(metadata.vocabulary.restrict_to_attributes(["Curvature"]),
                                 segments[::-1], name="m", seed=0, reverse_classes=True)
    aligned = align_to_evaluation_set(reversed_, reference_set, ["Curvature", "Lane width"],
                                      missing_attribute_policy="invalid")
    assert aligned.segment_ids == reference_set.segment_ids
    assert aligned.attribute_to_irap_codes == {
        a: metadata.vocabulary.get_irap_codes(a) for a in ["Curvature", "Lane width"]}
    rows = [segments[::-1].index(s) for s in reference_set.segment_ids]
    np.testing.assert_array_equal(aligned.probs["Curvature"],
                                  reversed_.probs["Curvature"][rows][:, ::-1])
    assert not aligned.is_valid["Lane width"].any()
    with pytest.raises(PredictionFormatError, match="not predicted"):
        align_to_evaluation_set(reversed_, reference_set, ["Lane width"])


def test_null_policies(reference_set, predictions):
    with_invalid = invalidate_attribute_predictions(predictions, "Lane width",
                                                    np.arange(predictions.num_segments) % 2 == 0)
    first_class = evaluate_predictions(with_invalid, reference_set, null_policy="first_class")
    excluded = evaluate_predictions(with_invalid, reference_set, null_policy="exclude")
    num_labeled = reference_set.attr_to_num_labeled["Lane width"]
    assert first_class.metrics.per_attribute["n"]["Lane width"] == num_labeled
    assert excluded.metrics.per_attribute["n"]["Lane width"] == \
           num_labeled - excluded.num_invalid["Lane width"]
    assert excluded.num_invalid["Lane width"] > 0
    all_invalid = invalidate_attribute_predictions(predictions, "Lane width",
                                                   np.zeros(predictions.num_segments, bool))
    with pytest.raises(ValueError, match=r"\['Lane width'\] are invalid"):
        evaluate_predictions(all_invalid, reference_set, null_policy="exclude")


def test_first_class_scores_invalid_attribute_predictions_as_uniform_distributions(
        metadata, reference_set, predictions):
    invalid = invalidate_attribute_predictions(predictions, "Lane width",
                                               np.zeros(predictions.num_segments, bool))
    metrics = evaluate_predictions(invalid, reference_set,
                                   null_policy="first_class").metrics.per_attribute
    num_classes = len(metadata.vocabulary.get_irap_codes("Lane width"))
    # Finite, unlike the NLL of a one-hot distribution of class 0.
    assert metrics["NLL"]["Lane width"] == pytest.approx(np.log(num_classes))
    assert metrics["Brier"]["Lane width"] == pytest.approx(1 - 1 / num_classes)
    # Every invalid attribute prediction counts as a prediction of class 0.
    labels = reference_set.class_indices["Lane width"]
    assert metrics["A"]["Lane width"] == pytest.approx(
        np.mean(labels[labels != IGNORE_LABEL_INDEX] == 0))


def test_only_distributions_get_the_manual_review_proportion(reference_set, predictions):
    assert {"aMRP90", "MRP90"} <= set(get_default_metric_names("probs"))
    hard = to_hard_predictions(predictions)
    assert hard.output_kind == "hard"
    hard_names = evaluate_predictions(hard, reference_set).metrics.names
    assert hard_names == get_default_metric_names("hard")
    assert not {"aMRP90", "MRP90"} & set(hard_names)


def test_manual_review_proportion_of_invalid_predictions_is_that_of_a_review_in_random_order(
        reference_set, predictions):
    # All invalid attribute predictions have the same confidence, that of the uniform
    # distribution.
    invalid = invalidate_attribute_predictions(predictions, "Lane width",
                                               np.zeros(predictions.num_segments, bool))
    metrics = evaluate_predictions(invalid, reference_set).metrics.per_attribute
    macro_f1 = metrics["mF1"]["Lane width"]
    assert macro_f1 < 0.9
    assert metrics["MRP90"]["Lane width"] == pytest.approx((0.9 - macro_f1) / (1 - macro_f1))


def test_class_metrics(metadata, reference_set, predictions):
    result = evaluate_predictions(predictions, reference_set)
    lane_width = compute_class_metrics(result)["Lane width"]
    assert lane_width.irap_codes == metadata.vocabulary.get_irap_codes("Lane width")
    assert list(lane_width.metrics) == ["P", "R", "F1", "IoU", "cNLL", "cBrier"]
    statistics = sum_statistics(result.statistics["Lane width"])
    np.testing.assert_array_equal(lane_width.support, statistics.support)
    np.testing.assert_allclose(lane_width.metrics["F1"],
                               compute_classification_metrics(statistics, ["F1"])["F1"])
    table = to_class_table(lane_width)
    assert list(table.columns) == ["value", "support", "P", "R", "F1", "IoU", "cNLL", "cBrier"]
    assert list(table.index) == list(lane_width.irap_codes)
    with pytest.raises(ValueError, match="Not per-class"):
        compute_class_metrics(result, ["mF1"])


def test_results_do_not_keep_per_class_metrics(reference_set, predictions):
    with pytest.raises(ValueError, match="compute_class_metrics"):
        evaluate_predictions(predictions, reference_set, metric_names=["mF1", "F1"])


def test_evaluation_requires_every_segment_and_attribute(metadata, reference_set):
    segments = list(reference_set.segment_ids)
    with pytest.raises(PredictionFormatError, match="no prediction"):
        evaluate_predictions(make_predictions(metadata.vocabulary, segments[1:], name="m", seed=0),
                             reference_set)
    partial = make_predictions(metadata.vocabulary.restrict_to_attributes(["Curvature"]),
                               segments, name="m", seed=0)
    with pytest.raises(PredictionFormatError, match="not predicted"):
        evaluate_predictions(partial, reference_set)
    assert evaluate_predictions(partial, reference_set, attributes=["Curvature"]).attributes == \
           ("Curvature",)


def test_missing_attributes_score_as_invalid_attribute_predictions(metadata, reference_set,
                                                                   predictions):
    without = dc.replace(predictions, attribute_to_irap_codes={
        a: c for a, c in predictions.attribute_to_irap_codes.items() if a != "Lane width"},
        probs={a: p for a, p in predictions.probs.items() if a != "Lane width"})
    result = evaluate_predictions(without, reference_set, missing_attribute_policy="invalid")
    all_invalid = evaluate_predictions(
        invalidate_attribute_predictions(predictions, "Lane width",
                                         np.zeros(predictions.num_segments, bool)),
        reference_set)
    np.testing.assert_equal(dc.asdict(result.metrics), dc.asdict(all_invalid.metrics))
    assert result.num_invalid == all_invalid.num_invalid
    assert result.missing_attributes == ("Lane width",)
    assert all_invalid.missing_attributes == ()
    assert to_json_dict(result)["missing_attributes"] == ["Lane width"]
    with pytest.raises(ValueError, match="exclude"):
        evaluate_predictions(without, reference_set, missing_attribute_policy="invalid",
                             null_policy="exclude")
    # Without missing attributes, 'exclude' is allowed.
    evaluate_predictions(predictions, reference_set, missing_attribute_policy="invalid",
                         null_policy="exclude")
    with pytest.raises(ValueError, match="missing_attribute_policy"):
        evaluate_predictions(predictions, reference_set, missing_attribute_policy="skip")


def test_select_result_attributes_equals_scoring_the_subset(reference_set, predictions):
    result = evaluate_predictions(predictions, reference_set)
    subset = ["Curvature", "Number of lanes"]  # Not in the order of the result.
    assert result.attributes == ("Carriageway label", "Number of lanes", "Lane width",
                                 "Curvature")
    selected = select_result_attributes(result, subset)
    expected = evaluate_predictions(predictions, reference_set,
                                    attributes=[a for a in result.attributes if a in subset])
    assert selected.attributes == expected.attributes
    np.testing.assert_equal(dc.asdict(selected.metrics), dc.asdict(expected.metrics))
    np.testing.assert_equal(dc.asdict(select_metric_attributes(result.metrics, subset)),
                            dc.asdict(expected.metrics))
    assert selected.num_invalid == expected.num_invalid
    assert selected.intervals is None
    np.testing.assert_equal(dc.asdict(compute_bootstrap_intervals([selected], num_resamples=20)),
                            dc.asdict(compute_bootstrap_intervals([expected], num_resamples=20)))
    with pytest.raises(ValueError, match="not scored"):
        select_result_attributes(result, ["Lane width", "Speed limit"])
    with pytest.raises(ValueError, match="At least one"):
        select_result_attributes(result, [])
