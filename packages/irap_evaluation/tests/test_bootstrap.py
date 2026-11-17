import dataclasses as dc

import numpy as np
import pytest

from irap_evaluation.bootstrap import (
    BootstrapInterval,
    MetricDifference,
    combine_model_resamples,
    compare_methods,
    compute_bootstrap_intervals,
    compute_percentile_interval,
    draw_group_weights,
    resample_model_metrics,
)
from irap_evaluation.evaluation import evaluate_predictions, select_result_attributes
from irap_evaluation.evaluation_sets import get_evaluation_set
from irap_evaluation.methods import compute_method_metrics
from irap_evaluation.metrics import (
    MetricValues,
    compute_grouped_metrics,
    select_metric_attributes,
)

from synthetic_vietnam import (
    SPLIT,
    evaluate_models,
    invalidate_attribute_predictions,
    make_predictions,
    with_model,
)


def test_bootstrap_intervals_contain_the_point_estimate(metadata, predictions):
    evaluation_set = get_evaluation_set(metadata, "vietnam", SPLIT, (0,), "all")
    result = evaluate_predictions(predictions, evaluation_set)
    intervals = compute_bootstrap_intervals([result], num_resamples=200, seed=1)
    assert intervals == compute_bootstrap_intervals([result], num_resamples=200, seed=1)
    for name in ["amF1", "aNLL"]:
        interval = intervals.averages[name]
        assert interval.low <= result.metrics.averages[name] <= interval.high
    assert set(intervals.per_attribute["mF1"]) == set(result.attributes)


def test_bootstrap_argument_errors(metadata, reference_set, predictions):
    result = evaluate_predictions(predictions, reference_set)
    with pytest.raises(ValueError, match="At least one resample"):
        compute_bootstrap_intervals([result], num_resamples=0)
    with pytest.raises(ValueError, match="At least one resample"):
        compare_methods([result], [result], num_resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        compute_bootstrap_intervals([result], num_resamples=10, confidence=95)
    with pytest.raises(ValueError, match="at least one model"):
        compute_bootstrap_intervals([], num_resamples=10)
    # No segment has a context window of 100 segments.
    empty_set = get_evaluation_set(metadata, "vietnam", SPLIT, (0, -100), "empty")
    empty_result = evaluate_predictions(predictions, empty_set, attributes=["Lane width"])
    with pytest.raises(ValueError, match="no road sequences"):
        compute_bootstrap_intervals([empty_result], num_resamples=10)


def test_compare_methods(metadata, reference_set, predictions):
    predictions_b = make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]),
                                     name="b", seed=1)
    result_a, result_b = (evaluate_predictions(p, reference_set)
                          for p in (predictions, predictions_b))
    self_difference = compare_methods([result_a], [result_a], ["amF1"],
                                      num_resamples=50).averages["amF1"]
    assert self_difference == MetricDifference(0.0, BootstrapInterval(0.0, 0.0, 0))
    per_attribute = compare_methods([result_a], [result_b], ["mF1"], num_resamples=20)
    assert set(per_attribute.per_attribute["mF1"]) == set(result_a.attributes)
    assert compare_methods([result_a], [result_b], num_resamples=20).names == \
           tuple(result_a.metrics.averages)

    with pytest.raises(ValueError, match="not in both methods"):
        compare_methods([result_a], [result_b], ["amIoU"], num_resamples=20)
    other_set = evaluate_predictions(
        predictions_b, get_evaluation_set(metadata, "vietnam", SPLIT, (0,), "all"))
    with pytest.raises(ValueError, match="different segments"):
        compare_methods([result_a], [other_set])
    with pytest.raises(ValueError, match="null policies"):
        compare_methods([result_a], [evaluate_predictions(predictions_b, reference_set,
                                                          null_policy="exclude")])
    half_invalid = invalidate_attribute_predictions(predictions, "Lane width",
                                                    np.arange(predictions.num_segments) % 2 == 0)
    with pytest.raises(ValueError, match="exclude"):
        compare_methods(
            [evaluate_predictions(half_invalid, reference_set, null_policy="exclude")],
            [evaluate_predictions(predictions_b, reference_set, null_policy="exclude")])


def _to_bounds(intervals):
    """(num_intervals, 2) the bounds of `MetricValues` of intervals."""
    per_attribute = [i for attr_to_interval in intervals.per_attribute.values()
                     for i in attr_to_interval.values()]
    return np.array([(i.low, i.high) for i in [*intervals.averages.values(), *per_attribute]])


def test_identical_models_add_no_uncertainty(metadata, reference_set):
    [model] = evaluate_models(metadata, reference_set, "m", [0])
    copies = [with_model(model, "m", seed=i) for i in range(3)]
    assert compute_method_metrics(copies) == model.metrics
    # The mean of the resampled values can differ from the value by rounding.
    np.testing.assert_allclose(
        _to_bounds(compute_bootstrap_intervals(copies, num_resamples=50, seed=4)),
        _to_bounds(compute_bootstrap_intervals([model], num_resamples=50, seed=4)), rtol=1e-12)


def test_models_share_the_resamples_of_road_sequences(metadata, reference_set):
    models = evaluate_models(metadata, reference_set, "m", [0, 1])
    weights = draw_group_weights(len(reference_set.sequence_ids), 20, np.random.default_rng(3))
    for model in models:
        np.testing.assert_array_equal(
            resample_model_metrics(model, ["amF1"], num_resamples=20, seed=3).averages["amF1"],
            compute_grouped_metrics(model.statistics, ["amF1"], weights).averages["amF1"])


def test_resampled_averages_over_a_subset_equal_resampling_the_subset(reference_set,
                                                                      predictions):
    result = evaluate_predictions(predictions, reference_set)
    subset = ["Curvature", "Number of lanes"]
    np.testing.assert_equal(
        dc.asdict(select_metric_attributes(resample_model_metrics(result, num_resamples=20),
                                           subset)),
        dc.asdict(resample_model_metrics(select_result_attributes(result, subset),
                                         num_resamples=20)))


def test_resampled_method_values_add_the_model_term(metadata, reference_set):
    models = evaluate_models(metadata, reference_set, "m", [0, 1, 2])
    model_resamples = [resample_model_metrics(r, ["amF1", "mF1"], num_resamples=20)
                       for r in models]
    resampled = combine_model_resamples(model_resamples, "m")
    assert set(resampled.per_attribute["mF1"]) == set(models[0].attributes)

    def compute_t_values(select):
        model_values = np.stack([select(v) for v in model_resamples])
        return ((select(resampled) - model_values.mean(0))
                / (model_values.std(0, ddof=1) / np.sqrt(3)))

    # One t draw per resample, shared by the metrics of the method.
    attribute = models[0].attributes[0]
    np.testing.assert_allclose(compute_t_values(lambda v: v.per_attribute["mF1"][attribute]),
                               compute_t_values(lambda v: v.averages["amF1"]))
    np.testing.assert_array_equal(
        combine_model_resamples(model_resamples, "m").averages["amF1"],
        resampled.averages["amF1"])
    assert not np.allclose(combine_model_resamples(model_resamples, "other").averages["amF1"],
                           resampled.averages["amF1"])
    with pytest.raises(ValueError, match="At least one model"):
        combine_model_resamples([], "m")
    # Models without values, e.g. of metrics that hard predictions lack.
    without_values = MetricValues(averages={}, per_attribute={})
    assert combine_model_resamples([without_values] * 2, "m") == without_values
    with pytest.raises(ValueError, match="number of resamples"):
        combine_model_resamples(
            [model_resamples[0], resample_model_metrics(models[1], ["amF1", "mF1"],
                                                        num_resamples=10)], "m")

    point = compute_method_metrics(models)
    assert point.averages["amF1"] == pytest.approx(
        np.mean([r.metrics.averages["amF1"] for r in models]))
    interval = compute_bootstrap_intervals(models, num_resamples=200).averages["amF1"]
    assert interval.low <= point.averages["amF1"] <= interval.high


def test_an_infinite_model_mean_stays_infinite(metadata, reference_set, predictions):
    # Probability 0 for every class but the first gives an infinite NLL.
    probs = np.zeros_like(predictions.probs["Lane width"])
    probs[:, 0] = 1
    certain = dc.replace(predictions, probs={**predictions.probs, "Lane width": probs})
    models = [with_model(evaluate_predictions(predictions, reference_set), "m", seed=0),
              with_model(evaluate_predictions(certain, reference_set), "m", seed=1)]
    assert compute_method_metrics(models).averages["aNLL"] == np.inf
    interval = compute_bootstrap_intervals(models, num_resamples=50).averages["aNLL"]
    assert (interval.low, interval.high) == (np.inf, np.inf)


@pytest.mark.parametrize("values, expected", [
    ([1.0] * 3 + [np.inf] * 97, (np.inf, np.inf)),
    ([1.0] * 50 + [np.inf] * 50, (1.0, np.inf)),
    ([np.inf] * 10, (np.inf, np.inf)),
    ([-np.inf] * 50 + [0.0] * 50, (-np.inf, 0.0)),
    ([np.nan] * 5, (np.nan, np.nan)),
])
def test_intervals_of_infinite_values(values, expected):
    interval = compute_percentile_interval(np.array(values), 0.95)
    np.testing.assert_equal((interval.low, interval.high), expected)
    assert interval.num_undefined == np.isnan(values).sum()


def test_intervals_of_finite_values_are_numpy_percentiles():
    values = np.random.default_rng(0).normal(size=101)
    interval = compute_percentile_interval(np.concatenate([values, [np.nan]]), 0.9)
    np.testing.assert_allclose((interval.low, interval.high), np.quantile(values, [0.05, 0.95]))
    assert interval.num_undefined == 1
