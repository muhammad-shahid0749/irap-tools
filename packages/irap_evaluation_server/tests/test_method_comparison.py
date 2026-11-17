import dataclasses as dc
import math

import irap_evaluation as ie
import pytest
from irap_evaluation.reports.evaluation_report import to_labeled_metric_values
from run_helpers import (
    add_model,
    assert_intervals_close,
    make_model,
    score_and_get_models,
    score_submission_again,
)
from synthetic_vietnam import SPLIT

from irap_evaluation_server.components.score_tables import get_per_attribute_table_metric_names
from irap_evaluation_server.method_comparison import (
    compare_with_reference,
    get_better_method,
    get_comparison_error,
)
from irap_evaluation_server.method_scores import (
    compute_method_scores,
    group_scored_models_by_method,
)
from irap_evaluation_server.scoring import RESAMPLED_METRIC_NAMES


@pytest.fixture
def method_to_models(archive, dataset_contexts, worker):
    """Method 'a' with two models, 'b' with one, and 'hard' with one of hard predictions."""
    submissions = [
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "a", 1, random_seed=1)),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "a", 2, random_seed=2)),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "b", random_seed=3)),
        add_model(archive, dataset_contexts,
                  make_model(dataset_contexts, "hard", random_seed=4, is_hard=True))]
    return group_scored_models_by_method(
        score_and_get_models(worker, [s.id for s in submissions]))


def _compare(worker, method_to_models, name_a, name_b, attributes):
    """Compares method a with the reference b from their stored resampled values, as the Scores
    page does."""
    row_a, row_b = (compute_method_scores(n, method_to_models[n], attributes)
                    for n in (name_a, name_b))
    method_to_resamples = {r.method_name: worker.get_method_resamples(r, RESAMPLED_METRIC_NAMES)
                           for r in (row_a, row_b)}
    method_to_differences, errors = compare_with_reference(
        [row_a, row_b], method_to_resamples, row_b, worker.settings.confidence)
    assert errors == [] and list(method_to_differences) == [name_a]
    return method_to_differences[name_a]


def test_comparison_equals_compare_methods(dataset_contexts, worker, method_to_models):
    subset = method_to_models["a"][0].scores.attributes[1:]
    differences = _compare(worker, method_to_models, "a", "b", subset)

    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    results_a, results_b = (
        [ie.select_result_attributes(score_submission_again(worker, m.submission, reference_set),
                                     subset) for m in method_to_models[name]]
        for name in ("a", "b"))
    expected = ie.compare_methods(
        results_a, results_b, differences.names, num_resamples=worker.settings.num_resamples,
        seed=0)
    expected_differences = to_labeled_metric_values(expected)
    for label, difference in to_labeled_metric_values(differences).items():
        assert difference.difference == pytest.approx(expected_differences[label].difference,
                                                      rel=1e-12)

    def select_intervals(values):
        return ie.map_metric_values(lambda d: d.interval, values)

    assert_intervals_close(select_intervals(differences), select_intervals(expected))


def test_hard_predictions_are_compared_on_the_common_metrics(worker, method_to_models):
    attributes = method_to_models["a"][0].scores.attributes
    for names in [("a", "hard"), ("hard", "a")]:
        differences = _compare(worker, method_to_models, *names, attributes)
        assert "amF1" in differences.averages and "aNLL" not in differences.averages
        assert "mF1" in differences.per_attribute


def test_a_method_has_no_resampled_values_of_metrics_that_it_lacks(worker, method_to_models):
    # As in the per-attribute view of a metric of distributions, next to a hard method.
    attributes = method_to_models["a"][0].scores.attributes
    row_a, row_hard = (compute_method_scores(n, method_to_models[n], attributes)
                       for n in ("a", "hard"))
    for metric_name in ("NLL", "MRP90"):
        names = get_per_attribute_table_metric_names(metric_name)
        assert worker.get_method_resamples(row_a, names).names == (
            ie.get_attribute_average_name(metric_name), metric_name)
        hard_resamples = worker.get_method_resamples(row_hard, names)
        assert hard_resamples.names == ()
        differences, errors = compare_with_reference(
            [row_a, row_hard], {"a": worker.get_method_resamples(row_a, names),
                                "hard": hard_resamples}, row_a, worker.settings.confidence)
        assert errors == [] and differences["hard"].names == ()


def test_get_comparison_error(method_to_models):
    attributes = method_to_models["a"][0].scores.attributes
    row_a, row_b = (compute_method_scores(n, method_to_models[n], attributes) for n in "ab")
    assert get_comparison_error(row_a, row_b) is None
    [model_b] = row_b.scored_models
    other_set = dc.replace(row_b, scored_models=(dc.replace(model_b, scores=dc.replace(
        model_b.scores, evaluation_set_fingerprint="other")),))
    assert "different evaluation sets" in get_comparison_error(row_a, other_set)


def test_get_better_method():
    def interval(low, high):
        return ie.BootstrapInterval(low=low, high=high, num_undefined=0)

    assert get_better_method(interval(0.05, 0.15), "amF1") == "A"
    assert get_better_method(interval(-0.15, -0.05), "amF1") == "B"
    # Lower is better.
    assert get_better_method(interval(0.05, 0.15), "aNLL") == "B"
    assert get_better_method(interval(-0.15, -0.05), "aNLL") == "A"
    # The interval includes 0, is undefined, or there is none.
    for undecided in (interval(-0.1, 0.1), interval(0.0, 0.1), interval(math.nan, math.nan),
                      None):
        assert get_better_method(undecided, "amF1") is None
