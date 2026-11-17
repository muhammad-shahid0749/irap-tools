import irap_evaluation as ie
import pytest
from run_helpers import (
    add_model,
    add_models_of_different_context_offsets,
    assert_close,
    assert_intervals_close,
    make_model,
    score_and_get_models,
    score_submission_again,
)
from synthetic_vietnam import SPLIT

from irap_evaluation_server.method_scores import (
    compute_method_scores,
    find_common_evaluation_set,
    group_scored_models_by_method,
)
from irap_evaluation_server.scoring import RESAMPLED_METRIC_NAMES


def get_method_intervals(worker, row):
    """Computes the intervals of a method from its stored resampled values, as the pages do."""
    return ie.map_metric_values(
        lambda v: ie.compute_percentile_interval(v, worker.settings.confidence),
        worker.get_method_resamples(row, RESAMPLED_METRIC_NAMES))


def test_method_scores_and_intervals_over_a_subset(archive, dataset_contexts, worker):
    submissions = [
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", 1, random_seed=1)),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", 2, random_seed=2)),
        add_model(archive, dataset_contexts,
                  make_model(dataset_contexts, "other", random_seed=3))]
    models = score_and_get_models(worker, [s.id for s in submissions])
    method_to_models = group_scored_models_by_method(models)
    assert [m.submission.id for m in method_to_models["m"]] == [s.id for s in submissions[:2]]
    subset = method_to_models["m"][0].scores.attributes[1:]

    method_scores = compute_method_scores("m", method_to_models["m"], subset)
    assert method_scores.error is None and method_scores.num_models == 2
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    results = [ie.select_result_attributes(score_submission_again(worker, s, reference_set),
                                           subset)
               for s in submissions[:2]]
    assert_close(method_scores.metrics, ie.compute_method_metrics(results))

    # Intervals over the subset, from the stored resampled values of the models.
    assert_intervals_close(
        get_method_intervals(worker, method_scores),
        ie.compute_bootstrap_intervals(results, num_resamples=worker.settings.num_resamples,
                                       seed=0))
    # Only the requested metrics.
    assert worker.get_method_resamples(method_scores, ["amF1", "mF1"]).names == ("amF1", "mF1")
    assert worker.get_method_resamples(
        compute_method_scores("m", method_to_models["m"], ["unscored"]), ["amF1"]) is None


def test_missing_attribute(archive, dataset_contexts, worker):
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    attr = ie.select_evaluated_attributes(reference_set)[0]
    submission = add_model(archive, dataset_contexts,
                           make_model(dataset_contexts, "m", drop_attribute=attr))
    [model] = score_and_get_models(worker, [submission.id])
    assert model.scores.missing_attributes == (attr,)
    assert compute_method_scores("m", [model], model.scores.attributes).missing_attributes == (
        attr,)
    assert compute_method_scores("m", [model],
                                 model.scores.attributes[1:]).missing_attributes == ()


def test_models_that_cannot_be_combined(archive, dataset_contexts, worker):
    submissions = [
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", 1)),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", 2, is_hard=True))]
    models = score_and_get_models(worker, [s.id for s in submissions])
    method_scores = compute_method_scores("m", models, models[0].scores.attributes)
    assert method_scores.metrics is None
    assert "differ in metrics" in method_scores.error


def test_find_common_evaluation_set_of_model_compatible_sets(archive, dataset_contexts, worker):
    submissions = add_models_of_different_context_offsets(archive, dataset_contexts)
    models = score_and_get_models(worker, [s.id for s in submissions],
                                  ie.MODEL_COMPATIBLE_SET_NAME)
    offsets_to_model = {m.scores.context_offsets: m for m in models}
    context = dataset_contexts["vietnam"]
    for offsets, model in offsets_to_model.items():
        assert find_common_evaluation_set(context, [model]) is context.get_model_compatible_set(
            SPLIT, offsets)
    assert len(offsets_to_model) == 2 and find_common_evaluation_set(context, models) is None
    with pytest.raises(ValueError, match="At least one model"):
        find_common_evaluation_set(context, [])
