import dataclasses as dc
import math
import re
from datetime import datetime, timezone

import irap_evaluation as ie
import numpy as np
import pytest
from run_helpers import add_model, make_model, score_and_get_models

from irap_evaluation_server.archive import Model, Submission
from irap_evaluation_server.components.attribute_filter import AttributeSubset
from irap_evaluation_server.components.score_tables import (
    MODEL_SCORES_METRIC_NAMES,
    ScoresScope,
    make_method_scores_html,
    make_method_table_html,
    make_model_scores_html,
    make_per_attribute_table_html,
)
from irap_evaluation_server.components.view_queries import select_matching_names
from irap_evaluation_server.label_counts import AttributeLabelCounts
from irap_evaluation_server.method_ranking import (
    DEFAULT_SORT_METRIC,
    RANKING_METRIC_NAMES,
    CellHighlight,
    ShownMetrics,
    compute_highlights,
    is_not_significantly_worse,
    rank_methods,
    select_model_metrics,
    select_shown_methods,
    sort_keys_by_value,
    sort_method_scores,
    sort_scored_models,
)
from irap_evaluation_server.method_scores import MethodScores, ScoredModel, compute_method_scores
from irap_evaluation_server.model_listing import summarize_models_by_method
from irap_evaluation_server.scoring import EvaluationSetScores

_TIME = datetime(2026, 10, 3, tzinfo=timezone.utc)


def make_model_row(model_id, method_name="m", seed=None, method_display_name=None):
    return Model(id=model_id, dataset="vietnam", method_name=method_name,
                 method_display_name=method_display_name, seed=seed, description="",
                 created_at=_TIME)


def make_submission(submission_id, method_name="m", seed=None, training_splits=(), model=None,
                    split="val", num_attributes=2):
    return Submission(
        id=submission_id, model=model or make_model_row(submission_id, method_name, seed),
        split=split, file_sha256="0" * 64, output_kind="probs", context_offsets=(0,),
        training_splits=training_splits, early_stopping_splits=(), num_segments=10,
        num_attributes=num_attributes, submitter="Bo", uploaded_at=_TIME)


def make_scored_model(submission_id, attribute_to_value, method_name="m",
                      missing_attributes=(), training_splits=()):
    """Makes a scored model whose iRAP metrics of each attribute equal the given value."""
    metrics = ie.MetricValues(
        averages={ie.get_attribute_average_name(n): math.nan
                  for n in ie.IRAP_ATTRIBUTE_METRIC_NAMES},
        per_attribute={n: dict(attribute_to_value) for n in ie.IRAP_ATTRIBUTE_METRIC_NAMES})
    scores = EvaluationSetScores(
        evaluation_set="reference", context_offsets=(0,), evaluation_set_fingerprint="f",
        attributes=tuple(attribute_to_value),
        metrics=ie.select_metric_attributes(metrics, list(attribute_to_value)),
        num_invalid={a: 3 for a in attribute_to_value}, missing_attributes=missing_attributes)
    return ScoredModel(make_submission(submission_id, method_name, seed=submission_id,
                                       training_splits=training_splits), scores)


def make_method_scores(method_name, averages, error=None, training_splits=(), attributes=("x",)):
    return MethodScores(
        method_name=method_name,
        scored_models=(make_scored_model(1, {"x": 0.5}, method_name,
                                         training_splits=training_splits),),
        attributes=attributes,
        metrics=None if error else ie.MetricValues(averages=averages, per_attribute={}),
        error=error)


def test_sort_method_scores():
    rows = [make_method_scores("a", {"amF1": 0.5, "aNLL": 1.0, "aMRP90": 0.4}),
            make_method_scores("b", {"amF1": 0.7, "aNLL": 2.0, "aMRP90": 0.3}),
            make_method_scores("c", {"amF1": math.nan, "aNLL": math.inf, "aMRP90": math.nan}),
            make_method_scores("d", {}, error="differ"),
            # Hard predictions, without the metrics of distributions.
            make_method_scores("e", {"amF1": 0.6})]
    assert [r.method_name for r in sort_method_scores(rows, "amF1")] == ["b", "e", "a", "c", "d"]
    # Lower is better, infinite is worst.
    assert [r.method_name for r in sort_method_scores(rows, "aNLL")] == ["a", "b", "c", "d", "e"]
    assert [r.method_name for r in sort_method_scores(rows, "aMRP90")] == ["b", "a", "c", "d", "e"]


def test_ties_are_sorted_by_shown_method_name():
    def make_row(method_name, method_display_name):
        row = make_method_scores(method_name, {"amF1": 0.5})
        [model] = row.scored_models
        submission = dc.replace(model.submission, model=make_model_row(
            1, method_name, method_display_name=method_display_name))
        return dc.replace(row, scored_models=(dc.replace(model, submission=submission),))

    rows = [make_row("a", "z"), make_row("b", None)]
    assert [r.shown_method_name for r in rows] == ["z", "b"]
    assert [r.method_name for r in sort_method_scores(rows, "amF1")] == ["b", "a"]


def test_rank_methods_means_over_models():
    models = [make_scored_model(1, {"x": 0.2, "y": 0.4}),
              make_scored_model(2, {"x": 0.4, "y": 0.6}),
              make_scored_model(3, {"x": 0.5, "y": 0.1}, method_name="n")]
    rows = rank_methods(models, ["x"], "amF1")
    assert [(r.method_name, r.num_models) for r in rows] == [("n", 1), ("m", 2)]
    assert rows[1].metrics.averages["amF1"] == pytest.approx(0.3)
    [row] = rank_methods(models[:1], ["x", "z"], "amF1")
    assert "not scored on z" in row.error


def test_sort_scored_models_by_the_average_over_a_subset():
    models = [make_scored_model(1, {"x": 0.2, "y": 0.9}),
              make_scored_model(2, {"x": 0.4, "y": 0.1}), make_scored_model(3, {"x": 0.3})]
    assert select_model_metrics(models[0], ["x", "y"]).averages["amF1"] == pytest.approx(0.55)
    assert select_model_metrics(models[2], ["x", "y"]) is None  # It is not scored on y.
    assert select_model_metrics(models[0], []) is None
    assert [m.submission.id for m in sort_scored_models(models, "amF1", ["x"])] == [2, 3, 1]
    assert [m.submission.id for m in sort_scored_models(models, "amF1", ["x", "y"])] == [
        1, 2, 3]


def test_select_shown_methods():
    rows = [make_method_scores("a-1", {"amF1": 0.6}),
            make_method_scores("x", {}, error="differ"), make_method_scores("B-2", {"amF1": 0.5}),
            make_method_scores("t", {"amF1": 0.9}, training_splits=("val",))]

    def select(text, reference_method="", hidden_training_split=None):
        shown, num_hidden = select_shown_methods(
            rows, select_matching_names(text, [r.method_name for r in rows]), reference_method,
            hidden_training_split)
        return [r.method_name for r in shown], num_hidden

    assert select("") == (["a-1", "x", "B-2", "t"], 0)
    assert select("b") == (["B-2"], 0)  # A part of the name, case-insensitive.
    assert select(r"^(a|b)-\d$") == (["a-1", "B-2"], 0)
    # The reference, unmatched.
    assert select("^x$", reference_method="B-2") == (["x", "B-2"], 0)
    assert select("nothing") == ([], 0)
    # The methods trained on the split are hidden, except the reference.
    assert select("", hidden_training_split="val") == (["a-1", "x", "B-2"], 1)
    assert select("", hidden_training_split="test") == (["a-1", "x", "B-2", "t"], 0)
    assert select("", "t", hidden_training_split="val") == (["a-1", "x", "B-2", "t"], 0)
    assert select("^a", hidden_training_split="val") == (["a-1"], 0)  # Only matching rows.


def test_select_matching_names_refuses_invalid_and_slow_patterns():
    with pytest.raises(ValueError, match="Invalid regular expression"):
        select_matching_names("a(", ["a"])
    # Exponential backtracking, which `regex` does not avoid, unlike for e.g. '(a+)+#'.
    with pytest.raises(ValueError, match="too long"):
        select_matching_names(r"^(\w|\w\w)*$", ["a" * 40 + "!"])


def test_summarize_models_by_method():
    models = [make_model_row(1, "a", seed=1), make_model_row(2, "a", seed=0),
              make_model_row(3, "b"), make_model_row(4, "c")]
    submissions = [make_submission(10, model=models[0], split="val", num_attributes=3),
                   make_submission(11, model=models[0], split="test", num_attributes=5),
                   make_submission(13, model=models[1]),
                   make_submission(14, model=models[2], training_splits=None)]
    later = dc.replace(submissions[3], uploaded_at=_TIME.replace(year=2027))
    groups = summarize_models_by_method(models, [*submissions[:3], later])
    assert [g.method_name for g in groups] == ["b", "a", "c"]  # The latest updated first.
    a = groups[1]
    assert [m.model.seed for m in a.models] == [0, 1]
    summary = a.models[1]
    assert {s: x.id for s, x in summary.split_to_submission.items()} == {"val": 10, "test": 11}
    assert (summary.num_segments, summary.attribute_range, summary.output_kind) == (
        20, (3, 5), "probs")
    assert groups[0].model_info.training_splits is None
    c = groups[2].models[0]
    assert (c.split_to_submission, c.output_kind, c.attribute_range, c.model_info) == (
        {}, None, None, None)


def test_attribute_subset():
    options = ("a", "b", "c")
    assert AttributeSubset.from_query([], options).is_all
    subset = AttributeSubset.from_query(["c", "x", "b"], options)
    assert (subset.selected, subset.ignored) == (("b", "c"), ("x",))
    assert subset.to_query() == ("b", "c")
    assert subset.label == "2 of 3 attributes"
    # Unknown values alone select all, and are reported.
    unknown = AttributeSubset.from_query(["x"], options)
    assert unknown.is_all and unknown.ignored == ("x",)
    assert unknown.to_query() == () and unknown.label == "all 3 attributes"
    assert subset.select(["c", "a"]).selected == ("a", "c")
    assert subset.select([]).selected == ()


def test_compute_highlights_strength():
    values = [0.2, None, 0.6, math.nan, 0.4, math.inf]
    strengths = [h and h.strength for h in compute_highlights(values, None, False)]
    assert strengths == [0.0, None, 1.0, None, pytest.approx(0.5), None]
    lower_better = compute_highlights(values, None, True)
    assert [h and h.strength for h in lower_better] == [1.0, None, 0.0, None, pytest.approx(0.5),
                                                       None]
    assert not any(h and h.is_not_significantly_worse for h in lower_better)  # Without tests.
    assert compute_highlights([0.5, 0.5], None, False) == [CellHighlight(1.0, False)] * 2
    # Nothing to compare with.
    assert compute_highlights([0.5, None, math.nan], None, False) == [None] * 3
    with pytest.raises(ValueError):
        compute_highlights([0.5, 0.6], [None], False)


def test_compute_highlights_tests_paired_differences_with_the_best():
    rng = np.random.default_rng(0)
    roads = rng.normal(0.7, 0.1, size=1000)  # The variation that the methods share.
    values = [0.7, 0.69, 0.69, 0.5, 0.6, 0.65]
    resamples = [
        roads,
        roads - 0.01,  # Worse in every resample, despite overlapping intervals.
        roads - 0.01 + rng.normal(0, 0.05, size=1000),  # A noisy near-tie.
        roads - 0.2, None,  # Without resamples.
        np.full(1000, np.nan)]  # Undefined.
    is_not_worse = [h.is_not_significantly_worse
                    for h in compute_highlights(values, resamples, False)]
    assert is_not_worse == [True, False, True, False, False, False]
    # Lower is better.
    is_not_worse = [h.is_not_significantly_worse for h in compute_highlights(
        [-v for v in values], [None if r is None else -r for r in resamples], True)]
    assert is_not_worse == [True, False, True, False, False, False]
    # Without resamples of the best, only the best.
    is_not_worse = [h.is_not_significantly_worse
                    for h in compute_highlights(values, [None, *resamples[1:]], False)]
    assert is_not_worse == [True, False, False, False, False, False]


def test_is_not_significantly_worse_is_a_one_sided_test():
    # 5 % of the resampled advantages are not negative: the boundary of the test at 5 %.
    advantages = np.arange(-950, 51) / 1000
    assert is_not_significantly_worse(advantages)
    assert not is_not_significantly_worse(advantages - 0.01)


def test_method_tables_escape_names():
    models = [make_scored_model(1, {"x": 0.2, "y": 0.4}, method_name="<m>",
                                missing_attributes=("y",))]
    rows = [*rank_methods(models, ["x", "y"], "amF1"),
            make_method_scores("<n>", {}, error="Models <differ>.", attributes=("x", "y"))]
    resamples = ie.MetricValues(averages={"amF1": np.linspace(0.1, 0.3, 20)}, per_attribute={})
    scope = ScoresScope("vietnam", "val", ("x",))
    table = make_method_table_html(rows, {"<m>": resamples}, 0.95, ["amF1"], "amF1", scope)
    assert "&lt;m&gt;" in table and "<m>" not in table
    assert "method=%3Cm%3E" in table  # The link to the Method page.
    assert "Models &lt;differ&gt;." in table

    per_attribute = make_per_attribute_table_html(rows, {}, 0.95, "mF1", scope,
                                                  attribute_to_label_counts=None,
                                                  shows_class_counts=True)
    assert "<m>" not in per_attribute and "<n>" not in per_attribute


def test_sort_keys_by_value():
    attributes = list("abcdefg")
    values = {"a": 0.5, "b": math.nan, "c": 0.2, "d": None, "e": 0.5, "f": math.inf}
    # Ties keep their order, and keys without a value are last.
    assert sort_keys_by_value(attributes, values, is_descending=False) == list("caefbdg")
    assert sort_keys_by_value(attributes, values, is_descending=True) == list("faecbdg")


def test_per_attribute_table_sorts_the_attributes_by_a_column():
    attributes = ["x", "y", "z"]
    [row] = rank_methods([make_scored_model(1, {"x": 0.4, "y": 0.2, "z": 0.6})], attributes,
                         "amF1")
    row = dc.replace(row, metrics=dc.replace(row.metrics, per_attribute={
        **row.metrics.per_attribute, "NLL": {"x": 0.4, "y": 0.2, "z": 0.6}}))
    label_counts = {
        a: AttributeLabelCounts(num_labeled_segments=num_segments, num_labeled_classes=num_classes,
                                num_minority_class_segments=5, num_smallest_class_segments=1)
        for a, num_segments, num_classes in (("x", 30, 2), ("y", 20, 4), ("z", 10, 3))}

    def get_attribute_order(attribute_sort, metric_name="mF1", shows_class_counts=True,
                            attribute_to_label_counts=label_counts):
        table = make_per_attribute_table_html(
            [row], {}, 0.95, metric_name, ScoresScope("vietnam", "val", ()),
            attribute_to_label_counts, shows_class_counts, attribute_sort=attribute_sort)
        return re.findall(r"<tr><td>(\w)</td>", table)  # Without the Average row.

    assert get_attribute_order("") == attributes
    # A method: worst first, then best first.
    assert get_attribute_order("method:m") == ["y", "x", "z"]
    assert get_attribute_order("-method:m") == ["z", "x", "y"]
    assert get_attribute_order("method:m", metric_name="NLL") == ["z", "x", "y"]
    assert get_attribute_order("-method:m", metric_name="NLL") == ["y", "x", "z"]
    # A label count: smallest first, then largest first.
    assert get_attribute_order("count:examples") == ["z", "y", "x"]
    assert get_attribute_order("-count:examples") == ["x", "y", "z"]
    assert get_attribute_order("count:classes") == ["x", "z", "y"]
    # Not by a column that is hidden or not in the table.
    assert get_attribute_order("count:classes", shows_class_counts=False) == attributes
    assert get_attribute_order("count:examples", shows_class_counts=False) == ["z", "y", "x"]
    assert get_attribute_order("count:examples", attribute_to_label_counts=None) == attributes
    assert get_attribute_order("-method:other") == attributes


def test_model_scores_table_sorts_the_attributes_by_a_column():
    attributes = ["x", "y", "z"]
    model = make_scored_model(1, {"x": 0.4, "y": 0.2, "z": 0.6})
    model = dc.replace(model, scores=dc.replace(model.scores,
                                                num_invalid={"x": 0, "y": 2, "z": 5}))
    scores = compute_method_scores("m", [model], attributes)

    def get_attribute_order(attribute_sort):
        tables_html = make_model_scores_html(scores, None, 0.95, ShownMetrics.from_hidden(()),
                                             attribute_sort)
        return re.findall(r"<tr><td>(\w)</td>", tables_html)  # Without the Average row.

    assert get_attribute_order("") == attributes
    # A metric: worst first, then best first.
    assert get_attribute_order("mF1") == ["y", "x", "z"]
    assert get_attribute_order("-mF1") == ["z", "x", "y"]
    # The invalid predictions: largest first, then smallest first.
    assert get_attribute_order("invalid") == ["z", "y", "x"]
    assert get_attribute_order("-invalid") == attributes
    assert get_attribute_order("MRP90") == attributes  # Not a column of the table.


def test_method_scores_have_the_mean_of_the_models():
    attributes = ["x", "y"]
    models = [make_scored_model(i, {"x": 0.2 * i, "y": 0.1}) for i in (1, 2)]
    models[1] = dc.replace(models[1], scores=dc.replace(models[1].scores,
                                                        num_invalid={"x": 4, "y": 6}))
    seed_scores = [compute_method_scores(m.submission.label, [m], attributes) for m in models]
    scope = ScoresScope("vietnam", "val", ())

    def make_html(method_scores):
        return make_method_scores_html(method_scores, None, seed_scores, [None, None], 0.95,
                                       ShownMetrics.from_hidden(()), scope)

    tables_html = make_html(compute_method_scores("m", models, attributes))
    # Invalid predictions of x, y, and their mean in the Average row.
    assert all(n in tables_html for n in ("3.5", "4.5", "4.0"))
    # Without the attribute table of a mean with an error.
    offset_models = [models[0], dc.replace(models[1], scores=dc.replace(
        models[1].scores, context_offsets=(1,), evaluation_set_fingerprint="g"))]
    tables_html = make_html(compute_method_scores("m", offset_models, attributes))
    assert "cannot be combined" in tables_html and "Invalid predictions" not in tables_html


def test_shown_metrics_leave_out_the_hidden_ones():
    assert ShownMetrics.from_hidden(()).average_names == RANKING_METRIC_NAMES
    # The main metric and unknown names cannot be hidden.
    shown = ShownMetrics.from_hidden({"amP", "aBrier", DEFAULT_SORT_METRIC, "aUnknown"})
    assert shown.average_names == tuple(n for n in RANKING_METRIC_NAMES
                                        if n not in ("amP", "aBrier"))
    assert "Brier" not in shown.per_attribute_names and "NLL" in shown.per_attribute_names
    assert shown.attribute_table_names == ("mF1", "mR", "MCC", "A", "MRP90")
    assert ShownMetrics.from_hidden({"aMRP90"}).attribute_table_names == (
        "mF1", "mP", "mR", "MCC", "A")


def test_only_models_with_distributions_have_the_manual_review_proportion(
        archive, dataset_contexts, worker):
    submissions = [
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "probs")),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "hard", is_hard=True))]
    method_to_model = {m.submission.model.method_name: m
                       for m in score_and_get_models(worker, [s.id for s in submissions])}
    for method_name, has_manual_review_proportion in (("probs", True), ("hard", False)):
        model = method_to_model[method_name]
        scores = compute_method_scores(method_name, [model], model.scores.attributes)
        assert ("aMRP90" in scores.metrics.averages) == has_manual_review_proportion
        resamples = worker.get_method_resamples(scores, MODEL_SCORES_METRIC_NAMES)
        assert ("MRP90" in resamples.per_attribute) == has_manual_review_proportion
        # The tables of a hard model have no row and no column of the manual review proportion.
        tables_html = make_model_scores_html(scores, resamples, 0.95, ShownMetrics.from_hidden(()))
        assert ("MRP90" in tables_html) == has_manual_review_proportion


def test_method_submitters_are_in_the_order_of_their_first_upload():
    def make_model(submission_id, submitter, day):
        model = make_scored_model(submission_id, {"x": 0.5})
        submission = dc.replace(model.submission, submitter=submitter,
                                uploaded_at=_TIME.replace(day=day))
        return dc.replace(model, submission=submission)

    models = [make_model(1, "Cy", 5), make_model(2, "Al", 4), make_model(3, "Cy", 3)]
    assert compute_method_scores("m", models, ["x"]).submitters == ["Cy", "Al"]
