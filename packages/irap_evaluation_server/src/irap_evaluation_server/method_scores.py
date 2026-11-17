"""Scores of the methods: means over the models of a method on an evaluation set over a subset of
the attributes (`compute_method_scores`), and their resampled values, from which intervals and
paired comparisons follow (`combine_method_resamples`). They are computed from the stored scores
(`scoring`) when they are shown.
"""

import dataclasses as dc
import typing as T

import irap_evaluation as ie
import numpy as np

from .archive import Submission, get_method_description, list_submitters
from .datasets import DatasetContext
from .scoring import (
    RESAMPLED_METRIC_NAMES,
    EvaluationSetScores,
    SubmissionScoring,
    metric_values_to_float32,
)


# Methods ##########################################################################################

@dc.dataclass(frozen=True)
class ScoredModel:
    """A model with the scores of its submission of one split on one evaluation set."""

    submission: Submission
    scores: EvaluationSetScores


def get_model_evaluation_set(context: DatasetContext,
                             scored_model: ScoredModel) -> ie.EvaluationSet:
    """Returns the evaluation set that the scores of a model are of.

    Raises:
        ValueError: If the submission no longer has the set, or the set has changed since the
            scoring (its fingerprint differs).
    """
    submission, scores = scored_model.submission, scored_model.scores
    name_to_set = context.get_submission_evaluation_sets(submission.split,
                                                         submission.context_offsets)
    if scores.evaluation_set not in name_to_set:
        raise ValueError(f"File #{submission.id} of {submission.label} has no evaluation set"
                         f" {scores.evaluation_set!r}.")
    if name_to_set[scores.evaluation_set].fingerprint != scores.evaluation_set_fingerprint:
        raise ValueError(f"The {scores.evaluation_set} set of file #{submission.id} of"
                         f" {submission.label} has changed since it was scored.")
    return name_to_set[scores.evaluation_set]


@dc.dataclass(frozen=True)
class MethodScores:
    """The scores of a method on an evaluation set over a subset of the attributes: the means
    over its models (`irap_evaluation.compute_mean_metrics`).

    Attributes:
        scored_models: In the order of their submission ids.
        attributes: The subset of the attributes.
        metrics: None if there is an `error`.
        error: Why the means cannot be computed, e.g. the models differ in output kind or are not
            scored on an attribute of the subset, or None.
    """

    method_name: str
    scored_models: tuple[ScoredModel, ...]
    attributes: tuple[str, ...]
    metrics: ie.MetricValues[float] | None
    error: str | None

    @property
    def num_models(self) -> int:
        return len(self.scored_models)

    @property
    def missing_attributes(self) -> tuple[str, ...]:
        """The attributes of the subset that some model does not predict, in the order of the
        subset."""
        unpredicted = {a for m in self.scored_models for a in m.scores.missing_attributes}
        return tuple(a for a in self.attributes if a in unpredicted)

    @property
    def shown_method_name(self) -> str:
        """The shown method name of the models (`archive.Model.shown_method_name`), which is the
        same for the models of a method, or `method_name` if there are no models."""
        return (self.scored_models[0].submission.model.shown_method_name if self.scored_models
                else self.method_name)

    @property
    def description(self) -> str:
        """`archive.get_method_description` of the models."""
        return get_method_description(m.submission.model for m in self.scored_models)

    @property
    def submitters(self) -> list[str]:
        """The distinct submitters of the submissions of the models
        (`archive.list_submitters`)."""
        return list_submitters(m.submission for m in self.scored_models)


def select_scored_models(submissions: T.Iterable[Submission],
                         submission_id_to_scoring: T.Mapping[int, SubmissionScoring],
                         evaluation_set: str) -> list[ScoredModel]:
    """Selects the submissions that are scored on the evaluation set with this name, with those
    scores.

    Args:
        submissions: Of one split, so that the submissions of a method are its models.
        submission_id_to_scoring: The current scorings (see `scoring.is_scoring_current`).
    """
    models = []
    for submission in submissions:
        scoring = submission_id_to_scoring.get(submission.id)
        if scoring is not None and (scores := scoring.get_scores(evaluation_set)) is not None:
            models.append(ScoredModel(submission, scores))
    return models


def group_scored_models_by_method(
        models: T.Iterable[ScoredModel]) -> dict[str, list[ScoredModel]]:
    """Groups models by method name, in the order of their submission ids."""
    method_to_models: dict[str, list[ScoredModel]] = {}
    for model in sorted(models, key=lambda m: m.submission.id):
        method_to_models.setdefault(model.submission.model.method_name, []).append(model)
    return method_to_models


def are_scored_on_same_set(models: T.Iterable[ScoredModel]) -> bool:
    """Checks whether models are scored on the same evaluation set (the same fingerprint), so
    that their resampled values are paired: of the same resamples of the same road sequences."""
    return len({m.scores.evaluation_set_fingerprint for m in models}) <= 1


def find_common_evaluation_set(context: DatasetContext,
                               models: T.Sequence[ScoredModel]) -> ie.EvaluationSet | None:
    """Returns the evaluation set that all models are scored on (`are_scored_on_same_set`), or
    None if they are scored on different sets, e.g. model-compatible sets of different context
    offsets.

    Raises:
        ValueError: If `models` is empty, and see `get_model_evaluation_set`.
    """
    if not models:
        raise ValueError("At least one model is required.")
    return get_model_evaluation_set(context, models[0]) if are_scored_on_same_set(models) else None


def get_combination_error(models: T.Sequence[ScoredModel]) -> str | None:
    """Returns why the models cannot be combined in means (`compute_method_scores`) or resampled
    values (`combine_method_resamples`), or None if they can."""
    if not models:
        return "At least one model is required."
    if not are_scored_on_same_set(models):
        offsets = ", ".join(f"{m.submission.label}: {m.scores.context_offsets}" for m in models)
        return (f"The models are scored on different segments (context offsets {offsets}), so"
                f" they cannot be combined.")
    if len({(m.scores.metrics.names, m.scores.attributes) for m in models}) > 1:
        return ("The models differ in metrics or attributes, e.g. hard and probabilistic"
                " predictions, so they cannot be combined.")
    return None


def compute_method_scores(method_name: str, models: T.Sequence[ScoredModel],
                          attributes: T.Sequence[str]) -> MethodScores:
    """Computes the means over the models of a method of their metrics over `attributes`.

    Errors of combining the models, and attributes that they are not scored on, are in
    `MethodScores.error`, so that a page can show them next to the other methods.

    Raises:
        ValueError: If `attributes` is empty.
    """
    if not attributes:
        raise ValueError("At least one attribute is required.")
    models = tuple(sorted(models, key=lambda m: m.submission.id))
    error = get_combination_error(models)
    if error is None and (unscored := [a for a in attributes
                                       if a not in models[0].scores.attributes]):
        error = (f"The models are not scored on {', '.join(unscored)}, e.g. since their"
                 f" evaluation set has no labels of them.")
    metrics = None if error is not None else ie.compute_mean_metrics(
        [ie.select_metric_attributes(m.scores.metrics, attributes) for m in models])
    return MethodScores(method_name=method_name, scored_models=models,
                        attributes=tuple(attributes), metrics=metrics, error=error)


# Resampled values #################################################################################

def combine_method_resamples(method_name: str,
                             model_resamples: T.Sequence[ie.MetricValues[np.ndarray]],
                             attributes: T.Sequence[str], metric_names: T.Collection[str],
                             seed: int) -> ie.MetricValues[np.ndarray]:
    """Computes the resampled values of a method over `attributes` from those of its models
    (`scoring.ScoreStore.get_resamples`): their averages over the attributes
    (`irap_evaluation.select_metric_attributes`), combined with the variation between the models
    (`irap_evaluation.combine_model_resamples`). The values are float32, like the stored ones.

    Args:
        metric_names: The metrics to compute, e.g. those that a page shows, since all metrics of
            all attributes take tens of milliseconds per method. Those that the models lack are
            left out.
        seed: `scoring.ScoringSettings.bootstrap_seed`.

    Raises:
        ValueError: If `metric_names` has metrics that are not stored (`RESAMPLED_METRIC_NAMES`),
            `attributes` is empty or has attributes without values, or the models differ in
            metrics or attributes.
    """
    if unstored := [n for n in metric_names if n not in RESAMPLED_METRIC_NAMES]:
        raise ValueError(f"The resampled values of {unstored} are not stored.")
    # The averages need their per-attribute values.
    needed_names = ie.add_per_attribute_metric_names(list(metric_names))
    return metric_values_to_float32(ie.combine_model_resamples(
        [ie.select_metric_names(
            ie.select_metric_attributes(ie.select_metric_names(r, needed_names), attributes),
            metric_names)
         for r in model_resamples], method_name, seed=seed))
