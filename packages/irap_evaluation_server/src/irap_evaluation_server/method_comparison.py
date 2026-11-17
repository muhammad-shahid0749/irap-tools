"""The comparison of two methods on a shared evaluation set of a dataset split
(`irap_evaluation.SHARED_EVALUATION_SET_NAMES`), as
`irap_evaluation.compare_methods` compares them: the differences of their means over models, with
bootstrap intervals over the same resampled road sequences, which include the variation between
models (see `irap_evaluation/docs/evaluation.md`, "Comparing methods").

The comparisons are computed from the resampled values of the methods
(`scoring_worker.ScoringWorker.get_method_resamples`).
"""

import typing as T

import irap_evaluation as ie
import numpy as np

from .method_scores import MethodScores, are_scored_on_same_set


def list_comparable_methods(rows: T.Iterable[MethodScores]) -> list[MethodScores]:
    """Lists the methods that can be compared, those without `MethodScores.error`, in their
    order."""
    return [r for r in rows if r.error is None]


def get_comparison_error(row: MethodScores, reference: MethodScores) -> str | None:
    """Returns why a method cannot be compared with the reference method, or None if it can.

    Both are means over the same attributes (`method_scores.compute_method_scores`) without an
    error.
    """
    if not are_scored_on_same_set((*row.scored_models, *reference.scored_models)):
        return "The methods are scored on different evaluation sets."
    return None


def compare_with_reference(
        rows: T.Iterable[MethodScores],
        method_to_resamples: T.Mapping[str, ie.MetricValues[np.ndarray] | None],
        reference: MethodScores, confidence: float
) -> tuple[dict[str, ie.MetricValues[ie.MetricDifference]], list[str]]:
    """Computes the differences of the methods minus the reference method
    (`irap_evaluation.compute_resampled_differences`), for the metrics that both have, e.g.
    without the probabilistic ones if one has hard predictions.

    Args:
        rows: Means over the same attributes as `reference` (`method_scores.compute_method_scores`).
        method_to_resamples: Method name -> the resampled values of the method over the
            attributes, or None, e.g. for a method with an error, for the rows and the reference.
        reference: The reference method, without an error.

    Returns:
        Method name -> its differences from the reference, for the rows with resampled values,
        and the errors of the rows that cannot be compared with the reference
        (`get_comparison_error`).
    """
    method_to_differences, errors = {}, []
    if (reference_resamples := method_to_resamples[reference.method_name]) is None:
        return method_to_differences, errors
    for row in rows:
        row_resamples = method_to_resamples[row.method_name]
        if row.method_name == reference.method_name or row_resamples is None:
            continue
        if (error := get_comparison_error(row, reference)) is not None:
            errors.append(f"{row.shown_method_name} cannot be compared with"
                          f" {reference.shown_method_name}: {error}")
            continue
        common_names = set(row_resamples.names) & set(reference_resamples.names)
        method_to_differences[row.method_name] = ie.compute_resampled_differences(
            ie.select_metric_names(row_resamples, common_names), reference_resamples, row.metrics,
            reference.metrics, confidence)
    return method_to_differences, errors


def get_better_method(interval: ie.BootstrapInterval | None,
                      metric_name: str) -> T.Literal["A", "B"] | None:
    """Determines which method is clearly better by the interval of a difference A − B: the one
    that it favours if it excludes 0, otherwise None (also for an undefined interval). There are
    no p-values (`irap_evaluation/docs/evaluation.md`, "Comparing methods")."""
    if interval is None or not interval.low <= interval.high:  # Also NaN bounds.
        return None
    if interval.low <= 0 <= interval.high:
        return None
    is_a_higher = interval.low > 0
    return "A" if is_a_higher != ie.is_lower_better(metric_name) else "B"
