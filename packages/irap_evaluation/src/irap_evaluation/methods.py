"""Methods: the models of one training recipe, e.g. with several seeds – grouping, checks and
means over models.

`docs/evaluation.md#methods` defines how the models of a method are combined.
"""

import collections
import typing as T

import numpy as np

from .evaluation import EvaluationResult
from .metrics import MetricValues, map_metric_values
from .predictions import ModelInfo, PredictionHeader


# Models ###########################################################################################

def check_method_models(models: T.Iterable[ModelInfo]) -> None:
    """Checks that the models of each method can be averaged: they are distinct (different seeds,
    or a single model without a seed), and have the same training and early stopping splits,
    since a method is one training recipe.

    Args:
        models: The models of one dataset, e.g. of the files of one split, each once.

    Raises:
        ValueError: If a method has a seed twice, several models of which one lacks a seed, or
            models with different training or early stopping splits.
    """
    name_to_models: dict[str, list[ModelInfo]] = {}
    for model in models:
        name_to_models.setdefault(model.method_name, []).append(model)
    for name, method_models in name_to_models.items():
        seeds = [m.seed for m in method_models]
        if len(seeds) > 1 and None in seeds:
            raise ValueError(f"Method {name!r} has {len(seeds)} models, so each needs a seed.")
        if len(set(seeds)) < len(seeds):
            raise ValueError(f"Method {name!r} has several models with the same seed.")
        if len({(m.training_splits, m.early_stopping_splits) for m in method_models}) > 1:
            raise ValueError(f"The models of method {name!r} have different training or early"
                             f" stopping splits. Models trained on other data belong to another"
                             f" method.")


def check_prediction_file_models(headers: T.Sequence[PredictionHeader]) -> None:
    """Checks the models of prediction files, e.g. of several splits: the files of one model (a
    method and seed) have the same `ModelInfo`, and the models of each split and of each dataset
    can be averaged (`check_method_models`).

    Raises:
        ValueError: If they cannot.
    """
    key_to_models: dict[tuple[str, ...], list[ModelInfo]] = {}
    for h in headers:
        key_to_models.setdefault((h.dataset, h.split), []).append(h.model)
    for (dataset, split), models in key_to_models.items():
        try:
            check_method_models(models)
        except ValueError as e:
            raise ValueError(f"The files of {dataset}/{split}: {e}") from e
    dataset_to_models = {h.dataset: set() for h in headers}
    for h in headers:
        dataset_to_models[h.dataset].add(h.model)
    for dataset, models in dataset_to_models.items():
        if differing := sorted(label for label, n in collections.Counter(
                m.label for m in models).items() if n > 1):
            raise ValueError(f"The files of the models {differing} on {dataset} differ in their"
                             f" training or early stopping splits.")
        check_method_models(models)


def group_models_by_method(
        results: T.Iterable[EvaluationResult]) -> dict[str, tuple[EvaluationResult, ...]]:
    """Groups results by method name, in the order of `results`.

    Raises:
        ValueError: See `check_method_models`.
    """
    results = list(results)
    check_method_models(r.model for r in results)
    name_to_results: dict[str, list[EvaluationResult]] = {}
    for result in results:
        name_to_results.setdefault(result.model.method_name, []).append(result)
    return {name: tuple(rs) for name, rs in name_to_results.items()}


# Results ##########################################################################################

def check_results_are_comparable(result_a: EvaluationResult, result_b: EvaluationResult) -> None:
    """Checks that two results can be averaged or compared: they are scored on the same segments,
    road sequences and attributes, with the same null policy.

    Raises:
        ValueError: If they differ, or use `null_policy='exclude'` with invalid attribute
            predictions, which scores them on different segments.
    """
    set_a, set_b = result_a.evaluation_set, result_b.evaluation_set
    label_a, label_b = result_a.model.label, result_b.model.label
    if (set_a.segment_ids, set_a.segment_sequence_ids) != \
            (set_b.segment_ids, set_b.segment_sequence_ids):
        raise ValueError(f"{label_a!r} and {label_b!r} are on different segments. Compare them on"
                         f" the reference set, which is the same for every method.")
    if result_a.attributes != result_b.attributes:
        raise ValueError(f"{label_a!r} and {label_b!r} score different attributes:"
                         f" {sorted(set(result_a.attributes) ^ set(result_b.attributes))}.")
    if result_a.null_policy != result_b.null_policy:
        raise ValueError(f"{label_a!r} and {label_b!r} use different null policies:"
                         f" {result_a.null_policy!r} and {result_b.null_policy!r}.")
    if result_a.null_policy == "exclude" and any(
            n > 0 for r in (result_a, result_b) for n in r.num_invalid.values()):
        raise ValueError("With null_policy='exclude', the invalid attribute predictions of each"
                         " model are left out, so the models are scored on different segments."
                         " Use null_policy='first_class'.")


def _values_have_same_metrics(values_a: MetricValues, values_b: MetricValues) -> bool:
    """Whether the values are of the same metrics and, per metric, of the same attributes."""
    return values_a.names == values_b.names and all(
        values_a.per_attribute[n].keys() == values_b.per_attribute[n].keys()
        for n in values_a.per_attribute)


def check_models_have_same_metrics(model_values: T.Sequence[MetricValues]) -> None:
    """Checks that the values of the models of a method, e.g. their metrics or resampled values,
    are of the same metrics and attributes.

    Raises:
        ValueError: If there are no models, or they differ in metrics or attributes.
    """
    if not model_values:
        raise ValueError("At least one model is required.")
    if not all(_values_have_same_metrics(v, model_values[0]) for v in model_values[1:]):
        raise ValueError("The models differ in metrics or attributes.")


def check_method_results(models: T.Sequence[EvaluationResult]) -> None:
    """Checks that `models` are results of the models of one method that can be averaged.

    Raises:
        ValueError: If `models` is empty, the results are not of the models of one method that
            `check_method_models` accepts, with the same metrics, or two models are not
            comparable (`check_results_are_comparable`).
    """
    if not models:
        raise ValueError("A method needs at least one model.")
    if len(names := {r.model.method_name for r in models}) > 1:
        raise ValueError(f"The models are of different methods: {sorted(names)}.")
    check_method_models(r.model for r in models)
    for model in models[1:]:
        check_results_are_comparable(models[0], model)
        if not _values_have_same_metrics(model.metrics, models[0].metrics):
            raise ValueError(f"The models {models[0].model.label!r} and {model.model.label!r}"
                             f" have different metrics.")


# Means over models ################################################################################

def _compute_mean(*values: float) -> float:
    """The mean, or the value itself if all values are equal, which keeps ints (e.g. `n`) and
    avoids rounding."""
    if all(v == values[0] for v in values):
        return values[0]
    return float(np.mean(values))


def compute_mean_metrics(model_metrics: T.Sequence[MetricValues[float]]) -> MetricValues[float]:
    """Computes the mean of each metric over models, e.g. of their metrics over a subset of the
    attributes (`metrics.select_metric_attributes`). `compute_method_metrics` also checks the
    models.

    Raises:
        ValueError: See `check_models_have_same_metrics`.
    """
    check_models_have_same_metrics(model_metrics)
    return map_metric_values(_compute_mean, *model_metrics)


def compute_method_metrics(models: T.Sequence[EvaluationResult]) -> MetricValues[float]:
    """Computes the metrics of a method: the mean of each metric over its models.

    Args:
        models: The results of the models of the method.

    Raises:
        ValueError: See `check_method_results`.
    """
    check_method_results(models)
    return compute_mean_metrics([r.metrics for r in models])
