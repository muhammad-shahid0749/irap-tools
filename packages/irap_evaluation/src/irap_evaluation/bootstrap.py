"""Confidence intervals of methods and of their differences.

Each resample draws road sequences with replacement and recomputes the metrics of each model from
the per-sequence statistics (`resample_model_metrics`). The resampled values of a method add the
uncertainty of the mean over its models with a Student-t draw (`combine_model_resamples`).
Intervals are percentiles of the resampled values (`compute_percentile_interval`), and comparisons
are those of the differences of two methods on the same resamples
(`compute_resampled_differences`). `docs/evaluation.md#confidence-intervals` defines them.
"""

import dataclasses as dc
import hashlib
import math
import typing as T

import numpy as np

from .evaluation import EvaluationResult
from .methods import (
    check_method_results,
    check_models_have_same_metrics,
    check_results_are_comparable,
    compute_method_metrics,
)
from .metrics import MetricValues, compute_grouped_metrics, map_metric_values


@dc.dataclass(frozen=True)
class BootstrapInterval:
    """A bootstrap confidence interval.

    Attributes:
        num_undefined: The number of resamples whose value is NaN, which the bounds leave out,
            e.g. resamples with an undefined MCC, or with infinite values of both compared
            methods. The bounds are NaN if every resample is undefined.
    """

    low: float
    high: float
    num_undefined: int

    @property
    def is_partially_undefined(self) -> bool:
        """Whether some but not all resamples are undefined. The bounds then hold only for the
        resamples where the value is defined, unlike NaN bounds, which show that every resample
        is undefined."""
        return self.num_undefined > 0 and not math.isnan(self.low)


@dc.dataclass(frozen=True)
class MetricDifference:
    """The difference `a - b` of a metric of two methods on the same evaluation set.

    Attributes:
        difference: The difference of the means over the models of each method.
        interval: Its bootstrap confidence interval.
    """

    difference: float
    interval: BootstrapInterval


def draw_group_weights(num_groups: int, num_resamples: int,
                       rng: np.random.Generator) -> np.ndarray:
    """(num_resamples, num_groups) int64: how often each resample draws each group.

    Raises:
        ValueError: If there are no groups, e.g. for an empty evaluation set, or no resamples.
    """
    if num_groups < 1:
        raise ValueError("The evaluation set has no road sequences to resample.")
    if num_resamples < 1:
        raise ValueError(f"At least one resample is required, got {num_resamples}.")
    return rng.multinomial(num_groups, np.full(num_groups, 1 / num_groups), size=num_resamples)


def _draw_model_t_values(num_models: int, num_resamples: int, method_name: str,
                         seed: int) -> np.ndarray | None:
    """(num_resamples,) Student-t(num_models - 1) values, or None for a single model.

    They are drawn from a stream of `seed` and `method_name`, which is independent of the road
    sequences (`draw_group_weights`) and of the draws of other methods.
    """
    if num_models == 1:
        return None
    method_key = int.from_bytes(hashlib.sha256(method_name.encode()).digest()[:8], "little")
    rng = np.random.default_rng([seed, method_key])
    return rng.standard_t(num_models - 1, size=num_resamples)


def _compute_quantile(sorted_values: np.ndarray, quantile: float) -> float:
    """The quantile of non-empty sorted values with linear interpolation, like `np.quantile`.

    Unlike `np.quantile`, which computes inf - inf = NaN, it interpolates between equal infinite
    values to that value, and between an infinite and a finite value to the infinite one.
    """
    position = (len(sorted_values) - 1) * quantile
    index = int(np.floor(position))
    fraction = position - index
    low = float(sorted_values[index])
    high = float(sorted_values[min(index + 1, len(sorted_values) - 1)])
    if fraction == 0 or low == high:
        return low
    return (1 - fraction) * low + fraction * high


def compute_percentile_interval(values: np.ndarray, confidence: float) -> BootstrapInterval:
    """Computes the percentile interval of resampled values, e.g. of `combine_model_resamples`:
    from the (1 - confidence) / 2 to the (1 + confidence) / 2 quantile of the values that are not
    NaN (`docs/evaluation.md#confidence-intervals`).

    Raises:
        ValueError: If `confidence` is not in [0, 1].
    """
    _check_confidence(confidence)
    values = np.asarray(values, dtype=np.float64)
    is_undefined = np.isnan(values)
    num_undefined = int(is_undefined.sum())
    values = np.sort(values[~is_undefined])
    if values.size == 0:
        return BootstrapInterval(float("nan"), float("nan"), num_undefined)
    tail = (1 - confidence) / 2
    return BootstrapInterval(_compute_quantile(values, tail), _compute_quantile(values, 1 - tail),
                    num_undefined)


def _check_confidence(confidence: float) -> None:
    if not 0 <= confidence <= 1:
        raise ValueError(f"The confidence must be in [0, 1], got {confidence}.")


def resample_model_metrics(result: EvaluationResult,
                           metric_names: T.Sequence[str] | None = None, *,
                           num_resamples: int = 1000, seed: int = 0) -> MetricValues[np.ndarray]:
    """Computes the metrics of a model in each resample of the road sequences of its evaluation
    set: (num_resamples,) float arrays.

    The resamples depend only on `seed`, `num_resamples` and the number of road sequences, so
    they are the same for all models on an evaluation set, which pairs comparisons of methods.

    Args:
        metric_names: None means the metrics of `result`.

    Raises:
        ValueError: If the evaluation set is empty, or `num_resamples` < 1.
    """
    weights = draw_group_weights(len(result.evaluation_set.sequence_ids), num_resamples,
                                 np.random.default_rng(seed))
    return compute_grouped_metrics(
        result.statistics, result.metrics.names if metric_names is None else metric_names,
        weights)


def _get_num_resamples(values: MetricValues[np.ndarray]) -> int:
    """Returns the number of resamples of resampled values, which have at least one value."""
    return len(next(values.iterate_values()))


def combine_model_resamples(model_resamples: T.Sequence[MetricValues[np.ndarray]],
                            method_name: str, *, seed: int = 0) -> MetricValues[np.ndarray]:
    """Computes the resampled values of a method from those of its models
    (`resample_model_metrics` with the same seed).

    Each value is the mean over the models plus their standard deviation over sqrt(M) times a
    Student-t(M - 1) draw per resample, or the value of the model for a single model. A value
    whose mean is infinite is that mean. The t draws depend on `seed` and `method_name`, so they
    are independent between methods, and the same for a method in every comparison. Models
    without metric values, e.g. after a selection of metrics that they lack, give a method
    without metric values.

    Raises:
        ValueError: If there are no models, or they differ in metrics, attributes or the number
            of resamples.
    """
    check_models_have_same_metrics(model_resamples)
    if next(model_resamples[0].iterate_values(), None) is None:
        return model_resamples[0]
    if len(num_resamples := {_get_num_resamples(v) for v in model_resamples}) > 1:
        raise ValueError(f"The models differ in the number of resamples: {sorted(num_resamples)}.")
    model_t_values = _draw_model_t_values(len(model_resamples), num_resamples.pop(), method_name,
                                          seed)

    def combine(*model_values) -> np.ndarray:
        values = np.stack([np.asarray(v, dtype=np.float64) for v in model_values])  # (M, B)
        mean = values.mean(0)
        if model_t_values is None:
            return mean
        with np.errstate(invalid="ignore"):  # the deviation of infinite values is NaN
            uncertainty = values.std(0, ddof=1) / np.sqrt(len(values)) * model_t_values
        return np.where(np.isinf(mean), mean, mean + uncertainty)

    return map_metric_values(combine, *model_resamples)


def _resample_method_metrics(models: T.Sequence[EvaluationResult], metric_names: T.Sequence[str],
                             num_resamples: int, seed: int) -> MetricValues[np.ndarray]:
    return combine_model_resamples(
        [resample_model_metrics(r, metric_names, num_resamples=num_resamples, seed=seed)
         for r in models],
        models[0].model.method_name, seed=seed)


def compute_bootstrap_intervals(
    models: T.Sequence[EvaluationResult],
    *,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricValues[BootstrapInterval]:
    """Computes confidence intervals of the metrics of a method (`methods.compute_method_metrics`).

    Args:
        models: The results of the models of the method. With a single model, the intervals
            cover only the sampling of road sequences, and can be attached to it with
            `dataclasses.replace(result, intervals=...)`.

    Raises:
        ValueError: If the evaluation set is empty, `num_resamples` < 1, `confidence` is not in
            [0, 1], or `models` are not valid for `methods.check_method_results`.
    """
    _check_confidence(confidence)
    check_method_results(models)
    resampled = _resample_method_metrics(models, models[0].metrics.names, num_resamples, seed)
    return map_metric_values(lambda v: compute_percentile_interval(v, confidence), resampled)


def compare_methods(
    models_a: T.Sequence[EvaluationResult],
    models_b: T.Sequence[EvaluationResult],
    metric_names: T.Sequence[str] | None = None,
    *,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricValues[MetricDifference]:
    """Compares two methods on the same resampled road sequences.

    Args:
        models_a: The results of the models of method a.
        models_b: The results of the models of method b.
        metric_names: Metrics of both methods, attribute averages or per-attribute metrics.
            None means the attribute averages that both have.

    Raises:
        ValueError: If the models of a method are not valid for `methods.check_method_results`,
            the models of a are not comparable with those of b
            (`methods.check_results_are_comparable`), a method lacks a metric, `num_resamples` < 1
            or `confidence` is not in [0, 1].
    """
    _check_confidence(confidence)
    check_method_results(models_a)
    check_method_results(models_b)
    check_results_are_comparable(models_a[0], models_b[0])
    metrics_a, metrics_b = models_a[0].metrics, models_b[0].metrics
    if metric_names is None:
        metric_names = [n for n in metrics_a.averages if n in metrics_b.averages]
    common_names = set(metrics_a.names) & set(metrics_b.names)
    if missing := [n for n in metric_names if n not in common_names]:
        raise ValueError(f"Metrics not in both methods: {missing}. Method a has"
                         f" {list(metrics_a.names)}.")
    resampled_a, resampled_b = (
        _resample_method_metrics(models, metric_names, num_resamples, seed)
        for models in (models_a, models_b))
    return compute_resampled_differences(resampled_a, resampled_b,
                                         compute_method_metrics(models_a),
                                         compute_method_metrics(models_b), confidence)


def compute_resampled_differences(resampled_a: MetricValues[np.ndarray],
                                  resampled_b: MetricValues[np.ndarray],
                                  metrics_a: MetricValues[float], metrics_b: MetricValues[float],
                                  confidence: float) -> MetricValues[MetricDifference]:
    """Computes the differences a - b of the metrics of two methods, for the metrics of
    `resampled_a`, with the percentile intervals of the differences of their resampled values.

    Args:
        resampled_a: The resampled values of method a (`combine_model_resamples`).
        resampled_b: Those of method b, on the same resamples, e.g. on the same evaluation set
            with the same seed.
        metrics_a: The metrics of method a, e.g. `methods.compute_method_metrics`.
        metrics_b: Those of method b.

    Raises:
        ValueError: If `confidence` is not in [0, 1].
    """
    _check_confidence(confidence)

    def get_difference(resampled_value_a, resampled_value_b, point_a, point_b
                       ) -> MetricDifference:
        return MetricDifference(float(point_a - point_b),
                                compute_percentile_interval(resampled_value_a - resampled_value_b,
                                                            confidence))

    return map_metric_values(get_difference, resampled_a, resampled_b, metrics_a, metrics_b)
