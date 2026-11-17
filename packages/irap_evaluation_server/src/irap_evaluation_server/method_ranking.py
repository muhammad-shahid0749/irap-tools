"""What the Scores page shows: the methods of a dataset split, ranked by a metric averaged over a
subset of the attributes (see `scoring` for the scores), the order of table rows by their values,
and the highlights of compared values."""

import dataclasses as dc
import math
import typing as T

import irap_evaluation as ie
import numpy as np

from .method_scores import (
    MethodScores,
    ScoredModel,
    compute_method_scores,
    group_scored_models_by_method,
)
from .scoring import AVERAGE_METRIC_NAMES

K = T.TypeVar("K", bound=T.Hashable)

#: The columns of the method table, of which admins can hide some (`ShownMetrics`).
RANKING_METRIC_NAMES = AVERAGE_METRIC_NAMES
DEFAULT_SORT_METRIC = ie.get_attribute_average_name(ie.IRAP_MAIN_METRIC)
#: The metrics that admins can hide: all but the main one, which is the default sort.
HIDEABLE_METRIC_NAMES = tuple(n for n in RANKING_METRIC_NAMES if n != DEFAULT_SORT_METRIC)
#: The metrics of the tables of each attribute's scores (`ShownMetrics.attribute_table_names`).
#: The accuracy is next to the manual review proportion, which is read against the error rate.
ATTRIBUTE_TABLE_METRIC_NAMES = (*ie.IRAP_ATTRIBUTE_METRIC_NAMES, "A",
                                ie.MANUAL_REVIEW_PROPORTION_METRIC)


@dc.dataclass(frozen=True)
class ShownMetrics:
    """The metrics that the pages show, those of `RANKING_METRIC_NAMES` that admins have not
    hidden (`display_settings.DisplaySettingsStore`). A hidden attribute average hides its
    per-attribute metric too, e.g. aBrier hides Brier.

    Attributes:
        average_names: The shown attribute averages, in the order of `RANKING_METRIC_NAMES`.
    """

    average_names: tuple[str, ...]

    @classmethod
    def from_hidden(cls, hidden: T.Collection[str]) -> T.Self:
        """Args:
            hidden: The hidden attribute averages. Those that are not in `HIDEABLE_METRIC_NAMES`,
                e.g. stored before a library update, are shown.
        """
        return cls(tuple(n for n in RANKING_METRIC_NAMES
                         if n not in hidden or n not in HIDEABLE_METRIC_NAMES))

    @property
    def per_attribute_names(self) -> tuple[str, ...]:
        """The per-attribute metrics of the shown averages, e.g. mF1 of amF1, for the
        per-attribute view."""
        return tuple(ie.parse_metric_name(n).name for n in self.average_names)

    @property
    def attribute_table_names(self) -> tuple[str, ...]:
        """The metrics of `ATTRIBUTE_TABLE_METRIC_NAMES` whose averages are shown, e.g. for the
        tables of the metrics of each attribute."""
        return tuple(n for n in ATTRIBUTE_TABLE_METRIC_NAMES if n in self.per_attribute_names)


#: The level of the one-sided test of whether a value is worse than the best
#: (`CellHighlight.is_not_significantly_worse`).
NOT_WORSE_SIGNIFICANCE_LEVEL = 0.05


@dc.dataclass(frozen=True)
class CellHighlight:
    """How a value stands out among the values that it is compared with.

    Attributes:
        strength: 0 for the worst value, 1 for the best (min–max over the compared values).
        is_not_significantly_worse: Whether the value is the best, or not significantly worse
            than a best value (`is_not_significantly_worse`).
    """

    strength: float
    is_not_significantly_worse: bool


def is_not_significantly_worse(resampled_advantages: np.ndarray) -> bool:
    """Tests whether a value is not significantly worse than another, by a one-sided paired
    bootstrap test at `NOT_WORSE_SIGNIFICANCE_LEVEL`: whether the upper bound of the percentile
    interval of the resampled advantages at confidence 1 - 2α reaches 0. If the interval is
    undefined, the value is not tested to be not worse.

    Args:
        resampled_advantages: The resampled values of the value minus those of the other, on
            the same resamples, with the sign that makes a better value positive.
    """
    interval = ie.compute_percentile_interval(resampled_advantages,
                                              1 - 2 * NOT_WORSE_SIGNIFICANCE_LEVEL)
    return interval.high >= 0  # Also false for NaN bounds.


def compute_highlights(values: T.Sequence[float | None],
                       resamples: T.Sequence[np.ndarray | None] | None,
                       is_lower_better: bool) -> list[CellHighlight | None]:
    """Computes the highlights of values that are compared with each other, e.g. of the methods
    in a method table column.

    Only finite values are compared. The others, or all values if fewer than 2 are finite, have
    None.

    Args:
        resamples: The resampled values of each value (None if it lacks them), all on the same
            resamples, e.g. of methods on the same evaluation set, or None for no
            `CellHighlight.is_not_significantly_worse`. A value without them is not compared
            with the best.
    """
    is_test_shown = resamples is not None
    pairs = list(zip(values, resamples if is_test_shown else [None] * len(values), strict=True))
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return [None] * len(values)
    best, worst = (min(finite), max(finite)) if is_lower_better else (max(finite), min(finite))
    best_resamples = [r for v, r in pairs if v == best and r is not None]
    sign = -1 if is_lower_better else 1

    def highlight(value: float | None, value_resamples: np.ndarray | None
                  ) -> CellHighlight | None:
        if value is None or not math.isfinite(value):
            return None
        strength = 1.0 if best == worst else (value - worst) / (best - worst)
        is_not_worse = value == best or value_resamples is not None and any(
            is_not_significantly_worse(sign * (value_resamples - b)) for b in best_resamples)
        return CellHighlight(strength, is_not_significantly_worse=is_test_shown and is_not_worse)

    return [highlight(v, r) for v, r in pairs]


def collect_scored_attributes(models: T.Iterable[ScoredModel]) -> tuple[str, ...]:
    """Collects the attributes that the models are scored on, in first-seen order.

    The models of a view normally have the same attributes, since the attribute predictions of
    missing attributes are scored as invalid. Model-compatible sets can lack labels of some
    attributes.
    """
    return tuple(dict.fromkeys(a for m in models for a in m.scores.attributes))


def _get_value_sort_key(value: float | None, is_descending: bool) -> tuple[bool, float]:
    """Returns a sort key that puts the missing values (None or NaN) last."""
    if value is None or math.isnan(value):
        return True, 0.0
    return False, -value if is_descending else value


def _get_sort_key(metrics: ie.MetricValues[float] | None, metric: str,
                  name: str) -> tuple[bool, float, str]:
    """Returns the sort key of `sort_method_scores` and `sort_scored_models`."""
    value = None if metrics is None else metrics.averages.get(metric)
    return (*_get_value_sort_key(value, is_descending=not ie.is_lower_better(metric)), name)


def sort_method_scores(rows: T.Iterable[MethodScores], metric: str) -> list[MethodScores]:
    """Sorts methods best first by the attribute average `metric` (see
    `irap_evaluation.is_lower_better`).

    Rows without a value (an error, a metric that the method lacks, or NaN) are last. Ties are
    ordered by shown method name.
    """
    return sorted(rows, key=lambda r: _get_sort_key(r.metrics, metric, r.shown_method_name))


def select_model_metrics(model: ScoredModel,
                         attributes: T.Sequence[str]) -> ie.MetricValues[float] | None:
    """Selects the metrics of a model over `attributes`
    (`irap_evaluation.select_metric_attributes`), None if it is not scored on all of them, or
    `attributes` is empty."""
    if not attributes or any(a not in model.scores.attributes for a in attributes):
        return None
    return ie.select_metric_attributes(model.scores.metrics, attributes)


def sort_scored_models(models: T.Iterable[ScoredModel], metric: str,
                       attributes: T.Sequence[str]) -> list[ScoredModel]:
    """Sorts models best first by the attribute average `metric` over `attributes`
    (`select_model_metrics`), as `sort_method_scores` sorts methods. Ties are ordered by shown
    model label."""
    return sorted(models, key=lambda m: _get_sort_key(select_model_metrics(m, attributes),
                                                      metric, m.submission.shown_label))


def sort_keys_by_value(keys: T.Iterable[K], key_to_value: T.Mapping[K, float | None],
                       is_descending: bool) -> list[K]:
    """Sorts keys, e.g. attributes, by their values, e.g. of a method in a per-attribute metric.

    Keys without a value (None, NaN, or not in `key_to_value`) are last. Ties keep their order.
    """
    return sorted(keys, key=lambda k: _get_value_sort_key(key_to_value.get(k), is_descending))


def rank_methods(models: T.Iterable[ScoredModel], attributes: T.Sequence[str],
                 sort_metric: str) -> list[MethodScores]:
    """Computes the scores of each method over `attributes` (`method_scores.compute_method_scores`),
    sorted by `sort_metric` (`sort_method_scores`).

    Raises:
        ValueError: If `attributes` is empty.
    """
    return sort_method_scores(
        [compute_method_scores(name, method_models, attributes)
         for name, method_models in group_scored_models_by_method(models).items()],
        sort_metric)


def get_method_split_use(row: MethodScores, split: str) -> ie.SplitUse:
    """Returns what the models of a method used `split` for
    (`irap_evaluation.ModelInfo.get_split_use`), which is the same for all of them
    (`irap_evaluation.check_method_models`)."""
    return row.scored_models[0].submission.model_info.get_split_use(split)


def describe_split_use(split_use: ie.SplitUse, split: str) -> str:
    """Returns a short mark of what a model used a split for, '' if the split is held out.
    `irap_evaluation.explain_split_use` gives the explanation."""
    return {"training": f"trained on {split}", "early_stopping": f"early stopping on {split}",
            "unknown": "training splits unknown", "held_out": ""}[split_use]


def select_shown_methods(rows: T.Iterable[MethodScores],
                         matching_method_names: T.AbstractSet[str],
                         reference_method: str = "",
                         hidden_training_split: str | None = None
                         ) -> tuple[list[MethodScores], int]:
    """Selects the rows of the methods in `matching_method_names`, e.g. those that a filter
    matches (`components.view_queries.select_matching_names`), and the row of
    `reference_method`, in their order.

    Args:
        reference_method: The method that the others are compared with, shown even if it is not
            in `matching_method_names` or was trained on `hidden_training_split`.
        hidden_training_split: If not None, the methods trained on this split
            (`get_method_split_use`) are hidden.

    Returns:
        The selected rows, and the number of matching rows that are hidden.
    """
    shown, num_hidden = [], 0
    for row in rows:
        if row.method_name == reference_method:
            shown.append(row)
        elif row.method_name not in matching_method_names:
            continue
        elif (hidden_training_split is not None
              and get_method_split_use(row, hidden_training_split) == "training"):
            num_hidden += 1
        else:
            shown.append(row)
    return shown, num_hidden
