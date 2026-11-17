"""Scoring `Predictions` on an `EvaluationSet`."""

import dataclasses as dc
import typing as T

import numpy as np

from irap_data.attrs import filter_labeled_attrs, get_attrs_to_include
from irap_data.metadata import IGNORE_LABEL_INDEX, ClassVocabulary

from .evaluation_sets import EvaluationSet
from .metric_names import (
    MACRO_METRIC_TO_PER_CLASS,
    MANUAL_REVIEW_PROPORTION_METRIC,
    PROBABILISTIC_METRICS,
    add_per_attribute_metric_names,
    get_attribute_average_name,
    get_irap_metric_names,
    is_per_class_metric,
    parse_metric_name,
)
from .metrics import (
    ClassificationStatistics,
    MetricValues,
    ReviewCurve,
    compute_classification_metrics,
    compute_classification_statistics,
    compute_grouped_metrics,
    compute_review_curve,
    map_metric_values,
    sum_statistics,
)
from .predictions import (
    ModelInfo,
    OutputKind,
    PredictionFormatError,
    Predictions,
    add_invalid_attributes,
    align_classes,
    select_segments,
    to_class_indices,
)

#: How an invalid attribute prediction (`Predictions.is_valid` False) is scored:
#: - 'first_class': as class 0 in the metrics of the predicted classes and as the uniform
#:   distribution in NLL and Brier.
#: - 'exclude': not scored, like a missing label. Results with invalid attribute predictions are
#:   then scored on different segments, so the means over models (`methods`) and the comparisons
#:   of `bootstrap` refuse them.
#: `docs/evaluation.md#invalid-attribute-predictions` gives the reasons.
NullPolicy = T.Literal["first_class", "exclude"]
NULL_POLICIES: tuple[str, ...] = T.get_args(NullPolicy)

#: How an evaluated attribute that the predictions lack is scored:
#: - 'error': not at all, `evaluate_predictions` raises an error.
#: - 'invalid': as if every attribute prediction of it were invalid, so that models that lack
#:   attributes are scored on the same attributes as the others and are not rewarded for leaving
#:   out attributes.
#: `docs/evaluation.md#missing-attributes` gives the reasons.
MissingAttributePolicy = T.Literal["error", "invalid"]
MISSING_ATTRIBUTE_POLICIES: tuple[str, ...] = T.get_args(MissingAttributePolicy)

#: 'canonical': the iRAP attribute subset (`irap_data.attrs.get_attrs_to_include`). 'all': every
#: attribute of the dataset.
AttributeSelection = T.Literal["canonical", "all"]
ATTRIBUTE_SELECTIONS: tuple[str, ...] = T.get_args(AttributeSelection)


def _check_attributes_in_vocabulary(attributes: T.Iterable[str],
                                    vocabulary: ClassVocabulary) -> None:
    if unknown := [a for a in attributes if a not in vocabulary.attribute_names]:
        raise ValueError(f"Attributes not in the dataset: {unknown}.")


def select_evaluated_attributes(
    evaluation_set: EvaluationSet,
    attributes: AttributeSelection | T.Sequence[str] = "canonical",
) -> tuple[str, ...]:
    """The attributes to score: the candidates that have a label in `evaluation_set`.

    Args:
        attributes: The candidates, an `AttributeSelection` or attribute names. The 'canonical'
            selection can contain attributes that the dataset lacks, which are left out.

    Raises:
        ValueError: For an unknown selection, or attribute names that are not in the dataset.
    """
    if isinstance(attributes, str):
        if attributes not in ATTRIBUTE_SELECTIONS:
            raise ValueError(f"attributes must be one of {ATTRIBUTE_SELECTIONS} or attribute"
                             f" names, got {attributes!r}.")
        attributes = (get_attrs_to_include() if attributes == "canonical"
                      else evaluation_set.vocabulary.attribute_names)
    else:
        _check_attributes_in_vocabulary(attributes, evaluation_set.vocabulary)
    return filter_labeled_attrs(attributes, evaluation_set.attr_to_num_labeled)


def get_default_metric_names(output_kind: OutputKind) -> tuple[str, ...]:
    """The metrics that `evaluate_predictions` computes by default: the iRAP protocol
    (`metric_names.get_irap_metric_names`) for the output kind, and for 'probs' also the
    attribute average of the manual review proportion
    (`metric_names.MANUAL_REVIEW_PROPORTION_METRIC`). Each average is followed by its per-attribute
    value (`metric_names.add_per_attribute_metric_names`), so that
    `metrics.select_metric_attributes` can average the metrics over a subset of the attributes."""
    names = get_irap_metric_names(output_kind=output_kind)
    averages = [n for n in names if parse_metric_name(n).is_averaged]
    if output_kind == "probs":
        averages.append(get_attribute_average_name(MANUAL_REVIEW_PROPORTION_METRIC))
    return add_per_attribute_metric_names(
        (*averages, *(n for n in names if not parse_metric_name(n).is_averaged)))


@dc.dataclass(frozen=True)
class EvaluationResult:
    """Scores of one set of predictions on one evaluation set.

    Attributes:
        model: The model of the evaluated predictions.
        null_policy: See `NullPolicy`.
        statistics: Attribute -> statistics with a leading (G,) axis, one entry per road
            sequence of `evaluation_set.sequence_ids`, from which any metric can be recomputed,
            e.g. for a bootstrap resample. Its keys are the evaluated attributes (`attributes`).
        num_invalid: Attribute -> number of labeled segments with an invalid prediction.
        metrics: The scores. The values of `n` and `nc_suppN` are ints, the others floats. For
            per-class metrics, see `compute_class_metrics`.
        intervals: `bootstrap.BootstrapInterval`s of the metrics of this model alone, which cover
            the sampling of road sequences but not of models (see
            `bootstrap.compute_bootstrap_intervals`), or None.
        missing_attributes: The evaluated attributes that the predictions lack, scored as
            invalid attribute predictions (`MissingAttributePolicy` 'invalid'). `num_invalid`
            counts them too.
    """

    model: ModelInfo
    output_kind: OutputKind
    evaluation_set: EvaluationSet
    null_policy: NullPolicy
    statistics: T.Mapping[str, ClassificationStatistics]
    num_invalid: T.Mapping[str, int]
    metrics: MetricValues[float]
    intervals: MetricValues | None = None
    missing_attributes: tuple[str, ...] = ()

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(self.statistics)


def align_to_evaluation_set(
    predictions: Predictions,
    evaluation_set: EvaluationSet,
    attributes: T.Sequence[str],
    *,
    missing_attribute_policy: MissingAttributePolicy = "error",
) -> Predictions:
    """Selects and orders the predictions to match the labels of `evaluation_set`: its segments
    in its order, `attributes`, and the classes of each attribute in the order of its
    vocabulary.

    So `to_class_indices` of the result compares directly with `evaluation_set.class_indices`.
    `evaluate_predictions` scores the predictions aligned this way.

    Args:
        missing_attribute_policy: See `MissingAttributePolicy`.

    Raises:
        ValueError: For attributes that are not in the dataset, or an unknown
            `missing_attribute_policy`.
        PredictionFormatError: If the release or split differ, an attribute is not predicted
            and `missing_attribute_policy='error'`, the predicted iRAP codes of an attribute
            differ from the dataset's, or a segment of the evaluation set has no prediction.
    """
    header = predictions.header
    if (header.dataset, header.split) != (evaluation_set.dataset, evaluation_set.split):
        raise PredictionFormatError(
            f"Predictions of {header.dataset}/{header.split} cannot be scored on"
            f" {evaluation_set.dataset}/{evaluation_set.split}.")
    if missing_attribute_policy not in MISSING_ATTRIBUTE_POLICIES:
        raise ValueError(f"missing_attribute_policy must be one of {MISSING_ATTRIBUTE_POLICIES},"
                         f" got {missing_attribute_policy!r}.")
    _check_attributes_in_vocabulary(attributes, evaluation_set.vocabulary)
    attribute_to_irap_codes = evaluation_set.vocabulary.restrict_to_attributes(
        attributes).attribute_to_irap_codes
    if missing_attribute_policy == "invalid":
        predictions = add_invalid_attributes(predictions, attribute_to_irap_codes)
    return select_segments(align_classes(predictions, attribute_to_irap_codes),
                           evaluation_set.segment_ids)


def evaluate_predictions(
    predictions: Predictions,
    evaluation_set: EvaluationSet,
    *,
    attributes: T.Sequence[str] | None = None,
    null_policy: NullPolicy = "first_class",
    missing_attribute_policy: MissingAttributePolicy = "error",
    metric_names: T.Sequence[str] | None = None,
) -> EvaluationResult:
    """Scores predictions on every segment of an evaluation set.

    Predictions of other segments, e.g. outside the reference set, are not used.

    Args:
        predictions: Predictions of the same release and split as `evaluation_set`.
        attributes: The attributes to score. None means `select_evaluated_attributes` with its
            defaults.
        null_policy: See `NullPolicy`.
        missing_attribute_policy: See `MissingAttributePolicy`.
        metric_names: None means `get_default_metric_names` of the output kind of the
            predictions.

    Raises:
        ValueError: For probabilistic metrics of hard predictions, per-class metric names, if
            there is no attribute to score, if `null_policy='exclude'` would leave an attribute
            without scored segments, since its attribute predictions are invalid on every labeled
            segment or, with `missing_attribute_policy='invalid'`, it is not predicted, and as
            `align_to_evaluation_set` does.
        PredictionFormatError: See `align_to_evaluation_set`.
    """
    if null_policy not in NULL_POLICIES:
        raise ValueError(f"null_policy must be one of {NULL_POLICIES}, got {null_policy!r}.")
    if metric_names is None:
        metric_names = get_default_metric_names(predictions.output_kind)
    if per_class := [n for n in metric_names if is_per_class_metric(n)]:
        raise ValueError(f"Per-class metrics {per_class} are not kept in results. Use"
                         f" compute_class_metrics(result).")
    attributes = tuple(select_evaluated_attributes(evaluation_set) if attributes is None
                       else attributes)
    if not attributes:
        raise ValueError("There is no attribute to score.")
    missing_attributes = (tuple(a for a in attributes if a not in predictions.attributes)
                          if missing_attribute_policy == "invalid" else ())
    aligned = align_to_evaluation_set(predictions, evaluation_set, attributes,
                                      missing_attribute_policy=missing_attribute_policy)
    num_invalid = {attr: int((~aligned.is_valid[attr]
                              & (evaluation_set.class_indices[attr] != IGNORE_LABEL_INDEX)).sum())
                   for attr in attributes}
    if null_policy == "exclude":
        attr_to_num_labeled = evaluation_set.attr_to_num_labeled
        if all_invalid := [a for a in attributes
                           if 0 < num_invalid[a] == attr_to_num_labeled[a]]:
            raise ValueError(f"The attribute predictions of {all_invalid} are invalid or missing"
                             f" on every labeled segment, so null_policy='exclude' would leave"
                             f" them without scored segments. Use null_policy='first_class'.")
    has_probabilities = predictions.output_kind == "probs"
    attr_to_predicted_indices = to_class_indices(aligned)

    def get_statistics(attr: str) -> ClassificationStatistics:
        is_valid = aligned.is_valid[attr]
        targets = evaluation_set.class_indices[attr]
        if null_policy == "exclude":
            targets = np.where(is_valid, targets, IGNORE_LABEL_INDEX)
        num_classes = len(aligned.attribute_to_irap_codes[attr])
        # See NullPolicy.
        predicted_indices = np.where(is_valid, attr_to_predicted_indices[attr], 0)
        probs = (np.where(is_valid[:, None], aligned.probs[attr], np.float32(1 / num_classes))
                 if has_probabilities else None)
        return compute_classification_statistics(
            targets, predicted_indices, num_classes, probs=probs,
            group_indices=evaluation_set.segment_sequence_indices,
            num_groups=len(evaluation_set.sequence_ids))

    statistics = {attr: get_statistics(attr) for attr in attributes}
    return EvaluationResult(
        model=predictions.header.model, output_kind=predictions.output_kind,
        evaluation_set=evaluation_set, null_policy=null_policy,
        statistics=statistics, num_invalid=num_invalid,
        metrics=_compute_result_metrics(statistics, metric_names),
        missing_attributes=missing_attributes)


def _compute_result_metrics(statistics: T.Mapping[str, ClassificationStatistics],
                            metric_names: T.Sequence[str]) -> MetricValues[float]:
    """`EvaluationResult.metrics` from its statistics."""
    return map_metric_values(lambda v: np.asarray(v).item(),
                             compute_grouped_metrics(statistics, metric_names))


def select_result_attributes(result: EvaluationResult,
                             attributes: T.Collection[str]) -> EvaluationResult:
    """The result of a subset of its attributes, with the same metrics computed again from the
    statistics, e.g. for intervals of averages over the subset.

    The attributes keep the order of `result.attributes`, so that the metrics equal those of
    scoring only the subset. The intervals are None.

    Raises:
        ValueError: If `attributes` is empty or has attributes that `result` does not score.
    """
    if not attributes:
        raise ValueError("At least one attribute is required.")
    if unknown := sorted(set(attributes) - set(result.attributes)):
        raise ValueError(f"Attributes not scored in {result.model.label!r}: {unknown}.")
    selected = [a for a in result.attributes if a in attributes]
    statistics = {a: result.statistics[a] for a in selected}
    return dc.replace(
        result, statistics=statistics, num_invalid={a: result.num_invalid[a] for a in selected},
        metrics=_compute_result_metrics(statistics, result.metrics.names), intervals=None,
        missing_attributes=tuple(a for a in result.missing_attributes if a in attributes))


@dc.dataclass(frozen=True)
class ClassMetrics:
    """Per-class metrics of one attribute on an evaluation set, in vocabulary class order.

    Attributes:
        irap_codes: (K,) the iRAP code of each class.
        values: (K,) the class values, e.g. 'Wide (>= 3.25m)'.
        support: (K,) int64 number of labeled segments per class.
        confusion_matrix: (K, K) int64 counts, rows = ground truth, columns = prediction.
        metrics: Name -> (K,) float64 values, e.g. 'P', 'R', 'F1' (see
            `metric_names.PER_CLASS_METRICS`).
    """

    irap_codes: tuple[int, ...]
    values: tuple[str, ...]
    support: np.ndarray
    confusion_matrix: np.ndarray
    metrics: T.Mapping[str, np.ndarray]


def compute_class_metrics(result: EvaluationResult,
                          metric_names: T.Sequence[str] | None = None) -> dict[str, ClassMetrics]:
    """Attribute -> the per-class metrics of `result`, from its statistics.

    Args:
        metric_names: Per-class metric names. None means 'P', 'R', 'F1' and 'IoU', and also
            'cNLL' and 'cBrier' for 'probs' predictions.

    Raises:
        ValueError: For a name that is not a per-class metric, or a probabilistic metric of hard
            predictions.
    """
    if metric_names is None:
        metric_names = [n for n in MACRO_METRIC_TO_PER_CLASS.values()
                        if result.output_kind == "probs" or n not in PROBABILISTIC_METRICS]
    if not_per_class := [n for n in metric_names if not is_per_class_metric(n)]:
        raise ValueError(f"Not per-class metrics: {not_per_class}.")
    vocabulary = result.evaluation_set.vocabulary

    def compute(attr: str) -> ClassMetrics:
        statistics = sum_statistics(result.statistics[attr])
        return ClassMetrics(
            irap_codes=vocabulary.get_irap_codes(attr), values=vocabulary.get_values(attr),
            support=statistics.support, confusion_matrix=statistics.confusion_matrix,
            metrics=compute_classification_metrics(statistics, metric_names))

    return {attr: compute(attr) for attr in result.attributes}


def compute_review_curves(result: EvaluationResult) -> dict[str, ReviewCurve]:
    """Attribute -> the review curve of `result` (`metrics.compute_review_curve`), from its
    statistics.

    Raises:
        ValueError: For hard predictions.
    """
    return {attr: compute_review_curve(sum_statistics(result.statistics[attr]))
            for attr in result.attributes}
