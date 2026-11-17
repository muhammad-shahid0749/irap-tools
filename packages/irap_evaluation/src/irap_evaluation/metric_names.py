"""Metric names of the iRAP evaluation protocol: their form, parsing and descriptions.

Metric names have the form `[a]<base>[_suppN]`:

- `<base>` is a per-attribute metric (see `CONFUSION_MATRIX_METRICS`, `PROBABILISTIC_METRICS`
  and `MANUAL_REVIEW_METRICS`).
- The `a` prefix averages it over attributes, leaving out attributes where it is undefined (NaN).
- The `_suppN` suffix restricts a mean over classes to the classes with at least N ground-truth
  examples. `nc_suppN` is the number of such classes.
"""

import typing as T

from .predictions import OutputKind

SUPPORT_SUFFIX = "_supp"

#: Metrics computed from the confusion matrix. `P`, `R`, `F1` and `IoU` are per-class arrays and
#: the `m` versions their means over classes.
CONFUSION_MATRIX_METRICS = frozenset(
    {"A", "mP", "mR", "mF1", "mIoU", "P", "R", "F1", "IoU", "n", "MCC", "kappa"})
#: Metrics computed from predicted distributions. `NLL` and `Brier` are means over examples,
#: `cNLL` and `cBrier` per-class means, and `mNLL` and `mBrier` means over classes.
PROBABILISTIC_METRICS = frozenset({"NLL", "Brier", "cNLL", "cBrier", "mNLL", "mBrier"})
#: The metric of a simulated review (`metrics.compute_review_curve`): the share of the labeled
#: examples that a reviewer corrects, least confident first, until mF1 reaches
#: `MANUAL_REVIEW_TARGET_MF1`.
MANUAL_REVIEW_TARGET_MF1 = 0.9
MANUAL_REVIEW_PROPORTION_METRIC = f"MRP{round(MANUAL_REVIEW_TARGET_MF1 * 100)}"
#: Metrics computed from the confidences of predicted distributions.
MANUAL_REVIEW_METRICS = frozenset({MANUAL_REVIEW_PROPORTION_METRIC})
#: Metrics that need predicted distributions, so hard predictions lack them.
DISTRIBUTION_METRICS = PROBABILISTIC_METRICS | MANUAL_REVIEW_METRICS
#: Metrics whose lower values are better, which are those of predicted distributions.
LOWER_IS_BETTER_METRICS = DISTRIBUTION_METRICS
#: Metrics that exist only with a support threshold.
SUPPORT_ONLY_METRICS = frozenset({"nc"})
KNOWN_BASE_METRICS = CONFUSION_MATRIX_METRICS | DISTRIBUTION_METRICS | SUPPORT_ONLY_METRICS
#: Mean-over-classes metric -> the per-class array it is the mean of.
MACRO_METRIC_TO_PER_CLASS = {"mP": "P", "mR": "R", "mF1": "F1", "mIoU": "IoU",
                             "mNLL": "cNLL", "mBrier": "cBrier"}
PER_CLASS_METRICS = frozenset(MACRO_METRIC_TO_PER_CLASS.values())

# The iRAP protocol.
IRAP_ATTRIBUTE_METRIC_NAMES = ("mF1", "mP", "mR", "MCC")
IRAP_MAIN_METRIC = "mF1"
IRAP_IGNORE_MISSING_CLASSES = True
IRAP_CLASS_SUPPORT_THRESHOLDS = (5, 10)


def get_irap_metric_names(min_class_supports: T.Sequence[int] = IRAP_CLASS_SUPPORT_THRESHOLDS,
                          output_kind: OutputKind = "probs") -> tuple[str, ...]:
    """The metric names of the iRAP protocol, attribute averages first.

    Args:
        min_class_supports: Support thresholds of the restricted versions of the main metric.
        output_kind: The metrics of predicted distributions are included for 'probs'.
    """
    is_probabilistic = output_kind == "probs"
    averaged = [*IRAP_ATTRIBUTE_METRIC_NAMES, "A"]  # The per-attribute names of the averages.
    per_attribute = ["mF1", "A", "n", "MCC"]
    if is_probabilistic:
        averaged += ["NLL", "Brier", "mNLL"]
        per_attribute += ["NLL", "Brier", "mNLL"]
    for n in min_class_supports:
        main_name = f"{IRAP_MAIN_METRIC}{SUPPORT_SUFFIX}{n}"
        averaged.append(main_name)
        per_attribute += [main_name, f"nc{SUPPORT_SUFFIX}{n}"]
        if is_probabilistic:
            averaged.append(f"mNLL{SUPPORT_SUFFIX}{n}")
    return (*map(get_attribute_average_name, averaged), *per_attribute)


# Parsing ##########################################################################################

class ParsedMetricName(T.NamedTuple):
    is_averaged: bool  # has the 'a' prefix
    name: str  # the per-attribute name, without the 'a' prefix
    base: str  # without the prefix and the support suffix, e.g. 'mF1'
    min_support: int | None  # the threshold of a '_suppN' suffix


def parse_metric_name(name: str) -> ParsedMetricName:
    """Parses an `[a]<base>[_suppN]` metric name.

    Raises:
        ValueError: For an unknown base name, a threshold on a metric that is not a mean over
            classes, a non-integer threshold, or a support-only metric without a threshold.
    """
    base, separator, threshold = name.partition(SUPPORT_SUFFIX)
    if separator and not threshold.isdigit():
        raise ValueError(f"Metric {name!r} has a non-integer support threshold {threshold!r}.")
    min_support = int(threshold) if separator else None
    is_averaged = False
    if base not in KNOWN_BASE_METRICS and base.startswith("a"):
        is_averaged, base = True, base[1:]
    if base not in KNOWN_BASE_METRICS:
        raise ValueError(f"Unknown metric {name!r}. Known base names:"
                         f" {sorted(KNOWN_BASE_METRICS)}.")
    if min_support is None and base in SUPPORT_ONLY_METRICS:
        raise ValueError(f"Metric {name!r} exists only with a support threshold,"
                         f" e.g. {base}{SUPPORT_SUFFIX}10.")
    if min_support is not None and base not in MACRO_METRIC_TO_PER_CLASS \
            and base not in SUPPORT_ONLY_METRICS:
        raise ValueError(f"Metric {name!r} restricts {base!r} by support, but {base!r} is not a"
                         f" mean over classes.")
    return ParsedMetricName(is_averaged=is_averaged, name=name[1:] if is_averaged else name,
                            base=base, min_support=min_support)


def get_attribute_average_name(name: str) -> str:
    """Returns the name of the attribute average of a per-attribute metric, e.g. 'amF1' for
    'mF1'.

    Raises:
        ValueError: For an invalid name (`parse_metric_name`) or an attribute average.
    """
    if parse_metric_name(name).is_averaged:
        raise ValueError(f"Metric {name!r} is already an attribute average.")
    return "a" + name


def is_per_class_metric(name: str) -> bool:
    """Whether the metric is a per-class array, e.g. 'F1'."""
    return parse_metric_name(name).base in PER_CLASS_METRICS


def is_lower_better(name: str) -> bool:
    """Whether lower values of the metric are better (`LOWER_IS_BETTER_METRICS`), e.g. for
    sorting methods."""
    return parse_metric_name(name).base in LOWER_IS_BETTER_METRICS


def add_per_attribute_metric_names(metric_names: T.Sequence[str]) -> tuple[str, ...]:
    """`metric_names` followed by the per-attribute name of each attribute average that they
    lack, e.g. 'NLL' for 'aNLL', so that `metrics.select_metric_attributes` can average the
    results over a subset of the attributes."""
    names = dict.fromkeys(metric_names)
    for name in metric_names:
        if (parsed := parse_metric_name(name)).is_averaged:
            names.setdefault(parsed.name)
    return tuple(names)


# Descriptions #####################################################################################

#: Base metric name -> its description (`describe_metric`), with the placeholders `{name}` for the
#: metric name, e.g. 'mF1_supp10', and `{classes}` for the classes of a mean over classes.
_BASE_METRIC_DESCRIPTIONS = {
    "A": "Accuracy ({name}): the fraction of labeled examples whose predicted class matches the"
         " label.",
    "P": "Precision ({name}): among examples predicted as this class, the fraction whose label is"
         " this class.",
    "R": "Recall ({name}): among examples whose label is this class, the fraction predicted as"
         " this class.",
    "F1": "{name}: the harmonic mean of this class's precision and recall.",
    "IoU": "Intersection over union ({name}): for this class, the number of examples both labeled"
           " and predicted as the class, divided by the number labeled or predicted as the class.",
    "mP": "Macro precision ({name}): the mean of precision over {classes}.",
    "mR": "Macro recall ({name}): the mean of recall over {classes}.",
    "mF1": "Macro F1 ({name}): the mean of F1 over {classes}.",
    "mIoU": "Mean intersection over union ({name}): the mean of IoU over {classes}.",
    "n": "{name}: the number of labeled examples.",
    "nc": "{name}: the number of {classes}.",
    "MCC": "Matthews correlation coefficient ({name}): the correlation between labels and predicted"
           " classes; 1 indicates perfect agreement and 0 indicates no correlation.",
    "kappa": "Cohen's kappa ({name}): the agreement of predicted classes with labels beyond"
             " chance, 1 for perfect predictions and 0 for agreement by chance.",
    "NLL": "Negative log-likelihood ({name}): the mean over examples of −ln of the predicted"
           " probability of the label. Infinite if a label gets probability 0.",
    "Brier": "Brier score ({name}): the mean, over examples, of the sum of squared differences"
             " between the predicted class probabilities and the one-hot label (from 0 to 2).",
    "cNLL": "Class NLL ({name}): the mean negative log-likelihood of examples whose label is this"
            " class.",
    "cBrier": "Class Brier score ({name}): the mean Brier score of examples whose label is this"
              " class.",
    "mNLL": "Class-balanced negative log-likelihood ({name}): the mean of class NLL over"
            " {classes}.",
    "mBrier": "Class-balanced Brier score ({name}): the mean of class Brier score over"
              " {classes}.",
    MANUAL_REVIEW_PROPORTION_METRIC: (
        "Manual review proportion ({name}): the fraction of labeled examples that a reviewer has"
        " to correct, starting with the least confident predictions, until mF1 reaches"
        f" {MANUAL_REVIEW_TARGET_MF1}. 0 if mF1 already reaches it."),
}
#: Metrics that count something, so that neither higher nor lower values are better.
_COUNT_METRICS = frozenset({"n", "nc"})


def describe_metric(name: str,
                    ignore_missing_classes: bool = IRAP_IGNORE_MISSING_CLASSES) -> str:
    """Describes a metric in words, e.g. 'Macro F1 (mF1): the mean of F1 over …', for a tooltip. An
    attribute average gets a sentence about the average, then a line break and the description of
    its per-attribute metric.

    Args:
        ignore_missing_classes: The setting of `metrics.compute_classification_metrics` that the
            metric was computed with.

    Raises:
        ValueError: For an invalid name (`parse_metric_name`).
    """
    parsed = parse_metric_name(name)
    if parsed.min_support is not None:
        classes = f"classes with at least {parsed.min_support} labeled examples"
    elif parsed.base in PROBABILISTIC_METRICS:
        classes = "labeled classes"
    elif ignore_missing_classes:
        classes = "classes that are labeled or predicted"
    else:
        classes = "all classes"
    description = _BASE_METRIC_DESCRIPTIONS[parsed.base].format(name=parsed.name, classes=classes)
    if parsed.base not in _COUNT_METRICS:
        description += " Lower is better." if is_lower_better(name) else " Higher is better."
    if not parsed.is_averaged:
        return description
    return f"{describe_attribute_average(parsed.name)}\n{description}"


def describe_attribute_average(name: str) -> str:
    """Describes the attribute average of a metric in a sentence, e.g. 'The mean over attributes of
    mF1, …' for `name='mF1'`.

    Args:
        name: The name of the per-attribute metric, or words for it, e.g. 'the metric'.
    """
    return f"The mean over attributes of {name}, leaving out those where it is undefined."
