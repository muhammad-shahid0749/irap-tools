"""The view settings that several pages share, mostly query parameters."""

import time
import typing as T

import irap_evaluation as ie
import regex  # Unlike `re`, it has timeouts, so that a user's pattern cannot block the server.

from ..datasets import EVALUATION_SET_NAMES, DatasetContext

#: The longest time that a name filter may take over all names (`select_matching_names`).
#: Plain filters take microseconds, while a catastrophic pattern, e.g. '^(\w|\w\w)*$', can take
#: minutes.
NAME_FILTER_TIMEOUT_S = 0.1


def parse_dataset(query: T.Mapping[str, str], dataset_contexts: T.Mapping[str, DatasetContext],
                  default: str | None = None) -> str:
    """Parses the `dataset` parameter.

    Args:
        default: The dataset without the parameter, by default the first one.

    Raises:
        ValueError: For an unknown dataset.
    """
    dataset = query.get("dataset") or default or next(iter(dataset_contexts))
    if dataset not in dataset_contexts:
        raise ValueError(f"Unknown dataset {dataset!r}. Datasets: {', '.join(dataset_contexts)}.")
    return dataset


def select_default_split(split_names: T.Sequence[str]) -> str:
    """Selects the split that a page shows without the `split` parameter: the first that is not
    "train", or the first if there is no other."""
    return next((s for s in split_names if s != "train"), split_names[0])


def parse_dataset_and_split(query: T.Mapping[str, str],
                            dataset_contexts: T.Mapping[str, DatasetContext]) -> tuple[str, str]:
    """Parses the `dataset` and `split` parameters of a page of any split. They default to the first
    dataset and its default split (`select_default_split`).

    Raises:
        ValueError: For an unknown dataset or split.
    """
    dataset = parse_dataset(query, dataset_contexts)
    context = dataset_contexts[dataset]
    split = query.get("split") or select_default_split(context.split_names)
    if split not in context.split_names:
        raise ValueError(f"The dataset {dataset!r} has no split {split!r}. Its splits:"
                         f" {', '.join(context.split_names)}.")
    return dataset, split


def parse_evaluation_set(query: T.Mapping[str, str]) -> str:
    """Parses the `set` parameter, which defaults to the reference set.

    Raises:
        ValueError: For an unknown evaluation set.
    """
    evaluation_set = query.get("set") or ie.REFERENCE_SET_NAME
    if evaluation_set not in EVALUATION_SET_NAMES:
        raise ValueError(f"Unknown evaluation set {evaluation_set!r}. Evaluation sets:"
                         f" {', '.join(EVALUATION_SET_NAMES)}.")
    return evaluation_set


def make_view_options(per_attribute_names: T.Iterable[str]) -> dict[str, str]:
    """Makes the options of the View select of a page with the attribute averages or one
    per-attribute metric (the `metric` parameter, `parse_per_attribute_metric`).

    Args:
        per_attribute_names: The shown per-attribute metrics
            (`method_ranking.ShownMetrics.per_attribute_names`).
    """
    return {"": "Attribute averages", **{m: f"Per attribute: {m}" for m in per_attribute_names}}


def parse_per_attribute_metric(query: T.Mapping[str, str],
                               per_attribute_names: T.Sequence[str]) -> str:
    """Parses the `metric` parameter: a per-attribute metric, or '' for the attribute averages.

    Args:
        per_attribute_names: The shown per-attribute metrics
            (`method_ranking.ShownMetrics.per_attribute_names`).

    Raises:
        ValueError: For a metric that is unknown or hidden.
    """
    metric_name = query.get("metric", "")
    if metric_name and metric_name not in per_attribute_names:
        raise ValueError(f"Unknown or hidden per-attribute metric {metric_name!r}. Metrics:"
                         f" {', '.join(per_attribute_names)}.")
    return metric_name


def select_matching_names(text: str, names: T.Iterable[str]) -> set[str]:
    """Selects the names that the case-insensitive regular expression of a name filter, e.g. of
    the methods, matches in any part. '' matches all names.

    Raises:
        ValueError: If `text` is not a valid regular expression, or matching it takes longer than
            `NAME_FILTER_TIMEOUT_S` in total.
    """
    try:
        pattern = regex.compile(text, regex.IGNORECASE)
    except regex.error as e:
        raise ValueError(f"Invalid regular expression {text!r}: {e}.") from None
    deadline_s = time.perf_counter() + NAME_FILTER_TIMEOUT_S
    matching = set()
    try:
        for name in names:
            remaining_s = deadline_s - time.perf_counter()
            if remaining_s <= 0:
                raise TimeoutError
            if pattern.search(name, timeout=remaining_s) is not None:
                matching.add(name)
    except TimeoutError:
        raise ValueError(f"The regular expression {text!r} takes too long. Simplify it, e.g."
                         f" without nested repetitions such as '(a+)+'.") from None
    return matching
