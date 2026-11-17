"""The classes of the attributes of a model with their scores, for finding the classes that are
hard for the model: how well its predictions tell each class from the other classes of its
attribute, and what the class is confused with.

How well a class is told apart is measured one vs. rest, with measures whose value by chance is the
same for every class size (`compute_class_discrimination`), unlike F1, which a predictor that
ignores the input already gets between p and 2p / (1 + p) for a class of proportion p.
"""

import dataclasses as dc
import typing as T

import numpy as np
from irap_data.metadata import IGNORE_LABEL_INDEX


@dc.dataclass(frozen=True)
class ClassDiscrimination:
    """How well the predictions of a model tell each class of an attribute from its other classes.

    Both measures are 0.5 and 0 for every predictor that ignores the input, at any proportion of
    the class, e.g. also for one that always predicts the class. They are NaN for a class without
    labeled segments, and for one with all of them, which has no other classes to be told from.

    Attributes:
        auroc: (K,) float64 one-vs-rest area under the ROC curve of the predicted probability of
            each class: the probability that a segment labeled with the class gets a higher
            probability of it than a segment labeled with another class, with ties counting
            half. It does not depend on a decision threshold.
        informedness: (K,) float64 one-vs-rest informedness (Youden's J) of the predicted
            classes: the recall of the class minus the proportion of the segments of other classes
            that are predicted as it. For hard predictions without invalid ones, `auroc` is
            (1 + informedness) / 2.
    """

    auroc: np.ndarray
    informedness: np.ndarray


def _compute_average_ranks(values: np.ndarray) -> np.ndarray:
    """Returns the 1-based ranks of (N,) values, with the mean rank for equal values."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last_ranks = np.cumsum(counts)
    return (last_ranks - (counts - 1) / 2)[inverse]


def compute_class_discrimination(labels: np.ndarray, predicted_indices: np.ndarray,
                                 probs: np.ndarray) -> ClassDiscrimination:
    """Computes how well predictions tell each class of an attribute from its other classes.

    An invalid prediction counts as the first class and the uniform distribution, as in the scores
    (`null_policy='first_class'`).

    Args:
        labels: (N,) class indices, `IGNORE_LABEL_INDEX` for unlabeled segments.
        predicted_indices: (N,) predicted class indices, `IGNORE_LABEL_INDEX` for invalid
            predictions.
        probs: (N, K) predicted distributions, NaN rows for invalid predictions, one-hot for hard
            predictions.
    """
    num_classes = probs.shape[1]
    is_labeled = labels != IGNORE_LABEL_INDEX
    labels = labels[is_labeled]
    is_valid = predicted_indices[is_labeled] != IGNORE_LABEL_INDEX
    predicted_indices = np.where(is_valid, predicted_indices[is_labeled], 0)
    probs = np.where(is_valid[:, None], probs[is_labeled], 1 / num_classes)
    auroc, informedness = np.full(num_classes, np.nan), np.full(num_classes, np.nan)
    for k in range(num_classes):
        is_positive = labels == k
        num_positive = int(is_positive.sum())
        num_negative = len(labels) - num_positive
        if num_positive == 0 or num_negative == 0:
            continue
        positive_rank_sum = _compute_average_ranks(probs[:, k])[is_positive].sum()
        auroc[k] = ((positive_rank_sum - num_positive * (num_positive + 1) / 2)
                    / (num_positive * num_negative))
        is_predicted = predicted_indices == k
        informedness[k] = ((is_predicted & is_positive).sum() / num_positive
                           - (is_predicted & ~is_positive).sum() / num_negative)
    return ClassDiscrimination(auroc, informedness)


@dc.dataclass(frozen=True)
class Confusion:
    """The segments that are labeled with one class and predicted as another, the class of
    `irap_code` and `class_name`."""

    irap_code: int
    class_name: str
    num_segments: int


def _to_optional(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


@dc.dataclass(frozen=True)
class ClassScores:
    """A class of an attribute that has labeled segments, with the scores of a model.

    Attributes:
        class_index: The index of the class among the classes of the attribute.
        support: The number of segments labeled with the class.
        auroc, informedness: See `ClassDiscrimination`. None if they are undefined, for the only
            labeled class of an attribute.
        num_missed: The number of segments labeled with the class and predicted as another.
        largest_confusion: The confusion with the class that most of the missed segments are
            predicted as, the first of several such classes. None if no segment is missed.
    """

    attribute: str
    class_index: int
    irap_code: int
    class_name: str
    support: int
    f1: float
    auroc: float | None
    informedness: float | None
    num_missed: int
    largest_confusion: Confusion | None


def list_class_scores(
        class_reports: T.Mapping[str, T.Mapping[str, T.Any]],
        attribute_to_discrimination: T.Mapping[str, ClassDiscrimination]
) -> list[ClassScores]:
    """Lists the classes with labeled segments of all attributes of a model, in the order of the
    attributes and their classes.

    Args:
        class_reports: Attribute -> the scores of its classes, the `classes` of an
            `irap_evaluation` report (`evaluation_report.to_json_dict`), whose confusion matrices
            have the labels in rows and count an invalid prediction as the first class.
        attribute_to_discrimination: `compute_class_discrimination` of each attribute of
            `class_reports`.
    """
    class_scores = []
    for attribute, report in class_reports.items():
        discrimination = attribute_to_discrimination[attribute]
        confusion_matrix = np.array(report["confusion_matrix"])
        for k, support in enumerate(report["support"]):
            if support == 0:
                continue
            confusions = confusion_matrix[k].copy()
            confusions[k] = 0
            largest_confusion = None
            if confusions.any():
                predicted = int(confusions.argmax())
                largest_confusion = Confusion(report["irap_codes"][predicted],
                                              report["values"][predicted],
                                              int(confusions[predicted]))
            class_scores.append(ClassScores(
                attribute=attribute, class_index=k, irap_code=report["irap_codes"][k],
                class_name=report["values"][k], support=support, f1=report["f1"][k],
                auroc=_to_optional(discrimination.auroc[k]),
                informedness=_to_optional(discrimination.informedness[k]),
                num_missed=int(confusions.sum()), largest_confusion=largest_confusion))
    return class_scores
