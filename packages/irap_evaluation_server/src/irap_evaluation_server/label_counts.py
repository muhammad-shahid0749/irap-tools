"""Counts of the labels of attributes in an evaluation set: how many segments and classes the
metric values of an attribute are computed from. They describe the evaluation set, not the
predictions."""

import dataclasses as dc
import typing as T

import irap_evaluation as ie
import numpy as np
from irap_data.metadata import compute_num_segments_per_class


@dc.dataclass(frozen=True)
class AttributeLabelCounts:
    """The labels of an attribute in an evaluation set.

    Attributes:
        num_labeled_segments: The number of segments with a label of the attribute.
        num_labeled_classes: The number of classes of the attribute with at least one labeled
            segment.
        num_minority_class_segments: The number of labeled segments outside the majority class,
            the class with the most labeled segments. 0 if the attribute has no labels.
        num_smallest_class_segments: The number of labeled segments of the class with the fewest,
            among the classes with at least one. None if the attribute has no labels.
    """

    num_labeled_segments: int
    num_labeled_classes: int
    num_minority_class_segments: int
    num_smallest_class_segments: int | None


def count_attribute_labels(evaluation_set: ie.EvaluationSet,
                           attributes: T.Iterable[str]) -> dict[str, AttributeLabelCounts]:
    """Counts the labels of each attribute in an evaluation set.

    Raises:
        ValueError: See `irap_data.metadata.compute_num_segments_per_class`.
    """
    return {a: _summarize_class_supports(compute_num_segments_per_class(
                evaluation_set.class_indices[a], len(evaluation_set.vocabulary.get_irap_codes(a))))
            for a in attributes}


def _summarize_class_supports(class_supports: np.ndarray) -> AttributeLabelCounts:
    labeled_class_supports = class_supports[class_supports > 0]
    num_labeled_segments = int(labeled_class_supports.sum())
    return AttributeLabelCounts(
        num_labeled_segments=num_labeled_segments,
        num_labeled_classes=int(labeled_class_supports.size),
        num_minority_class_segments=(num_labeled_segments
                                     - int(labeled_class_supports.max(initial=0))),
        num_smallest_class_segments=(int(labeled_class_supports.min())
                                     if labeled_class_supports.size else None))
