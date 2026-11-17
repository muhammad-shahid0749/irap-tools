"""Class frequencies and other statistics of the splits of an iRAP release, from metadata only.

`irap_data.reports.statistics_report` formats them as HTML, JSON and CSV, and
`irap_data.reports.class_frequency_plot` plots them.
"""

import dataclasses as dc
import typing as T
from pathlib import Path

import numpy as np

from ..metadata import (
    IGNORE_LABEL_INDEX,
    ClassVocabulary,
    DatasetPreset,
    IRAPMetadata,
    compute_label_matrix,
    compute_num_segments_per_class,
    compute_split_labels,
    get_dataset_preset,
    get_segment_location,
    is_unlabeled_split,
    select_preset_split_segments,
)

# Segment selections ###############################################################################

SelectionKind = T.Literal["labeled", "with_context", "model"]

#: What each selection counts of a split.
SELECTION_DESCRIPTIONS: dict[SelectionKind, str] = {
    "labeled": "The segments that have an image and labels, as the loading preset of the release"
               " requires them (see `irap_data.metadata.compute_split_labels`): the dataset as"
               " annotated.",
    "with_context": "The labeled segments whose context window with the context offsets of the"
                    " release (`irap_data.DatasetPreset.reference_context_offsets`) is complete:"
                    " every offset stays inside the road sequence of the segment and has an"
                    " image. The segments near the ends of a road sequence are left out. A model"
                    " whose context offsets lie within the window can be scored on all of"
                    " them, so they are an evaluation set common to such models.",
    "model": "The labeled segments whose context window with the context offsets of a model is"
             " complete: the segments that the model is trained and evaluated on.",
}


def select_segments(
    metadata: IRAPMetadata,
    dataset_name: str,
    split: str,
    selection: SelectionKind,
    context_offsets: T.Sequence[int] | None = None,
) -> dict[str, list[int]]:
    """Selects the segments of `split` that `selection` counts (see `SELECTION_DESCRIPTIONS`).

    Args:
        dataset_name: A key of `DATASET_PRESETS`.
        context_offsets: The context offsets of the model, for the 'model' selection only.

    Returns:
        Segment ID -> class indices, in `splits.json` order.

    Raises:
        ValueError: If `context_offsets` are given for a selection other than 'model' or are
            missing for it.
    """
    if (selection == "model") != (context_offsets is not None):
        raise ValueError("Context offsets are needed for the 'model' selection and only for it.")
    if selection == "labeled":
        return compute_split_labels(
            metadata, split,
            allow_missing_attributes=get_dataset_preset(dataset_name).allow_missing_attributes)
    if selection in ("with_context", "model"):
        selected = select_preset_split_segments(metadata, dataset_name, split, context_offsets)
        return {sid: selected.segment_id_to_labels[sid] for sid in selected.segment_ids}
    raise ValueError(f"Unknown selection {selection!r}. Choose from {list(SELECTION_DESCRIPTIONS)}.")


def compute_sequence_indices(
    sequence_index: T.Mapping[str, tuple[str, int]],
    segment_ids: T.Sequence[str],
) -> np.ndarray:
    """Numbers the road sequences of segments in the order of their first segment.

    Args:
        sequence_index: Segment ID -> (road ID, position), e.g. `IRAPMetadata.sequence_index`.

    Returns:
        (N,) int64 sequence numbers in `[0, number of sequences)`. A segment that is in no road
        sequence is a sequence of its own.
    """
    key_to_index: dict[T.Hashable, int] = {}
    # A segment without a sequence is keyed by a tuple, which cannot equal a road ID (a string).
    keys = [sequence_index[sid][0] if sid in sequence_index else (sid,) for sid in segment_ids]
    return np.array([key_to_index.setdefault(key, len(key_to_index)) for key in keys],
                    dtype=np.int64)


# Class distributions ##############################################################################

def compute_num_sequences_per_class(
    class_indices: np.ndarray,
    sequence_indices: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Counts the road sequences that contain each class of one attribute.

    Args:
        class_indices: (N,) class indices, `IGNORE_LABEL_INDEX` where a segment has no label.
        sequence_indices: (N,) sequence numbers from `compute_sequence_indices`.
        num_classes: The number of classes of the attribute.

    Returns:
        (num_classes,) int64 array whose element `c` is the number of sequences with a segment
        of class `c`.
    """
    is_labeled = class_indices != IGNORE_LABEL_INDEX
    class_sequence_pairs = np.unique(
        np.stack([class_indices[is_labeled], sequence_indices[is_labeled]]), axis=1)
    return np.bincount(class_sequence_pairs[0], minlength=num_classes)


@dc.dataclass(frozen=True, eq=False)
class AttributeDistribution:
    """The class frequencies of one attribute in a set of segments.

    The derived measures are NaN for an attribute without a labeled segment.

    Attributes:
        attribute: The attribute name.
        values: (K,) the value names of the classes.
        irap_codes: (K,) the iRAP codes of the classes.
        num_segments: (K,) int64, the number of segments of each class.
        num_sequences: (K,) int64, the number of road sequences that contain each class.
            Consecutive segments of a sequence are strongly correlated, so this is closer to the
            number of independent examples than `num_segments`.
        num_missing: The number of segments without a label.
    """

    attribute: str
    values: tuple[str, ...]
    irap_codes: tuple[int, ...]
    num_segments: np.ndarray
    num_sequences: np.ndarray
    num_missing: int

    @property
    def num_classes(self) -> int:
        return len(self.values)

    @property
    def num_labeled(self) -> int:
        return int(self.num_segments.sum())

    @property
    def num_total(self) -> int:
        """The number of segments, labeled or not."""
        return self.num_labeled + self.num_missing

    @property
    def is_labeled(self) -> bool:
        return self.num_labeled > 0

    @property
    def num_absent_classes(self) -> int:
        return int((self.num_segments == 0).sum())

    @property
    def shares(self) -> np.ndarray:
        """(K,) the share of all segments, labeled or not, in each class."""
        if not self.num_total:
            return np.full(self.num_classes, np.nan)
        return self.num_segments / self.num_total

    @property
    def labeled_shares(self) -> np.ndarray:
        """(K,) the share of the labeled segments in each class. The class-balance measures
        use it."""
        if not self.is_labeled:
            return np.full(self.num_classes, np.nan)
        return self.num_segments / self.num_labeled

    @property
    def majority_share(self) -> float:
        """The labeled share of the most frequent class: the accuracy of predicting it."""
        return float(self.labeled_shares.max())

    @property
    def imbalance_ratio(self) -> float:
        """The ratio of the most to the least frequent class that has a segment."""
        observed = self.num_segments[self.num_segments > 0]
        return float(observed.max() / observed.min()) if observed.size else np.nan

    @property
    def effective_num_classes(self) -> float:
        """The exponential of the class entropy: K for K equally frequent classes, and 1 for a
        single class."""
        shares = self.labeled_shares[self.num_segments > 0]
        return float(np.exp(-(shares * np.log(shares)).sum())) if shares.size else np.nan


def compute_attribute_distributions(
    vocabulary: ClassVocabulary,
    labels: np.ndarray,
    sequence_indices: np.ndarray,
) -> tuple[AttributeDistribution, ...]:
    """Computes the class distribution of each attribute.

    Args:
        labels: (N, A) class indices in `vocabulary` order, `IGNORE_LABEL_INDEX` for no label.
        sequence_indices: (N,) sequence numbers from `compute_sequence_indices`.

    Returns:
        One distribution per attribute, in `vocabulary` order.
    """
    return tuple(
        AttributeDistribution(
            attribute=attr, values=vocabulary.get_values(attr),
            irap_codes=vocabulary.get_irap_codes(attr),
            num_segments=compute_num_segments_per_class(labels[:, i], num_classes),
            num_sequences=compute_num_sequences_per_class(labels[:, i], sequence_indices,
                                                          num_classes),
            num_missing=int((labels[:, i] == IGNORE_LABEL_INDEX).sum()))
        for i, (attr, num_classes) in enumerate(zip(vocabulary.attribute_names,
                                                    vocabulary.class_counts)))


def compute_total_variation_distance(a: AttributeDistribution, b: AttributeDistribution) -> float:
    """The total variation distance between the class distributions of two segment sets: the
    share of labels that would have to change class for the distributions to match. NaN if one
    of them has no label."""
    return float(0.5 * np.abs(a.labeled_shares - b.labeled_shares).sum())


def compute_majority_accuracy(train: AttributeDistribution,
                              evaluated: AttributeDistribution) -> float:
    """The accuracy on `evaluated` of predicting the most frequent class of `train`. NaN if one
    of them has no label."""
    if not (train.is_labeled and evaluated.is_labeled):
        return np.nan
    return float(evaluated.labeled_shares[np.argmax(train.num_segments)])


# Ordering #########################################################################################

_ATTRIBUTE_SORT_KEYS: dict[str, T.Callable[[AttributeDistribution], float]] = {
    "count": lambda d: -d.num_labeled,
    # An attribute without a labeled segment has no ratio. It goes last, as NaN would make the
    # order arbitrary.
    "imbalance": lambda d: (-d.imbalance_ratio if np.isfinite(d.imbalance_ratio)
                            else np.inf),
}

#: Orders of attributes. 'schema' is the vocabulary order, 'count' is by descending number of
#: labeled segments, and 'imbalance' is by descending `imbalance_ratio`.
ATTRIBUTE_ORDERS = ("schema", *_ATTRIBUTE_SORT_KEYS)

#: Orders of the classes of an attribute. 'schema' is the class-index order, which is meaningful
#: for ordinal attributes such as speed limits, and 'count' is by descending number of segments.
CLASS_ORDERS = ("schema", "count")


def sort_distributions(
    distributions: T.Sequence[AttributeDistribution],
    *,
    attribute_order: str = "schema",
    class_order: str = "schema",
) -> list[AttributeDistribution]:
    """Reorders attributes and the classes of each attribute. The sort is stable, so ties keep
    the schema order.

    Args:
        attribute_order: One of `ATTRIBUTE_ORDERS`.
        class_order: One of `CLASS_ORDERS`.

    Raises:
        ValueError: For an unknown order.
    """
    if attribute_order not in ATTRIBUTE_ORDERS:
        raise ValueError(f"{attribute_order=} is not one of {ATTRIBUTE_ORDERS}.")
    if class_order not in CLASS_ORDERS:
        raise ValueError(f"{class_order=} is not one of {CLASS_ORDERS}.")
    distributions = list(distributions)
    if class_order == "count":
        distributions = [_sort_classes_by_count(d) for d in distributions]
    if attribute_order != "schema":
        distributions.sort(key=_ATTRIBUTE_SORT_KEYS[attribute_order])
    return distributions


def _sort_classes_by_count(d: AttributeDistribution) -> AttributeDistribution:
    order = np.argsort(-d.num_segments, kind="stable")
    return dc.replace(d, values=tuple(d.values[i] for i in order),
                      irap_codes=tuple(d.irap_codes[i] for i in order),
                      num_segments=d.num_segments[order], num_sequences=d.num_sequences[order])


# Split and dataset statistics #####################################################################

@dc.dataclass(frozen=True, eq=False)
class SelectionStatistics:
    """Statistics of the segments of a split in one selection.

    Attributes:
        num_segments: The number of segments.
        num_unlabeled: The number of segments without a label for any attribute. Releases that
            allow missing attributes keep such segments.
        num_without_sequence: The number of segments that are in no road sequence of
            `road_id_to_segment_id_sequence.json`. Each is counted as a sequence of its own.
        sequence_lengths: (S,) int64, the number of segments of each road sequence.
        road_length_m: The summed length of the segments that have road data.
        distributions: One per attribute, in vocabulary order.
    """

    num_segments: int
    num_unlabeled: int
    num_without_sequence: int
    sequence_lengths: np.ndarray
    road_length_m: float
    distributions: tuple[AttributeDistribution, ...]

    @property
    def num_sequences(self) -> int:
        return len(self.sequence_lengths)


def compute_selection_statistics(
    metadata: IRAPMetadata,
    segment_id_to_labels: T.Mapping[str, T.Sequence[int]],
) -> SelectionStatistics:
    """Computes the statistics of the segments that `select_segments` selects."""
    segment_ids = tuple(segment_id_to_labels)
    vocabulary = metadata.vocabulary
    labels = compute_label_matrix(segment_id_to_labels, segment_ids,
                                  len(vocabulary.attribute_names))
    sequence_indices = compute_sequence_indices(metadata.sequence_index, segment_ids)
    road_data = metadata.segment_id_to_road_data
    return SelectionStatistics(
        num_segments=len(segment_ids),
        num_unlabeled=int((labels == IGNORE_LABEL_INDEX).all(axis=1).sum()),
        num_without_sequence=sum(sid not in metadata.sequence_index for sid in segment_ids),
        sequence_lengths=np.bincount(sequence_indices),
        road_length_m=float(sum(get_segment_location(road_data[sid]).length_m
                                for sid in segment_ids if sid in road_data)),
        distributions=compute_attribute_distributions(vocabulary, labels, sequence_indices))


@dc.dataclass(frozen=True)
class SplitStatistics:
    """Statistics of one split.

    Attributes:
        split: The split name.
        num_listed: The number of segments of the split in `splits.json`.
        num_with_image: The number of those that have an image.
        selections: Selection -> statistics. Empty for an unlabeled split.
        selection_to_segment_ids: Selection -> the segments in it, as a dataset of the split
            yields them, in `splits.json` order. An unlabeled split has no 'labeled' selection,
            and its selections with context hold its segments with a complete context window.
    """

    split: str
    num_listed: int
    num_with_image: int
    selections: T.Mapping[SelectionKind, SelectionStatistics]
    selection_to_segment_ids: T.Mapping[SelectionKind, tuple[str, ...]]

    @property
    def num_selected(self) -> dict[SelectionKind, int]:
        return {k: len(v) for k, v in self.selection_to_segment_ids.items()}


@dc.dataclass(frozen=True)
class DatasetStatistics:
    """Statistics of the splits of a release.

    Attributes:
        dataset: The release name, a key of `DATASET_PRESETS`.
        metadata_dir: The metadata directory.
        preset: The loading preset of the release.
        model_context_offsets: The context offsets of the 'model' selection, or None if it is not
            computed.
        splits: In `splits.json` order.
    """

    dataset: str
    metadata_dir: Path
    preset: DatasetPreset
    model_context_offsets: tuple[int, ...] | None
    splits: tuple[SplitStatistics, ...]

    @property
    def selections(self) -> tuple[SelectionKind, ...]:
        return _get_selections(self.model_context_offsets)

    def get_context_offsets(self, selection: SelectionKind) -> tuple[int, ...] | None:
        """The context offsets that `selection` requires a complete window for, if any."""
        return {"labeled": None, "with_context": self.preset.reference_context_offsets,
                "model": self.model_context_offsets}[selection]

    @property
    def is_model_selection_same_as_with_context(self) -> bool:
        """Whether there is a 'model' selection and it has the same segments as the
        'with_context' selection in every labeled split, so that its statistics repeat those of
        'with_context'."""
        return self.model_context_offsets is not None and all(
            s.selection_to_segment_ids["model"] == s.selection_to_segment_ids["with_context"]
            for s in self.splits if not is_unlabeled_split(s.split))

    def get_split_distributions(
        self, selection: SelectionKind,
    ) -> dict[str, tuple[AttributeDistribution, ...]]:
        """Split -> attribute distributions, for the splits with segments in `selection`."""
        return {s.split: s.selections[selection].distributions for s in self.splits
                if s.selections and s.selections[selection].num_segments}


def compute_dataset_statistics(
    metadata: IRAPMetadata,
    dataset_name: str,
    *,
    splits: T.Sequence[str] | None = None,
    model_context_offsets: T.Sequence[int] | None = None,
) -> DatasetStatistics:
    """Computes the statistics of the splits of a release in the 'labeled' and
    'with_context' selections, and in the 'model' selection if `model_context_offsets` are given.

    Args:
        dataset_name: A key of `DATASET_PRESETS`.
        splits: The splits to include, or None for all in `splits.json`.
        model_context_offsets: The context offsets of the 'model' selection.

    Raises:
        ValueError: For an unknown release or split.
    """
    preset = get_dataset_preset(dataset_name)
    splits = list(metadata.splits) if splits is None else list(splits)
    if unknown := [s for s in splits if s not in metadata.splits]:
        raise ValueError(f"Splits not in splits.json: {unknown}. Available:"
                         f" {', '.join(metadata.splits)}.")
    if model_context_offsets is not None:
        model_context_offsets = tuple(model_context_offsets)
    paths = metadata.segment_id_to_data_paths_rel

    def compute_split_statistics(split: str) -> SplitStatistics:
        listed = metadata.splits[split]
        is_unlabeled = is_unlabeled_split(split)
        selection_to_segments = {
            selection: select_segments(metadata, dataset_name, split, selection,
                                       model_context_offsets if selection == "model" else None)
            for selection in _get_selections(model_context_offsets)
            if not (is_unlabeled and selection == "labeled")}
        selections = {} if is_unlabeled else {
            selection: compute_selection_statistics(metadata, segment_id_to_labels)
            for selection, segment_id_to_labels in selection_to_segments.items()}
        return SplitStatistics(split=split, num_listed=len(listed),
                               num_with_image=sum(sid in paths for sid in listed),
                               selections=selections,
                               selection_to_segment_ids={
                                   k: tuple(v) for k, v in selection_to_segments.items()})

    return DatasetStatistics(dataset=dataset_name, metadata_dir=metadata.metadata_dir,
                             preset=preset, model_context_offsets=model_context_offsets,
                             splits=tuple(map(compute_split_statistics, splits)))


def _get_selections(model_context_offsets: tuple[int, ...] | None) -> tuple[SelectionKind, ...]:
    return ("labeled", "with_context") + (() if model_context_offsets is None else ("model",))
