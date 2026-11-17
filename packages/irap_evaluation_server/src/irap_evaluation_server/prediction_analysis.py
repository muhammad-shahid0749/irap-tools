"""The analysis of the predictions of models on an evaluation set, as on the Analysis page: the
outcome of each segment for an attribute, the confusion matrix, the segments of a matrix cell or
the missed segments of a class, and the details of a segment.

The predictions of a submission are analysed as `irap_evaluation.evaluate_predictions` scores them,
with the scoring settings (`align_predictions`), so that the counts agree with its scores.
"""

import collections
import dataclasses as dc
import functools
import threading
import typing as T

import irap_evaluation as ie
import numpy as np
from irap_data.metadata import IGNORE_LABEL_INDEX, IRAPMetadata

from .archive import ModelArchive, Submission
from .scoring import ScoringSettings, select_scored_attributes

#: The outcomes of the segments for one model, by outcome code (the `MODEL_*` constants).
MODEL_OUTCOME_NAMES = ("correct", "wrong", "invalid", "unlabeled")
MODEL_CORRECT, MODEL_WRONG, MODEL_INVALID, MODEL_UNLABELED = range(len(MODEL_OUTCOME_NAMES))
#: The outcomes of the segments for two models, A and B, by outcome code (the `COMPARISON_*`
#: constants). A model is correct on a segment if its prediction is valid and equals the label.
COMPARISON_OUTCOME_NAMES = ("both correct", "only A correct", "only B correct", "neither correct",
                            "unlabeled")
(COMPARISON_BOTH_CORRECT, COMPARISON_ONLY_A_CORRECT, COMPARISON_ONLY_B_CORRECT,
 COMPARISON_NEITHER_CORRECT, COMPARISON_UNLABELED) = range(len(COMPARISON_OUTCOME_NAMES))


@dc.dataclass(frozen=True)
class AlignedPredictions:
    """The predictions of a submission on an evaluation set (see `align_predictions`).

    Attributes:
        predictions: Of the scored attributes and the segments of the set, with the classes and
            segments in the order of the set.
        predicted_indices: Attribute -> (N,) int16 predicted class index of each segment of the
            set, `IGNORE_LABEL_INDEX` where the prediction is invalid.
    """

    submission: Submission
    evaluation_set: ie.EvaluationSet
    predictions: ie.Predictions
    predicted_indices: T.Mapping[str, np.ndarray]


def align_predictions(submission: Submission, predictions: ie.Predictions,
                      evaluation_set: ie.EvaluationSet,
                      settings: ScoringSettings) -> AlignedPredictions:
    """
    Args:
        predictions: The contents of the file of `submission`.

    Raises:
        irap_evaluation.PredictionFormatError: See `irap_evaluation.align_to_evaluation_set`.
    """
    aligned = ie.align_to_evaluation_set(
        predictions, evaluation_set, select_scored_attributes(evaluation_set, settings),
        missing_attribute_policy=settings.missing_attribute_policy)
    return AlignedPredictions(
        submission, evaluation_set, aligned,
        {a: v.astype(np.int16) for a, v in ie.to_class_indices(aligned).items()})


class AlignedPredictionsCache:
    """The aligned predictions that pages show, read from the prediction files on first use.

    It keeps the `max_size` most recently used ones, since a file of Vietnam train takes about
    0.3 s to read and align, and 10 MB of memory. Open pages also keep the predictions that they
    show. The cache is shared by the threads of the pages, and two threads that miss the same
    predictions both read them.
    """

    def __init__(self, archive: ModelArchive, settings: ScoringSettings, max_size: int = 8):
        self.archive = archive
        self.settings = settings
        self.max_size = max_size
        self._key_to_aligned: collections.OrderedDict[tuple, AlignedPredictions] = (
            collections.OrderedDict())
        self._lock = threading.Lock()

    def get(self, submission: Submission,
            evaluation_set: ie.EvaluationSet) -> AlignedPredictions:
        """
        Raises:
            irap_evaluation.PredictionFormatError: See `align_predictions`.
        """
        key = (submission.id, submission.file_sha256, evaluation_set.fingerprint)
        with self._lock:
            if (aligned := self._key_to_aligned.get(key)) is not None:
                self._key_to_aligned.move_to_end(key)
                return aligned
        predictions = ie.read_predictions(self.archive.get_predictions_path(submission.id))
        aligned = align_predictions(submission, predictions, evaluation_set, self.settings)
        with self._lock:
            self._key_to_aligned[key] = aligned
            while len(self._key_to_aligned) > self.max_size:
                self._key_to_aligned.popitem(last=False)
        return aligned


# Outcomes #########################################################################################

@dc.dataclass(frozen=True)
class AttributeOutcomes:
    """The outcomes of the segments of an evaluation set for one attribute.

    Attributes:
        outcome_names: `MODEL_OUTCOME_NAMES`, or `COMPARISON_OUTCOME_NAMES` for two models.
        segment_outcomes: (N,) int8 index into `outcome_names` of each segment of the set.
        confusion_matrix: (K, K + 1) int64 counts of the labeled segments for the first model:
            rows are labels, columns are predicted classes, and the last column counts invalid
            predictions. Added to the first column, the last one gives the matrix of
            `null_policy='first_class'`, which scores an invalid attribute prediction as class
            0.
    """

    outcome_names: tuple[str, ...]
    segment_outcomes: np.ndarray
    confusion_matrix: np.ndarray


def _get_num_classes(evaluation_set: ie.EvaluationSet, attribute: str) -> int:
    """Returns K, the number of classes of an attribute, also of the aligned predictions."""
    return len(evaluation_set.vocabulary.get_irap_codes(attribute))


def get_predicted_columns(aligned: AlignedPredictions, attribute: str) -> np.ndarray:
    """Returns (N,) the confusion-matrix column of each segment: the predicted class index, or K
    for an invalid prediction."""
    num_classes = _get_num_classes(aligned.evaluation_set, attribute)
    predicted = aligned.predicted_indices[attribute]
    return np.where(predicted == IGNORE_LABEL_INDEX, num_classes, predicted)


def compute_attribute_outcomes(aligned_a: AlignedPredictions, attribute: str,
                               aligned_b: AlignedPredictions | None = None) -> AttributeOutcomes:
    """
    Raises:
        ValueError: If `aligned_b` is of another evaluation set.
    """
    evaluation_set = aligned_a.evaluation_set
    labels = evaluation_set.class_indices[attribute]
    is_labeled = labels != IGNORE_LABEL_INDEX
    predicted_a = aligned_a.predicted_indices[attribute]
    # Invalid predictions are IGNORE_LABEL_INDEX, so they are correct only on unlabeled segments,
    # whose outcome is 'unlabeled'.
    is_correct_a = predicted_a == labels
    if aligned_b is None:
        outcome_names = MODEL_OUTCOME_NAMES
        outcomes = np.select([~is_labeled, predicted_a == IGNORE_LABEL_INDEX, is_correct_a],
                             [MODEL_UNLABELED, MODEL_INVALID, MODEL_CORRECT], MODEL_WRONG)
    else:
        if aligned_b.evaluation_set.fingerprint != evaluation_set.fingerprint:
            raise ValueError("The predictions are of different evaluation sets.")
        is_correct_b = aligned_b.predicted_indices[attribute] == labels
        outcome_names = COMPARISON_OUTCOME_NAMES
        outcomes = np.select(
            [~is_labeled, is_correct_a & is_correct_b, is_correct_a, is_correct_b],
            [COMPARISON_UNLABELED, COMPARISON_BOTH_CORRECT, COMPARISON_ONLY_A_CORRECT,
             COMPARISON_ONLY_B_CORRECT], COMPARISON_NEITHER_CORRECT)
    num_columns = _get_num_classes(evaluation_set, attribute) + 1
    cells = (labels[is_labeled] * num_columns
             + get_predicted_columns(aligned_a, attribute)[is_labeled])
    confusion_matrix = np.bincount(cells, minlength=(num_columns - 1) * num_columns).reshape(
        num_columns - 1, num_columns)
    return AttributeOutcomes(outcome_names=outcome_names,
                             segment_outcomes=outcomes.astype(np.int8),
                             confusion_matrix=confusion_matrix)


def select_cell_segments(aligned: AlignedPredictions, attribute: str, label_index: int,
                         predicted_index: int | None) -> np.ndarray:
    """Selects the segments of the set in a cell of the confusion matrix
    (`AttributeOutcomes.confusion_matrix`), whose `predicted_index` K is the invalid column, or,
    for `predicted_index` None, the missed segments of a class: those in its row but not in its
    correct predictions.

    For probabilistic predictions, the ones with the most confident valid predictions come
    first, so that errors are listed from the confident ones. Otherwise, and for invalid ones,
    which come last, the segments are in the order of the set.

    Returns:
        (M,) int64 indices into the segments of the set.
    """
    labels = aligned.evaluation_set.class_indices[attribute]
    columns = get_predicted_columns(aligned, attribute)
    is_in_cell = (columns != label_index if predicted_index is None
                  else columns == predicted_index)
    indices = np.flatnonzero((labels == label_index) & is_in_cell)
    if aligned.predictions.output_kind == "probs":
        num_classes = _get_num_classes(aligned.evaluation_set, attribute)
        cell_columns = columns[indices]
        is_valid = cell_columns < num_classes
        confidences = np.full(len(indices), -np.inf)
        confidences[is_valid] = aligned.predictions.probs[attribute][indices[is_valid],
                                                                     cell_columns[is_valid]]
        indices = indices[np.argsort(-confidences, kind="stable")]
    return indices


# Segments #########################################################################################

@dc.dataclass(frozen=True)
class MapPoints:
    """The segments of an evaluation set that have a location
    (`irap_data.metadata.IRAPMetadata.segment_coordinates`).

    Attributes:
        segment_indices: (P,) int64 indices into the segments of the set.
        coordinates: (P, 2) float64 latitude and longitude in degrees.
        num_segments: The number of segments of the set.
    """

    segment_indices: np.ndarray
    coordinates: np.ndarray
    num_segments: int

    @property
    def num_unlocated(self) -> int:
        return self.num_segments - len(self.segment_indices)

    @functools.cached_property
    def segment_to_point(self) -> np.ndarray:
        """(N,) int64 the point of each segment of the set, -1 if it has no location."""
        segment_to_point = np.full(self.num_segments, -1, dtype=np.int64)
        segment_to_point[self.segment_indices] = np.arange(len(self.segment_indices))
        return segment_to_point


def compute_map_points(metadata: IRAPMetadata, evaluation_set: ie.EvaluationSet) -> MapPoints:
    segment_coordinates = metadata.segment_coordinates
    pairs = [(i, segment_coordinates[s]) for i, s in enumerate(evaluation_set.segment_ids)
             if s in segment_coordinates]
    return MapPoints(segment_indices=np.array([i for i, _ in pairs], dtype=np.int64),
                     coordinates=np.array([c for _, c in pairs], dtype=np.float64).reshape(-1, 2),
                     num_segments=evaluation_set.num_segments)


def get_context_segment_ids(metadata: IRAPMetadata, segment_id: str,
                            context_offsets: T.Sequence[int]) -> list[str | None]:
    """Returns the ids of the segments at the context offsets in the road sequence of a segment,
    None outside the sequence."""
    if segment_id not in metadata.sequence_index:
        return [segment_id if offset == 0 else None for offset in context_offsets]
    road_id, position = metadata.sequence_index[segment_id]
    sequence = metadata.road_id_to_segment_id_sequence[road_id]
    return [sequence[position + o] if 0 <= position + o < len(sequence) else None
            for o in context_offsets]


@dc.dataclass(frozen=True)
class ModelPrediction:
    """
    Attributes:
        label: The name of the model on the page, e.g. 'Model A'.
        probs: (K,) float32 distribution in the class order of the evaluation set, None if the
            prediction is invalid.
    """

    label: str
    probs: np.ndarray | None

    @property
    def predicted_index(self) -> int | None:
        return None if self.probs is None else int(self.probs.argmax())


@dc.dataclass(frozen=True)
class SegmentDetail:
    """A segment of an evaluation set with its label and the predictions of models for one
    attribute.

    Attributes:
        road_id: None if the segment is in no road sequence.
        position: The position in the road sequence, None if it is in none.
        coordinates: Latitude and longitude in degrees, None if the segment has no location.
        label_index: None if the segment has no label.
        context_segment_ids: (offset, segment id or None outside the road sequence), by offset,
            for the context offsets of the first model, or (0,) if they are unknown.
    """

    segment_id: str
    road_id: str | None
    position: int | None
    coordinates: tuple[float, float] | None
    class_names: tuple[str, ...]
    irap_codes: tuple[int, ...]
    label_index: int | None
    model_predictions: tuple[ModelPrediction, ...]
    context_segment_ids: tuple[tuple[int, str | None], ...]


def get_segment_detail(metadata: IRAPMetadata,
                       labeled_aligned_seq: T.Sequence[tuple[str, AlignedPredictions]],
                       segment_index: int, attribute: str) -> SegmentDetail:
    """
    Args:
        labeled_aligned_seq: (`ModelPrediction.label`, predictions) of models on the same
            evaluation set, at least one.
        segment_index: Into the segments of the set.
    """
    _, first_aligned = labeled_aligned_seq[0]
    evaluation_set = first_aligned.evaluation_set
    segment_id = evaluation_set.segment_ids[segment_index]
    road_id, position = metadata.sequence_index.get(segment_id, (None, None))
    label_index = int(evaluation_set.class_indices[attribute][segment_index])
    offsets = sorted(first_aligned.submission.context_offsets or (0,))
    return SegmentDetail(
        segment_id=segment_id, road_id=road_id, position=position,
        coordinates=metadata.segment_coordinates.get(segment_id),
        class_names=evaluation_set.vocabulary.get_values(attribute),
        irap_codes=evaluation_set.vocabulary.get_irap_codes(attribute),
        label_index=None if label_index == IGNORE_LABEL_INDEX else label_index,
        model_predictions=tuple(
            ModelPrediction(label,
                            aligned.predictions.probs[attribute][segment_index]
                            if aligned.predictions.is_valid[attribute][segment_index] else None)
            for label, aligned in labeled_aligned_seq),
        context_segment_ids=tuple(zip(offsets, get_context_segment_ids(metadata, segment_id,
                                                                       offsets))))
