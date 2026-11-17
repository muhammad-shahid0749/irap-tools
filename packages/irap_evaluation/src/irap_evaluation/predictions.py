"""Model predictions for iRAP road segments: the `Predictions` datatype and conversions.

Classes are identified by iRAP code, not by class index, so predictions of models trained with
differently ordered metadata can be compared and combined (see `align_classes`).
"""

import dataclasses as dc
import functools
import typing as T

import numpy as np

from irap_data.metadata import IGNORE_LABEL_INDEX, validate_irap_codes

#: 'probs': predicted class distributions. 'hard': predicted classes without probabilities, e.g.
#: parsed from generated text, for which the probabilistic metrics are not computed.
OutputKind = T.Literal["probs", "hard"]
OUTPUT_KINDS: tuple[str, ...] = T.get_args(OutputKind)

#: The iRAP code of an invalid prediction in `Predictions.from_irap_codes` and `to_irap_codes`.
INVALID_IRAP_CODE = -1

#: Tolerance of the sum of a stored float32 distribution.
PROBABILITY_SUM_EPS = 1e-4


class PredictionFormatError(ValueError):
    """Predictions that do not satisfy the invariants of `Predictions` or the file format, or
    that do not match the other predictions or the evaluation set they are used with."""


#: What a model used the segments of a split for (`ModelInfo.get_split_use`):
#: - 'training': they were used to fit the model, so its scores on them are not of held-out data.
#: - 'early_stopping': they were used only for early stopping or checkpoint selection, so its
#:   scores on them are optimistic.
#: - 'unknown': the training splits of the model are unknown, so it may have been fitted on them.
#: - 'held_out': they were not used.
SplitUse = T.Literal["training", "early_stopping", "unknown", "held_out"]


def explain_split_use(split_use: SplitUse, split: str) -> str | None:
    """Explains what it means for the scores of a model on `split` that it used the split, or
    may have used it. None for 'held_out'."""
    match split_use:
        case "training":
            return (f"The model was trained on {split}, so its scores on it are not of held-out"
                    f" data.")
        case "early_stopping":
            return (f"{split} was used for early stopping or checkpoint selection of the model, so"
                    f" its scores on it are optimistic.")
        case "unknown":
            return (f"The training splits of the model are unknown, so its scores on {split} may"
                    f" not be of held-out data.")
        case "held_out":
            return None


def format_model_label(method_name: str, seed: int | None) -> str:
    """The label of a model in reports (`ModelInfo.label`): `<method>` or `<method>/seed<seed>`."""
    return method_name if seed is None else f"{method_name}/seed{seed}"


def _to_split_names(splits: T.Iterable[str], field: str) -> tuple[str, ...]:
    """The split names in sorted order, so that their order does not matter."""
    if isinstance(splits, str):
        raise PredictionFormatError(f"{field} must be a sequence of split names, not a string.")
    splits = tuple(splits)
    if not all(isinstance(s, str) for s in splits):
        raise PredictionFormatError(f"{field} must be split names, got {splits}.")
    if len(set(splits)) != len(splits):
        raise PredictionFormatError(f"{field} {splits} have duplicates.")
    return tuple(sorted(splits))


@dc.dataclass(frozen=True)
class ModelInfo:
    """What produced the predictions: one model of a method, e.g. one trained network.

    Two `ModelInfo`s are equal if they describe the same model: `details` are not compared. The
    split names are stored sorted, so their order does not matter.

    Attributes:
        method_name: Identifies the method, e.g. in file names and reports. It may be long, e.g.
            if it encodes the training configuration. The models with the same method name are
            models of one method, e.g. trained with different seeds, whose spread the bootstrap
            includes (see `bootstrap`).
        training_splits: The splits of the dataset whose segments (images or labels) were used to
            fit the model, () for none, e.g. for a zero-shot model, or None if unknown. Other
            training data belongs in `details`.
        early_stopping_splits: The splits that were used only for early stopping or checkpoint
            selection, () for none.
        seed: Identifies the model among the models of the method, e.g. the training seed, or None
            for a method with a single model.
        method_display_name: A human-readable name of the method, shown instead of the method
            name (`shown_method_name`), or None. It is stored stripped. It does not identify the
            method, so it is not compared, and the models of a method may have different ones.
        details: JSON-compatible free-form details, e.g. the checkpoint, the code commit
            (conventionally under 'commit') or the training configuration. They describe how the
            predictions were made, not which model made them, so they are not compared.

    Raises:
        PredictionFormatError: If a split field has duplicates or is not a sequence of strings, a
            split is both a training and an early stopping split, or the method display name is
            blank.
    """

    method_name: str
    training_splits: tuple[str, ...] | None
    early_stopping_splits: tuple[str, ...]
    seed: int | None = None
    method_display_name: str | None = dc.field(default=None, compare=False)
    details: T.Mapping[str, T.Any] | None = dc.field(default=None, compare=False)

    def __post_init__(self):
        if self.method_display_name is not None:
            if not (display_name := self.method_display_name.strip()):
                raise PredictionFormatError("The method display name must not be blank.")
            object.__setattr__(self, "method_display_name", display_name)
        training = (None if self.training_splits is None
                    else _to_split_names(self.training_splits, "The training splits"))
        early_stopping = _to_split_names(self.early_stopping_splits, "The early stopping splits")
        if both := set(training or ()) & set(early_stopping):
            raise PredictionFormatError(f"The splits {sorted(both)} are both training and early"
                                        f" stopping splits.")
        object.__setattr__(self, "training_splits", training)
        object.__setattr__(self, "early_stopping_splits", early_stopping)

    @property
    def label(self) -> str:
        return format_model_label(self.method_name, self.seed)

    @property
    def shown_method_name(self) -> str:
        """The method display name, or the method name if there is none."""
        return self.method_display_name or self.method_name

    def get_split_use(self, split: str) -> SplitUse:
        """Returns what the model used the segments of `split` for. Other training data in
        `details` is not considered."""
        if self.training_splits is None:
            return "unknown"
        if split in self.training_splits:
            return "training"
        return "early_stopping" if split in self.early_stopping_splits else "held_out"

    def to_json_dict(self) -> dict[str, T.Any]:
        """The JSON object of the model in the header of a prediction file
        (`docs/prediction_format.md`), also used for the members of an ensemble."""
        return {"method_name": self.method_name, "method_display_name": self.method_display_name,
                "seed": self.seed,
                "training_splits": (None if self.training_splits is None
                                    else list(self.training_splits)),
                "early_stopping_splits": list(self.early_stopping_splits),
                "details": None if self.details is None else dict(self.details)}


def check_split_names(model: ModelInfo, splits: T.Collection[str]) -> None:
    """Checks that the training and early stopping splits of `model` are among `splits`, e.g. the
    splits of the dataset's metadata (`IRAPMetadata.splits`).

    Raises:
        PredictionFormatError: For an unknown split.
    """
    named = [*(model.training_splits or ()), *model.early_stopping_splits]
    if unknown := [s for s in named if s not in splits]:
        raise PredictionFormatError(f"The model {model.label!r} names unknown splits {unknown}."
                                    f" The splits are {sorted(splits)}.")


@dc.dataclass(frozen=True)
class PredictionHeader:
    """The origin of a set of predictions: the model and the segments it was run on.

    Attributes:
        dataset: The iRAP release, a key of `irap_data.DATASET_PRESETS` ('bh', 'vietnam').
        split: The split the segments come from, e.g. 'val'.
        model: See `ModelInfo`.
        context_offsets: The distinct positions of the segments whose images the model reads,
            relative to the predicted segment in its road sequence, e.g. (0, -1, -4), where -1
            is the previous segment in driving order. With the dataset and split, they define the
            model's own evaluation set. None if unknown.

    Raises:
        PredictionFormatError: If `context_offsets` has duplicates.
    """

    dataset: str
    split: str
    model: ModelInfo
    context_offsets: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.context_offsets is not None:
            offsets = tuple(self.context_offsets)
            if len(set(offsets)) != len(offsets):
                raise PredictionFormatError(f"The context offsets {offsets} have duplicates.")
            object.__setattr__(self, "context_offsets", offsets)


def get_common_dataset_and_split(headers: T.Iterable[PredictionHeader]) -> tuple[str, str]:
    """The (dataset, split) that all headers share.

    Raises:
        PredictionFormatError: If they differ.
    """
    dataset_splits = {(h.dataset, h.split) for h in headers}
    if len(dataset_splits) != 1:
        raise PredictionFormatError(f"The predictions are of different releases or splits:"
                                    f" {sorted(dataset_splits)}.")
    return dataset_splits.pop()


@dc.dataclass(frozen=True)
class Predictions:
    """Per-attribute class distributions predicted for road segments.

    An attribute prediction is the prediction of one attribute for one segment, a row of
    `probs[attribute]`. It is invalid if the model gave no usable prediction, e.g. an unparseable
    VLM answer. The row of an invalid attribute prediction is NaN.

    Attributes:
        header: See `PredictionHeader`.
        output_kind: See `OutputKind`.
        attribute_to_irap_codes: Attribute -> the iRAP codes of its classes, in the column order
            of `probs`, which need not match the dataset's order (see
            `ClassVocabulary.attribute_to_irap_codes`). Its keys are the predicted attributes.
        segment_ids: (N,) unique segment IDs, the row order of `probs`.
        probs: Attribute -> (N, K) float32 distributions over the classes of
            `attribute_to_irap_codes`, one-hot for hard predictions. Stored as read-only views,
            so that they stay consistent with the cached `is_valid`.

    Raises:
        PredictionFormatError: If there is no attribute, the arrays do not match
            `attribute_to_irap_codes` and `segment_ids`, or a row is neither NaN nor a
            distribution (a one-hot one for hard predictions).
    """

    header: PredictionHeader
    output_kind: OutputKind
    attribute_to_irap_codes: T.Mapping[str, tuple[int, ...]]
    segment_ids: tuple[str, ...]
    probs: T.Mapping[str, np.ndarray]

    def __post_init__(self):
        object.__setattr__(self, "attribute_to_irap_codes",
                           {attr: tuple(codes)
                            for attr, codes in self.attribute_to_irap_codes.items()})
        object.__setattr__(self, "segment_ids", tuple(self.segment_ids))
        object.__setattr__(self, "probs", _to_read_only_views(self.probs))
        _validate_predictions(self)

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(self.attribute_to_irap_codes)

    @property
    def num_segments(self) -> int:
        return len(self.segment_ids)

    @functools.cached_property
    def is_valid(self) -> dict[str, np.ndarray]:
        """Attribute -> (N,) read-only bool, False for an invalid attribute prediction."""
        return _to_read_only_views({attr: ~np.isnan(self.probs[attr][:, 0])
                                    for attr in self.attributes})

    @classmethod
    def from_probabilities(
        cls,
        header: PredictionHeader,
        attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]],
        segment_ids: T.Sequence[str],
        probs: T.Mapping[str, np.ndarray],
        is_valid: T.Mapping[str, np.ndarray] | None = None,
    ) -> "Predictions":
        """Builds 'probs' predictions from distributions of any float type.

        Args:
            attribute_to_irap_codes: See `Predictions`.
            probs: Attribute -> (N, K) distributions in the class order of
                `attribute_to_irap_codes`.
            is_valid: Attribute -> (N,) bool, or None if all predictions are valid. The rows of
                invalid attribute predictions are ignored. A NaN in a valid row is an error, so
                that a numerical failure of the model does not pass for an invalid attribute
                prediction.
        """
        _check_attributes(probs, "probs", attribute_to_irap_codes)
        if is_valid is not None:
            _check_attributes(is_valid, "is_valid", attribute_to_irap_codes)

        def get_probs(attr: str) -> np.ndarray:
            values = np.asarray(probs[attr])
            attr_is_valid = (np.ones(len(values), dtype=bool) if is_valid is None
                             else np.asarray(is_valid[attr], dtype=bool))
            if np.isnan(values[attr_is_valid]).any():
                raise PredictionFormatError(f"{attr!r}: the distribution of a valid attribute"
                                            f" prediction contains NaN.")
            return _fill_invalid_rows(values, attr_is_valid)

        return cls(header, "probs", attribute_to_irap_codes, tuple(segment_ids),
                   {attr: get_probs(attr) for attr in probs})

    @classmethod
    def from_logits(
        cls,
        header: PredictionHeader,
        attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]],
        segment_ids: T.Sequence[str],
        logits: T.Mapping[str, np.ndarray],
        is_valid: T.Mapping[str, np.ndarray] | None = None,
    ) -> "Predictions":
        """Builds 'probs' predictions from (N, K) logits per attribute, as
        `from_probabilities`."""
        return cls.from_probabilities(header, attribute_to_irap_codes, segment_ids,
                                      {attr: _softmax(v) for attr, v in logits.items()},
                                      is_valid)

    @classmethod
    def from_class_indices(
        cls,
        header: PredictionHeader,
        attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]],
        segment_ids: T.Sequence[str],
        class_indices: T.Mapping[str, np.ndarray],
    ) -> "Predictions":
        """Builds 'hard' predictions from (N,) class indices per attribute.

        Args:
            attribute_to_irap_codes: See `Predictions`.
            class_indices: Attribute -> (N,) integer indices into the classes of
                `attribute_to_irap_codes`, `IGNORE_LABEL_INDEX` for an invalid prediction.

        Raises:
            PredictionFormatError: If the indices are not integers or not class indices.
        """
        _check_attributes(class_indices, "class_indices", attribute_to_irap_codes)

        def get_probs(attr: str) -> np.ndarray:
            num_classes = len(attribute_to_irap_codes[attr])
            indices = _to_integer_array(attr, class_indices[attr], "class indices")
            is_valid = indices != IGNORE_LABEL_INDEX
            if ((indices < 0) & is_valid).any() or (indices >= num_classes).any():
                raise PredictionFormatError(f"{attr!r}: class indices must be in"
                                            f" [0, {num_classes}) or {IGNORE_LABEL_INDEX}.")
            one_hot = np.eye(num_classes, dtype=np.float32)[np.where(is_valid, indices, 0)]
            return _fill_invalid_rows(one_hot, is_valid)

        return cls(header, "hard", attribute_to_irap_codes, tuple(segment_ids),
                   {attr: get_probs(attr) for attr in class_indices})

    @classmethod
    def from_irap_codes(
        cls,
        header: PredictionHeader,
        attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]],
        segment_ids: T.Sequence[str],
        irap_codes: T.Mapping[str, np.ndarray],
    ) -> "Predictions":
        """Builds 'hard' predictions from (N,) iRAP codes per attribute.

        Args:
            attribute_to_irap_codes: See `Predictions`. The classes the model chooses from.
            irap_codes: Attribute -> (N,) integer iRAP codes of the predicted classes,
                `INVALID_IRAP_CODE` for an invalid prediction.

        Raises:
            PredictionFormatError: If the codes are not integers, a code is not a class of its
                attribute, or `INVALID_IRAP_CODE` is the code of a class.
        """
        _check_attributes(irap_codes, "irap_codes", attribute_to_irap_codes)

        def get_class_indices(attr: str) -> np.ndarray:
            code_to_index = {code: i for i, code in enumerate(attribute_to_irap_codes[attr])}
            if INVALID_IRAP_CODE in code_to_index:
                raise PredictionFormatError(f"{attr!r}: {INVALID_IRAP_CODE} is the code of a"
                                            f" class, so it cannot mark invalid predictions.")
            distinct_codes, inverse = np.unique(
                _to_integer_array(attr, irap_codes[attr], "iRAP codes"), return_inverse=True)
            distinct_codes = distinct_codes.tolist()
            if unknown := sorted(set(distinct_codes) - set(code_to_index) - {INVALID_IRAP_CODE}):
                raise PredictionFormatError(f"{attr!r}: iRAP codes {unknown} are not among the"
                                            f" codes of the attribute, {sorted(code_to_index)}.")
            distinct_indices = [code_to_index.get(c, IGNORE_LABEL_INDEX) for c in distinct_codes]
            return np.array(distinct_indices, dtype=np.int64)[inverse.reshape(-1)]

        return cls.from_class_indices(header, attribute_to_irap_codes, segment_ids,
                                      {attr: get_class_indices(attr) for attr in irap_codes})


def _to_read_only_views(attribute_to_array: T.Mapping[str, T.Any]) -> dict[str, np.ndarray]:
    """Read-only views of the arrays. The arrays themselves stay writeable."""
    def to_read_only_view(array: T.Any) -> np.ndarray:
        view = np.asarray(array).view()
        view.flags.writeable = False
        return view

    return {attr: to_read_only_view(array) for attr, array in attribute_to_array.items()}


def _check_attributes(attribute_to_values: T.Mapping[str, T.Any], name: str,
                      attribute_to_irap_codes: T.Mapping[str, T.Any]) -> None:
    if set(attribute_to_values) != set(attribute_to_irap_codes):
        raise PredictionFormatError(
            f"The attributes of {name} differ from those of attribute_to_irap_codes: missing"
            f" {sorted(set(attribute_to_irap_codes) - set(attribute_to_values))}, extra"
            f" {sorted(set(attribute_to_values) - set(attribute_to_irap_codes))}.")


def _validate_attribute_irap_codes(attr: str, codes: T.Sequence[int]) -> None:
    """`irap_data.metadata.validate_irap_codes`, raising `PredictionFormatError`."""
    try:
        validate_irap_codes(attr, codes)
    except ValueError as e:
        raise PredictionFormatError(str(e)) from e


def _validate_predictions(predictions: Predictions) -> None:
    """Checks the invariants documented in `Predictions`."""
    p = predictions
    if p.output_kind not in OUTPUT_KINDS:
        raise PredictionFormatError(f"output_kind must be one of {OUTPUT_KINDS},"
                                    f" got {p.output_kind!r}.")
    if not p.attribute_to_irap_codes:
        raise PredictionFormatError("Predictions need at least one attribute.")
    num_segments = len(p.segment_ids)
    if not all(isinstance(sid, str) for sid in p.segment_ids):
        raise PredictionFormatError("Segment IDs must be strings.")
    if len(set(p.segment_ids)) != num_segments:
        raise PredictionFormatError("Segment IDs must be unique.")
    _check_attributes(p.probs, "probs", p.attribute_to_irap_codes)
    for attr, codes in p.attribute_to_irap_codes.items():
        _validate_attribute_irap_codes(attr, codes)
        probs = p.probs[attr]
        if probs.dtype != np.float32 or probs.shape != (num_segments, len(codes)):
            raise PredictionFormatError(
                f"{attr!r}: expected float32 probabilities of shape"
                f" {(num_segments, len(codes))}, got {probs.dtype} {probs.shape}.")
        is_nan = np.isnan(probs)
        is_invalid = is_nan.all(1)
        if (is_nan.any(1) & ~is_invalid).any():
            raise PredictionFormatError(f"{attr!r}: a row must be either NaN (an invalid attribute"
                                        f" prediction) or a distribution without NaN.")
        valid_probs = probs[~is_invalid]
        if not (np.isfinite(valid_probs).all() and (valid_probs >= 0).all()):
            raise PredictionFormatError(f"{attr!r}: probabilities must be finite and >= 0.")
        if not np.all(np.abs(valid_probs.sum(1, dtype=np.float64) - 1) <= PROBABILITY_SUM_EPS):
            raise PredictionFormatError(f"{attr!r}: each distribution must sum to 1.")
        if p.output_kind == "hard" and not (
                np.isin(valid_probs, (0, 1)).all() and (valid_probs.sum(1) == 1).all()):
            raise PredictionFormatError(f"{attr!r}: hard predictions must be one-hot.")


def _replace_unvalidated(predictions: Predictions, **changes) -> Predictions:
    """`dataclasses.replace` without `_validate_predictions`, for changes that keep the
    invariants: selecting segments or attributes, reordering classes, and adding invalid segments
    or attribute predictions.
    The checks scan every probability, which takes about a second for 100k segments and 50
    attributes."""
    if "probs" in changes:
        changes = {**changes, "probs": _to_read_only_views(changes["probs"])}
    replaced = object.__new__(Predictions)
    for field in dc.fields(Predictions):
        object.__setattr__(replaced, field.name,
                           changes.get(field.name, getattr(predictions, field.name)))
    return replaced


# Conversions ######################################################################################

def _fill_invalid_rows(probs: np.ndarray, is_valid: np.ndarray) -> np.ndarray:
    return np.where(is_valid[:, None], probs, np.float32(np.nan)).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis in float64."""
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(-1, keepdims=True))
    return exp / exp.sum(-1, keepdims=True)


def _to_integer_array(attr: str, values: T.Any, name: str) -> np.ndarray:
    """`values` as an int64 array.

    Raises:
        PredictionFormatError: If `values` is not empty and not of an integer type, since casting,
            e.g. of 2.7, would silently give another value.
    """
    array = np.asarray(values)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise PredictionFormatError(f"{attr!r}: {name} must be integers, got {array.dtype}.")
    return array.astype(np.int64)


def to_class_indices(predictions: Predictions) -> dict[str, np.ndarray]:
    """Attribute -> (N,) int64 argmax class indices, `IGNORE_LABEL_INDEX` where invalid."""
    return {attr: np.where(predictions.is_valid[attr],
                           np.nan_to_num(predictions.probs[attr], nan=0).argmax(1),
                           IGNORE_LABEL_INDEX)
            for attr in predictions.attributes}


def to_irap_codes(predictions: Predictions) -> dict[str, np.ndarray]:
    """Attribute -> (N,) int64 iRAP codes of the argmax classes, `INVALID_IRAP_CODE` where
    invalid."""
    return {attr: np.where(indices == IGNORE_LABEL_INDEX, INVALID_IRAP_CODE,
                           np.asarray(predictions.attribute_to_irap_codes[attr])[indices])
            for attr, indices in to_class_indices(predictions).items()}


def check_class_codes(predictions: Predictions,
                      attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]]) -> None:
    """Checks that the attributes of `attribute_to_irap_codes` are predicted with its iRAP codes,
    in any order.

    Raises:
        PredictionFormatError: If they are not, or `attribute_to_irap_codes` is empty.
    """
    predicted = predictions.attribute_to_irap_codes
    if not attribute_to_irap_codes:
        raise PredictionFormatError("Predictions need at least one attribute.")
    if missing := [a for a in attribute_to_irap_codes if a not in predicted]:
        raise PredictionFormatError(f"Attributes not predicted: {missing}.")
    for attr, codes in attribute_to_irap_codes.items():
        # Predicted codes are unique (`Predictions` validation).
        if len(codes) != len(predicted[attr]) or set(codes) != set(predicted[attr]):
            raise PredictionFormatError(
                f"{attr!r}: the predicted iRAP codes {sorted(predicted[attr])} differ from"
                f" {sorted(codes)}.")


def align_classes(predictions: Predictions,
                  attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]]) -> Predictions:
    """The predictions of the attributes of `attribute_to_irap_codes`, in its class order.

    Classes are matched by iRAP code. `ClassVocabulary.attribute_to_irap_codes` gives the class
    codes of a vocabulary.

    Raises:
        PredictionFormatError: See `check_class_codes`.
    """
    check_class_codes(predictions, attribute_to_irap_codes)
    predicted = predictions.attribute_to_irap_codes

    def get_column_order(attr: str) -> np.ndarray:
        code_to_index = {code: i for i, code in enumerate(predicted[attr])}
        return np.array([code_to_index[c] for c in attribute_to_irap_codes[attr]])

    return _replace_unvalidated(
        predictions,
        attribute_to_irap_codes={attr: tuple(codes)
                                 for attr, codes in attribute_to_irap_codes.items()},
        probs={attr: predictions.probs[attr][:, get_column_order(attr)]
               for attr in attribute_to_irap_codes})


def add_invalid_attributes(predictions: Predictions,
                           attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]]
                           ) -> Predictions:
    """The predictions with the attributes of `attribute_to_irap_codes` that they do not predict
    added, with every attribute prediction invalid. Predicted attributes are kept as they are.

    `docs/evaluation.md#missing-attributes` gives the reasons for scoring missing attributes so.

    Args:
        attribute_to_irap_codes: Attribute -> the iRAP codes of its classes, e.g. from
            `ClassVocabulary.attribute_to_irap_codes`.

    Raises:
        PredictionFormatError: If the iRAP codes of an added attribute are invalid
            (`irap_data.metadata.validate_irap_codes`).
    """
    missing = {attr: tuple(codes) for attr, codes in attribute_to_irap_codes.items()
               if attr not in predictions.attribute_to_irap_codes}
    for attr, codes in missing.items():
        _validate_attribute_irap_codes(attr, codes)
    if not missing:
        return predictions
    return _replace_unvalidated(
        predictions,
        attribute_to_irap_codes={**predictions.attribute_to_irap_codes, **missing},
        probs={**predictions.probs,
               **{attr: np.full((predictions.num_segments, len(codes)), np.nan, dtype=np.float32)
                  for attr, codes in missing.items()}})


def add_invalid_segments(predictions: Predictions, segment_ids: T.Iterable[str]) -> Predictions:
    """Appends the segments of `segment_ids` that the predictions do not predict, with every
    attribute prediction invalid. Predicted segments are kept as they are."""
    predicted = set(predictions.segment_ids)
    missing = tuple(dict.fromkeys(sid for sid in segment_ids if sid not in predicted))
    if not missing:
        return predictions
    return _replace_unvalidated(
        predictions, segment_ids=predictions.segment_ids + missing,
        probs={attr: np.concatenate([v, np.full((len(missing), v.shape[1]), np.nan, v.dtype)])
               for attr, v in predictions.probs.items()})


def select_segments(predictions: Predictions, segment_ids: T.Sequence[str]) -> Predictions:
    """The predictions of `segment_ids`, in that order.

    Raises:
        PredictionFormatError: If a segment has no prediction, or `segment_ids` has duplicates.
    """
    segment_ids = tuple(segment_ids)
    row_of = {sid: i for i, sid in enumerate(predictions.segment_ids)}
    if missing := [sid for sid in segment_ids if sid not in row_of]:
        raise PredictionFormatError(f"{len(missing)} of {len(segment_ids)} segments have no"
                                    f" prediction, e.g. {missing[:5]}.")
    if len(set(segment_ids)) != len(segment_ids):
        raise PredictionFormatError("Segment IDs must be unique.")
    rows = np.array([row_of[sid] for sid in segment_ids], dtype=np.int64)
    return _replace_unvalidated(predictions, segment_ids=segment_ids,
                                probs={attr: v[rows] for attr, v in predictions.probs.items()})
