"""Torch-free access to iRAP dataset metadata: class vocabularies, labels and segment selection.

`IRAPDataset` is built on these functions, and so are tools that need the labels of a split
without loading images, such as offline evaluation.
"""

import dataclasses as dc
import functools
import json
import os
import re
import pickle
import typing as T
import warnings
from pathlib import Path

import numpy as np

IGNORE_LABEL_INDEX: int = -1


class MetaFiles:
    """File names relative to a dataset metadata directory."""

    ATTRIBUTE_METADATA = "attribute_metadata.json"
    SPLITS = "splits.json"
    SEGMENT_ID_TO_DATA_PATHS = "segment_id_to_data_paths_rel.json"
    SEGMENT_ID_TO_ROAD_DATA = "segment_id_to_road_data.json"
    ROAD_ID_TO_SEGMENT_ID_SEQUENCE = "road_id_to_segment_id_sequence.json"
    UNLABELED_SEGMENT_ID_TO_LOCATION = "unlabeled_segment_id_to_location.json"
    NCONTEXT_SUBDIR = "seg_to_res"


# Dataset presets ##################################################################################

@dc.dataclass(frozen=True)
class DatasetPreset:
    """Loading defaults of an iRAP release.

    Attributes:
        allow_missing_attributes: Whether a segment with a missing or unknown attribute code is
            kept, with `IGNORE_LABEL_INDEX` for that attribute, rather than dropped.
        use_ncontext_filter: Whether labeled splits are restricted to the precomputed N-context
            subsets (see `load_ncontext_segment_ids`).
        reference_context_offsets: Context offsets that define the reference evaluation set: the
            segments whose context window with these offsets is complete. A model whose context
            offsets lie within this window can predict every reference segment, so all such
            models are scored on the same segments.
    """

    allow_missing_attributes: bool
    use_ncontext_filter: bool
    reference_context_offsets: tuple[int, ...]


#: Presets by release name, the same names as `irap_dataset.IRAP_DATASET_FACTORIES`.
DATASET_PRESETS: dict[str, DatasetPreset] = {
    # The N-context subsets already require 10 labeled neighbours on each side.
    "bh": DatasetPreset(allow_missing_attributes=False, use_ncontext_filter=True,
                        reference_context_offsets=tuple(range(-10, 11))),
    # Vietnam road sequences are short (median 29 segments in the 2026-07-31 metadata), so the
    # window holds only the past context that the default model offsets (0, -1, -4) need.
    "vietnam": DatasetPreset(allow_missing_attributes=True, use_ncontext_filter=False,
                             reference_context_offsets=tuple(range(-4, 1))),
}


def get_dataset_preset(dataset_name: str) -> DatasetPreset:
    try:
        return DATASET_PRESETS[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown iRAP dataset {dataset_name!r}."
                         f" Choose from {sorted(DATASET_PRESETS)}.") from None


# Paths ############################################################################################

def resolve_datasets_root() -> Path:
    if v := os.environ.get("IRAP_HOME"):
        return Path(v)
    elif v := os.environ.get("DATASETS_PATH"):
        return Path(v)
    else:
        raise RuntimeError(
            "Cannot resolve datasets root directory. Please set the IRAP_HOME or DATASETS_PATH"
            " environment variable.")


#: Release name -> (dataset directory, metadata directory) under `resolve_datasets_root()`.
#: The Vietnam images and metadata share a directory.
_RELEASE_SUBDIRS: dict[str, tuple[str, str]] = {
    "bh": ("IRAP_BIH", "IRAP_BIH_METADATA"),
    "vietnam": ("IRAP_Vietnam", "IRAP_Vietnam"),
}


def get_default_irap_paths(dataset_name: str) -> tuple[Path, Path]:
    """The (dataset directory, metadata directory) of a release under `resolve_datasets_root()`.

    Raises:
        ValueError: If `dataset_name` is not a key of `DATASET_PRESETS`.
    """
    get_dataset_preset(dataset_name)
    root = resolve_datasets_root()
    dataset_subdir, metadata_subdir = _RELEASE_SUBDIRS[dataset_name]
    return root / dataset_subdir, root / metadata_subdir


def resolve_irap_paths(
    *,
    dataset_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolves iRAP-BH dataset and metadata directories.

    A directory that is not given is resolved with `get_default_irap_paths`.

    Returns:
        Tuple of (dataset_dir, metadata_dir) as Path objects.
    """
    if dataset_dir is None:
        dataset_dir = get_default_irap_paths("bh")[0]
    if metadata_dir is None:
        metadata_dir = get_default_irap_paths("bh")[1]
    return Path(dataset_dir), Path(metadata_dir)


def _load_json(path: Path) -> T.Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_optional_json(path: Path) -> T.Any:
    """Loads a JSON file, or returns an empty dict if it does not exist."""
    return _load_json(path) if path.exists() else {}


# Class vocabulary #################################################################################

def to_irap_code(value: T.Any) -> int | None:
    """Parses an iRAP code, which the metadata stores as an int (Vietnam) or a string (BH).

    Returns:
        The code, or None for a missing code: None, `'None'` or a negative number (the Vietnam
        tables write -1 for a blank cell).

    Raises:
        ValueError: For any other value, e.g. a float or `'N/A'`, which a metadata writer should
            not produce. Reading it as a missing code would silently drop the label.
    """
    if value is None or value == "None":
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        code = value
    elif isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        code = int(value)
    else:
        raise ValueError(f"Not an iRAP code: {value!r}. Codes are ints or digit strings, and"
                         f" None, 'None' or a negative number mark a missing code.")
    return code if code >= 0 else None


def validate_irap_codes(attribute: str, codes: T.Sequence[int]) -> None:
    """Checks the iRAP codes of the classes of an attribute.

    Raises:
        ValueError: If there are no codes, a code is not an int, or two codes are equal.
    """
    if not codes:
        raise ValueError(f"Attribute {attribute!r} has no classes.")
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in codes):
        raise ValueError(f"Attribute {attribute!r} has iRAP codes that are not ints: {codes}.")
    if len(set(codes)) != len(codes):
        raise ValueError(f"Attribute {attribute!r} has duplicate iRAP codes: {codes}.")


def _get_ordered_attributes(attr_meta: T.Mapping[str, T.Any]) -> list[str]:
    """The attributes of `attribute_metadata.json` content, in canonical order."""
    attr_to_idx = attr_meta["attribute_to_idx"]
    return sorted(attr_to_idx, key=attr_to_idx.__getitem__)


@dc.dataclass(frozen=True, eq=False)
class ClassVocabulary:
    """The attributes of a dataset and the classes of each, in class-index order.

    Class `i` of attribute `a` is the `i`-th entry of `attribute_to_value_to_irap_code[a]`: a
    value name and its iRAP code. Attributes are in the canonical order (`attribute_to_idx` of
    `attribute_metadata.json`), and classes in the order of `attribute_value_to_irap_number`,
    which is the order of the class indices in `IRAPDataset` targets. The
    `attribute_irap_number_to_class_idx` mapping that some metadata files also contain is not
    read: it numbers classes in iRAP-code order, which differs for the attributes whose values
    are not in code order (both land-use sides and 'Pedestrian crossing - inspected road').

    Two vocabularies are equal if they have the same attributes, values and codes in the same
    order, since the class order is the column order of predicted probabilities.

    Raises:
        ValueError: If an attribute has no classes or two classes with the same iRAP code.
    """

    attribute_to_value_to_irap_code: T.Mapping[str, T.Mapping[str, int]]

    def __post_init__(self):
        for attr, value_to_code in self.attribute_to_value_to_irap_code.items():
            validate_irap_codes(attr, list(value_to_code.values()))

    def _get_entries(self) -> tuple:
        return tuple((a, tuple(v.items())) for a, v in self.attribute_to_value_to_irap_code.items())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassVocabulary):
            return NotImplemented
        return self._get_entries() == other._get_entries()

    __hash__ = None  # The mappings are mutable.

    @classmethod
    def from_attribute_metadata(cls, attr_meta: T.Mapping[str, T.Any]) -> "ClassVocabulary":
        """Builds the vocabulary from the content of `attribute_metadata.json`."""
        value_to_number = attr_meta["attribute_value_to_irap_number"]

        def parse_code(attr, value, raw_code):
            code = to_irap_code(raw_code)
            if code is None:
                raise ValueError(f"Attribute {attr!r} value {value!r} has an invalid iRAP code"
                                 f" {raw_code!r}.")
            return code

        return cls({attr: {value: parse_code(attr, value, raw_code)
                           for value, raw_code in value_to_number[attr].items()}
                    for attr in _get_ordered_attributes(attr_meta)})

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return tuple(self.attribute_to_value_to_irap_code)

    @property
    def class_counts(self) -> tuple[int, ...]:
        return tuple(len(v) for v in self.attribute_to_value_to_irap_code.values())

    @property
    def attribute_to_irap_codes(self) -> dict[str, tuple[int, ...]]:
        """Attribute -> the iRAP codes of its classes, in class-index order."""
        return {attr: tuple(value_to_code.values())
                for attr, value_to_code in self.attribute_to_value_to_irap_code.items()}

    @property
    def attr_to_value_to_class_idx(self) -> dict[str, dict[str, int]]:
        return {attr: {value: i for i, value in enumerate(value_to_code)}
                for attr, value_to_code in self.attribute_to_value_to_irap_code.items()}

    def get_values(self, attribute: str) -> tuple[str, ...]:
        return tuple(self.attribute_to_value_to_irap_code[attribute])

    def get_irap_codes(self, attribute: str) -> tuple[int, ...]:
        return tuple(self.attribute_to_value_to_irap_code[attribute].values())

    def get_irap_code_to_class_index(self, attribute: str) -> dict[int, int]:
        return {code: i for i, code in enumerate(self.get_irap_codes(attribute))}

    def restrict_to_attributes(self, attributes: T.Iterable[str]) -> "ClassVocabulary":
        """The vocabulary of `attributes`, in the given order.

        Raises:
            KeyError: If an attribute is not in this vocabulary.
        """
        attributes = tuple(attributes)
        unknown = [a for a in attributes if a not in self.attribute_to_value_to_irap_code]
        if unknown:
            raise KeyError(f"Attributes not in the vocabulary: {unknown}.")
        return ClassVocabulary({a: self.attribute_to_value_to_irap_code[a] for a in attributes})


def load_attribute_metadata(
    metadata_dir: str | Path,
) -> tuple[list[str], dict[str, dict[str, T.Any]]]:
    """Loads iRAP attribute metadata and returns attributes in canonical order.

    Args:
        metadata_dir: Metadata directory.

    Returns:
        ordered_attrs: Attribute names ordered by their index in the metadata.
        attribute_value_to_irap_number: Mapping attr -> {value -> irap_number}, with the iRAP
            numbers as stored in the file. `ClassVocabulary` parses them.
    """
    attr_meta = _load_json(Path(metadata_dir) / MetaFiles.ATTRIBUTE_METADATA)
    return _get_ordered_attributes(attr_meta), attr_meta["attribute_value_to_irap_number"]


def load_class_vocabulary(metadata_dir: str | Path) -> ClassVocabulary:
    return ClassVocabulary.from_attribute_metadata(
        _load_json(Path(metadata_dir) / MetaFiles.ATTRIBUTE_METADATA))


def get_class_counts(metadata_dir: str | Path) -> tuple[int, ...]:
    """Returns a tuple containing the number of classes for each attribute in canonical order."""
    return load_class_vocabulary(metadata_dir).class_counts


def get_bh_class_counts(metadata_dir: str | Path | None = None) -> tuple[int, ...]:
    _, metadata_dir = resolve_irap_paths(metadata_dir=metadata_dir)
    return get_class_counts(metadata_dir)


def get_bih_class_counts(metadata_dir: str | Path | None = None) -> tuple[int, ...]:
    """Deprecated: use `get_bh_class_counts`."""
    warnings.warn("get_bih_class_counts is deprecated. Use get_bh_class_counts.",
                  DeprecationWarning, stacklevel=2)
    return get_bh_class_counts(metadata_dir)


# Metadata files ###################################################################################

@dc.dataclass(frozen=True)
class IRAPMetadata:
    """The content of a dataset metadata directory.

    Attributes:
        metadata_dir: The directory the files were read from.
        vocabulary: From `attribute_metadata.json`.
        splits: Split name -> segment IDs, from `splits.json`.
        segment_id_to_data_paths_rel: Segment ID -> modality -> path relative to the dataset
            directory, or `'NONE'`.
        segment_id_to_road_data: Segment ID -> road data, whose `required_attributes` holds the
            iRAP code of each attribute. Unlabeled segments have no entry.
        road_id_to_segment_id_sequence: Road ID -> segment IDs in driving order. Empty if the
            file is absent.

    The files that only some uses need (`unlabeled_segment_id_to_location`,
    `ncontext_segment_ids`) are read on first access, and the values derived from the files
    (`sequence_index`, `segment_coordinates`) are computed on first access. Both are kept, so
    that the splits and evaluation sets built from one metadata object share them.
    """

    metadata_dir: Path
    vocabulary: ClassVocabulary
    splits: T.Mapping[str, T.Sequence[str]]
    segment_id_to_data_paths_rel: T.Mapping[str, T.Mapping[str, str]]
    segment_id_to_road_data: T.Mapping[str, T.Mapping[str, T.Any]]
    road_id_to_segment_id_sequence: T.Mapping[str, T.Sequence[str]]

    # `cached_property` writes to the instance `__dict__`, which a frozen dataclass allows.
    @functools.cached_property
    def sequence_index(self) -> dict[str, tuple[str, int]]:
        """Segment ID -> (road ID, position in the road sequence), see `index_road_sequences`."""
        return index_road_sequences(self.road_id_to_segment_id_sequence)

    @functools.cached_property
    def ncontext_segment_ids(self) -> set[str]:
        """The N-context subset in the `seg_to_res` subdirectory (see
        `load_ncontext_segment_ids`), for the releases with `use_ncontext_filter`."""
        return load_ncontext_segment_ids(self.metadata_dir / MetaFiles.NCONTEXT_SUBDIR,
                                         self.road_id_to_segment_id_sequence)

    @functools.cached_property
    def unlabeled_segment_id_to_location(self) -> dict[str, dict[str, float]]:
        """Segment ID -> `lat` and `lon` in degrees, for the unlabeled segments with a known
        location. Empty if the file is absent."""
        return _load_optional_json(self.metadata_dir / MetaFiles.UNLABELED_SEGMENT_ID_TO_LOCATION)

    @functools.cached_property
    def segment_coordinates(self) -> dict[str, tuple[float, float]]:
        """Segment ID -> (latitude, longitude) in degrees, for each segment that has a known
        location or lies between two such segments of its road sequence.

        A segment has a known location if it has road data or an entry in
        `unlabeled_segment_id_to_location`. A segment between two located segments of its road
        sequence is placed by linear interpolation in the sequence position, as the segments of
        a sequence are evenly spaced (20 m in iRAP-Vietnam). A segment before the first or after
        the last located segment of its sequence is not placed.
        """
        coordinates = {}
        for sid, data in self.segment_id_to_road_data.items():
            location = get_segment_location(data)
            coordinates[sid] = (location.latitude_deg, location.longitude_deg)
        for sid, location in self.unlabeled_segment_id_to_location.items():
            coordinates[sid] = (float(location["lat"]), float(location["lon"]))
        for sequence in self.road_id_to_segment_id_sequence.values():
            located = [i for i, sid in enumerate(sequence) if sid in coordinates]
            for start, end in zip(located, located[1:]):
                (lat0, lon0) = coordinates[sequence[start]]
                (lat1, lon1) = coordinates[sequence[end]]
                for i in range(start + 1, end):
                    t = (i - start) / (end - start)
                    coordinates[sequence[i]] = (lat0 + t * (lat1 - lat0), lon0 + t * (lon1 - lon0))
        return coordinates


def load_irap_metadata(metadata_dir: str | Path) -> IRAPMetadata:
    metadata_dir = Path(metadata_dir)
    return IRAPMetadata(
        metadata_dir=metadata_dir,
        vocabulary=load_class_vocabulary(metadata_dir),
        splits=_load_json(metadata_dir / MetaFiles.SPLITS),
        segment_id_to_data_paths_rel=_load_json(metadata_dir / MetaFiles.SEGMENT_ID_TO_DATA_PATHS),
        segment_id_to_road_data=_load_json(metadata_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA),
        road_id_to_segment_id_sequence=_load_optional_json(
            metadata_dir / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE),
    )


@dc.dataclass(frozen=True)
class SegmentLocation:
    """Where a segment lies on its road section."""

    section: str
    distance_m: float  # from the start of the section
    length_m: float
    latitude_deg: float
    longitude_deg: float


def get_segment_location(road_data: T.Mapping[str, T.Any]) -> SegmentLocation:
    """Reads the location from one entry of `segment_id_to_road_data.json`.

    Two layouts exist: the Vietnam one with top-level `section`, `distance_km`, `length_km`,
    `lat` and `lon`, and the BH one with the coding-table fields `Section`, `Distance` and
    `Length` (in km) and `Latitude` and `Longitude` under `metadata`.

    Raises:
        ValueError: If the entry has neither layout.
    """
    if "distance_km" in road_data:
        return SegmentLocation(
            section=str(road_data["section"]),
            distance_m=1000 * float(road_data["distance_km"]),
            length_m=1000 * float(road_data["length_km"]),
            latitude_deg=float(road_data["lat"]), longitude_deg=float(road_data["lon"]))
    if "metadata" in road_data and "Distance" in road_data["metadata"]:
        fields = road_data["metadata"]
        return SegmentLocation(
            section=str(fields["Section"]),
            distance_m=1000 * float(fields["Distance"]),
            length_m=1000 * float(fields["Length"]),
            latitude_deg=float(fields["Latitude"]), longitude_deg=float(fields["Longitude"]))
    raise ValueError(f"Road data without a known location layout. Keys: {sorted(road_data)}.")


# Labels ###########################################################################################

def compute_label_matrix(segment_id_to_labels, segment_ids, num_attributes) -> np.ndarray:
    """Stacks per-segment attribute labels into an array.

    Args:
        segment_id_to_labels: Mapping from segment ID to per-attribute class indices.
        segment_ids: Sequence of segment IDs to include, in output row order.
        num_attributes: Total number of attributes.

    Returns:
        Array of shape `(len(segment_ids), num_attributes)` with dtype int64.
        Unannotated entries contain `IGNORE_LABEL_INDEX` (-1).
    """
    return np.array([segment_id_to_labels[sid] for sid in segment_ids],
                    dtype=np.int64).reshape(len(segment_ids), num_attributes)


def compute_num_segments_per_class(class_indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Counts the segments of each class of one attribute.

    The ignore label is excluded rather than counted: as a negative index it would
    silently credit every unannotated segment to the last class.

    Args:
        class_indices: (N,) class indices, `IGNORE_LABEL_INDEX` where a segment has no label.
        num_classes: The number of classes of the attribute.

    Returns:
        (num_classes,) int64 array whose element `c` is the number of segments of class `c`.

    Raises:
        ValueError: If a class index is outside `[0, num_classes)` and not the ignore label.
    """
    observed = class_indices[class_indices != IGNORE_LABEL_INDEX]
    if observed.size and (observed.min() < 0 or observed.max() >= num_classes):
        invalid = observed[(observed < 0) | (observed >= num_classes)]
        raise ValueError(f"Class indices outside [0, {num_classes}) and other than the ignore"
                         f" label {IGNORE_LABEL_INDEX}: {sorted(set(invalid.tolist()))}.")
    return np.bincount(observed, minlength=num_classes)


def compute_class_occurrence_counts(
    info,
    segment_ids: T.Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Counts, per attribute, how many of a split's examples are labelled with each class.

    Args:
        info: Dataset info carrying `segment_ids`, `segment_id_to_labels`, `class_counts`
            and `attr_to_value_to_class_idx`.
        segment_ids: The segments to count, keys of `info.segment_id_to_labels`. None counts
            `info.segment_ids`, the examples of the dataset.

    Returns:
        Maps each attribute name, in schema order, to the counts of
        `compute_num_segments_per_class`.
    """
    labels = compute_label_matrix(
        info.segment_id_to_labels, info.segment_ids if segment_ids is None else segment_ids,
        len(info.class_counts))
    counts = {}
    for attr_index, (attr, num_classes) in enumerate(zip(info.attr_to_value_to_class_idx,
                                                          info.class_counts)):
        try:
            counts[attr] = compute_num_segments_per_class(labels[:, attr_index], num_classes)
        except ValueError as e:
            raise ValueError(f"Attribute {attr!r}: {e}") from e
    return counts


def compute_segment_labels(
    vocabulary: ClassVocabulary,
    segment_id_to_road_data: T.Mapping[str, T.Mapping[str, T.Any]],
    segment_ids: T.Iterable[str],
    allow_missing_attributes: bool,
) -> dict[str, list[int]]:
    """Maps the iRAP codes of each segment to class indices in `vocabulary` order.

    A code is missing if the segment has no road data, the attribute has no code, or the code
    is not a class of the attribute (see `to_irap_code`).

    Args:
        allow_missing_attributes: If True, a missing code becomes `IGNORE_LABEL_INDEX`.
            Otherwise a segment with a missing code is left out.

    Returns:
        Segment ID -> one class index per attribute, for the segments that are kept, in the
        order of `segment_ids`.
    """
    code_to_index = {a: vocabulary.get_irap_code_to_class_index(a)
                     for a in vocabulary.attribute_names}

    def get_labels(road_data) -> list[int] | None:
        codes = road_data.get("required_attributes", {})
        labels = [code_to_index[a].get(to_irap_code(codes.get(a)), IGNORE_LABEL_INDEX)
                  for a in vocabulary.attribute_names]
        return labels if allow_missing_attributes or IGNORE_LABEL_INDEX not in labels else None

    segment_id_to_labels = {sid: get_labels(segment_id_to_road_data.get(sid, {}))
                            for sid in segment_ids}
    return {sid: labels for sid, labels in segment_id_to_labels.items() if labels is not None}


def compute_num_labeled_per_attribute(
    vocabulary: ClassVocabulary,
    segment_id_to_labels: T.Mapping[str, T.Sequence[int]],
    segment_ids: T.Sequence[str],
) -> dict[str, int]:
    """The number of labels other than `IGNORE_LABEL_INDEX` per attribute over `segment_ids`."""
    label_matrix = compute_label_matrix(segment_id_to_labels, segment_ids,
                                        len(vocabulary.attribute_names))
    num_labeled = (label_matrix != IGNORE_LABEL_INDEX).sum(axis=0)
    return {a: int(n) for a, n in zip(vocabulary.attribute_names, num_labeled)}


# Context and segment selection ####################################################################

def index_road_sequences(
    road_id_to_segment_id_sequence: T.Mapping[str, T.Sequence[str]],
) -> dict[str, tuple[str, int]]:
    """Segment ID -> (road ID, position in the road sequence).

    A segment can be in several road sequences, e.g. an unlabeled iRAP-Vietnam segment between
    the labeled segments of two roads that cover the same stretch. It is a context frame in
    each of them, and is indexed in the longest one (the first of equally long ones), whose
    context window is the most likely to be complete.
    """
    index: dict[str, tuple[str, int]] = {}
    for road_id, sequence in sorted(road_id_to_segment_id_sequence.items(),
                                    key=lambda item: -len(item[1])):
        for i, sid in enumerate(sequence):
            index.setdefault(sid, (road_id, i))
    return index


def compute_segment_context_ids(
    road_id_to_segment_id_sequence: T.Mapping[str, T.Sequence[str]],
    segment_ids: T.Iterable[str],
    context_offsets: T.Sequence[int],
    available_segment_ids: T.Container[str],
    sequence_index: T.Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """The context segment IDs of each segment whose context window is complete.

    Context frames are drawn from the segment's own road sequence, so a window does not cross
    road boundaries. A window is complete if every offset stays inside the sequence and every
    context segment is in `available_segment_ids` (has an image). A segment that is not in a road
    sequence has no window.

    Args:
        sequence_index: `index_road_sequences(road_id_to_segment_id_sequence)` if already
            computed, e.g. `IRAPMetadata.sequence_index`.

    Returns:
        Segment ID -> the segment IDs at `context_offsets`, for the segments with a complete
        window, in the order of `segment_ids`.
    """
    if sequence_index is None:
        sequence_index = index_road_sequences(road_id_to_segment_id_sequence)

    def get_context_ids(sid: str) -> tuple[str, ...] | None:
        if sid not in sequence_index:
            return None
        road_id, position = sequence_index[sid]
        sequence = road_id_to_segment_id_sequence[road_id]
        indices = [position + offset for offset in context_offsets]
        if min(indices) < 0 or max(indices) >= len(sequence):
            return None
        context_ids = tuple(sequence[i] for i in indices)
        return context_ids if all(c in available_segment_ids for c in context_ids) else None

    segment_id_to_context_ids = {sid: get_context_ids(sid) for sid in segment_ids}
    return {sid: ids for sid, ids in segment_id_to_context_ids.items() if ids is not None}


def load_ncontext_segment_ids(
    seg_to_res_path: T.Union[str, Path],
    road_sequences: T.Union[str, Path, T.Mapping[str, T.Sequence[str]]],
    max_N: int = 10,
    splits: T.Sequence[str] = ("train", "val", "test"),
) -> set[str]:
    """Loads segment IDs from precomputed split results with per-split N-context filtering.

    For each split in `splits`, loads segment IDs from `<seg_to_res_path>/<split>.pickle`
    and filters them so each kept segment has `max_N` preceding and `max_N` following
    neighbors on the same road within that split.

    Args:
        seg_to_res_path: Directory containing `<split>.pickle` files with `segment_id_to_idx`.
        road_sequences: Path to `road_id_to_segment_id_sequence.json` or a mapping of
            road ID to ordered segment IDs.
        max_N: Number of context neighbors required on each side within the split.
        splits: Split names to load and filter.

    Returns:
        Set of segment IDs passing the N-context filter across all specified splits.
    """
    seg_to_res_path = Path(seg_to_res_path)

    if isinstance(road_sequences, T.Mapping):
        road_id_to_segment_id_sequence = road_sequences
    else:
        road_seq_path = Path(road_sequences)
        if not road_seq_path.exists():
            raise FileNotFoundError(
                f"Road sequence file not found: {road_seq_path}\n"
                f"This file is required for N-context filtering.")
        road_id_to_segment_id_sequence = _load_json(road_seq_path)

    def ncontext_filter_for_split(split_segment_ids: set[str]) -> set[str]:
        """Applies N-context filtering for a single split (matches NContextDataset.build_contexts)."""
        filtered = set()
        for segment_sequence in road_id_to_segment_id_sequence.values():
            n_segments = len(segment_sequence)
            for i in range(max_N, n_segments - max_N):
                current = segment_sequence[i]
                if current not in split_segment_ids:
                    continue
                before = segment_sequence[i - max_N : i]
                after = segment_sequence[i + 1 : i + 1 + max_N]
                # All context segments must also be in THIS split's segment set
                if all(seg_id in split_segment_ids for seg_id in list(before) + list(after)):
                    filtered.add(current)
        return filtered

    # Process each split separately (matching original train_local_rec.py behavior)
    total_from_pickles = 0
    filtered_segment_ids = set()

    for split in splits:
        pickle_path = seg_to_res_path / f"{split}.pickle"
        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Precomputed results file not found: {pickle_path}\n"
                f"This file is required for ncontext_segment_id_subset filtering."
            )

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        if "segment_id_to_idx" not in data:
            raise ValueError(
                f"Pickle file {pickle_path} does not contain 'segment_id_to_idx' key. "
                f"Available keys: {list(data.keys())}"
            )

        split_segment_ids = set(data["segment_id_to_idx"].keys())
        total_from_pickles += len(split_segment_ids)

        # Apply N-context filtering for THIS split only
        split_filtered = ncontext_filter_for_split(split_segment_ids)
        filtered_segment_ids.update(split_filtered)

        print(
            f"[load_ncontext_segment_ids] {split}: {len(split_segment_ids)} from pickle, "
            f"{len(split_filtered)} after N-context filter"
        )

    print(
        f"[load_ncontext_segment_ids] Total: {total_from_pickles} from pickles, "
        f"{len(filtered_segment_ids)} after N-context filter (max_N={max_N})"
    )

    return filtered_segment_ids


def is_unlabeled_split(split: str) -> bool:
    """Whether a split holds unlabeled segments (`unlabeled_train`, ...). They have no road data
    and are not in the N-context subsets."""
    return split.startswith("unlabeled")


@dc.dataclass(frozen=True)
class SplitSelection:
    """The segments of a split that a dataset yields, with their labels and context.

    Attributes:
        segment_ids: The selected segments, in `splits.json` order.
        segment_id_to_labels: Labels of the segments of the split that have an image and labels
            (see `compute_split_labels`), a superset of `segment_ids`. The context-window and
            N-context filters do not remove segments from it.
        segment_id_to_context_ids: Context segment IDs of the selected segments.
        attr_to_num_labeled: Number of labeled selected segments per attribute. An attribute can
            have classes in the vocabulary but no label in a dataset (e.g. the BH-only
            attributes of iRAP-Vietnam), which consumers use to leave it out of evaluation.
    """

    segment_ids: tuple[str, ...]
    segment_id_to_labels: T.Mapping[str, list[int]]
    segment_id_to_context_ids: T.Mapping[str, tuple[str, ...]]
    attr_to_num_labeled: T.Mapping[str, int]


def compute_split_labels(
    metadata: IRAPMetadata,
    split: str,
    *,
    allow_missing_attributes: bool,
) -> dict[str, list[int]]:
    """The labels of the segments of `split` that have an image (see `compute_segment_labels`).

    Splits whose name starts with `unlabeled` have no road data, so their segments are kept
    with all labels `IGNORE_LABEL_INDEX`, whatever `allow_missing_attributes` is.

    Returns:
        Segment ID -> class indices, in `splits.json` order.

    Raises:
        ValueError: If `split` is not in `splits.json`.
    """
    if split not in metadata.splits:
        raise ValueError(f"Invalid split {split!r}. Available in splits.json:"
                         f" {', '.join(sorted(metadata.splits))}.")
    paths = metadata.segment_id_to_data_paths_rel
    return compute_segment_labels(
        metadata.vocabulary, metadata.segment_id_to_road_data,
        [sid for sid in metadata.splits[split] if sid in paths],
        allow_missing_attributes or is_unlabeled_split(split))


def select_split_segments(
    metadata: IRAPMetadata,
    split: str,
    *,
    context_offsets: T.Sequence[int],
    allow_missing_attributes: bool,
    ncontext_segment_id_subset: T.Container[str] | None = None,
) -> SplitSelection:
    """Selects the segments of `split` that have an image, labels and a complete context window.

    The filters apply in this order: an image path exists and the labels are complete (see
    `compute_split_labels`), the context window is complete (see
    `compute_segment_context_ids`), and the segment is in `ncontext_segment_id_subset`.

    Raises:
        ValueError: If `split` is not in `splits.json`.
    """
    segment_id_to_labels = compute_split_labels(
        metadata, split, allow_missing_attributes=allow_missing_attributes)
    num_without_sequence = sum(sid not in metadata.sequence_index for sid in segment_id_to_labels)
    if num_without_sequence:
        warnings.warn(f"{num_without_sequence} of the {len(segment_id_to_labels)} segments of"
                      f" split {split!r} that have an image and labels are in no road sequence,"
                      f" so they have no context window and are left out.")
    segment_id_to_context_ids = compute_segment_context_ids(
        metadata.road_id_to_segment_id_sequence, segment_id_to_labels, context_offsets,
        available_segment_ids=metadata.segment_id_to_data_paths_rel,
        sequence_index=metadata.sequence_index)
    if ncontext_segment_id_subset is not None:
        segment_id_to_context_ids = {sid: ids for sid, ids in segment_id_to_context_ids.items()
                                     if sid in ncontext_segment_id_subset}
    segment_ids = tuple(segment_id_to_context_ids)
    return SplitSelection(
        segment_ids=segment_ids,
        segment_id_to_labels=segment_id_to_labels,
        segment_id_to_context_ids=segment_id_to_context_ids,
        attr_to_num_labeled=compute_num_labeled_per_attribute(
            metadata.vocabulary, segment_id_to_labels, segment_ids))


def select_preset_split_segments(
    metadata: IRAPMetadata,
    dataset_name: str,
    split: str,
    context_offsets: T.Sequence[int] | None = None,
) -> SplitSelection:
    """Selects the segments of `split` with the loading defaults of a release.

    Args:
        dataset_name: A key of `DATASET_PRESETS`.
        context_offsets: The context offsets of a model, or None for the reference offsets of
            the release (see `DatasetPreset`).
    """
    preset = get_dataset_preset(dataset_name)
    use_ncontext_filter = preset.use_ncontext_filter and not is_unlabeled_split(split)
    return select_split_segments(
        metadata, split,
        context_offsets=(preset.reference_context_offsets if context_offsets is None
                         else context_offsets),
        allow_missing_attributes=preset.allow_missing_attributes,
        ncontext_segment_id_subset=(metadata.ncontext_segment_ids if use_ncontext_filter
                                    else None))
