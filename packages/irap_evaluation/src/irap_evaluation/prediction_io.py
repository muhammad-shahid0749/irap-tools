"""Reading and writing `Predictions` as Parquet files.

`docs/prediction_format.md` specifies the file layout.
"""

import json
import typing as T
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from irap_data.metadata import IGNORE_LABEL_INDEX

from .file_names import shorten_file_name, to_valid_file_name
from .predictions import (
    ModelInfo,
    OutputKind,
    PredictionFormatError,
    PredictionHeader,
    Predictions,
    format_model_label,
    to_class_indices,
)

FORMAT_VERSION = 1
HEADER_METADATA_KEY = b"irap_predictions"
SEGMENT_ID_COLUMN = "segment_id"
_REQUIRED_HEADER_FIELDS = ("format_version", "dataset", "split", "model",
                           "attribute_to_irap_codes")


# Header ###########################################################################################

def _to_header_json_dict(predictions: Predictions) -> dict[str, T.Any]:
    header = predictions.header
    offsets = header.context_offsets
    return {
        "format_version": FORMAT_VERSION,
        "dataset": header.dataset,
        "split": header.split,
        "context_offsets": None if offsets is None else list(offsets),
        "model": header.model.to_json_dict(),
        "attribute_to_irap_codes": {attr: list(codes)
                                    for attr, codes in predictions.attribute_to_irap_codes.items()},
    }


def _is_int(value: T.Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_string_list(value: T.Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _from_model_json_dict(d: T.Any) -> ModelInfo:
    """Parses the object of `ModelInfo.to_json_dict`, ignoring unknown fields.

    Raises:
        PredictionFormatError: If it is malformed.
    """
    if not (isinstance(d, dict) and isinstance(d.get("method_name"), str)
            and isinstance(d.get("method_display_name"), (str, type(None)))
            and (d.get("seed") is None or _is_int(d["seed"]))
            and "training_splits" in d
            and (d["training_splits"] is None or _is_string_list(d["training_splits"]))
            and _is_string_list(d.get("early_stopping_splits"))
            and isinstance(d.get("details"), (dict, type(None)))):
        raise PredictionFormatError(
            "'model' must be an object with a string 'method_name', 'training_splits' (an array"
            " of split names, or null if unknown), 'early_stopping_splits' (an array of split"
            " names), an optional string 'method_display_name', an optional integer 'seed' and"
            " an optional object 'details'.")
    return ModelInfo(method_name=d["method_name"], training_splits=d["training_splits"],
                     early_stopping_splits=d["early_stopping_splits"], seed=d.get("seed"),
                     method_display_name=d.get("method_display_name"), details=d.get("details"))


def _from_header_json_dict(d: T.Any) -> tuple[PredictionHeader, dict[str, list]]:
    """Parses the header written by `_to_header_json_dict`, ignoring unknown fields.

    Returns:
        The header and `attribute_to_irap_codes`, whose codes `Predictions` validates.

    Raises:
        PredictionFormatError: If a required field is missing or has the wrong JSON type, or for
            an unsupported format version.
    """
    if not isinstance(d, dict):
        raise PredictionFormatError("The header must be a JSON object.")
    version = d.get("format_version")
    if not _is_int(version):
        raise PredictionFormatError(f"The format version must be an integer, got {version!r}.")
    if version != FORMAT_VERSION:
        raise PredictionFormatError(f"Format version {version} is not supported. The supported"
                                    f" version is {FORMAT_VERSION}.")
    if missing := [f for f in _REQUIRED_HEADER_FIELDS if f not in d]:
        raise PredictionFormatError(f"The header has no fields {missing}.")
    offsets, attribute_to_irap_codes = d.get("context_offsets"), d["attribute_to_irap_codes"]
    if not (isinstance(d["dataset"], str) and isinstance(d["split"], str)):
        raise PredictionFormatError("'dataset' and 'split' must be strings.")
    model = _from_model_json_dict(d["model"])
    if not (offsets is None or (isinstance(offsets, list) and all(map(_is_int, offsets)))):
        raise PredictionFormatError("'context_offsets' must be null or an array of integers.")
    if not (isinstance(attribute_to_irap_codes, dict)
            and all(isinstance(codes, list) for codes in attribute_to_irap_codes.values())):
        raise PredictionFormatError("'attribute_to_irap_codes' must map attributes to arrays of"
                                    " iRAP codes.")
    header = PredictionHeader(dataset=d["dataset"], split=d["split"], model=model,
                              context_offsets=offsets)
    return header, attribute_to_irap_codes


def _parse_header_metadata(
        metadata: T.Mapping[bytes, bytes]) -> tuple[PredictionHeader, dict[str, list]]:
    """Parses the header in the key-value metadata of a table or a Parquet file (see
    `_from_header_json_dict`).

    Raises:
        PredictionFormatError: If there is no header, or it does not match the format.
    """
    if HEADER_METADATA_KEY not in metadata:
        raise PredictionFormatError(f"The table has no {HEADER_METADATA_KEY.decode()!r} header.")
    try:
        header_json = json.loads(metadata[HEADER_METADATA_KEY].decode("utf-8"))
    except ValueError as e:
        raise PredictionFormatError(f"The header is not UTF-8 JSON: {e}") from e
    return _from_header_json_dict(header_json)


# Columns ##########################################################################################

def _to_probability_column(probs: np.ndarray, is_valid: np.ndarray) -> pa.ListArray:
    # A variable-size list, since Parquet readers do not restore null fixed-size lists.
    offsets = np.concatenate([[0], np.cumsum(np.where(is_valid, probs.shape[1], 0))])
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()),
                                    pa.array(probs[is_valid].reshape(-1)),
                                    mask=pa.array(~is_valid))


def _get_output_kind(attr: str, data_type: pa.DataType) -> OutputKind | None:
    """The output kind that a column type stores, None for the null type, which dataframe
    libraries give to a column with only null values."""
    if pa.types.is_null(data_type):
        return None
    if pa.types.is_integer(data_type):
        return "hard"
    if ((pa.types.is_list(data_type) or pa.types.is_large_list(data_type))
            and (pa.types.is_float32(data_type.value_type)
                 or pa.types.is_float64(data_type.value_type))):
        return "probs"
    raise PredictionFormatError(f"{attr!r}: expected a list of float32 or float64 probabilities"
                                f" or integer class indices, got {data_type}.")


def _read_probabilities(attr: str, column: pa.Array, num_classes: int,
                        is_valid: np.ndarray) -> np.ndarray:
    """(N, K) float32 distributions of a list column, NaN in the rows of null values."""
    column = column.cast(pa.list_(pa.float32()))
    lengths = column.value_lengths().to_numpy(zero_copy_only=False)
    if not (lengths[is_valid] == num_classes).all():
        raise PredictionFormatError(f"{attr!r}: every non-null list must have"
                                    f" {num_classes} probabilities.")
    probs = np.full((len(column), num_classes), np.nan, dtype=np.float32)
    probs[is_valid] = column.flatten().to_numpy(zero_copy_only=False).reshape(-1, num_classes)
    return probs


def _read_class_indices(attr: str, column: pa.Array, is_valid: np.ndarray) -> np.ndarray:
    """(N,) int64 class indices of an integer column, `IGNORE_LABEL_INDEX` for null values."""
    indices = column.cast(pa.int64()).fill_null(IGNORE_LABEL_INDEX).to_numpy(zero_copy_only=False)
    if (indices[is_valid] == IGNORE_LABEL_INDEX).any():
        raise PredictionFormatError(f"{attr!r}: a non-null value is {IGNORE_LABEL_INDEX}. An"
                                    f" invalid attribute prediction is null.")
    return indices


# Tables and files #################################################################################

def to_arrow_table(predictions: Predictions) -> pa.Table:
    """The table of `predictions`: list columns of probabilities, or integer columns of class
    indices for hard predictions, null for invalid attribute predictions."""
    attributes, is_valid = predictions.attributes, predictions.is_valid
    if SEGMENT_ID_COLUMN in attributes:
        raise PredictionFormatError(f"An attribute cannot be named {SEGMENT_ID_COLUMN!r}.")
    if predictions.output_kind == "hard":
        class_indices = to_class_indices(predictions)
        columns = {attr: pa.array(class_indices[attr], type=pa.int64(), mask=~is_valid[attr])
                   for attr in attributes}
    else:
        columns = {attr: _to_probability_column(predictions.probs[attr], is_valid[attr])
                   for attr in attributes}
    table = pa.table({SEGMENT_ID_COLUMN: pa.array(predictions.segment_ids, type=pa.string()),
                      **columns})
    try:
        header = json.dumps(_to_header_json_dict(predictions), ensure_ascii=False,
                            allow_nan=False)
    except (TypeError, ValueError) as e:
        raise PredictionFormatError(f"The header is not standard JSON, e.g. 'model.details' holds"
                                    f" NaN or a NumPy value: {e}") from e
    return table.replace_schema_metadata({HEADER_METADATA_KEY: header.encode("utf-8")})


def from_arrow_table(table: pa.Table,
                     metadata: T.Mapping[bytes, bytes] | None = None) -> Predictions:
    """Parses a table written by `to_arrow_table`.

    The column types give the output kind. The other column types that
    `docs/prediction_format.md` allows, e.g. those of Polars, are converted.

    Args:
        metadata: The key-value metadata that holds the header, by default the schema metadata of
            `table`.

    Raises:
        PredictionFormatError: If the header or a column does not match the format.
    """
    metadata = (table.schema.metadata if metadata is None else metadata) or {}
    header, attribute_to_irap_codes = _parse_header_metadata(metadata)
    expected_columns = [SEGMENT_ID_COLUMN, *attribute_to_irap_codes]
    if missing := [c for c in expected_columns if c not in table.column_names]:
        raise PredictionFormatError(f"The table has no columns {missing}.")
    if extra := [c for c in table.column_names if c not in expected_columns]:
        raise PredictionFormatError(f"The columns {extra} are not attributes of the header.")
    segment_ids = table.column(SEGMENT_ID_COLUMN).to_pylist()
    columns = {attr: table.column(attr).combine_chunks() for attr in attribute_to_irap_codes}
    output_kinds = {_get_output_kind(attr, column.type) for attr, column in columns.items()}
    output_kinds.discard(None)
    if not output_kinds:
        raise PredictionFormatError("Every attribute column has the null type, so the output kind"
                                    " is unknown.")
    if len(output_kinds) > 1:
        raise PredictionFormatError("Either all attribute columns hold probabilities or all hold"
                                    " class indices.")

    is_valid = {attr: ~column.is_null().to_numpy(zero_copy_only=False)
                for attr, column in columns.items()}
    if output_kinds == {"hard"}:
        class_indices = {attr: _read_class_indices(attr, column, is_valid[attr])
                         for attr, column in columns.items()}
        return Predictions.from_class_indices(header, attribute_to_irap_codes, segment_ids,
                                              class_indices)
    probs = {attr: _read_probabilities(attr, column, len(attribute_to_irap_codes[attr]),
                                       is_valid[attr])
             for attr, column in columns.items()}
    return Predictions.from_probabilities(header, attribute_to_irap_codes, segment_ids, probs,
                                          is_valid)


def make_prediction_file_name(method_name: str, seed: int | None, split: str) -> str:
    """The conventional name of a prediction file: `<method>.<split>.predictions.parquet`, or
    `<method>_seed<seed>.<split>.predictions.parquet` for a model with a seed.

    The method name and the split are made valid by `to_valid_file_name`. A method name that
    would make the name longer than `MAX_FILE_NAME_BYTES` is cut by `shorten_file_name`. The
    full method name is in the file's header.

    Raises:
        ValueError: If the split name leaves no room for the method name.
    """
    # The label is `<method>/seed<seed>` for a model with a seed, as for report directories.
    stem = to_valid_file_name(format_model_label(method_name, seed))
    seed_part = "" if seed is None else f"_seed{seed}"
    return shorten_file_name(stem.removesuffix(seed_part),
                             f"{seed_part}.{to_valid_file_name(split)}.predictions.parquet")


def write_predictions(path: str | Path, predictions: Predictions) -> None:
    """Writes `predictions` to a Parquet file, conventionally named by
    `make_prediction_file_name`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_arrow_table(predictions), path,
                   use_dictionary=predictions.output_kind == "hard")


def read_predictions(path: str | Path) -> Predictions:
    """Reads and validates a prediction file, e.g. one written by `write_predictions`.

    Raises:
        PredictionFormatError: If the file does not match the format.
    """
    # Reads the header from the file metadata, since pyarrow's schema metadata lacks it for files
    # of some writers, e.g. Polars.
    with pq.ParquetFile(path) as file:
        return from_arrow_table(file.read(), file.metadata.metadata)


def read_prediction_header(source: str | Path | T.BinaryIO) -> PredictionHeader:
    """Reads the header of a prediction file without its columns, e.g. to show the model of a
    file before it is read. The attributes and the columns are not checked, so `read_predictions`
    may still refuse the file.

    Args:
        source: A path or a seekable binary file, e.g. a member of a .zip archive.

    Raises:
        PredictionFormatError: If the header does not match the format.
    """
    with pq.ParquetFile(source) as file:
        return _parse_header_metadata(file.metadata.metadata or {})[0]
