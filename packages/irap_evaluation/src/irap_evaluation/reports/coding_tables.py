"""Export of predictions as iRAP coding tables."""

import dataclasses as dc
import functools
import json
import math
import typing as T
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from irap_data.metadata import IRAPMetadata, SegmentLocation, get_segment_location

from ..predictions import INVALID_IRAP_CODE, PredictionFormatError, Predictions, to_irap_codes

#: The fields that `predictions_to_coding_table` fills, besides the attributes.
CODING_TABLE_FIELDS = ("coder_name", "coding_date", "segment_id", "section", "distance_km",
                       "length_km", "latitude_start", "longitude_start", "latitude_end",
                       "longitude_end")


@dc.dataclass(frozen=True)
class CodingTableTemplate:
    """A coding-table layout.

    Attributes:
        columns: The column headers, in order.
        field_to_column: Each of `CODING_TABLE_FIELDS` -> its column.
        attribute_to_column: Attribute -> the column that receives its predicted iRAP code.

    Raises:
        ValueError: If a field is not mapped, or a field or attribute is mapped to a column that
            is not in `columns` or to the same column as another.
    """

    columns: tuple[str, ...]
    field_to_column: T.Mapping[str, str]
    attribute_to_column: T.Mapping[str, str]

    def __post_init__(self):
        if missing := [f for f in CODING_TABLE_FIELDS if f not in self.field_to_column]:
            raise ValueError(f"Fields without a column: {missing}.")
        mapped = [*self.field_to_column.values(), *self.attribute_to_column.values()]
        if unknown := sorted(set(mapped) - set(self.columns)):
            raise ValueError(f"Mapped columns not in the template: {unknown}.")
        if len(set(mapped)) != len(mapped):
            raise ValueError("Several fields or attributes are mapped to the same column.")


def load_coding_table_template(path: str | Path | None = None) -> CodingTableTemplate:
    """Loads a template JSON file. None loads the packaged iRAP layout."""
    text = (Path(path).read_text(encoding="utf-8") if path is not None else
            resources.files("irap_evaluation.reports").joinpath("data/coding_table_template.json")
            .read_text(encoding="utf-8"))
    d = json.loads(text)
    return CodingTableTemplate(tuple(d["columns"]), d["field_to_column"], d["attribute_to_column"])


def get_unmapped_attributes(template: CodingTableTemplate,
                            attributes: T.Iterable[str]) -> list[str]:
    """The attributes whose predictions have no column in the template."""
    return [a for a in attributes if a not in template.attribute_to_column]


def get_unfilled_columns(template: CodingTableTemplate, attributes: T.Iterable[str]) -> list[str]:
    """The template columns that stay blank for predictions of `attributes`."""
    filled = {*template.field_to_column.values(),
              *_get_attribute_columns(template, attributes).values()}
    return [c for c in template.columns if c not in filled]


@dc.dataclass(frozen=True)
class _RoadOrder:
    """The predicted segments in road order (`_locate_in_road_order`), and where the prediction
    of each is: row `rows[j]` of `predictions_seq[predictions_indices[j]]`."""

    segment_ids: list[str]
    predictions_indices: np.ndarray
    rows: np.ndarray


def _locate_in_road_order(predictions_seq: T.Sequence[Predictions],
                          metadata: IRAPMetadata) -> _RoadOrder:
    """Locates the predicted segments of all predictions, ordered by road sequence and position,
    then the segments of no sequence in the order of `predictions_seq` and their rows.

    Raises:
        ValueError: If `predictions_seq` is empty.
        PredictionFormatError: If the predictions are of different datasets or models, or a
            segment is in several of them.
    """
    if not predictions_seq:
        raise ValueError("At least one set of predictions is required.")
    first = predictions_seq[0].header
    if any((p.header.dataset, p.header.model) != (first.dataset, first.model)
           for p in predictions_seq[1:]):
        raise PredictionFormatError("A coding table is made of the predictions of one model on"
                                    " one dataset.")
    # (segment ID, index in `predictions_seq`, row)
    located = [(sid, i, row) for i, p in enumerate(predictions_seq)
               for row, sid in enumerate(p.segment_ids)]
    if len({sid for sid, _, _ in located}) != len(located):
        raise PredictionFormatError("A segment is in several of the predictions, e.g. of"
                                    " overlapping splits.")
    road_rank = {road_id: i for i, road_id in enumerate(metadata.road_id_to_segment_id_sequence)}
    sequence_index = metadata.sequence_index

    def get_road_position(segment_id: str) -> tuple[float, int]:
        if segment_id not in sequence_index:
            return (math.inf, 0)  # The sort is stable.
        road_id, position = sequence_index[segment_id]
        return (road_rank[road_id], position)

    located.sort(key=lambda item: get_road_position(item[0]))
    return _RoadOrder([sid for sid, _, _ in located],
                      np.array([i for _, i, _ in located], dtype=np.int64),
                      np.array([row for _, _, row in located], dtype=np.int64))


def _get_attributes(predictions_seq: T.Sequence[Predictions]) -> list[str]:
    """The attributes that any of the predictions predict, in order of appearance."""
    return list(dict.fromkeys(a for p in predictions_seq for a in p.attributes))


def _get_next_segment_ids_with_road_data(metadata: IRAPMetadata) -> dict[str, str]:
    """Segment ID -> the next segment with road data in its road sequence, if there is one.

    Segments without road data can lie between the two.
    """
    road_data = metadata.segment_id_to_road_data
    segment_id_to_next = {}
    for sequence in metadata.road_id_to_segment_id_sequence.values():
        next_id = None
        for sid in reversed(sequence):
            if next_id is not None:
                segment_id_to_next[sid] = next_id
            if sid in road_data:
                next_id = sid
    return segment_id_to_next


#: Tolerance of the distance at which the next segment of a section starts.
ADJACENCY_EPS_M = 1.0


def _is_adjacent(location: SegmentLocation, next_location: SegmentLocation) -> bool:
    """Whether `next_location` starts on the same section where `location` ends."""
    return (next_location.section == location.section and
            abs(next_location.distance_m - location.distance_m - location.length_m)
            <= ADJACENCY_EPS_M)


def _make_blank_table(template: CodingTableTemplate, num_rows: int) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series([None] * num_rows, dtype=object)
                         for column in template.columns})


def _get_attribute_columns(template: CodingTableTemplate,
                           attributes: T.Iterable[str]) -> dict[str, str]:
    """Attribute -> column, for the attributes that have a column."""
    return {a: template.attribute_to_column[a] for a in attributes
            if a in template.attribute_to_column}


def predictions_to_coding_table(
    predictions_seq: T.Sequence[Predictions],
    metadata: IRAPMetadata,
    template: CodingTableTemplate,
    *,
    coder_name: str,
    coding_date: str,
) -> pd.DataFrame:
    """Makes one coding-table row per predicted segment, with the argmax iRAP code per attribute.

    Rows are ordered by road sequence and position. The end coordinates are the start of the
    next segment with road data if it starts where the segment ends, else blank. Invalid
    predictions, the attributes that the predictions of a segment lack, and the columns that no
    field or predicted attribute fills are blank.

    Args:
        predictions_seq: The predictions of one model on disjoint segments, e.g. on several
            splits.
        metadata: The metadata of the release, for the segment locations.
        coder_name: Written to the coder-name column, e.g. the method name.
        coding_date: Written to the coding-date column, `YYYY-MM-DD`.

    Raises:
        ValueError: See `_locate_in_road_order`.
    """
    order = _locate_in_road_order(predictions_seq, metadata)
    road_data = metadata.segment_id_to_road_data
    segment_id_to_next = _get_next_segment_ids_with_road_data(metadata)
    get_location = functools.cache(lambda sid: get_segment_location(road_data[sid]))

    def get_location_fields(segment_id: str) -> dict[str, T.Any]:
        if segment_id not in road_data:
            return {}
        location = get_location(segment_id)
        fields = dict(section=location.section, distance_km=location.distance_m / 1000,
                      length_km=location.length_m / 1000, latitude_start=location.latitude_deg,
                      longitude_start=location.longitude_deg)
        next_id = segment_id_to_next.get(segment_id)
        if next_id is not None and _is_adjacent(location, end := get_location(next_id)):
            fields.update(latitude_end=end.latitude_deg, longitude_end=end.longitude_deg)
        return fields

    field_rows = [dict(coder_name=coder_name, coding_date=coding_date, segment_id=sid,
                       **get_location_fields(sid))
                  for sid in order.segment_ids]
    table = _make_blank_table(template, len(order.segment_ids))
    for field, column in template.field_to_column.items():
        table[column] = [row.get(field) for row in field_rows]
    codes_seq = [to_irap_codes(p) for p in predictions_seq]
    for attr, column in _get_attribute_columns(template, _get_attributes(predictions_seq)).items():
        codes = _gather_attribute_values(order, codes_seq, attr, INVALID_IRAP_CODE)
        table[column] = pd.arrays.IntegerArray(codes, codes == INVALID_IRAP_CODE)
    return table


def _gather_attribute_values(order: _RoadOrder,
                             values_seq: T.Sequence[T.Mapping[str, np.ndarray]], attr: str,
                             missing_value: T.Any) -> np.ndarray:
    """Gathers the value of `attr` of each segment in road order from the (N,) values of its
    predictions (`values_seq[predictions_index]`), `missing_value` where they lack the
    attribute."""
    dtype = next(v[attr].dtype for v in values_seq if attr in v)
    gathered = np.full(len(order.segment_ids), missing_value, dtype=dtype)
    for i, values in enumerate(values_seq):
        if attr in values:
            is_from_values = order.predictions_indices == i
            gathered[is_from_values] = values[attr][order.rows[is_from_values]]
    return gathered


def predictions_to_confidence_table(predictions_seq: T.Sequence[Predictions],
                                    metadata: IRAPMetadata,
                                    template: CodingTableTemplate) -> pd.DataFrame:
    """Makes a table of the probability of each exported code, in the layout and row order of
    `predictions_to_coding_table`.

    Only the segment ID column and the attribute columns are filled.

    Raises:
        ValueError: For hard predictions, which have no probabilities, and see
            `_locate_in_road_order`.
    """
    if any(p.output_kind == "hard" for p in predictions_seq):
        raise ValueError("Hard predictions have no probabilities for a confidence table.")
    order = _locate_in_road_order(predictions_seq, metadata)
    table = _make_blank_table(template, len(order.segment_ids))
    table[template.field_to_column["segment_id"]] = order.segment_ids
    attribute_to_column = _get_attribute_columns(template, _get_attributes(predictions_seq))
    confidences_seq = [{attr: probs.max(1) for attr, probs in p.probs.items()
                        if attr in attribute_to_column}
                       for p in predictions_seq]
    for attr, column in attribute_to_column.items():
        table[column] = _gather_attribute_values(order, confidences_seq, attr, np.nan)
    return table


def write_coding_table(path: str | Path, table: pd.DataFrame,
                       confidence_table: pd.DataFrame | None = None) -> list[Path]:
    """Writes a coding table as `.xlsx` or `.csv`, depending on the suffix of `path`.

    An `.xlsx` file gets the confidence table as a second sheet. With `.csv`, it goes to a
    separate `<stem>.confidence.csv` file. A `.csv` file is UTF-8 with a byte order mark, without
    which Excel reads it in the ANSI code page and garbles e.g. Vietnamese section names.

    Returns:
        The written files.

    Raises:
        ValueError: For another suffix.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".xlsx":
        _write_xlsx(path, {"Coding table": table} if confidence_table is None
                    else {"Coding table": table, "Confidence": confidence_table})
        return [path]
    if path.suffix == ".csv":
        table.to_csv(path, index=False, encoding="utf-8-sig")
        if confidence_table is None:
            return [path]
        confidence_path = path.with_suffix(".confidence.csv")
        confidence_table.to_csv(confidence_path, index=False, encoding="utf-8-sig")
        return [path, confidence_path]
    raise ValueError(f"Expected a .xlsx or .csv path, got {path}.")


def _write_xlsx(path: Path, sheet_name_to_table: T.Mapping[str, pd.DataFrame]) -> None:
    """Writes each table as a sheet, with the column names as the first row and missing values
    as blank cells.

    The rows are written with xlsxwriter, since `DataFrame.to_excel` is about 3 times slower
    with either engine (2026-10-04). Strings are written as text, not as formulas or URLs.
    """
    import xlsxwriter  # The `xlsx` extra.

    # `constant_memory` writes each row to disk when the next one starts.
    with xlsxwriter.Workbook(path, {"constant_memory": True}) as workbook:
        for sheet_name, table in sheet_name_to_table.items():
            sheet = workbook.add_worksheet(sheet_name)
            sheet.add_write_handler(
                str, lambda sheet, row, col, text, *args: sheet.write_string(row, col, text, *args))
            sheet.write_row(0, 0, table.columns)
            rows = table.astype(object).where(table.notna(), None).itertuples(index=False,
                                                                              name=None)
            for i, row in enumerate(rows, start=1):
                sheet.write_row(i, 0, row)
