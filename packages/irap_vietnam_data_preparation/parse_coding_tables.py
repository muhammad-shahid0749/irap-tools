"""Parse iRAP-Vietnam coding tables (per-segment iRAP attribute annotations) into a normalized per-segment table.

Usage:
    python parse_coding_tables.py <data_dir>

Reads (under ``<data_dir>/_raw/``):
    coding-tables.zip                # re-extracted into _work/coding-tables/
    attribute_metadata.json         # BH-compatible: attribute_to_idx +
                                     # attribute_value_to_irap_number

Writes (into ``<data_dir>/_work/``):
    rows.parquet                     # one row per kept segment
    parse_report.json                # counts, warnings, dropped-row breakdown

Validation rules (see irap_vietnam_data_preparation/vietnam_data_preparation.md):

- Header lookup is by name (case-insensitive after whitespace strip); column
  order is not assumed. Table columns are expected to match the BH attribute
  metadata names directly.
- Required scalar columns: Section, Distance, Length, Latitude start,
  Longitude start, Image Reference FPZ, Comments.
- Required attribute columns: every key of attribute_to_idx in the supplied
  metadata, minus the entries listed in ``incompatible_attributes.json``
  (``ignored_from_bh``). Missing any remaining required column => fatal
  error. Attributes whose Vietnam table header differs from the BH name are
  resolved via the ``bh_to_table_attribute_name`` map in
  ``incompatible_attributes.json`` (the output keeps the canonical BH name).
- Files lacking the labeling columns entirely (Section + at least one
  attribute) are skipped as "non-coding files".
- Rows are dropped (and counted) when:
    * a scalar cell (Distance, Length, coordinates, ``offset_distance_m``) is
      not numeric,
    * the optional ``offset_distance_m`` column (distance in meters between the
      row's FPZ image coordinates and its annotation coordinates) is greater
      than MAX_ANNOT_LOC_OFFSET_M (a blank offset is not checked),
    * Length != 0.02 (within 1e-6),
    * any required scalar field is missing,
    * Image Reference FPZ does not contain "seg. no. <N>",
    * the row has no genuine attribute value at all, as a non-coding or
      padding row.
- An attribute cell that is blank, not an integer, or without an iRAP code in
  attribute_value_to_irap_number becomes MISSING_ATTR_CODE (-1). With
  --drop-rows-missing-attributes, the row is dropped instead. Speed limit (in
  km/h) and Intersecting road volume (a vehicle count) are converted to iRAP
  codes (see ATTR_VALUE_CLASSIFIERS).
- Duplicate segment ids across the entire dataset are resolved by keeping
  the row with the most non-empty attribute cells, and the first one in a tie.

Every value that is set to missing or interpreted, and every file-level
irregularity, is recorded as an :class:`Issue` (see :class:`IssueKind`). The
issues are printed as warnings to stderr, summarized on stdout, and listed per
file in parse_report.json. Row numbers are Excel row numbers (the header is
row 1).
"""

import argparse
import difflib
import json
import math
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import typing as T

import pandas as pd
from tqdm import tqdm

import layout


SEG_ID_COLUMN = "Image Reference FPZ"
REQUIRED_SCALAR_COLUMNS = (
    "Section",
    "Distance",
    "Length",
    "Latitude start",
    "Longitude start",
    SEG_ID_COLUMN,
    "Comments",
)

EXPECTED_LENGTH_KM = 0.02
LENGTH_TOLERANCE = 1e-6
INCOMPATIBLE_ATTRIBUTES_FILE = "incompatible_attributes.json"
# Optional column: distance (m) between a row's FPZ image coordinates and its
# annotation coordinates. Rows whose annotation is farther than this from the
# FPZ point are dropped. The column is ignored when absent or blank.
ANNOT_LOC_OFFSET_COLUMN = "offset_distance_m"
MAX_ANNOT_LOC_OFFSET_M = 5
# Stored in an attribute column for a cell that is blank, not an integer, or
# without an iRAP code (see SET_TO_MISSING_KINDS).
MISSING_ATTR_CODE = -1

SEG_ID_RE = re.compile(r"seg\.?\s*no\.?\s*(\d+)", re.IGNORECASE)


def resolve_coding_tables_dir(data_dir: Path) -> Path:
    """Return a directory containing the coding tables.

    Always removes any existing ``_work/coding-tables/`` directory and
    re-extracts ``_raw/coding-tables.zip`` into a fresh one, then returns it.
    """
    unzipped = layout.coding_tables_dir(data_dir)
    zip_path = layout.coding_tables_zip_path(data_dir)
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"'{zip_path}' not found."
        )
    if unzipped.is_dir():
        print(f"Removing existing {unzipped}...")
        shutil.rmtree(unzipped)
    print(f"Unzipping {zip_path} -> {unzipped}...")
    unzipped.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(unzipped)
    return unzipped


def load_attribute_metadata(path: Path) -> tuple[list[str], dict[str, set[int]]]:
    """Load ``attribute_metadata.json``.

    Returns:
        attribute_names: ordered list (by ``attribute_to_idx`` value).
        valid_codes_per_attr: ``{attr_name: set(irap_codes)}``.
    """
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "attribute_to_idx" not in meta or "attribute_value_to_irap_number" not in meta:
        raise ValueError(
            f"{path} missing 'attribute_to_idx' or "
            f"'attribute_value_to_irap_number'."
        )
    attribute_names = ordered_attribute_names(meta)
    valid_codes = {
        attr: {int(c) for c in meta["attribute_value_to_irap_number"][attr].values()}
        for attr in attribute_names
    }
    return attribute_names, valid_codes


def ordered_attribute_names(meta: T.Mapping[str, T.Any]) -> list[str]:
    """Attribute names of an attribute-metadata dict, ordered by ``attribute_to_idx``."""
    idx_to_attr = {int(v): k for k, v in meta["attribute_to_idx"].items()}
    return [idx_to_attr[i] for i in sorted(idx_to_attr)]


def load_incompatible_attributes(script_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Load the mapping file in the script dir.

    Returns ``(ignored_from_bh, bh_to_table_attribute_name)``: the BH attribute
    names to exclude, and ``{bh_attribute_name: vietnam_table_column_name}`` used
    to resolve required columns whose Vietnam header differs from the canonical
    BH name (the output keeps the BH name). Both are empty if the file or key
    is absent.
    """
    path = script_dir / INCOMPATIBLE_ATTRIBUTES_FILE
    if not path.is_file():
        return [], {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (list(data.get("ignored_from_bh", [])),
            dict(data.get("bh_to_table_attribute_name", {})))


def find_table_files(coding_tables_dir: Path) -> list[Path]:
    """List candidate spreadsheet files (``.xlsx``/``.xls``) under the dir."""
    files = sorted(
        p for p in coding_tables_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".xlsx", ".xls")
        and not p.name.startswith("~$")  # Excel lock files
    )
    return files


def is_blank(value: T.Any) -> bool:
    """True if a cell is empty/NaN/blank-string."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def is_real_attribute_code(code: T.Any) -> bool:
    """True if an attribute ``code`` is a real annotation (not absent/blank)."""
    return code not in (None, MISSING_ATTR_CODE)


def normalize_header(name: T.Any) -> str:
    """Return a trimmed string header name (or empty string)."""
    return "" if is_blank(name) else str(name).strip()


def format_row_ranges(rows: T.Iterable[int]) -> str:
    """Collapse a set of 1-based row numbers into a list of continuous ranges.

    E.g. ``[92, 93, 468, 469, 470]`` -> ``"92-93, 468-470"``.
    """
    nums = sorted(set(rows))
    if not nums:
        return ""
    ranges: list[tuple[int, int]] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append((start, prev))
            start = prev = n
    ranges.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


class IssueKind:
    """Kinds of :class:`Issue`, grouped by their effect on the output."""

    # An attribute cell that is set to MISSING_ATTR_CODE.
    BLANK = "blank"
    NON_NUMERIC = "non_numeric"
    NON_INTEGER = "non_integer"
    UNKNOWN_VALUE = "unknown_value"  # a code or raw value without a valid iRAP code
    # A value that is used with an interpretation.
    SPEED_LIMIT_BELOW_30_KMH = "speed_limit_below_30_kmh"  # coded as <30km/h
    BLANK_ANNOT_LOC_OFFSET = "blank_annot_loc_offset"  # row kept without the offset check
    DUPLICATE_TIE = "duplicate_dropped_in_tie"  # an equally filled earlier row was kept
    DUPLICATE_CONFLICT = "duplicate_conflict"  # value: (kept code, dropped code)
    # A file-level irregularity.
    EXTRA_SHEETS_IGNORED = "extra_sheets_ignored"  # value: the ignored sheet names
    IGNORED_COLUMN_WITH_VALUES = "ignored_column_with_values"
    EMPTY_ATTRIBUTE = "empty_attribute"  # all cells blank
    NO_ANNOT_LOC_OFFSET_COLUMN = "no_annot_loc_offset_column"


SET_TO_MISSING_KINDS = (
    IssueKind.UNKNOWN_VALUE, IssueKind.NON_NUMERIC, IssueKind.NON_INTEGER, IssueKind.BLANK)
ISSUE_KINDS_BY_EFFECT: dict[str, tuple[str, ...]] = {
    f"Attribute cells set to {MISSING_ATTR_CODE}": SET_TO_MISSING_KINDS,
    "Values interpreted": (
        IssueKind.SPEED_LIMIT_BELOW_30_KMH, IssueKind.BLANK_ANNOT_LOC_OFFSET,
        IssueKind.DUPLICATE_TIE, IssueKind.DUPLICATE_CONFLICT),
    "File level": (
        IssueKind.EXTRA_SHEETS_IGNORED, IssueKind.IGNORED_COLUMN_WITH_VALUES,
        IssueKind.EMPTY_ATTRIBUTE, IssueKind.NO_ANNOT_LOC_OFFSET_COLUMN),
}
ISSUE_KIND_ORDER = [k for kinds in ISSUE_KINDS_BY_EFFECT.values() for k in kinds]


@dataclass(frozen=True)
class Issue:
    """A value that was set to missing or interpreted, or a file-level irregularity."""

    kind: str
    file: str
    row: int | None = None  # Excel row number, None for a file-level issue
    column: str | None = None  # canonical attribute name for an attribute column
    value: T.Hashable = None


class InvalidCellValueError(ValueError):
    """Raised for a non-blank cell value that cannot be converted. ``kind`` is an IssueKind."""

    def __init__(self, kind: str, value: T.Any):
        super().__init__(f"{kind}: {value!r}")
        self.kind = kind
        self.value = value


def parse_seg_id(cell: T.Any) -> int | None:
    """Extracts the segment id from an Image Reference FPZ cell ("seg. no. <N>"), or returns None."""
    if is_blank(cell):
        return None
    m = SEG_ID_RE.search(str(cell))
    return int(m.group(1)) if m else None


def parse_float(cell: T.Any) -> float | None:
    """Converts a cell to float, or returns None for a blank cell.

    Raises :class:`InvalidCellValueError` for a non-numeric cell.
    """
    if is_blank(cell):
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        raise InvalidCellValueError(IssueKind.NON_NUMERIC, cell) from None


def parse_int_code(cell: T.Any) -> int | None:
    """Converts a cell to int, or returns None for a blank cell.

    Raises :class:`InvalidCellValueError` for a non-numeric or non-integer cell.
    """
    f = parse_float(cell)  # Excel often loads integer codes as floats (e.g. 1.0).
    if f is None:
        return None
    if not f.is_integer():
        raise InvalidCellValueError(IssueKind.NON_INTEGER, cell)
    return int(f)


# The result of an attribute value classifier: the iRAP code (None for a blank
# cell), and the kind of the interpretation issue, if any.
ClassifiedValue = tuple[int | None, str | None]


def classify_road_volume(cell: T.Any) -> ClassifiedValue:
    """Maps an intersecting road volume (vehicle count) to an iRAP code.

    Codes match attribute_metadata.json:
        1: ≥15,000 vehicles
        2: 10,000 to <15,000 vehicles
        3: 5,000 to <10,000 vehicles
        4: 1,000 to <5,000 vehicles
        5: 100 to <1,000 vehicles
        6: 1 to <100 vehicles
        7: Not applicable (0 vehicles)

    Raises :class:`InvalidCellValueError` for a non-numeric count, and for a
    count that is negative or between 0 and 1.
    """
    v = parse_float(cell)
    if v is None:
        return None, None
    # The smallest count of the codes 1 to 6.
    for code, min_count in enumerate((15000, 10000, 5000, 1000, 100, 1), start=1):
        if v >= min_count:
            return code, None
    if v == 0:
        return 7, None
    raise InvalidCellValueError(IssueKind.UNKNOWN_VALUE, cell)


def classify_speed_limit_kmh(cell: T.Any) -> ClassifiedValue:
    """Maps a speed limit in km/h to an iRAP code.

    Codes match attribute_metadata.json:
        1: <30km/h (also 30 km/h, which has no code of its own)
        3: 40km/h, 5: 50km/h, ..., 23: 140km/h (each step of 10 km/h adds 2)
        25: ≥150km/h

    A speed limit below 30 km/h is reported (SPEED_LIMIT_BELOW_30_KMH), as it
    shares the code with 30 km/h. Raises :class:`InvalidCellValueError` for a
    speed limit that is not a positive integer, or that is between 30 and
    150 km/h but not a multiple of 10.
    """
    kmh = parse_int_code(cell)
    if kmh is None:
        return None, None
    if 0 < kmh < 30:
        return 1, IssueKind.SPEED_LIMIT_BELOW_30_KMH
    if kmh == 30:
        return 1, None
    if kmh >= 150:
        return 25, None
    if kmh > 30 and kmh % 10 == 0:
        return 3 + 2 * (kmh - 40) // 10, None
    raise InvalidCellValueError(IssueKind.UNKNOWN_VALUE, kmh)


# Attributes whose table cells contain raw values (not iRAP codes) that need
# classification into codes. The classifier is called instead of
# ``parse_int_code``, and its result is validated against
# ``valid_codes_per_attr`` like any other code, so that a mismatch with
# attribute_metadata.json is not silent.
ATTR_VALUE_CLASSIFIERS: dict[str, T.Callable[[T.Any], ClassifiedValue]] = {
    "Intersecting road volume": classify_road_volume,
    "Speed limit": classify_speed_limit_kmh,
}


def read_sheet(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Reads the first non-empty sheet of an XLS/XLSX file as a DataFrame.

    Returns the DataFrame (empty if all sheets are empty) and the names of the
    other non-empty sheets, which are ignored.
    """
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    sheets = pd.read_excel(path, sheet_name=None, engine=engine, dtype=object)
    non_empty_sheets = {n: df for n, df in sheets.items() if not df.empty}
    if not non_empty_sheets:
        return pd.DataFrame(), []
    first_name, *other_names = non_empty_sheets
    return non_empty_sheets[first_name], other_names


def excel_row(df_index: int) -> int:
    """Returns the Excel row number of a DataFrame row read by :func:`read_sheet` (the header is row 1)."""
    return df_index + 2


def file_is_coding_table(
    df: pd.DataFrame,
    attribute_names: T.Sequence[str],
) -> bool:
    """True if the file looks like a labeling sheet.

    Heuristic: must have a "Section" column AND at least one of the canonical
    attribute columns (case-insensitive after whitespace stripping). Files
    lacking both are silently skipped.
    """
    headers = {normalize_header(c).lower() for c in df.columns}
    return "section" in headers and any(a.lower() in headers for a in attribute_names)


def validate_columns(
    df: pd.DataFrame,
    attribute_names: T.Sequence[str],
    *,
    file: Path,
    ignored_attributes: list[str] | None = None,
    name_mapping: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Resolves the table column of every required column.

    Returns ``{logical_name: actual_column_name}`` for every required column,
    and the ignored attributes whose column is present in the table and has a
    non-blank value (so that callers can report that values are ignored).

    Header lookup is case-insensitive (after whitespace stripping). Attribute
    columns are looked up by their metadata name directly (table column names
    are expected to match); when that fails, ``name_mapping`` (BH name ->
    Vietnam table header) provides a fallback so the result still keys on the
    canonical BH name.

    Raises ``ValueError`` if any required column is missing. The message lists
    missing required columns next to their closest header found in the file,
    and separately lists headers present in the file but unused by the
    schema (so renames vs. truly-extra columns are easy to tell apart).
    """
    ignored_attributes = ignored_attributes or []
    name_mapping = name_mapping or {}

    actual = {normalize_header(c): c for c in df.columns}
    ci_index: dict[str, str] = {}
    for h in actual:
        ci_index.setdefault(h.lower(), h)

    def resolve_one(name: str) -> str | None:
        if name in actual:
            return name
        h = ci_index.get(name.lower())
        if h is not None:
            return h
        mapped = name_mapping.get(name)
        if mapped is not None:
            return mapped if mapped in actual else ci_index.get(mapped.lower())
        return None

    # Scalar columns: look up by their own name.
    resolved: dict[str, str | None] = {r: resolve_one(r) for r in REQUIRED_SCALAR_COLUMNS}

    # Attribute columns: look up by their metadata name (== table column name).
    for attr in attribute_names:
        resolved[attr] = resolve_one(attr)

    missing = [r for r, h in resolved.items() if h is None]
    if missing:
        matched = sorted(h for h in resolved.values() if h is not None)
        consumed = set(matched)
        unused_found = sorted(set(actual) - consumed)

        suggestion_lines = []
        for r in missing:
            cands = difflib.get_close_matches(r, unused_found, n=1, cutoff=0.6)
            if cands:
                ratio = difflib.SequenceMatcher(None, r, cands[0]).ratio()
                suggestion_lines.append(
                    f"  - {r!r}  ->  closest in table: {cands[0]!r} "
                    f"(sim={ratio:.2f})"
                )
            else:
                suggestion_lines.append(f"  - {r!r}  ->  no close match")
        matched_lines = [f"  - {h!r}" for h in matched]
        unused_lines = [f"  - {h!r}" for h in unused_found]

        raise ValueError(
            f"{file}: column header mismatch.\n"
            f"\nMissing required columns ({len(missing)}):\n"
            + "\n".join(suggestion_lines)
            + f"\n\nMatched required columns ({len(matched)}):\n"
            + "\n".join(matched_lines)
            + f"\n\nUnused columns in table ({len(unused_found)}):\n"
            + "\n".join(unused_lines)
        )

    result = {r: actual[h] for r, h in resolved.items() if h is not None}
    ignored_with_values = [
        attr for attr in ignored_attributes
        if (h := resolve_one(attr)) is not None
        and not df[actual[h]].apply(is_blank).all()
    ]
    return result, ignored_with_values


# -----------------------------------------------------------------------------
# Per-row processing
# -----------------------------------------------------------------------------


@dataclass
class ParseInputs:
    """Resolved inputs for parsing a dataset's coding tables."""

    attribute_names: list[str]
    valid_codes: dict[str, set[int]]
    ignored_attributes: list[str]
    name_mapping: dict[str, str]
    coding_tables_dir: Path
    files: list[Path]


@dataclass
class ParseReport:
    """Mutable bookkeeping gathered while parsing, used to build parse_report.json.

    Pass an instance to :func:`collect_all_rows` to collect it; pass ``None``
    there to parse without the (otherwise unused) bookkeeping.
    """

    non_empty_rows_per_file: dict[str, int] = field(default_factory=dict)
    drop_counts_per_file: dict[str, Counter] = field(default_factory=dict)
    rows_kept_per_file: dict[str, int] = field(default_factory=dict)
    skipped_files: dict[str, str] = field(default_factory=dict)
    processed_files: list[str] = field(default_factory=list)
    # Issues of the parsed files and the kept rows, before duplicate resolution.
    issues: list[Issue] = field(default_factory=list)

    @property
    def drop_counts(self) -> Counter:
        return sum(self.drop_counts_per_file.values(), Counter())


class RowDropReason:
    NON_NUMERIC_SCALAR = "non_numeric_scalar"
    LENGTH_MISMATCH = "length_mismatch"
    MISSING_SCALAR = "missing_scalar"
    BAD_SEG_ID = "bad_seg_id"
    NO_ATTRIBUTES = "no_attributes"
    MISSING_ATTRIBUTE = "missing_attribute"  # with drop_missing_attrs
    INVALID_ATTRIBUTE_VALUE = "invalid_attribute_value"  # with drop_missing_attrs
    ANNOT_LOC_OFFSET_TOO_FAR = "annot_loc_offset_too_far"


def parse_row_attributes(
    raw: T.Mapping[T.Any, T.Any],
    attribute_names: T.Sequence[str],
    cols: T.Mapping[str, str],
    valid_codes_per_attr: T.Mapping[str, set[int]],
    file: str,
    row: int,
) -> tuple[dict[str, int], list[Issue]]:
    """Resolves every attribute code of one row.

    Returns the codes, and the issues of the cells, which have the given
    ``file`` and ``row``. The cells of classifier-backed attributes are
    converted from raw values to codes, the others are parsed as integer
    codes. A cell that is blank, not an integer, or without a code in
    ``valid_codes_per_attr`` becomes MISSING_ATTR_CODE, with an issue of a
    kind in SET_TO_MISSING_KINDS.
    """
    attrs = dict.fromkeys(attribute_names, MISSING_ATTR_CODE)
    issues: list[Issue] = []
    for attr in attribute_names:
        cell = raw[cols[attr]]
        classifier = ATTR_VALUE_CLASSIFIERS.get(attr)
        try:
            code, interpretation = (classifier(cell) if classifier is not None
                                    else (parse_int_code(cell), None))
        except InvalidCellValueError as e:
            issues.append(Issue(e.kind, file, row, attr, e.value))
            continue
        if code is None:
            issues.append(Issue(IssueKind.BLANK, file, row, attr))
        elif code not in valid_codes_per_attr[attr]:
            issues.append(Issue(IssueKind.UNKNOWN_VALUE, file, row, attr, code))
        else:
            attrs[attr] = code
            if interpretation is not None:
                issues.append(Issue(interpretation, file, row, attr, cell))
    return attrs, issues


def process_file(
    df: pd.DataFrame,
    path: Path,
    inputs: ParseInputs,
    report: ParseReport,
    *,
    drop_missing_attrs: bool = False,
) -> list[dict]:
    """Parse one coding-table sheet (``df``, read from ``path``) into normalized row dicts.

    Records are dicts with keys:
        section, distance, length, lat, lon, seg_id, comments, source_file,
        source_row (Excel row number), and one entry per attribute (iRAP code
        as int).

    Mutates the per-file tallies and issues in ``report`` for the parse report.
    Issues are recorded for the kept rows only. The dropped rows are counted
    by drop reason. Blank cells of an attribute that is blank in the whole
    file are reported once, as EMPTY_ATTRIBUTE.
    """
    attribute_names = inputs.attribute_names
    cols, ignored_with_values = validate_columns(
        df, attribute_names, file=path,
        ignored_attributes=inputs.ignored_attributes,
        name_mapping=inputs.name_mapping,
    )

    report.issues.extend(Issue(IssueKind.IGNORED_COLUMN_WITH_VALUES, path.name, column=attr)
                         for attr in ignored_with_values)

    # Optional FPZ-to-annotation offset column (not a required column).
    norm_to_actual = {normalize_header(c).lower(): c for c in df.columns}
    annot_loc_offset_col = norm_to_actual.get(ANNOT_LOC_OFFSET_COLUMN.lower())
    if annot_loc_offset_col is None:
        report.issues.append(Issue(IssueKind.NO_ANNOT_LOC_OFFSET_COLUMN, path.name,
                                   column=ANNOT_LOC_OFFSET_COLUMN))

    attr_blank = pd.DataFrame(
        {a: df[cols[a]].apply(is_blank) for a in attribute_names}
    )
    empty_attrs = [attr for attr in attribute_names if attr_blank[attr].all()]
    report.issues.extend(Issue(IssueKind.EMPTY_ATTRIBUTE, path.name, column=attr)
                         for attr in empty_attrs)

    # A row with no values for any attribute is treated as empty (padding /
    # non-coding row); only non-empty rows count toward num_rows_non_empty.
    report.non_empty_rows_per_file[path.name] = int((~attr_blank.all(axis=1)).sum())

    rows: list[dict] = []
    file_drops: Counter = Counter()

    # Plain dicts: per-cell lookups on them are far cheaper than on the
    # ``pd.Series`` rows that ``iterrows`` yields.
    for idx, raw in zip(df.index, df.to_dict("records")):
        # Skip wholly empty rows (XLSX often pads with blanks).
        if all(is_blank(raw[cols[k]]) for k in REQUIRED_SCALAR_COLUMNS):
            continue
        row = excel_row(idx)

        try:
            annot_loc_offset = (parse_float(raw[annot_loc_offset_col])
                                if annot_loc_offset_col is not None else None)
            length = parse_float(raw[cols["Length"]])
            distance = parse_float(raw[cols["Distance"]])
            lat = parse_float(raw[cols["Latitude start"]])
            lon = parse_float(raw[cols["Longitude start"]])
        except InvalidCellValueError:
            file_drops[RowDropReason.NON_NUMERIC_SCALAR] += 1
            continue

        # Drop rows whose annotation sits too far from the FPZ point.
        if annot_loc_offset is not None and annot_loc_offset > MAX_ANNOT_LOC_OFFSET_M:
            file_drops[RowDropReason.ANNOT_LOC_OFFSET_TOO_FAR] += 1
            continue

        if length is None or abs(length - EXPECTED_LENGTH_KM) > LENGTH_TOLERANCE:
            file_drops[RowDropReason.LENGTH_MISMATCH] += 1
            continue

        section = raw[cols["Section"]]
        if is_blank(section) or distance is None or lat is None or lon is None:
            file_drops[RowDropReason.MISSING_SCALAR] += 1
            continue
        section = str(section).strip()

        seg_id = parse_seg_id(raw[cols[SEG_ID_COLUMN]])
        if seg_id is None:
            file_drops[RowDropReason.BAD_SEG_ID] += 1
            continue

        attrs, attr_issues = parse_row_attributes(raw, attribute_names, cols, inputs.valid_codes,
                                                  path.name, row)
        # A row with no genuine attribute value (all blank / file-level-empty)
        # is a non-coding / padding row carrying no labeling signal; drop it
        # rather than keep an all-(-1) record.
        if not any(is_real_attribute_code(v) for v in attrs.values()):
            file_drops[RowDropReason.NO_ATTRIBUTES] += 1
            continue
        if drop_missing_attrs:
            missing_kinds = {i.kind for i in attr_issues if i.kind in SET_TO_MISSING_KINDS}
            if missing_kinds:
                file_drops[RowDropReason.MISSING_ATTRIBUTE if missing_kinds == {IssueKind.BLANK}
                           else RowDropReason.INVALID_ATTRIBUTE_VALUE] += 1
                continue

        report.issues.extend(i for i in attr_issues
                             if not (i.kind == IssueKind.BLANK and i.column in empty_attrs))
        if annot_loc_offset_col is not None and annot_loc_offset is None:
            report.issues.append(Issue(IssueKind.BLANK_ANNOT_LOC_OFFSET, path.name, row,
                                       ANNOT_LOC_OFFSET_COLUMN))

        comments = raw[cols["Comments"]]
        comments = "" if is_blank(comments) else str(comments)

        rec = {
            "section": section,
            "distance": distance,
            "length": length,
            "lat": lat,
            "lon": lon,
            "seg_id": str(seg_id),
            "comments": comments,
            "source_file": path.name,
            # Provenance for downstream tooling. Dropped before rows.parquet.
            "source_row": row,
        }
        rec.update(attrs)
        rows.append(rec)

    if file_drops:
        report.drop_counts_per_file[path.name] = file_drops
    return rows


def num_filled_attribute_cells(rec: dict, attribute_names: T.Sequence[str]) -> int:
    """Count genuinely-coded attribute cells in a record (used for dup resolution)."""
    return sum(1 for a in attribute_names if is_real_attribute_code(rec.get(a)))


def resolve_duplicates(
    rows: list[dict],
    attribute_names: T.Sequence[str],
) -> tuple[list[dict], list[dict], list[Issue]]:
    """Keeps one row per seg_id: the one with the most filled cells, and the first one in a tie.

    Returns the kept rows, the dropped duplicate rows, and the issues of the
    dropped rows: DUPLICATE_TIE for a row dropped in a tie, and
    DUPLICATE_CONFLICT (value: ``(kept code, dropped code)``) for each
    attribute where the dropped row has a genuine code that differs from the
    kept one, including where the kept row's cell is missing.
    """
    by_seg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_seg[r["seg_id"]].append(r)

    kept: list[dict] = []
    dropped: list[dict] = []
    issues: list[Issue] = []
    for group in by_seg.values():
        # Stable sort by the number of filled cells: keep the first.
        scored = sorted(((num_filled_attribute_cells(r, attribute_names), r) for r in group),
                        key=lambda filled_and_row: -filled_and_row[0])
        (winner_filled, winner), losers = scored[0], scored[1:]
        kept.append(winner)
        for num_filled, r in losers:
            dropped.append(r)
            file, row = r["source_file"], r["source_row"]
            if num_filled == winner_filled:
                issues.append(Issue(IssueKind.DUPLICATE_TIE, file, row, SEG_ID_COLUMN))
            issues.extend(
                Issue(IssueKind.DUPLICATE_CONFLICT, file, row, attr, (winner[attr], r[attr]))
                for attr in attribute_names
                if is_real_attribute_code(r[attr]) and r[attr] != winner[attr])
    return kept, dropped, issues


# -----------------------------------------------------------------------------
# Parsing entry points (shared by the CLI and other tools, e.g. cross-annotator
# evaluation, which need the pre-dedup rows)
# -----------------------------------------------------------------------------


class ParseSetupError(Exception):
    """Raised when the dataset layout/inputs are unusable. Carries an exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def prepare_parse_inputs(data_dir: Path) -> ParseInputs:
    """Resolve attribute metadata, ignore/rename config, and the coding-table files.

    Loads ``attribute_metadata.json`` and the ``incompatible_attributes.json``
    mapping, applies the ignored-attribute filter, and (re-)extracts the coding
    tables. Raises :class:`ParseSetupError` (with a CLI exit code) if the raw
    dir, metadata file, or any spreadsheet files are missing.
    """
    raw = layout.raw_dir(data_dir)
    if not raw.is_dir():
        raise ParseSetupError(
            f"{raw} not found. Expected a directory containing "
            f"attribute_metadata.json and coding-tables[.zip].",
            2,
        )
    attr_meta_path = layout.attr_meta_path(data_dir)
    if not attr_meta_path.is_file():
        raise ParseSetupError(f"{attr_meta_path} not found.", 2)

    attribute_names, valid_codes = load_attribute_metadata(attr_meta_path)
    ignored_attributes, name_mapping = load_incompatible_attributes(Path(__file__).parent)
    if ignored_attributes:
        ignored_set = set(ignored_attributes)
        attribute_names = [a for a in attribute_names if a not in ignored_set]
        print("Attributes:")
        for attr in attribute_names:
            print(f"  {attr}")
        valid_codes = {k: v for k, v in valid_codes.items() if k not in ignored_set}
        print(f"Ignoring {len(ignored_attributes)} attribute(s) per mapping file: "
              + ", ".join(repr(a) for a in sorted(ignored_attributes)))
    if name_mapping:
        print(f"Translating {len(name_mapping)} attribute name(s) to their "
              f"Vietnam table header per mapping file.")
    coding_tables_dir = resolve_coding_tables_dir(data_dir)
    files = find_table_files(coding_tables_dir)
    if not files:
        raise ParseSetupError(
            f"no .xls/.xlsx files under {coding_tables_dir}", 1
        )
    print(f"Found {len(files)} spreadsheet file(s) under {coding_tables_dir}")
    return ParseInputs(
        attribute_names=attribute_names,
        valid_codes=valid_codes,
        ignored_attributes=ignored_attributes,
        name_mapping=name_mapping,
        coding_tables_dir=coding_tables_dir,
        files=files,
    )


def collect_all_rows(
    inputs: ParseInputs,
    *,
    drop_missing_attrs: bool = False,
    report: ParseReport | None = None,
) -> list[dict]:
    """Parse every coding-table file and return the pre-dedup rows.

    This is the row list *before* :func:`resolve_duplicates`, so a ``seg_id``
    coded by several annotators appears once per source file. When ``report`` is
    given its bookkeeping fields are populated for parse_report.json; when
    ``None`` the tallies are gathered into a scratch instance and discarded.
    """
    rep = report if report is not None else ParseReport()
    all_rows: list[dict] = []
    for path in tqdm(inputs.files, desc="Parsing", unit="file"):
        try:
            df, ignored_sheets = read_sheet(path)
        except Exception as e:
            print(f"ERROR reading {path}: {e}", file=sys.stderr)
            raise
        if ignored_sheets:
            rep.issues.append(Issue(IssueKind.EXTRA_SHEETS_IGNORED, path.name,
                                    value=tuple(ignored_sheets)))
        if df.empty:
            print(f"INFO: {path.name}: skipping – empty sheet", file=sys.stderr)
            rep.skipped_files[path.name] = "empty sheet"
            continue
        if not file_is_coding_table(df, inputs.attribute_names):
            print(f"INFO: {path.name}: skipping – not a coding table "
                  f"(no 'Section' column or no recognizable attribute column)",
                  file=sys.stderr)
            rep.skipped_files[path.name] = (
                "not a coding table (no 'Section' column or no recognizable "
                "attribute column)"
            )
            continue
        try:
            rows = process_file(df, path, inputs, rep,
                                drop_missing_attrs=drop_missing_attrs)
        except ValueError as e:
            print(f"WARN: {path.name}: skipping – column validation failed:\n{e}",
                  file=sys.stderr)
            rep.skipped_files[path.name] = f"column validation failed: {e}"
            continue
        rep.processed_files.append(path.name)
        rep.rows_kept_per_file[path.name] = len(rows)
        all_rows.extend(rows)
    return all_rows


# -----------------------------------------------------------------------------
# Issue reporting
# -----------------------------------------------------------------------------

MAX_SUMMARY_LINES_PER_ISSUE_KIND = 10

IssueGroupKey = tuple[str, str, str | None, T.Hashable]  # (file, kind, column, value)


def group_issues(issues: T.Iterable[Issue]) -> dict[IssueGroupKey, list[int]]:
    """Groups issues by (file, kind, column, value), with the Excel rows of each group.

    Groups are ordered by file, kind (as in ISSUE_KINDS_BY_EFFECT), column and
    value. The rows of a file-level group are empty.
    """
    def sort_key(i: Issue):
        return (i.file, ISSUE_KIND_ORDER.index(i.kind), i.column or "", str(i.value),
                i.row or 0)

    groups: dict[IssueGroupKey, list[int]] = {}
    for i in sorted(issues, key=sort_key):
        rows = groups.setdefault((i.file, i.kind, i.column, i.value), [])
        if i.row is not None:
            rows.append(i.row)
    return groups


def issue_label(column: str | None, value: T.Hashable) -> str:
    """Returns ``"<column> = <value>"``, leaving out the parts that are None."""
    parts = [] if column is None else [column]
    if value is not None:
        parts.append(repr(value))
    return " = ".join(parts)


def to_json_value(value: T.Any) -> T.Any:
    """Converts an issue value to a JSON-serializable value (str for unusual cell types)."""
    if isinstance(value, tuple):
        return [to_json_value(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def issue_group_entry(kind: str, column: str | None, value: T.Hashable, rows: list[int]) -> dict:
    """Returns the parse_report.json entry of an issue group from :func:`group_issues`."""
    return {
        "kind": kind,
        **({"column": column} if column is not None else {}),
        **({"value": to_json_value(value)} if value is not None else {}),
        **({"num_rows": len(rows), "rows": format_row_ranges(rows)} if rows else {}),
    }


def count_issues_by_kind(issues: T.Iterable[Issue]) -> dict[str, int]:
    """Counts issues by kind, in the order of ISSUE_KINDS_BY_EFFECT."""
    counts = Counter(i.kind for i in issues)
    return {kind: counts[kind] for kind in ISSUE_KIND_ORDER if counts[kind]}


def print_issue_warnings(issue_groups: T.Mapping[IssueGroupKey, list[int]]) -> None:
    """Prints a warning per issue group of :func:`group_issues` to stderr."""
    for (fname, kind, column, value), rows in issue_groups.items():
        label = issue_label(column, value)
        where = f" in {len(rows)} row(s): {format_row_ranges(rows)}" if rows else ""
        print(f"WARN: {fname}: {kind}{f' ({label})' if label else ''}{where}", file=sys.stderr)


def print_issue_summary(issues: T.Sequence[Issue]) -> None:
    """Prints the numbers of issues by effect, kind, column and value.

    Shows at most MAX_SUMMARY_LINES_PER_ISSUE_KIND (column, value) lines per
    kind, the most frequent first.
    """
    if not issues:
        return
    kind_to_counts: dict[str, Counter] = defaultdict(Counter)  # kind -> (column, value) -> number
    for i in issues:
        kind_to_counts[i.kind][i.column, i.value] += 1
    print("Issues (cells, rows or files, with the duplicate issues of the dropped duplicates):")
    for effect, kinds in ISSUE_KINDS_BY_EFFECT.items():
        present_kinds = [kind for kind in kinds if kind in kind_to_counts]
        if present_kinds:
            print(f"  {effect}:")
        for kind in present_kinds:
            counts = sorted(kind_to_counts[kind].items(),
                            key=lambda c: (-c[1], c[0][0] or "", str(c[0][1])))
            print(f"    {kind}: {sum(n for _, n in counts)}")
            for (column, value), n in counts[:MAX_SUMMARY_LINES_PER_ISSUE_KIND]:
                print(f"      {issue_label(column, value)}: {n}")
            if len(counts) > MAX_SUMMARY_LINES_PER_ISSUE_KIND:
                print(f"      ... and {len(counts) - MAX_SUMMARY_LINES_PER_ISSUE_KIND} more")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument("data_dir", type=Path,
                        help="iRAP-Vietnam dataset root.")
    parser.add_argument(
        "--drop-rows-missing-attributes", action="store_true",
        help="Drop rows with any blank or invalid attribute cell. "
             "By default such rows are kept with the cells set to "
             f"{MISSING_ATTR_CODE} (MISSING_ATTR_CODE).",
    )
    args = parser.parse_args(argv)

    try:
        inputs = prepare_parse_inputs(args.data_dir)
    except ParseSetupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.exit_code

    attribute_names = inputs.attribute_names

    rep = ParseReport()
    all_rows = collect_all_rows(
        inputs,
        drop_missing_attrs=args.drop_rows_missing_attributes,
        report=rep,
    )

    kept, dup_dropped, dup_issues = resolve_duplicates(all_rows, attribute_names)
    issues = rep.issues + dup_issues
    issue_groups = group_issues(issues)
    # Print the warnings before the summary, so that the summary is not buried
    # above them. The summary is one contiguous block, in the order rows flow
    # through the pipeline.
    print_issue_warnings(issue_groups)
    total_dropped = sum(rep.drop_counts.values())
    if rep.drop_counts:
        total_rows = len(all_rows) + total_dropped
        print(f"Rows dropped during parsing ({total_dropped}/{total_rows} total):")
        for reason, count in rep.drop_counts.most_common():
            print(f"  {reason}: {count}")
    if rep.drop_counts_per_file:
        print("Rows dropped during parsing, per file:")
        for fname in sorted(
            rep.drop_counts_per_file,
            key=lambda f: (-sum(rep.drop_counts_per_file[f].values()), f),
        ):
            fc = rep.drop_counts_per_file[fname]
            file_dropped = sum(fc.values())
            file_total = rep.rows_kept_per_file.get(fname, 0) + file_dropped
            print(f"  {fname} ({file_dropped}/{file_total} total):")
            for reason, count in fc.most_common():
                print(f"    {reason}: {count}")
    print(f"Collected {len(all_rows)} rows before duplicate resolution.")
    print(f"After duplicate resolution: {len(kept)} kept, "
          f"{len(dup_dropped)} duplicate row(s) dropped.")

    # Missing-attribute tallies computed over the final kept rows so every
    # count below refers to the written output.
    missing_per_attr_kept: Counter = Counter()
    total_rows_with_missing = 0
    for rec in kept:
        row_has_missing = False
        for attr in attribute_names:
            if rec.get(attr) == MISSING_ATTR_CODE:
                missing_per_attr_kept[attr] += 1
                row_has_missing = True
        if row_has_missing:
            total_rows_with_missing += 1
    num_files_empty_per_attr = Counter(
        i.column for i in issues if i.kind == IssueKind.EMPTY_ATTRIBUTE)
    if total_rows_with_missing:
        attr_missing_totals = sorted(
            ((attr, missing_per_attr_kept.get(attr, 0)) for attr in attribute_names),
            key=lambda kv: (-kv[1], kv[0]),
        )
        print(f"Rows kept with at least one missing ({MISSING_ATTR_CODE}) attribute: "
              f"{total_rows_with_missing}")
        for attr, count in attr_missing_totals:
            if count:
                nfe = num_files_empty_per_attr[attr]
                suffix = f"  [entirely blank in {nfe} file(s)]" if nfe > 0 else ""
                print(f"  {attr}: {count}{suffix}")
    print_issue_summary(issues)

    layout.work_dir(args.data_dir).mkdir(parents=True, exist_ok=True)
    # ``source_row`` is provenance for other tools (e.g. cross-annotator eval);
    # keep it out of rows.parquet so the public schema stays stable.
    df_out = pd.DataFrame(kept).drop(columns=["source_row"], errors="ignore")
    parquet_path = layout.rows_path(args.data_dir)
    df_out.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} ({len(df_out)} rows)")

    def file_entry(fname: str) -> dict:
        entry: dict = {
            "status": "processed",
            "num_rows_non_empty": rep.non_empty_rows_per_file.get(fname, 0),
            "num_rows_kept": rep.rows_kept_per_file.get(fname, 0),
        }
        file_drops = rep.drop_counts_per_file.get(fname)
        if file_drops:
            entry["rows_dropped_by_reason"] = dict(file_drops)
        return entry

    files_section: dict[str, dict] = {}
    for fname in rep.processed_files:
        files_section[fname] = file_entry(fname)
    for fname, reason in rep.skipped_files.items():
        files_section[fname] = {"status": "skipped", "skip_reason": reason}
    # Every file with an issue is processed or skipped.
    for (fname, kind, column, value), rows in issue_groups.items():
        files_section[fname].setdefault("issues", []).append(
            issue_group_entry(kind, column, value, rows))
    files_section = dict(sorted(files_section.items()))

    issues_per_column: dict[str | None, list[Issue]] = defaultdict(list)
    for i in issues:
        issues_per_column[i.column].append(i)

    report = {
        "meta": {
            "parsed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "coding_tables_dir": str(
                inputs.coding_tables_dir.relative_to(args.data_dir)
            ),
            "attribute_metadata_path": str(
                layout.attr_meta_path(args.data_dir).relative_to(args.data_dir)
            ),
            "ignored_attributes": sorted(inputs.ignored_attributes),
        },
        "summary": {
            "files": {
                "total": len(inputs.files),
                "processed": len(rep.processed_files),
                "skipped": len(rep.skipped_files),
            },
            "rows": {
                "kept": len(kept),
                "with_any_missing_attribute": total_rows_with_missing,
                "dropped_total": total_dropped,
                "dropped_by_reason": dict(rep.drop_counts),
                "dropped_duplicates": len(dup_dropped),
            },
            "num_sections": df_out["section"].nunique() if not df_out.empty else 0,
            "issues": count_issues_by_kind(issues),
        },
        "files": files_section,
        # Per attribute: the number of kept rows (after duplicate resolution)
        # in which it is missing, and the numbers of its issues by kind.
        "attributes": {
            attr: {
                **({"num_rows_missing": n} if (n := missing_per_attr_kept.get(attr, 0)) > 0 else {}),
                **({"issues": count_issues_by_kind(issues_per_column[attr])}
                   if attr in issues_per_column else {}),
            }
            for attr in attribute_names
        },
    }
    report_path = layout.parse_report_path(args.data_dir)
    report_text = json.dumps(report, indent=2, ensure_ascii=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
