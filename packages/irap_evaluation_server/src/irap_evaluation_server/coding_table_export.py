"""The export of the submissions of a model, e.g. one per split, as one iRAP coding table (see
`irap_evaluation.reports.coding_tables`), with the packaged template."""

import dataclasses as dc
import datetime
import typing as T
from pathlib import Path

import irap_evaluation as ie
from irap_data.metadata import IRAPMetadata
from irap_evaluation.reports import coding_tables

from .archive import ModelArchive, make_label_file_name
from .database import get_utc_now
from .datasets import DatasetContext

#: The file formats (suffixes without the dot).
CODING_TABLE_FORMATS = ("xlsx", "csv")
#: The first characters that make spreadsheet programs read a .csv cell as a formula, which a
#: coder name must not start with, since it can be a method name that a user chose. An .xlsx file
#: has it as text (`irap_evaluation.reports.coding_tables.write_coding_table`).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
#: Excel does not open a file whose path is longer than 218 characters, so the file name leaves
#: about 58 characters for the directory, e.g. a downloads directory.
MAX_CODING_TABLE_FILE_NAME_BYTES = 160


@dc.dataclass(frozen=True)
class CodingTableJob:
    """A checked coding table of a model, with what `write_model_coding_table` needs, which can
    be pickled for another process.

    Attributes:
        file_name: The name of the downloaded file.
        prediction_paths: Of the submissions of the chosen splits, in the order of `file_name`.
    """

    file_name: str
    prediction_paths: tuple[Path, ...]
    metadata: IRAPMetadata
    coder_name: str
    coding_date: str


def parse_coding_date(text: str) -> str:
    """Parses the coding date of an export as `YYYY-MM-DD`, today (in UTC) for ''.

    Raises:
        ValueError: If it is not a date in that format.
    """
    if not text:
        return get_utc_now().date().isoformat()
    try:
        return datetime.date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError(f"The coding date must be YYYY-MM-DD, got {text!r}.") from None


def write_model_coding_table(prediction_paths: T.Sequence[Path], metadata: IRAPMetadata,
                             path: Path, *, coder_name: str, coding_date: str) -> Path:
    """Writes the coding table of the predictions of one model to `path`, a `.xlsx` or `.csv`
    file, with the segments of all of them in road order.

    A `.xlsx` file of probabilistic predictions gets the confidence table as its second sheet
    (`coding_tables.predictions_to_confidence_table`). A `.csv` file has only the codes, since
    the confidence table would be a second file.

    It takes tens of seconds for a `.xlsx` file of large splits. In a thread, it delayed the
    server's event loop by over 10 s (2026-10-09), so the server runs it in a process
    (`run.cpu_bound`).

    Args:
        prediction_paths: Of one model, at least one, e.g. its submissions of some splits.
        metadata: Of the dataset, which gives the segment locations.
        coding_date: `YYYY-MM-DD`.

    Returns:
        `path`, so that a caller of `run.cpu_bound` can tell a written file from a cancelled
        call, which returns None.

    Raises:
        ValueError: For another suffix, or no predictions.
    """
    if path.suffix.lstrip(".") not in CODING_TABLE_FORMATS:
        raise ValueError(f"Expected a .xlsx or .csv path, got {path}.")
    if not prediction_paths:
        raise ValueError("At least one prediction file is required.")
    predictions_seq = [ie.read_predictions(p) for p in prediction_paths]
    template = coding_tables.load_coding_table_template()
    table = coding_tables.predictions_to_coding_table(
        predictions_seq, metadata, template, coder_name=coder_name, coding_date=coding_date)
    confidence_table = (
        coding_tables.predictions_to_confidence_table(predictions_seq, metadata, template)
        if path.suffix == ".xlsx" and all(p.output_kind == "probs" for p in predictions_seq)
        else None)
    coding_tables.write_coding_table(path, table, confidence_table)
    return path


def plan_model_coding_table(archive: ModelArchive,
                            dataset_contexts: T.Mapping[str, DatasetContext], model_id: int,
                            splits: T.Sequence[str], file_format: str, *, coder_name: str,
                            coding_date: str) -> CodingTableJob:
    """Checks a coding table of the files of some splits of a model and collects what writing it
    needs (`write_model_coding_table`).

    Args:
        splits: The splits of the files, in the order of the file name.
        file_format: One of `CODING_TABLE_FORMATS`.
        coder_name: '' for the shown method name (`archive.Model.shown_method_name`).
        coding_date: See `parse_coding_date`.

    Raises:
        LookupError: If there is no such model.
        ValueError: For another format, an invalid date, a dataset that is not configured, no
            splits, a split without a file, or a .csv coder name that spreadsheet programs read
            as a formula.
    """
    if file_format not in CODING_TABLE_FORMATS:
        raise ValueError(f"Unknown format {file_format!r}.")
    coding_date = parse_coding_date(coding_date)
    model = archive.get_model(model_id)
    context = dataset_contexts.get(model.dataset)
    if context is None:
        raise ValueError(f"The dataset {model.dataset!r} is not configured.")
    split_to_submission = {s.split: s for s in archive.list_model_submissions(model_id)}
    if not splits:
        raise ValueError("Choose at least one split.")
    if unknown := [s for s in splits if s not in split_to_submission]:
        raise ValueError(f"The model has no files of the splits {unknown}.")
    submissions = [split_to_submission[s] for s in dict.fromkeys(splits)]
    file_name = make_label_file_name(
        model.label, f".{'+'.join(s.split for s in submissions)}.coding_table.{file_format}",
        max_bytes=MAX_CODING_TABLE_FILE_NAME_BYTES)
    coder_name = coder_name or model.shown_method_name
    if file_format == "csv" and coder_name.startswith(_FORMULA_PREFIXES):
        raise ValueError(f"The coder name {coder_name!r} starts with {coder_name[0]!r}, which"
                         f" spreadsheet programs read as a formula. Enter another coder name.")
    return CodingTableJob(
        file_name=file_name,
        prediction_paths=tuple(archive.get_predictions_path(s.id) for s in submissions),
        metadata=context.metadata, coder_name=coder_name, coding_date=coding_date)
