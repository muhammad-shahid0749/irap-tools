"""Uploads of a prediction file, or of a .zip archive of the files of one model, one per split.
An upload is checked and planned as an update of the model (`archive.ModelUpdate`), and applied
after the user confirms its changes, if any (`ModelUpdate.confirmations`).

An uploaded file is at `ModelArchive.make_upload_path`. A planned upload keeps it until it is
applied, so that a refused or cancelled upload can be tried again, e.g. with another seed.
"""

import dataclasses as dc
import typing as T
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import irap_evaluation as ie

from .archive import (
    ARCHIVE_SUFFIX,
    ModelArchive,
    ModelUpdate,
    NewSubmission,
    Submission,
    compute_file_sha256,
    describe_model_fields,
    get_differing_model_fields,
)
from .datasets import DatasetContext

#: The most bytes of an uploaded file, and of the files extracted from an uploaded archive, whose
#: parquet files hardly compress. As `request_body` of `/api/uploads` in deployment/Caddyfile.
MAX_UPLOAD_NUM_BYTES = 2 ** 30
#: `MAX_UPLOAD_NUM_BYTES` for messages.
MAX_UPLOAD_SIZE_TEXT = f"{MAX_UPLOAD_NUM_BYTES / 2**30:g} GiB"
_COPY_CHUNK_NUM_BYTES = 2 ** 20


def parse_seed(text: str) -> int | None:
    """Parses the seed that a user entered, None for a blank text.

    Raises:
        ValueError: If it is not an integer.
    """
    if not text.strip():
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"The seed must be an integer, not {text.strip()!r}.") from None


def parse_optional_text(text: str) -> str | None:
    """Parses an optional text that a user entered, e.g. a description or a method display name
    with an upload, None for a blank text, which keeps the stored one."""
    return text.strip() or None


def is_archive_upload(uploaded_path: Path) -> bool:
    """Returns whether an upload at `ModelArchive.make_upload_path` is a .zip archive."""
    return uploaded_path.name.endswith(ARCHIVE_SUFFIX)


def _override_seed(model: ie.ModelInfo, seed: int | None) -> ie.ModelInfo:
    """Returns the model with the seed given at the upload, if it is not None."""
    return model if seed is None else dc.replace(model, seed=seed)


def _replace_model(predictions: ie.Predictions, model: ie.ModelInfo) -> ie.Predictions:
    return dc.replace(predictions, header=dc.replace(predictions.header, model=model))


def _get_rewritten_path(uploaded_path: Path) -> Path:
    """Returns where `_prepare_file` writes an uploaded file with a replaced seed."""
    return uploaded_path.with_name(f"{uploaded_path.name}.rewritten")


def _prepare_file(
    dataset_contexts: T.Mapping[str, DatasetContext],
    uploaded: ie.Predictions,
    uploaded_path: Path,
    *,
    seed: int | None,
    details: T.Mapping[str, T.Any],
) -> NewSubmission:
    """Checks the contents of an uploaded prediction file, read from `uploaded_path`, and makes
    its new submission. The dataset must be configured, and `DatasetContext.check_new_predictions`
    must accept the contents.

    If the seed is replaced, the file is rewritten to `_get_rewritten_path(uploaded_path)`, and
    the new submission is of that file.

    Args:
        details: The details of the upload in the action log, without the notes and seeds.

    Raises:
        irap_evaluation.PredictionFormatError: See `DatasetContext.check_new_predictions`.
        ValueError: For a dataset that is not configured, and see
            `DatasetContext.check_new_predictions`.
    """
    header = uploaded.header
    model = _override_seed(header.model, seed)
    if header.dataset not in dataset_contexts:
        raise ValueError(f"The predictions are of the dataset {header.dataset!r}, which is not"
                         f" configured. The configured datasets: {', '.join(dataset_contexts)}.")
    notes = dataset_contexts[header.dataset].check_new_predictions(uploaded)
    if model.seed == header.model.seed:
        return NewSubmission.from_predictions(uploaded, uploaded_path, details=details,
                                              notes=notes)
    # After the checks, since it validates the predictions again.
    predictions = _replace_model(uploaded, model)
    path = _get_rewritten_path(uploaded_path)
    ie.write_predictions(path, predictions)
    details = {**details, "original_seed": header.model.seed,
               "uploaded_file_sha256": compute_file_sha256(uploaded_path)}
    return NewSubmission.from_predictions(predictions, path, details=details, notes=notes)


def _is_macos_metadata(info: zipfile.ZipInfo) -> bool:
    """Returns whether a member is metadata that macOS adds to archives (AppleDouble files)."""
    return (info.filename.startswith("__MACOSX/")
            or PurePosixPath(info.filename).name.startswith("._"))


def _list_archive_members(zip_file: zipfile.ZipFile, file_name: str) -> list[zipfile.ZipInfo]:
    members = [info for info in zip_file.infolist()
               if not info.is_dir() and not _is_macos_metadata(info)]
    if not members:
        raise ValueError(f"The archive {file_name} has no files.")
    if others := [m.filename for m in members if not m.filename.lower().endswith(".parquet")]:
        raise ValueError(f"The archive {file_name} may hold only prediction files (.parquet)."
                         f" Other files: {', '.join(others)}.")
    return members


def _check_same_model(new: NewSubmission, first: NewSubmission, first_member_name: str) -> None:
    """Checks that two files of an archive are of one model, with the same `MODEL_FILE_FIELDS`.

    Raises:
        ValueError: If they are not.
    """
    def identify(s: NewSubmission) -> str:
        return f"{s.header.model.label} on {s.header.dataset}"

    if identify(new) != identify(first):
        raise ValueError(f"It is of the model {identify(new)}, but {first_member_name} is of"
                         f" {identify(first)}. An archive holds the files of one model.")
    if differing := get_differing_model_fields([new, first]):
        raise ValueError(f"It has the {describe_model_fields(new, differing)}, but"
                         f" {first_member_name} has the {describe_model_fields(first, differing)}."
                         f" The files of a model must have the same output kind, context offsets,"
                         f" and training and early stopping splits.")


#: The errors of reading a member of an archive that cannot be extracted, e.g. one that is
#: encrypted, corrupt, or compressed with an unsupported method.
_EXTRACTION_ERRORS = (zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error, EOFError)


def _open_archive(uploaded_path: Path, file_name: str) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(uploaded_path)
    except zipfile.BadZipFile as e:
        raise ValueError(f"{file_name} is not a .zip archive: {e}") from e


def _extract_archive_member(zip_file: zipfile.ZipFile, member: zipfile.ZipInfo, path: Path,
                            max_num_bytes: int) -> int:
    """Extracts a member to `path`. Returns its size.

    Raises:
        ValueError: If the member cannot be extracted, e.g. it is encrypted, corrupt, or
            compressed with an unsupported method, or it has more than `max_num_bytes`, e.g.
            a zip bomb.
    """
    num_bytes = 0
    try:
        with zip_file.open(member) as source, path.open("wb") as target:
            while chunk := source.read(_COPY_CHUNK_NUM_BYTES):
                num_bytes += len(chunk)
                if num_bytes > max_num_bytes:
                    raise ValueError(f"The files of the archive have more than"
                                     f" {MAX_UPLOAD_SIZE_TEXT}.")
                target.write(chunk)
    except _EXTRACTION_ERRORS as e:
        raise ValueError(f"It cannot be extracted: {e}") from e
    return num_bytes


def _prepare_archive(archive: ModelArchive, dataset_contexts: T.Mapping[str, DatasetContext],
                     uploaded_path: Path, *, file_name: str, seed: int | None,
                     new_paths: list[Path]) -> list[NewSubmission]:
    """Extracts and checks the files of an uploaded .zip archive (see `plan_upload`).

    Args:
        new_paths: Receives the paths of the extracted and rewritten files, which the caller
            removes.
    """
    prepared: list[NewSubmission] = []
    num_extracted_bytes = 0
    with _open_archive(uploaded_path, file_name) as zip_file:
        split_to_member_name: dict[str, str] = {}
        for member in _list_archive_members(zip_file, file_name):
            path = archive.make_upload_path(member.filename)
            new_paths += [path, _get_rewritten_path(path)]
            try:
                num_extracted_bytes += _extract_archive_member(
                    zip_file, member, path, MAX_UPLOAD_NUM_BYTES - num_extracted_bytes)
                uploaded = ie.read_predictions(path)
                split = uploaded.header.split
                if split in split_to_member_name:
                    raise ValueError(f"It is of the split {split!r}, like"
                                     f" {split_to_member_name[split]}. An archive holds one file"
                                     f" per split.")
                split_to_member_name[split] = member.filename
                new = _prepare_file(
                    dataset_contexts, uploaded, path, seed=seed,
                    details={"file_name": file_name, "archive_member": member.filename})
                if prepared:
                    _check_same_model(new, prepared[0], prepared[0].details["archive_member"])
                prepared.append(new)
            except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
                raise ValueError(f"{member.filename}: {e}") from e
    return prepared


def read_upload_headers(uploaded_path: Path, *, file_name: str) -> list[ie.PredictionHeader]:
    """Reads the headers of an uploaded prediction file or of the files of a .zip archive
    (`irap_evaluation.read_prediction_header`), without checking the files (see `plan_upload`),
    e.g. to show what the upload will set.

    Args:
        uploaded_path: At `archive.make_upload_path`.
        file_name: The name of the file on the uploader's computer, for messages.

    Raises:
        ValueError: If a header cannot be read (also irap_evaluation.PredictionFormatError), the
            archive is refused, or its files would have more than `MAX_UPLOAD_NUM_BYTES`. A
            message about a file of an archive starts with the file's name.
    """
    if not is_archive_upload(uploaded_path):
        return [ie.read_prediction_header(uploaded_path)]
    headers = []
    with _open_archive(uploaded_path, file_name) as zip_file:
        members = _list_archive_members(zip_file, file_name)
        # A member is read up to its size in the archive's directory, which bounds the reading.
        if sum(m.file_size for m in members) > MAX_UPLOAD_NUM_BYTES:
            raise ValueError(f"The files of the archive have more than {MAX_UPLOAD_SIZE_TEXT}.")
        for member in members:
            try:
                with zip_file.open(member) as file:
                    headers.append(ie.read_prediction_header(file))
            except _EXTRACTION_ERRORS as e:
                raise ValueError(f"{member.filename}: It cannot be extracted: {e}") from e
            except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
                raise ValueError(f"{member.filename}: {e}") from e
    return headers


@dc.dataclass(frozen=True)
class PlannedUpload:
    """An upload that is checked and planned (`plan_upload`), to be applied (`apply_upload`) or
    discarded (`discard`).

    Attributes:
        uploaded_path: The uploaded file or archive, which `discard` keeps.
        temporary_paths: The files that are extracted from an archive or rewritten with another
            seed, and that are not yet stored.
    """

    update: ModelUpdate
    uploaded_path: Path
    temporary_paths: tuple[Path, ...]

    @property
    def notes(self) -> list[str]:
        """The notes of the files (`archive.NewSubmission.notes`), with their splits."""
        return [f"{new.header.split}: {note}" for new in self.update.new_submissions
                for note in new.notes]

    def discard(self) -> None:
        """Removes the temporary files, but not the uploaded file, so that the upload can be
        tried again. The file of each new submission is the uploaded file or a temporary one."""
        for path in self.temporary_paths:
            path.unlink(missing_ok=True)


def plan_upload(
    archive: ModelArchive,
    dataset_contexts: T.Mapping[str, DatasetContext],
    uploaded_path: Path,
    *,
    file_name: str,
    seed: int | None = None,
    description: str | None = None,
    method_display_name: str | None = None,
) -> PlannedUpload:
    """Checks an uploaded prediction file or .zip archive and plans storing its files as
    submissions of their model (`ModelArchive.plan_model_update`).

    A file must be of a configured dataset, name splits of the dataset, predict attributes of its
    vocabulary with their iRAP codes, and be accepted by `irap_evaluation.select_evaluation_sets`.
    The files of an archive must also be of one model, with the same `archive.MODEL_FILE_FIELDS`,
    and of distinct splits, and all members but directories and macOS metadata must be .parquet
    files.

    Args:
        uploaded_path: At `archive.make_upload_path`. It is kept, also if the upload is refused.
        file_name: The name of the file on the uploader's computer, for the action log, which
            also records the name of each file in an archive.
        seed: If not None, it replaces the seed in each header (`irap_evaluation.ModelInfo.seed`),
            e.g. to add another model of a method, or to give a seed to files without one, which
            the archive requires (see `ModelArchive.plan_model_update`). The files are then
            rewritten, and the action log records the original seed and the hash of the
            uploaded file.
        description: The new description of the model, or None to keep it.
        method_display_name: The new display name of the method, or None to take it from the
            headers (see `ModelArchive.plan_model_update`). The files are not rewritten.

    Raises:
        irap_evaluation.PredictionFormatError: If an uploaded prediction file is not valid,
            names unknown splits, or its attributes or iRAP codes differ from the dataset's.
        ValueError: If a file or the archive is refused otherwise, or the update is refused (see
            `ModelArchive.plan_model_update`). A refused file of an archive, also an invalid
            one, raises a ValueError whose message starts with the file's name.
    """
    temporary_paths: list[Path] = []
    try:
        if is_archive_upload(uploaded_path):
            new_submissions = _prepare_archive(archive, dataset_contexts, uploaded_path,
                                               file_name=file_name, seed=seed,
                                               new_paths=temporary_paths)
        else:
            temporary_paths.append(_get_rewritten_path(uploaded_path))
            new_submissions = [_prepare_file(
                dataset_contexts, ie.read_predictions(uploaded_path), uploaded_path, seed=seed,
                details={"file_name": file_name})]
        update = archive.plan_model_update(new_submissions, action="upload",
                                           description=description,
                                           method_display_name=method_display_name)
    except BaseException:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    return PlannedUpload(update, uploaded_path, tuple(temporary_paths))


def apply_upload(archive: ModelArchive, planned: PlannedUpload, *,
                 submitter: str) -> list[Submission]:
    """Applies a planned upload (`ModelArchive.apply_model_update`) and removes its files that
    are not stored. If the upload is refused, the uploaded file is kept, so that the upload can
    be planned and tried again.

    Raises:
        ValueError: See `ModelArchive.apply_model_update`.
    """
    try:
        submissions = archive.apply_model_update(planned.update, submitter=submitter)
    finally:
        planned.discard()
    planned.uploaded_path.unlink(missing_ok=True)  # Moved into the archive, or not stored.
    return submissions
