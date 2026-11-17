"""The archive of models: their prediction files on disk, an SQLite index of them and an action
log.

A model (`Model`) is one model of a method on a dataset, e.g. one trained network, identified by
its dataset, method name and seed. Its submissions (`Submission`) are its prediction files, at most
one per split. A submission of a split that the model already has replaces the old one, which is
deleted. Deletion is permanent: deleting a submission removes its row and its file, and the score
cache then deletes its scores (`scoring.ScoreStore.delete_scorings_of_deleted_submissions`).
Deleting a model removes it with all its submissions, and deleting a method removes all its models
on a dataset. The last submission of a model is deleted only with the model, so each model has a
submission. The action log keeps the actions on deleted models and submissions. Each action
records who did it, the name of an account that can write. The archive does not check
permissions, its callers do (`accounts.can_write`).

A method on a dataset can have a display name, which the pages show instead of the method name.
An update sets it, given with the update or from the files that have one
(`irap_evaluation.ModelInfo.method_display_name`), and it can be edited.

The data directory holds:

- `archive.sqlite3`: the index of the models and submissions, the method display names, and the
  action log,
- `submissions/<id>/predictions.parquet`: the stored files, which are not modified until they are
  deleted,
- `uploads/`: files that are not stored yet: uploaded files and .zip archives, the files
  extracted from archives or rewritten with another seed, ensembles, and the .zip archives of
  downloads while they are sent.
"""

import dataclasses as dc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import time
import typing as T
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import irap_evaluation as ie

from .database import Migration, begin_write, connect, get_utc_now, initialize_schema

_logger = logging.getLogger(__name__)

#: The actions that add submissions: an upload, or an ensemble of stored models (see
#: `ensembles`).
AddingActionKind = T.Literal["upload", "ensemble"]
ADDING_ACTION_KINDS: tuple[str, ...] = T.get_args(AddingActionKind)
ActionKind = AddingActionKind | T.Literal[
    "delete", "delete_model", "edit_description", "edit_method_display_name"]

DATABASE_FILE_NAME = "archive.sqlite3"
#: The version of the database schema (`PRAGMA user_version`). A database of an older version is
#: migrated (`_MIGRATIONS`). Version 1 is the schema of commit c084f1a (2026-10-12), the first in
#: production.
SCHEMA_VERSION = 3


def _drop_score_cache(connection: sqlite3.Connection) -> None:
    """Migrates version 1 to 2: drops the score cache, which is now in its own database
    (`scoring.ScoreStore`), and its trigger, which would make deletions fail."""
    for statement in ["DROP TRIGGER IF EXISTS delete_submission_results",
                      "DROP TABLE IF EXISTS result_submissions",
                      "DROP TABLE IF EXISTS result_cache",
                      "DROP TABLE IF EXISTS submission_scores",
                      "DROP TABLE IF EXISTS submission_scorings",
                      "DROP TABLE IF EXISTS score_cache_version"]:
        connection.execute(statement)


def _replace_intersect_segments(connection: sqlite3.Connection) -> None:
    """Migrates version 2 to 3: replaces 'intersect_segments' in the details of each ensemble
    action by 'segment_combination' and 'attribute_combination' (`ensembles.prepare_ensemble`).
    Ensembles of version 2 required the same attributes.

    Raises:
        KeyError: If an ensemble action has no 'intersect_segments'.
    """
    rows = connection.execute("SELECT id, details FROM actions WHERE action = 'ensemble'")
    for action_id, details_json in rows.fetchall():
        old_details = json.loads(details_json)
        if "intersect_segments" not in old_details:
            raise KeyError(f"The ensemble action {action_id} has no 'intersect_segments'.")
        combinations = {
            "segment_combination": "intersection" if old_details["intersect_segments"] else "same",
            "attribute_combination": "same"}
        # The combinations take the place of 'intersect_segments' in the action log.
        details = {}
        for key, value in old_details.items():
            details.update(combinations if key == "intersect_segments" else {key: value})
        connection.execute("UPDATE actions SET details = ? WHERE id = ?",
                           (json.dumps(details), action_id))


#: Version v -> the migration from v to v + 1 (`database.initialize_schema`).
_MIGRATIONS: dict[int, Migration] = {1: _drop_score_cache, 2: _replace_intersect_segments}

#: SQLite stores integers as signed 64-bit values.
_SQLITE_INTEGER_RANGE = range(-2**63, 2**63)

PREDICTIONS_SUFFIX = ".predictions.parquet"
ARCHIVE_SUFFIX = ".zip"
#: The names of `ModelArchive.make_upload_path`.
_UPLOAD_NAME_PATTERN = re.compile(
    rf"[0-9a-f]{{32}}({re.escape(PREDICTIONS_SUFFIX)}|{re.escape(ARCHIVE_SUFFIX)})")

#: AUTOINCREMENT ids of models and submissions are not reused after a deletion, so that the
#: action log, the details of ensembles and old links do not refer to new ones.
_SCHEMA = """
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    method_name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX model_identity ON models (dataset, method_name, seed);
-- The methods with a display name.
CREATE TABLE methods (
    dataset TEXT NOT NULL,
    method_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    PRIMARY KEY (dataset, method_name)
);
CREATE TABLE submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL REFERENCES models (id),
    split TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    output_kind TEXT NOT NULL,
    context_offsets TEXT,
    training_splits TEXT,
    early_stopping_splits TEXT NOT NULL,
    num_segments INTEGER NOT NULL,
    num_attributes INTEGER NOT NULL,
    submitter TEXT NOT NULL,
    uploaded_at TEXT NOT NULL
);
CREATE UNIQUE INDEX submission_split ON submissions (model_id, split);
-- Without foreign keys, since the actions outlive deleted models and submissions.
CREATE TABLE actions (
    id INTEGER PRIMARY KEY,
    time TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    model_id INTEGER NOT NULL,
    model_label TEXT NOT NULL,
    submission_id INTEGER,
    details TEXT NOT NULL
);
"""

_METHOD_JOIN = ("LEFT JOIN methods d ON d.dataset = m.dataset"
                " AND d.method_name = m.method_name")
_MODEL_QUERY = f"SELECT m.*, d.display_name AS method_display_name FROM models m {_METHOD_JOIN}"
_SUBMISSION_QUERY = ("SELECT s.*, m.dataset, m.method_name, m.seed, m.description, m.created_at,"
                     " d.display_name AS method_display_name"
                     f" FROM submissions s JOIN models m ON m.id = s.model_id {_METHOD_JOIN}")


@dc.dataclass(frozen=True)
class Model:
    """A stored model: one model of a method on a dataset (see the module docstring).

    Attributes:
        method_display_name: The display name of the method on the dataset, or None.
        description: Markdown about the model, e.g. how it was trained, whose first line
            summarizes it (`get_description_summary`).
        created_at: The time of its first submission, in UTC.
    """

    id: int
    dataset: str
    method_name: str
    method_display_name: str | None
    seed: int
    description: str
    created_at: datetime

    @property
    def label(self) -> str:
        """Identifies the model, e.g. in messages and file names."""
        return ie.format_model_label(self.method_name, self.seed)

    @property
    def shown_method_name(self) -> str:
        """The method display name, or the method name if there is none."""
        return self.method_display_name or self.method_name

    @property
    def shown_label(self) -> str:
        """The label shown on the pages: `label` with the method display name."""
        return ie.format_model_label(self.shown_method_name, self.seed)

    @property
    def seed_label(self) -> str:
        """Identifies the model among the models of its method, e.g. 'seed3'."""
        return f"seed{self.seed}"


@dc.dataclass(frozen=True)
class Submission:
    """A stored prediction file of a model on one split.

    The header fields are copies of the header of the file, so that listing does not read the
    files.

    Attributes:
        training_splits, early_stopping_splits: See `irap_evaluation.ModelInfo`.
        submitter: The account name of the writer who uploaded it.
        uploaded_at: The time of the upload, in UTC.
    """

    id: int
    model: Model
    split: str
    file_sha256: str
    output_kind: ie.OutputKind
    context_offsets: tuple[int, ...] | None
    training_splits: tuple[str, ...] | None
    early_stopping_splits: tuple[str, ...]
    num_segments: int
    num_attributes: int
    submitter: str
    uploaded_at: datetime

    @property
    def dataset(self) -> str:
        return self.model.dataset

    @property
    def label(self) -> str:
        return self.model.label

    @property
    def shown_label(self) -> str:
        return self.model.shown_label

    @property
    def model_info(self) -> ie.ModelInfo:
        """The model of the header of the file, without its details."""
        return ie.ModelInfo(method_name=self.model.method_name,
                            training_splits=self.training_splits,
                            early_stopping_splits=self.early_stopping_splits, seed=self.model.seed)


def list_submitters(submissions: T.Iterable[Submission]) -> list[str]:
    """Lists the distinct submitters of submissions, in the order of their first upload."""
    return list(dict.fromkeys(
        s.submitter for s in sorted(submissions, key=lambda s: s.uploaded_at)))


def get_method_description(models: T.Iterable[Model]) -> str:
    """Returns the description of the model of a method with the lowest seed, or '' if there are
    no models. `ModelArchive.set_method_description` gives the models of a method the same one."""
    lowest_seed_model = min(models, key=lambda m: m.seed, default=None)
    return "" if lowest_seed_model is None else lowest_seed_model.description


def get_description_summary(description: str) -> str:
    """Returns the first line of a description, which the tables show instead of the whole
    Markdown text."""
    return description.partition("\n")[0].rstrip()


def describe_description_change(label: str, old_description: str,
                                description: str) -> str | None:
    """Describes a change of the description of a model for the user to confirm, or returns None
    if it replaces no other description. A description of several lines is shown by its first
    line and '…'."""
    if not old_description or description == old_description:
        return None

    def shorten(text: str) -> str:
        summary = get_description_summary(text)
        return summary if summary == text else f"{summary} …"

    return (f"Changes the description of {label} from {shorten(old_description)!r} to"
            f" {shorten(description)!r}.")


@dc.dataclass(frozen=True)
class ActionLogEntry:
    """An action on the archive.

    Attributes:
        time: In UTC.
        actor: The account name of the writer who did it.
        model_id: The model that it is on, which may have been deleted.
        model_label: The `Model.label` of the model, e.g. for a deleted one.
        submission_id: The submission that it is on, which may have been deleted, None for an
            action on the model.
        details: JSON-compatible details, e.g. the name of an uploaded file.
    """

    id: int
    time: datetime
    actor: str
    action: ActionKind
    model_id: int
    model_label: str
    submission_id: int | None
    details: T.Mapping[str, T.Any]


@dc.dataclass(frozen=True)
class NewSubmission:
    """A checked prediction file to store with `ModelArchive.apply_model_update`.

    It holds only what the index needs, so that the predictions of several files need not be in
    memory together.

    Attributes:
        file_path: The file, in `ModelArchive.uploads_dir`.
        details: The details of the action that adds it in the action log, e.g. the name of the
            uploaded file.
        notes: The notes of `DatasetContext.check_new_predictions`, which the action log
            records with the details (as 'notes').
    """

    header: ie.PredictionHeader
    output_kind: ie.OutputKind
    num_segments: int
    num_attributes: int
    file_path: Path
    details: T.Mapping[str, T.Any]
    notes: tuple[str, ...]

    @classmethod
    def from_predictions(cls, predictions: ie.Predictions, file_path: Path, *,
                         details: T.Mapping[str, T.Any], notes: T.Sequence[str]) -> T.Self:
        """Args:
            predictions: The contents of the file, e.g. from `irap_evaluation.read_predictions`.
        """
        return cls(header=predictions.header, output_kind=predictions.output_kind,
                   num_segments=predictions.num_segments,
                   num_attributes=len(predictions.attributes), file_path=file_path,
                   details=details, notes=tuple(notes))

    @property
    def split(self) -> str:
        return self.header.split

    @property
    def context_offsets(self) -> tuple[int, ...] | None:
        return self.header.context_offsets

    @property
    def training_splits(self) -> tuple[str, ...] | None:
        return self.header.model.training_splits

    @property
    def early_stopping_splits(self) -> tuple[str, ...]:
        return self.header.model.early_stopping_splits


#: The fields that the files of a model share: name -> the getter of a `Submission` or a
#: `NewSubmission`.
MODEL_FILE_FIELDS: T.Mapping[str, T.Callable[[Submission | NewSubmission], T.Any]] = {
    "output kind": lambda s: s.output_kind,
    "context offsets": lambda s: s.context_offsets,
    "training splits": lambda s: s.training_splits,
    "early stopping splits": lambda s: s.early_stopping_splits,
}


def get_differing_model_fields(files: T.Iterable[Submission | NewSubmission]) -> list[str]:
    """Returns the names of the `MODEL_FILE_FIELDS` in which the files differ."""
    files = list(files)
    return [name for name, get in MODEL_FILE_FIELDS.items() if len({get(f) for f in files}) > 1]


def get_download_file_name(submission: Submission) -> str:
    return ie.make_prediction_file_name(submission.model.method_name, submission.model.seed,
                                        submission.split)


def make_label_file_name(label: str, suffix: str, *,
                         max_bytes: int = ie.MAX_FILE_NAME_BYTES) -> str:
    """Makes the name of a downloaded file from a label, e.g. of a model, with a long label cut
    like in `ie.make_prediction_file_name`, so that `suffix` is kept whole."""
    return ie.shorten_file_name(ie.to_valid_file_name(label), ie.to_valid_file_name(suffix),
                                max_bytes=max_bytes)


def describe_model_fields(file: Submission | NewSubmission, names: T.Iterable[str]) -> str:
    """Formats the values of some `MODEL_FILE_FIELDS` of a file, e.g. those that differ."""
    return ", ".join(f"{name} {MODEL_FILE_FIELDS[name](file)}" for name in names)


@dc.dataclass(frozen=True)
class ModelUpdate:
    """New submissions of one model, e.g. of an upload or an ensemble, as planned on the state of
    the archive (`ModelArchive.plan_model_update`).

    `ModelArchive.apply_model_update` applies it if the archive has not changed since.

    Attributes:
        new_submissions: Of distinct splits and one model, the model of their headers, which has
            a seed.
        action: The action of each new submission in the action log.
        description: The new description of the model, or None to keep it.
        method_display_name: The new display name of the method, given with the update or that
            of the new submissions, or None to keep the method's.
        model: The model as planned, or None if the update creates it.
        old_method_display_name: The display name of the method as planned.
        replaced: Split -> the submission that a new one replaces, which is deleted.
        replaces_all_files: Whether the new submissions replace all files of the model, e.g.
            those of an ensemble, whose files are made of the same members.
        replaced_other_splits: Split -> a submission of another split, which the update deletes
            if it replaces all files of the model.
        source_submissions: The submissions that the new ones are made of, e.g. the members of
            an ensemble, which must still exist when the update is applied.
    """

    new_submissions: tuple[NewSubmission, ...]
    action: AddingActionKind
    description: str | None
    method_display_name: str | None
    model: Model | None
    old_method_display_name: str | None
    replaced: T.Mapping[str, Submission]
    replaces_all_files: bool
    replaced_other_splits: T.Mapping[str, Submission]
    source_submissions: tuple[Submission, ...]

    @property
    def dataset(self) -> str:
        return self.new_submissions[0].header.dataset

    @property
    def method_name(self) -> str:
        return self.new_submissions[0].header.model.method_name

    @property
    def seed(self) -> int:
        return self.new_submissions[0].header.model.seed

    @property
    def label(self) -> str:
        return ie.format_model_label(self.method_name, self.seed)

    @property
    def confirmations(self) -> list[str]:
        """The changes that the user confirms before the update is applied: replaced files, which
        are deleted, and a description or a method display name that replaces another one."""
        def describe(submission: Submission) -> str:
            return (f"file #{submission.id} of {self.label}, uploaded"
                    f" {submission.uploaded_at:%Y-%m-%d %H:%M} UTC by {submission.submitter},"
                    f" with its scores")

        copy = "To keep a copy, download it first."
        items = []
        for split, submission in self.replaced.items():
            items.append(f"Replaces and permanently deletes the {split} {describe(submission)}."
                         f" {copy}")
        for split, submission in self.replaced_other_splits.items():
            items.append(f"Permanently deletes the {split} {describe(submission)}, since the new"
                         f" files replace all files of the model. {copy}")
        if (self.model is not None and self.description is not None
                and (change := describe_description_change(self.label, self.model.description,
                                                           self.description)) is not None):
            items.append(change)
        if (self.old_method_display_name is not None and self.method_display_name is not None
                and self.method_display_name != self.old_method_display_name):
            items.append(f"Changes the display name of the method {self.method_name!r} on"
                         f" {self.dataset} from {self.old_method_display_name!r} to"
                         f" {self.method_display_name!r}.")
        return items


def _get_method_display_name(headers: T.Iterable[ie.PredictionHeader]) -> str | None:
    """Returns the method display name of the headers that have one, None if none has.

    Raises:
        ValueError: If they have different ones.
    """
    names = {n for h in headers if (n := h.model.method_display_name) is not None}
    if len(names) > 1:
        raise ValueError(f"The files have different method display names: {sorted(names)}.")
    return next(iter(names), None)


def discard_model_update(update: ModelUpdate) -> None:
    """Removes the files of the new submissions of an update that is not applied, or those that
    remain after it is applied (the others are moved into the archive)."""
    for new in update.new_submissions:
        new.file_path.unlink(missing_ok=True)


class _DryRunFinished(Exception):
    """Ends the transaction of a dry run of `ModelArchive._apply`, which rolls it back."""


def compute_file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _to_optional_json(values: T.Sequence | None) -> str | None:
    return None if values is None else json.dumps(list(values))


def _from_optional_json(text: str | None) -> tuple | None:
    return None if text is None else tuple(json.loads(text))


def _to_model(row: sqlite3.Row, id_column: str = "id") -> Model:
    return Model(id=row[id_column], dataset=row["dataset"], method_name=row["method_name"],
                 method_display_name=row["method_display_name"], seed=row["seed"],
                 description=row["description"],
                 created_at=datetime.fromisoformat(row["created_at"]))


def _to_submission(row: sqlite3.Row) -> Submission:
    """Converts a row of `_SUBMISSION_QUERY`."""
    model = _to_model(row, id_column="model_id")
    return Submission(
        id=row["id"], model=model, split=row["split"],
        file_sha256=row["file_sha256"], output_kind=row["output_kind"],
        context_offsets=_from_optional_json(row["context_offsets"]),
        training_splits=_from_optional_json(row["training_splits"]),
        early_stopping_splits=tuple(json.loads(row["early_stopping_splits"])),
        num_segments=row["num_segments"], num_attributes=row["num_attributes"],
        submitter=row["submitter"], uploaded_at=datetime.fromisoformat(row["uploaded_at"]))


def _to_action_log_entry(row: sqlite3.Row) -> ActionLogEntry:
    return ActionLogEntry(id=row["id"], time=datetime.fromisoformat(row["time"]),
                          actor=row["actor"], action=row["action"], model_id=row["model_id"],
                          model_label=row["model_label"], submission_id=row["submission_id"],
                          details=json.loads(row["details"]))


def _check_actor(actor: str) -> str:
    """Strips the account name of who does an action.

    Raises:
        ValueError: If it is empty.
    """
    if not actor.strip():
        raise ValueError("The actor of an action must not be empty.")
    return actor.strip()


def _check_seed(seed: int | None) -> int:
    """Checks the seed of a model in its file headers. Unlike `irap_evaluation`, the archive
    requires a seed also for the only model of a method, so that a method never has a model
    without a seed next to models with one (`irap_evaluation.check_method_models`), which could be
    given a seed only by storing its files again.

    Raises:
        ValueError: If `seed` is None or does not fit SQLite's INTEGER.
    """
    if seed is None:
        raise ValueError("The model has no seed. Enter one, e.g. 0 for the first model of the"
                         " method.")
    if seed not in _SQLITE_INTEGER_RANGE:
        raise ValueError(f"The seed {seed} does not fit a signed 64-bit integer.")
    return seed


def _get_model(connection: sqlite3.Connection, model_id: int) -> Model:
    row = connection.execute(f"{_MODEL_QUERY} WHERE m.id = ?", (model_id,)).fetchone()
    if row is None:
        raise LookupError(f"There is no model #{model_id}.")
    return _to_model(row)


def _find_model(connection: sqlite3.Connection, dataset: str, method_name: str,
                seed: int) -> Model | None:
    row = connection.execute(
        f"{_MODEL_QUERY} WHERE m.dataset = ? AND m.method_name = ? AND m.seed = ?",
        (dataset, method_name, seed)).fetchone()
    return None if row is None else _to_model(row)


def _get_method_models(connection: sqlite3.Connection, dataset: str,
                       method_name: str) -> list[Model]:
    rows = connection.execute(
        f"{_MODEL_QUERY} WHERE m.dataset = ? AND m.method_name = ? ORDER BY m.id",
        (dataset, method_name))
    if not (models := [_to_model(row) for row in rows]):
        raise LookupError(f"The method {method_name!r} has no models on {dataset}.")
    return models


def _get_submission(connection: sqlite3.Connection, submission_id: int) -> Submission:
    row = connection.execute(f"{_SUBMISSION_QUERY} WHERE s.id = ?", (submission_id,)).fetchone()
    if row is None:
        raise LookupError(f"There is no file #{submission_id}.")
    return _to_submission(row)


def _list_submissions(connection: sqlite3.Connection, where: str | None = None,
                      parameters: T.Sequence[T.Any] = ()) -> list[Submission]:
    """Lists the submissions, newest first, optionally only those that match an SQL condition."""
    where_clause = "" if where is None else f" WHERE {where}"
    rows = connection.execute(f"{_SUBMISSION_QUERY}{where_clause} ORDER BY s.id DESC",
                              parameters)
    return [_to_submission(row) for row in rows]


def _list_model_submissions(connection: sqlite3.Connection, model_id: int) -> list[Submission]:
    return _list_submissions(connection, "s.model_id = ?", (model_id,))


def _list_method_submissions(connection: sqlite3.Connection, dataset: str,
                             method_name: str) -> list[Submission]:
    return _list_submissions(connection, "m.dataset = ? AND m.method_name = ?",
                             (dataset, method_name))


def _log_action(connection: sqlite3.Connection, actor: str, action: ActionKind, model: Model,
                submission_id: int | None, details: T.Mapping[str, T.Any]) -> None:
    connection.execute(
        "INSERT INTO actions (time, actor, action, model_id, model_label, submission_id, details)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (get_utc_now().isoformat(), actor, action, model.id, model.label, submission_id,
         json.dumps(details)))


def _delete_submission_row(connection: sqlite3.Connection, submission: Submission, actor: str,
                           details: T.Mapping[str, T.Any]) -> None:
    """Deletes a submission from the index and logs it."""
    connection.execute("DELETE FROM submissions WHERE id = ?", (submission.id,))
    _log_action(connection, actor, "delete", submission.model, submission.id,
                {"split": submission.split, **details})


def _delete_model_rows(connection: sqlite3.Connection, model: Model, actor: str,
                       details: T.Mapping[str, T.Any]) -> list[Submission]:
    """Deletes a model and its submissions from the index, and the display name of its method if
    no other model of the method on the dataset remains, and logs it. Returns the deleted
    submissions."""
    submissions = _list_model_submissions(connection, model.id)
    for submission in submissions:
        _delete_submission_row(connection, submission, actor, {"by": "delete_model"})
    connection.execute("DELETE FROM models WHERE id = ?", (model.id,))
    connection.execute(
        "DELETE FROM methods WHERE dataset = ? AND method_name = ? AND NOT EXISTS"
        " (SELECT 1 FROM models WHERE dataset = ? AND method_name = ?)",
        (model.dataset, model.method_name, model.dataset, model.method_name))
    _log_action(connection, actor, "delete_model", model, None, details)
    return submissions


def _set_description(connection: sqlite3.Connection, model: Model, description: str,
                     actor: str) -> None:
    """Changes the description of a model, if it differs, and logs it."""
    if description != model.description:
        connection.execute("UPDATE models SET description = ? WHERE id = ?",
                           (description, model.id))
        _log_action(connection, actor, "edit_description", model, None,
                    {"old_description": model.description, "description": description})


def _read_method_display_name(connection: sqlite3.Connection, dataset: str,
                              method_name: str) -> str | None:
    row = connection.execute(
        "SELECT display_name FROM methods WHERE dataset = ? AND method_name = ?",
        (dataset, method_name)).fetchone()
    return None if row is None else row["display_name"]


def _set_method_display_name(connection: sqlite3.Connection, model: Model,
                             display_name: str | None, actor: str,
                             details: T.Mapping[str, T.Any]) -> None:
    """Sets or, for None, clears the display name of the method of a model, if it differs, and
    logs it on the model.

    Args:
        details: More details of the action, e.g. the action that it is part of ('by').
    """
    if display_name == model.method_display_name:
        return
    if display_name is None:
        connection.execute("DELETE FROM methods WHERE dataset = ? AND method_name = ?",
                           (model.dataset, model.method_name))
    else:
        connection.execute(
            "INSERT INTO methods (dataset, method_name, display_name) VALUES (?, ?, ?)"
            " ON CONFLICT (dataset, method_name)"
            " DO UPDATE SET display_name = excluded.display_name",
            (model.dataset, model.method_name, display_name))
    _log_action(connection, actor, "edit_method_display_name", model, None,
                {"old_display_name": model.method_display_name, "display_name": display_name,
                 **details})


def _delete_replaced_submissions(connection: sqlite3.Connection, model: Model,
                                 update: ModelUpdate, actor: str) -> None:
    """Deletes the submissions that an update replaces from the index, as planned
    (`ModelUpdate.replaced` and `ModelUpdate.replaced_other_splits`), and logs it.

    Raises:
        ValueError: If the submissions of the model have changed since the update was planned,
            e.g. by another user.
    """
    new_splits = {new.split for new in update.new_submissions}
    replaced = [s for s in _list_model_submissions(connection, model.id)
                if update.replaces_all_files or s.split in new_splits]
    planned = {**update.replaced, **update.replaced_other_splits}
    if {s.split: s.id for s in replaced} != {split: s.id for split, s in planned.items()}:
        raise ValueError(f"The files of {model.label} have changed, e.g. by another user."
                         f" Try again.")
    for submission in replaced:
        _delete_submission_row(connection, submission, actor, {"by": update.action})


def _check_model_invariant(connection: sqlite3.Connection, model_id: int) -> None:
    """Checks that the submissions of a model have the same `MODEL_FILE_FIELDS`.

    Raises:
        ValueError: If they differ.
    """
    submissions = _list_model_submissions(connection, model_id)
    if differing := get_differing_model_fields(submissions):
        # Without the submission ids, which a dry run (`ModelArchive._apply`) rolls back.
        values = "; ".join(f"{s.split}: {describe_model_fields(s, differing)}"
                           for s in sorted(submissions, key=lambda s: s.split))
        raise ValueError(f"The files of the splits of {submissions[0].label} would differ in"
                         f" {', '.join(differing)}: {values}. The files of a model must have the"
                         f" same output kind, context offsets, and training and early stopping"
                         f" splits. To change them, replace all its files together, e.g. with a"
                         f" .zip archive, or delete the other files first.")


def _check_method_invariant(connection: sqlite3.Connection, dataset: str,
                            method_name: str) -> None:
    """Checks that the models of a method can be averaged (`irap_evaluation.check_method_models`).
    Each model has a submission, which gives its splits.

    Raises:
        ValueError: If they cannot.
    """
    submissions = _list_method_submissions(connection, dataset, method_name)
    model_id_to_info = {s.model.id: s.model_info for s in submissions}  # Equal per model.
    try:
        ie.check_method_models(model_id_to_info.values())
    except ValueError as e:
        # Distinct per model (`models.model_identity`). Without the model ids, which a dry run
        # (`ModelArchive._apply`) rolls back.
        seeds = ", ".join(f"seed {seed}" for seed in sorted({s.model.seed for s in submissions}))
        raise ValueError(f"{e} Models on {dataset}: {seeds}.") from e


def _check_sources_exist(connection: sqlite3.Connection,
                         sources: T.Iterable[Submission]) -> None:
    """Checks that the submissions that new ones are made of still exist
    (`ModelUpdate.source_submissions`).

    Raises:
        ValueError: If one does not.
    """
    for source in sources:
        if connection.execute("SELECT 1 FROM submissions WHERE id = ?",
                              (source.id,)).fetchone() is None:
            raise ValueError(f"The {source.split} file #{source.id} of {source.label}, which the"
                             f" new files are made of, has been replaced or deleted, e.g. by"
                             f" another user. Try again.")


class ModelArchive:
    """The archive in a data directory (see the module docstring), created on first use.

    Each operation opens its own database connection, so the archive can be used from several
    threads.

    A database of an older schema version is migrated (`database.initialize_schema`).

    Raises:
        ValueError: If the database cannot be migrated to `SCHEMA_VERSION`, or its tables differ
            from those of its version (see `database.initialize_schema`).
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(exist_ok=True)
        initialize_schema(self.database_path, _SCHEMA, SCHEMA_VERSION, "archive", _MIGRATIONS)

    @property
    def database_path(self) -> Path:
        return self.data_dir / DATABASE_FILE_NAME

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def submissions_dir(self) -> Path:
        return self.data_dir / "submissions"

    def get_predictions_path(self, submission_id: int) -> Path:
        return self.submissions_dir / str(submission_id) / "predictions.parquet"

    def _list_deleted_submission_dirs(self, connection: sqlite3.Connection) -> list[Path]:
        """Lists the directories in `submissions_dir` of the submissions that are deleted as
        `connection` sees the database, e.g. in the transaction of a deletion: those deleted in
        it and those whose removal failed before (`_remove_submission_dirs`).

        An id above the last one given out (`sqlite_sequence`) is skipped. Its directory has a
        file of an update that is being stored, or being moved back after its transaction
        failed (`_apply`), or that a crash left before COMMIT, which the next submission with
        that id overwrites.
        """
        row = connection.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'submissions'").fetchone()
        if row is None:
            return []
        ids = {r[0] for r in connection.execute("SELECT id FROM submissions")}
        return [p for p in self.submissions_dir.iterdir()
                if p.name.isdigit() and int(p.name) <= row[0] and int(p.name) not in ids]

    @staticmethod
    def _remove_submission_dirs(paths: T.Iterable[Path]) -> None:
        """Removes the directories of deleted submissions, after the deletion is committed. A
        directory that cannot be removed, e.g. because its file is being read on Windows, is
        logged and removed by a later deletion."""
        for path in paths:
            try:
                shutil.rmtree(path)
            except OSError:  # The deletion is committed, so it is not refused.
                _logger.warning("Removing the files of the deleted submission #%s failed.",
                                path.name, exc_info=True)

    # Uploads ######################################################################################

    def make_upload_path(self, file_name: str) -> Path:
        """Makes a new path in `uploads_dir` for an uploaded file: a .zip archive if `file_name`
        has that suffix, otherwise a prediction file.

        Args:
            file_name: The name of the file on the uploader's computer.
        """
        suffix = (ARCHIVE_SUFFIX if file_name.lower().endswith(ARCHIVE_SUFFIX)
                  else PREDICTIONS_SUFFIX)
        return self.uploads_dir / f"{uuid.uuid4().hex}{suffix}"

    def make_download_path(self, suffix: str) -> Path:
        """Makes a new path in `uploads_dir` for a file that is prepared for a download, e.g. a
        coding table, which `remove_uploads` also removes. `get_upload_path` does not accept its
        name, so that a browser cannot pass it as an upload.

        Args:
            suffix: E.g. '.xlsx'.
        """
        return self.uploads_dir / f"download-{uuid.uuid4().hex}{suffix}"

    def get_upload_path(self, upload_name: str) -> Path | None:
        """Returns the path of an upload by the name of its file, None if `make_upload_path` does
        not make such names, e.g. if a browser sent a name that points outside `uploads_dir`."""
        if _UPLOAD_NAME_PATTERN.fullmatch(upload_name) is None:
            return None
        return self.uploads_dir / upload_name

    def remove_uploads(self, *, older_than_s: float | None = None) -> int:
        """Removes the files in `uploads_dir`, e.g. those left when the server stopped, or those
        of abandoned uploads.

        Args:
            older_than_s: If not None, only the files last modified more than this many seconds
                ago are removed.

        Returns:
            The number of removed files.
        """
        max_mtime_s = math.inf if older_than_s is None else time.time() - older_than_s
        num_removed = 0
        for path in self.uploads_dir.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < max_mtime_s:
                    path.unlink()
                    num_removed += 1
            except FileNotFoundError:  # Removed by a concurrent call.
                pass
        return num_removed

    # Reading ######################################################################################

    def list_models(self) -> list[Model]:
        """Lists the models, newest first."""
        with connect(self.database_path) as connection:
            rows = connection.execute(f"{_MODEL_QUERY} ORDER BY m.id DESC")
            return [_to_model(row) for row in rows]

    def get_model(self, model_id: int) -> Model:
        """
        Raises:
            LookupError: If there is no such model, e.g. since it has been deleted.
        """
        with connect(self.database_path) as connection:
            return _get_model(connection, model_id)

    def list_submissions(self) -> list[Submission]:
        """Lists the submissions of all models, newest first."""
        with connect(self.database_path) as connection:
            return _list_submissions(connection)

    def list_model_submissions(self, model_id: int) -> list[Submission]:
        """Lists the submissions of a model, newest first."""
        with connect(self.database_path) as connection:
            return _list_model_submissions(connection, model_id)

    def get_submission(self, submission_id: int) -> Submission:
        """
        Raises:
            LookupError: If there is no such submission, e.g. since it has been deleted.
        """
        with connect(self.database_path) as connection:
            return _get_submission(connection, submission_id)

    def get_method_models(self, dataset: str, method_name: str) -> list[Model]:
        """Returns the models of a method on a dataset, by id.

        Raises:
            LookupError: If the method has no models on the dataset, e.g. since they have been
                deleted.
        """
        with connect(self.database_path) as connection:
            return _get_method_models(connection, dataset, method_name)

    def list_method_submissions(self, dataset: str, method_name: str) -> list[Submission]:
        """Lists the submissions of the models of a method on a dataset, newest first."""
        with connect(self.database_path) as connection:
            return _list_method_submissions(connection, dataset, method_name)

    def list_actions(self, *, model_ids: T.Collection[int] | None = None) -> list[ActionLogEntry]:
        """Lists the actions, newest first, optionally only those on some models and their
        submissions."""
        query, parameters = "SELECT * FROM actions", ()
        if model_ids is not None:
            parameters = tuple(model_ids)
            query += f" WHERE model_id IN ({', '.join('?' * len(parameters))})"
        with connect(self.database_path) as connection:
            rows = connection.execute(query + " ORDER BY id DESC", parameters)
            return [_to_action_log_entry(row) for row in rows]

    # Model updates ################################################################################

    def get_inherited_method_display_name(
            self, headers: T.Sequence[ie.PredictionHeader]) -> str | None:
        """Returns the method display name that an update with files of these headers sets when
        none is given (see `plan_model_update`): that of the headers, or else the method's.

        Args:
            headers: Of one method on a dataset, e.g. those of an upload.

        Raises:
            ValueError: If there are no headers, or they have different method display names.
        """
        if not headers:
            raise ValueError("At least one header is required.")
        if (display_name := _get_method_display_name(headers)) is not None:
            return display_name
        with connect(self.database_path) as connection:
            return _read_method_display_name(connection, headers[0].dataset,
                                             headers[0].model.method_name)

    def plan_model_update(self, new_submissions: T.Sequence[NewSubmission], *,
                          action: AddingActionKind, description: str | None,
                          method_display_name: str | None = None,
                          replaces_all_files: bool = False,
                          source_submissions: T.Sequence[Submission] = ()) -> ModelUpdate:
        """Plans storing new submissions of one model, the model of their headers.

        The update creates the model if it does not exist, e.g. after it was deleted, replaces
        the submissions of the same splits, which are deleted, and sets the method display name
        of the new submissions, if they have one. It is tried and rolled back, so that
        a refused update is refused before the user confirms it (`ModelUpdate.confirmations`).

        Args:
            description: The new description of the model, or None to keep it ('' for a new
                model).
            method_display_name: The new display name of the method, which is stripped, or None
                to take it from the headers. The files keep the display names of their headers.
            replaces_all_files: Whether the new submissions replace all files of the model, so
                that the submissions of other splits are deleted, e.g. for an ensemble, whose
                files are made of the same members.
            source_submissions: See `ModelUpdate.source_submissions`.

        Raises:
            ValueError: If there are no new submissions, they are of several models, have a
                split twice, `method_display_name` is blank, or it is None and they have
                different method display names, the model has no seed or one that does not fit
                SQLite's INTEGER, or the update is refused (see `apply_model_update`).
        """
        if not new_submissions:
            raise ValueError("At least one new submission is required.")
        headers = [new.header for new in new_submissions]
        first = headers[0]
        identities = {(h.dataset, h.model.method_name, h.model.seed) for h in headers}
        if len(identities) > 1:
            raise ValueError(f"The files are of several models: {sorted(map(str, identities))}.")
        splits = [h.split for h in headers]
        if len(set(splits)) < len(splits):
            raise ValueError(f"The files have a split twice: {splits}.")
        if method_display_name is None:
            method_display_name = _get_method_display_name(headers)
        elif not (method_display_name := method_display_name.strip()):
            raise ValueError("The method display name must not be blank.")
        seed = _check_seed(first.model.seed)
        with connect(self.database_path) as connection:
            model = _find_model(connection, first.dataset, first.model.method_name, seed)
            stored = {} if model is None else {
                s.split: s for s in _list_model_submissions(connection, model.id)}
            old_method_display_name = _read_method_display_name(connection, first.dataset,
                                                                first.model.method_name)
        update = ModelUpdate(
            new_submissions=tuple(new_submissions), action=action, description=description,
            method_display_name=method_display_name, model=model,
            old_method_display_name=old_method_display_name,
            replaced={s: stored[s] for s in splits if s in stored},
            replaces_all_files=replaces_all_files,
            replaced_other_splits=({s: a for s, a in stored.items() if s not in splits}
                                   if replaces_all_files else {}),
            source_submissions=tuple(source_submissions))
        try:
            self._apply(update, actor="dry run", is_dry_run=True)
        except _DryRunFinished:
            pass
        return update

    def apply_model_update(self, update: ModelUpdate, *, submitter: str) -> list[Submission]:
        """Stores the new submissions of an update, moved from their `file_path`, deletes the
        submissions that they replace, with their files, and logs the actions:
        the action of each new submission (`ModelUpdate.action`), with the id of the submission
        that it replaces ('replaced_submission_id'), the deletion of each replaced submission,
        also of other splits (`ModelUpdate.replaced_other_splits`), and the changes of the
        description and the method display name, if any.

        Either all are stored or none. Files that are not stored stay at their `file_path`. A
        file of a replaced submission that cannot be removed after the update is stored, e.g.
        because it is being read on Windows, is logged and removed by a later deletion.

        Returns:
            The new submissions, in the order of `update.new_submissions`.

        Raises:
            ValueError: If `submitter` is empty, the model, its method display name, the
                submission of a split, or a source submission has changed since the update was
                planned, the submissions of the model would differ (see `_check_model_invariant`),
                or the models of the method could not be averaged (see `_check_method_invariant`).
        """
        return self._apply(update, _check_actor(submitter), is_dry_run=False)

    def _apply(self, update: ModelUpdate, actor: str, is_dry_run: bool) -> list[Submission]:
        """Applies an update (`apply_model_update`), or, for a dry run, does its checks and
        writes in a transaction that is rolled back, without moving the files. A dry run ends
        with `_DryRunFinished`."""
        file_sha256s = ([""] * len(update.new_submissions) if is_dry_run
                        else [compute_file_sha256(new.file_path)
                              for new in update.new_submissions])
        moved_paths: list[tuple[Path, Path]] = []  # (target, source)
        try:
            with begin_write(self.database_path) as connection:
                _check_sources_exist(connection, update.source_submissions)
                model = self._update_model(connection, update, actor)
                self._update_method_display_name(connection, model, update, actor)
                _delete_replaced_submissions(connection, model, update, actor)
                submission_ids = [
                    self._insert_submission(connection, model, new, file_sha256, update, actor)
                    for new, file_sha256 in zip(update.new_submissions, file_sha256s, strict=True)]
                _check_model_invariant(connection, model.id)
                _check_method_invariant(connection, model.dataset, model.method_name)
                if is_dry_run:
                    raise _DryRunFinished()
                for new, submission_id in zip(update.new_submissions, submission_ids):
                    target_path = self.get_predictions_path(submission_id)
                    # A directory left by a crash between the move and COMMIT is overwritten.
                    target_path.parent.mkdir(exist_ok=True)
                    os.replace(new.file_path, target_path)
                    moved_paths.append((target_path, new.file_path))
                submissions = [_get_submission(connection, i) for i in submission_ids]
                deleted_dirs = self._list_deleted_submission_dirs(connection)
        except BaseException:
            # The transaction is rolled back, also if COMMIT failed.
            for target_path, source_path in reversed(moved_paths):
                os.replace(target_path, source_path)
            raise
        self._remove_submission_dirs(deleted_dirs)
        return submissions

    @staticmethod
    def _update_model(connection: sqlite3.Connection, update: ModelUpdate, actor: str) -> Model:
        """Checks that the model is as planned, and creates or describes it.

        Raises:
            ValueError: If the model has changed since the update was planned.
        """
        model = _find_model(connection, update.dataset, update.method_name, update.seed)
        if model != update.model:
            raise ValueError(f"The model {update.label} has changed since the files were checked,"
                             f" e.g. by another user. Try again.")
        description = None if update.description is None else update.description.strip()
        if model is None:
            cursor = connection.execute(
                "INSERT INTO models (dataset, method_name, seed, description, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (update.dataset, update.method_name, update.seed, description or "",
                 get_utc_now().isoformat()))
            return _get_model(connection, cursor.lastrowid)
        if description is not None:
            _set_description(connection, model, description, actor)
        return _get_model(connection, model.id)

    @staticmethod
    def _update_method_display_name(connection: sqlite3.Connection, model: Model,
                                    update: ModelUpdate, actor: str) -> None:
        """Checks that the method display name is as planned, and sets the one of the new
        submissions, if any.

        Raises:
            ValueError: If the method display name has changed since the update was planned.
        """
        if model.method_display_name != update.old_method_display_name:
            raise ValueError(f"The display name of the method {model.method_name!r} has changed"
                             f" since the files were checked, e.g. by another user. Try again.")
        if (display_name := update.method_display_name) is not None:
            _set_method_display_name(connection, model, display_name, actor,
                                     {"by": update.action})

    @staticmethod
    def _insert_submission(connection: sqlite3.Connection, model: Model, new: NewSubmission,
                           file_sha256: str, update: ModelUpdate, actor: str) -> int:
        """Inserts a new submission and logs it, with the id of the submission that it replaces
        (`ModelUpdate.replaced`), if any. Returns its id."""
        header = new.header
        replaced = update.replaced.get(header.split)
        cursor = connection.execute(
            "INSERT INTO submissions (model_id, split, file_sha256, output_kind, context_offsets,"
            " training_splits, early_stopping_splits, num_segments, num_attributes, submitter,"
            " uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (model.id, header.split, file_sha256, new.output_kind,
             _to_optional_json(header.context_offsets),
             _to_optional_json(header.model.training_splits),
             json.dumps(list(header.model.early_stopping_splits)), new.num_segments,
             new.num_attributes, actor, get_utc_now().isoformat()))
        details = {**new.details, "notes": list(new.notes)}
        if replaced is not None:
            details["replaced_submission_id"] = replaced.id
        _log_action(connection, actor, update.action, model, cursor.lastrowid, details)
        return cursor.lastrowid

    # Other changes ################################################################################

    def delete_submission(self, submission_id: int, *, actor: str) -> Submission:
        """Deletes a submission permanently, with its file, and logs it.
        Returns it as it was.

        Raises:
            LookupError: If there is no such submission, e.g. since another user deleted it.
            ValueError: If `actor` is empty, or it is the last submission of its model, which is
                deleted with `delete_model` instead.
        """
        actor = _check_actor(actor)
        with begin_write(self.database_path) as connection:
            submission = _get_submission(connection, submission_id)
            if [s.id for s in _list_model_submissions(connection, submission.model.id)] == [
                    submission_id]:
                raise ValueError(f"File #{submission_id} is the last file of {submission.label}."
                                 f" Delete the model instead.")
            _delete_submission_row(connection, submission, actor, {})
            deleted_dirs = self._list_deleted_submission_dirs(connection)
        self._remove_submission_dirs(deleted_dirs)
        return submission

    def delete_model(self, model_id: int, *, actor: str) -> list[Submission]:
        """Deletes a model permanently, with all its submissions and their files, and logs it.
        Also deletes the display name of its method if no other model of the method on the
        dataset remains. Returns the deleted submissions.

        Raises:
            LookupError: If there is no such model, e.g. since another user deleted it.
            ValueError: If `actor` is empty.
        """
        actor = _check_actor(actor)
        with begin_write(self.database_path) as connection:
            model = _get_model(connection, model_id)
            submissions = _delete_model_rows(connection, model, actor, {})
            deleted_dirs = self._list_deleted_submission_dirs(connection)
        self._remove_submission_dirs(deleted_dirs)
        return submissions

    def delete_method(self, dataset: str, method_name: str, *, actor: str) -> list[Submission]:
        """Deletes all models of a method on a dataset permanently, as `delete_model` does, with
        their display name. Returns the deleted submissions.

        Raises:
            LookupError: If the method has no models on the dataset, e.g. since another user
                deleted them.
            ValueError: If `actor` is empty.
        """
        actor = _check_actor(actor)
        with begin_write(self.database_path) as connection:
            submissions = []
            for model in _get_method_models(connection, dataset, method_name):
                submissions += _delete_model_rows(connection, model, actor,
                                                  {"by": "delete_method"})
            deleted_dirs = self._list_deleted_submission_dirs(connection)
        self._remove_submission_dirs(deleted_dirs)
        return submissions

    def write_model_predictions_zip(self, model_id: int, target: T.BinaryIO) -> None:
        """Writes a .zip archive of the files of a model, which an upload accepts as the files of
        one model.

        Raises:
            LookupError: If there is no such model.
        """
        with connect(self.database_path) as connection:
            _get_model(connection, model_id)
            submissions = _list_model_submissions(connection, model_id)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as zip_file:
            for submission in sorted(submissions, key=lambda s: s.split):
                zip_file.write(self.get_predictions_path(submission.id),
                               get_download_file_name(submission))

    def set_model_description(self, model_id: int, description: str, *, actor: str) -> Model:
        """Changes the description of a model and logs it.

        Raises:
            LookupError: If there is no such model.
            ValueError: If `actor` is empty, or the description is unchanged.
        """
        actor = _check_actor(actor)
        description = description.strip()
        with begin_write(self.database_path) as connection:
            model = _get_model(connection, model_id)
            if description == model.description:
                raise ValueError("The description is unchanged.")
            _set_description(connection, model, description, actor)
            return _get_model(connection, model_id)

    def set_method_description(self, dataset: str, method_name: str, description: str, *,
                               actor: str) -> list[Model]:
        """Changes the description of each model of a method on a dataset to the same one, and
        logs it on each changed model.

        Returns:
            The models of the method, with the new description, by id.

        Raises:
            LookupError: If the method has no models on the dataset.
            ValueError: If `actor` is empty, or all models already have the description.
        """
        actor = _check_actor(actor)
        description = description.strip()
        with begin_write(self.database_path) as connection:
            models = _get_method_models(connection, dataset, method_name)
            if all(m.description == description for m in models):
                raise ValueError("The description is unchanged.")
            for model in models:
                _set_description(connection, model, description, actor)
            return _get_method_models(connection, dataset, method_name)

    def set_method_display_name(self, dataset: str, method_name: str, display_name: str | None,
                                *, actor: str) -> list[Model]:
        """Changes the display name of a method on a dataset, which all its models share, and
        logs it on its first model (the lowest id), since the action log is of models.

        Args:
            display_name: The new display name, None or blank to clear it.

        Returns:
            The models of the method, with the new display name, by id.

        Raises:
            LookupError: If the method has no models on the dataset.
            ValueError: If `actor` is empty, or the display name is unchanged.
        """
        actor = _check_actor(actor)
        display_name = (display_name or "").strip() or None
        with begin_write(self.database_path) as connection:
            models = _get_method_models(connection, dataset, method_name)
            if display_name == models[0].method_display_name:
                raise ValueError("The display name is unchanged.")
            _set_method_display_name(connection, models[0], display_name, actor, {})
            return _get_method_models(connection, dataset, method_name)
