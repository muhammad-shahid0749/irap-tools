"""The Models page (the list of models with the upload form and the form that creates an ensemble
of selected models), and the endpoints for uploading and downloading prediction files."""

import dataclasses as dc
import typing as T
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from nicegui import app, run, ui

from ..accounts import can_write
from ..archive import ModelArchive, Submission, get_download_file_name
from ..components.account_sessions import AccountSessions
from ..components.confirmation_dialog import confirm_changes
from ..components.ensemble_form import EnsembleForm
from ..components.formatting import make_code_spans_html
from ..components.models_table import (
    MODEL_CHECKBOX_CHANGE_HANDLER,
    make_check_models_js,
    make_models_table_html,
)
from ..components.native_controls import (
    create_file_upload_input,
    create_native_button,
    create_native_input,
    create_native_select,
    set_native_input_placeholder,
    set_status,
)
from ..components.page_frame import create_page_frame, create_write_permission_panel
from ..components.routes import (
    MODELS_PATH,
    PREDICTIONS_DOWNLOAD_PATH,
    SCORES_PATH,
    UPLOAD_PATH,
    get_model_path,
    get_models_path,
)
from ..components.view_queries import select_matching_names
from ..datasets import DatasetContext
from ..model_listing import MethodGroup, ModelSummary, summarize_models_by_method
from ..scoring_worker import ScoringWorker
from ..uploads import (
    MAX_UPLOAD_NUM_BYTES,
    MAX_UPLOAD_SIZE_TEXT,
    apply_upload,
    parse_optional_text,
    parse_seed,
    plan_upload,
    read_upload_headers,
)

#: Uploaded files that are not stored are removed after this time, e.g. those of a closed page
#: whose client NiceGUI has not deleted.
_ABANDONED_UPLOAD_AGE_S = 24 * 3600
_UPLOAD_TOO_LARGE_MESSAGE = f"A file can have at most {MAX_UPLOAD_SIZE_TEXT}."
_UPLOAD_FORM_TITLE = "Upload predictions"

async def _write_upload(request: Request, path: Path) -> None:
    """Writes the body of an upload request to `path`. Removes `path` if that fails, also if the
    client disconnects.

    Raises:
        HTTPException: 413 if the body has more than `uploads.MAX_UPLOAD_NUM_BYTES`.
    """
    num_bytes = 0
    try:
        with path.open("wb") as file:
            async for chunk in request.stream():
                num_bytes += len(chunk)
                if num_bytes > MAX_UPLOAD_NUM_BYTES:
                    raise HTTPException(status_code=413, detail=_UPLOAD_TOO_LARGE_MESSAGE)
                file.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@dc.dataclass
class _UploadFormState:
    """The state of the upload form of one page.

    Attributes:
        upload_id: The latest upload of the file input (see `create_file_upload_input`).
        upload_path: The file of that upload (see `ModelArchive.get_upload_path`), from
            when it is uploaded until a check takes it.
        is_uploading: Whether the latest upload is in progress.
        is_checking: Whether an upload is being checked and stored after a click on Add.
    """

    upload_id: int | None = None
    upload_path: Path | None = None
    is_uploading: bool = False
    is_checking: bool = False
    file_name: str = ""
    description: str = ""
    method_display_name: str = ""
    seed_text: str = ""


def _make_method_display_name_placeholder(archive: ModelArchive, upload_path: Path,
                                          file_name: str) -> str:
    """Makes the placeholder of the method display name input for an upload, with the display
    name that a blank input keeps (`ModelArchive.get_inherited_method_display_name`)."""
    headers = read_upload_headers(upload_path, file_name=file_name)
    return f"blank: {archive.get_inherited_method_display_name(headers) or 'none'}"


def _create_upload_form(archive: ModelArchive,
                        dataset_contexts: T.Mapping[str, DatasetContext],
                        worker: ScoringWorker, get_actor: T.Callable[[], str]) -> None:
    """Creates the form for uploading a prediction file or a .zip archive of the files of one
    model.

    The upload is checked and planned (`uploads.plan_upload`), and the user confirms its changes,
    if any (`archive.ModelUpdate.confirmations`). A stored upload opens the page of its model.

    Args:
        get_actor: Returns the name of the signed-in account that can write, or raises
            `ValueError`.
    """
    form = _UploadFormState()

    def remove_pending_upload() -> None:
        if form.upload_path is not None:
            form.upload_path.unlink(missing_ok=True)
            form.upload_path = None

    async def show_inherited_method_display_name(upload_path: Path, upload_id: int,
                                                 file_name: str) -> None:
        """Shows the method display name that a blank input keeps as the input's placeholder.
        It shows none if the headers cannot be read, e.g. of an invalid file, since Add then
        reports why."""
        client = ui.context.client
        try:
            placeholder = await run.io_bound(_make_method_display_name_placeholder, archive,
                                             upload_path, file_name)
        except (OSError, ValueError):  # Also irap_evaluation.PredictionFormatError.
            return
        # None if the server is stopping.
        if placeholder is not None and form.upload_id == upload_id and not client.is_deleted:
            set_native_input_placeholder(display_name_input, placeholder)

    def update_add_button() -> None:
        add_button.props["disabled"] = form.is_uploading or form.is_checking
        add_button.update()

    async def on_upload_status_changed(status: dict) -> None:
        if status["status"] == "uploading":
            remove_pending_upload()
            form.upload_id, form.is_uploading = status["upload_id"], True
            update_add_button()
            set_native_input_placeholder(display_name_input, "")
            set_status(status_label, f"Uploading {status['file_name']}…")
            return
        upload_path = (archive.get_upload_path(status["upload_name"])
                       if status["status"] == "uploaded" else None)
        if status["upload_id"] != form.upload_id:  # A newer upload replaced this one.
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)
            return
        form.is_uploading = False
        update_add_button()
        if upload_path is not None:
            form.upload_path, form.file_name = upload_path, status["file_name"]
            set_status(status_label, f"{form.file_name} is uploaded. Click Add to check and"
                                     f" store it.")
            await show_inherited_method_display_name(upload_path, form.upload_id,
                                                     form.file_name)
        else:
            set_status(status_label, f"The upload failed: {status.get('message', status)}",
                       is_error=True)

    def set_checking(is_checking: bool) -> None:
        form.is_checking = is_checking
        update_add_button()

    async def on_add_clicked() -> None:
        if form.upload_path is None:
            set_status(status_label, "Choose a file first.", is_error=True)
            return
        if not form.upload_path.exists():
            form.upload_path = None
            set_status(status_label, f"{form.file_name} is no longer on the server, since"
                                     f" uploads are removed after"
                                     f" {_ABANDONED_UPLOAD_AGE_S / 3600:g} h. Choose it again.",
                       is_error=True)
            return
        try:
            actor = get_actor()
            seed = parse_seed(form.seed_text)
        except ValueError as e:
            set_status(status_label, str(e), is_error=True)
            return
        # The check takes the file, so that a new upload does not remove it.
        upload_path, upload_id, file_name = form.upload_path, form.upload_id, form.file_name
        form.upload_path = None

        client = ui.context.client

        def give_back_upload() -> None:
            # Kept for a retry, e.g. with another seed, unless the page is closed
            # (`remove_pending_upload`) or has a newer upload.
            if form.upload_id == upload_id and not client.is_deleted:
                form.upload_path = upload_path
            else:
                upload_path.unlink(missing_ok=True)

        set_status(status_label, f"Checking {file_name}…")
        set_checking(True)
        try:
            submissions = await _check_and_store(archive, dataset_contexts, upload_path,
                                                 file_name=file_name, seed=seed,
                                                 description=form.description,
                                                 method_display_name=form.method_display_name,
                                                 actor=actor)
        except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
            give_back_upload()
            if not client.is_deleted:
                set_status(status_label, f"{file_name} is not stored: {e}", is_error=True)
            return
        except Exception as e:
            give_back_upload()
            if not client.is_deleted:
                set_status(status_label, f"Internal error while checking {file_name}: {e!r}."
                                         f" See the server log.", is_error=True)
            raise
        finally:
            if not client.is_deleted:
                set_checking(False)
        if not submissions:  # Cancelled, or the server is stopping.
            give_back_upload()
            if not client.is_deleted:
                set_status(status_label, f"{file_name} is not stored.")
            return
        await run.io_bound(worker.update_model_submissions, submissions[0].model)
        if not client.is_deleted:
            ui.navigate.to(get_model_path(submissions[0].model.id, submissions[0].split))

    ui.context.client.on_delete(remove_pending_upload)

    with ui.element("section").classes("panel"):
        ui.label(_UPLOAD_FORM_TITLE).classes("section-title")
        ui.html(make_code_spans_html(
            "A `.predictions.parquet` file in the `irap_evaluation` format, whose header gives"
            " the dataset, split, method, seed and training splits, or a `.zip` archive with"
            " one such file per split of one model, stored only if all are accepted. A file"
            " without a seed needs one in the Seed field, also for a method's only model. A file"
            " of a split that the model already has replaces its file, which is deleted"
            " permanently."),
            sanitize=False).classes("muted")
        with ui.element("div").classes("form-row"):
            create_file_upload_input("File", UPLOAD_PATH, ".parquet,.zip",
                                     on_upload_status_changed)
            display_name_input = create_native_input(
                "Method display name", "", lambda v: setattr(form, "method_display_name", v),
                size=30)
            create_native_input("Seed", "", lambda v: setattr(form, "seed_text", v),
                                placeholder="from the file", size=10)
            create_native_input("Description", "", lambda v: setattr(form, "description", v),
                                placeholder="blank: keep the model's", size=40)
            add_button = create_native_button("Add", on_add_clicked)
        status_label = ui.label().classes("muted")


async def _check_and_store(archive: ModelArchive,
                           dataset_contexts: T.Mapping[str, DatasetContext], upload_path: Path, *,
                           file_name: str, seed: int | None, description: str,
                           method_display_name: str, actor: str) -> list[Submission]:
    """Plans an upload, asks the user to confirm its changes, if any, and applies it.

    Args:
        description, method_display_name: As entered, blank to keep the stored one.
        actor: The name of the signed-in account.

    Returns:
        The new submissions, or [] if the user cancelled, closed the page, or the server is
        stopping.

    Raises:
        ValueError: See `uploads.plan_upload` and `uploads.apply_upload`.
    """
    client = ui.context.client
    planned = await run.io_bound(plan_upload, archive, dataset_contexts, upload_path,
                                 file_name=file_name, seed=seed,
                                 description=parse_optional_text(description),
                                 method_display_name=parse_optional_text(method_display_name))
    if planned is None:  # The server is stopping.
        return []
    update = planned.update
    try:
        if client.is_deleted:
            return []
        if update.confirmations and not await confirm_changes(
                f"Upload of {update.label} on {update.dataset}", update.confirmations, "Store"):
            return []
        return await run.io_bound(apply_upload, archive, planned, submitter=actor) or []
    finally:
        planned.discard()  # Also if the page is closed, or the server is stopping.


@dc.dataclass(frozen=True)
class _ModelsView:
    """What the Models page shows, which its URL keeps.

    Attributes:
        dataset: The shown dataset, or '' for all.
        model_pattern: The regular expression of the shown model labels
            (`select_matching_names`), '' for all.
    """

    dataset: str = ""
    model_pattern: str = ""

    def to_path(self) -> str:
        return get_models_path(self.dataset, self.model_pattern)


def _load_groups(archive: ModelArchive,
                 dataset_names: T.Iterable[str]) -> dict[str, list[MethodGroup]]:
    """Loads the models of each dataset by method (`summarize_models_by_method`)."""
    models = archive.list_models()
    submissions = archive.list_submissions()
    return {d: summarize_models_by_method([m for m in models if m.dataset == d], submissions)
            for d in dataset_names}


def _list_group_models(dataset_to_groups: T.Mapping[str, T.Iterable[MethodGroup]]
                       ) -> list[ModelSummary]:
    return [m for groups in dataset_to_groups.values() for g in groups for m in g.models]


def _filter_groups(groups: T.Iterable[MethodGroup],
                   shown_labels: T.Collection[str]) -> list[MethodGroup]:
    """Leaves out the models whose shown label is not in `shown_labels`, and the methods without
    models left."""
    filtered = (dc.replace(g, models=tuple(m for m in g.models
                                           if m.model.shown_label in shown_labels))
                for g in groups)
    return [g for g in filtered if g.models]


def register_models_pages(archive: ModelArchive,
                          dataset_contexts: T.Mapping[str, DatasetContext],
                          worker: ScoringWorker, sessions: AccountSessions) -> None:
    @app.post(UPLOAD_PATH)
    async def receive_upload(request: Request, file_name: str = "") -> dict:
        # Before the body is read, so that visitors cannot make the server store it.
        try:
            sessions.require_writer_name()
        except ValueError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        if int(request.headers.get("content-length", 0)) > MAX_UPLOAD_NUM_BYTES:
            raise HTTPException(status_code=413, detail=_UPLOAD_TOO_LARGE_MESSAGE)
        await run.io_bound(archive.remove_uploads, older_than_s=_ABANDONED_UPLOAD_AGE_S)
        path = archive.make_upload_path(file_name)
        await _write_upload(request, path)
        return {"status": "uploaded", "upload_name": path.name, "file_name": file_name}

    @app.get(PREDICTIONS_DOWNLOAD_PATH)
    def download_predictions(submission_id: int) -> FileResponse:
        try:
            submission = archive.get_submission(submission_id)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return FileResponse(archive.get_predictions_path(submission_id),
                            filename=get_download_file_name(submission),
                            media_type="application/vnd.apache.parquet")

    @ui.page("/")
    def index_page() -> RedirectResponse:
        return RedirectResponse(SCORES_PATH)

    @ui.page(MODELS_PATH, title="Models · iRAP evaluation")
    def models_page(dataset: str = "", models: str = "") -> None:
        """
        Args:
            dataset, models: `_ModelsView.dataset` and `model_pattern`.
        """
        account = sessions.get_account()
        if dataset and dataset not in dataset_contexts:
            with create_page_frame(MODELS_PATH, account):
                ui.label(f"Unknown dataset {dataset!r}. Datasets:"
                         f" {', '.join(dataset_contexts)}.").classes("error")
            return
        view = _ModelsView(dataset, models)
        # The models of all datasets, loaded again after each change of the view or the archive.
        dataset_to_all_groups = _load_groups(archive, dataset_contexts)

        async def on_view_changed(**changes: str) -> None:
            nonlocal view
            view = dc.replace(view, **changes)
            ui.navigate.history.replace(view.to_path())
            await reload()

        async def reload() -> None:
            """Loads the models again, e.g. with a new ensemble, and shows them."""
            nonlocal dataset_to_all_groups
            client = ui.context.client
            loaded = await run.io_bound(_load_groups, archive, dataset_contexts)
            if loaded is None or client.is_deleted:  # None if the server is stopping.
                return
            dataset_to_all_groups = loaded
            if form is not None:
                form.set_models(_list_group_models(loaded))
            show_tables.refresh()

        def list_shown_groups() -> dict[str, list[MethodGroup]]:
            """Lists the models of the shown datasets that the filter matches, by method, without
            the datasets without such models.

            Raises:
                ValueError: See `select_matching_names`.
            """
            matching_labels = select_matching_names(
                view.model_pattern,
                {m.model.shown_label for m in _list_group_models(dataset_to_all_groups)})
            datasets = [view.dataset] if view.dataset else list(dataset_contexts)
            return {d: groups for d in datasets
                    if (groups := _filter_groups(dataset_to_all_groups[d], matching_labels))}

        # Only with a form.
        def check_selected_models() -> None:
            """Shows the form's selection in the checkboxes of the tables, after a selection by
            the server."""
            ui.run_javascript(make_check_models_js(form.selected_model_ids))

        def select_shown(is_selected: bool) -> None:
            try:
                dataset_to_groups = list_shown_groups()
            except ValueError:  # Shown instead of the tables.
                return
            form.set_selected([m.model.id for m in _list_group_models(dataset_to_groups)],
                              is_selected)
            check_selected_models()

        @ui.refreshable
        def show_tables() -> None:
            """Shows the tables, with checkboxes if there is an ensemble form. A click on a
            checkbox only selects its model in the form, since the checkbox shows it."""
            try:
                dataset_to_groups = list_shown_groups()
            except ValueError as e:
                ui.label(str(e)).classes("error")
                return
            if not dataset_to_groups:
                ui.label("No models match the filter." if view.model_pattern
                         else "No models.").classes("muted")
            for name, groups in dataset_to_groups.items():
                ui.label(name).classes("subsection-title")
                table = ui.html(make_models_table_html(
                    groups, dataset_contexts[name].split_names,
                    selected_model_ids=None if form is None else form.selected_model_ids),
                    sanitize=False).classes("w-full overflow-x-auto")
                if form is not None:
                    table.on("change", lambda e, f=form: f.set_selected([e.args[0]], e.args[1]),
                             js_handler=MODEL_CHECKBOX_CHANGE_HANDLER)

        with create_page_frame(MODELS_PATH, account):
            if can_write(account):
                _create_upload_form(archive, dataset_contexts, worker,
                                    sessions.require_writer_name)
            else:
                create_write_permission_panel(account, _UPLOAD_FORM_TITLE,
                                              "upload predictions and create ensembles")
            models_panel = ui.element("section").classes("panel")
            # Below the models, since they are selected first.
            form = (EnsembleForm(archive, dataset_contexts, worker, sessions.require_writer_name,
                                 on_selection_changed=check_selected_models, on_created=reload)
                    if can_write(account) else None)
            if form is not None:
                form.set_models(_list_group_models(dataset_to_all_groups))
            with models_panel:
                ui.label("Models").classes("section-title")
                with ui.element("div").classes("form-row"):
                    create_native_select("Dataset", {"": "All", **{n: n for n in dataset_contexts}},
                                         view.dataset, lambda v: on_view_changed(dataset=v))
                    create_native_input("Model filter (regex)", view.model_pattern,
                                        lambda v: on_view_changed(model_pattern=v),
                                        placeholder="all", size=24)
                    if form is not None:
                        # Hidden with the checkboxes while the ensemble form is collapsed.
                        create_native_button("Select shown", lambda: select_shown(True)
                                             ).classes("model-selection")
                        create_native_button("Unselect shown", lambda: select_shown(False)
                                             ).classes("model-selection")
                ui.label("A method name opens the Method page with the scores of all its seeds."
                         " A ✓ opens the Model page with the scores of the split's file."
                         " Hover over it for the file details. Below it: whether the model used"
                         " the split for training or early stopping.").classes("muted")
                show_tables()
