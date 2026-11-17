"""The Method page: a method's details, models, and the scores of its seeds and their mean on a
split. Its display name, which all its models share, is edited here, and a Markdown description
that is given to all its models, whose first line the tables show."""

import dataclasses as dc
import functools
import typing as T

import irap_evaluation as ie
from fastapi import Request
from nicegui import run, ui

from ..accounts import can_write
from ..archive import ActionLogEntry, ModelArchive, describe_description_change
from ..components.account_sessions import AccountSessions
from ..components.attribute_filter import AttributeSubset
from ..components.changes import load_after_change, run_change_in_thread, run_refreshes
from ..components.confirmation_dialog import confirm_changes
from ..components.formatting import (
    format_context_offsets,
    format_evaluation_set_title,
    format_training_split_rows,
    make_action_table_html,
    make_truncated_name_html,
    make_unscored_files_html,
)
from ..components.models_table import make_models_table_html
from ..components.native_controls import (
    create_native_button,
    create_native_input,
    create_native_text_form,
    create_native_textarea,
)
from ..components.page_frame import (
    create_collapsible_panel,
    create_description,
    create_key_values,
    create_page_frame,
)
from ..components.split_scores_panel import ShownSplitScores, create_split_scores_panel
from ..components.view_queries import parse_dataset
from ..components.routes import (
    ATTRIBUTE_PARAMETER,
    ATTRIBUTE_SORT_PARAMETER,
    METHOD_PATH,
    MODELS_PATH,
    get_method_path,
    get_models_path,
)
from ..components.score_tables import (
    MODEL_SCORES_METRIC_NAMES,
    ScoresScope,
    make_method_scores_html,
    make_method_scores_note_html,
)
from ..datasets import DatasetContext
from ..display_settings import DisplaySettingsStore
from ..method_ranking import ShownMetrics
from ..method_scores import MethodScores, ScoredModel, compute_method_scores
from ..model_listing import MethodGroup, group_submissions_by_split, summarize_models_by_method
from ..scoring_worker import ScoringWorker


@dc.dataclass(frozen=True)
class _MethodSnapshot:
    """A method with its models and their actions, as the panels show them."""

    group: MethodGroup
    actions: list[ActionLogEntry]

    @classmethod
    def load(cls, archive: ModelArchive, dataset: str, method_name: str,
             with_actions: bool) -> T.Self:
        """
        Args:
            with_actions: Whether to load the actions, which are otherwise empty.

        Raises:
            LookupError: See `ModelArchive.get_method_models`.
        """
        models = archive.get_method_models(dataset, method_name)
        [group] = summarize_models_by_method(
            models, archive.list_method_submissions(dataset, method_name))
        actions = archive.list_actions(model_ids=[m.id for m in models]) if with_actions else []
        return cls(group, actions)


def _make_set_score_tables_html(worker: ScoringWorker, method_name: str,
                                set_to_models: T.Mapping[str, T.Sequence[ScoredModel]],
                                subset: AttributeSubset, shown_metrics: ShownMetrics,
                                scope: ScoresScope, attribute_sort: str) -> dict[str, str]:
    """Makes the score tables of a method (`make_method_scores_html`) over the attributes of
    `subset` on each evaluation set, with a row per model, by seed.

    Args:
        set_to_models: Evaluation set name -> the models of the method scored on it.
        shown_metrics: The metrics of the tables.
        attribute_sort: See `make_method_scores_html`.
    """
    def get_resamples(scores: MethodScores) -> ie.MetricValues | None:
        # The metrics of the Model page also for the models, whose resampled values are then
        # cached for both pages.
        return worker.get_method_resamples(scores, MODEL_SCORES_METRIC_NAMES)

    set_to_tables_html = {}
    for name, models in set_to_models.items():
        method_scores = compute_method_scores(method_name, models, subset.selected)
        seed_scores = [compute_method_scores(m.submission.label, [m], subset.selected)
                       for m in sorted(models, key=lambda m: m.submission.model.seed)]
        set_to_tables_html[name] = make_method_scores_html(
            method_scores, get_resamples(method_scores), seed_scores,
            [get_resamples(s) for s in seed_scores], worker.settings.confidence, shown_metrics,
            scope, attribute_sort)
    return set_to_tables_html


class _MethodPage:
    """The panels of the page and their refreshes after a change of the method. A user who cannot
    write sees them without the display name and description inputs and without the action
    log."""

    def __init__(self, archive: ModelArchive, context: DatasetContext, worker: ScoringWorker,
                 get_actor: T.Callable[[], str], can_write: bool, snapshot: _MethodSnapshot):
        """
        Args:
            context: The context of the dataset of the method.
            get_actor: See `run_change_in_thread`.
            can_write: Whether the account that was signed in when the page was created could
                write.
        """
        self.archive = archive
        self.context = context
        self.worker = worker
        self.get_actor = get_actor
        self.can_write = can_write
        self.dataset = snapshot.group.dataset
        self.method_name = snapshot.group.method_name
        #: What the panels show, loaded again after each change.
        self.snapshot = snapshot
        self.client = ui.context.client
        #: The refreshes of the panels after a change.
        self.refreshes: list[T.Callable[[], None]] = []

    async def run_change(self, change: T.Callable[[str], T.Any], status_label: ui.label) -> None:
        """Runs a change of the method (`run_change_in_thread`), rescores what it affects and
        shows the method as it is then, also after a refused change, e.g. of a method that another
        user has changed. Goes to the Models page if the method no longer has models.

        Args:
            change: Gets the actor (`get_actor`).
        """
        if await run_change_in_thread(change, self.get_actor, status_label):
            await run.io_bound(self.worker.update_method_submissions, self.dataset,
                               self.method_name)
        snapshot = await load_after_change(
            functools.partial(_MethodSnapshot.load, self.archive, self.dataset, self.method_name,
                              self.can_write),
            get_models_path(self.dataset), self.client)
        if snapshot is not None:
            self.snapshot = snapshot
            await run_refreshes(self.refreshes)

    # Details and models ###########################################################################

    def create_details(self) -> None:
        @ui.refreshable
        def show_details() -> None:
            group = self.snapshot.group
            rows = {
                "Dataset": group.dataset,
                "Method": group.method_name,
                **format_training_split_rows(group.model_info),
                "Models": str(len(group.models)),
            }
            if not self.can_write:  # Otherwise it is edited below.
                rows["Display name"] = group.method_display_name or "–"
            ui.html(f"Method {make_truncated_name_html(group.shown_method_name)}",
                    sanitize=False).classes("section-title truncated-line")
            create_key_values(rows)
            create_description(group.description)
            if self.can_write:
                create_native_button("Delete method", on_delete_method_clicked)

        async def on_display_name_saved(display_name: str) -> None:
            await self.run_change(lambda actor: self.archive.set_method_display_name(
                self.dataset, self.method_name, display_name, actor=actor), status_label)

        async def on_description_saved(description: str) -> None:
            description = description.strip()
            changes = [c for m in self.snapshot.group.models
                       if (c := describe_description_change(m.model.seed_label,
                                                            m.model.description,
                                                            description)) is not None]
            if changes and not await confirm_changes("Save description", changes, "Save"):
                return
            await self.run_change(lambda actor: self.archive.set_method_description(
                self.dataset, self.method_name, description, actor=actor), status_label)

        async def on_delete_method_clicked() -> None:
            group = self.snapshot.group
            seeds = ", ".join(m.model.seed_label for m in group.models)
            display_name_text = ("" if group.method_display_name is None
                                 else ", and its display name")
            changes = [f"Permanently deletes the method {group.method_name} on {self.dataset}:"
                       f" its {len(group.models)} models ({seeds}) with all their files and their"
                       f" scores{display_name_text}. To keep a copy, first download the files"
                       f" (.zip) of each model on its Model page, which can be uploaded again."]
            if await confirm_changes(f"Delete method {group.shown_method_name}", changes,
                                     "Delete"):
                await self.run_change(lambda actor: self.archive.delete_method(
                    self.dataset, self.method_name, actor=actor), status_label)

        with ui.element("section").classes("panel"):
            show_details()
            # Not refreshed with the details, so that a refresh does not undo an edit.
            if self.can_write:
                create_native_text_form(create_native_input,
                                        "Display name (for all models of the method)",
                                        self.snapshot.group.method_display_name or "",
                                        "Save display name", on_display_name_saved,
                                        placeholder="none", size=40)
                create_native_text_form(
                    create_native_textarea,
                    "Description (Markdown, for all models of the method; the tables show its"
                    " first line)", self.snapshot.group.description, "Save description",
                    on_description_saved, placeholder="none")
            status_label = ui.label().classes("muted")
        self.refreshes.append(show_details.refresh)

    def create_models(self) -> None:
        @ui.refreshable
        def show_models() -> None:
            ui.html(make_models_table_html([self.snapshot.group], self.context.split_names,
                                           has_method_rows=False, has_description_column=False),
                    sanitize=False).classes("w-full overflow-x-auto")

        with ui.element("section").classes("panel"):
            ui.label("Models").classes("section-title")
            ui.label("A ✓ opens the Model page with the scores of the split's file.").classes(
                "muted")
            show_models()
        self.refreshes.append(show_models.refresh)

    # Scores #######################################################################################

    def create_scores(self, split_query: str, attribute_query: T.Sequence[str],
                      attribute_sort_query: str, shown_metrics: ShownMetrics) -> None:
        create_split_scores_panel(
            self.worker, "method", split_query, attribute_query, attribute_sort_query,
            self.context.metadata.splits,
            lambda: group_submissions_by_split(self.snapshot.group.models,
                                               self.context.split_names),
            lambda split, attributes, attribute_sort: get_method_path(
                self.dataset, self.method_name, split, attributes, attribute_sort),
            lambda set_to_models, subset, split, attribute_sort: _make_set_score_tables_html(
                self.worker, self.method_name, set_to_models, subset, shown_metrics,
                ScoresScope(self.dataset, split, subset.to_query()), attribute_sort),
            self._show_split_scores)

    def _show_split_scores(self, shown: ShownSplitScores) -> None:
        worker = self.worker
        scorings = shown.scorings
        if not shown.set_to_models:
            ui.label("No model is scored on this split.").classes("muted")
            # E.g. a split without labels.
            for note in sorted({n for s in scorings.current.values() for n in s.notes}):
                ui.label(note).classes("muted")
        elif not shown.subset.selected:
            ui.label("Select at least one attribute.").classes("error")
        else:
            split_use = scorings.submissions[0].model_info.get_split_use(shown.split)
            if (explanation := ie.explain_split_use(split_use, shown.split)) is not None:
                ui.label(explanation).classes("error" if split_use == "training" else "muted")
            for name, tables_html in shown.set_to_tables_html.items():
                # The models of a method can differ in context offsets, and then not be averaged.
                offsets = sorted({format_context_offsets(m.scores.context_offsets)
                                  for m in shown.set_to_models[name]})
                ui.label(format_evaluation_set_title(name, " / ".join(offsets))).classes(
                    "subsection-title")
                ui.label(f"Averages over {shown.subset.label}").classes("muted")
                ui.html(tables_html, sanitize=False).classes("w-full overflow-x-auto")
            num_models = max(len(m) for m in shown.set_to_models.values())
            ui.html(make_method_scores_note_html(worker.settings.confidence, num_models),
                    sanitize=False)
        if unscored := scorings.list_unscored():
            ui.label("Unscored files").classes("subsection-title")
            ui.html(make_unscored_files_html(unscored, lambda s: s.model.seed_label),
                    sanitize=False)

    # Actions ######################################################################################

    def create_actions(self) -> None:
        if not self.can_write:
            return

        @ui.refreshable
        def show_actions() -> None:
            model_id_to_model = {m.model.id: m.model for m in self.snapshot.group.models}
            ui.html(make_action_table_html(self.snapshot.actions, model_id_to_model),
                    sanitize=False)

        with create_collapsible_panel("Action log of its models"):
            show_actions()
        self.refreshes.append(show_actions.refresh)


def register_method_page(archive: ModelArchive,
                         dataset_contexts: T.Mapping[str, DatasetContext],
                         worker: ScoringWorker, sessions: AccountSessions,
                         display_settings: DisplaySettingsStore) -> None:
    @ui.page(METHOD_PATH, title="Method · iRAP evaluation")
    def method_page(request: Request) -> None:
        account = sessions.get_account()
        shown_metrics = display_settings.get_shown_metrics()
        query = request.query_params
        with create_page_frame(MODELS_PATH, account):
            try:
                dataset = parse_dataset(query, dataset_contexts)
                snapshot = _MethodSnapshot.load(archive, dataset, query.get("method", ""),
                                                can_write(account))
            except (ValueError, LookupError) as e:
                ui.label(str(e)).classes("error")
                return
            page = _MethodPage(archive, dataset_contexts[dataset], worker,
                               sessions.require_writer_name, can_write(account), snapshot)
            page.create_details()
            page.create_models()
            page.create_scores(query.get("split", ""), query.getlist(ATTRIBUTE_PARAMETER),
                               query.get(ATTRIBUTE_SORT_PARAMETER, ""), shown_metrics)
            page.create_actions()
