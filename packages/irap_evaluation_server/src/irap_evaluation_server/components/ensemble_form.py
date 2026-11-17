"""The panel of the Models page that creates an ensemble of the models selected in its table (see
`ensembles`)."""

import dataclasses as dc
import typing as T

import irap_evaluation as ie
from nicegui import run, ui

from ..archive import Model, ModelArchive, Submission, discard_model_update
from ..datasets import DatasetContext
from ..ensembles import EnsembleMember, EnsemblePlan, plan_ensemble, prepare_ensemble
from ..model_listing import ModelSummary
from ..scoring_worker import ScoringWorker
from ..uploads import parse_optional_text, parse_seed
from .confirmation_dialog import confirm_changes
from .formatting import format_splits, make_truncated_model_link_html
from .native_controls import (
    create_native_button,
    create_native_checkbox,
    create_native_input,
    create_native_select,
    set_native_input_placeholder,
    set_status,
)
from .page_frame import create_collapsible_panel

_NO_MEMBERS_TEXT = "Select at least 2 models in the table above."
_COMBINATION_TEXTS: dict[ie.MemberCombination, str] = {
    "same": "refuse",
    "intersection": "those that every member predicts",
    "union": "all that any member predicts",
}


def _parse_members(models: T.Sequence[Model],
                   weight_texts: T.Mapping[int, str]) -> list[EnsembleMember]:
    """Parses the weights of the members. A blank or missing weight is 1.

    Args:
        weight_texts: By model id.

    Raises:
        ValueError: If a weight is not a number.
    """
    members = []
    for model in models:
        text = weight_texts.get(model.id, "1").strip() or "1"
        try:
            weight = float(text)
        except ValueError:
            raise ValueError(f"The weight of {model.shown_label} must be a number, got {text!r}."
                             ) from None
        members.append(EnsembleMember(model, weight))
    return members


def _describe_plan(plan: EnsemblePlan, model: ie.ModelInfo) -> str:
    """
    Args:
        model: `plan.make_model_info()`.
    """
    text = f"Creates the ensemble on the splits {', '.join(plan.split_to_submissions)}."
    for split, labels in plan.split_to_missing.items():
        text += f" Skips {split}, which has no file of {', '.join(labels)}."
    if model.training_splits is None:
        text += " Its training splits are unknown, since a member's are."
    else:
        text += (f" It is trained on {format_splits(model.training_splits)}, the union of those"
                 f" of the members.")
    if model.early_stopping_splits:
        text += f" Its early stopping splits are {format_splits(model.early_stopping_splits)}."
    return text


@dc.dataclass
class _ModelInputs:
    """The texts of the inputs of the ensemble's model, as entered."""

    method_name: str = ""
    display_name: str = ""
    seed_text: str = "0"
    description: str = ""
    segment_combination: ie.MemberCombination = "same"
    attribute_combination: ie.MemberCombination = "same"


class EnsembleForm:
    """The "Create an ensemble" panel: the selected models, i.e. the members, with their weights,
    the plan of the ensemble, and the inputs of its model.

    The models are selected in the table of the Models page (`set_selected`). They can be
    unselected in the panel's list of members, which also shows those that the table hides. The
    panel is collapsed at first, and while it is, the selection in the table is hidden
    (`model-selection` in `styles.css`).
    """

    def __init__(self, archive: ModelArchive, dataset_contexts: T.Mapping[str, DatasetContext],
                 worker: ScoringWorker, get_actor: T.Callable[[], str], *,
                 on_selection_changed: T.Callable[[], None],
                 on_created: T.Callable[[], T.Awaitable[None]]):
        """Creates the panel in the current container. It has no models until `set_models`.

        Args:
            get_actor: Returns the name of the signed-in account that can write, or raises
                `ValueError`.
            on_selection_changed: Called after a member is unselected in the panel, e.g. to
                uncheck it in the table.
            on_created: Awaited after an ensemble is stored, e.g. to show it in the table.
        """
        self.archive = archive
        self.dataset_contexts = dataset_contexts
        self.worker = worker
        self.get_actor = get_actor
        self.on_selection_changed = on_selection_changed
        self.on_created = on_created
        #: The models that can be members, with their submissions, which the shown plan is made
        #: of, in the order of the list of members. Create plans with new submissions.
        self.summaries: list[ModelSummary] = []
        self._selected_model_ids: set[int] = set()
        #: Model id -> the text of its weight input, also after it is unselected, so that it
        #: keeps its weight if it is selected again.
        self.weight_texts: dict[int, str] = {}
        self.inputs = _ModelInputs()
        self._create_panel()

    @property
    def selected_model_ids(self) -> frozenset[int]:
        return frozenset(self._selected_model_ids)

    def set_models(self, summaries: T.Sequence[ModelSummary]) -> None:
        """Sets the models that can be members, e.g. after the table is loaded again.

        Args:
            summaries: In the order of the list of members, e.g. of the table.
        """
        self.summaries = list(summaries)
        self._show_members.refresh()
        self._update_plan()

    def set_selected(self, model_ids: T.Iterable[int], is_selected: bool) -> None:
        """Selects or unselects models, without `on_selection_changed`."""
        if is_selected:
            self._selected_model_ids.update(model_ids)
        else:
            self._selected_model_ids.difference_update(model_ids)
        self._show_members.refresh()
        self._update_plan()

    def _list_members(self) -> list[ModelSummary]:
        return [s for s in self.summaries if s.model.id in self._selected_model_ids]

    def _plan(self, members: T.Sequence[ModelSummary],
              submissions: T.Iterable[Submission]) -> EnsemblePlan:
        """Plans the ensemble of the members (`_list_members`) on the dataset of the first.

        Args:
            submissions: Those of the members, or more.

        Raises:
            ValueError: If there are no members, and see `_parse_members` and
                `ensembles.plan_ensemble`.
        """
        if not members:
            raise ValueError(_NO_MEMBERS_TEXT)
        dataset = members[0].model.dataset
        return plan_ensemble(submissions, dataset,
                             _parse_members([m.model for m in members], self.weight_texts),
                             self.dataset_contexts[dataset].split_names)

    def _update_plan(self) -> None:
        members = self._list_members()
        if not members:
            set_status(self._plan_label, _NO_MEMBERS_TEXT)
            self._set_default_name("")
            return
        try:
            plan = self._plan(members, [s for m in members for s in m.submissions])
        except ValueError as e:
            set_status(self._plan_label, str(e), is_error=True)
            self._set_default_name("")
            return
        model = plan.make_model_info()
        set_status(self._plan_label, _describe_plan(plan, model))
        self._set_default_name(model.method_name)

    def _set_default_name(self, default_name: str) -> None:
        set_native_input_placeholder(self._name_input,
                                     f"blank: {default_name}" if default_name else "")

    def _unselect_member(self, model_id: int) -> None:
        self.set_selected([model_id], False)
        self.on_selection_changed()

    def _set_weight(self, model_id: int, text: str) -> None:
        self.weight_texts[model_id] = text
        self._update_plan()

    @ui.refreshable_method
    def _show_members(self) -> None:
        """Shows a row of each member: a checkbox that unselects it, its weight and its splits."""
        with ui.element("div").classes("ensemble-members"):
            for summary in self._list_members():
                model = summary.model
                create_native_checkbox(
                    make_truncated_model_link_html(model.id, model.shown_label), True,
                    lambda _, i=model.id: self._unselect_member(i), is_label_html=True)
                # Text, since a number input sends '' for text that is not a number.
                create_native_input("", self.weight_texts.get(model.id, "1"),
                                    lambda text, i=model.id: self._set_weight(i, text), size=6)
                split_names = self.dataset_contexts[model.dataset].split_names
                ui.label(", ".join(summary.list_splits(split_names))).classes("muted")

    def _set_creating(self, is_creating: bool) -> None:
        self._create_button.props["disabled"] = is_creating
        self._create_button.update()

    async def _on_create_clicked(self) -> None:
        client = ui.context.client
        # New submissions, since the shown plan's may have changed.
        submissions = await run.io_bound(self.archive.list_submissions)
        if submissions is None or client.is_deleted:  # None if the server is stopping.
            return
        try:
            plan = self._plan(self._list_members(), submissions)
            actor = self.get_actor()
            seed = parse_seed(self.inputs.seed_text)
        except ValueError as e:
            set_status(self._status_label, str(e), is_error=True)
            return
        set_status(self._status_label, "Creating the ensemble…")
        self._set_creating(True)
        try:
            created = await _create(
                self.archive, self.dataset_contexts[plan.dataset], plan,
                method_name=parse_optional_text(self.inputs.method_name),
                method_display_name=parse_optional_text(self.inputs.display_name),
                seed=seed, segment_combination=self.inputs.segment_combination,
                attribute_combination=self.inputs.attribute_combination,
                description=parse_optional_text(self.inputs.description), actor=actor)
        except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
            if not client.is_deleted:
                set_status(self._status_label, str(e), is_error=True)
            return
        except Exception as e:
            if not client.is_deleted:
                set_status(self._status_label, f"Internal error: {e!r}. See the server log.",
                           is_error=True)
            raise
        finally:
            if not client.is_deleted:
                self._set_creating(False)
        if created:
            await run.io_bound(self.worker.update_model_submissions, created[0].model)
        if client.is_deleted:
            return
        if not created:  # Cancelled, or the server is stopping.
            set_status(self._status_label, "The ensemble is not stored.")
            return
        self._status_label.set_text("")
        model = created[0].model
        self._result_html.set_content(
            f"Created {make_truncated_model_link_html(model.id, model.shown_label)} on the splits"
            f" {', '.join(s.split for s in created)}. It is scored in the background.")
        await self.on_created()

    def _create_panel(self) -> None:
        # Collapsed at first, which hides the selection in the table (`styles.css`).
        with create_collapsible_panel("Create an ensemble", "ensemble-section"):
            with ui.element("div").classes("flex flex-col gap-2"):
                ui.label("The ensemble predicts the weighted mean of the members' distributions."
                         " A hard prediction counts as one-hot, so an ensemble of hard predictions"
                         " is a weighted vote. An attribute prediction is invalid only if it is"
                         " invalid in every member. The ensemble gets a file for each split that"
                         " every member has. If its model exists, these files replace all of the"
                         " model's files, which are deleted permanently. It counts as trained on"
                         " the training splits of all members.").classes("muted")
                ui.label("Members and weights").classes("subsection-title")
                self._show_members()
                self._plan_label = ui.label(_NO_MEMBERS_TEXT).classes("muted")
                with ui.element("div").classes("form-row"):
                    self._name_input = create_native_input(
                        "Method name", "", lambda v: setattr(self.inputs, "method_name", v),
                        size=40)
                    create_native_input("Display name", "",
                                        lambda v: setattr(self.inputs, "display_name", v),
                                        placeholder="blank: keep the method's", size=30)
                    create_native_input("Seed", self.inputs.seed_text,
                                        lambda v: setattr(self.inputs, "seed_text", v), size=10)
                    create_native_input("Description", "",
                                        lambda v: setattr(self.inputs, "description", v),
                                        placeholder="blank: keep the model's", size=40)
                with ui.element("div").classes("form-row"):
                    create_native_select(
                        "Segments if the members predict different ones", _COMBINATION_TEXTS,
                        "same", lambda v: setattr(self.inputs, "segment_combination", v))
                    create_native_select(
                        "Attributes if the members predict different ones", _COMBINATION_TEXTS,
                        "same", lambda v: setattr(self.inputs, "attribute_combination", v))
                ui.label("Members with different context offsets predict different segments."
                         " With all segments or attributes, a member that does not predict one"
                         " counts as invalid, so an attribute that only one member predicts is"
                         " that member's prediction.").classes("muted")
                with ui.element("div").classes("form-row items-center"):
                    self._create_button = create_native_button("Create",
                                                               self._on_create_clicked)
                    self._status_label = ui.label().classes("muted")
                self._result_html = ui.html("", sanitize=False)


async def _create(archive: ModelArchive, context: DatasetContext, plan: EnsemblePlan, *,
                  method_name: str | None, method_display_name: str | None, seed: int | None,
                  segment_combination: ie.MemberCombination,
                  attribute_combination: ie.MemberCombination, description: str | None,
                  actor: str) -> list[Submission]:
    """Makes the ensemble files (`ensembles.prepare_ensemble`), asks the user to confirm the
    changes, if any, and stores them.

    Args:
        method_name, method_display_name, segment_combination, attribute_combination,
            description: See `ensembles.prepare_ensemble`.
        actor: The name of the signed-in account.

    Returns:
        The new submissions, or [] if the user cancelled, closed the page, or the server is
        stopping.

    Raises:
        ValueError: See `ensembles.prepare_ensemble` and
            `archive.ModelArchive.apply_model_update`.
    """
    client = ui.context.client
    update = await run.io_bound(prepare_ensemble, archive, context, plan,
                                method_name=method_name, method_display_name=method_display_name,
                                seed=seed, segment_combination=segment_combination,
                                attribute_combination=attribute_combination,
                                description=description)
    if update is None:  # The server is stopping.
        return []
    try:
        if client.is_deleted:
            return []
        if update.confirmations and not await confirm_changes(
                f"Ensemble {update.label} on {update.dataset}", update.confirmations, "Store"):
            return []
        return await run.io_bound(archive.apply_model_update, update, submitter=actor) or []
    finally:
        discard_model_update(update)  # The files that are not moved into the archive.
