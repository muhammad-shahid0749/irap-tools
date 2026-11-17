"""The Scores panel of the Model and Method pages: the score tables of the files of a chosen split
over the attributes of an attribute filter, loaded and computed in threads."""

import dataclasses as dc
import typing as T

from nicegui import background_tasks, run, ui

from ..archive import Submission
from ..method_ranking import collect_scored_attributes
from ..method_scores import ScoredModel
from ..scoring_worker import ScoringWorker, SubmissionScorings
from .attribute_filter import AttributeFilter, AttributeSubset
from .formatting import make_split_options
from .latest_updates import LatestUpdates
from .native_controls import create_native_select
from .refresh_timer import RefreshTimer
from .row_sorting import SORT_CLICK_JS_HANDLER


@dc.dataclass(frozen=True)
class ShownSplitScores:
    """What the Scores panel shows of the files of the chosen split.

    Attributes:
        split_error: Why the split of the URL is not shown, or ''.
        splits: The splits with files, the options of the select.
        scorings: The scorings of the files of the split.
        set_to_models: Evaluation set name -> the files scored on it, for the sets with some
            (`SubmissionScorings.group_scored_models_by_set`).
        subset: The attribute subset, None if no file has current scores.
        set_to_tables_html: Evaluation set name -> its score tables, none if no attribute is
            selected.
    """

    split: str
    split_error: str
    splits: list[str]
    scorings: SubmissionScorings
    set_to_models: dict[str, list[ScoredModel]]
    subset: AttributeSubset | None
    set_to_tables_html: dict[str, str]


def create_split_scores_panel(
        worker: ScoringWorker, owner_noun: str, split_query: str,
        attribute_query: T.Sequence[str], attribute_sort_query: str,
        split_to_segment_ids: T.Mapping[str, T.Sequence[str]],
        group_submissions_by_split: T.Callable[[], T.Mapping[str, T.Sequence[Submission]]],
        make_path: T.Callable[[str, T.Sequence[str], str], str],
        make_set_score_tables_html: T.Callable[
            [T.Mapping[str, T.Sequence[ScoredModel]], AttributeSubset, str, str], dict[str, str]],
        show_split_scores: T.Callable[[ShownSplitScores], None]
) -> T.Callable[[], T.Awaitable[None]]:
    """Creates the Scores panel, with a select of the split and an attribute filter, and starts
    its first update. Returns the update, e.g. for after a change of the files.

    A click on a header with a `data-sort` attribute in the panel
    (`row_sorting.SORT_CLICK_JS_HANDLER`) sorts the attribute rows of the score tables.

    Args:
        owner_noun: What has the files, e.g. 'model', for the messages.
        split_query: The split of the URL, '' for the first one.
        attribute_sort_query: The order of the attribute rows of the URL, '' for that of the
            attribute subset.
        split_to_segment_ids: The segments of each split of the dataset
            (`irap_data.metadata.IRAPMetadata.splits`), whose numbers the select shows.
        group_submissions_by_split: Gets split -> its files, in the order of the select.
        make_path: Makes the path of the page with a split, an attribute subset
            (`AttributeSubset.to_query`) and the order of the attribute rows, for the URL after
            a choice.
        make_set_score_tables_html: Makes evaluation set name -> its score tables of the files
            of the chosen split (`ShownSplitScores.set_to_models`) over the attributes of the
            subset, of the split, with the attribute rows in the given order. Runs in a thread.
        show_split_scores: Shows the scores below the select.
    """
    state = {"split": split_query, "attribute_sort": attribute_sort_query}
    latest_updates = LatestUpdates()

    async def on_split_changed(split: str) -> None:
        state["split"] = split
        update_url()
        await update_scores()

    async def on_subset_changed(subset: AttributeSubset) -> None:
        update_url(subset)
        await update_scores()

    async def on_attribute_sort_changed(attribute_sort: str) -> None:
        state["attribute_sort"] = attribute_sort
        update_url()
        await update_scores()

    def update_url(subset: AttributeSubset | None = None) -> None:
        subset = subset or attribute_filter.subset
        ui.navigate.history.replace(make_path(state["split"], subset.to_query(),
                                              state["attribute_sort"]))

    async def update_scores() -> None:
        """Loads and computes the scores of the chosen split in threads, and shows them if no
        later update has started meanwhile."""
        is_latest = latest_updates.start()
        split_to_submissions = group_submissions_by_split()
        if not split_to_submissions:
            show_scores.refresh(f"The {owner_noun} has no files.")
            return
        split_error = ""
        if state["split"] not in split_to_submissions:
            if state["split"]:
                split_error = f"The {owner_noun} has no file of the split {state['split']!r}."
            state["split"] = next(iter(split_to_submissions))
        split = state["split"]
        scorings = await run.io_bound(worker.load_scorings, split_to_submissions[split])
        if scorings is None or not is_latest():
            return
        set_to_models = scorings.group_scored_models_by_set()
        subset, set_to_tables_html = None, {}
        if scorings.current:
            subset = attribute_filter.set_options(collect_scored_attributes(
                m for models in set_to_models.values() for m in models))
            if subset.selected:
                set_to_tables_html = await run.io_bound(
                    make_set_score_tables_html, set_to_models, subset, split,
                    state["attribute_sort"])
                if set_to_tables_html is None or not is_latest():
                    return
        show_scores.refresh(ShownSplitScores(split, split_error, list(split_to_submissions),
                                             scorings, set_to_models, subset,
                                             set_to_tables_html))

    @ui.refreshable
    def show_scores(shown: ShownSplitScores | str) -> None:
        """
        Args:
            shown: The scores, or a message instead, e.g. while they are loaded.
        """
        if isinstance(shown, str):
            ui.label(shown).classes("muted")
            return
        refresh_timer.watch([lambda i=i: worker.is_scoring_pending(i)
                             for i in shown.scorings.pending_ids])
        if shown.split_error:
            ui.label(shown.split_error).classes("error")
        with ui.element("div").classes("form-row"):
            create_native_select("Split", make_split_options(shown.splits, split_to_segment_ids),
                                 shown.split, on_split_changed)
        show_split_scores(shown)

    with ui.element("section").classes("panel").on(
            "click", lambda e: on_attribute_sort_changed(e.args),
            js_handler=SORT_CLICK_JS_HANDLER):
        ui.label("Scores").classes("section-title")
        attribute_filter = AttributeFilter(attribute_query, on_subset_changed)
        refresh_timer = RefreshTimer(update_scores)
        show_scores("Loading the scores…")
    background_tasks.create(update_scores(), name="update scores")
    return update_scores
