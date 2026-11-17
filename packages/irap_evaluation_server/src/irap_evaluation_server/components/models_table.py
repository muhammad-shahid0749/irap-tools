"""HTML of the table of models of the Models and Method pages: the models of methods
(`model_listing.MethodGroup`) with a column per split."""

import html
import json
import typing as T

import irap_evaluation as ie

from ..archive import get_description_summary
from ..model_listing import MethodGroup, ModelSummary
from .formatting import (
    format_splits,
    format_utc_time,
    make_header_cell_html,
    make_method_link_html,
    make_model_link_html,
    make_table_html_from_header,
    make_title_attribute,
    make_truncated_name_html,
)
from .native_controls import NATIVE_CHOICE_ATTRIBUTES

#: The `js_handler` of the 'change' event of a `ui.html` with a table with checkboxes
#: (`make_models_table_html`), which emits [the model id, whether it is checked].
MODEL_CHECKBOX_CHANGE_HANDLER = ("(e) => { const id = e.target.dataset.modelId;"
                                 " if (id !== undefined) emit([Number(id), e.target.checked]); }")
#: The columns of the models table after the split columns.
_SUMMARY_COLUMNS = ("Output", "Segments", "Attributes", "Description", "Submitter", "Updated")
_SUMMARY_NUMBER_COLUMNS = ("Segments", "Attributes")
#: The mark of what a model used a split for, in the column of the split, which the header names
#: (`method_ranking.describe_split_use` for marks without it).
_SPLIT_USE_MARKS: dict[ie.SplitUse, str] = {"training": "training", "early_stopping": "early stop",
                                            "unknown": "unknown", "held_out": ""}


def _make_submission_cell_html(summary: ModelSummary, split: str) -> str:
    """Makes the cell of a split: a link to the model page with the scores of its submission,
    the submission details in the title, and a mark if the model used the split, with
    the explanation in its title."""
    submission = summary.split_to_submission.get(split)
    if submission is None:
        return '<td class="mark"></td>'
    title = (f"File #{submission.id}, uploaded {format_utc_time(submission.uploaded_at)} by"
             f" {submission.submitter}, {submission.num_segments} segments,"
             f" {submission.num_attributes} attributes")
    link = make_model_link_html(summary.model.id, "✓", split)
    split_use = submission.model_info.get_split_use(split)
    explanation = ie.explain_split_use(split_use, split)
    mark_html = ("" if explanation is None else
                 f'<div class="interval" title="{html.escape(explanation)}">'
                 f"{_SPLIT_USE_MARKS[split_use]}</div>")
    return f'<td class="mark" title="{html.escape(title)}">{link}{mark_html}</td>'


def _make_summary_cells_html(summary: ModelSummary, has_description_column: bool) -> str:
    attribute_range = summary.attribute_range
    attributes = ("" if attribute_range is None else str(attribute_range[0])
                  if attribute_range[0] == attribute_range[1]
                  else f"{attribute_range[0]}–{attribute_range[1]}")
    description = (f"<td>{html.escape(get_description_summary(summary.model.description))}</td>"
                   if has_description_column else "")
    return (f"<td>{summary.output_kind or ''}</td>"
            f'<td class="number">{summary.num_segments or ""}</td>'
            f'<td class="number">{attributes}</td>'
            f"{description}"
            f"<td>{html.escape(', '.join(summary.submitters))}</td>"
            f"<td>{format_utc_time(summary.updated_at)}</td>")


def _make_method_row_html(group: MethodGroup, num_columns: int) -> str:
    """Makes the row of a method above its models: its shown name as a link to the Method page,
    its method name if it differs, both cut off at the end of the row if long, the method name
    first, and its training and early stopping splits."""
    info = group.model_info
    splits = ""
    if info is not None:
        splits = f"trained on {format_splits(info.training_splits)}"
        if info.early_stopping_splits:
            splits += f", early stopping on {format_splits(info.early_stopping_splits)}"
        splits = f' <span class="muted">· {html.escape(splits)}</span>'
    method_name = ("" if group.shown_method_name == group.method_name else
                   f' <span class="muted truncated-first">'
                   f"({make_truncated_name_html(group.method_name)})</span>")
    link_html = make_method_link_html(group.dataset, group.method_name, group.shown_method_name)
    return (f'<tr class="method-row"><td colspan="{num_columns}"><div class="truncated-line">'
            f"<b>{make_truncated_name_html(group.shown_method_name, link_html)}</b>{method_name}"
            f"{splits}</div></td></tr>")


def make_models_table_html(groups: T.Sequence[MethodGroup], split_names: T.Sequence[str], *,
                           has_method_rows: bool = True, has_description_column: bool = True,
                           selected_model_ids: T.Collection[int] | None = None) -> str:
    """Makes the table of the models of methods of a dataset
    (`model_listing.summarize_models_by_method`), with a column per split. A model is labeled by
    its seed, with its full label as the tooltip, since the method row above it or the page names
    its method.

    Args:
        split_names: The splits of the dataset, the order of the split columns.
        has_method_rows: Whether the models of each method are under a row of the method, e.g.
            not on the page of the method.
        has_description_column: Whether the first line of each model's description is shown,
            e.g. not on the page of the method, which shows the description above the table.
        selected_model_ids: The models whose checkbox is checked, or None for no checkbox column.
            The cells of the column are `model-selection` (`styles.css`). See
            `MODEL_CHECKBOX_CHANGE_HANDLER` and `make_check_models_js`.
    """
    header_cells = [
        *([] if selected_model_ids is None else ['<th class="model-selection"></th>']),
        "<th>Seed</th>", *(f'<th class="mark">{html.escape(s)}</th>' for s in split_names),
        *(make_header_cell_html(c, c in _SUMMARY_NUMBER_COLUMNS) for c in _SUMMARY_COLUMNS
          if has_description_column or c != "Description")]
    rows = []
    for group in groups:
        if has_method_rows:
            rows.append(_make_method_row_html(group, len(header_cells)))
        for summary in group.models:
            model = summary.model
            checkbox_html = ("" if selected_model_ids is None else
                             f'<td class="model-selection"><input type="checkbox"'
                             f' {NATIVE_CHOICE_ATTRIBUTES} data-model-id="{model.id}"'
                             f'{" checked" if model.id in selected_model_ids else ""}></td>')
            label_html = (f"<span{make_title_attribute(model.shown_label)}>"
                          f"{make_model_link_html(model.id, model.seed_label)}</span>")
            rows.append(
                f"<tr>{checkbox_html}<td>{label_html}</td>"
                f"{''.join(_make_submission_cell_html(summary, s) for s in split_names)}"
                f"{_make_summary_cells_html(summary, has_description_column)}</tr>")
    return make_table_html_from_header("".join(header_cells), rows)


def make_check_models_js(selected_model_ids: T.Collection[int]) -> str:
    """Makes JavaScript that checks the checkboxes of the selected models in the tables of the
    page (`make_models_table_html`) and unchecks the others, e.g. after a selection by the
    server. Unlike new table HTML, this is not ignored by a `ui.html` whose content is
    unchanged, which it may be after a click on a checkbox."""
    return (f"const ids = new Set({json.dumps(sorted(selected_model_ids))});"
            f" document.querySelectorAll('input[data-model-id]')"
            f".forEach(c => c.checked = ids.has(Number(c.dataset.modelId)));")
