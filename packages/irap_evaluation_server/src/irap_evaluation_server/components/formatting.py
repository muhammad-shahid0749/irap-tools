"""Display texts and HTML fragments shared by the pages."""

import html
import typing as T
from datetime import datetime

import irap_evaluation as ie

from ..archive import ActionLogEntry, Model, Submission
from .routes import get_method_path, get_model_path

#: The display name of each evaluation set (`datasets.EVALUATION_SET_NAMES`).
EVALUATION_SET_LABELS = {ie.REFERENCE_SET_NAME: "Reference set", ie.FULL_SET_NAME: "Full set",
                         ie.MODEL_COMPATIBLE_SET_NAME: "Model-compatible set"}

_ACTION_DETAIL_LABELS = {"file_name": "File", "archive_member": "File in the archive",
                         "notes": "Notes", "original_seed": "Seed in the file",
                         "uploaded_file_sha256": "SHA-256 of the uploaded file",
                         "replaced_submission_id": "Replaced file",
                         "members": "Members", "segment_combination": "Combined segments",
                         "attribute_combination": "Combined attributes",
                         "old_description": "Old description", "description": "Description",
                         "old_display_name": "Old display name", "display_name": "Display name",
                         "split": "Split", "by": "Part of"}


def format_utc_time(time: datetime) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC")


def format_splits(splits: T.Sequence[str] | None) -> str:
    """Formats training or early stopping splits (`irap_evaluation.ModelInfo`): 'unknown' for
    None, 'none' for ()."""
    if splits is None:
        return "unknown"
    return ", ".join(splits) or "none"


def format_training_split_rows(info: ie.ModelInfo | None) -> dict[str, str]:
    """Formats the training and early stopping splits of a model as rows of details
    (`create_key_values`), '–' without model info."""
    return {"Training splits": "–" if info is None else format_splits(info.training_splits),
            "Early stopping splits": ("–" if info is None
                                      else format_splits(info.early_stopping_splits))}


def format_context_offsets(offsets: T.Sequence[int] | None) -> str:
    """Formats context offsets: 'unknown' for None."""
    return "unknown" if offsets is None else ", ".join(map(str, offsets))


def make_split_options(splits: T.Iterable[str],
                       split_to_segment_ids: T.Mapping[str, T.Sequence[str]]) -> dict[str, str]:
    """Makes the options of a split select: each split with its number of segments.

    Args:
        split_to_segment_ids: Has the splits, e.g. `irap_data.metadata.IRAPMetadata.splits`.
    """
    return {s: f"{s} ({len(split_to_segment_ids[s])})" for s in splits}


def make_evaluation_set_options(
        name_to_num_segments: T.Mapping[str, int],
        labels: T.Mapping[str, str] = EVALUATION_SET_LABELS) -> dict[str, str]:
    """Makes the options of an evaluation set select: the display names, with the numbers of
    segments of the sets that have one, i.e. not of the model-compatible set, which depends on
    the model.

    Args:
        name_to_num_segments: Evaluation set name -> its number of segments.
        labels: Evaluation set name -> its display name.
    """
    return {n: f"{l} ({name_to_num_segments[n]})" if n in name_to_num_segments else l
            for n, l in labels.items()}


def format_evaluation_set_title(name: str, context_offsets_text: str) -> str:
    """Formats the title of the scores on an evaluation set, with the context offsets
    (`format_context_offsets`) on the model-compatible set."""
    if name in ie.SHARED_EVALUATION_SET_NAMES:
        return EVALUATION_SET_LABELS[name]
    return f"{EVALUATION_SET_LABELS[name]}, context offsets {context_offsets_text}"


def make_link_html(path: str, text: str) -> str:
    return f'<a href="{html.escape(path)}">{html.escape(text)}</a>'


def make_model_link_html(model_id: int, text: str, split: str = "",
                         attributes: T.Sequence[str] = ()) -> str:
    """Makes a link to the Model page (`routes.get_model_path`) with an escaped text."""
    return make_link_html(get_model_path(model_id, split, attributes), text)


def make_method_link_html(dataset: str, method_name: str, text: str, split: str = "",
                          attributes: T.Sequence[str] = ()) -> str:
    """Makes a link to the Method page (`routes.get_method_path`) with an escaped text."""
    return make_link_html(get_method_path(dataset, method_name, split, attributes), text)


def make_unscored_files_html(unscored: T.Iterable[tuple[Submission, str]],
                             get_label: T.Callable[[Submission], str]) -> str:
    """Makes a line per submission without current scores, with a link to its Model page and the
    reason (`scoring_worker.SubmissionScorings.list_unscored`)."""
    return "".join(f"<div>{make_truncated_model_link_html(s.model.id, get_label(s), s.split)}:"
                   f" {html.escape(reason)}</div>" for s, reason in unscored)


def make_truncated_name_html(name: str, content_html: str | None = None) -> str:
    """Makes a `truncated-name` (`styles.css`) of the escaped name, or of `content_html`, e.g. a
    link with the name as its text, with the name as its tooltip."""
    escaped_name = html.escape(name)
    content_html = escaped_name if content_html is None else content_html
    return f'<div class="truncated-name" title="{escaped_name}">{content_html}</div>'


def make_truncated_model_link_html(model_id: int, label: str, split: str = "") -> str:
    """Makes a link to the Model page (`make_model_link_html`) whose text is a model label, cut
    off if it is long (`make_truncated_name_html`)."""
    return make_truncated_name_html(label, make_model_link_html(model_id, label, split))


def make_code_spans_html(text: str) -> str:
    """Escapes `text` and puts each span between backticks in `<code>`, e.g. a file name in a
    note.

    Raises:
        ValueError: If a backtick is unpaired.
    """
    parts = text.split("`")
    if len(parts) % 2 == 0:
        raise ValueError(f"Unpaired backtick in {text!r}.")
    return "".join(f"<code>{html.escape(p)}</code>" if i % 2 == 1 else html.escape(p)
                   for i, p in enumerate(parts))


def make_table_html(header_cells: T.Sequence[str], rows: T.Sequence[str],
                    number_headers: T.Collection[str] = (),
                    header_titles: T.Mapping[str, str] | None = None) -> str:
    """Makes a `data-table` (`styles.css`) of escaped header texts and `<tr>` rows.

    Args:
        number_headers: The headers of the columns of `number` cells, which are right-aligned like
            the cells.
        header_titles: Header -> its tooltip, e.g. a description of a metric.

    Raises:
        ValueError: If a header in `number_headers` or `header_titles` is not in `header_cells`.
    """
    header_titles = header_titles or {}
    if unknown := (set(number_headers) | set(header_titles)) - set(header_cells):
        raise ValueError(f"Headers not in the header: {sorted(unknown)}.")
    return make_table_html_from_header(
        "".join(make_header_cell_html(c, c in number_headers, header_titles.get(c, ""))
                for c in header_cells), rows)


def make_data_click_js_handler(data_attribute: str) -> str:
    """Makes the JavaScript handler of a click on an element of many clickable items, e.g. the
    buttons of an HTML list, which emits the value of `data_attribute`, e.g. 'data-segment', of
    the clicked item, so that one handler serves all items."""
    return (f"(e) => {{ const item = e.target.closest('[{data_attribute}]');"
            f" if (item) emit(item.getAttribute('{data_attribute}')); }}")


def make_title_attribute(title: str) -> str:
    """Makes a ` title="…"` attribute of an escaped tooltip, or '' if `title` is ''."""
    return f' title="{html.escape(title)}"' if title else ""


def make_header_cell_html(text: str, is_number: bool = False, title: str = "") -> str:
    """Makes a `<th>` of an escaped text, right-aligned like the cells if it is the header of
    `number` cells.

    Args:
        title: The tooltip, e.g. a description of a metric, or '' for none.
    """
    class_attribute = ' class="number"' if is_number else ""
    return f"<th{class_attribute}{make_title_attribute(title)}>{html.escape(text)}</th>"


def make_table_html_from_header(header_html: str, rows: T.Sequence[str]) -> str:
    """Makes a `data-table` (`styles.css`) of `<th>` header cells and `<tr>` rows."""
    return (f'<table class="data-table"><thead><tr>{header_html}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def make_ensemble_members_html(members: T.Sequence[T.Mapping[str, T.Any]]) -> str:
    """Formats the members of an ensemble from the details of its action (`ensembles`), with
    links to their models."""
    return ", ".join(f"{make_truncated_model_link_html(m['model_id'], m['label'])}"
                     f" #{m['submission_id']} (weight {m['weight']:g})" for m in members)


def _make_action_details_html(details: T.Mapping[str, T.Any]) -> str:
    def to_html(key: str, value: T.Any) -> str:
        if key == "members":
            return make_ensemble_members_html(value)
        if key == "replaced_submission_id":
            return f"#{value}"
        if key == "by":
            return _ACTION_TEXTS[value]
        if isinstance(value, list):
            return "<br>".join(html.escape(str(v)) for v in value) or "–"
        return html.escape("–" if value in (None, "") else str(value))

    return "".join(f'<div><span class="muted">{html.escape(_ACTION_DETAIL_LABELS.get(k, k))}:'
                   f"</span> {to_html(k, v)}</div>" for k, v in details.items())


#: The text of each `archive.ActionKind` and of the action that did another, its detail "by"
#: (e.g. `ModelArchive.delete_method`), in the action log.
_ACTION_TEXTS = {"upload": "upload", "ensemble": "ensemble", "delete": "delete file",
                 "delete_model": "delete model", "edit_description": "edit description",
                 "edit_method_display_name": "edit method display name",
                 "delete_method": "delete method"}


def make_action_table_html(entries: T.Sequence[ActionLogEntry],
                           model_id_to_model: T.Mapping[int, Model]) -> str:
    """Makes the table of actions, with links to their models.

    Args:
        model_id_to_model: Has the models that are not deleted, e.g. all of them.
    """
    def make_row(entry: ActionLogEntry) -> str:
        model = model_id_to_model.get(entry.model_id)
        model_html = (make_truncated_name_html(f"{entry.model_label} (deleted)") if model is None
                      else make_truncated_model_link_html(model.id, model.shown_label))
        submission = "" if entry.submission_id is None else f"#{entry.submission_id}"
        return (f"<tr><td>{format_utc_time(entry.time)}</td><td>{html.escape(entry.actor)}</td>"
                f"<td>{_ACTION_TEXTS[entry.action]}</td><td>{model_html}</td>"
                f'<td>{submission}</td><td class="breaks-anywhere">'
                f"{_make_action_details_html(entry.details)}</td></tr>")

    return make_table_html(["Time", "Who", "Action", "Model", "File", "Details"],
                           [make_row(e) for e in entries])
