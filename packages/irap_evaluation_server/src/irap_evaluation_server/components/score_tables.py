"""HTML of the score tables of the Scores, Method and Model pages.

A value cell shows the value and, below it, its bootstrap interval, computed from the resampled
values of its row (`scoring_worker.ScoringWorker.get_method_resamples`). Its background is shaded by
how good the value is among the values that it is compared with, and it is bold if it is not
significantly worse than the best (`method_ranking.compute_highlights`). With a reference method,
the cells of the other methods also show their difference from it (`ReferenceComparisons`). The
notes below the tables are short, with links to the Methodology page (`methodology`).
"""

import dataclasses as dc
import html
import math
import typing as T

import irap_evaluation as ie
import numpy as np

from ..archive import get_description_summary
from ..label_counts import AttributeLabelCounts
from ..method_comparison import get_better_method
from ..method_ranking import (
    ATTRIBUTE_TABLE_METRIC_NAMES,
    RANKING_METRIC_NAMES,
    CellHighlight,
    ShownMetrics,
    compute_highlights,
    describe_split_use,
    get_method_split_use,
)
from ..method_scores import MethodScores, are_scored_on_same_set
from ..methodology import (
    COMPARISONS_SECTION,
    EVALUATION_SETS_SECTION,
    INTERVALS_SECTION,
    METRICS_SECTION,
)
from .formatting import (
    format_context_offsets,
    make_link_html,
    make_method_link_html,
    make_model_link_html,
    make_table_html,
    make_table_html_from_header,
    make_title_attribute,
    make_truncated_name_html,
)
from .routes import get_methodology_path
from .row_sorting import (
    RowSorting,
    SortColumn,
    make_metric_sort_column,
)

V = T.TypeVar("V")

#: Gets the value of a cell from the values of a row, e.g. its resampled values, None if they
#: lack it.
ValueSelector = T.Callable[[ie.MetricValues[V]], V | None]

#: The resampled values of a row (`scoring_worker.ScoringWorker.get_method_resamples`), or None.
RowResamples = ie.MetricValues[np.ndarray] | None

#: The mark of a method with a single model (`_is_single_model`), which `_SINGLE_MODEL_NOTE`
#: explains.
_SINGLE_MODEL_MARK = "single model"
_SINGLE_MODEL_NOTE = (f'"{_SINGLE_MODEL_MARK}": its intervals do not cover the variation between'
                      f" seeds.")

#: The metrics of `make_model_scores_html` and `make_method_scores_html`, e.g. for
#: `ScoringWorker.get_method_resamples`.
MODEL_SCORES_METRIC_NAMES = (*RANKING_METRIC_NAMES, *ATTRIBUTE_TABLE_METRIC_NAMES)


@dc.dataclass(frozen=True)
class ScoresScope:
    """What the scores of a table are of: a dataset split and an attribute subset, which the
    links of the table keep.

    Attributes:
        attributes_query: The attribute subset (`AttributeSubset.to_query`).
    """

    dataset: str
    split: str
    attributes_query: tuple[str, ...]

    def make_method_link_html(self, method_name: str, text: str) -> str:
        return make_method_link_html(self.dataset, method_name, text, self.split,
                                     self.attributes_query)

    def make_model_link_html(self, model_id: int, text: str) -> str:
        return make_model_link_html(model_id, text, self.split, self.attributes_query)


def select_average_value(metric_name: str) -> ValueSelector:
    return lambda values: values.averages.get(metric_name)


def select_attribute_value(metric_name: str, attribute: str) -> ValueSelector:
    return lambda values: values.per_attribute.get(metric_name, {}).get(attribute)


def _select_row_value(values: ie.MetricValues[V] | None, select_value: ValueSelector) -> V | None:
    return None if values is None else select_value(values)


def _compute_cell_interval(resamples: np.ndarray | None,
                           confidence: float) -> ie.BootstrapInterval | None:
    return None if resamples is None else ie.compute_percentile_interval(resamples, confidence)


def _compute_row_interval(resamples: RowResamples, select_value: ValueSelector,
                          confidence: float) -> ie.BootstrapInterval | None:
    return _compute_cell_interval(_select_row_value(resamples, select_value), confidence)


def _make_methodology_link_html(section_id: str, text: str) -> str:
    """Makes a link to a section of the Methodology page with an escaped text."""
    return make_link_html(get_methodology_path(section_id), text)


def _make_note_lines_html(*lines_html: str) -> str:
    return "".join(f'<div class="muted">{line}</div>' for line in lines_html)


def _make_different_sets_reason_html() -> str:
    """Explains why the methods of a table are not compared on the same segments."""
    segments = _make_methodology_link_html(EVALUATION_SETS_SECTION, "different segments")
    return f"since the methods are scored on {segments}"


def _make_shading_note(axis: T.Literal["column", "row"]) -> str:
    """Explains the shading (`CellHighlight.strength`).

    Args:
        axis: Along which the values are compared with each other.
    """
    return f"Darker: better in its {axis}."


def _make_interval_link_html(confidence: float) -> str:
    return _make_methodology_link_html(INTERVALS_SECTION, f"{confidence:.0%} confidence interval")


def make_model_scores_note_html(confidence: float) -> str:
    """Explains the intervals and the shading of `make_model_scores_html`."""
    return _make_note_lines_html(
        f"Below each value: its {_make_interval_link_html(confidence)}, which does not cover the"
        f" variation between seeds. {_make_shading_note('column')}")


def make_method_scores_note_html(confidence: float, num_models: int) -> str:
    """Explains the intervals and the shading of `make_method_scores_html`.

    Args:
        num_models: The number of models of the mean.
    """
    if num_models == 1:
        return make_model_scores_note_html(confidence)
    return _make_note_lines_html(
        f"Below each value: its {_make_interval_link_html(confidence)}. In the rows of the seeds,"
        f" it does not cover the variation between seeds. {_make_shading_note('column')}")


def _is_single_model(row: MethodScores) -> bool:
    """Returns whether the intervals of the row cover only one model (`_SINGLE_MODEL_NOTE`)."""
    return row.num_models == 1 and row.error is None


def format_metric_value(value: float, num_decimals: int) -> str:
    """Formats a value without a leading zero, e.g. '.123' or '-.123', or as 'NaN', '∞' or
    '-∞'."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    text = f"{value:.{num_decimals}f}"
    sign, unsigned = ("-", text[1:]) if text.startswith("-") else ("", text)
    return sign + unsigned[1:] if unsigned.startswith("0.") else text


def format_optional_metric_value(value: float | None, num_decimals: int) -> str:
    """Formats a value like `format_metric_value`, or None as '–', e.g. a metric that a model
    lacks."""
    return "–" if value is None else format_metric_value(value, num_decimals)


def format_signed_metric_value(value: float, num_decimals: int) -> str:
    """Formats a value like `format_metric_value`, with '+' before a positive value, e.g. a
    difference."""
    return ("+" if value > 0 else "") + format_metric_value(value, num_decimals)


def _get_interval_text(interval: ie.BootstrapInterval | None,
                       is_signed: bool = False) -> tuple[str, str]:
    """Returns the text of an interval, '–' for None, and a title that explains it ('' for
    none).

    Args:
        is_signed: Whether the bounds have a sign, e.g. those of a difference.
    """
    if interval is None:
        return "–", ""
    format_bound = format_signed_metric_value if is_signed else format_metric_value
    text = (f"{format_bound(interval.low, num_decimals=3)}–"
            f"{format_bound(interval.high, num_decimals=3)}")
    if not interval.is_partially_undefined:
        return text, ""
    return text + "*", (f"Number of resamples left out, since the value is undefined:"
                        f" {interval.num_undefined}.")


def _make_interval_html(interval: ie.BootstrapInterval | None) -> str:
    text, title = _get_interval_text(interval)
    return f'<div class="interval"{make_title_attribute(title)}>{text}</div>'


def make_value_cell_html(value: float | None, interval: ie.BootstrapInterval | None,
                         note: str = "", highlight: CellHighlight | None = None,
                         difference_html: str = "") -> str:
    """Makes a cell with the value and its interval, or '–' if the value is None.

    Args:
        note: Shown instead of the interval, e.g. 'not predicted'.
        difference_html: Shown below the interval, e.g. the difference from a reference method
            (`_make_difference_html`).
    """
    if value is None:
        return '<td class="number">–</td>'
    below = (f'<div class="interval">{html.escape(note)}</div>' if note
             else _make_interval_html(interval)) + difference_html
    text = format_metric_value(value, num_decimals=4)
    if highlight is not None and highlight.is_not_significantly_worse:
        text = f"<b>{text}</b>"
    # The shade is `--strength` in `styles.css`.
    style = (f' style="--strength: {highlight.strength:.2f}"'
             if highlight is not None and highlight.strength else "")
    return f'<td class="number"{style}>{text}{below}</td>'


# Comparisons with a reference method ##############################################################

@dc.dataclass(frozen=True)
class ReferenceComparisons:
    """The comparisons of the methods of a table with a reference method
    (`method_comparison.compare_with_reference`).

    Attributes:
        method_to_differences: Method name -> its differences from the reference, for the
            compared methods.
    """

    reference: MethodScores
    method_to_differences: T.Mapping[str, ie.MetricValues[ie.MetricDifference]]


def _make_comparison_note(comparisons: ReferenceComparisons, confidence: float) -> str:
    """Explains the differences from the reference method."""
    return (f"{_make_methodology_link_html(COMPARISONS_SECTION, 'Δ')}: the difference from"
            f" {html.escape(comparisons.reference.shown_method_name)}, with its {confidence:.0%}"
            f" confidence interval. better/worse: the interval excludes 0.")


def _make_difference_html(difference: ie.MetricDifference, metric_name: str) -> str:
    """Makes the difference of a value from that of the reference method, with its interval, and
    whether the value is clearly better or worse (`method_comparison.get_better_method`)."""
    interval_text, title = _get_interval_text(difference.interval, is_signed=True)
    better = get_better_method(difference.interval, metric_name)
    verdict = {"A": " <b>better</b>", "B": " <b>worse</b>", None: ""}[better]
    return (f'<div class="interval"{make_title_attribute(title)}>'
            f"Δ {format_signed_metric_value(difference.difference, 4)} ({interval_text})"
            f"{verdict}</div>")


def _make_method_label_html(row: MethodScores, comparisons: ReferenceComparisons | None,
                            scope: ScoresScope) -> str:
    """Makes the shown method name as a link to the Method page, cut off with a tooltip if it is
    long (`truncated-name` in `styles.css`), and below it, whether it is the reference method,
    used the split of the scores (`method_ranking.describe_split_use`) or has a single model."""
    marks = []
    if comparisons is not None and row.method_name == comparisons.reference.method_name:
        marks.append("reference")
    if split_use := describe_split_use(get_method_split_use(row, scope.split), scope.split):
        marks.append(split_use)
    if _is_single_model(row):
        marks.append(_SINGLE_MODEL_MARK)
    marks_html = (f'<div class="muted">({html.escape(", ".join(marks))})</div>' if marks
                  else "")
    link_html = scope.make_method_link_html(row.method_name, row.shown_method_name)
    return make_truncated_name_html(row.shown_method_name, link_html) + marks_html


# Tables ###########################################################################################

def _make_compared_cells_html(rows: T.Sequence[MethodScores],
                              row_resamples: T.Sequence[RowResamples],
                              select_value: ValueSelector, metric_name: str, confidence: float,
                              is_tested: bool, comparisons: ReferenceComparisons | None,
                              notes: T.Sequence[str] | None = None) -> list[str]:
    """Makes the cells of the values of rows that are compared with each other, with their
    intervals and differences from the reference method.

    Args:
        row_resamples: The resampled values of each row.
        select_value: Gets the value of the cells from the metrics, resampled values and
            differences of a row.
        metric_name: The metric of the value.
        is_tested: Whether the values are tested against the best (`compute_highlights`), which
            needs them on the same resamples, e.g. of methods on the same evaluation set.
        notes: The note of each cell (`make_value_cell_html`).
    """
    values = [_select_row_value(r.metrics, select_value) for r in rows]
    resamples = [_select_row_value(r, select_value) for r in row_resamples]
    highlights = compute_highlights(values, resamples if is_tested else None,
                                    ie.is_lower_better(metric_name))

    def get_difference_html(row: MethodScores) -> str:
        if comparisons is None:
            return ""
        difference = _select_row_value(comparisons.method_to_differences.get(row.method_name),
                                       select_value)
        return "" if difference is None else _make_difference_html(difference, metric_name)

    return [make_value_cell_html(v, _compute_cell_interval(r, confidence), note=n, highlight=h,
                                 difference_html=get_difference_html(row))
            for row, v, r, n, h in zip(rows, values, resamples, notes or [""] * len(rows),
                                       highlights, strict=True)]


def _make_seed_links_html(row: MethodScores, scope: ScoresScope) -> str:
    """Makes a link to each model of a row, labeled by its seed."""
    return ", ".join(scope.make_model_link_html(m.submission.model.id,
                                                m.submission.model.seed_label)
                     for m in row.scored_models)


def _make_model_links_html(row: MethodScores, scope: ScoresScope) -> str:
    """Makes the number of models, with a link to each, labeled by seed."""
    return f'{row.num_models} <span class="muted">({_make_seed_links_html(row, scope)})</span>'


def _make_row_cells_html(row: MethodScores, metric_cells: T.Iterable[str],
                         num_metrics: int) -> str:
    """Makes the metric cells of a row, or one cell with its error across them."""
    if row.metrics is None:
        return f'<td colspan="{num_metrics}" class="error">{html.escape(row.error or "")}</td>'
    return "".join(metric_cells)


def _are_tested(rows: T.Iterable[MethodScores]) -> bool:
    """Returns whether the values of the rows are tested against the best
    (`compute_highlights`), which needs the rows with values on the same evaluation set
    (`method_scores.are_scored_on_same_set`)."""
    return are_scored_on_same_set(m for r in rows if r.metrics is not None for m in r.scored_models)


def _make_method_notes_html(rows: T.Sequence[MethodScores], confidence: float,
                            axis: T.Literal["column", "row"], is_tested: bool,
                            comparisons: ReferenceComparisons | None) -> str:
    """Explains a table of methods: their values and intervals, the highlights, which are tested
    against the best if `is_tested` (`_are_tested`), the single models, and the differences from
    the reference method.

    Args:
        axis: Along which the values are compared with each other.
    """
    if is_tested:
        bold = (f"{_make_methodology_link_html(COMPARISONS_SECTION, 'Bold')}: not significantly"
                f" worse than the best.")
    else:
        bold = f"No bold, {_make_different_sets_reason_html()}."
    lines = [(f"Values: means over the seeds of each method. Below each: its"
              f" {_make_interval_link_html(confidence)}. {_make_shading_note(axis)} {bold}")]
    if any(_is_single_model(r) for r in rows):
        lines.append(_SINGLE_MODEL_NOTE)
    if comparisons is not None:
        lines.append(_make_comparison_note(comparisons, confidence))
    return _make_note_lines_html(*lines)


def _make_notes_html(row: MethodScores) -> str:
    notes = []
    if row.missing_attributes:
        title_attribute = make_title_attribute(", ".join(row.missing_attributes))
        notes.append(f"<span{title_attribute}>{len(row.missing_attributes)} not predicted</span>")
    first_scores = row.scored_models[0].scores
    if first_scores.evaluation_set == ie.MODEL_COMPATIBLE_SET_NAME:
        notes.append(f"context offsets {format_context_offsets(first_scores.context_offsets)}")
    return "<br>".join(notes)


def _make_sortable_metric_header_html(metric_name: str, is_sorted: bool) -> str:
    """Makes the header of a metric column with a `data-sort` attribute (`make_method_table_html`)
    and the description of the metric as its tooltip."""
    return (f'<th class="number sortable{" sorted" if is_sorted else ""}"'
            f' data-sort="{metric_name}"{make_title_attribute(ie.describe_metric(metric_name))}>'
            f'{metric_name}{" ▾" if is_sorted else ""}</th>')


def make_method_table_html(rows: T.Sequence[MethodScores],
                           method_to_resamples: T.Mapping[str, RowResamples], confidence: float,
                           metric_names: T.Sequence[str], sort_metric: str, scope: ScoresScope,
                           comparisons: ReferenceComparisons | None = None) -> str:
    """Makes a table of method rows × attribute averages.

    The metric headers have a `data-sort` attribute with the metric name, for a click handler.

    Args:
        method_to_resamples: Method name -> its resampled values of `metric_names`, or None.
        confidence: The confidence of the intervals.
        metric_names: The attribute averages of the columns, e.g. the shown ones
            (`ShownMetrics.average_names`).
        scope: For the links to the methods and models, and the marks of the methods that used
            the split.
        comparisons: The comparisons with a reference method, or None.
    """
    metric_headers = "".join(_make_sortable_metric_header_html(n, n == sort_metric)
                             for n in metric_names)
    header = (f"<th>Method</th><th>Models</th>{metric_headers}<th>Submitter</th><th>Notes</th>"
              f"<th>Description</th>")
    row_resamples = [method_to_resamples.get(r.method_name) for r in rows]
    is_tested = _are_tested(rows)
    metric_columns = [
        _make_compared_cells_html(rows, row_resamples, select_average_value(n), n, confidence,
                                  is_tested=is_tested, comparisons=comparisons)
        for n in metric_names]
    html_rows = []
    for row, metric_cells in zip(rows, zip(*metric_columns)):
        cells = _make_row_cells_html(row, metric_cells, len(metric_names))
        # Cut off, so that a long first line does not widen the table.
        description = make_truncated_name_html(get_description_summary(row.description))
        html_rows.append(f"<tr><td>{_make_method_label_html(row, comparisons, scope)}</td>"
                         f"<td>{_make_model_links_html(row, scope)}</td>{cells}"
                         f"<td>{html.escape(', '.join(row.submitters))}</td>"
                         f"<td>{_make_notes_html(row)}</td><td>{description}</td></tr>")
    return (make_table_html_from_header(header, html_rows)
            + _make_method_notes_html(rows, confidence, "column", is_tested, comparisons))


def get_per_attribute_table_metric_names(metric_name: str) -> tuple[str, str]:
    """Returns the metrics of `make_per_attribute_table_html` of `metric_name`, e.g. for
    `ScoringWorker.get_method_resamples`: the metric and its attribute average."""
    return metric_name, ie.get_attribute_average_name(metric_name)


@dc.dataclass(frozen=True)
class _LabelCountColumn:
    """A label count column of `make_per_attribute_table_html`.

    Attributes:
        key: Identifies the column in a row sort (`_get_label_count_sort_key`), e.g. in URLs, so
            that it does not change with the header.
        title: The tooltip of the header.
        reveals_classes: Whether the count reveals the class distribution, so that it is hidden
            like the per-class scores (`accounts.can_see_class_scores`).
    """

    key: str
    header: str
    title: str
    get_count: T.Callable[[AttributeLabelCounts], int | None]
    reveals_classes: bool


_LABEL_COUNT_COLUMNS = (
    _LabelCountColumn(
        "classes", "Class count", "The number of classes of the attribute that occur as a label"
                                  " in the evaluation set.",
        lambda c: c.num_labeled_classes, reveals_classes=True),
    _LabelCountColumn(
        "examples", "Example count", "The number of labeled examples: the segments of the"
                                     " evaluation set with a label of the attribute.",
        lambda c: c.num_labeled_segments, reveals_classes=False),
    _LabelCountColumn(
        "minority", "Minority-class example count",
        "The number of labeled examples outside the attribute's majority class, the class with"
        " the most labeled examples.",
        lambda c: c.num_minority_class_segments, reveals_classes=True),
    _LabelCountColumn(
        "smallest", "Smallest-class example count",
        "The number of labeled examples of the attribute's rarest class among those that occur"
        " as a label.",
        lambda c: c.num_smallest_class_segments, reveals_classes=True),
)


def _is_label_count_hidden(column: _LabelCountColumn, shows_class_counts: bool) -> bool:
    return column.reveals_classes and not shows_class_counts


def _make_label_count_cells_html(counts: AttributeLabelCounts | None,
                                 shows_class_counts: bool) -> str:
    """Makes the label count cells of an attribute, with '–' for counts that are None, undefined
    or hidden."""
    def format_count(column: _LabelCountColumn) -> str:
        is_hidden = _is_label_count_hidden(column, shows_class_counts)
        count = None if counts is None or is_hidden else column.get_count(counts)
        return "–" if count is None else str(count)

    return "".join(f'<td class="number">{format_count(c)}</td>' for c in _LABEL_COUNT_COLUMNS)


def _make_label_count_note(split: str,
                           attribute_to_label_counts: T.Mapping[str, AttributeLabelCounts] | None,
                           shows_class_counts: bool) -> str:
    """Explains the label count columns, and why some are '–' (`_make_label_count_cells_html`)."""
    columns = _make_methodology_link_html(
        METRICS_SECTION, ", ".join(c.header for c in _LABEL_COUNT_COLUMNS))
    if attribute_to_label_counts is None:
        return f"{columns}: not shown, {_make_different_sets_reason_html()}."
    note = (f"{columns}: counts of the labels of each attribute in the evaluation set, the same"
            f" for all methods.")
    if not shows_class_counts:
        *hidden_headers, last_hidden_header = (c.header for c in _LABEL_COUNT_COLUMNS
                                               if c.reveals_classes)
        note += (f" {', '.join(hidden_headers)} and {last_hidden_header} are hidden, since the"
                 f" labels of the {html.escape(split)} split are protected.")
    return note


def _get_label_count_sort_key(column: _LabelCountColumn) -> str:
    return f"count:{column.key}"


def _get_method_sort_key(row: MethodScores) -> str:
    return f"method:{row.method_name}"


def _get_per_attribute_table_sort_columns(
        rows: T.Iterable[MethodScores], metric_name: str,
        attribute_to_label_counts: T.Mapping[str, AttributeLabelCounts] | None,
        shows_class_counts: bool) -> dict[str, SortColumn[str]]:
    """Returns the columns of `make_per_attribute_table_html` that its attribute rows can be
    sorted by, by their keys in a row sort (`row_sorting`): the label counts that are shown,
    smallest first, and the methods with values of the metric, worst first. A hidden label count
    is left out, since the order of the attributes would reveal it.

    Args: See `make_per_attribute_table_html`, with `rows` without errors.
    """
    key_to_column = {}
    if attribute_to_label_counts is not None:
        for column in _LABEL_COUNT_COLUMNS:
            if not _is_label_count_hidden(column, shows_class_counts):
                key_to_column[_get_label_count_sort_key(column)] = SortColumn(
                    {a: column.get_count(c) for a, c in attribute_to_label_counts.items()},
                    is_descending_first=False)
    for row in rows:
        if (attribute_to_value := row.metrics.per_attribute.get(metric_name)) is not None:
            key_to_column[_get_method_sort_key(row)] = make_metric_sort_column(
                attribute_to_value, metric_name)
    return key_to_column


def make_per_attribute_table_html(
        rows: T.Sequence[MethodScores], method_to_resamples: T.Mapping[str, RowResamples],
        confidence: float, metric_name: str, scope: ScoresScope,
        attribute_to_label_counts: T.Mapping[str, AttributeLabelCounts] | None,
        shows_class_counts: bool, comparisons: ReferenceComparisons | None = None,
        attribute_sort: str = "") -> str:
    """Makes a table of attribute rows × method columns of one per-attribute metric, for the rows
    without an error, in their order. The first row has the attribute average of the metric.
    Before the method columns, label count columns describe the evaluation set of the methods
    (`label_counts.AttributeLabelCounts`).

    The attribute rows can be sorted by a label count or a method
    (`_get_per_attribute_table_sort_columns`), by a click on its header (`row_sorting`).

    Args:
        rows: Over the same attribute subset.
        method_to_resamples: Method name -> its resampled values of
            `get_per_attribute_table_metric_names(metric_name)`, or None.
        scope: For the links to the methods, and the marks of the methods that used the split.
        attribute_to_label_counts: Attribute -> its label counts in the evaluation set of the
            rows, for the attributes of the rows, or None if the rows are scored on different
            sets.
        shows_class_counts: Whether the label counts that reveal the class distribution are
            shown (`accounts.can_see_class_scores`).
        comparisons: The comparisons with a reference method, or None.
        attribute_sort: The row sort of the attribute rows (`row_sorting`), '' for the order of
            the attribute subset.

    Raises:
        ValueError: If there are no rows, or they are over different attribute subsets.
    """
    if len(subsets := {r.attributes for r in rows}) != 1:
        raise ValueError(f"The rows must be over one attribute subset, not {len(subsets)}.")
    [attributes] = subsets
    rows = [r for r in rows if r.metrics is not None]
    sorting = RowSorting(
        _get_per_attribute_table_sort_columns(rows, metric_name, attribute_to_label_counts,
                                              shows_class_counts), attribute_sort)
    row_resamples = [method_to_resamples.get(r.method_name) for r in rows]
    is_tested = _are_tested(rows)
    average_name = ie.get_attribute_average_name(metric_name)
    average_cells = _make_compared_cells_html(
        rows, row_resamples, select_average_value(average_name), average_name, confidence,
        is_tested=is_tested, comparisons=comparisons)
    html_rows = [f"<tr><td{make_title_attribute(ie.describe_metric(average_name))}>Average</td>"
                 f'<td colspan="{len(_LABEL_COUNT_COLUMNS)}"></td>'
                 f"{''.join(average_cells)}</tr>"]
    for attribute in sorting.sort(attributes):
        label_count_cells = _make_label_count_cells_html(
            None if attribute_to_label_counts is None else attribute_to_label_counts[attribute],
            shows_class_counts)
        cells = _make_compared_cells_html(
            rows, row_resamples, select_attribute_value(metric_name, attribute), metric_name,
            confidence, is_tested=is_tested, comparisons=comparisons,
            notes=["not predicted" if attribute in r.missing_attributes else "" for r in rows])
        html_rows.append(f"<tr><td>{html.escape(attribute)}</td>{label_count_cells}"
                         f"{''.join(cells)}</tr>")
    label_count_headers = "".join(
        sorting.make_column_header_html(html.escape(c.header), c.title,
                                        _get_label_count_sort_key(c), is_vertical=True)
        for c in _LABEL_COUNT_COLUMNS)
    method_headers = "".join(
        sorting.make_column_header_html(_make_method_label_html(r, comparisons, scope), "",
                                        _get_method_sort_key(r), is_vertical=True) for r in rows)
    return (make_table_html_from_header(
                f"{sorting.make_unsorted_header_html('Attribute')}{label_count_headers}"
                f"{method_headers}", html_rows)
            + _make_note_lines_html(
                html.escape(ie.describe_metric(metric_name)),
                _make_label_count_note(scope.split, attribute_to_label_counts, shows_class_counts))
            + _make_method_notes_html(rows, confidence, "row", is_tested, comparisons))


def _make_row_value_cell_html(row: MethodScores, resamples: RowResamples, average_name: str,
                              confidence: float) -> str:
    """Makes the cell of an attribute average of a row, with its interval, without shading."""
    select_value = select_average_value(average_name)
    return make_value_cell_html(_select_row_value(row.metrics, select_value),
                                _compute_row_interval(resamples, select_value, confidence))


def _get_nums_invalid(scores: MethodScores, attributes: T.Iterable[str]) -> list[int]:
    """Returns the number of invalid predictions of each model for each of the attributes."""
    return [m.scores.num_invalid[a] for m in scores.scored_models for a in attributes]


def _format_num_invalid(scores: MethodScores, attributes: T.Sequence[str]) -> str:
    """Formats the number of invalid predictions of the attributes, the mean over the models and
    the attributes if there are several."""
    counts = _get_nums_invalid(scores, attributes)
    return str(counts[0]) if len(counts) == 1 else f"{sum(counts) / len(counts):.1f}"


#: The key of the invalid predictions column of `_make_attribute_scores_table_html` in a row sort
#: (`row_sorting`). The keys of its metric columns are the metric names.
_NUM_INVALID_SORT_KEY = "invalid"


def _make_attribute_scores_table_html(scores: MethodScores, resamples: RowResamples,
                                     confidence: float, metric_names: T.Sequence[str],
                                     attribute_sort: str = "") -> str:
    """Makes a table of the metrics and invalid predictions of each attribute of a subset, of one
    model or the mean over the models of a method. The attributes are compared with each other
    in each metric column, without tests against the best (bold), which are meant for comparing
    methods. The first row has the attribute averages and the mean number of invalid predictions,
    and is not shaded.

    The attribute rows can be sorted by a metric, worst first, or by the invalid predictions,
    largest first, by a click on its header (`row_sorting`).

    Args:
        scores: `method_scores.compute_method_scores`, without an error.
        resamples: Its resampled values of `metric_names`, or None.
        confidence: The confidence of the intervals.
        metric_names: The metrics of the columns, of `method_ranking.ATTRIBUTE_TABLE_METRIC_NAMES`,
            e.g. the shown ones (`ShownMetrics.attribute_table_names`). Those that the scores
            lack, e.g. the manual review proportion of hard predictions, get no column.
        attribute_sort: The row sort of the attribute rows (`row_sorting`), '' for the order of
            the attribute subset.

    Raises:
        ValueError: If `scores` has an error.
    """
    if (metrics := scores.metrics) is None:
        raise ValueError(f"The scores of {scores.method_name} have an error: {scores.error}")
    metric_names = [n for n in metric_names if n in metrics.per_attribute]
    sorting = RowSorting({
        **{n: make_metric_sort_column(metrics.per_attribute[n], n) for n in metric_names},
        _NUM_INVALID_SORT_KEY: SortColumn(
            {a: sum(_get_nums_invalid(scores, [a])) for a in scores.attributes},
            is_descending_first=True)}, attribute_sort)
    attributes = sorting.sort(scores.attributes)
    metric_columns = []
    for n in metric_names:
        values = [metrics.per_attribute[n].get(a) for a in attributes]
        highlights = compute_highlights(values, None, ie.is_lower_better(n))
        intervals = [_compute_row_interval(resamples, select_attribute_value(n, a), confidence)
                     for a in attributes]
        metric_columns.append([make_value_cell_html(v, i, highlight=h)
                               for v, i, h in zip(values, intervals, highlights)])
    average_cells = [_make_row_value_cell_html(scores, resamples,
                                               ie.get_attribute_average_name(n), confidence)
                     for n in metric_names]
    average_title = make_title_attribute(ie.describe_attribute_average("the metric"))
    attribute_rows = [f"<tr><td{average_title}>Average</td>{''.join(average_cells)}"
                      f'<td class="number">{_format_num_invalid(scores, attributes)}</td></tr>']
    for attribute, metric_cells in zip(attributes, zip(*metric_columns)):
        is_missing = attribute in scores.missing_attributes
        num_invalid = (f"{_format_num_invalid(scores, [attribute])}"
                       f"{' (not predicted)' if is_missing else ''}")
        attribute_rows.append(f"<tr><td>{html.escape(attribute)}</td>{''.join(metric_cells)}"
                              f'<td class="number">{num_invalid}</td></tr>')
    invalid_header = ("Invalid predictions" if scores.num_models == 1
                      else "Invalid predictions (mean)")
    headers = [sorting.make_unsorted_header_html("Attribute"),
               *(sorting.make_column_header_html(n, ie.describe_metric(n), n)
                 for n in metric_names),
               sorting.make_column_header_html(invalid_header, "", _NUM_INVALID_SORT_KEY)]
    return make_table_html_from_header("".join(headers), attribute_rows)


def make_model_scores_html(scores: MethodScores, resamples: RowResamples, confidence: float,
                           shown_metrics: ShownMetrics, attribute_sort: str = "") -> str:
    """Makes the tables of the scores of one model over a subset of the attributes: the attribute
    averages, and the metrics and invalid predictions of each attribute
    (`_make_attribute_scores_table_html`). `make_model_scores_note_html` explains them.

    Args:
        scores: `method_scores.compute_method_scores` of the model alone.
        resamples: Its resampled values of `MODEL_SCORES_METRIC_NAMES`, or None.
        confidence: The confidence of the intervals.
        shown_metrics: The metrics of the tables, those that the model has.
        attribute_sort: The row sort of the attribute rows (`row_sorting`).
    """
    if scores.metrics is None:
        return f'<div class="error">{html.escape(scores.error or "")}</div>'
    average_rows = [f"<tr><td{make_title_attribute(ie.describe_metric(n))}>{n}</td>"
                    f"{_make_row_value_cell_html(scores, resamples, n, confidence)}</tr>"
                    for n in shown_metrics.average_names if n in scores.metrics.averages]
    average_table = make_table_html(["Average", "Value"], average_rows, number_headers=["Value"])
    attribute_table = _make_attribute_scores_table_html(
        scores, resamples, confidence, shown_metrics.attribute_table_names, attribute_sort)
    return f'<div class="score-tables">{average_table}{attribute_table}</div>'


def make_method_scores_html(method_scores: MethodScores, method_resamples: RowResamples,
                            seed_scores: T.Sequence[MethodScores],
                            seed_resamples: T.Sequence[RowResamples], confidence: float,
                            shown_metrics: ShownMetrics, scope: ScoresScope,
                            attribute_sort: str = "") -> str:
    """Makes the tables of the scores of a method over a subset of the attributes: the attribute
    averages of the mean over its models and of each model alone (`_make_seed_table_html`), and
    the metrics and invalid predictions of each attribute of the mean
    (`_make_attribute_scores_table_html`), unless the mean has an error.
    `make_method_scores_note_html` explains them.

    Args:
        method_scores: `method_scores.compute_method_scores` of the models of the method.
        method_resamples: Its resampled values of `MODEL_SCORES_METRIC_NAMES`, or None.
        seed_scores: `method_scores.compute_method_scores` of each model alone, in the order of the
            rows.
        seed_resamples: The resampled values of each of `seed_scores` of the shown attribute
            averages, or None.
        confidence: The confidence of the intervals.
        shown_metrics: The metrics of the tables.
        scope: For the links to the models.
        attribute_sort: The row sort of the attribute rows (`row_sorting`).
    """
    seed_table = _make_seed_table_html(method_scores, method_resamples, seed_scores,
                                       seed_resamples, confidence,
                                       shown_metrics.average_names, scope)
    tables_html = f"<div>{seed_table}</div>"
    if method_scores.metrics is not None:
        attribute_table = _make_attribute_scores_table_html(
            method_scores, method_resamples, confidence, shown_metrics.attribute_table_names,
            attribute_sort)
        tables_html += f'<div><div class="muted">Mean over the seeds</div>{attribute_table}</div>'
    return f'<div class="score-tables">{tables_html}</div>'


def _make_seed_table_html(method_scores: MethodScores, method_resamples: RowResamples,
                          seed_scores: T.Sequence[MethodScores],
                          seed_resamples: T.Sequence[RowResamples], confidence: float,
                          metric_names: T.Sequence[str], scope: ScoresScope) -> str:
    """Makes a table of the mean over the models of a method and of each model alone (labeled by
    seed, with a link) × the attribute averages `metric_names`. The models are compared with each
    other in each column, without tests against the best (bold), which are meant for comparing
    methods. The mean is not shaded.

    Args: See `make_method_scores_html`.
    """
    mean_cells = [_make_row_value_cell_html(method_scores, method_resamples, n, confidence)
                  for n in metric_names]
    seed_columns = [
        _make_compared_cells_html(seed_scores, seed_resamples, select_average_value(n), n,
                                  confidence, is_tested=False, comparisons=None)
        for n in metric_names]
    num_metrics = len(metric_names)
    html_rows = [f"<tr><td>Mean</td>"
                 f"{_make_row_cells_html(method_scores, mean_cells, num_metrics)}</tr>"]
    html_rows += [f"<tr><td>{_make_seed_links_html(row, scope)}</td>"
                  f"{_make_row_cells_html(row, metric_cells, num_metrics)}</tr>"
                  for row, metric_cells in zip(seed_scores, zip(*seed_columns), strict=True)]
    return make_table_html(["Seed", *metric_names], html_rows, metric_names,
                           header_titles={n: ie.describe_metric(n) for n in metric_names})
