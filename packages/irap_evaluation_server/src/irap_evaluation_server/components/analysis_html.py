"""HTML of the Analysis page: the tabs of its panels, the confusion matrix, the class table, the
classes of all attributes, the review curve, the segments of a matrix cell or the missed segments
of a class, and the details of a segment (see `prediction_analysis`).

The matrix, the table of the classes of all attributes and the segment list are single HTML
elements with `data-cell`, `data-class-row` and `data-segment` attributes for one delegated click
handler each, since hundreds of NiceGUI elements would be slow.
"""

import dataclasses as dc
import html
import typing as T

import irap_evaluation as ie
import numpy as np

from ..class_analysis import ClassDiscrimination, ClassScores
from ..prediction_analysis import SegmentDetail
from .analysis_view import format_cell
from .deck_map import OUTCOME_COLORS
from .formatting import (
    make_header_cell_html,
    make_table_html,
    make_table_html_from_header,
    make_title_attribute,
    make_truncated_name_html,
)
from .row_sorting import RowSorting, SortColumn, make_metric_sort_column
from .score_tables import format_metric_value, format_optional_metric_value


def _format_class(irap_code: int, class_name: str) -> str:
    return f"{irap_code}. {class_name}"


def _get_cell_color(label_index: int, predicted_index: int, num_classes: int) -> tuple[int, ...]:
    if predicted_index == num_classes:
        return OUTCOME_COLORS["invalid"]
    return OUTCOME_COLORS["correct" if label_index == predicted_index else "wrong"]


def make_confusion_matrix_html(confusion_matrix: np.ndarray, class_names: T.Sequence[str],
                               irap_codes: T.Sequence[int],
                               selected_cell: tuple[int, int | None] | None) -> str:
    """Makes the matrix as a grid of cells, shaded by their share of the row, with their counts
    scaled to fit the cells.

    The cells and the labels of the rows with missed segments have a `data-cell` attribute
    (`analysis_view.format_cell`) for a click handler. A row label selects the missed segments
    of its class.

    Args:
        confusion_matrix: (K, K + 1) counts with the invalid predictions in the last column (see
            `prediction_analysis.AttributeOutcomes.confusion_matrix`).
        selected_cell: See `analysis_view.AnalysisView.cell`.
    """
    num_classes = len(class_names)
    parts = ['<div class="header">label ↓ predicted →</div>']
    parts += [f'<div class="header center" title="{html.escape(n)}">{c}</div>'
              for c, n in zip(irap_codes, class_names, strict=True)]
    parts.append('<div class="header center" title="Invalid predictions">inv.</div>')
    for i, (code, name) in enumerate(zip(irap_codes, class_names, strict=True)):
        label = html.escape(_format_class(code, name))
        row_sum = int(confusion_matrix[i].sum())
        num_missed = row_sum - int(confusion_matrix[i, i])
        is_missed_selected = selected_cell == (i, None)
        if num_missed:
            parts.append(
                f'<div class="header row-label{" selected" if is_missed_selected else ""}"'
                f' data-cell="{format_cell((i, None))}"'
                f' title="{label} – click to list its {num_missed} missed segments">{label}</div>')
        else:
            parts.append(f'<div class="header" title="{label}">{label}</div>')
        for j in range(num_classes + 1):
            count = int(confusion_matrix[i, j])
            predicted = "invalid" if j == num_classes else _format_class(irap_codes[j],
                                                                          class_names[j])
            share = count / row_sum if row_sum else 0
            is_selected = (selected_cell == (i, j)
                           or (is_missed_selected and j != i and count > 0))
            classes = "cell" + (" selected" if is_selected else "")
            attributes = (f' data-cell="{format_cell((i, j))}"' if count else " disabled")
            color = ", ".join(map(str, _get_cell_color(i, j, num_classes)))
            parts.append(
                f'<div class="{classes}"{attributes}'
                f' style="background: color-mix(in srgb, rgb({color}) {round(share * 100)}%,'
                f' white)" title="{label} → {html.escape(predicted)}: {count}">'
                f'<span>{count}</span></div>')
    num_digits = len(str(int(confusion_matrix.max())))
    return (f'<div class="confusion-matrix" style="--num-count-columns: {num_classes + 1};'
            f' --num-digits: {num_digits}">{"".join(parts)}</div>')


def make_panel_tabs_html(panel_to_title: T.Mapping[str, str], selected_panel: str) -> str:
    """Makes the tabs of the panels of which one is shown, with a `data-panel` attribute for a
    click handler.

    Args:
        panel_to_title: The key of a panel -> its title.
    """
    tabs = "".join(
        f'<button type="button" data-panel="{html.escape(panel)}"'
        f'{" class=selected" if panel == selected_panel else ""}>{html.escape(title)}</button>'
        for panel, title in panel_to_title.items())
    return f'<div class="panel-tabs">{tabs}</div>'


#: The headers of the one-vs-rest AUROC and informedness of a class (`ClassDiscrimination`), in
#: `make_class_table_html` and `make_class_analysis_html`.
AUROC_HEADER, INFORMEDNESS_HEADER = "AUROC", "J"
_SUPPORT_TITLE = "The number of segments labeled with the class."
_AUROC_TITLE = ("One-vs-rest AUROC: the probability that a segment of the class gets a higher"
                " predicted probability of it than a segment of another class. 0.5 by chance, at"
                " any class size.")
_INFORMEDNESS_TITLE = ("Informedness (Youden's J): the recall of the class minus the proportion"
                       " of the other classes' segments predicted as it. 0 by chance, at any class"
                       " size.")


def _make_discrimination_cells_html(auroc: float | None, informedness: float | None) -> str:
    """Makes the cells of the one-vs-rest AUROC and informedness of a class."""
    return "".join(f'<td class="number">{format_optional_metric_value(v, 3)}</td>'
                   for v in (auroc, informedness))


def make_class_table_html(class_report: T.Mapping[str, T.Any], confusion_matrix: np.ndarray,
                          discrimination: ClassDiscrimination) -> str:
    """Makes a table of the support, precision, recall, F1, one-vs-rest AUROC and informedness
    of each class, and its invalid predictions.

    Args:
        class_report: The attribute's entry of `classes` in an
            `evaluation_report.to_json_dict` document, whose undefined values are None.
        confusion_matrix: See `make_confusion_matrix_html`.
        discrimination: `class_analysis.compute_class_discrimination` of the attribute.
    """
    def make_discrimination_cells_html(k: int) -> str:
        return _make_discrimination_cells_html(
            *(None if np.isnan(v) else float(v)
              for v in (discrimination.auroc[k], discrimination.informedness[k])))

    rows = []
    for k, (code, name) in enumerate(zip(class_report["irap_codes"], class_report["values"],
                                         strict=True)):
        cells = "".join(f'<td class="number">'
                        f"{format_optional_metric_value(class_report[key][k], 3)}</td>"
                        for key in ("precision", "recall", "f1"))
        rows.append(f"<tr><td>{html.escape(_format_class(code, name))}</td>"
                    f'<td class="number">{class_report["support"][k]}</td>{cells}'
                    f"{make_discrimination_cells_html(k)}"
                    f'<td class="number">{int(confusion_matrix[k, -1])}</td></tr>')
    metric_names = ("P", "R", "F1")
    header_titles = {"Support": _SUPPORT_TITLE,
                     **{n: ie.describe_metric(n) for n in metric_names},
                     AUROC_HEADER: _AUROC_TITLE, INFORMEDNESS_HEADER: _INFORMEDNESS_TITLE}
    number_headers = ["Support", *metric_names, AUROC_HEADER, INFORMEDNESS_HEADER, "Invalid"]
    return make_table_html(["Class", *number_headers], rows, number_headers, header_titles)


def _make_class_sort_columns(class_scores: T.Sequence[ClassScores]
                             ) -> dict[str, SortColumn[int]]:
    """Returns the columns of `make_class_analysis_html` that its rows can be sorted by, by
    their keys in a row sort. A row is the index of its class in `class_scores`."""
    def map_rows(get_value: T.Callable[[ClassScores], float | None]) -> dict[int, float | None]:
        return {i: get_value(c) for i, c in enumerate(class_scores)}

    return {
        "class": SortColumn({i: i for i in range(len(class_scores))}, False),
        "support": SortColumn(map_rows(lambda c: c.support), is_descending_first=True),
        "f1": make_metric_sort_column(map_rows(lambda c: c.f1), "F1"),
        "auroc": SortColumn(map_rows(lambda c: c.auroc), False),
        "informedness": SortColumn(map_rows(lambda c: c.informedness), False),
        "missed": SortColumn(map_rows(lambda c: c.num_missed), is_descending_first=True)}


def make_class_analysis_html(class_scores: T.Sequence[ClassScores],
                             shown_indices: T.Iterable[int], class_sort: str,
                             selected_index: int | None) -> str:
    """Makes a table of classes of several attributes with their support, F1, one-vs-rest AUROC
    and informedness, and what they are confused with.

    The rows have a `data-class-row` attribute with the index of the class in `class_scores`,
    for a click handler, and can be sorted by a click on a column header (`row_sorting`).

    Args:
        class_scores: `class_analysis.list_class_scores` of a model.
        shown_indices: The indices of the shown classes in `class_scores`, e.g. those of the
            selected attributes.
        class_sort: The row sort of the classes.
        selected_index: The index of the class whose row is marked, or None.
    """
    sorting = RowSorting(_make_class_sort_columns(class_scores), class_sort, rows_noun="classes")
    rows = []
    for i in sorting.sort(list(shown_indices)):
        scores = class_scores[i]
        confusion = scores.largest_confusion
        confusion_cell = "<td></td>"
        if confusion is not None:
            title = (f"Of the {scores.num_missed} missed segments, {confusion.num_segments} are"
                     f" predicted as {_format_class(confusion.irap_code, confusion.class_name)}.")
            confusion_cell = (f'<td class="number"{make_title_attribute(title)}>'
                              f'{confusion.irap_code}'
                              f' <span class="muted">({confusion.num_segments})</span></td>')
        rows.append(
            f'<tr data-class-row="{i}"{" class=selected" if i == selected_index else ""}>'
            f"<td><div>{html.escape(_format_class(scores.irap_code, scores.class_name))}</div>"
            f'<div class="muted">{html.escape(scores.attribute)}</div></td>'
            f'<td class="number">{scores.support}</td>'
            f'<td class="number">{format_metric_value(scores.f1, 3)}</td>'
            f"{_make_discrimination_cells_html(scores.auroc, scores.informedness)}"
            f'<td class="number">{scores.num_missed}</td>{confusion_cell}</tr>')
    headers = [
        sorting.make_column_header_html(
            "Class", "The class, with its iRAP code, and its attribute.", "class",
            is_number=False),
        sorting.make_column_header_html("Support", _SUPPORT_TITLE, "support"),
        sorting.make_column_header_html("F1", ie.describe_metric("F1"), "f1"),
        sorting.make_column_header_html(AUROC_HEADER, _AUROC_TITLE, "auroc"),
        sorting.make_column_header_html(INFORMEDNESS_HEADER, _INFORMEDNESS_TITLE, "informedness"),
        sorting.make_column_header_html(
            "Missed", "The number of segments labeled with the class and predicted as another"
                      " class.", "missed"),
        make_header_cell_html(
            "Mostly as", is_number=True, title="The iRAP code of the class that most of the"
                                               " missed segments are predicted as, with their"
                                               " number.")]
    return make_table_html_from_header("".join(headers), rows)


# Review curve #####################################################################################

#: The role of a model -> the (R, G, B) of its lines, which are the outcome colours of the map
#: for a model alone and for B.
_ROLE_COLORS = {"A": OUTCOME_COLORS["correct"], "B": OUTCOME_COLORS["only B correct"]}
_VIEW_BOX_WIDTH, _VIEW_BOX_HEIGHT = 400, 200
#: The plot area of the chart in SVG units, inside the view box with the tick labels.
_PLOT_LEFT, _PLOT_RIGHT, _PLOT_TOP, _PLOT_BOTTOM = 24, 392, 8, 162
#: The step of the reviewed proportions whose scores the tooltips show.
_TOOLTIP_STEP = 0.05


@dc.dataclass(frozen=True)
class ModelReviewCurve:
    """The review curve of a model for one attribute.

    Attributes:
        role: 'A' or 'B'.
        curve: Of the attribute, without batch dimensions, None for hard predictions.
        manual_review_proportion: The model's `irap_evaluation.MANUAL_REVIEW_PROPORTION_METRIC`
            of the attribute, None if the model lacks it or it is undefined.
    """

    role: str
    curve: ie.ReviewCurve | None
    manual_review_proportion: float | None


def _to_chart_point(reviewed_proportion: float, score: float) -> tuple[float, float]:
    """Returns the SVG coordinates of a reviewed proportion and a score, both in [0, 1]."""
    return (round(_PLOT_LEFT + reviewed_proportion * (_PLOT_RIGHT - _PLOT_LEFT), 1),
            round(_PLOT_BOTTOM - score * (_PLOT_BOTTOM - _PLOT_TOP), 1))


def _make_chart_axes_html() -> str:
    """Makes the gridlines, the line of the target mF1 and the tick labels."""
    parts = []
    target = ie.MANUAL_REVIEW_TARGET_MF1
    for score, label, css_class in ((0, "0", "grid"), (0.5, ".5", "grid"),
                                    (target, format_metric_value(target, 1), "target"),
                                    (1, "1", "grid")):
        x, y = _to_chart_point(0, score)
        parts.append(f'<line class="{css_class}" x1="{x}" x2="{_PLOT_RIGHT}" y1="{y}" y2="{y}"/>'
                     f'<text x="{x - 4}" y="{y + 3.5}" text-anchor="end">{label}</text>')
    for percent in (0, 25, 50, 75, 100):
        x, y = _to_chart_point(percent / 100, 0)
        parts.append(f'<text x="{x}" y="{y + 13}" text-anchor="middle">{percent}</text>')
    x, y = _to_chart_point(0.5, 0)
    parts.append(f'<text x="{x}" y="{y + 32}" text-anchor="middle">'
                 f"% of the segments reviewed, least confident first</text>")
    return "".join(parts)


def _make_model_lines_html(model_curve: ModelReviewCurve) -> str:
    """Makes the lines of mF1 and accuracy of a model with a curve, and the point where mF1
    reaches the target."""
    color = f"rgb{_ROLE_COLORS[model_curve.role]}"
    parts = []
    curve = model_curve.curve
    for css_class, scores in (("accuracy", curve.accuracy), ("macro-f1", curve.macro_f1)):
        points = " ".join("{},{}".format(*_to_chart_point(proportion, score))
                          for proportion, score in zip(curve.review_proportions, scores,
                                                       strict=True))
        parts.append(f'<polyline class="{css_class}" points="{points}" stroke="{color}"/>')
    if model_curve.manual_review_proportion is not None:
        x, y = _to_chart_point(model_curve.manual_review_proportion, ie.MANUAL_REVIEW_TARGET_MF1)
        parts.append(f'<circle class="manual-review-proportion" cx="{x}" cy="{y}" r="4"'
                     f' fill="{color}"/>')
    return "".join(parts)


def _make_tooltip_strips_html(model_curves: T.Sequence[ModelReviewCurve]) -> str:
    """Makes a strip of the plot area around each multiple of `_TOOLTIP_STEP`, with a tooltip of
    the scores of the models at this reviewed proportion, and a line that shows on hover."""
    strips = []
    for proportion in np.arange(0, 1 + _TOOLTIP_STEP / 2, _TOOLTIP_STEP):
        lines = [f"{round(proportion * 100)} % reviewed"]
        for model_curve in model_curves:
            curve = model_curve.curve
            macro_f1, accuracy = (float(np.interp(proportion, curve.review_proportions, scores))
                                  for scores in (curve.macro_f1, curve.accuracy))
            lines.append(f"{model_curve.role}: mF1 {format_metric_value(macro_f1, 3)}, accuracy"
                         f" {format_metric_value(accuracy, 3)}")
        x, _ = _to_chart_point(proportion, 0)
        left, _ = _to_chart_point(max(proportion - _TOOLTIP_STEP / 2, 0), 0)
        right, _ = _to_chart_point(min(proportion + _TOOLTIP_STEP / 2, 1), 0)
        strips.append(
            f'<g class="tooltip-strip"><title>{html.escape(chr(10).join(lines))}</title>'
            f'<line x1="{x}" x2="{x}" y1="{_PLOT_TOP}" y2="{_PLOT_BOTTOM}"/>'
            f'<rect x="{left}" y="{_PLOT_TOP}" width="{round(right - left, 1)}"'
            f' height="{_PLOT_BOTTOM - _PLOT_TOP}"/></g>')
    return "".join(strips)


def _make_line_key_html(css_class: str, color: str) -> str:
    """Makes a short line of a legend, drawn as the lines of `css_class` in the chart."""
    return (f'<svg class="line-key" viewBox="0 0 18 8"><polyline class="{css_class}"'
            f' points="1,4 17,4" stroke="{color}"/></svg>')


def make_review_curve_html(model_curves: T.Sequence[ModelReviewCurve]) -> str:
    """Makes the chart of the review curves of models for one attribute: mF1 and accuracy
    against the proportion of the segments that are reviewed
    (`irap_evaluation.compute_review_curve`), with the point where mF1 reaches the target of the
    manual review proportion. A model with hard predictions gets a note instead of a curve.

    Args:
        model_curves: Of the models of the page, e.g. A and B.
    """
    with_curve = [c for c in model_curves if c.curve is not None]
    notes = "".join(f'<div class="muted">Model {c.role}: no review curve, since its predictions'
                    f" are hard.</div>" for c in model_curves if c.curve is None)
    if not with_curve:
        return f'<div class="review-curve">{notes}</div>'
    metric_name = ie.MANUAL_REVIEW_PROPORTION_METRIC
    legend = "".join(
        f'<span>{_make_line_key_html("macro-f1", f"rgb{_ROLE_COLORS[c.role]}")}Model {c.role}'
        f" ({metric_name} = {format_optional_metric_value(c.manual_review_proportion, 3)})</span>"
        for c in with_curve)
    legend += (f'<span class="muted">{_make_line_key_html("macro-f1", "currentColor")}mF1'
               f' {_make_line_key_html("accuracy", "currentColor")}accuracy</span>')
    description = html.escape(ie.describe_metric(metric_name))
    return (f'<div class="review-curve"><div class="review-legend" title="{description}">'
            f"{legend}</div>"
            f'<svg class="chart" viewBox="0 0 {_VIEW_BOX_WIDTH} {_VIEW_BOX_HEIGHT}"'
            f' style="--view-box-width: {_VIEW_BOX_WIDTH}" role="img"'
            f' aria-label="mF1 and accuracy against the proportion of the segments reviewed">'
            f'{_make_chart_axes_html()}{"".join(map(_make_model_lines_html, with_curve))}'
            f"{_make_tooltip_strips_html(with_curve)}</svg>{notes}</div>")


def format_predicted_class(column: int, irap_codes: T.Sequence[int],
                           class_names: T.Sequence[str]) -> tuple[str, str]:
    """Formats a column of the confusion matrix, K for invalid predictions, for
    `make_segment_list_html`: as (the iRAP code or 'inv.', the class with its code or
    'invalid')."""
    if column == len(irap_codes):
        return "inv.", "invalid"
    return str(irap_codes[column]), _format_class(irap_codes[column], class_names[column])


def make_segment_list_html(segment_indices: T.Sequence[int], segment_ids: T.Sequence[str],
                           selected_index: int | None,
                           predicted_classes: T.Sequence[tuple[str, str]] | None = None) -> str:
    """Makes a button with the id of each of the given segments of the set.

    Args:
        segment_indices: Into the segments of the set, in the `data-segment` attribute.
        segment_ids: The ids of all segments of the set.
        predicted_classes: (short, long) `format_predicted_class` of each of the given segments,
            to show the short one after its id and the long one in its tooltip, e.g. when they
            are not all predicted as the same class. None leaves them out.
    """
    buttons = []
    for k, i in enumerate(segment_indices):
        content, title = html.escape(segment_ids[i]), ""
        if predicted_classes is not None:
            short, long = map(html.escape, predicted_classes[k])
            content += f' <span class="predicted">→ {short}</span>'
            title = f' title="Predicted as {long}"'
        buttons.append(f'<button type="button" data-segment="{i}"'
                       f'{" class=selected" if i == selected_index else ""}{title}>'
                       f"{content}</button>")
    return "".join(buttons)


def _make_probability_table_html(detail: SegmentDetail) -> str:
    header = "".join(f"<th>{make_truncated_name_html(p.label)}</th>"
                     for p in detail.model_predictions)
    rows = []
    for k, (code, name) in enumerate(zip(detail.irap_codes, detail.class_names, strict=True)):
        cells = []
        for prediction in detail.model_predictions:
            if prediction.probs is None:
                cells.append(f'<td class="muted">{"invalid" if k == 0 else ""}</td>')
                continue
            probability = float(prediction.probs[k])
            bar_class = "bar predicted" if prediction.predicted_index == k else "bar"
            cells.append(f'<td class="bar-cell"><div class="{bar_class}"'
                         f' style="width: {probability * 100:.1f}%"></div>'
                         f"<span>{probability:.2f}</span></td>")
        label_class = ' class="label"' if k == detail.label_index else ""
        rows.append(f"<tr><td{label_class}>{html.escape(_format_class(code, name))}</td>"
                    f'{"".join(cells)}</tr>')
    return (f'<table class="probabilities"><thead><tr><th>Class</th>{header}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def make_context_images_html(detail: SegmentDetail,
                             get_image_url: T.Callable[[str], str]) -> str:
    """Makes the context images of a segment, one below the other, from the largest offset, e.g.
    the segment itself (0) above the previous ones (-1, -4).

    Args:
        get_image_url: Segment id -> the URL of its image.
    """
    figures = []
    for offset, segment_id in sorted(detail.context_segment_ids, key=lambda p: p[0],
                                     reverse=True):
        image = (f'<img src="{html.escape(get_image_url(segment_id))}" alt="No image">'
                 if segment_id is not None
                 else '<div class="muted">Outside the road sequence</div>')
        figures.append(f"<figure>{image}<figcaption>offset {offset}</figcaption></figure>")
    return f'<div class="context-images">{"".join(figures)}</div>'


def make_segment_facts_html(detail: SegmentDetail) -> str:
    """Makes the predicted distributions of a segment, and below them its id, road position and
    location. The table shows the label, so the text below only says when there is none."""
    road = ("no road sequence" if detail.road_id is None
            else f"road {detail.road_id}, position {detail.position}")
    location = ("no location" if detail.coordinates is None
                else f"{detail.coordinates[0]:.5f}, {detail.coordinates[1]:.5f}")
    parts = [detail.segment_id, road, location]
    if detail.label_index is None:
        parts.append("no label")
    info = html.escape(" · ".join(parts))
    return (f'<div class="segment-facts">{_make_probability_table_html(detail)}'
            f'<div class="muted">{info}</div></div>')
