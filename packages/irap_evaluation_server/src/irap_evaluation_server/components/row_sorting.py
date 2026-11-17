"""Sorting of the rows of an HTML table by a click on a column header.

A row sort is the key of the column that the rows are sorted by, after `REVERSED_SORT_PREFIX` for
the reverse of the first order of the column, or '' for the unsorted rows. The header of a column
that the rows can be sorted by has a `data-sort` attribute with the row sort that a click selects
(`SORT_CLICK_JS_HANDLER`): the first order of the column, or its reverse if the rows are in the
first order.
"""

import dataclasses as dc
import html
import typing as T

import irap_evaluation as ie

from ..method_ranking import sort_keys_by_value
from .formatting import make_header_cell_html, make_title_attribute

K = T.TypeVar("K", bound=T.Hashable)

#: The JavaScript handler of a click on a table whose headers have a `data-sort` attribute. It
#: emits the attribute of the clicked header, unless the click is on a link in it, e.g. to a
#: method.
SORT_CLICK_JS_HANDLER = ("(e) => { const th = e.target.closest('th[data-sort]');"
                         " if (th && !e.target.closest('a')) emit(th.dataset.sort); }")

REVERSED_SORT_PREFIX = "-"


@dc.dataclass(frozen=True)
class SortColumn(T.Generic[K]):
    """A column that the rows of a table can be sorted by.

    Attributes:
        row_to_value: Row -> its value in the column, None if it has none.
        is_descending_first: Whether the first order of the column is descending.
    """

    row_to_value: T.Mapping[K, float | None]
    is_descending_first: bool


def make_metric_sort_column(row_to_value: T.Mapping[K, float | None],
                            metric_name: str) -> SortColumn[K]:
    """Makes a column of the values of a metric, sorted worst first."""
    return SortColumn(row_to_value, is_descending_first=ie.is_lower_better(metric_name))


@dc.dataclass(frozen=True)
class RowSorting(T.Generic[K]):
    """The order of the rows of a table, and its headers.

    Attributes:
        key_to_column: The key of a column in a row sort -> the column, for the columns that the
            rows can be sorted by.
        row_sort: The row sort of the table. The rows are not sorted for '' and for a key that
            is not in `key_to_column`, e.g. of a method that is not shown.
        rows_noun: What the rows are, for the tooltips.
    """

    key_to_column: T.Mapping[str, SortColumn[K]]
    row_sort: str
    rows_noun: str = "attributes"

    @property
    def _sorted_key(self) -> str:
        return self.row_sort.removeprefix(REVERSED_SORT_PREFIX)

    @property
    def _is_reversed(self) -> bool:
        return self.row_sort.startswith(REVERSED_SORT_PREFIX)

    def sort(self, rows: T.Sequence[K]) -> T.Sequence[K]:
        if (column := self.key_to_column.get(self._sorted_key)) is None:
            return rows
        return sort_keys_by_value(rows, column.row_to_value,
                                  is_descending=column.is_descending_first != self._is_reversed)

    def make_unsorted_header_html(self, text: str) -> str:
        """Makes a header with an escaped text, e.g. of the names of the rows. While the rows are
        sorted, a click on it selects the unsorted rows."""
        if self._sorted_key not in self.key_to_column:
            return make_header_cell_html(text)
        return (f'<th class="sortable" data-sort="" title="Click to undo the sorting of the'
                f' {self.rows_noun}.">{html.escape(text)}</th>')

    def make_column_header_html(self, content_html: str, title: str, sort_key: str, *,
                                is_number: bool = True, is_vertical: bool = False) -> str:
        """Makes the header of a column. If the rows can be sorted by the column, a mark after
        the header shows their order: ↑ ascending, ↓ descending, ↕ not sorted by the column.

        Args:
            title: The tooltip, '' for none.
            sort_key: The key of the column in a row sort.
            is_number: Whether the cells of the column are `number` cells, which the header is
                aligned with.
            is_vertical: Whether the header is read from bottom to top, with the mark below it,
                so that the column is as narrow as its values (`vertical-header` in
                `styles.css`).
        """
        classes, sort_attribute, mark_html = ["number"] if is_number else [], "", ""
        if is_vertical:
            classes.append("vertical-header")
            content_html = f"<div>{content_html}</div>"
        if (column := self.key_to_column.get(sort_key)) is not None:
            is_sorted = sort_key == self._sorted_key
            next_sort = (REVERSED_SORT_PREFIX + sort_key if is_sorted and not self._is_reversed
                         else sort_key)
            classes.append("sortable")
            sort_attribute = f' data-sort="{html.escape(next_sort)}"'
            if is_sorted:
                classes.append("sorted")
                mark = "↓" if column.is_descending_first != self._is_reversed else "↑"
            else:
                mark = "↕"
            mark_html = f' <span class="sort-mark">{mark}</span>'
        class_attribute = f' class="{" ".join(classes)}"' if classes else ""
        return (f"<th{class_attribute}{sort_attribute}{make_title_attribute(title)}>"
                f"{content_html}{mark_html}</th>")
