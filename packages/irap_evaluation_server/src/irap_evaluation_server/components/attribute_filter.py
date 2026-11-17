"""The attribute filter: which of the scored attributes the averages of a page cover.

The subset is in the URL as repeated `attribute` query parameters (`routes.ATTRIBUTE_PARAMETER`),
and no parameter means all attributes.
"""

import dataclasses as dc
import typing as T

from nicegui import ui
from nicegui.elements.mixins.text_element import TextElement

from .native_controls import (
    create_native_button,
    create_native_checkbox,
    create_native_input,
    set_native_checkbox_checked,
)


@dc.dataclass(frozen=True)
class AttributeSubset:
    """
    Attributes:
        options: The scored attributes.
        selected: The attributes that the averages cover, in the order of `options`.
        ignored: Attributes of the URL that are not options, e.g. of another dataset.
    """

    options: tuple[str, ...]
    selected: tuple[str, ...]
    ignored: tuple[str, ...] = ()

    @classmethod
    def from_query(cls, values: T.Sequence[str], options: T.Sequence[str]) -> T.Self:
        """Makes the subset of the query values, all options if none of them is an option."""
        options = tuple(options)
        selected = tuple(a for a in options if a in values)
        return cls(options=options, selected=selected or options,
                   ignored=tuple(dict.fromkeys(v for v in values if v not in options)))

    def select(self, attributes: T.Collection[str]) -> T.Self:
        return type(self)(options=self.options,
                          selected=tuple(a for a in self.options if a in attributes))

    @property
    def is_all(self) -> bool:
        return len(self.selected) == len(self.options)

    def to_query(self) -> tuple[str, ...]:
        """Returns the query values, none if all attributes are selected."""
        return () if self.is_all else self.selected

    @property
    def count_label(self) -> str:
        """E.g. '12 of 34', or 'all 34'."""
        num_options = len(self.options)
        return (f"all {num_options}" if self.is_all
                else f"{len(self.selected)} of {num_options}")

    @property
    def label(self) -> str:
        """E.g. '12 of 34 attributes'."""
        return f"{self.count_label} attributes"


class AttributeFilter:
    """The attribute filter of a page whose options are known only when it renders its scores,
    e.g. after a scoring: a collapsible panel in a fixed place, created again when the options
    change.

    The panel has a checkbox per option, a text filter of the shown checkboxes, and All and None
    buttons for the shown ones, so that None, a filter text and All select the matching
    attributes. It is not created again on changes of the subset, so it stays open.

    Attributes:
        subset: The current subset, None before the first `set_options`.
    """

    def __init__(self, query_values: T.Sequence[str],
                 on_change: T.Callable[[AttributeSubset], T.Any]):
        """
        Args:
            query_values: The subset of the URL.
            on_change: Called after a change of the subset by the user. It can be async.
        """
        self.subset: AttributeSubset | None = None
        self._query_values = tuple(query_values)
        self._on_change = on_change
        self._container = ui.element("div")
        self._summary: TextElement | None = None
        self._option_to_checkbox: dict[str, ui.element] = {}
        self._option_to_container: dict[str, ui.element] = {}

    def set_options(self, options: T.Sequence[str]) -> AttributeSubset:
        """Sets the options of the latest render, and returns the subset. New options keep the
        selected attributes that are among them."""
        if self.subset is None or self.subset.options != tuple(options):
            query_values = self._query_values if self.subset is None else self.subset.to_query()
            self.subset = AttributeSubset.from_query(query_values, options)
            self._container.clear()
            if options:
                with self._container:
                    self._create_panel()
        return self.subset

    def _create_panel(self) -> None:
        subset = self.subset
        self._option_to_checkbox, self._option_to_container = {}, {}
        with ui.element("details").classes("collapsible-section attribute-filter"):
            self._summary = TextElement(tag="summary", text=f"Attributes: {subset.count_label}")
            with ui.element("div").classes("form-row"):
                create_native_input("Filter", "", self._filter_options,
                                    placeholder="part of a name", size=24)
                create_native_button(
                    "All", lambda: self._select(set(self.subset.selected) | self._get_shown()))
                create_native_button(
                    "None", lambda: self._select(set(self.subset.selected) - self._get_shown()))
            with ui.element("div").classes("attribute-options"):
                for option in subset.options:
                    with ui.element("div") as container:
                        self._option_to_checkbox[option] = create_native_checkbox(
                            option, option in subset.selected,
                            lambda is_checked, o=option: self._set_checked(o, is_checked))
                    self._option_to_container[option] = container
            if subset.ignored:
                ui.label(f"Ignored, since they are not scored here: {', '.join(subset.ignored)}."
                         ).classes("muted")

    def _select(self, selected: T.Collection[str]) -> T.Any:
        """Selects attributes. Returns the result of `on_change`, so that NiceGUI awaits that of
        an async one."""
        self.subset = self.subset.select(selected)
        for option, checkbox in self._option_to_checkbox.items():
            set_native_checkbox_checked(checkbox, option in self.subset.selected)
        self._summary.set_text(f"Attributes: {self.subset.count_label}")
        return self._on_change(self.subset)

    def _set_checked(self, option: str, is_checked: bool) -> T.Any:
        selected = set(self.subset.selected)
        if is_checked:
            selected.add(option)
        else:
            selected.discard(option)
        return self._select(selected)

    def _filter_options(self, text: str) -> None:
        for option, container in self._option_to_container.items():
            container.set_visibility(text.strip().lower() in option.lower())

    def _get_shown(self) -> set[str]:
        return {o for o, container in self._option_to_container.items() if container.visible}
