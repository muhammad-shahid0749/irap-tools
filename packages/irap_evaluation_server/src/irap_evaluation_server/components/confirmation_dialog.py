"""A modal dialog in which the user confirms changes, e.g. the replacements of an upload."""

import typing as T

from nicegui import ui
from nicegui.elements.mixins.text_element import TextElement

from .native_controls import create_native_button


async def confirm_changes(title: str, items: T.Sequence[str],
                          confirm_text: str = "Confirm") -> bool:
    """Asks the user to confirm changes. Returns whether they confirmed. Escape and a click
    outside the dialog cancel.

    Args:
        items: The changes, as sentences.
    """
    # A div, since Quasar's dialog takes pointer events only on a div (`.q-dialog__inner > div`).
    with ui.dialog() as dialog, ui.element("div").classes("panel confirmation"):
        ui.label(title).classes("section-title")
        with ui.element("ul"):
            for item in items:
                TextElement(tag="li", text=item)
        with ui.element("div").classes("form-row"):
            create_native_button(confirm_text, lambda: dialog.submit(True))
            create_native_button("Cancel", lambda: dialog.submit(False))
    try:
        return await dialog is True  # None if the dialog is closed otherwise.
    finally:
        # A closed page has removed its elements already, and deleting one again raises.
        if not dialog.client.is_deleted:
            dialog.delete()
