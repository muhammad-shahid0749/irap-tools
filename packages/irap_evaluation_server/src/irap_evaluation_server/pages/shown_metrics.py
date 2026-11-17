"""The Shown metrics page, on which admins hide some attribute averages from the pages (see
`display_settings`)."""

import html

import irap_evaluation as ie
from nicegui import ui

from ..accounts import is_admin
from ..components.account_sessions import AccountSessions
from ..components.changes import run_admin_change
from ..components.native_controls import create_native_checkbox
from ..components.page_frame import create_page_frame
from ..components.routes import SHOWN_METRICS_PATH
from ..display_settings import DisplaySettingsStore
from ..method_ranking import HIDEABLE_METRIC_NAMES, RANKING_METRIC_NAMES


def _make_metric_label_html(metric_name: str, is_hideable: bool) -> str:
    """Makes the name of an attribute average with the description of its per-attribute
    metric, which the per-attribute view shows too."""
    description = ie.describe_metric(ie.parse_metric_name(metric_name).name)
    if not is_hideable:
        description += " The main metric, always shown."
    return f'<b>{metric_name}</b> <span class="muted">{html.escape(description)}</span>'


def _create_metric_checkboxes(store: DisplaySettingsStore, sessions: AccountSessions) -> None:
    """Creates a checkbox per attribute average, which shows or hides it at once. The store checks
    that the signed-in account is still an admin when a change runs."""
    @ui.refreshable
    def show_checkboxes() -> None:
        hidden = store.get_hidden_metrics()
        for name in RANKING_METRIC_NAMES:
            is_hideable = name in HIDEABLE_METRIC_NAMES
            checkbox = create_native_checkbox(
                _make_metric_label_html(name, is_hideable), name not in hidden,
                lambda is_shown, n=name: on_changed(n, is_shown), is_label_html=True)
            if not is_hideable:
                checkbox.props["disabled"] = True

    async def on_changed(metric_name: str, is_shown: bool) -> None:
        await run_admin_change(
            lambda actor: store.set_metric_hidden(metric_name, not is_shown, actor), sessions,
            status_label, f"{metric_name} is now {'shown' if is_shown else 'hidden'}. Open pages"
                          f" show the change after a reload.",
            show_checkboxes.refresh)

    with ui.element("div").classes("flex flex-col gap-1"):
        show_checkboxes()
    status_label = ui.label().classes("muted")


def register_shown_metrics_page(store: DisplaySettingsStore, sessions: AccountSessions) -> None:
    @ui.page(SHOWN_METRICS_PATH, title="Shown metrics · iRAP evaluation")
    def shown_metrics_page() -> None:
        account = sessions.get_account()
        with create_page_frame(SHOWN_METRICS_PATH, account), \
                ui.element("section").classes("panel"):
            ui.label("Shown metrics").classes("section-title")
            if not is_admin(account):
                ui.label("Only admins can change the shown metrics.").classes("error")
                return
            ui.label("The attribute averages that the pages show, with their per-attribute"
                     " metrics, e.g. aBrier with Brier. A hidden metric is still scored and in"
                     " the downloaded scores, so showing it again needs no scoring.").classes(
                "muted")
            _create_metric_checkboxes(store, sessions)
