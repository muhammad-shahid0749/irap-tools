"""The Action log page: all changes of the archive, with who made them and when, for accounts
that can write."""

from nicegui import ui

from ..accounts import can_write
from ..archive import ModelArchive
from ..components.account_sessions import AccountSessions
from ..components.formatting import make_action_table_html
from ..components.page_frame import create_page_frame, create_write_permission_panel
from ..components.routes import ACTION_LOG_PATH


def register_action_log_page(archive: ModelArchive, sessions: AccountSessions) -> None:
    @ui.page(ACTION_LOG_PATH, title="Action log · iRAP evaluation")
    def action_log_page() -> None:
        account = sessions.get_account()
        with create_page_frame(ACTION_LOG_PATH, account):
            if not can_write(account):
                create_write_permission_panel(account, "Action log", "see the action log")
                return
            with ui.element("section").classes("panel"):
                ui.label("Action log").classes("section-title")
                if actions := archive.list_actions():
                    models = {m.id: m for m in archive.list_models()}
                    ui.html(make_action_table_html(actions, models),
                            sanitize=False).classes("w-full")
                else:
                    ui.label("No actions.").classes("muted")
