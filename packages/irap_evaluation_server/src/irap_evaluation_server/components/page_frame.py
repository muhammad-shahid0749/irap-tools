"""The frame of every page: the styles, the navigation and the sign-in, and the panels and
elements shared by the pages."""

import contextlib
import html
import typing as T
from pathlib import Path

from nicegui import ui
from nicegui.elements.mixins.text_element import TextElement

from ..accounts import Account, can_write, is_admin
from .routes import (
    ACCOUNTS_PATH,
    ACTION_LOG_PATH,
    ANALYSIS_PATH,
    METHODOLOGY_PATH,
    MODELS_PATH,
    SCORES_PATH,
    SHOWN_METRICS_PATH,
    SIGN_OUT_ENDPOINT_PATH,
    get_register_path,
    get_sign_in_path,
)

#: (Title, path) of the pages in the navigation of everyone, also of visitors.
NAVIGATION_PAGES = (("Scores", SCORES_PATH), ("Analysis", ANALYSIS_PATH),
                    ("Models", MODELS_PATH), ("Methodology", METHODOLOGY_PATH))
#: (Title, path) of the pages in the Admin menu of the navigation, which only admins see.
ADMIN_PAGES = (("Accounts", ACCOUNTS_PATH), ("Shown metrics", SHOWN_METRICS_PATH))

_STYLES = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")


def _get_page_url() -> str:
    """Returns the path and query of the current page, as it was opened."""
    url = ui.context.client.request.url
    return f"{url.path}?{url.query}" if url.query else url.path


def make_post_form_html(action: str, content_html: str, next_path: str, submit_text: str) -> str:
    """Makes a plain form that posts to the endpoint `action` with the field `next`, e.g. to
    sign in, so that browsers can save the password.

    Args:
        content_html: The inputs, or other content before the submit button.
    """
    return (f'<form method="post" action="{action}" class="form-row">{content_html}'
            f'<input type="hidden" name="next" value="{html.escape(next_path)}">'
            f'<button class="native-control" type="submit">{submit_text}</button></form>')


def _make_sign_in_html(account: Account | None) -> str:
    """Makes the sign-in and registration links, or the account's name and permission with a
    sign-out button, all of which return to the current page."""
    page_url = _get_page_url()
    if account is None:
        return (f'<a href="{html.escape(get_sign_in_path(page_url))}">Sign in</a> · '
                f'<a href="{html.escape(get_register_path(page_url))}">Register</a>')
    return make_post_form_html(SIGN_OUT_ENDPOINT_PATH,
                               f'<span>{html.escape(account.name)}'
                               f' <span class="muted">({account.permission})</span></span>',
                               page_url, "Sign out")


def _create_admin_menu(current_path: str) -> None:
    """Creates the Admin menu of the navigation, whose links `styles.css` shows while the pointer
    is over its title or the title has the focus, e.g. after a tap, without state that a render of
    the page would reset."""
    is_current = any(path == current_path for _, path in ADMIN_PAGES)
    with ui.element("div").classes("nav-menu"):
        TextElement(tag="span", text="Admin ▾").classes(
            "nav-menu-title current" if is_current else "nav-menu-title").props("tabindex=0")
        with ui.element("div").classes("nav-menu-items"):
            for title, path in ADMIN_PAGES:
                ui.link(title, path).classes("current" if path == current_path else "")


@contextlib.contextmanager
def create_page_frame(current_path: str, account: Account | None,
                      fills_window: bool = False) -> T.Iterator[ui.element]:
    """Creates the top bar and the content container, in which the page creates its content.

    Args:
        current_path: The path of the navigation entry to mark as current.
        account: The signed-in account (`AccountSessions.get_account`), or None.
        fills_window: Whether the content fills the rest of the window, without padding or
            scrolling, e.g. for a map. Otherwise it is a column of panels that scrolls.
    """
    ui.add_css(_STYLES)
    ui.query(".nicegui-content").classes("p-0 gap-0" + (" h-screen" if fills_window else ""))
    pages = NAVIGATION_PAGES + ((("Action log", ACTION_LOG_PATH),) if can_write(account) else ())
    with ui.element("header").classes("top-bar w-full"):
        ui.label("iRAP evaluation").classes("title")
        with ui.element("nav"):
            for title, path in pages:
                ui.link(title, path).classes("current" if path == current_path else "")
            if is_admin(account):
                _create_admin_menu(current_path)
        ui.html(_make_sign_in_html(account), sanitize=False).classes("sign-in")
    with ui.element("main").classes(
            "page-content fills-window" if fills_window else "page-content") as content:
        yield content


def create_write_permission_panel(account: Account | None, title: str, action: str) -> None:
    """Creates the panel of a form that the user cannot use, with why: a link to the Sign-in
    page, or that the account can only view.

    Args:
        title: The title of the form's panel.
        action: What the form does, e.g. 'create ensembles'.
    """
    with ui.element("section").classes("panel"):
        ui.label(title).classes("section-title")
        if account is None:
            ui.link(f"Sign in to {action}.", get_sign_in_path(_get_page_url()))
        else:
            ui.label(f"Your account can only view. An admin can give it write permission, so"
                     f" that you can {action}.").classes("muted")


@contextlib.contextmanager
def create_collapsible_panel(title: str, classes: str = "") -> T.Iterator[ui.element]:
    """Creates a panel that is collapsed at first, with `title` as its summary, in which the
    caller creates the content. A refresh of the content does not close it.

    Args:
        classes: More classes of its `details` element.
    """
    with ui.element("section").classes("panel"), \
            ui.element("details").classes(f"collapsible-section {classes}") as details:
        TextElement(tag="summary", text=title).classes("section-title")
        yield details


def create_key_values(rows: T.Mapping[str, str | tuple[str, str]]) -> None:
    """Creates a grid of keys and their values, a text, or the text and the path of a link."""
    with ui.element("div").classes("key-values"):
        for key, value in rows.items():
            ui.label(key).classes("key")
            if isinstance(value, tuple):
                ui.link(*value)
            else:
                ui.label(value)


#: The extras of the Markdown converter (markdown2): NiceGUI's default ones, and 'code-friendly',
#: which keeps the underscores of names and formulas, e.g. `seed_1_probs` and `m_i`, from
#: becoming emphasis.
_MARKDOWN_EXTRAS = ["fenced-code-blocks", "tables", "code-friendly"]


def create_markdown_text(markdown: str, *, sanitize: bool) -> ui.markdown:
    """Shows Markdown with the styles of a text (`.markdown-text`).

    Args:
        sanitize: Whether the browser sanitizes the HTML (DOMPurify), which the Markdown that
            users write needs.
    """
    return ui.markdown(markdown, extras=_MARKDOWN_EXTRAS, sanitize=sanitize).classes(
        "markdown-text")


def create_description(description: str) -> None:
    """Shows the description of a model or method, which is Markdown, under a title, unless it is
    empty."""
    if description:
        ui.label("Description").classes("subsection-title")
        create_markdown_text(description, sanitize=True).classes("description")


def show_view_error(message: str, page_path: str) -> None:
    """Shows why the URL of a page has no view, e.g. an unknown dataset, with a link to the page
    without parameters.

    Args:
        page_path: The path of a page in `NAVIGATION_PAGES`.
    """
    title = dict((path, title) for title, path in NAVIGATION_PAGES)[page_path]
    ui.label(message).classes("error")
    ui.link(f"Open the default {title} page", page_path)
