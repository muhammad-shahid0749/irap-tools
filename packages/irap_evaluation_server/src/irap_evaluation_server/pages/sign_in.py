"""The Sign-in and Register pages (`accounts`), and the endpoints of their forms and of the
sign-out button. All are plain forms, so that browsers can save the password."""

import asyncio
import typing as T

from fastapi import Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from nicegui import app, run, ui

from ..accounts import MAX_ACCOUNT_NAME_LENGTH, MIN_PASSWORD_LENGTH, Account
from ..components.account_sessions import AccountSessions
from ..components.page_frame import create_page_frame, make_post_form_html
from ..components.routes import (
    REGISTER_ENDPOINT_PATH,
    REGISTER_PATH,
    SCORES_PATH,
    SIGN_IN_ENDPOINT_PATH,
    SIGN_IN_PATH,
    SIGN_OUT_ENDPOINT_PATH,
    get_register_path,
    get_sign_in_path,
    make_query_path,
    to_local_path,
)

#: The key in the user storage of why a registration was refused, which the Register page shows
#: once. It is not in the URL, so that a link cannot make the page show any text.
_REGISTRATION_ERROR_KEY = "registration_error"
#: The most passwords that are hashed at once (about 16 MiB and 50 ms each,
#: `accounts.hash_password`), so that many sign-in or registration requests cannot take the
#: memory or the threads of `run.io_bound`, which other work also uses.
_MAX_CONCURRENT_HASHES = 2


def _make_input_html(label: str, attributes: str) -> str:
    return (f'<label class="field">{label}'
            f'<input class="native-control" required size="24" {attributes}></label>')


def _make_sign_in_form_html(next_path: str) -> str:
    inputs_html = (_make_input_html("Account", 'name="name" autocomplete="username"')
                   + _make_input_html("Password", 'type="password" name="password"'
                                                  ' autocomplete="current-password"'))
    return make_post_form_html(SIGN_IN_ENDPOINT_PATH, inputs_html, next_path, "Sign in")


def _make_registration_form_html(next_path: str) -> str:
    new_password = (f'type="password" autocomplete="new-password"'
                    f' minlength="{MIN_PASSWORD_LENGTH}"')
    inputs_html = (_make_input_html("Account name", f'name="name" autocomplete="username"'
                                                    f' maxlength="{MAX_ACCOUNT_NAME_LENGTH}"')
                   + _make_input_html(f"Password (at least {MIN_PASSWORD_LENGTH} characters)",
                                      f'name="password" {new_password}')
                   + _make_input_html("Repeat the password",
                                      f'name="repeated_password" {new_password}'))
    return make_post_form_html(REGISTER_ENDPOINT_PATH, inputs_html, next_path, "Register")


def _to_next_path(url: str) -> str:
    """Returns the local path to return to after signing in or out (`routes.to_local_path`), by
    default the Scores page."""
    return to_local_path(url, SCORES_PATH)


def _create_form_panel(title: str, intro: str, account: Account | None, error: str | None,
                       form_html: str, other_link: tuple[str, str], next_path: str) -> None:
    """Creates the panel of the Sign-in or Register page.

    Args:
        account: The signed-in account, if any.
        other_link: The text and path of a link to the other page.
    """
    with ui.element("section").classes("panel"):
        ui.label(title).classes("section-title")
        ui.label(intro).classes("muted")
        if account is not None:
            ui.label(f"You are signed in as {account.name}.")
        if error is not None:
            ui.label(error).classes("error")
        ui.html(form_html, sanitize=False)
        with ui.element("div").classes("form-row"):
            ui.link(*other_link)
            ui.link("Back", next_path)


def register_sign_in_pages(sessions: AccountSessions) -> None:
    password_hashing = asyncio.Semaphore(_MAX_CONCURRENT_HASHES)

    @app.post(SIGN_IN_ENDPOINT_PATH)
    async def sign_in(name: T.Annotated[str, Form()], password: T.Annotated[str, Form()],
                      next_url: T.Annotated[str, Form(alias="next")] = "") -> RedirectResponse:
        next_path = _to_next_path(next_url)
        async with password_hashing:
            # None also if the server is stopping.
            account = await run.io_bound(sessions.accounts.verify, name.strip(), password)
        if account is None:
            return RedirectResponse(make_query_path(SIGN_IN_PATH,
                                                    {"next": next_path, "failed": "1"}),
                                    status_code=303)
        sessions.sign_in(account)
        return RedirectResponse(next_path, status_code=303)

    @app.post(REGISTER_ENDPOINT_PATH)
    async def register(name: T.Annotated[str, Form()], password: T.Annotated[str, Form()],
                       repeated_password: T.Annotated[str, Form()],
                       next_url: T.Annotated[str, Form(alias="next")] = "") -> RedirectResponse:
        next_path = _to_next_path(next_url)
        try:
            if repeated_password != password:
                raise ValueError("The passwords differ.")
            async with password_hashing:
                account = await run.io_bound(sessions.accounts.register, name, password)
        except ValueError as e:
            app.storage.user[_REGISTRATION_ERROR_KEY] = str(e)
            return RedirectResponse(get_register_path(next_path), status_code=303)
        if account is None:
            raise HTTPException(status_code=503, detail="The server is stopping.")
        sessions.sign_in(account)
        return RedirectResponse(next_path, status_code=303)

    @app.post(SIGN_OUT_ENDPOINT_PATH)
    def sign_out(next_url: T.Annotated[str, Form(alias="next")] = "") -> RedirectResponse:
        sessions.sign_out()
        return RedirectResponse(_to_next_path(next_url), status_code=303)

    @ui.page(SIGN_IN_PATH, title="Sign in · iRAP evaluation")
    def sign_in_page(request: Request, failed: bool = False) -> None:
        next_path = _to_next_path(request.query_params.get("next", ""))
        account = sessions.get_account()
        with create_page_frame(SIGN_IN_PATH, account):
            _create_form_panel(
                "Sign in", "Anyone can view the archive. Accounts with write permission can also"
                           " change it.", account,
                "The account name or password is wrong." if failed else None,
                _make_sign_in_form_html(next_path), ("Register", get_register_path(next_path)),
                next_path)

    @ui.page(REGISTER_PATH, title="Register · iRAP evaluation")
    def register_page(request: Request) -> None:
        next_path = _to_next_path(request.query_params.get("next", ""))
        account = sessions.get_account()
        with create_page_frame(REGISTER_PATH, account):
            _create_form_panel(
                "Register", "Anyone can view the archive. The first account is an admin. Later"
                            " accounts can view, until an admin gives them write permission to"
                            " change the archive.", account,
                app.storage.user.pop(_REGISTRATION_ERROR_KEY, None),
                _make_registration_form_html(next_path), ("Sign in", get_sign_in_path(next_path)),
                next_path)
