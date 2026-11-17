"""Running work that a user starts on a page off the event loop, e.g. a change of the archive or
of an account, with refusals shown in a status label."""

import functools
import inspect
import typing as T

from nicegui import Client, run, ui

from ..accounts import is_admin
from .account_sessions import AccountSessions
from .native_controls import set_status

R = T.TypeVar("R")


def _show_error(status_label: ui.label, text: str) -> None:
    if not status_label.client.is_deleted:
        set_status(status_label, text, is_error=True)


def _show_internal_error(status_label: ui.label, text: str) -> None:
    _show_error(status_label, f"Internal error: {text}. See the server log.")


async def run_in_thread_showing_errors(function: T.Callable[[], R],
                                       status_label: ui.label) -> R | None:
    """Runs `function` in a thread, e.g. since it may wait for a database lock or take seconds.
    Shows a refusal or an internal error in `status_label`.

    Args:
        function: Raises `ValueError` or `LookupError` if the work is refused.

    Returns:
        The result of `function`, or None if it was refused or the server is stopping.
    """
    try:
        return await run.io_bound(function)
    except (ValueError, LookupError) as e:
        _show_error(status_label, str(e))
        return None
    except Exception as e:
        _show_internal_error(status_label, repr(e))
        raise


async def run_in_process_showing_errors(function: T.Callable[[], R],
                                        status_label: ui.label) -> R | None:
    """Runs `function` in a process of `run.cpu_bound`, e.g. since it takes seconds of CPU time,
    which would delay the event loop in a thread. Shows an internal error in `status_label`.

    Args:
        function: Pickled for the process, e.g. a `functools.partial` of a module-level
            function.

    Returns:
        The result of `function`, or None if the call was cancelled or the server is stopping.
    """
    try:
        return await run.cpu_bound(function)
    except Exception as e:
        # The text of a `SubprocessException` has the traceback, which the server log shows.
        _show_internal_error(status_label, f"{e.original_type}: {e.original_message}"
                             if isinstance(e, run.SubprocessException) else repr(e))
        raise


async def run_change_in_thread(change: T.Callable[[str], T.Any], get_actor: T.Callable[[], str],
                               status_label: ui.label) -> bool:
    """Runs a change with `run_in_thread_showing_errors`. Returns whether it was done.

    Args:
        change: Gets the actor. It returns something other than None, and raises `ValueError`
            or `LookupError` if the change is refused.
        get_actor: Returns the name of the signed-in account that may make the change, e.g.
            `AccountSessions.require_writer_name`, or raises `ValueError`. It is called on the
            event loop, since the user storage is not available in the thread of the change.
    """
    try:
        actor = get_actor()
    except ValueError as e:
        _show_error(status_label, str(e))
        return False
    return await run_in_thread_showing_errors(functools.partial(change, actor),
                                              status_label) is not None


async def run_admin_change(change: T.Callable[[str], T.Any], sessions: AccountSessions,
                           status_label: ui.label, done_text: str,
                           refresh: T.Callable[[], T.Any]) -> None:
    """Runs a change that only admins may make, whose store checks that the signed-in account is
    still an admin when it runs (`run_change_in_thread`). Then shows what it changed as it is
    (`refresh`), also after a refused change, or reloads the page if the user is no longer an
    admin.

    Args:
        done_text: The status after the change is done.
    """
    is_done = await run_change_in_thread(change, sessions.require_account_name, status_label)
    if status_label.client.is_deleted:
        return
    if not is_admin(sessions.get_account()):
        ui.navigate.reload()
        return
    if is_done:
        set_status(status_label, done_text)
    refresh()


async def load_after_change(load: T.Callable[[], R], missing_path: str,
                            client: Client) -> R | None:
    """Loads what a page shows again in a thread after a change, also after a refused one, e.g.
    of something that another user has changed. Goes to `missing_path` if it no longer exists.

    Args:
        load: Raises `LookupError` if what the page shows no longer exists.
        client: The client of the page.

    Returns:
        The result of `load`, or None if it no longer exists, the page is closed, or the server
        is stopping.
    """
    try:
        loaded = await run.io_bound(load)
    except LookupError:  # E.g. deleted by the change or by another user.
        if not client.is_deleted:
            ui.navigate.to(missing_path)
        return None
    return None if client.is_deleted else loaded


async def run_refreshes(refreshes: T.Iterable[T.Callable[[], T.Any]]) -> None:
    """Runs the refreshes of the panels of a page in order, awaiting those that return an
    awaitable."""
    for refresh in refreshes:
        if inspect.isawaitable(result := refresh()):
            await result
