"""The polling of work that a page waits for."""

import typing as T

from nicegui import ui


class RefreshTimer:
    """Refreshes a view while work that it shows is pending, e.g. scorings of the scoring worker:
    each time some of the work is done, so that results appear one by one.

    After each render, the view passes the pending work to `watch`. The refresh renders again,
    which may watch new work.
    """

    def __init__(self, on_progress: T.Callable[[], T.Any], interval_s: float = 1.0):
        """
        Args:
            on_progress: Renders the view again. It can be async.
        """
        self._on_progress = on_progress
        self._pending_checks: list[T.Callable[[], bool]] = []
        self._timer = ui.timer(interval_s, self._check, active=False)

    def watch(self, pending_checks: T.Iterable[T.Callable[[], bool]]) -> None:
        """Watches the work of the latest render, replacing the work watched before.

        Args:
            pending_checks: One per item of work that is pending, true while it is pending.
        """
        self._pending_checks = list(pending_checks)
        self._timer.active = bool(self._pending_checks)

    def _check(self) -> T.Any:
        """Returns the result of `on_progress`, if called, so that the timer awaits that of an
        async one."""
        if not all(check() for check in self._pending_checks):
            self._timer.active = False
            return self._on_progress()
        return None
