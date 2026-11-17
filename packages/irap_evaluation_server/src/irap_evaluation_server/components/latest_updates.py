"""The updates of a view whose work runs off the event loop, e.g. in `nicegui.run.io_bound`."""

import typing as T

from nicegui import ui


class LatestUpdates:
    """Tells whether an update of a view is the latest, so that an update whose work was overtaken
    by a later one, e.g. of another click, does not show its outdated result."""

    def __init__(self) -> None:
        self._client = ui.context.client
        self._num_started = 0

    def start(self) -> T.Callable[[], bool]:
        """Starts an update. Returns a check of whether it is still the latest update and the page
        is still open."""
        self._num_started += 1
        number = self._num_started
        return lambda: number == self._num_started and not self._client.is_deleted
