"""What the pages show, which admins choose: the hidden metrics (`method_ranking.ShownMetrics`).

Hiding a metric changes only the pages, not the scoring or the downloads, so a metric that is shown
again appears at once.

The settings are in `display_settings.sqlite3` in the data directory, apart from the archive and
the accounts.
"""

import logging
import sqlite3
from pathlib import Path

from .accounts import AccountStore, is_admin
from .database import Migration, begin_write, connect, initialize_schema
from .method_ranking import HIDEABLE_METRIC_NAMES, ShownMetrics

_logger = logging.getLogger(__name__)

DATABASE_FILE_NAME = "display_settings.sqlite3"
#: The version of the database schema (`PRAGMA user_version`). A database of an older version is
#: migrated (`_MIGRATIONS`).
SCHEMA_VERSION = 1
#: Version v -> the migration from v to v + 1 (`database.initialize_schema`).
_MIGRATIONS: dict[int, Migration] = {}

_SCHEMA = """
CREATE TABLE hidden_metrics (
    metric_name TEXT PRIMARY KEY
);
"""


def _read_hidden_metrics(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(row["metric_name"] for row in
                     connection.execute("SELECT metric_name FROM hidden_metrics"))


class DisplaySettingsStore:
    """The display settings database of a data directory (see the module docstring), created on
    first use.

    Each operation opens its own database connection, so the store can be used from several
    threads.

    Raises:
        ValueError: If the database cannot be migrated to `SCHEMA_VERSION`, or its tables differ
            from those of its version (see `database.initialize_schema`).
    """

    def __init__(self, data_dir: Path, accounts: AccountStore):
        """Args:
            accounts: Whose admins can change the settings.
        """
        self.database_path = data_dir / DATABASE_FILE_NAME
        self.accounts = accounts
        initialize_schema(self.database_path, _SCHEMA, SCHEMA_VERSION, "display settings",
                          _MIGRATIONS)

    def get_hidden_metrics(self) -> frozenset[str]:
        """Returns the hidden attribute averages. Those that are not in `HIDEABLE_METRIC_NAMES`,
        e.g. stored before a library update, are shown (`ShownMetrics.from_hidden`)."""
        with connect(self.database_path) as connection:
            return _read_hidden_metrics(connection)

    def get_shown_metrics(self) -> ShownMetrics:
        """Returns the metrics that the pages show."""
        return ShownMetrics.from_hidden(self.get_hidden_metrics())

    def set_metric_hidden(self, metric_name: str, is_hidden: bool, actor: str) -> frozenset[str]:
        """Hides or shows an attribute average. Changing one metric at a time keeps the changes
        of other metrics, e.g. by another admin meanwhile.

        Args:
            actor: The name of the admin who changes it.

        Returns:
            The hidden attribute averages after the change.

        Raises:
            ValueError: If `actor` is not an admin, e.g. after another admin changed it, or the
                metric is not in `HIDEABLE_METRIC_NAMES`.
        """
        if metric_name not in HIDEABLE_METRIC_NAMES:
            raise ValueError(f"The metric {metric_name!r} cannot be hidden. Metrics:"
                             f" {', '.join(HIDEABLE_METRIC_NAMES)}.")
        with begin_write(self.database_path) as connection:
            # After the lock is taken, so that the check is of the permission at the write.
            if not is_admin(self.accounts.find_account(actor)):
                raise ValueError("Only admins can change the shown metrics.")
            if is_hidden:
                connection.execute("INSERT OR IGNORE INTO hidden_metrics (metric_name)"
                                   " VALUES (?)", (metric_name,))
            else:
                connection.execute("DELETE FROM hidden_metrics WHERE metric_name = ?",
                                   (metric_name,))
            hidden = _read_hidden_metrics(connection)
        _logger.info("%s %s the metric %s.", actor, "hid" if is_hidden else "showed", metric_name)
        return hidden

    def warn_about_unknown_metrics(self) -> None:
        """Logs a warning about stored hidden metrics that are not in `HIDEABLE_METRIC_NAMES`,
        which have no effect, e.g. at the start of the server."""
        if unknown := sorted(self.get_hidden_metrics() - set(HIDEABLE_METRIC_NAMES)):
            _logger.warning("The display settings %s hide the unknown metrics [%s], which are"
                            " ignored.", self.database_path, ", ".join(unknown))
