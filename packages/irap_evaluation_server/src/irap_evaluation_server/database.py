"""Connections to the SQLite databases of the data directory: those of the archive, the score cache
and the accounts, and the migrations of their schemas. Times are stored in UTC as ISO 8601
(`get_utc_now`)."""

import contextlib
import logging
import sqlite3
import typing as T
from datetime import UTC, datetime
from pathlib import Path

_logger = logging.getLogger(__name__)

#: Changes the tables of a database from one schema version to the next (`initialize_schema`),
#: with `sqlite3.Connection.execute`, since `executescript` would commit the transaction.
Migration = T.Callable[[sqlite3.Connection], None]


def get_utc_now() -> datetime:
    """Returns the current time in UTC, to whole seconds."""
    return datetime.now(UTC).replace(microsecond=0)


@contextlib.contextmanager
def connect(database_path: Path) -> T.Iterator[sqlite3.Connection]:
    """Opens a connection in autocommit mode, so that transactions are explicit (`begin_write`),
    with foreign keys enforced, e.g. so that deleting a scoring deletes its scores
    (`scoring.ScoreStore`).

    Each operation opens its own connection, so that the database can be used from several
    threads.
    """
    connection = sqlite3.connect(database_path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
    finally:
        connection.close()


def initialize_schema(database_path: Path, schema: str, schema_version: int,
                      database_name: str, migrations: T.Mapping[int, Migration]) -> None:
    """Creates the tables of a new database, or migrates a database of an older schema version,
    and checks that its tables are those of `schema` (`check_schema`).

    A migration first copies the database to a backup file next to it, e.g.
    'archive.v1.2026-10-12T150000Z.backup.sqlite3'. It then runs the migrations from the version
    of the database to `schema_version` in one transaction, with foreign keys off, as SQLite's
    procedure for changing tables requires, and checks the foreign keys and the tables before the
    commit.

    Args:
        schema: The SQL statements that create the tables of `schema_version`.
        schema_version: Stored as the `PRAGMA user_version` of the database.
        database_name: What the database holds, for the messages, e.g. 'archive'.
        migrations: Version v -> the migration from v to v + 1, for each version that a database
            in use can have.

    Raises:
        ValueError: If the database has a newer schema version, one without migrations, or tables
            that differ from `schema`, e.g. of a version that was changed in place.
    """
    with connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        num_tables = connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]
        if version == 0 and not num_tables:
            connection.executescript(f"BEGIN; {schema} PRAGMA user_version = {schema_version};"
                                     f" COMMIT;")
        elif version == schema_version:
            check_schema(connection, schema, database_name)
        else:
            _migrate(connection, database_path, version, schema, schema_version, database_name,
                     migrations)


def _migrate(connection: sqlite3.Connection, database_path: Path, version: int, schema: str,
             schema_version: int, database_name: str,
             migrations: T.Mapping[int, Migration]) -> None:
    """Migrates a database from `version` to `schema_version` (see `initialize_schema`)."""
    if version > schema_version:
        raise ValueError(f"The {database_name} database {database_path} has the schema version"
                         f" {version}, which is newer than that of this server"
                         f" ({schema_version}). Use a newer server.")
    if any(v not in migrations for v in range(version, schema_version)):
        raise ValueError(f"The {database_name} database {database_path} has the schema version"
                         f" {version}, which cannot be migrated to {schema_version}. Use a new"
                         f" data directory.")
    backup_path = _back_up_database(connection, database_path, version)
    _logger.warning("Migrating the %s database %s from schema version %d to %d. The backup is"
                    " %s.", database_name, database_path, version, schema_version, backup_path)
    connection.execute("PRAGMA foreign_keys = OFF")  # Has no effect in a transaction.
    try:
        with _write_transaction(connection):
            for from_version in range(version, schema_version):
                migrations[from_version](connection)
            if violations := connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError(f"The migration of the {database_name} database breaks foreign"
                                 f" keys: {[tuple(v) for v in violations]}.")
            check_schema(connection, schema, database_name)
            connection.execute(f"PRAGMA user_version = {schema_version}")
    finally:
        connection.execute("PRAGMA foreign_keys = ON")


def _back_up_database(connection: sqlite3.Connection, database_path: Path, version: int) -> Path:
    """Copies the database to a file next to it, named after its version and the time. A backup
    of the same second is replaced, which is of the same database, since a failed migration is
    rolled back."""
    backup_path = database_path.with_name(
        f"{database_path.stem}.v{version}.{get_utc_now():%Y-%m-%dT%H%M%SZ}.backup"
        f"{database_path.suffix}")
    with contextlib.closing(sqlite3.connect(backup_path)) as backup:
        connection.backup(backup)
    return backup_path


def _describe_schema(connection: sqlite3.Connection,
                     names: T.Collection[str] | None = None) -> dict[str, tuple]:
    """Describes the tables (columns and foreign keys, without CHECK constraints) and indexes
    (table, uniqueness and columns) of a database, those in `names`, or all."""
    descriptions = {}
    for row in connection.execute("SELECT type, name, tbl_name FROM sqlite_master"
                                  " WHERE type IN ('table', 'index') AND sql IS NOT NULL"
                                  " AND name NOT LIKE 'sqlite_%'"):
        kind, name, table = row
        if names is not None and name not in names:
            continue
        if kind == "table":
            descriptions[name] = (
                kind, [tuple(c) for c in connection.execute(f"PRAGMA table_info({name})")],
                # Without the id of the foreign key, which is its position.
                sorted(tuple(k)[1:] for k in connection.execute(
                    f"PRAGMA foreign_key_list({name})")))
        else:
            is_unique = next(i["unique"] for i in connection.execute(f"PRAGMA index_list({table})")
                             if i["name"] == name)
            descriptions[name] = (kind, table, is_unique,
                                  [c["name"] for c in connection.execute(
                                      f"PRAGMA index_info({name})")])
    return descriptions


def check_schema(connection: sqlite3.Connection, schema: str, database_name: str) -> None:
    """Checks that the tables and indexes that `schema` creates are in the database as `schema`
    creates them. The database can have others.

    Raises:
        ValueError: For each that differs or is missing.
    """
    with contextlib.closing(sqlite3.connect(":memory:")) as expected_connection:
        expected_connection.row_factory = sqlite3.Row
        expected_connection.executescript(schema)
        expected = _describe_schema(expected_connection)
    actual = _describe_schema(connection, expected)
    if differing := sorted(n for n in expected if actual.get(n) != expected[n]):
        raise ValueError(f"The tables or indexes {differing} of the {database_name} database"
                         f" differ from those of its schema version, e.g. since the database is"
                         f" of an older server whose schema was changed without a migration.")


@contextlib.contextmanager
def begin_write(database_path: Path) -> T.Iterator[sqlite3.Connection]:
    """Opens a write transaction, which is committed at the end of the block, or rolled back on
    an exception. It holds the database lock from the start, so that a check and the write that
    depends on it are not interleaved with another write."""
    with connect(database_path) as connection, _write_transaction(connection):
        yield connection


@contextlib.contextmanager
def _write_transaction(connection: sqlite3.Connection) -> T.Iterator[None]:
    """Runs the block in a write transaction of `connection` (see `begin_write`)."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        # SQLite has already rolled back after some errors, e.g. a full disk.
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")
