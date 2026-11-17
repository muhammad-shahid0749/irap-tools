import sqlite3

import pytest

from irap_evaluation_server.database import check_schema, connect, initialize_schema

_SCHEMA_1 = """
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
"""
_SCHEMA_2 = """
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    seed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES models (id)
);
CREATE UNIQUE INDEX file_model ON files (model_id);
"""


def _add_seed(connection):
    connection.execute("ALTER TABLE models ADD COLUMN seed INTEGER NOT NULL DEFAULT 0")


def _add_files(connection):
    connection.execute("CREATE TABLE files (id INTEGER PRIMARY KEY,"
                       " model_id INTEGER NOT NULL REFERENCES models (id))")
    connection.execute("CREATE UNIQUE INDEX file_model ON files (model_id)")


def _read(path, query):
    with connect(path) as connection:
        return [tuple(r) for r in connection.execute(query)]


def _make_database_of_version_1(tmp_path):
    path = tmp_path / "archive.sqlite3"
    initialize_schema(path, _SCHEMA_1, 1, "archive", {})
    with connect(path) as connection:
        connection.execute("INSERT INTO models (name) VALUES ('m')")
    return path


def test_an_older_database_is_migrated_after_a_backup(tmp_path):
    path = _make_database_of_version_1(tmp_path)
    initialize_schema(path, _SCHEMA_2, 3, "archive", {1: _add_seed, 2: _add_files})
    assert _read(path, "PRAGMA user_version") == [(3,)]
    assert _read(path, "SELECT name, seed FROM models") == [("m", 0)]
    [backup] = tmp_path.glob("archive.v1.*.backup.sqlite3")
    assert _read(backup, "PRAGMA user_version") == [(1,)]
    assert _read(backup, "SELECT * FROM models") == [(1, "m")]
    # The foreign keys are enforced again.
    with connect(path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute("INSERT INTO files (model_id) VALUES (99)")


def test_a_failed_migration_changes_nothing(tmp_path):
    path = _make_database_of_version_1(tmp_path)

    def fail(connection):
        connection.execute("DELETE FROM models")
        raise RuntimeError("Failed.")

    with pytest.raises(RuntimeError):
        initialize_schema(path, _SCHEMA_2, 3, "archive", {1: _add_seed, 2: fail})
    assert _read(path, "PRAGMA user_version") == [(1,)]
    assert _read(path, "SELECT * FROM models") == [(1, "m")]
    # A migration that does not give the schema, or breaks foreign keys.
    with pytest.raises(ValueError, match=r"\['file_model', 'files'\] of the archive"):
        initialize_schema(path, _SCHEMA_2, 3, "archive", {1: _add_seed, 2: lambda c: None})

    def add_broken_file(connection):
        _add_files(connection)
        connection.execute("INSERT INTO files (model_id) VALUES (99)")

    with pytest.raises(ValueError, match="breaks foreign keys"):
        initialize_schema(path, _SCHEMA_2, 3, "archive", {1: _add_seed, 2: add_broken_file})
    assert _read(path, "PRAGMA user_version") == [(1,)]
    assert _read(path, "SELECT * FROM models") == [(1, "m")]
    for backup in tmp_path.glob("archive.v1.*.backup.sqlite3"):
        assert _read(backup, "SELECT * FROM models") == [(1, "m")]


def test_databases_that_cannot_be_migrated_are_refused(tmp_path):
    path = _make_database_of_version_1(tmp_path)
    with pytest.raises(ValueError, match="version 1, which cannot be migrated to 3"):
        initialize_schema(path, _SCHEMA_2, 3, "archive", {2: _add_files})
    with pytest.raises(ValueError, match="newer than that of this server"):
        initialize_schema(path, "", 0, "archive", {})
    # Tables without a version, e.g. of another program.
    other_path = tmp_path / "other.sqlite3"
    with connect(other_path) as connection:
        connection.execute("CREATE TABLE models (id INTEGER PRIMARY KEY)")
    with pytest.raises(ValueError, match="version 0, which cannot be migrated"):
        initialize_schema(other_path, _SCHEMA_1, 1, "archive", {})
    assert not list(tmp_path.glob("*.backup.sqlite3"))


def test_a_schema_changed_without_a_new_version_is_refused(tmp_path):
    path = _make_database_of_version_1(tmp_path)
    with pytest.raises(ValueError, match=r"\['models'\] of the archive database differ"):
        initialize_schema(path, _SCHEMA_1.replace("name TEXT", "label TEXT"), 1, "archive",
                          {})
    with connect(path) as connection:
        # Other tables of the database are not checked, but the indexes of the schema are.
        connection.execute("CREATE TABLE other (id INTEGER)")
        check_schema(connection, _SCHEMA_1, "archive")
        connection.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL"
                           " REFERENCES models (id))")
        connection.execute("CREATE INDEX file_model ON files (model_id)")  # Not unique.
        with pytest.raises(ValueError, match=r"\['file_model', 'models'\]"):
            check_schema(connection, _SCHEMA_2, "archive")
