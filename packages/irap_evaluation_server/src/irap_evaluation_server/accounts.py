"""The accounts of the users, who register in the app.

Anyone can view the archive, also without an account. The permission of an account
(`Permission`) says what else it may do: 'view' nothing else, 'write' also change the archive
and see the action log, and 'admin' also download the per-class scores of the protected splits
(`config.DatasetConfig.is_protected_split`), change the permissions of the accounts and remove
them, and choose the shown metrics (`display_settings`). The first account to register is an
admin, and later ones can view until an admin gives them more. There is always an admin.

The accounts are in `accounts.sqlite3` in the data directory, apart from the archive. Passwords
are stored as scrypt hashes. Account names are unique regardless of the case of ASCII letters.
"""

import dataclasses as dc
import hashlib
import hmac
import secrets
import sqlite3
import typing as T
from datetime import datetime
from pathlib import Path

from .config import DatasetConfig
from .database import Migration, begin_write, connect, get_utc_now, initialize_schema

Permission = T.Literal["view", "write", "admin"]
#: From the least to the most.
PERMISSIONS: tuple[Permission, ...] = T.get_args(Permission)
PERMISSION_DESCRIPTIONS: T.Mapping[Permission, str] = {
    "view": "Can view, like visitors without an account.",
    "write": "Can also upload, replace and delete files, delete models and methods, edit"
             " descriptions, create ensembles and see the action log.",
    "admin": "Can also download the per-class scores of the splits that are not analysed, e.g."
             " test, change the permissions of the accounts and remove them, and choose the"
             " shown metrics.",
}

DATABASE_FILE_NAME = "accounts.sqlite3"
MIN_PASSWORD_LENGTH = 8
MAX_ACCOUNT_NAME_LENGTH = 64
#: The version of the database schema (`PRAGMA user_version`). A database of an older version is
#: migrated (`_MIGRATIONS`).
SCHEMA_VERSION = 1
#: Version v -> the migration from v to v + 1 (`database.initialize_schema`).
_MIGRATIONS: dict[int, Migration] = {}
# The scrypt costs (RFC 7914), about 16 MiB and 50 ms per hash.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1
_SCRYPT_MAX_MEMORY = 64 * 2 ** 20
_HASH_LENGTH = 32
#: The hash of a random password that no account has, checked for an unknown name (`verify`).
_DUMMY_HASH = ("scrypt$16384$8$1$e00ff21556850a8850592442ecbced8b"
               "$5fb435b9ca946bd7a9823a413bb5680274aa86ef47420783e3353d5f79cf9c30")

_SCHEMA = f"""
CREATE TABLE accounts (
    name TEXT PRIMARY KEY COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    permission TEXT NOT NULL CHECK (permission IN {PERMISSIONS}),
    registered_at TEXT NOT NULL
);
"""


@dc.dataclass(frozen=True)
class Account:
    """
    Attributes:
        name: The name in the action log.
        password_hash: See `hash_password`.
        registered_at: In UTC.
    """

    name: str
    password_hash: str
    permission: Permission
    registered_at: datetime

    @property
    def session_key(self) -> str:
        """Identifies the password without revealing its hash, so that a session ends when the
        account is removed, also if an account of the same name registers again."""
        return hashlib.sha256(self.password_hash.encode("ascii")).hexdigest()


def can_write(account: Account | None) -> bool:
    """Returns whether an account, or a visitor (None), may change the archive."""
    return account is not None and account.permission in ("write", "admin")


def is_admin(account: Account | None) -> bool:
    return account is not None and account.permission == "admin"


def can_see_class_scores(account: Account | None, config: DatasetConfig, split: str) -> bool:
    """Returns whether an account, or a visitor (None), can see the per-class scores of a split,
    which are only for admins on a protected split (`DatasetConfig.is_protected_split`)."""
    return is_admin(account) or not config.is_protected_split(split)


def hash_password(password: str) -> str:
    """Hashes a password with a new salt, as 'scrypt$N$r$p$<salt hex>$<hash hex>'."""
    salt = secrets.token_bytes(16)
    password_hash = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R,
                                   p=_SCRYPT_P, maxmem=_SCRYPT_MAX_MEMORY, dklen=_HASH_LENGTH)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${password_hash.hex()}"


def is_password_correct(password: str, password_hash: str) -> bool:
    """
    Raises:
        ValueError: If `password_hash` is not of the form of `hash_password`.
    """
    try:
        kind, n, r, p, salt, expected = password_hash.split("$")
        if kind != "scrypt":
            raise ValueError
        expected_bytes = bytes.fromhex(expected)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n),
                                r=int(r), p=int(p), maxmem=_SCRYPT_MAX_MEMORY,
                                dklen=len(expected_bytes))
    except ValueError:
        raise ValueError("The password hash is not of the form"
                         " 'scrypt$N$r$p$<salt hex>$<hash hex>'.") from None
    return hmac.compare_digest(actual, expected_bytes)


def check_account_name(name: str) -> str:
    """Strips an account name.

    Raises:
        ValueError: If it is empty, longer than `MAX_ACCOUNT_NAME_LENGTH`, or has characters
            that are not printable.
    """
    name = name.strip()
    if not name or len(name) > MAX_ACCOUNT_NAME_LENGTH or not name.isprintable():
        raise ValueError(f"An account name must have 1 to {MAX_ACCOUNT_NAME_LENGTH} printable"
                         f" characters, got {name!r}.")
    return name


def check_new_password(password: str) -> None:
    """
    Raises:
        ValueError: If it is shorter than `MIN_PASSWORD_LENGTH`.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"A password must have at least {MIN_PASSWORD_LENGTH} characters.")


def _to_account(row: sqlite3.Row) -> Account:
    return Account(name=row["name"], password_hash=row["password_hash"],
                   permission=row["permission"],
                   registered_at=datetime.fromisoformat(row["registered_at"]))


def _find_account(connection: sqlite3.Connection, name: str) -> Account | None:
    row = connection.execute("SELECT * FROM accounts WHERE name = ?", (name,)).fetchone()
    return None if row is None else _to_account(row)


def _check_name_is_free(connection: sqlite3.Connection, name: str) -> None:
    """
    Raises:
        ValueError: If an account has the name.
    """
    if _find_account(connection, name) is not None:
        raise ValueError(f"The account name {name!r} is taken.")


def _get_account_changed_by_admin(connection: sqlite3.Connection, name: str,
                                  actor: str) -> Account:
    """Returns the account that the admin `actor` changes.

    Raises:
        ValueError: If `actor` is not an admin, e.g. after another admin changed it.
        LookupError: If there is no such account.
    """
    if not is_admin(_find_account(connection, actor)):
        raise ValueError("Only admins can change accounts.")
    if (account := _find_account(connection, name)) is None:
        raise LookupError(f"There is no account {name!r}.")
    return account


def _has_accounts(connection: sqlite3.Connection) -> bool:
    return connection.execute("SELECT 1 FROM accounts").fetchone() is not None


def _check_has_admin(connection: sqlite3.Connection) -> None:
    """
    Raises:
        ValueError: If no account is an admin, so that the change in the transaction is rolled
            back.
    """
    if connection.execute("SELECT 1 FROM accounts WHERE permission = 'admin'").fetchone() is None:
        raise ValueError("There must remain an admin. Make another account an admin first.")


class AccountStore:
    """The accounts database of a data directory (see the module docstring), created on first use.

    Each operation opens its own database connection, so the store can be used from several
    threads.

    A database of an older schema version is migrated (`database.initialize_schema`).

    Raises:
        ValueError: If the database cannot be migrated to `SCHEMA_VERSION`, or its tables differ
            from those of its version (see `database.initialize_schema`).
    """

    def __init__(self, data_dir: Path):
        self.database_path = data_dir / DATABASE_FILE_NAME
        initialize_schema(self.database_path, _SCHEMA, SCHEMA_VERSION, "accounts", _MIGRATIONS)

    def list_accounts(self) -> list[Account]:
        """Returns the accounts in the order of registration."""
        with connect(self.database_path) as connection:
            rows = connection.execute("SELECT * FROM accounts ORDER BY rowid").fetchall()
        return [_to_account(row) for row in rows]

    def has_accounts(self) -> bool:
        with connect(self.database_path) as connection:
            return _has_accounts(connection)

    def find_account(self, name: str) -> Account | None:
        with connect(self.database_path) as connection:
            return _find_account(connection, name)

    def verify(self, name: str, password: str) -> Account | None:
        """Returns the account if the password is correct, otherwise None. Takes about as long
        for an unknown name, so that the time does not reveal which names exist."""
        account = self.find_account(name)
        if account is None:
            is_password_correct(password, _DUMMY_HASH)
            return None
        return account if is_password_correct(password, account.password_hash) else None

    def register(self, name: str, password: str) -> Account:
        """Adds an account: an admin if it is the first one, otherwise one that can view.

        Raises:
            ValueError: See `check_account_name` and `check_new_password`, or if the name is
                taken.
        """
        name = check_account_name(name)
        check_new_password(password)
        with connect(self.database_path) as connection:  # Before the slow hashing.
            _check_name_is_free(connection, name)
        # Before the transaction, so that it does not hold the lock during the hashing.
        password_hash = hash_password(password)
        with begin_write(self.database_path) as connection:
            _check_name_is_free(connection, name)  # Also if it was registered meanwhile.
            account = Account(name, password_hash,
                              "view" if _has_accounts(connection) else "admin", get_utc_now())
            connection.execute(
                "INSERT INTO accounts (name, password_hash, permission, registered_at)"
                " VALUES (?, ?, ?, ?)",
                (account.name, account.password_hash, account.permission,
                 account.registered_at.isoformat()))
        return account

    def set_permission(self, name: str, permission: Permission, actor: str) -> Account:
        """Changes the permission of an account. Its sessions get it on their next check.

        Args:
            actor: The name of the admin who changes it.

        Returns:
            The changed account.

        Raises:
            ValueError: If `permission` is not one of `PERMISSIONS`, `actor` is not an admin, or
                no admin would remain.
            LookupError: If there is no such account.
        """
        if permission not in PERMISSIONS:
            raise ValueError(f"The permission must be one of {PERMISSIONS}, got {permission!r}.")
        with begin_write(self.database_path) as connection:
            account = _get_account_changed_by_admin(connection, name, actor)
            connection.execute("UPDATE accounts SET permission = ? WHERE name = ?",
                               (permission, account.name))
            _check_has_admin(connection)
        return dc.replace(account, permission=permission)

    def remove_account(self, name: str, actor: str) -> Account:
        """Removes an account, which ends its sessions. The action log keeps its name.

        Args:
            actor: The name of the admin who removes it.

        Returns:
            The removed account.

        Raises:
            ValueError: If `actor` is not an admin, or no admin would remain.
            LookupError: If there is no such account.
        """
        with begin_write(self.database_path) as connection:
            account = _get_account_changed_by_admin(connection, name, actor)
            connection.execute("DELETE FROM accounts WHERE name = ?", (account.name,))
            _check_has_admin(connection)
        return account
