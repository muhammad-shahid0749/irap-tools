import concurrent.futures

import pytest

from irap_evaluation_server.accounts import (
    MIN_PASSWORD_LENGTH,
    AccountStore,
    can_write,
    hash_password,
    is_admin,
    is_password_correct,
)
from irap_evaluation_server.components.request_checks import is_same_origin
from irap_evaluation_server.components.routes import to_local_path

PASSWORD = "correct horse battery"


def test_hash_password():
    password_hash = hash_password(PASSWORD)
    assert is_password_correct(PASSWORD, password_hash)
    assert not is_password_correct(PASSWORD + "!", password_hash)
    assert hash_password(PASSWORD) != password_hash  # A new salt.
    with pytest.raises(ValueError, match="not of the form"):
        is_password_correct(PASSWORD, "plain")


def test_register_and_verify(tmp_path):
    store = AccountStore(tmp_path)
    assert store.list_accounts() == []
    admin = store.register(" Al ", PASSWORD)
    assert (admin.name, admin.permission) == ("Al", "admin")
    viewer = store.register("Bo", PASSWORD)
    assert viewer.permission == "view" and not can_write(viewer) and not can_write(None)
    assert store.list_accounts() == [admin, viewer]
    assert store.verify("Al", PASSWORD) == admin
    assert store.verify("al", PASSWORD) == admin  # The stored name.
    assert store.verify("Al", "wrong password") is None
    assert store.verify("Cid", PASSWORD) is None
    with pytest.raises(ValueError, match="taken"):
        store.register("AL", PASSWORD)


@pytest.mark.parametrize("name, password, message", [
    ("", PASSWORD, "printable"),
    ("A\nl", PASSWORD, "printable"),
    ("A" * 65, PASSWORD, "printable"),
    ("Al", "x" * (MIN_PASSWORD_LENGTH - 1), "at least"),
])
def test_register_refuses(tmp_path, name, password, message):
    with pytest.raises(ValueError, match=message):
        AccountStore(tmp_path).register(name, password)


def test_concurrent_first_registrations_make_one_admin(tmp_path):
    store = AccountStore(tmp_path)
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        accounts = list(executor.map(lambda name: store.register(name, PASSWORD),
                                     ["A", "B", "C", "D"]))
    assert sorted(a.permission for a in accounts) == ["admin", "view", "view", "view"]


def test_admin_changes_accounts(tmp_path):
    store = AccountStore(tmp_path)
    store.register("Al", PASSWORD)
    bo = store.register("Bo", PASSWORD)
    with pytest.raises(ValueError, match="Only admins"):
        store.set_permission("Bo", "write", actor="Bo")
    assert can_write(store.set_permission("Bo", "write", actor="Al"))
    assert store.find_account("Bo").permission == "write"
    with pytest.raises(ValueError, match="Only admins"):  # Write is not enough.
        store.remove_account("Al", actor="Bo")
    with pytest.raises(ValueError, match="one of"):
        store.set_permission("Bo", "owner", actor="Al")
    with pytest.raises(LookupError, match="no account"):
        store.set_permission("Cid", "write", actor="Al")

    store.remove_account("Bo", actor="Al")
    assert store.find_account("Bo") is None
    # A new account of the same name does not continue the sessions of the removed one.
    assert store.register("Bo", PASSWORD).session_key != bo.session_key


def test_last_admin_remains(tmp_path):
    store = AccountStore(tmp_path)
    store.register("Al", PASSWORD)
    store.register("Bo", PASSWORD)
    for change in (lambda: store.set_permission("Al", "write", actor="Al"),
                   lambda: store.remove_account("Al", actor="Al")):
        with pytest.raises(ValueError, match="remain an admin"):
            change()
    assert is_admin(store.find_account("Al"))  # Rolled back.

    store.set_permission("Bo", "admin", actor="Al")
    store.set_permission("Al", "view", actor="Al")
    with pytest.raises(ValueError, match="Only admins"):  # Al is no longer one.
        store.set_permission("Al", "admin", actor="Al")


@pytest.mark.parametrize("url, expected", [
    ("/models/3?split=val", "/models/3?split=val"),
    ("", "/scores"),
    ("models", "/scores"),
    ("https://example.com/", "/scores"),
    ("//example.com/", "/scores"),
    ("/\\example.com/", "/scores"),
    ("/\t/example.com/", "/scores"),
])
def test_to_local_path(url, expected):
    assert to_local_path(url, "/scores") == expected


@pytest.mark.parametrize("origin, expected", [
    (None, True),
    ("https://eval.example", True),
    ("https://EVAL.example", True),
    ("http://localhost:8600", False),
    ("https://other.eval.example", False),
    ("https://eval.example.evil", False),
    ("null", False),
])
def test_is_same_origin(origin, expected):
    assert is_same_origin(origin, "eval.example") == expected
