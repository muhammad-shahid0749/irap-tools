import pytest

from irap_evaluation_server.accounts import AccountStore
from irap_evaluation_server.display_settings import DisplaySettingsStore
from irap_evaluation_server.method_ranking import DEFAULT_SORT_METRIC

PASSWORD = "correct horse battery"


@pytest.fixture
def accounts(tmp_path):
    accounts = AccountStore(tmp_path)
    accounts.register("Al", PASSWORD)  # The admin.
    accounts.register("Bo", PASSWORD)
    accounts.set_permission("Bo", "write", "Al")
    return accounts


def test_hide_and_show_metrics(tmp_path, accounts):
    store = DisplaySettingsStore(tmp_path, accounts)
    assert store.get_hidden_metrics() == frozenset()
    assert store.set_metric_hidden("aBrier", True, "Al") == {"aBrier"}
    assert store.set_metric_hidden("amP", True, "Al") == {"aBrier", "amP"}
    assert store.set_metric_hidden("amP", True, "Al") == {"aBrier", "amP"}
    assert store.set_metric_hidden("aBrier", False, "Al") == {"amP"}
    # Opened again, e.g. after a restart.
    assert DisplaySettingsStore(tmp_path, accounts).get_hidden_metrics() == {"amP"}


@pytest.mark.parametrize("metric_name, actor, message", [
    ("aBrier", "Bo", "Only admins"),
    ("aBrier", "Cid", "Only admins"),  # No such account.
    (DEFAULT_SORT_METRIC, "Al", "cannot be hidden"),
    ("aUnknown", "Al", "cannot be hidden"),
])
def test_set_metric_hidden_refuses(tmp_path, accounts, metric_name, actor, message):
    store = DisplaySettingsStore(tmp_path, accounts)
    with pytest.raises(ValueError, match=message):
        store.set_metric_hidden(metric_name, True, actor)
    assert store.get_hidden_metrics() == frozenset()
