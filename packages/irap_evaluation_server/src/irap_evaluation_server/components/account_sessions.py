"""The signed-in account of a browser (see `accounts`), in NiceGUI's per-browser user storage."""

import hmac

from nicegui import app

from ..accounts import Account, AccountStore, can_write

_SESSION_KEY = "account"


class AccountSessions:
    """The sessions of the accounts. The methods use the user storage of the current page or
    request, so they cannot be called in the thread of `run.io_bound`.

    The account is read from the store on each check, so that a change of its permission applies
    at once. A session ends when its account is removed (`Account.session_key`).
    """

    def __init__(self, accounts: AccountStore):
        self.accounts = accounts

    def get_account(self) -> Account | None:
        """Returns the signed-in account, or None for a visitor."""
        session = app.storage.user.get(_SESSION_KEY)
        if session is None:
            return None
        account = self.accounts.find_account(session["name"])
        if account is None or not hmac.compare_digest(account.session_key,
                                                      session["session_key"]):
            return None
        return account

    def require_account_name(self) -> str:
        """Returns the name of the signed-in account, e.g. as the actor of a change of an account,
        whose permission the store checks (`AccountStore.set_permission`).

        Raises:
            ValueError: If no one is signed in, e.g. after a sign-out in another tab.
        """
        if (account := self.get_account()) is None:
            raise ValueError("You are not signed in.")
        return account.name

    def require_writer_name(self) -> str:
        """Returns the name of the signed-in account if it can write, as the actor of a change of
        the archive.

        Raises:
            ValueError: If no one is signed in or the account cannot write.
        """
        if (account := self.get_account()) is None:
            raise ValueError("You are not signed in. Sign in to change the archive.")
        if not can_write(account):
            raise ValueError("Your account can only view. An admin can give it write"
                             " permission.")
        return account.name

    def sign_in(self, account: Account) -> None:
        app.storage.user[_SESSION_KEY] = {"name": account.name,
                                          "session_key": account.session_key}

    def sign_out(self) -> None:
        app.storage.user.pop(_SESSION_KEY, None)
