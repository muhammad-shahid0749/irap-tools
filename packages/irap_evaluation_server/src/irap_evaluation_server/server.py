"""The `irap-eval-server <config.toml>` command, which runs the web app.

`config.load_server_config` describes the configuration file.
"""

import argparse
import http.client
import os
import secrets
import time
import typing as T
from pathlib import Path

from .accounts import AccountStore
from .archive import ModelArchive
from .config import load_server_config
from .datasets import load_dataset_contexts
from .display_settings import DisplaySettingsStore
from .prediction_analysis import AlignedPredictionsCache
from .scoring import ScoreStore
from .scoring_worker import ScoringWorker
from .vendor_libraries import (
    MAP_LIBRARY_FILES,
    MATH_LIBRARY_FILES,
    LibraryFile,
    ensure_library_files,
)

#: The length of the key of the session cookies, in hex digits (256 bits).
_STORAGE_SECRET_LENGTH = 64


def _load_storage_secret(data_dir: Path) -> str:
    """Loads the key of NiceGUI's session cookies, created on first use, so that the
    per-browser storage (e.g. the signed-in account) survives restarts.

    Raises:
        ValueError: If the stored key is shorter than `_STORAGE_SECRET_LENGTH`, e.g. empty after
            a crash, since cookies signed with it could be forged.
    """
    path = data_dir / "storage_secret.txt"
    if not path.is_file():
        # Readable only by the server's user, since the key signs the session cookies.
        file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file.write(secrets.token_hex(_STORAGE_SECRET_LENGTH // 2))
    secret = path.read_text(encoding="utf-8").strip()
    if len(secret) < _STORAGE_SECRET_LENGTH:
        raise ValueError(f"{path} has a key of {len(secret)} characters, fewer than"
                         f" {_STORAGE_SECRET_LENGTH}. Delete it to make a new one, which signs"
                         f" out all users.")
    return secret


def _download_libraries(vendor_dir: Path, files: T.Iterable[LibraryFile], name: str,
                        consequence: str) -> str | None:
    """Downloads the files of libraries (`ensure_library_files`) and returns None, or prints why
    they could not be downloaded and returns an error message for the pages.

    Args:
        name: The subject of the messages, e.g. 'The map libraries'.
        consequence: What is missing without them, e.g. 'The Analysis page shows no map.'.
    """
    try:
        ensure_library_files(vendor_dir, files)
    except (OSError, ValueError, http.client.HTTPException) as e:
        print(f"{name} could not be downloaded into {vendor_dir}: {e} {consequence}")
        # Without the path and the error, which visitors need not see.
        return f"{name} could not be downloaded when the server started."
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Runs the iRAP evaluation web app.")
    parser.add_argument("config", type=Path, help="The configuration file (TOML).")
    args = parser.parse_args(argv)
    config = load_server_config(args.config)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    # NiceGUI reads the path of its per-browser storage when it is imported.
    os.environ["NICEGUI_STORAGE_PATH"] = str(config.data_dir / "nicegui_storage")
    from nicegui import app, run, ui

    from .components.account_sessions import AccountSessions
    from .components.request_checks import SameOriginMiddleware
    from .components.routes import REGISTER_PATH, VENDOR_PATH
    from .pages.accounts import register_accounts_page
    from .pages.action_log import register_action_log_page
    from .pages.analysis import register_analysis_page
    from .pages.method import register_method_page
    from .pages.methodology import register_methodology_page
    from .pages.model import register_model_page
    from .pages.models import register_models_pages
    from .pages.scores import register_scores_page
    from .pages.shown_metrics import register_shown_metrics_page
    from .pages.sign_in import register_sign_in_pages

    vendor_dir = config.data_dir / "vendor"
    map_error = _download_libraries(vendor_dir, MAP_LIBRARY_FILES, "The map libraries",
                                    "The Analysis page shows no map.")
    math_error = _download_libraries(vendor_dir, MATH_LIBRARY_FILES, "MathJax",
                                     "The Methodology page shows its formulas as TeX.")
    start_time_s = time.perf_counter()
    dataset_contexts = load_dataset_contexts(config.datasets)
    print(f"Loaded the metadata of {', '.join(dataset_contexts)} in"
          f" {time.perf_counter() - start_time_s:.1f} s.")
    archive = ModelArchive(config.data_dir)
    if num_removed := archive.remove_uploads():
        print(f"Removed {num_removed} unfinished uploads.")
    scoring_worker = ScoringWorker(archive, ScoreStore(config.data_dir), dataset_contexts)
    # Scores what is new or out of date, e.g. after a metadata update, in the background.
    scoring_worker.update_all_submissions()
    scoring_worker.start()
    # Without arguments: NiceGUI passes the app to a handler with one parameter.
    app.on_shutdown(lambda: scoring_worker.stop())

    accounts = AccountStore(config.data_dir)
    if not accounts.has_accounts():
        print(f"There are no accounts yet. The first to register at {REGISTER_PATH} becomes the"
              f" admin.")
    display_settings = DisplaySettingsStore(config.data_dir, accounts)
    display_settings.warn_about_unknown_metrics()
    sessions = AccountSessions(accounts)
    # The session cookie is also sent with the requests of pages on other hosts of the domain.
    app.add_middleware(SameOriginMiddleware)
    register_sign_in_pages(sessions)
    register_accounts_page(sessions)
    register_shown_metrics_page(display_settings, sessions)
    register_scores_page(dataset_contexts, scoring_worker, sessions, display_settings)
    register_analysis_page(dataset_contexts, scoring_worker,
                           AlignedPredictionsCache(archive, scoring_worker.settings), map_error,
                           sessions)
    register_models_pages(archive, dataset_contexts, scoring_worker, sessions)
    register_model_page(archive, dataset_contexts, scoring_worker, sessions, display_settings)
    register_method_page(archive, dataset_contexts, scoring_worker, sessions, display_settings)
    register_methodology_page(scoring_worker.settings, list(dataset_contexts), math_error,
                              sessions, display_settings)
    register_action_log_page(archive, sessions)
    vendor_dir.mkdir(exist_ok=True)
    app.add_static_files(VENDOR_PATH, vendor_dir)
    # The processes of `run.cpu_bound`, e.g. for coding tables, start fresh on every platform.
    # Linux defaults to "fork" before Python 3.14, which copies the state of the server's threads,
    # e.g. held locks.
    run.process_pool_start_method = "spawn"
    # No ripple effects on clicks, and no loading bar on requests.
    app.config.quasar_config["ripple"] = False
    app.config.quasar_config["loadingBar"]["skipHijack"] = True
    # No sad face on NiceGUI's error pages (`nicegui/error.py`, whose SVG has the id "Ebene_1").
    ui.add_css("div:has(> svg#Ebene_1) { display: none; }", shared=True)
    # Browsers refuse a cookie with the prefix __Host- from other hosts of the domain, e.g. one
    # that a page on another host sets before a user signs in, so that its browser would share
    # the session (session fixation). The prefix needs HTTPS.
    session_cookie = "__Host-session" if config.is_served_over_https else "session"
    ui.run(host=config.host, port=config.port, title="iRAP evaluation", favicon="🛣️",
           reload=False, show=False, storage_secret=_load_storage_secret(config.data_dir),
           session_middleware_kwargs={"https_only": config.is_served_over_https,
                                      "session_cookie": session_cookie})


if __name__ == "__main__":
    main()
