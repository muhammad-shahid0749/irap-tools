from pathlib import Path

import pytest

from irap_evaluation_server.config import DEFAULT_HOST, DEFAULT_PORT, load_server_config


def write_config(config_dir: Path, text: str) -> Path:
    config_dir.mkdir(exist_ok=True)
    path = config_dir / "server.toml"
    path.write_text(text, encoding="utf-8")
    return path


VALID_CONFIG = """
data_dir = "data"

[datasets.vietnam]
dataset_dir = "IRAP_Vietnam"
analysis_splits = ["train", "val"]

[datasets.bh]
dataset_dir = "IRAP_BIH"
metadata_dir = "{metadata_dir}"
analysis_splits = []
"""


def test_load_server_config(tmp_path):
    metadata_dir = tmp_path.resolve() / "elsewhere" / "IRAP_BIH_METADATA"
    config = load_server_config(write_config(
        tmp_path / "config", VALID_CONFIG.replace("{metadata_dir}", metadata_dir.as_posix())))
    base_dir = (tmp_path / "config").resolve()
    assert config.data_dir == base_dir / "data"
    assert (config.host, config.port) == (DEFAULT_HOST, DEFAULT_PORT)
    vietnam, bh = config.datasets["vietnam"], config.datasets["bh"]
    assert vietnam.dataset_dir == vietnam.metadata_dir == base_dir / "IRAP_Vietnam"
    assert vietnam.analysis_splits == ("train", "val")
    assert bh.dataset_dir == base_dir / "IRAP_BIH"
    assert bh.metadata_dir == metadata_dir  # An absolute path is kept.
    assert bh.analysis_splits == ()


def test_load_server_config_host_and_port(tmp_path):
    config = load_server_config(write_config(
        tmp_path, 'host = "0.0.0.0"\nport = 9000\nis_served_over_https = true\n' + VALID_CONFIG))
    assert (config.host, config.port, config.is_served_over_https) == ("0.0.0.0", 9000, True)


@pytest.mark.parametrize("text, message", [
    (VALID_CONFIG.replace('data_dir = "data"', ""), "missing ['data_dir']"),
    ("colour = 1\n" + VALID_CONFIG, "unknown ['colour']"),
    ("port = true\n" + VALID_CONFIG, "expected int"),
    ('port = "80"\n' + VALID_CONFIG, "expected int"),
    ("is_served_over_https = 1\n" + VALID_CONFIG, "expected bool"),
    (VALID_CONFIG.replace("[datasets.bh]", "[datasets.croatia]"), "unknown datasets"),
    (VALID_CONFIG.replace('analysis_splits = []', ""), "missing ['analysis_splits']"),
    (VALID_CONFIG.replace('dataset_dir = "IRAP_BIH"', 'images_dir = "IRAP_BIH"'),
     "missing ['dataset_dir']"),
    (VALID_CONFIG.replace('analysis_splits = []', 'analysis_splits = "val"'), "expected list"),
    ('data_dir = "data"\ndatasets = {}\n', "no datasets"),
])
def test_load_server_config_refuses(tmp_path, text, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_server_config(write_config(tmp_path, text))
