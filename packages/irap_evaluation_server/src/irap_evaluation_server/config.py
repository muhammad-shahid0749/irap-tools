"""The server configuration, read from a TOML file.

Example::

    data_dir = "/srv/irap_eval"   # the archive
    host = "0.0.0.0"              # optional, default 127.0.0.1
    port = 8600                   # optional
    is_served_over_https = true   # optional, default false

    [datasets.vietnam]            # a key of irap_data.DATASET_PRESETS
    dataset_dir = "/data/IRAP_Vietnam"
    analysis_splits = ["train", "val"]

    [datasets.bh]
    dataset_dir = "/data/IRAP_BIH"
    metadata_dir = "/data/IRAP_BIH_METADATA"   # optional, default dataset_dir
    analysis_splits = ["train", "val"]
"""

import dataclasses as dc
import tomllib
import typing as T
from pathlib import Path

from irap_data.metadata import DATASET_PRESETS

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8600
DEFAULT_IS_SERVED_OVER_HTTPS = False


@dc.dataclass(frozen=True)
class DatasetConfig:
    """A dataset that the server scores submissions on.

    Attributes:
        dataset_dir: The dataset directory, which the image paths of the metadata are relative
            to (`segment_id_to_data_paths_rel.json`).
        metadata_dir: The metadata directory of the release (see `irap_data.load_irap_metadata`),
            e.g. `dataset_dir` for Vietnam.
        analysis_splits: The splits of the Analysis page, e.g. without test to keep its labels
            unseen. All splits are scored. The others are protected (`is_protected_split`).
    """

    dataset_dir: Path
    metadata_dir: Path
    analysis_splits: tuple[str, ...]

    def is_protected_split(self, split: str) -> bool:
        """Returns whether the labels of a split are protected: it is not one of
        `analysis_splits`, e.g. test. Everyone sees its per-attribute and averaged scores, but
        only admins its per-class scores, and no one its segments on the Analysis page."""
        return split not in self.analysis_splits


@dc.dataclass(frozen=True)
class ServerConfig:
    """The configuration of a server instance.

    Attributes:
        data_dir: The directory of the archive (see `archive.ModelArchive`).
        datasets: Dataset name, a key of `irap_data.DATASET_PRESETS` and the `dataset` of the
            prediction headers -> its configuration.
        is_served_over_https: Whether users open the app over HTTPS, e.g. through a reverse
            proxy. Then the browser sends the session cookie, which identifies a signed-in
            account, only over HTTPS.
    """

    data_dir: Path
    datasets: T.Mapping[str, DatasetConfig]
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    is_served_over_https: bool = DEFAULT_IS_SERVED_OVER_HTTPS


def _check_keys(table: T.Mapping[str, T.Any], required: set[str], optional: set[str],
                where: str) -> None:
    if missing := required - table.keys():
        raise ValueError(f"{where}: missing {sorted(missing)}.")
    if unknown := table.keys() - required - optional:
        raise ValueError(f"{where}: unknown {sorted(unknown)}.")


def _check_type(value: T.Any, expected_type: type, where: str) -> None:
    # bool is an int subclass, but `port = true` is a mistake.
    if not isinstance(value, expected_type) or (expected_type is int and isinstance(value, bool)):
        raise ValueError(f"{where}: expected {expected_type.__name__}, got {value!r}.")


def _to_dataset_config(table: T.Any, base_dir: Path, where: str) -> DatasetConfig:
    _check_type(table, dict, where)
    _check_keys(table, {"dataset_dir", "analysis_splits"}, {"metadata_dir"}, where)
    for key in ("dataset_dir", "metadata_dir"):
        if key in table:
            _check_type(table[key], str, f"{where}.{key}")
    splits = table["analysis_splits"]
    _check_type(splits, list, f"{where}.analysis_splits")
    for split in splits:
        _check_type(split, str, f"{where}.analysis_splits")
    return DatasetConfig(dataset_dir=base_dir / table["dataset_dir"],
                         metadata_dir=base_dir / table.get("metadata_dir", table["dataset_dir"]),
                         analysis_splits=tuple(splits))


def load_server_config(path: str | Path) -> ServerConfig:
    """Reads and checks a configuration file (see the module docstring).

    Relative paths are relative to the directory of the file. The splits are checked against the
    metadata when it is loaded (`datasets.load_dataset_contexts`).

    Raises:
        ValueError: If the file is not valid TOML, a key is missing or unknown, a value has the
            wrong type, there are no datasets, or a dataset is not a key of
            `irap_data.DATASET_PRESETS`.
    """
    path = Path(path)
    with path.open("rb") as file:
        table = tomllib.load(file)
    base_dir = path.parent.resolve()
    _check_keys(table, {"data_dir", "datasets"}, {"host", "port", "is_served_over_https"},
                str(path))
    _check_type(table["data_dir"], str, f"{path}: data_dir")
    _check_type(table.get("host", DEFAULT_HOST), str, f"{path}: host")
    _check_type(table.get("port", DEFAULT_PORT), int, f"{path}: port")
    _check_type(table.get("is_served_over_https", DEFAULT_IS_SERVED_OVER_HTTPS), bool,
                f"{path}: is_served_over_https")
    _check_type(table["datasets"], dict, f"{path}: datasets")
    if not table["datasets"]:
        raise ValueError(f"{path}: no datasets.")
    if unknown := table["datasets"].keys() - DATASET_PRESETS.keys():
        raise ValueError(f"{path}: unknown datasets {sorted(unknown)}."
                         f" Choose from {sorted(DATASET_PRESETS)}.")
    return ServerConfig(
        data_dir=base_dir / table["data_dir"],
        datasets={name: _to_dataset_config(t, base_dir, f"{path}: datasets.{name}")
                  for name, t in table["datasets"].items()},
        host=table.get("host", DEFAULT_HOST), port=table.get("port", DEFAULT_PORT),
        is_served_over_https=table.get("is_served_over_https", DEFAULT_IS_SERVED_OVER_HTTPS))
