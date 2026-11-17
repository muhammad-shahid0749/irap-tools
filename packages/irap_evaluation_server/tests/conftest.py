import json
import sys
from pathlib import Path

import pytest
from irap_data.metadata import MetaFiles

# The synthetic metadata and predictions of the irap_evaluation tests.
sys.path.insert(0, str(Path(__file__).parents[2] / "irap_evaluation" / "tests"))

from run_helpers import OTHER_SPLIT
from synthetic_vietnam import SPLIT, write_metadata_dir

from irap_evaluation_server.archive import ModelArchive
from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.datasets import load_dataset_contexts
from irap_evaluation_server.scoring import ScoreStore, ScoringSettings
from irap_evaluation_server.scoring_worker import ScoringWorker


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    return write_metadata_dir(tmp_path / "IRAP_Vietnam")


@pytest.fixture
def dataset_contexts(metadata_dir):
    return load_dataset_contexts({"vietnam": DatasetConfig(
        dataset_dir=metadata_dir, metadata_dir=metadata_dir, analysis_splits=(SPLIT,))})


@pytest.fixture
def two_split_contexts(metadata_dir):
    """The dataset with the split `OTHER_SPLIT`, which has the segments of `SPLIT`."""
    splits_path = metadata_dir / MetaFiles.SPLITS
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits[OTHER_SPLIT] = splits[SPLIT]
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    return load_dataset_contexts({"vietnam": DatasetConfig(
        dataset_dir=metadata_dir, metadata_dir=metadata_dir, analysis_splits=())})


@pytest.fixture
def archive(tmp_path: Path) -> ModelArchive:
    return ModelArchive(tmp_path / "data")


@pytest.fixture
def settings() -> ScoringSettings:
    # Fewer resamples than the server, for speed.
    return ScoringSettings(num_resamples=50)


@pytest.fixture
def store(archive) -> ScoreStore:
    return ScoreStore(archive.data_dir)


@pytest.fixture
def worker(archive, store, dataset_contexts, settings) -> ScoringWorker:
    return ScoringWorker(archive, store, dataset_contexts, settings)
