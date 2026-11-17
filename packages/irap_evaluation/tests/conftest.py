from pathlib import Path

import pytest

from irap_data.metadata import load_irap_metadata
from irap_evaluation.evaluation_sets import get_full_set, get_reference_set

from synthetic_vietnam import SPLIT, make_predictions, write_metadata_dir


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    return write_metadata_dir(tmp_path / "IRAP_Vietnam")


@pytest.fixture
def metadata(metadata_dir):
    return load_irap_metadata(metadata_dir)


@pytest.fixture
def reference_set(metadata):
    return get_reference_set(metadata, "vietnam", SPLIT)


@pytest.fixture
def full_set(metadata):
    return get_full_set(metadata, "vietnam", SPLIT)


@pytest.fixture
def predictions(metadata):
    """Predictions of model 'm' for every segment of the split."""
    return make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]), name="m", seed=0)
