"""Tests of `irap_data.reports.statistics`."""

import json

from irap_data.metadata import MetaFiles, load_irap_metadata
from irap_data.reports.statistics import compute_dataset_statistics


def test_statistics_count_the_segments_that_a_dataset_of_an_unlabeled_split_yields(
        metadata_dir):
    (metadata_dir / MetaFiles.SPLITS).write_text(json.dumps(
        {"val": ["S0", "S1", "S2", "S4", "S5", "T1"], "unlabeled_val": ["T0"]}), encoding="utf-8")
    statistics = compute_dataset_statistics(load_irap_metadata(metadata_dir), "vietnam",
                                            model_context_offsets=(0,))
    unlabeled = statistics.splits[1]
    assert unlabeled.selections == {}
    # T0 is the first segment of r1, so only the model window (0,) is complete.
    assert unlabeled.num_selected == {"with_context": 0, "model": 1}
