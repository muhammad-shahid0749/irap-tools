"""Tests of `irap_data.reports.segment_map`."""

from irap_data.reports.segment_map import compute_split_segment_maps
from irap_data.reports.statistics import compute_dataset_statistics


def test_split_segment_maps_mark_the_selected_segments(metadata):
    statistics = compute_dataset_statistics(metadata, "vietnam", model_context_offsets=(0, -1))
    val, unlabeled_val = compute_split_segment_maps(metadata, statistics, "model")
    # S3 has no image. S0 and T0 have no predecessor, and S4 needs S3.
    assert val.segment_ids == ("S0", "S1", "S2", "S4", "S5", "T0", "T1")
    assert [sid for sid, s in zip(val.segment_ids, val.is_selected) if s] == \
           ["S1", "S2", "S5", "T1"]
    assert val.coordinates.shape == (7, 2) and val.num_unplaced == 0
    assert not unlabeled_val.is_labeled and unlabeled_val.group == "val"
    assert unlabeled_val.segment_ids == ()
