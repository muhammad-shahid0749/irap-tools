"""Fixtures of a small metadata directory, shared by the tests of the metadata and the reports."""

import json
from pathlib import Path

import pytest

from irap_data.metadata import IRAPMetadata, MetaFiles, load_irap_metadata

# BH stores codes as strings, and class order is the value order, not the code order.
ATTRIBUTE_METADATA = {
    "attribute_to_idx": {"B": 1, "A": 0},
    "attribute_value_to_irap_number": {"A": {"a20": "20", "a10": "10"},
                                       "B": {"b1": 1, "b2": 2, "b3": 3}},
}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    """Two roads: r0 = S0..S5 (S3 has no image), r1 = T0..T1. S4 misses the code of B.

    The 'val' split has all segments, and 'unlabeled_val' none.
    """
    road_to_seq = {"r0": [f"S{i}" for i in range(6)], "r1": ["T0", "T1"]}
    all_segments = [s for seq in road_to_seq.values() for s in seq]
    codes = {sid: {"A": 10, "B": 2} for sid in all_segments}
    codes["S1"] = {"A": "20", "B": "3"}
    codes["S4"] = {"A": 10, "B": -1}
    _write_json(tmp_path / MetaFiles.ATTRIBUTE_METADATA, ATTRIBUTE_METADATA)
    _write_json(tmp_path / MetaFiles.SEGMENT_ID_TO_DATA_PATHS,
                {sid: {"rgb": f"{sid}.png"} for sid in all_segments if sid != "S3"})
    _write_json(tmp_path / MetaFiles.SEGMENT_ID_TO_ROAD_DATA, {
        sid: {"section": "r0", "distance_km": 0.02 * i, "length_km": 0.02, "lat": 1.0,
              "lon": 2.0, "required_attributes": codes[sid]}
        for i, sid in enumerate(all_segments)})
    _write_json(tmp_path / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE, road_to_seq)
    _write_json(tmp_path / MetaFiles.SPLITS, {"val": all_segments, "unlabeled_val": []})
    return tmp_path


@pytest.fixture
def metadata(metadata_dir: Path) -> IRAPMetadata:
    return load_irap_metadata(metadata_dir)
