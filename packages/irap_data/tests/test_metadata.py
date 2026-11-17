"""Tests of the torch-free metadata API in `irap_data.metadata`."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from irap_data.metadata import (
    DATASET_PRESETS,
    IGNORE_LABEL_INDEX,
    ClassVocabulary,
    IRAPMetadata,
    MetaFiles,
    compute_class_occurrence_counts,
    compute_num_segments_per_class,
    compute_segment_context_ids,
    compute_segment_labels,
    get_segment_location,
    select_preset_split_segments,
    select_split_segments,
    to_irap_code,
)


def test_to_irap_code():
    assert [to_irap_code(v) for v in (3, "3", " 7 ", -1, "-1", "None", None)] == \
           [3, 3, 7, None, None, None, None]
    # A value no writer should produce is an error rather than a silently dropped label.
    for value in (True, 2.0, "2.0", "N/A", ""):
        with pytest.raises(ValueError, match="Not an iRAP code"):
            to_irap_code(value)


def test_vocabulary_equality_depends_on_order():
    vocabulary = ClassVocabulary({"a": {"x": 10, "y": 20}, "b": {"z": 1}})
    assert vocabulary == ClassVocabulary({"a": {"x": 10, "y": 20}, "b": {"z": 1}})
    assert vocabulary != ClassVocabulary({"a": {"y": 20, "x": 10}, "b": {"z": 1}})
    assert vocabulary != ClassVocabulary({"b": {"z": 1}, "a": {"x": 10, "y": 20}})
    with pytest.raises(TypeError):
        hash(vocabulary)


def test_vocabulary_follows_attribute_index_and_value_order(metadata):
    vocabulary = metadata.vocabulary
    assert vocabulary.attribute_names == ("A", "B")
    assert vocabulary.get_irap_codes("A") == (20, 10)
    assert vocabulary.get_values("A") == ("a20", "a10")
    assert vocabulary.class_counts == (2, 3)
    assert vocabulary.attr_to_value_to_class_idx["A"] == {"a20": 0, "a10": 1}
    assert vocabulary.restrict_to_attributes(["B"]).attribute_names == ("B",)
    with pytest.raises(KeyError):
        vocabulary.restrict_to_attributes(["C"])


def test_vocabulary_rejects_duplicate_codes():
    with pytest.raises(ValueError, match="duplicate"):
        ClassVocabulary({"A": {"x": 1, "y": 1}})


def test_compute_segment_labels_maps_codes_and_handles_missing(metadata):
    segment_ids = ["S0", "S1", "S4", "unknown"]
    loose = compute_segment_labels(metadata.vocabulary, metadata.segment_id_to_road_data,
                                   segment_ids, allow_missing_attributes=True)
    assert loose == {"S0": [1, 1], "S1": [0, 2], "S4": [1, IGNORE_LABEL_INDEX],
                     "unknown": [IGNORE_LABEL_INDEX, IGNORE_LABEL_INDEX]}
    strict = compute_segment_labels(metadata.vocabulary, metadata.segment_id_to_road_data,
                                    segment_ids, allow_missing_attributes=False)
    assert list(strict) == ["S0", "S1"]


def test_class_counting_excludes_the_ignore_label():
    labels = np.array([0, 2, IGNORE_LABEL_INDEX, 2])
    assert compute_num_segments_per_class(labels, 3).tolist() == [1, 0, 2]
    with pytest.raises(ValueError, match="outside"):
        compute_num_segments_per_class(labels, 2)
    info = SimpleNamespace(segment_ids=["s0", "s1"], class_counts=(2,),
                           segment_id_to_labels={"s0": [1], "s1": [5], "s2": [0]},
                           attr_to_value_to_class_idx={"A": {"x": 0, "y": 1}})
    with pytest.raises(ValueError, match="Attribute 'A'"):
        compute_class_occurrence_counts(info)
    info.segment_id_to_labels["s1"] = [IGNORE_LABEL_INDEX]
    assert compute_class_occurrence_counts(info)["A"].tolist() == [0, 1]
    # s2 has labels but is not an example, e.g. for lack of a complete context window.
    assert compute_class_occurrence_counts(
        info, segment_ids=list(info.segment_id_to_labels))["A"].tolist() == [1, 1]


def test_context_stays_on_the_road_and_needs_images(metadata):
    context = compute_segment_context_ids(
        metadata.road_id_to_segment_id_sequence, ["S0", "S1", "S2", "S4", "S5", "T0", "T1"],
        context_offsets=(0, -1), available_segment_ids=metadata.segment_id_to_data_paths_rel)
    # S0 and T0 have no predecessor, and S4 needs S3, which has no image.
    assert context == {"S1": ("S1", "S0"), "S2": ("S2", "S1"), "S5": ("S5", "S4"),
                       "T1": ("T1", "T0")}


def test_select_split_segments(metadata):
    selection = select_split_segments(metadata, "val", context_offsets=(0, -1),
                                      allow_missing_attributes=True)
    assert selection.segment_ids == ("S1", "S2", "S5", "T1")
    assert selection.attr_to_num_labeled == {"A": 4, "B": 4}
    subset = select_split_segments(metadata, "val", context_offsets=(0, -1),
                                   allow_missing_attributes=True,
                                   ncontext_segment_id_subset={"S2", "T1"})
    assert subset.segment_ids == ("S2", "T1")
    with pytest.raises(ValueError, match="Invalid split"):
        select_split_segments(metadata, "test", context_offsets=(0,),
                              allow_missing_attributes=True)


def test_select_preset_split_segments_uses_reference_offsets(metadata):
    # The -4..0 Vietnam reference window of S4 and S5 includes S3, which has no image, and r1 is
    # too short.
    assert select_preset_split_segments(metadata, "vietnam", "val").segment_ids == ()
    model = select_preset_split_segments(metadata, "vietnam", "val", context_offsets=(0,))
    assert model.segment_ids == ("S0", "S1", "S2", "S4", "S5", "T0", "T1")
    with pytest.raises(ValueError, match="Unknown iRAP dataset"):
        select_preset_split_segments(metadata, "atlantis", "val")


def test_select_split_segments_matches_irap_dataset(metadata_dir, metadata):
    pytest.importorskip("torch")
    from irap_data import IRAPDataset

    for offsets in [(0,), (0, -1), (1, 0)]:
        dataset = IRAPDataset(metadata_dir / "images", "val", metadata_dir=metadata_dir,
                              context_offsets=offsets, allow_missing_attributes=True)
        selection = select_split_segments(metadata, "val", context_offsets=offsets,
                                          allow_missing_attributes=True)
        assert tuple(dataset.segment_ids) == selection.segment_ids
        assert dataset.info.attr_to_num_labeled == selection.attr_to_num_labeled


def test_dataset_presets_cover_the_dataset_factories():
    pytest.importorskip("torch")
    from irap_data import IRAP_DATASET_FACTORIES

    assert set(DATASET_PRESETS) == set(IRAP_DATASET_FACTORIES)


def test_get_segment_location_reads_both_layouts():
    vietnam = get_segment_location({"section": "s", "distance_km": 0.04, "length_km": 0.02,
                                    "lat": 1.5, "lon": 2.5})
    bh = get_segment_location({"metadata": {"Section": "s", "Distance": "0.04",
                                            "Length": "0.02", "Latitude": "1.5",
                                            "Longitude": "2.5"}})
    assert vietnam == bh
    assert vietnam.distance_m == pytest.approx(40.0)
    with pytest.raises(ValueError, match="location layout"):
        get_segment_location({"required_attributes": {}})


def test_segment_coordinates_interpolate_between_located_segments(tmp_path: Path):
    def to_road_data(lat, lon):
        return {"section": "s", "distance_km": 0.0, "length_km": 0.02, "lat": lat, "lon": lon}

    (tmp_path / MetaFiles.UNLABELED_SEGMENT_ID_TO_LOCATION).write_text(json.dumps(
        {"u5": {"lat": 6.0, "lon": 3.0}, "v": {"lat": 7.0, "lon": 8.0}}), encoding="utf-8")
    metadata = IRAPMetadata(
        metadata_dir=tmp_path, vocabulary=ClassVocabulary({"A": {"a": 1}}), splits={},
        segment_id_to_data_paths_rel={},
        segment_id_to_road_data={"a": to_road_data(0.0, 0.0), "b": to_road_data(3.0, 6.0),
                                 "c": to_road_data(9.0, 9.0)},
        road_id_to_segment_id_sequence={"r": ["u0", "a", "u1", "u2", "b", "u3", "u4", "u5"]})
    coordinates = metadata.segment_coordinates
    # u0 is before the located segments of the road, and c and v are on no road.
    assert set(coordinates) == {"a", "u1", "u2", "b", "u3", "u4", "u5", "c", "v"}
    assert coordinates["u1"] == pytest.approx((1.0, 2.0))
    assert coordinates["u2"] == pytest.approx((2.0, 4.0))
    assert coordinates["u4"] == pytest.approx((5.0, 4.0))
    assert coordinates["u5"] == (6.0, 3.0)
    assert coordinates["c"] == (9.0, 9.0)
    assert coordinates["v"] == (7.0, 8.0)


def test_metadata_api_does_not_import_torch():
    code = ("import sys, irap_data, irap_data.metadata, irap_data.reports.statistics, "
            "irap_data.reports.statistics_report; "
            "sys.exit(int('torch' in sys.modules or 'matplotlib' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
