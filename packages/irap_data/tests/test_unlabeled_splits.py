"""Tests of unlabeled splits and of labeled-sample counts in `IRAPDataset` splits."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from irap_data import make_bh_data, make_vietnam_data
from irap_data.attrs import filter_labeled_attrs
from irap_data.irap_dataset import IRAPDataset
from irap_data.metadata import IGNORE_LABEL_INDEX, MetaFiles

ATTRS = ("A", "B")
IRAP_CODES = {"A": [10, 20], "B": [30, 40, 50]}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_dataset(root: Path, coded_attrs=ATTRS) -> tuple[Path, Path]:
    """Writes one road of labeled segments L0..L3 interleaved with unlabeled U0..U3.

    The road sequence mirrors `interleave_unlabeled_into_road_sequences` of the Vietnam data
    preparation. Labeled segments have the first code of each attribute in `coded_attrs`.
    """
    data_dir, meta_dir = root / "DATA", root / "DATA_METADATA"
    labeled = [f"L{i}" for i in range(4)]
    road = [s for i in range(4) for s in (f"L{i}", f"U{i}")]
    for sid in road:
        path = data_dir / "images" / f"{sid}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.zeros((12, 16, 3), dtype=np.uint8))
    _write_json(meta_dir / MetaFiles.ATTRIBUTE_METADATA, {
        "attribute_to_idx": {a: i for i, a in enumerate(ATTRS)},
        "attribute_value_to_irap_number": {a: {f"val_{c}": c for c in IRAP_CODES[a]}
                                           for a in ATTRS}})
    _write_json(meta_dir / MetaFiles.SEGMENT_ID_TO_DATA_PATHS,
                {sid: {"rgb": f"images/{sid}.png"} for sid in road})
    _write_json(meta_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA,
                {sid: {"required_attributes": {a: IRAP_CODES[a][0] for a in coded_attrs}}
                 for sid in labeled})
    _write_json(meta_dir / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE, {"road0": road})
    _write_json(meta_dir / MetaFiles.SPLITS, {
        "train": ["L0", "L1"], "val": ["L2"], "test": ["L3"],
        "unlabeled_train": ["U0", "U1"], "unlabeled_val": ["U2"], "unlabeled_test": ["U3"],
        "unlabeled_unlocated": []})
    return data_dir, meta_dir


def _make_splits(root: Path, coded_attrs=ATTRS, **kwargs):
    data_dir, meta_dir = write_dataset(root, coded_attrs)
    # context_offsets=(0,) keeps the tests independent of road-boundary edge cases.
    return make_bh_data(dataset_dir=data_dir, metadata_dir=meta_dir, context_offsets=(0,),
                        use_ncontext_filter=False, input_dim_rgb=(16, 12, 3), **kwargs)


def test_every_split_is_loaded_including_empty_ones(tmp_path):
    splits = _make_splits(tmp_path)
    assert {k: len(v) for k, v in splits.items()} == {
        "train": 2, "val": 1, "test": 1,
        "unlabeled_train": 2, "unlabeled_val": 1, "unlabeled_test": 1, "unlabeled_unlocated": 0}


def test_targets_are_class_indices_and_ignored_in_unlabeled_splits(tmp_path):
    splits = _make_splits(tmp_path)
    # The first code of each attribute is class 0.
    assert splits["train"][0]["target"].tolist() == [0, 0]
    unlabeled = splits["unlabeled_train"][0]
    assert unlabeled["rgb"].ndim == 4  # (S, 3, H, W)
    assert unlabeled["target"].tolist() == [IGNORE_LABEL_INDEX] * len(ATTRS)


def test_unknown_subset_error_lists_available_splits(tmp_path):
    data_dir, meta_dir = write_dataset(tmp_path)
    with pytest.raises(ValueError, match=r"Available in splits.json:.*unlabeled_train"):
        IRAPDataset(data_dir, subset="not_a_real_split", metadata_dir=meta_dir,
                    context_offsets=(0,))


def test_vietnam_data_reads_the_metadata_from_the_dataset_directory(tmp_path):
    data_dir, meta_dir = write_dataset(tmp_path)
    for path in meta_dir.iterdir():
        shutil.copy(path, data_dir)
    kwargs = dict(context_offsets=(0,), input_dim_rgb=(16, 12, 3))
    assert len(make_vietnam_data(dataset_dir=data_dir, **kwargs)["train"]) == 2
    assert len(make_vietnam_data(dataset_dir=data_dir, metadata_dir=data_dir, **kwargs)) > 0
    with pytest.raises(ValueError, match="metadata_dir"):
        make_vietnam_data(dataset_dir=data_dir, metadata_dir=meta_dir, **kwargs)


# With every attribute coded, as in iRAP-BH, filtering keeps every attribute. An attribute that
# is never coded, like the BH-only attributes in iRAP-Vietnam, keeps its value vocabulary but
# is dropped, since its metrics would be NaN.
@pytest.mark.parametrize("coded_attrs", [("A", "B"), ("A",)])
def test_filter_labeled_attrs_keeps_the_coded_attributes(tmp_path, coded_attrs):
    splits = _make_splits(tmp_path, coded_attrs, allow_missing_attributes=True)
    train_info = splits["train"].info
    assert train_info.attr_to_value_to_class_idx.keys() == set(ATTRS)
    assert train_info.attr_to_num_labeled == {a: 2 if a in coded_attrs else 0 for a in ATTRS}
    assert splits["unlabeled_train"].info.attr_to_num_labeled == {a: 0 for a in ATTRS}
    assert filter_labeled_attrs(ATTRS, train_info.attr_to_num_labeled) == coded_attrs
