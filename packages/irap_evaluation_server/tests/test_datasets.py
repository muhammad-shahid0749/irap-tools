import concurrent.futures
import json
import time

import irap_evaluation as ie
import pytest
from irap_data.metadata import MetaFiles
from run_helpers import with_split
from synthetic_vietnam import SPLIT, make_predictions

from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.datasets import load_dataset_contexts


def test_load_dataset_contexts_checks_the_analysis_splits(metadata_dir):
    with pytest.raises(ValueError, match=r"analysis splits \['test'\]"):
        load_dataset_contexts({"vietnam": DatasetConfig(
            dataset_dir=metadata_dir, metadata_dir=metadata_dir, analysis_splits=("test",))})


def test_unlabeled_split_is_not_scored(metadata_dir):
    splits_path = metadata_dir / MetaFiles.SPLITS
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits["unlabeled_val"] = splits[SPLIT][:15]
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    # Unlabeled segments have no road data.
    road_data_path = metadata_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA
    road_data = json.loads(road_data_path.read_text(encoding="utf-8"))
    for segment_id in splits["unlabeled_val"]:
        del road_data[segment_id]
    road_data_path.write_text(json.dumps(road_data), encoding="utf-8")
    context = load_dataset_contexts({"vietnam": DatasetConfig(
        dataset_dir=metadata_dir, metadata_dir=metadata_dir, analysis_splits=())})["vietnam"]
    predictions = make_predictions(context.metadata.vocabulary, splits["unlabeled_val"],
                                   name="m", seed=0)
    evaluation_sets, notes = context.select_evaluation_sets(
        with_split(predictions, "unlabeled_val"))
    assert evaluation_sets == []
    assert len(notes) == 3 and all("has no labels" in note for note in notes)


def test_shared_evaluation_sets(dataset_contexts):
    context = dataset_contexts["vietnam"]
    assert context.count_shared_evaluation_set_segments(SPLIT) == {
        ie.REFERENCE_SET_NAME: context.get_reference_set(SPLIT).num_segments,
        ie.FULL_SET_NAME: len(context.metadata.splits[SPLIT])}
    assert list(context.get_submission_evaluation_sets(SPLIT, (0, -1))) == [
        ie.REFERENCE_SET_NAME, ie.FULL_SET_NAME, ie.MODEL_COMPATIBLE_SET_NAME]


def test_segment_images_must_be_in_the_dataset_directory(dataset_contexts, tmp_path):
    context = dataset_contexts["vietnam"]
    segment_id = context.metadata.splits[SPLIT][0]
    path = (context.config.dataset_dir
            / context.metadata.segment_id_to_data_paths_rel[segment_id]["rgb"])
    assert context.find_segment_image(segment_id) is None  # No image file.
    path.parent.mkdir(parents=True)
    path.write_bytes(b"png")
    assert context.find_segment_image(segment_id) == path.resolve()
    (tmp_path / "outside.png").write_bytes(b"png")
    context.metadata.segment_id_to_data_paths_rel[segment_id] = {"rgb": "../outside.png"}
    assert context.find_segment_image(segment_id) is None


def test_evaluation_sets_are_computed_once(dataset_contexts, monkeypatch):
    context = dataset_contexts["vietnam"]
    num_calls = 0
    get_shared_evaluation_sets = ie.get_shared_evaluation_sets

    def get_shared_evaluation_sets_slowly(*args):
        nonlocal num_calls
        num_calls += 1
        time.sleep(0.1)
        return get_shared_evaluation_sets(*args)

    monkeypatch.setattr(ie, "get_shared_evaluation_sets", get_shared_evaluation_sets_slowly)
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        reference_sets = list(executor.map(context.get_reference_set, [SPLIT] * 4))
    assert num_calls == 1
    assert all(s is reference_sets[0] for s in reference_sets)
