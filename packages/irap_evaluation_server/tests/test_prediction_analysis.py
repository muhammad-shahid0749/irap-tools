import dataclasses as dc
import hashlib

import irap_evaluation as ie
import numpy as np
import pytest
from irap_data.metadata import IGNORE_LABEL_INDEX
from run_helpers import add_model, make_model
from synthetic_vietnam import SPLIT

from irap_evaluation_server.vendor_libraries import LibraryFile, ensure_library_files
from irap_evaluation_server.prediction_analysis import (
    COMPARISON_OUTCOME_NAMES,
    MODEL_OUTCOME_NAMES,
    AlignedPredictionsCache,
    align_predictions,
    compute_attribute_outcomes,
    get_context_segment_ids,
    select_cell_segments,
)
from irap_evaluation_server.scoring import SCORING_SETTINGS, score_submission


@pytest.fixture
def reference_set(dataset_contexts):
    return dataset_contexts["vietnam"].get_reference_set(SPLIT)


def _invalidate_every_third(predictions: ie.Predictions, attribute: str) -> ie.Predictions:
    is_valid = np.arange(predictions.num_segments) % 3 != 0
    probs = np.where(is_valid[:, None], predictions.probs[attribute], np.nan).astype(np.float32)
    return dc.replace(predictions, probs={**predictions.probs, attribute: probs})


def _align(archive, dataset_contexts, predictions, evaluation_set):
    submission = add_model(archive, dataset_contexts, predictions)
    return align_predictions(submission, predictions, evaluation_set, SCORING_SETTINGS)


@pytest.mark.parametrize("is_hard", [False, True])
def test_confusion_matrix_is_that_of_the_scores(archive, dataset_contexts, reference_set,
                                                is_hard):
    predictions = _invalidate_every_third(
        make_model(dataset_contexts, drop_attribute="Curvature", is_hard=is_hard), "Lane width")
    aligned = _align(archive, dataset_contexts, predictions, reference_set)
    result = ie.evaluate_predictions(predictions, reference_set, null_policy="first_class",
                                     missing_attribute_policy="invalid")
    class_metrics = ie.compute_class_metrics(result, ["F1"])
    _, reports, _ = score_submission(archive, dataset_contexts["vietnam"], aligned.submission,
                                     SCORING_SETTINGS)
    for attribute in result.attributes:
        matrix = compute_attribute_outcomes(aligned, attribute).confusion_matrix
        merged = matrix[:, :-1].copy()
        merged[:, 0] += matrix[:, -1]
        np.testing.assert_array_equal(merged, class_metrics[attribute].confusion_matrix)
        np.testing.assert_array_equal(
            merged, reports[ie.REFERENCE_SET_NAME]["classes"][attribute]["confusion_matrix"])
        assert matrix[:, -1].sum() == result.num_invalid[attribute]
    assert compute_attribute_outcomes(aligned, "Lane width").confusion_matrix[:, -1].sum() > 0
    # A missing attribute is invalid in every labeled segment.
    curvature = compute_attribute_outcomes(aligned, "Curvature").confusion_matrix
    assert curvature[:, :-1].sum() == 0 and curvature[:, -1].sum() > 0


def test_outcomes_of_one_and_two_models(archive, dataset_contexts, reference_set):
    aligned_a = _align(archive, dataset_contexts,
                       _invalidate_every_third(make_model(dataset_contexts, "a"), "Curvature"),
                       reference_set)
    aligned_b = _align(archive, dataset_contexts,
                       make_model(dataset_contexts, "b", random_seed=1), reference_set)
    labels = reference_set.class_indices["Curvature"]
    predicted_a = aligned_a.predicted_indices["Curvature"]
    predicted_b = aligned_b.predicted_indices["Curvature"]
    single = compute_attribute_outcomes(aligned_a, "Curvature")
    assert single.outcome_names == MODEL_OUTCOME_NAMES
    expected = np.where(labels == IGNORE_LABEL_INDEX, "unlabeled",
                        np.where(predicted_a == IGNORE_LABEL_INDEX, "invalid",
                                 np.where(predicted_a == labels, "correct", "wrong")))
    np.testing.assert_array_equal(np.array(MODEL_OUTCOME_NAMES)[single.segment_outcomes],
                                  expected)
    assert {"correct", "wrong", "invalid", "unlabeled"} <= set(expected)

    compared = compute_attribute_outcomes(aligned_a, "Curvature", aligned_b)
    assert compared.outcome_names == COMPARISON_OUTCOME_NAMES
    is_correct_a, is_correct_b = predicted_a == labels, predicted_b == labels
    names = np.array(COMPARISON_OUTCOME_NAMES)[compared.segment_outcomes]
    is_labeled = labels != IGNORE_LABEL_INDEX
    np.testing.assert_array_equal(names == "unlabeled", ~is_labeled)
    np.testing.assert_array_equal(names == "both correct", is_correct_a & is_correct_b)
    np.testing.assert_array_equal(names == "only A correct", is_correct_a & ~is_correct_b)
    np.testing.assert_array_equal(names == "only B correct", ~is_correct_a & is_correct_b)
    np.testing.assert_array_equal(names == "neither correct",
                                  is_labeled & ~is_correct_a & ~is_correct_b)
    # The matrix is of model A.
    np.testing.assert_array_equal(compared.confusion_matrix, single.confusion_matrix)


def test_cell_segments_list_confident_predictions_first(archive, dataset_contexts,
                                                        reference_set):
    aligned = _align(archive, dataset_contexts,
                     _invalidate_every_third(make_model(dataset_contexts), "Lane width"),
                     reference_set)
    matrix = compute_attribute_outcomes(aligned, "Lane width").confusion_matrix
    labels = reference_set.class_indices["Lane width"]
    num_classes = matrix.shape[0]
    for label_index in range(num_classes):
        for predicted_index in range(num_classes + 1):
            segments = select_cell_segments(aligned, "Lane width", label_index, predicted_index)
            assert len(segments) == matrix[label_index, predicted_index]
            assert (labels[segments] == label_index).all()
            if predicted_index == num_classes:
                assert not aligned.predictions.is_valid["Lane width"][segments].any()
                assert (np.diff(segments) > 0).all()
            else:
                assert (aligned.predicted_indices["Lane width"][segments]
                        == predicted_index).all()
                confidences = aligned.predictions.probs["Lane width"][segments, predicted_index]
                assert (np.diff(confidences) <= 0).all()


def test_missed_segments_list_confident_errors_first_and_invalid_ones_last(
        archive, dataset_contexts, reference_set):
    aligned = _align(archive, dataset_contexts,
                     _invalidate_every_third(make_model(dataset_contexts), "Lane width"),
                     reference_set)
    matrix = compute_attribute_outcomes(aligned, "Lane width").confusion_matrix
    labels = reference_set.class_indices["Lane width"]
    predicted = aligned.predicted_indices["Lane width"]
    is_valid = aligned.predictions.is_valid["Lane width"]
    for label_index in range(matrix.shape[0]):
        segments = select_cell_segments(aligned, "Lane width", label_index, None)
        assert len(segments) == matrix[label_index].sum() - matrix[label_index, label_index]
        assert (labels[segments] == label_index).all()
        assert (predicted[segments] != label_index).all()
        num_valid = int(is_valid[segments].sum())
        assert is_valid[segments[:num_valid]].all()
        valid = segments[:num_valid]
        confidences = aligned.predictions.probs["Lane width"][valid, predicted[valid]]
        assert (np.diff(confidences) <= 0).all()
        assert (np.diff(segments[num_valid:]) > 0).all()


def test_aligned_predictions_cache_reads_each_file_once(archive, dataset_contexts,
                                                        reference_set, monkeypatch):
    submissions = [add_model(archive, dataset_contexts,
                             make_model(dataset_contexts, "m", seed, seed))
                   for seed in range(3)]
    read_paths = []
    read_predictions = ie.read_predictions
    monkeypatch.setattr(ie, "read_predictions",
                        lambda path: read_paths.append(path) or read_predictions(path))
    cache = AlignedPredictionsCache(archive, SCORING_SETTINGS, max_size=2)
    aligned = cache.get(submissions[0], reference_set)
    assert aligned.predictions.segment_ids == reference_set.segment_ids
    assert cache.get(submissions[0], reference_set) is aligned
    cache.get(submissions[1], reference_set)
    cache.get(submissions[2], reference_set)  # Drops the least recently used, submission 0.
    cache.get(submissions[2], reference_set)
    cache.get(submissions[0], reference_set)
    assert read_paths == [archive.get_predictions_path(submissions[i].id) for i in (0, 1, 2, 0)]


def test_context_segments_outside_the_road_are_none(dataset_contexts):
    metadata = dataset_contexts["vietnam"].metadata
    sequence = metadata.road_id_to_segment_id_sequence["road0"]
    assert get_context_segment_ids(metadata, sequence[1], (0, -1, -4)) == [
        sequence[1], sequence[0], None]
    assert get_context_segment_ids(metadata, sequence[-1], (1,)) == [None]


def test_ensure_library_files_checks_the_hashes(tmp_path):
    url_to_data = {"https://a/x.js": b"x", "https://a/y.css": b"y"}
    files = [LibraryFile(f"lib/{url[-4:]}", url, hashlib.sha256(data).hexdigest())
             for url, data in url_to_data.items()]
    fetched_urls = []

    def fetch_url(url: str) -> bytes:
        fetched_urls.append(url)
        return url_to_data[url]

    ensure_library_files(tmp_path, files, fetch_url)
    assert (tmp_path / "lib/x.js").read_bytes() == b"x"
    ensure_library_files(tmp_path, files, fetch_url)  # The files are present.
    assert fetched_urls == list(url_to_data)
    (tmp_path / "lib/x.js").write_bytes(b"changed")
    url_to_data["https://a/x.js"] = b"tampered"
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_library_files(tmp_path, files, fetch_url)
    assert (tmp_path / "lib/x.js").read_bytes() == b"changed"
