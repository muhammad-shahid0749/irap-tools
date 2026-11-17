"""Synthetic predictions, their uploads and scoring, for the tests."""

import dataclasses as dc
import math
from pathlib import Path

import irap_evaluation as ie
import numpy as np
import pytest
from irap_evaluation.reports.evaluation_report import to_labeled_metric_values
from synthetic_vietnam import SPLIT, make_predictions, to_hard_predictions

from irap_evaluation_server.archive import ModelArchive
from irap_evaluation_server.method_scores import select_scored_models
from irap_evaluation_server.scoring import score_predictions
from irap_evaluation_server.uploads import apply_upload, plan_upload

#: The second split of the `two_split_contexts` fixture.
OTHER_SPLIT = "test"


def with_split(predictions: ie.Predictions, split: str) -> ie.Predictions:
    return dc.replace(predictions, header=dc.replace(predictions.header, split=split))


def with_context_offsets(predictions: ie.Predictions,
                         context_offsets: tuple[int, ...]) -> ie.Predictions:
    return dc.replace(predictions,
                      header=dc.replace(predictions.header, context_offsets=context_offsets))


def with_model(predictions: ie.Predictions, **changes) -> ie.Predictions:
    """Returns the predictions with changed fields of their `irap_evaluation.ModelInfo`."""
    header = predictions.header
    return dc.replace(predictions,
                      header=dc.replace(header, model=dc.replace(header.model, **changes)))


def assert_close(a, b, label=""):
    """Asserts that metric values, intervals or floats are equal, with NaN equal to NaN."""
    if isinstance(a, ie.MetricValues):
        a, b = to_labeled_metric_values(a), to_labeled_metric_values(b)
        assert a.keys() == b.keys()
        for key in a:
            assert_close(a[key], b[key], key)
    elif dc.is_dataclass(a):
        for field in dc.fields(a):
            assert_close(getattr(a, field.name), getattr(b, field.name), f"{label}.{field.name}")
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b), label
    else:
        assert a == pytest.approx(b, rel=1e-12, abs=0), label


def assert_intervals_close(actual, expected):
    """Asserts that intervals from the stored float32 resamples equal those of irap_evaluation,
    up to the float32 rounding, for the metrics of `actual`. The rounding of the model values is
    amplified in a method's values by the Student-t draws (`combine_model_resamples`), e.g.
    about 9 times at the 97.5 % quantile of t(1) for 2 models."""
    expected = to_labeled_metric_values(expected)
    for label, interval in to_labeled_metric_values(actual).items():
        assert interval.num_undefined == expected[label].num_undefined, label
        np.testing.assert_allclose([interval.low, interval.high],
                                   [expected[label].low, expected[label].high],
                                   rtol=1e-5, atol=1e-6, err_msg=label)


def score_and_get_models(worker, submission_ids, evaluation_set="reference"):
    """Scores the submissions with the worker. Returns the scored models of all submissions."""
    worker.update_submissions([worker.archive.get_submission(i) for i in submission_ids])
    worker.run_pending()
    submissions = worker.archive.list_submissions()
    return select_scored_models(submissions, worker.get_current_scorings(submissions),
                                evaluation_set)


def score_submission_again(worker, submission, evaluation_set):
    return score_predictions(
        ie.read_predictions(worker.archive.get_predictions_path(submission.id)), evaluation_set,
        worker.settings)


def make_model(dataset_contexts, name="m", model_seed=0, random_seed=0,
               drop_attribute=None, is_hard=False, training_splits=()) -> ie.Predictions:
    """Makes predictions of a model on `SPLIT`, without `drop_attribute`, and hard if
    `is_hard`."""
    metadata = dataset_contexts["vietnam"].metadata
    predictions = make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]), name=name,
                                   seed=random_seed, model_seed=model_seed,
                                   training_splits=training_splits)
    if drop_attribute is not None:
        predictions = dc.replace(
            predictions,
            attribute_to_irap_codes={a: c for a, c in predictions.attribute_to_irap_codes.items()
                                     if a != drop_attribute},
            probs={a: p for a, p in predictions.probs.items() if a != drop_attribute})
    return to_hard_predictions(predictions) if is_hard else predictions


def write_upload(archive: ModelArchive, predictions: ie.Predictions) -> Path:
    path = archive.make_upload_path("model.predictions.parquet")
    ie.write_predictions(path, predictions)
    return path


def add_model(archive, dataset_contexts, predictions, description=None):
    """Uploads and stores predictions, replacing the submission of their split, if any.
    Returns the new submission."""
    planned = plan_upload(archive, dataset_contexts, write_upload(archive, predictions),
                          file_name="model.parquet", description=description)
    [submission] = apply_upload(archive, planned, submitter="Al")
    return submission


def add_models_of_different_context_offsets(archive, dataset_contexts):
    """Adds a model of each of the methods "m" and "n", whose context offsets differ, so that
    their model-compatible sets differ. Returns the submissions."""
    return [add_model(archive, dataset_contexts,
                      with_context_offsets(make_model(dataset_contexts, name), offsets))
            for name, offsets in (("m", (0, -1)), ("n", (0, -2)))]
