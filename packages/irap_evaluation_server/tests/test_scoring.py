import dataclasses as dc
import json
import math
from datetime import datetime, timezone

import irap_evaluation as ie
import numpy as np
import pytest
from irap_data.metadata import MetaFiles
from irap_evaluation.reports.evaluation_report import to_json_dict
from run_helpers import (
    add_model,
    assert_close,
    make_model,
    score_and_get_models,
    score_submission_again,
)
from synthetic_vietnam import SPLIT

from irap_evaluation_server import scoring as scoring_module
from irap_evaluation_server import scoring_worker as scoring_worker_module
from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.database import connect
from irap_evaluation_server.datasets import load_dataset_contexts
from irap_evaluation_server.method_scores import compute_method_scores
from irap_evaluation_server.scoring import (
    ScoreStore,
    ScoringSettings,
    SubmissionScoring,
    compute_scoring_code_fingerprint,
    get_unscored_reason,
    is_scoring_current,
    remove_class_scores,
)
from irap_evaluation_server.scoring_worker import ScoringWorker


def test_scoring_equals_irap_evaluation(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    [model] = score_and_get_models(worker, [submission.id])

    scoring = store.get_scoring(submission.id)
    assert (scoring.status, scoring.message) == ("scored", "")
    assert is_scoring_current(scoring, submission, dataset_contexts["vietnam"], worker.settings)
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    result = score_submission_again(worker, submission, reference_set)
    assert model.scores.attributes == result.attributes
    assert_close(model.scores.metrics, result.metrics)
    assert model.scores.num_invalid == result.num_invalid
    # The JSON round trip keeps undefined values as None, like the report.
    assert store.get_score_report(submission.id, "reference") == json.loads(
        json.dumps(to_json_dict(result)))
    # The context offsets (0, -1) select more segments than the reference window (-4..0).
    model_compatible_set = dataset_contexts["vietnam"].get_model_compatible_set(SPLIT, (0, -1))
    assert model_compatible_set.num_segments > reference_set.num_segments
    assert scoring.notes == ()
    compatible_scores = scoring.get_scores(ie.MODEL_COMPATIBLE_SET_NAME)
    assert compatible_scores.context_offsets == (0, -1)
    assert_close(compatible_scores.metrics,
                 score_submission_again(worker, submission, model_compatible_set).metrics)


def test_remove_class_scores(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    report = store.get_score_report(submission.id, "reference")
    original = json.loads(json.dumps(report))

    reduced = remove_class_scores(report)
    assert report["classes"] and reduced["classes"] is None
    assert {k: v for k, v in reduced.items() if k != "classes"} == {
        k: v for k, v in report.items() if k != "classes"}
    assert report == original


def test_deletion_deletes_the_cached_values(archive, dataset_contexts, store, worker):
    submissions = [add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", seed))
                   for seed in (1, 2)]
    models = score_and_get_models(worker, [s.id for s in submissions])
    attributes = models[0].scores.attributes
    deleted = models[0].submission
    row = compute_method_scores("m", models, attributes)
    assert worker.get_method_resamples(row, ["amF1"]) is not None

    archive.delete_model(deleted.model.id, actor="Bo")
    worker.update_model_submissions(deleted.model)  # As the pages do after a change.
    assert store.get_scoring(deleted.id) is None
    assert store.get_score_report(deleted.id, "reference") is None
    with pytest.raises(LookupError):
        store.get_resamples(deleted.id, "reference", models[0].scores.evaluation_set_fingerprint)
    # A page that read the scores before the deletion, with a cold cache.
    new_worker = ScoringWorker(archive, store, dataset_contexts, worker.settings)
    assert new_worker.get_method_resamples(row, ["amF1"]) is None


def test_values_of_deleted_submissions_are_not_stored(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    scoring = store.get_scoring(submission.id)
    resamples = {s.evaluation_set: store.get_resamples(submission.id, s.evaluation_set,
                                                       s.evaluation_set_fingerprint)
                 for s in scoring.evaluation_set_scores}
    archive.delete_model(submission.model.id, actor="Bo")

    # E.g. values computed while the submission was deleted.
    store.set_scoring(scoring, {s.evaluation_set: {} for s in scoring.evaluation_set_scores},
                      resamples)
    assert store.get_scoring(submission.id) is None
    worker._score(submission.id)  # Queued before the deletion.
    assert store.get_scoring(submission.id) is None


def test_a_current_scoring_is_not_scored_again(archive, dataset_contexts, store, worker,
                                               monkeypatch):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts))
    score_and_get_models(worker, [submission.id])

    def fail(*args):
        raise RuntimeError("Scored again.")

    monkeypatch.setattr(scoring_worker_module, "score_submission", fail)
    worker._score(submission.id)  # E.g. queued again while it was being scored.
    assert store.get_scoring(submission.id).status == "scored"  # Not an error of `fail`.


def test_out_of_date_scorings_are_scored_again(archive, dataset_contexts, metadata_dir, store,
                                                worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    first = store.get_scoring(submission.id)

    # Other settings.
    other_worker = ScoringWorker(archive, store, dataset_contexts,
                                 ScoringSettings(num_resamples=20))
    assert other_worker.get_current_scorings([submission]) == {}
    other_worker.update_all_submissions()
    assert other_worker.is_scoring_pending(submission.id)
    other_worker.run_pending()
    assert set(other_worker.get_current_scorings([submission])) == {submission.id}
    assert worker.get_current_scorings([submission]) == {}

    # A new metadata build with another label of a reference segment.
    path = metadata_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA
    road_data = json.loads(path.read_text(encoding="utf-8"))
    segment_id = dataset_contexts["vietnam"].get_reference_set(SPLIT).segment_ids[0]
    codes = road_data[segment_id]["required_attributes"]
    codes["Number of lanes"] = 1 if codes["Number of lanes"] != 1 else 2
    path.write_text(json.dumps(road_data), encoding="utf-8")
    new_contexts = load_dataset_contexts({"vietnam": DatasetConfig(
        dataset_dir=metadata_dir, metadata_dir=metadata_dir, analysis_splits=(SPLIT,))})
    new_worker = ScoringWorker(archive, store, new_contexts, ScoringSettings(num_resamples=20))
    assert new_worker.get_current_scorings([submission]) == {}
    new_worker.update_all_submissions()
    new_worker.run_pending()
    rescored = store.get_scoring(submission.id)
    assert rescored.status == "scored"
    assert (rescored.get_scores("reference").evaluation_set_fingerprint
            != first.get_scores("reference").evaluation_set_fingerprint)


def test_scoring_errors(archive, dataset_contexts, store, worker):
    submissions = [add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", seed))
                   for seed in (1, 2)]
    # An invalid file fails, and is not scored again while it is current.
    archive.get_predictions_path(submissions[0].id).write_bytes(b"not parquet")
    # A missing file is an unexpected error, which is scored again.
    archive.get_predictions_path(submissions[1].id).unlink()
    worker.update_submissions(submissions)
    worker.run_pending()
    failed, error = (store.get_scoring(s.id) for s in submissions)
    assert failed.status == "failed" and failed.message
    assert error.status == "error" and "FileNotFoundError" in error.message
    assert set(worker.get_current_scorings(submissions)) == {submissions[0].id}
    worker.update_all_submissions()
    assert worker.is_scoring_pending(submissions[1].id)
    assert not worker.is_scoring_pending(submissions[0].id)
    worker.run_pending()


def test_get_unscored_reason():
    scoring = SubmissionScoring(submission_id=1, status="scored", message="", notes=(),
                                settings_fingerprint="s", evaluation_sets_fingerprint="e",
                                scored_at=datetime(2026, 10, 3, tzinfo=timezone.utc))
    assert get_unscored_reason(scoring, is_current=True, is_pending=False) is None
    assert get_unscored_reason(scoring, True, is_pending=True) == "Scoring…"
    assert "current settings" in get_unscored_reason(None, False, False)
    failed = dc.replace(scoring, status="failed", message="No predictions for 3 segments.")
    assert get_unscored_reason(failed, True, False) == failed.message
    error = dc.replace(scoring, status="error", message="Unexpected error.")
    assert "scored again" in get_unscored_reason(error, False, False)


def test_scorings_of_other_scoring_code_are_out_of_date(archive, dataset_contexts, worker,
                                                        monkeypatch):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    compute_uncached = compute_scoring_code_fingerprint.__wrapped__
    assert compute_uncached(worker.settings) == compute_uncached(worker.settings)

    resample_result = scoring_module.resample_result
    monkeypatch.setattr(scoring_module, "resample_result", lambda result, settings: (
        ie.map_metric_values(lambda v: v + 1, resample_result(result, settings))))
    compute_scoring_code_fingerprint.cache_clear()
    try:
        # A new instance, since the fingerprint is cached.
        new_worker = ScoringWorker(archive, worker.store, dataset_contexts,
                                   dc.replace(worker.settings))
        assert new_worker.get_current_scorings([submission]) == {}
    finally:
        compute_scoring_code_fingerprint.cache_clear()


def test_a_score_cache_of_another_version_is_created_again(archive, dataset_contexts, store,
                                                           worker, caplog):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    with connect(store.database_path) as connection:
        connection.execute("PRAGMA user_version = 99")

    new_store = ScoreStore(archive.data_dir)
    assert "version 99, so it is created again with version 1" in caplog.text
    assert new_store.get_scoring(submission.id) is None
    new_worker = ScoringWorker(archive, new_store, dataset_contexts, worker.settings)
    new_worker.update_all_submissions()
    new_worker.run_pending()
    assert new_store.get_scoring(submission.id).status == "scored"
    # The cache of the current version is kept.
    caplog.clear()
    assert ScoreStore(archive.data_dir).get_scoring(submission.id).status == "scored"
    assert not caplog.text

    # A cache whose tables changed without a new version.
    with connect(store.database_path) as connection:
        connection.execute("ALTER TABLE submission_scores ADD COLUMN extra TEXT")
    with pytest.raises(ValueError, match="submission_scores'] of the score cache"):
        ScoreStore(archive.data_dir)



def test_worker_thread(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    worker.start()
    try:
        worker.update_submissions([submission])
        assert worker.wait_until_idle(timeout_s=60)
        assert store.get_scoring(submission.id).status == "scored"
    finally:
        worker.stop(timeout_s=60)


def test_store_keeps_undefined_and_infinite_values(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    scoring = store.get_scoring(submission.id)
    scores = scoring.get_scores("reference")
    metrics = dc.replace(scores.metrics, averages={**scores.metrics.averages,
                                                   "amF1": math.nan, "aNLL": math.inf})
    resamples = store.get_resamples(submission.id, "reference", scores.evaluation_set_fingerprint)
    special = np.resize([math.nan, math.inf, -math.inf, 0.25], len(resamples.averages["aNLL"]))
    resamples = dc.replace(resamples, averages={**resamples.averages, "aNLL": special})
    store.set_scoring(
        dc.replace(scoring, evaluation_set_scores=(dc.replace(scores, metrics=metrics),)),
        {"reference": {}}, {"reference": resamples})
    new_store = ScoreStore(archive.data_dir)
    averages = new_store.get_scoring(submission.id).get_scores("reference").metrics.averages
    assert math.isnan(averages["amF1"]) and averages["aNLL"] == math.inf
    assert np.isfinite(averages["aA"])
    stored = new_store.get_resamples(submission.id, "reference",
                                     scores.evaluation_set_fingerprint)
    np.testing.assert_equal(dc.asdict(stored), dc.asdict(resamples))
