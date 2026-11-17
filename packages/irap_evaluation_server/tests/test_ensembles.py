import dataclasses as dc

import irap_evaluation as ie
import numpy as np
import pytest
from run_helpers import OTHER_SPLIT, add_model, make_model, with_split
from synthetic_vietnam import SPLIT

from irap_evaluation_server.archive import discard_model_update
from irap_evaluation_server.ensembles import (
    EnsembleMember,
    plan_ensemble,
    prepare_ensemble,
)

SPLITS = (SPLIT, OTHER_SPLIT)


@pytest.fixture
def submissions(archive, two_split_contexts):
    """Models a/seed1 and b/seed0 on both splits, c/seed0 only on `SPLIT`, and a deleted model
    d/seed0, whose `Model` the tests use as a member."""
    added = {}
    for name, seed, splits in [("a", 1, SPLITS), ("b", 0, SPLITS), ("c", 0, (SPLIT,)),
                               ("d", 0, SPLITS)]:
        predictions = make_model(two_split_contexts, name, seed, random_seed=len(added))
        for split in splits:
            added[(name, split)] = add_model(archive, two_split_contexts,
                                             with_split(predictions, split))
    archive.delete_model(added[("d", SPLIT)].model.id, actor="Bo")
    return added


def member(submissions, name, weight=1.0):
    return EnsembleMember(submissions[(name, SPLIT)].model, weight)


def test_plan(archive, submissions):
    members = [member(submissions, "a", 2.0), member(submissions, "b"),
               member(submissions, "c")]
    plan = plan_ensemble(archive.list_submissions(), "vietnam", members, SPLITS)
    assert plan.split_to_submissions == {SPLIT: tuple(submissions[(n, SPLIT)] for n in "abc")}
    assert plan.split_to_missing == {OTHER_SPLIT: ("c/seed0",)}
    for wrong_members, message in [
            (members[:1], "at least 2"), ([members[0], members[0]], "only once"),
            ([members[0], EnsembleMember(dc.replace(members[1].model, dataset="other"))],
             "one dataset"),
            ([dc.replace(members[0], weight=0.0), members[1]], "positive"),
            ([members[0], member(submissions, "d")], "No split")]:
        with pytest.raises(ValueError, match=message):
            plan_ensemble(archive.list_submissions(), "vietnam", wrong_members, SPLITS)


def test_create_ensemble(archive, two_split_contexts, submissions):
    members = [member(submissions, "a", 2.0), member(submissions, "b")]
    plan = plan_ensemble(archive.list_submissions(), "vietnam", members, SPLITS)
    update = prepare_ensemble(archive, two_split_contexts["vietnam"], plan, method_name="ens",
                              method_display_name="Ensemble", seed=7,
                              segment_combination="same", attribute_combination="same",
                              description="test")
    assert update.confirmations == []
    created = archive.apply_model_update(update, submitter="Bo")
    assert [(s.split, s.label, s.shown_label, s.model.description) for s in created] == [
        (SPLIT, "ens/seed7", "Ensemble/seed7", "test"),
        (OTHER_SPLIT, "ens/seed7", "Ensemble/seed7", "test")]
    for submission in created:
        stored = ie.read_predictions(archive.get_predictions_path(submission.id))
        expected = ie.ensemble_predictions(
            [ie.read_predictions(archive.get_predictions_path(submissions[(n, submission.split)]
                                                              .id)) for n in "ab"],
            "ens", method_display_name="Ensemble", seed=7, weights=[2.0, 1.0])
        assert stored.header == expected.header
        assert stored.header.model.method_display_name == "Ensemble"
        assert stored.header.model.training_splits == ()
        for attribute in expected.attributes:
            np.testing.assert_array_equal(stored.probs[attribute], expected.probs[attribute])
        [action] = [a for a in archive.list_actions() if a.submission_id == submission.id]
        assert action.action == "ensemble"
        assert action.details["members"] == [
            {"submission_id": submissions[("a", submission.split)].id,
             "model_id": submissions[("a", SPLIT)].model.id, "label": "a/seed1", "weight": 2.0},
            {"submission_id": submissions[("b", submission.split)].id,
             "model_id": submissions[("b", SPLIT)].model.id, "label": "b/seed0", "weight": 1.0}]
    assert not list(archive.uploads_dir.iterdir())

    # The model exists now, so another ensemble with its name and seed replaces its files.
    update = prepare_ensemble(archive, two_split_contexts["vietnam"], plan, method_name="ens",
                              method_display_name=None, seed=7,
                              segment_combination="same", attribute_combination="same",
                              description=None)
    assert len(update.confirmations) == 2
    discard_model_update(update)
    assert not list(archive.uploads_dir.iterdir())


def test_an_ensemble_gets_the_default_method_name(archive, two_split_contexts, submissions):
    plan = plan_ensemble(archive.list_submissions(), "vietnam",
                         [member(submissions, "b"), member(submissions, "a")], SPLITS)
    assert plan.make_model_info().method_name == "a_seed1+b_seed0"
    update = prepare_ensemble(archive, two_split_contexts["vietnam"], plan, method_name=None,
                              method_display_name=None, seed=0,
                              segment_combination="same", attribute_combination="same",
                              description=None)
    created = archive.apply_model_update(update, submitter="Bo")
    assert {(s.label, s.model.method_display_name) for s in created} == {
        ("a_seed1+b_seed0/seed0", None)}


def _prepare(archive, two_split_contexts, submissions, names):
    plan = plan_ensemble(archive.list_submissions(), "vietnam",
                         [member(submissions, n) for n in names], SPLITS)
    return prepare_ensemble(archive, two_split_contexts["vietnam"], plan, method_name="ens",
                            method_display_name=None, seed=0,
                            segment_combination="same", attribute_combination="same",
                            description=None)


def test_an_ensemble_replaces_all_files_of_its_model(archive, two_split_contexts, submissions):
    [first, second] = archive.apply_model_update(
        _prepare(archive, two_split_contexts, submissions, "ab"), submitter="Bo")
    # c has no file of OTHER_SPLIT, so the ensemble of a and c has only SPLIT.
    update = _prepare(archive, two_split_contexts, submissions, "ac")
    assert (list(update.replaced), list(update.replaced_other_splits)) == ([SPLIT], [OTHER_SPLIT])
    assert any(f"Permanently deletes the {OTHER_SPLIT} file #{second.id}" in c
               for c in update.confirmations)
    [third] = archive.apply_model_update(update, submitter="Bo")
    assert archive.list_submissions() == [third, *archive.list_submissions()[1:]]
    assert archive.list_model_submissions(third.model.id) == [third]
    assert not archive.get_predictions_path(second.id).parent.exists()
    ensemble, *deletions = archive.list_actions(model_ids=[third.model.id])[:3]
    assert (ensemble.action, ensemble.details["replaced_submission_id"]) == ("ensemble", first.id)
    assert [(a.action, a.submission_id, a.details) for a in deletions] == [
        ("delete", first.id, {"split": SPLIT, "by": "ensemble"}),
        ("delete", second.id, {"split": OTHER_SPLIT, "by": "ensemble"})]


def test_an_ensemble_is_refused_if_a_member_changes(archive, two_split_contexts, submissions):
    update = _prepare(archive, two_split_contexts, submissions, "ab")
    archive.delete_submission(submissions[("b", OTHER_SPLIT)].id, actor="Al")
    with pytest.raises(ValueError, match=f"{OTHER_SPLIT} file #.* of b/seed0, which the new files"
                                         f" are made of, has been replaced or deleted"):
        archive.apply_model_update(update, submitter="Bo")
    discard_model_update(update)


def test_an_ensemble_is_refused_if_its_model_gets_another_split(archive, two_split_contexts,
                                                                 submissions):
    archive.apply_model_update(_prepare(archive, two_split_contexts, submissions, "ac"),
                               submitter="Bo")
    update = _prepare(archive, two_split_contexts, submissions, "ac")
    # Another user adds a file of the other split meanwhile.
    other = _prepare(archive, two_split_contexts, submissions, "ab")
    archive.apply_model_update(archive.plan_model_update(
        [n for n in other.new_submissions if n.split == OTHER_SPLIT], action="upload",
        description=None), submitter="Al")
    discard_model_update(other)
    with pytest.raises(ValueError, match="files of ens/seed0 have changed"):
        archive.apply_model_update(update, submitter="Bo")
    discard_model_update(update)


def test_ensemble_files_are_removed_if_it_fails(archive, two_split_contexts, submissions,
                                                monkeypatch):
    plan = plan_ensemble(archive.list_submissions(), "vietnam",
                         [member(submissions, "a"), member(submissions, "b")], SPLITS)
    ensemble_predictions = ie.ensemble_predictions

    def fail_on_second_split(predictions_seq, method_name, **kwargs):
        if predictions_seq[0].header.split == OTHER_SPLIT:
            raise ie.PredictionFormatError("Refused.")
        return ensemble_predictions(predictions_seq, method_name, **kwargs)

    monkeypatch.setattr(ie, "ensemble_predictions", fail_on_second_split)
    with pytest.raises(ie.PredictionFormatError, match="Refused"):
        prepare_ensemble(archive, two_split_contexts["vietnam"], plan, method_name="ens",
                         method_display_name=None, seed=0,
                         segment_combination="same", attribute_combination="same",
                         description=None)
    assert not list(archive.uploads_dir.iterdir())
