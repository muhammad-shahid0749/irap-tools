"""Ensembles of stored models (`irap_evaluation.ensemble_predictions`): the weighted mean of their
predicted distributions, stored as a model with a submission for each split where every member
has one, which replace all files of the model.

The action log records each new submission with its member submissions and weights (action
'ensemble'), and the prediction header records the member models (see
`irap_evaluation/docs/prediction_format.md`).
"""

import dataclasses as dc
import math
import typing as T

import irap_evaluation as ie

from .archive import Model, ModelArchive, ModelUpdate, NewSubmission, Submission
from .datasets import DatasetContext


@dc.dataclass(frozen=True)
class EnsembleMember:
    """A member of an ensemble: a model and its weight."""

    model: Model
    weight: float = 1.0


@dc.dataclass(frozen=True)
class EnsemblePlan:
    """The submissions that an ensemble is made of, by split.

    Attributes:
        split_to_submissions: Split -> the submission of each member, in the order of `members`,
            for the splits where every member has a submission.
        split_to_missing: Split -> the shown labels of the members without a submission of it,
            for the other splits that some member has one of.
    """

    dataset: str
    members: tuple[EnsembleMember, ...]
    split_to_submissions: T.Mapping[str, tuple[Submission, ...]]
    split_to_missing: T.Mapping[str, tuple[str, ...]]

    def make_model_info(self) -> ie.ModelInfo:
        """Makes the model of the ensemble, with the default method name and its training and
        early stopping splits (`irap_evaluation.make_ensemble_model_info`). The files of a member
        have the same training and early stopping splits, so those of any split are used."""
        submissions = next(iter(self.split_to_submissions.values()))
        return ie.make_ensemble_model_info(s.model_info for s in submissions)


def plan_ensemble(submissions: T.Iterable[Submission], dataset: str,
                  members: T.Sequence[EnsembleMember],
                  split_names: T.Sequence[str]) -> EnsemblePlan:
    """
    Args:
        submissions: E.g. all submissions (`ModelArchive.list_submissions`).
        split_names: The splits of the dataset, in the order of the plan.

    Raises:
        ValueError: If there are fewer than 2 members, a member is not of `dataset` or is given
            twice, a weight is not positive and finite, or no split has a submission of every
            member.
    """
    if len(members) < 2:
        raise ValueError("An ensemble needs at least 2 members.")
    if other := [m.model.shown_label for m in members if m.model.dataset != dataset]:
        raise ValueError(f"The members must be of one dataset, but {', '.join(other)} are not"
                         f" of {dataset}.")
    if len({m.model.id for m in members}) < len(members):
        raise ValueError("Each model can be a member only once.")
    if bad := [m.model.shown_label for m in members
               if not (math.isfinite(m.weight) and m.weight > 0)]:
        raise ValueError(f"The weights of {', '.join(bad)} must be positive and finite.")
    split_and_model_to_submission = {(s.split, s.model.id): s for s in submissions
                                     if s.dataset == dataset}
    split_to_submissions, split_to_missing = {}, {}
    for split in split_names:
        found = [split_and_model_to_submission.get((split, m.model.id)) for m in members]
        if all(s is not None for s in found):
            split_to_submissions[split] = tuple(found)
        elif any(s is not None for s in found):
            split_to_missing[split] = tuple(m.model.shown_label
                                            for m, s in zip(members, found) if s is None)
    if not split_to_submissions:
        raise ValueError("No split has a file of every member.")
    return EnsemblePlan(dataset=dataset, members=tuple(members),
                        split_to_submissions=split_to_submissions,
                        split_to_missing=split_to_missing)


def prepare_ensemble(
    archive: ModelArchive,
    context: DatasetContext,
    plan: EnsemblePlan,
    *,
    method_name: str | None,
    method_display_name: str | None,
    seed: int | None,
    segment_combination: ie.MemberCombination,
    attribute_combination: ie.MemberCombination,
    description: str | None,
) -> ModelUpdate:
    """Makes the ensemble of each split of a plan, checks each as an upload
    (`DatasetContext.check_new_predictions`), and plans storing them as the submissions of their
    model (`ModelArchive.plan_model_update`). They replace all files of the model, so that its
    files are of the same members, and the update is refused if a member's file changes before
    it is applied.

    The files are in `archive.uploads_dir` until the update is applied. The caller removes those
    that remain (`archive.discard_model_update`).

    Args:
        context: The dataset of the plan.
        method_name: The method name of the ensemble's model, None for the default
            (`irap_evaluation.make_ensemble_method_name`).
        method_display_name: The method display name of the ensemble's model, or None to keep
            the method's.
        seed: The seed of the ensemble's model.
        segment_combination: Which segments to combine if the members predict different ones,
            e.g. members with different context offsets (`irap_evaluation.MemberCombination`).
        attribute_combination: Which attributes to combine if the members predict different
            ones (`irap_evaluation.MemberCombination`).
        description: The new description of the model, or None to keep it.

    Raises:
        irap_evaluation.PredictionFormatError: If the members cannot be combined, e.g. they
            predict other segments and `segment_combination` is 'same'.
        ValueError: If `irap_evaluation.select_evaluation_sets` refuses an ensemble, or
            `ModelArchive.plan_model_update` refuses the update.
    """
    weights = [m.weight for m in plan.members]
    new_submissions = []
    paths = []
    try:
        for submissions in plan.split_to_submissions.values():
            ensemble = ie.ensemble_predictions(
                [ie.read_predictions(archive.get_predictions_path(s.id)) for s in submissions],
                method_name, method_display_name=method_display_name, seed=seed, weights=weights,
                segment_combination=segment_combination,
                attribute_combination=attribute_combination)
            notes = context.check_new_predictions(ensemble)
            paths.append(path := archive.make_upload_path("ensemble.predictions.parquet"))
            ie.write_predictions(path, ensemble)
            new_submissions.append(NewSubmission.from_predictions(
                ensemble, path, details={
                    "members": [{"submission_id": s.id, "model_id": s.model.id,
                                 "label": s.label, "weight": w}
                                for s, w in zip(submissions, weights, strict=True)],
                    "segment_combination": segment_combination,
                    "attribute_combination": attribute_combination}, notes=notes))
        return archive.plan_model_update(
            new_submissions, action="ensemble", description=description, replaces_all_files=True,
            source_submissions=[s for ss in plan.split_to_submissions.values() for s in ss])
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
