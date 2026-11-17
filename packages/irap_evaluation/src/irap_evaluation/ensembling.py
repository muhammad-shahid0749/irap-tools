"""Combining the predictions of several methods or models."""

import itertools
import typing as T

import numpy as np

from .predictions import (
    ModelInfo,
    PredictionFormatError,
    PredictionHeader,
    Predictions,
    add_invalid_attributes,
    add_invalid_segments,
    align_classes,
    get_common_dataset_and_split,
    select_segments,
)

MemberCombination = T.Literal["same", "intersection", "union"]
"""Which segments or attributes an ensemble combines when its members predict different ones:
'same' refuses such members, 'intersection' combines those that every member predicts, and 'union'
those that any member predicts, where a member that does not predict one counts as invalid."""
MEMBER_COMBINATIONS: tuple[str, ...] = T.get_args(MemberCombination)


def _format_examples(keys: T.Sequence[str], max_num_shown: int = 5) -> str:
    shown = ", ".join(keys[:max_num_shown])
    return f"{shown}, …" if len(keys) > max_num_shown else shown


def _combine_keys(labels: T.Sequence[str], key_seqs: T.Sequence[T.Sequence[str]],
                  combination: MemberCombination, noun: str) -> tuple[str, ...]:
    """Combines the segment IDs or the attributes of the members.

    Args:
        labels: The labels of the members, for error messages.
        key_seqs: The keys of each member.
        noun: The plural noun of the keys, for error messages, e.g. 'segments'.

    Returns:
        The keys in the order of the first member, followed by those of later members that are
        new, in their order.

    Raises:
        PredictionFormatError: If the members predict different keys and `combination` is 'same',
            or they have no common key and `combination` is 'intersection'.
        ValueError: For an unknown `combination`.
    """
    if combination not in MEMBER_COMBINATIONS:
        raise ValueError(f"The combination of {noun} must be one of {MEMBER_COMBINATIONS}, got"
                         f" {combination!r}.")
    key_sets = [set(keys) for keys in key_seqs]
    all_keys = tuple(dict.fromkeys(itertools.chain.from_iterable(key_seqs)))
    if combination == "union":
        return all_keys
    if combination == "same":
        lacking = []
        for label, keys in zip(labels, key_sets):
            if missing := [k for k in all_keys if k not in keys]:
                lacking.append(f"{label} lacks {len(missing)} of the {len(all_keys)} {noun}"
                               f" ({_format_examples(missing)})")
        if lacking:
            raise PredictionFormatError(
                f"Every member must predict the same {noun}, but {'; '.join(lacking)}. Combine"
                f" their intersection or union instead.")
        return all_keys
    common = set.intersection(*key_sets)
    if not common:
        raise PredictionFormatError(
            f"The members have no common {noun}: "
            + "; ".join(f"{label} predicts {len(keys)} ({_format_examples(keys)})"
                        for label, keys in zip(labels, key_seqs)) + ".")
    return tuple(k for k in key_seqs[0] if k in common)


def _get_attribute_to_irap_codes(labels: T.Sequence[str],
                                 predictions_seq: T.Sequence[Predictions],
                                 attributes: T.Iterable[str]) -> dict[str, tuple[int, ...]]:
    """Attribute -> the iRAP codes of the first member that predicts it, in its class order.

    Raises:
        PredictionFormatError: If members predict different classes of an attribute.
    """
    attribute_to_irap_codes = {}
    for attr in attributes:
        label_and_codes = [(label, p.attribute_to_irap_codes[attr])
                           for label, p in zip(labels, predictions_seq)
                           if attr in p.attribute_to_irap_codes]
        # The codes of an attribute are unique within a member (`Predictions` validation).
        if len({frozenset(codes) for _, codes in label_and_codes}) > 1:
            raise PredictionFormatError(
                f"The members predict different classes of {attr!r}: "
                + "; ".join(f"{label} has iRAP codes {sorted(codes)}"
                            for label, codes in label_and_codes) + ".")
        attribute_to_irap_codes[attr] = tuple(label_and_codes[0][1])
    return attribute_to_irap_codes


def _get_union_of_context_offsets(
        headers: T.Iterable[PredictionHeader]) -> tuple[int, ...] | None:
    offsets = [h.context_offsets for h in headers]
    if any(o is None for o in offsets):
        return None
    return tuple(sorted(set().union(*offsets), reverse=True))


def _format_seeds(seeds: T.Sequence[int]) -> str:
    """Formats sorted seeds, writing runs of at least 3 consecutive seeds as `<first>-<last>`,
    e.g. '0-4,7,9'."""
    # The seeds of a run have the same difference to their index.
    runs = [[s for _, s in run]
            for _, run in itertools.groupby(enumerate(seeds), key=lambda p: p[1] - p[0])]
    return ",".join(f"{run[0]}-{run[-1]}" if len(run) >= 3 else ",".join(map(str, run))
                    for run in runs)


def make_ensemble_method_name(members: T.Iterable[ModelInfo]) -> str:
    """Makes the default method name of an ensemble from the method names and seeds of its
    members, e.g. 'resnet-seq_seeds0-4+vit-l' or 'resnet-seq_seeds1,3+vit-l_seed2'.

    The methods are sorted and joined with '+'. A method is written as `<method>` for a model
    without a seed, `<method>_seed<seed>` for one seed, and `<method>_seeds<seeds>` for several
    (see `_format_seeds`). The seeds are kept so that an ensemble of the models of one method is
    not named like the method.

    Raises:
        ValueError: For no members.
    """
    method_to_seeds: dict[str, list[int | None]] = {}
    for member in members:
        method_to_seeds.setdefault(member.method_name, []).append(member.seed)
    if not method_to_seeds:
        raise ValueError("At least one member is required.")
    parts = []
    for method_name, seeds in sorted(method_to_seeds.items()):
        parts.extend([method_name] * seeds.count(None))
        seeds = sorted(s for s in seeds if s is not None)
        if len(seeds) == 1:
            parts.append(f"{method_name}_seed{seeds[0]}")
        elif seeds:
            parts.append(f"{method_name}_seeds{_format_seeds(seeds)}")
    return "+".join(parts)


def make_ensemble_model_info(members: T.Iterable[ModelInfo], method_name: str | None = None, *,
                             method_display_name: str | None = None, seed: int | None = None,
                             details: T.Mapping[str, T.Any] | None = None) -> ModelInfo:
    """Makes the model of an ensemble, which uses every split that a member uses:
    - Its training splits are the union of the members' training splits, or None if those of a
      member are unknown.
    - Its early stopping splits are the union of the members' early stopping splits that are not
      training splits of the ensemble: a member that was fitted on a split makes the ensemble
      fitted on it.

    Args:
        method_name: The method name of the ensemble, None for `make_ensemble_method_name`.
    """
    members = list(members)
    training = (None if any(m.training_splits is None for m in members)
                else set().union(*(m.training_splits for m in members)))
    early_stopping = set().union(*(m.early_stopping_splits for m in members)) - (training or set())
    if method_name is None:
        method_name = make_ensemble_method_name(members)
    return ModelInfo(method_name=method_name, training_splits=training,
                     early_stopping_splits=early_stopping, seed=seed,
                     method_display_name=method_display_name, details=details)


def _compute_weighted_mean(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The weighted mean of (M, N, K) distributions with (M, N) weights, 0 for invalid attribute
    predictions.

    Returns:
        (N, K) distributions, NaN in rows with no weight.
    """
    probs = np.nan_to_num(probs.astype(np.float64), nan=0.0, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.einsum("mn,mnk->nk", weights, probs) / weights.sum(0)[:, None]


def ensemble_predictions(
    predictions_seq: T.Sequence[Predictions],
    method_name: str | None = None,
    *,
    method_display_name: str | None = None,
    seed: int | None = None,
    details: T.Mapping[str, T.Any] | None = None,
    weights: T.Sequence[float] | None = None,
    segment_combination: MemberCombination = "same",
    attribute_combination: MemberCombination = "same",
) -> Predictions:
    """Computes the weighted mean of the distributions of several members (methods or models) of
    the same release and split. For hard predictions, it gives weighted vote shares.

    Classes are matched by iRAP code, and the output uses the class order of the first member
    that predicts the attribute. An attribute prediction that is invalid in a member is left out
    of the mean for that attribute and segment, and it is invalid in the ensemble only if it is
    invalid in every member. With the 'union' combination of segments or attributes, a member's
    missing segment or attribute counts as invalid. The ensemble uses whatever its members use:
    - Its context offsets are the union of the members' offsets, since it reads every frame a
      member reads, or None if the offsets of a member are unknown.
    - Its training and early stopping splits are those of `make_ensemble_model_info`.

    Args:
        predictions_seq: The members.
        method_name: The method name of the ensemble's model, None for
            `make_ensemble_method_name`.
        method_display_name: The method display name of the ensemble's model.
        seed: The seed of the ensemble's model.
        details: The details of the ensemble's model. The combinations and the members with their
            context offsets, weights, attributes and numbers of segments are added under
            'ensemble'.
        weights: A finite positive weight per member. None weights them equally.
        segment_combination: Which segments to combine if the members predict different ones
            (`MemberCombination`).
        attribute_combination: Which attributes to combine if the members predict different
            ones (`MemberCombination`).

    Raises:
        PredictionFormatError: If the members differ in release, split or the iRAP codes of an
            attribute, or in segments or attributes as their combination does not allow
            (`MemberCombination`).
        ValueError: For no members, invalid weights or an unknown combination.
    """
    if not predictions_seq:
        raise ValueError("At least one member is required.")
    weights = np.ones(len(predictions_seq)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(predictions_seq),) or not (np.isfinite(weights) & (weights > 0)).all():
        raise ValueError(f"Expected {len(predictions_seq)} finite positive weights, got {weights}.")
    headers = [p.header for p in predictions_seq]
    dataset, split = get_common_dataset_and_split(headers)
    labels = [h.model.label for h in headers]
    attributes = _combine_keys(labels, [p.attributes for p in predictions_seq],
                               attribute_combination, "attributes")
    segment_ids = _combine_keys(labels, [p.segment_ids for p in predictions_seq],
                                segment_combination, "segments")
    attribute_to_irap_codes = _get_attribute_to_irap_codes(labels, predictions_seq, attributes)

    def align_member(predictions: Predictions) -> Predictions:
        selected = select_segments(add_invalid_segments(predictions, segment_ids), segment_ids)
        return align_classes(add_invalid_attributes(selected, attribute_to_irap_codes),
                             attribute_to_irap_codes)

    members = [align_member(p) for p in predictions_seq]

    def combine_attribute(attr: str) -> np.ndarray:
        cell_weights = weights[:, None] * np.stack([m.is_valid[attr] for m in members])
        mean = _compute_weighted_mean(np.stack([m.probs[attr] for m in members]), cell_weights)
        return mean.astype(np.float32)

    ensemble = {"segment_combination": segment_combination,
                "attribute_combination": attribute_combination,
                "members": [{"model": p.header.model.to_json_dict(),
                             "context_offsets": (None if p.header.context_offsets is None
                                                 else list(p.header.context_offsets)),
                             "weight": float(w),
                             "attributes": list(p.attributes),
                             "num_segments": p.num_segments}
                            for p, w in zip(predictions_seq, weights)]}
    model = make_ensemble_model_info([h.model for h in headers], method_name,
                                     method_display_name=method_display_name, seed=seed,
                                     details={**(details or {}), "ensemble": ensemble})
    header = PredictionHeader(dataset=dataset, split=split, model=model,
                              context_offsets=_get_union_of_context_offsets(headers))
    return Predictions(header, "probs", attribute_to_irap_codes, segment_ids,
                       {attr: combine_attribute(attr) for attr in attribute_to_irap_codes})
