import numpy as np
import pytest

from irap_data.metadata import IGNORE_LABEL_INDEX
from irap_evaluation.ensembling import ensemble_predictions, make_ensemble_method_name
from irap_evaluation.predictions import (
    ModelInfo,
    PredictionFormatError,
    PredictionHeader,
    Predictions,
)

CLASS_CODES = {"a": (1, 2)}
REVERSED_CLASS_CODES = {"a": (2, 1)}


def _header(method_name, offsets=None, training_splits=("train",), early_stopping_splits=(),
            seed=None):
    model = ModelInfo(method_name=method_name, training_splits=training_splits,
                      early_stopping_splits=early_stopping_splits, seed=seed)
    return PredictionHeader("vietnam", "val", model, context_offsets=offsets)


def _probs(attribute_to_irap_codes, values, header, is_valid=None):
    return Predictions.from_probabilities(header, attribute_to_irap_codes, ["s1", "s2"],
                                          {"a": np.array(values)},
                                          None if is_valid is None else {"a": np.array(is_valid)})


def test_mean_aligns_classes_by_code_and_skips_invalid_cells():
    first = _probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], _header("m1", offsets=(0, -1)))
    # The same distributions over (x, y) written in the order (y, x), and s2 invalid.
    second = _probs(REVERSED_CLASS_CODES, [[0.6, 0.4], [0.5, 0.5]],
                    _header("m2", offsets=(0, -4)), is_valid=[True, False])
    ensemble = ensemble_predictions([first, second], "e", weights=[1, 3])
    np.testing.assert_allclose(ensemble.probs["a"][0], [0.25 * 0.8 + 0.75 * 0.4,
                                                        0.25 * 0.2 + 0.75 * 0.6], rtol=1e-6)
    np.testing.assert_allclose(ensemble.probs["a"][1], [0.6, 0.4], rtol=1e-6)
    assert ensemble.header.context_offsets == (0, -1, -4)
    members = ensemble.header.model.details["ensemble"]["members"]
    assert [(m["model"]["method_name"], m["context_offsets"], m["weight"]) for m in members] == \
           [("m1", [0, -1], 1), ("m2", [0, -4], 3)]


@pytest.mark.parametrize("member_splits, training, early_stopping", [
    ([(("unlabeled_val", "train"), ()), (("train",), ("val",))],
     ("train", "unlabeled_val"), ("val",)),
    # A member's training split is not an early stopping split of the ensemble.
    ([(("train",), ("val",)), (("train", "val"), ())], ("train", "val"), ()),
    # With a member of unknown training splits, every split use of the ensemble is unknown.
    ([(("train",), ("val",)), (None, ())], None, ("val",)),
])
def test_ensemble_uses_the_splits_of_its_members(member_splits, training, early_stopping):
    members = [_probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]],
                      _header(f"m{i}", training_splits=t, early_stopping_splits=e))
               for i, (t, e) in enumerate(member_splits)]
    model = ensemble_predictions(members, "e", seed=1, details={"commit": "abc"}).header.model
    assert (model.method_name, model.seed, model.details["commit"]) == ("e", 1, "abc")
    assert (model.training_splits, model.early_stopping_splits) == (training, early_stopping)
    if training is None:
        assert model.get_split_use("val") == "unknown"


@pytest.mark.parametrize("members, name", [
    ([("b", None), ("a", None)], "a+b"),
    ([("m", 3)], "m_seed3"),
    ([("m", 2), ("m", 1)], "m_seeds1,2"),
    ([("m", 4), ("m", 0), ("m", 2), ("m", 3), ("m", 1), ("m", 7), ("m", 9), ("m", 10)],
     "m_seeds0-4,7,9,10"),
    ([("vit", 2), ("resnet", 3), ("resnet", 1), ("dino", None)],
     "dino+resnet_seeds1,3+vit_seed2"),
])
def test_default_method_name_lists_the_methods_with_their_seeds(members, name):
    models = [ModelInfo(method_name=m, training_splits=(), early_stopping_splits=(), seed=s)
              for m, s in members]
    assert make_ensemble_method_name(models) == name


def test_ensemble_gets_the_default_method_name_and_a_display_name():
    members = [_probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], _header("m", seed=i))
               for i in range(3)]
    model = ensemble_predictions(members).header.model
    assert (model.method_name, model.method_display_name) == ("m_seeds0-2", None)
    model = ensemble_predictions(members, "e", method_display_name="Ensemble").header.model
    assert (model.method_name, model.method_display_name) == ("e", "Ensemble")


def test_mean_of_hard_predictions_gives_vote_shares():
    votes = [Predictions.from_class_indices(_header(f"h{i}"), CLASS_CODES, ["s1", "s2"],
                                            {"a": np.array(v)})
             for i, v in enumerate([[0, 1], [1, IGNORE_LABEL_INDEX], [1, IGNORE_LABEL_INDEX]])]
    ensemble = ensemble_predictions(votes, "e")
    np.testing.assert_allclose(ensemble.probs["a"], [[1 / 3, 2 / 3], [0, 1]], rtol=1e-6)
    assert ensemble.output_kind == "probs"


@pytest.mark.parametrize("weights", [[1, 0], [1, -1], [1, np.inf], [1, np.nan], [1]])
def test_weights_must_be_finite_and_positive_per_member(weights):
    members = [_probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], _header(f"m{i}")) for i in range(2)]
    with pytest.raises(ValueError, match="finite positive weights"):
        ensemble_predictions(members, "e", weights=weights)


def test_segment_combinations():
    first = _probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], _header("m1"))
    other = Predictions.from_probabilities(_header("m2"), CLASS_CODES, ["s2", "s3"],
                                           {"a": np.array([[0.5, 0.5], [0.1, 0.9]])})
    with pytest.raises(PredictionFormatError,
                       match=r"same segments, but m1 lacks 1 of the 3 segments \(s3\); m2 lacks 1"
                             r" of the 3 segments \(s1\)\. Combine their intersection or union"):
        ensemble_predictions([first, other], "e")
    ensemble = ensemble_predictions([first, other], "e", segment_combination="intersection")
    assert ensemble.segment_ids == ("s2",)
    ensemble = ensemble_predictions([first, other], "e", segment_combination="union")
    assert ensemble.segment_ids == ("s1", "s2", "s3")
    np.testing.assert_allclose(ensemble.probs["a"], [[0.8, 0.2], [0.55, 0.45], [0.1, 0.9]],
                               rtol=1e-6)
    details = ensemble.header.model.details["ensemble"]
    assert (details["segment_combination"], details["attribute_combination"]) == ("union", "same")
    assert [m["num_segments"] for m in details["members"]] == [2, 2]
    disjoint = Predictions.from_probabilities(_header("m3"), CLASS_CODES, ["s3"],
                                              {"a": np.array([[0.5, 0.5]])})
    with pytest.raises(PredictionFormatError,
                       match=r"no common segments: m1 predicts 2 \(s1, s2\); m3 predicts 1 \(s3\)"):
        ensemble_predictions([first, disjoint], "e", segment_combination="intersection")


def test_attribute_combinations():
    first = _probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], _header("m1"))
    other = Predictions.from_probabilities(
        _header("m2"), {**REVERSED_CLASS_CODES, "b": (3, 4)}, ["s1", "s2"],
        {"a": np.array([[0.6, 0.4], [0.2, 0.8]]), "b": np.array([[0.3, 0.7], [0.9, 0.1]])})
    with pytest.raises(PredictionFormatError,
                       match=r"same attributes, but m1 lacks 1 of the 2 attributes \(b\)\."):
        ensemble_predictions([first, other], "e")
    ensemble = ensemble_predictions([first, other], "e", attribute_combination="intersection")
    assert ensemble.attributes == ("a",)
    ensemble = ensemble_predictions([first, other], "e", attribute_combination="union")
    assert ensemble.attribute_to_irap_codes == {"a": (1, 2), "b": (3, 4)}
    # `other` predicts the classes of 'a' in the order (2, 1).
    np.testing.assert_allclose(ensemble.probs["a"], [[0.6, 0.4], [0.7, 0.3]], rtol=1e-6)
    np.testing.assert_allclose(ensemble.probs["b"], other.probs["b"], rtol=1e-6)
    members = ensemble.header.model.details["ensemble"]["members"]
    assert [m["attributes"] for m in members] == [["a"], ["a", "b"]]


def test_members_must_predict_the_same_classes_of_an_attribute():
    first = _probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], _header("m1"))
    other = _probs({"a": (1, 3)}, [[0.5, 0.5], [0.1, 0.9]], _header("m2"))
    with pytest.raises(PredictionFormatError,
                       match=r"different classes of 'a': m1 has iRAP codes \[1, 2\]; m2 has iRAP"
                             r" codes \[1, 3\]\."):
        ensemble_predictions([first, other], "e")


def test_unknown_combination_is_an_error():
    members = [_probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], _header(f"m{i}")) for i in range(2)]
    with pytest.raises(ValueError, match="combination of segments must be one of"):
        ensemble_predictions(members, "e", segment_combination="all")
