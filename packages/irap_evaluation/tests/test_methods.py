import dataclasses as dc

import numpy as np
import pytest

from irap_evaluation.evaluation import evaluate_predictions, select_result_attributes
from irap_evaluation.methods import (
    check_prediction_file_models,
    compute_mean_metrics,
    compute_method_metrics,
    group_models_by_method,
)
from irap_evaluation.metrics import select_metric_attributes
from irap_evaluation.predictions import ModelInfo, PredictionHeader

from synthetic_vietnam import SPLIT, evaluate_models, make_predictions, with_model


def test_compute_mean_metrics_equals_the_method_metrics(metadata, reference_set):
    models = evaluate_models(metadata, reference_set, "m", [0, 1, 2])
    np.testing.assert_equal(dc.asdict(compute_mean_metrics([r.metrics for r in models])),
                            dc.asdict(compute_method_metrics(models)))
    subset = ["Number of lanes", "Lane width"]
    np.testing.assert_equal(
        dc.asdict(compute_mean_metrics([select_metric_attributes(r.metrics, subset)
                                        for r in models])),
        dc.asdict(compute_method_metrics([select_result_attributes(r, subset) for r in models])))
    with pytest.raises(ValueError, match="differ"):
        compute_mean_metrics([models[0].metrics,
                              select_metric_attributes(models[1].metrics, subset)])
    with pytest.raises(ValueError, match="At least one"):
        compute_mean_metrics([])


def test_models_must_be_distinct_comparable_models_of_one_method(metadata, reference_set):
    model_0, model_1 = evaluate_models(metadata, reference_set, "m", [0, 1])
    for models, message in [
        ([model_0, with_model(model_1, "m", seed=0)], "same seed"),
        ([model_0, with_model(model_1, "m")], "each needs a seed"),
        ([model_0, with_model(model_1, "other", seed=1)], "different methods"),
        ([model_0, with_model(model_1, "m", seed=1, early_stopping_splits=("val",))],
         "different training or early stopping splits"),
        ([model_0, dc.replace(model_1, null_policy="exclude")], "null policies"),
        ([model_0, evaluate_predictions(
            make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]), name="m",
                             seed=1, model_seed=1),
            reference_set, metric_names=["amF1"])], "different metrics"),
    ]:
        with pytest.raises(ValueError, match=message):
            compute_method_metrics(models)
    with pytest.raises(ValueError, match="same seed"):
        group_models_by_method([model_0, model_0])
    assert group_models_by_method([model_0, model_1]) == {"m": (model_0, model_1)}


def _header(split, seed=None, training_splits=("train",), method_name="m"):
    return PredictionHeader("vietnam", split, ModelInfo(method_name, training_splits, (), seed))


def test_files_of_one_model_on_several_splits():
    check_prediction_file_models([_header("val"), _header("test"), _header("val", method_name="n")])
    check_prediction_file_models([_header("val", 0), _header("val", 1), _header("test", 0)])
    for headers, message in [
        ([_header("val"), _header("test", training_splits=("train", "val"))],
         r"models \['m'\] on vietnam differ"),
        ([_header("val", 0), _header("test", 1, training_splits=None)],
         "different training or early stopping splits"),
        ([_header("val"), _header("val")], "vietnam/val: Method 'm' has 2 models"),
    ]:
        with pytest.raises(ValueError, match=message):
            check_prediction_file_models(headers)
