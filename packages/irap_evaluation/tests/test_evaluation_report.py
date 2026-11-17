import dataclasses as dc
import json

import numpy as np
import pytest

from irap_evaluation.bootstrap import BootstrapInterval
from irap_evaluation.evaluation import evaluate_predictions
from irap_evaluation.metrics import MetricValues
from irap_evaluation.reports.evaluation_report import (
    get_partially_undefined_intervals,
    to_json_dict,
    to_long_table,
    to_per_attribute_table,
    write_evaluation_reports,
)
from synthetic_vietnam import to_hard_predictions


def test_only_partially_undefined_intervals_are_reported():
    intervals = MetricValues(
        averages={"amF1": BootstrapInterval(0.1, 0.2, 0), "aNLL": BootstrapInterval(-1, 1, 3)},
        per_attribute={"MCC": {"Lane width": BootstrapInterval(np.nan, np.nan, 10)}})
    assert get_partially_undefined_intervals(intervals) == {"aNLL": 3}


def test_json_report_has_no_nan(reference_set, predictions):
    # Predicting one class everywhere makes MCC undefined (NaN).
    probs = np.zeros_like(predictions.probs["Lane width"])
    probs[:, 0] = 1
    constant = dc.replace(predictions, probs={**predictions.probs, "Lane width": probs})
    report = to_json_dict(evaluate_predictions(constant, reference_set))
    assert report["metrics"]["per_attribute"]["MCC"]["Lane width"] is None
    json.dumps(report, allow_nan=False)


def test_json_report_has_the_review_curve_of_distributions(reference_set, predictions):
    report = to_json_dict(evaluate_predictions(predictions, reference_set))
    curve = report["classes"]["Lane width"]["review_curve"]
    assert curve["review_proportions"][0] == 0 and curve["review_proportions"][-1] == 1
    assert curve["macro_f1"][0] == pytest.approx(
        report["metrics"]["per_attribute"]["mF1"]["Lane width"])
    assert curve["macro_f1"][-1] == 1 and curve["accuracy"][-1] == 1
    hard_report = to_json_dict(evaluate_predictions(to_hard_predictions(predictions),
                                                    reference_set))
    assert hard_report["classes"]["Lane width"]["review_curve"] is None


def test_per_attribute_tables_need_per_attribute_metrics(reference_set, predictions):
    long_table = to_long_table([evaluate_predictions(predictions, reference_set)])
    with pytest.raises(ValueError, match="no per-attribute values of 'amF1'"):
        to_per_attribute_table(long_table, "amF1")


def test_reports_refuse_evaluation_sets_that_map_to_one_file(reference_set, predictions,
                                                             tmp_path):
    results = [evaluate_predictions(predictions, dc.replace(reference_set, name=name))
               for name in ["a b", "a_b"]]
    with pytest.raises(ValueError, match="map to the files"):
        write_evaluation_reports(results, tmp_path / "out", table_metrics=["mF1"])
    assert not (tmp_path / "out").exists()
