"""End-to-end runs of the `irap-eval` commands on the synthetic dataset."""

import dataclasses as dc
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from irap_data.metadata import MetaFiles, get_dataset_preset
from irap_evaluation.prediction_io import read_predictions, write_predictions
from irap_evaluation.tools.cli import main

from synthetic_vietnam import SPLIT, make_predictions


def _write_model_files(metadata, tmp_path):
    segments = list(metadata.splits[SPLIT])
    paths = []
    for name, seed, offsets in [("a", 0, (0, -1)), ("b", 1, None)]:
        path = tmp_path / f"{name}.predictions.parquet"
        write_predictions(path, make_predictions(metadata.vocabulary, segments, name=name,
                                                 seed=seed, context_offsets=offsets))
        paths.append(str(path))
    return paths


def test_evaluate_refuses_models_without_labels(metadata_dir, metadata, tmp_path, capsys):
    paths = _write_model_files(metadata, tmp_path)
    # Without road data, no segment has labels, as in an unlabeled split.
    (metadata_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA).write_text("{}", encoding="utf-8")
    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), *paths, "--out", str(out)]) == 1
    assert "nothing is scored" in capsys.readouterr().err
    assert not out.exists()


def test_evaluate_ensemble_compare_export(metadata_dir, metadata, tmp_path, capsys):
    paths = _write_model_files(metadata, tmp_path)
    assert main(["validate", *paths]) == 0

    ensemble_path = str(tmp_path / "ens.predictions.parquet")
    assert main(["ensemble", *paths, "--name", "ens", "-o", ensemble_path]) == 0
    assert read_predictions(ensemble_path).header.context_offsets is None
    default_path = str(tmp_path / "default.predictions.parquet")
    assert main(["ensemble", *paths, "--display-name", "Ensemble", "-o", default_path]) == 0
    model = read_predictions(default_path).header.model
    assert (model.method_name, model.method_display_name) == ("a+b", "Ensemble")

    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), *paths, ensemble_path, "--out", str(out),
                 "--context-offsets", "b=0", "--bootstrap", "50",
                 "--table-metrics", "mF1", "MRP90"]) == 0
    assert (out / "per_attribute_MRP90_vietnam_val_reference.csv").exists()
    long_table = pd.read_csv(out / "summary_long.csv")
    assert set(long_table["model"]) == {"a", "b", "ens"}
    assert (long_table["ci_num_undefined"] >= 0).all()
    # All predict every segment. a is also scored on its own set, the offsets 0 of b select the
    # full set, and ens has no offsets.
    assert long_table.groupby("model")["evaluation_set"].unique().map(tuple).to_dict() == {
        "a": ("reference", "full", "compatible"), "b": ("reference", "full"),
        "ens": ("reference", "full")}
    per_attribute = pd.read_csv(out / "per_attribute_mF1_vietnam_val_reference.csv",
                                index_col=0)
    assert list(per_attribute.columns) == ["a", "b", "ens"]
    assert per_attribute.index[-1] == "average"
    report = json.loads((out / "a" / "metrics_vietnam_val_compatible.json").read_text())
    assert report["context_offsets"] == [0, -1]
    assert report["intervals"]["averages"]["amF1"]["low"] <= report["metrics"]["averages"]["amF1"]
    assert "Attribute averages per model, vietnam/val, reference set" in capsys.readouterr().out

    assert main(["compare", str(metadata_dir), *paths, "--a", "a", "--b", "b", "--bootstrap",
                 "50"]) == 0
    assert "a - b" in capsys.readouterr().out

    table_path = tmp_path / "coding.csv"
    assert main(["export-coding-table", str(metadata_dir), paths[0], "-o", str(table_path),
                 "--confidence-table"]) == 0
    assert (tmp_path / "coding.confidence.csv").exists()


def test_export_of_a_coding_table_of_several_files(metadata_dir, metadata, tmp_path):
    segments = list(metadata.splits[SPLIT])
    paths = []
    for i, part in enumerate([segments[::2], segments[1::2]]):
        paths.append(str(tmp_path / f"m_{i}.predictions.parquet"))
        predictions = make_predictions(metadata.vocabulary, part, name="m", seed=i)
        # Files of one model written at different commits.
        model = dc.replace(predictions.header.model, details={"commit": str(i)})
        write_predictions(paths[-1], dc.replace(
            predictions, header=dc.replace(predictions.header, model=model)))
    table_path = tmp_path / "coding.csv"
    assert main(["export-coding-table", str(metadata_dir), *paths, "-o", str(table_path)]) == 0
    assert len(pd.read_csv(table_path)) == len(segments)


def test_scores_on_a_training_split_are_noted(metadata_dir, metadata, tmp_path, capsys):
    path = tmp_path / "t.predictions.parquet"
    write_predictions(path, make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]),
                                             name="t", seed=0, training_splits=(SPLIT,)))
    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), str(path), "--out", str(out)]) == 0
    assert f"Model 't': The model was trained on {SPLIT}" in capsys.readouterr().out
    assert set(pd.read_csv(out / "summary_long.csv")["split_use"]) == {"training"}
    report = json.loads((out / "t" / "metrics_vietnam_val_reference.json").read_text())
    assert report["split_use"] == "training"
    averages = pd.read_csv(out / "averages_vietnam_val_reference.csv", index_col=0)
    assert averages.loc["t", "split_use"] == "training"


def test_unknown_training_split_names_are_an_error(metadata_dir, metadata, tmp_path, capsys):
    path = tmp_path / "t.predictions.parquet"
    write_predictions(path, make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]),
                                             name="t", seed=0, training_splits=("trian",)))
    assert main(["evaluate", str(metadata_dir), str(path), "--out", str(tmp_path / "r")]) == 1
    assert "unknown splits ['trian']" in capsys.readouterr().err


def _write_method_model_files(metadata, tmp_path, method_to_num_models):
    segments = list(metadata.splits[SPLIT])
    paths = []
    for name, num_models in method_to_num_models.items():
        for model_seed in range(num_models):
            path = tmp_path / f"{name}_{model_seed}.predictions.parquet"
            write_predictions(path, make_predictions(
                metadata.vocabulary, segments, name=name, seed=len(paths),
                model_seed=model_seed, context_offsets=None))
            paths.append(str(path))
    return paths


def test_methods_with_several_models(metadata_dir, metadata, tmp_path, capsys):
    paths = _write_method_model_files(metadata, tmp_path, {"a": 3, "b": 2, "c": 1})
    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), *paths, "--out", str(out),
                 "--bootstrap", "50"]) == 0
    assert "means over models" in capsys.readouterr().out
    assert (out / "a_seed0" / "metrics_vietnam_val_reference.json").exists()
    long_table = pd.read_csv(out / "summary_long.csv")
    assert set(long_table["method"]) == {"a", "b", "c"}
    methods = pd.read_csv(out / "method_averages_vietnam_val_reference.csv", index_col=0)
    assert methods["models"].to_dict() == {"a": 3, "b": 2}  # c has a single model
    a_models = long_table[(long_table["method"] == "a") & (long_table["attribute"] == "average")
                          & (long_table["metric"] == "mF1")
                          & (long_table["evaluation_set"] == "reference")]
    assert methods.loc["a", "mF1"] == pytest.approx(a_models["value"].mean())
    assert methods.loc["a", "mF1 low"] <= methods.loc["a", "mF1"] <= methods.loc["a", "mF1 high"]

    assert main(["compare", str(metadata_dir), *paths[:5], "--a", "a", "--b", "b",
                 "--metrics", "amF1", "--bootstrap", "50"]) == 0
    assert "a = a (3 models), b = b (2 models)" in capsys.readouterr().out
    assert main(["compare", str(metadata_dir), *paths, "--a", "a", "--b", "b"]) == 1
    assert "other methods than a and b: ['c']" in capsys.readouterr().err
    assert main(["compare", str(metadata_dir), *paths[:3], "--a", "a", "--b", "b"]) == 1
    assert "No files of the methods ['b']" in capsys.readouterr().err


def test_means_over_models_need_the_first_class_null_policy(metadata_dir, metadata, tmp_path,
                                                            capsys):
    segments = list(metadata.splits[SPLIT])
    paths = []
    for model_seed in range(2):
        predictions = make_predictions(metadata.vocabulary, segments, name="vlm", seed=model_seed,
                                       model_seed=model_seed, context_offsets=None)
        # The models have invalid attribute predictions in different segments.
        is_valid = np.arange(len(segments)) % 2 == model_seed
        probs = np.where(is_valid[:, None], predictions.probs["Lane width"], np.nan)
        paths.append(str(tmp_path / f"vlm_{model_seed}.predictions.parquet"))
        write_predictions(paths[-1], dc.replace(predictions, probs={
            **predictions.probs, "Lane width": probs.astype(np.float32)}))

    def evaluate(null_policy):
        out = tmp_path / null_policy
        assert main(["evaluate", str(metadata_dir), *paths, "--out", str(out),
                     "--null-policy", null_policy]) == 0
        return out, pd.read_csv(out / "summary_long.csv")

    excluded_out, excluded_models = evaluate("exclude")
    assert "No means over the models of a method" in capsys.readouterr().out
    assert not list(excluded_out.glob("method_averages_*.csv"))
    first_class_out, first_class_models = evaluate("first_class")
    assert "means over models" in capsys.readouterr().out
    methods = pd.read_csv(first_class_out / "method_averages_vietnam_val_reference.csv",
                          index_col=0)
    assert methods.loc["vlm", "null_policy"] == "first_class"
    assert set(excluded_models["null_policy"]) == {"exclude"}

    def get_lane_width_n(models):
        return models[(models["attribute"] == "Lane width") & (models["metric"] == "n")]["value"]

    assert (get_lane_width_n(excluded_models).to_numpy()
            < get_lane_width_n(first_class_models).to_numpy()).all()
    assert main(["compare", str(metadata_dir), *paths[:1], *_write_method_model_files(
        metadata, tmp_path, {"other": 1}), "--a", "vlm", "--b", "other",
        "--bootstrap", "20"]) == 0


def test_missing_attributes(metadata_dir, metadata, tmp_path, capsys):
    attributes = [a for a in metadata.vocabulary.attribute_names if a != "Lane width"]
    path = tmp_path / "partial.predictions.parquet"
    write_predictions(path, make_predictions(
        metadata.vocabulary.restrict_to_attributes(attributes), list(metadata.splits[SPLIT]),
        name="partial", seed=0))

    def evaluate(policy):
        out = tmp_path / policy
        exit_code = main(["evaluate", str(metadata_dir), str(path), "--out", str(out),
                          "--missing-attribute-policy", policy])
        return exit_code, out

    assert evaluate("error")[0] == 1
    assert "not predicted: ['Lane width']" in capsys.readouterr().err
    exit_code, out = evaluate("invalid")
    assert exit_code == 0
    assert "scored as invalid, since not predicted: Lane width" in capsys.readouterr().out
    invalid = pd.read_csv(out / "summary_long.csv")
    assert "Lane width" in set(invalid["attribute"])
    report = json.loads((out / "partial" / "metrics_vietnam_val_reference.json").read_text())
    assert report["missing_attributes"] == ["Lane width"]


@pytest.mark.parametrize("names, message", [
    (["m", "m"], "each needs a seed"),
    (["conv seq", "conv_seq"], "output directories"),
])
def test_clashing_method_names_are_an_error(metadata_dir, metadata, tmp_path, capsys, names,
                                            message):
    segments = list(metadata.splits[SPLIT])
    paths = []
    for name in names:
        paths.append(str(tmp_path / f"{len(paths)}.predictions.parquet"))
        write_predictions(paths[-1], make_predictions(metadata.vocabulary, segments, name=name,
                                                      seed=0))
    assert main(["evaluate", str(metadata_dir), *paths, "--out", str(tmp_path / "r")]) == 1
    assert message in capsys.readouterr().err  # without a traceback


def test_models_that_miss_reference_segments_are_scored_on_their_own_set(
        metadata_dir, metadata, tmp_path, capsys):
    # Offset -6 reaches beyond the -4..0 reference window, so the first segments of each road that
    # are in the reference set have no prediction.
    segments = [sid for sid in metadata.splits[SPLIT] if int(sid[1:]) >= 6]
    path = tmp_path / "far.predictions.parquet"
    write_predictions(path, make_predictions(metadata.vocabulary, segments, name="far", seed=0,
                                             context_offsets=(0, -6)))
    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), str(path), "--out", str(out)]) == 0
    assert "segments of the reference set" in capsys.readouterr().out
    long_table = pd.read_csv(out / "summary_long.csv")
    assert set(long_table["evaluation_set"]) == {"compatible"}


def test_a_model_that_cannot_be_scored_on_any_set_is_an_error(metadata_dir, metadata,
                                                              reference_set, tmp_path, capsys):
    complete, incomplete = tmp_path / "a.predictions.parquet", tmp_path / "b.predictions.parquet"
    segments = list(metadata.splits[SPLIT])
    reference_segment = reference_set.segment_ids[0]
    write_predictions(complete, make_predictions(metadata.vocabulary, segments, name="a", seed=0))
    write_predictions(incomplete, make_predictions(
        metadata.vocabulary, [sid for sid in segments if sid != reference_segment], name="b",
        seed=0, context_offsets=None))
    assert main(["evaluate", str(metadata_dir), str(complete), str(incomplete), "--out",
                 str(tmp_path / "r")]) == 1
    assert "Model 'b': No predictions for 1 of" in capsys.readouterr().err


def test_a_model_with_the_reference_offsets_is_not_scored_on_its_model_compatible_set(
        metadata_dir, metadata, tmp_path, capsys):
    offsets = get_dataset_preset("vietnam").reference_context_offsets
    path = tmp_path / "ref.predictions.parquet"
    write_predictions(path, make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]),
                                             name="ref", seed=0, context_offsets=offsets))
    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), str(path), "--out", str(out)]) == 0
    assert "model-compatible set is left out" in capsys.readouterr().out
    assert (sorted(p.name for p in (out / "ref").iterdir())
            == ["metrics_vietnam_val_full.json", "metrics_vietnam_val_reference.json"])


def test_the_library_does_not_import_the_reports_or_the_tools():
    code = ("import sys, irap_evaluation;"
            " print(sorted(m for m in sys.modules if m.startswith('irap_evaluation.')))")
    modules = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True).stdout
    assert "reports" not in modules and "tools" not in modules
