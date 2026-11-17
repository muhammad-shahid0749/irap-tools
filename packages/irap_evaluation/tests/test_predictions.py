import dataclasses as dc
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from irap_data.metadata import IGNORE_LABEL_INDEX
from irap_evaluation.prediction_io import (
    HEADER_METADATA_KEY,
    from_arrow_table,
    read_prediction_header,
    read_predictions,
    to_arrow_table,
    write_predictions,
)
from irap_evaluation.predictions import (
    INVALID_IRAP_CODE,
    ModelInfo,
    PredictionFormatError,
    PredictionHeader,
    Predictions,
    add_invalid_attributes,
    add_invalid_segments,
    align_classes,
    check_class_codes,
    check_split_names,
    select_segments,
    to_class_indices,
    to_irap_codes,
)

CLASS_CODES = {"a": (10, 20, 30), "b": (1, 2)}
HEADER = PredictionHeader(
    dataset="vietnam", split="val",
    model=ModelInfo(method_name="m", training_splits=("train",), early_stopping_splits=(),
                    details={"lr": 0.1}),
    context_offsets=(0, -1))


def _predictions():
    probs = {"a": np.array([[0.2, 0.5, 0.3], [0.9, 0.05, 0.05], [0, 0, 1]]),
             "b": np.array([[0.5, 0.5], [0.1, 0.9], [1, 0]])}
    is_valid = {"a": np.array([True, True, False]), "b": np.ones(3, dtype=bool)}
    return Predictions.from_probabilities(HEADER, CLASS_CODES, ["s1", "s2", "s3"], probs, is_valid)


def _hard_predictions():
    return Predictions.from_irap_codes(
        HEADER, CLASS_CODES, ["s1", "s2", "s3"],
        {"a": np.array([30, INVALID_IRAP_CODE, 10]), "b": np.array([2, 1, 1])})


def _assert_equal(loaded, predictions):
    assert loaded.header == predictions.header
    assert loaded.header.model.details == predictions.header.model.details  # Not compared above.
    assert loaded.output_kind == predictions.output_kind
    assert loaded.attribute_to_irap_codes == predictions.attribute_to_irap_codes
    assert loaded.segment_ids == predictions.segment_ids
    for attr in predictions.attributes:
        np.testing.assert_array_equal(loaded.probs[attr], predictions.probs[attr])


def _change_header(table, change):
    header = json.loads(table.schema.metadata[HEADER_METADATA_KEY])
    change(header)
    return table.replace_schema_metadata({HEADER_METADATA_KEY: json.dumps(header).encode()})


@pytest.mark.parametrize("make_predictions", [_predictions, _hard_predictions])
def test_round_trip(tmp_path, make_predictions):
    predictions = make_predictions()
    write_predictions(tmp_path / "m.predictions.parquet", predictions)
    _assert_equal(read_predictions(tmp_path / "m.predictions.parquet"), predictions)


def test_file_marks_invalid_attribute_predictions_as_null():
    table = to_arrow_table(_predictions())
    assert table.column("a").null_count == 1
    assert "output_kind" not in json.loads(table.schema.metadata[HEADER_METADATA_KEY])


def test_hard_file_stores_class_indices_in_the_header_order():
    predictions = Predictions.from_irap_codes(HEADER, {"a": (30, 20, 10)}, ["s1", "s2", "s3"],
                                              {"a": np.array([30, INVALID_IRAP_CODE, 10])})
    table = to_arrow_table(predictions)
    assert table.column("a").type == pa.int64()
    assert table.column("a").to_pylist() == [0, None, 2]


def test_invalid_rows_are_nan_and_argmax_ignores_them():
    predictions = _predictions()
    assert np.isnan(predictions.probs["a"][2]).all()
    np.testing.assert_array_equal(predictions.is_valid["a"], [True, True, False])
    np.testing.assert_array_equal(to_class_indices(predictions)["a"], [1, 0, IGNORE_LABEL_INDEX])
    np.testing.assert_array_equal(to_irap_codes(predictions)["a"], [20, 10, INVALID_IRAP_CODE])


@pytest.mark.parametrize("change, message", [
    (lambda p: {"probs": {**p.probs, "a": p.probs["a"] * 2}}, "sum to 1"),
    (lambda p: {"probs": {**p.probs, "a": p.probs["a"].astype(np.float64)}}, "float32"),
    (lambda p: {"probs": {**p.probs, "b": np.array([[0.5, np.nan]] * 3, np.float32)}}, "NaN"),
    (lambda p: {"segment_ids": ("s1", "s1", "s3")}, "unique"),
    (lambda p: {"probs": {"a": p.probs["a"]}}, "attributes"),
    (lambda p: {"attribute_to_irap_codes": {"a": (10, 10, 30), "b": (1, 2)}}, "duplicate"),
    (lambda p: {"output_kind": "hard"}, "one-hot"),
    (lambda p: {"attribute_to_irap_codes": {}, "probs": {}}, "at least one attribute"),
])
def test_validation_rejects_broken_predictions(change, message):
    predictions = _predictions()
    with pytest.raises(PredictionFormatError, match=message):
        dc.replace(predictions, **change(predictions))


def test_header_rejects_duplicate_context_offsets():
    with pytest.raises(PredictionFormatError, match="duplicates"):
        dc.replace(HEADER, context_offsets=(0, 0, -1))


def test_arrays_are_read_only_views():
    predictions = _predictions()
    for derived in [predictions, select_segments(predictions, ["s2"]),
                    align_classes(predictions, {"a": (30, 10, 20)})]:
        for arrays in [derived.probs, derived.is_valid]:
            assert not any(array.flags.writeable for array in arrays.values())
    probs = np.array([[0.5, 0.5]], dtype=np.float32)
    Predictions(HEADER, "probs", {"b": (1, 2)}, ["s1"], {"b": probs})
    assert probs.flags.writeable


def test_nan_in_a_valid_attribute_prediction_is_an_error():
    with pytest.raises(PredictionFormatError, match="valid attribute prediction contains NaN"):
        Predictions.from_probabilities(HEADER, {"b": (1, 2)}, ["s1"],
                                       {"b": np.array([[np.nan, np.nan]])})


def test_from_class_indices():
    predictions = Predictions.from_class_indices(HEADER, CLASS_CODES, ["s1", "s2"],
                                                 {"a": np.array([2, IGNORE_LABEL_INDEX]),
                                                  "b": np.array([0, 1])})
    assert predictions.output_kind == "hard"
    np.testing.assert_array_equal(predictions.probs["a"][0], [0, 0, 1])
    assert not predictions.is_valid["a"][1]
    with pytest.raises(PredictionFormatError, match="class indices"):
        Predictions.from_class_indices(HEADER, CLASS_CODES, ["s1"],
                                       {"a": np.array([3]), "b": np.array([0])})


def test_from_irap_codes():
    predictions = _hard_predictions()
    np.testing.assert_array_equal(to_irap_codes(predictions)["a"], [30, INVALID_IRAP_CODE, 10])
    np.testing.assert_array_equal(predictions.is_valid["a"], [True, False, True])
    with pytest.raises(PredictionFormatError, match=r"codes \[40\]"):
        Predictions.from_irap_codes(HEADER, CLASS_CODES, ["s1"],
                                    {"a": np.array([40]), "b": np.array([1])})
    with pytest.raises(PredictionFormatError, match="cannot mark"):
        Predictions.from_irap_codes(HEADER, {"a": (INVALID_IRAP_CODE, 1)}, ["s1"],
                                    {"a": np.array([1])})


@pytest.mark.parametrize("from_integers", [Predictions.from_class_indices,
                                           Predictions.from_irap_codes])
def test_integer_inputs_must_have_an_integer_type(from_integers):
    codes = {"b": (1, 2)}
    valid = from_integers(HEADER, codes, ["s1", "s2"], {"b": np.array([1, 1], dtype=np.int32)})
    assert valid.is_valid["b"].all()
    assert from_integers(HEADER, codes, [], {"b": []}).num_segments == 0
    # Casting would turn 1.5 into the valid 1.
    for values in [np.array([1.5, 1.0]), np.array([1.0, 1.0]), np.array([True, True])]:
        with pytest.raises(PredictionFormatError, match="must be integers"):
            from_integers(HEADER, codes, ["s1", "s2"], {"b": values})


def test_align_classes_reorders_classes_by_code():
    aligned = align_classes(_predictions(), {"a": (30, 10, 20)})
    assert aligned.attributes == ("a",)
    np.testing.assert_allclose(aligned.probs["a"][0], [0.3, 0.2, 0.5])
    np.testing.assert_array_equal(to_irap_codes(aligned)["a"], to_irap_codes(_predictions())["a"])
    with pytest.raises(PredictionFormatError, match="differ"):
        align_classes(_predictions(), {"a": (10, 20)})


def test_class_codes_must_match():
    check_class_codes(_predictions(), {"a": (30, 10, 20), "b": (1, 2)})
    for codes in [(10, 20), (10, 20, 20, 30), (10, 20, 40)]:
        with pytest.raises(PredictionFormatError, match="differ"):
            check_class_codes(_predictions(), {"a": codes})
    with pytest.raises(PredictionFormatError, match="not predicted"):
        check_class_codes(_predictions(), {"c": (1,)})
    with pytest.raises(PredictionFormatError, match="at least one attribute"):
        check_class_codes(_predictions(), {})


@pytest.mark.parametrize("make_predictions", [_predictions, _hard_predictions])
def test_add_invalid_attributes(make_predictions):
    predictions = make_predictions()
    added = add_invalid_attributes(predictions, {"b": (2, 1), "c": (5, 6, 7)})
    assert added.attributes == ("a", "b", "c")
    assert added.attribute_to_irap_codes["c"] == (5, 6, 7)
    assert added.output_kind == predictions.output_kind
    assert not added.is_valid["c"].any()
    for attr in predictions.attributes:  # Predicted attributes, also 'b', are kept.
        np.testing.assert_array_equal(added.probs[attr], predictions.probs[attr])
    assert added.attribute_to_irap_codes["b"] == CLASS_CODES["b"]
    # The result is valid.
    Predictions(added.header, added.output_kind, added.attribute_to_irap_codes,
                added.segment_ids, added.probs)
    assert add_invalid_attributes(predictions, {"a": CLASS_CODES["a"]}) is predictions
    with pytest.raises(PredictionFormatError, match="duplicate"):
        add_invalid_attributes(predictions, {"c": (5, 5)})


def test_select_segments():
    selected = select_segments(_predictions(), ["s3", "s1"])
    assert selected.segment_ids == ("s3", "s1")
    np.testing.assert_array_equal(selected.is_valid["a"], [False, True])
    with pytest.raises(PredictionFormatError, match="no prediction"):
        select_segments(_predictions(), ["s4"])
    with pytest.raises(PredictionFormatError, match="unique"):
        select_segments(_predictions(), ["s1", "s1"])


def test_add_invalid_segments():
    predictions = _predictions()
    added = add_invalid_segments(predictions, ["s4", "s1", "s4"])
    assert added.segment_ids == predictions.segment_ids + ("s4",)
    for attr in predictions.attributes:
        assert not added.is_valid[attr][-1]
        np.testing.assert_array_equal(added.probs[attr][:-1], predictions.probs[attr])
    assert add_invalid_segments(predictions, ["s1"]) is predictions


def test_read_prediction_header(tmp_path):
    path = tmp_path / "m.predictions.parquet"
    write_predictions(path, _predictions())
    with path.open("rb") as file:
        for header in (read_prediction_header(path), read_prediction_header(file)):
            assert header == HEADER
            assert header.model.details == HEADER.model.details


@pytest.mark.parametrize("read", [read_predictions, read_prediction_header])
def test_reading_rejects_other_parquet_files(tmp_path, read):
    table = to_arrow_table(_predictions()).replace_schema_metadata({})
    pq.write_table(table, tmp_path / "x.parquet")
    with pytest.raises(PredictionFormatError, match="header"):
        read(tmp_path / "x.parquet")


def test_probability_columns_are_not_dictionary_encoded(tmp_path):
    write_predictions(tmp_path / "m.predictions.parquet", _predictions())
    column = pq.read_metadata(tmp_path / "m.predictions.parquet").row_group(0).column(1)
    assert column.path_in_schema.startswith("a.")
    assert "RLE_DICTIONARY" not in column.encodings


@pytest.mark.parametrize("details", [{"lr": float("nan")}, {"lr": np.float32(0.1)}])
def test_writing_rejects_a_header_that_is_not_standard_json(details):
    header = dc.replace(HEADER, model=dc.replace(HEADER.model, details=details))
    with pytest.raises(PredictionFormatError, match="standard JSON"):
        to_arrow_table(dc.replace(_predictions(), header=header))


def _write_with_polars(table, path):
    polars = pytest.importorskip("polars")
    header = table.schema.metadata[HEADER_METADATA_KEY].decode()
    polars.from_arrow(table).write_parquet(path, metadata={HEADER_METADATA_KEY.decode(): header})


def _write_with_duckdb(table, path):
    duckdb = pytest.importorskip("duckdb")
    header = table.schema.metadata[HEADER_METADATA_KEY].decode().replace("'", "''")
    with duckdb.connect() as connection:
        connection.register("predictions", table)
        connection.execute(f"COPY predictions TO '{path.as_posix()}'"
                           f" (KV_METADATA {{{HEADER_METADATA_KEY.decode()}: '{header}'}})")


@pytest.mark.parametrize("write", [_write_with_polars, _write_with_duckdb])
@pytest.mark.parametrize("make_predictions", [_predictions, _hard_predictions])
def test_reading_files_of_other_writers(tmp_path, write, make_predictions):
    # Polars stores the header only in the Parquet key-value metadata, not in the Arrow schema.
    predictions = make_predictions()
    write(to_arrow_table(predictions), tmp_path / "m.predictions.parquet")
    _assert_equal(read_predictions(tmp_path / "m.predictions.parquet"), predictions)


def _set_column(table, name, column):
    return table.set_column(table.column_names.index(name), name, column)


@pytest.mark.parametrize("make_predictions", [_predictions, _hard_predictions])
def test_reading_accepts_a_null_typed_column_of_invalid_attribute_predictions(make_predictions):
    predictions = make_predictions()
    loaded = from_arrow_table(_set_column(to_arrow_table(predictions), "a", pa.nulls(3)))
    assert loaded.output_kind == predictions.output_kind
    assert not loaded.is_valid["a"].any()
    np.testing.assert_array_equal(loaded.probs["b"], predictions.probs["b"])


@pytest.mark.parametrize("make_predictions, attr, column_type", [
    (_predictions, "a", pa.large_list(pa.float64())),
    (_hard_predictions, "b", pa.int8()),
])
def test_reading_accepts_other_column_types(make_predictions, attr, column_type):
    predictions = make_predictions()
    table = to_arrow_table(predictions)
    table = _set_column(table, attr, table.column(attr).cast(column_type))
    _assert_equal(from_arrow_table(table), predictions)


@pytest.mark.parametrize("make_table, message", [
    (lambda: to_arrow_table(_predictions()).append_column("c", pa.array([1, 2, 3])),
     r"\['c'\] are not attributes"),
    (lambda: _set_column(to_arrow_table(_predictions()), "b", pa.array([1, 2, 2])),
     "all hold class indices"),
    (lambda: _set_column(to_arrow_table(_hard_predictions()), "a",
                         pa.array([2, IGNORE_LABEL_INDEX, 0])),
     "invalid attribute prediction is null"),
    (lambda: _set_column(to_arrow_table(_hard_predictions()), "a", pa.array([3, None, 0])),
     r"class indices must be in \[0, 3\)"),
    (lambda: _set_column(_set_column(to_arrow_table(_predictions()), "a", pa.nulls(3)),
                         "b", pa.nulls(3)),
     "null type"),
])
def test_reading_rejects_invalid_columns(make_table, message):
    with pytest.raises(PredictionFormatError, match=message):
        from_arrow_table(make_table())


@pytest.mark.parametrize("change, message", [
    (lambda h: h.pop("dataset"), r"no fields \['dataset'\]"),
    (lambda h: h.update(format_version="1"), "must be an integer"),
    (lambda h: h.update(format_version=2), "not supported"),
    (lambda h: h.update(format_version=0), "not supported"),
    (lambda h: h.update(context_offsets=[0, 0]), "duplicates"),
    (lambda h: h.pop("model"), r"no fields \['model'\]"),
    (lambda h: h.update(model={"training_splits": None, "early_stopping_splits": []}),
     "'model'"),
    (lambda h: h.update(model={"method_name": "m", "early_stopping_splits": []}),
     "'training_splits'"),
    (lambda h: h.update(model={"method_name": "m", "training_splits": []}),
     "'early_stopping_splits'"),
    (lambda h: h.update(model={"method_name": "m", "training_splits": [],
                               "early_stopping_splits": None}), "'early_stopping_splits'"),
    (lambda h: h.update(model={"method_name": "m", "training_splits": "train",
                               "early_stopping_splits": []}), "'model'"),
    (lambda h: h.update(model={"method_name": "m", "training_splits": [],
                               "early_stopping_splits": [], "seed": "1"}), "'model'"),
    (lambda h: h.update(model={"method_name": "m", "training_splits": [],
                               "early_stopping_splits": [], "seed": True}), "'model'"),
    (lambda h: h.update(model={"method_name": "m", "training_splits": [],
                               "early_stopping_splits": [], "method_display_name": 1}),
     "'method_display_name'"),
    (lambda h: h.update(model={"method_name": "m", "training_splits": [],
                               "early_stopping_splits": [], "method_display_name": " "}),
     "must not be blank"),
    (lambda h: h.update(model={"method_name": "m", "training_splits": ["a", "a"],
                               "early_stopping_splits": []}), "duplicates"),
    (lambda h: h.update(model={"method_name": "m", "training_splits": ["a"],
                               "early_stopping_splits": ["a"]}),
     "both training and early stopping"),
    (lambda h: h.update(context_offsets=["0"]), "context_offsets"),
    (lambda h: h.update(attribute_to_irap_codes={"a": [10, 10, 30], "b": [1, 2]}), "duplicate"),
    (lambda h: h.update(attribute_to_irap_codes={"a": [10, "20", 30], "b": [1, 2]}), "not ints"),
])
def test_reading_rejects_invalid_headers(change, message):
    with pytest.raises(PredictionFormatError, match=message):
        from_arrow_table(_change_header(to_arrow_table(_predictions()), change))


def test_model_needs_only_a_method_name_and_splits():
    def change(header):
        header["model"] = {"method_name": "m", "training_splits": None,
                           "early_stopping_splits": [], "epoch": 3}

    loaded = from_arrow_table(_change_header(to_arrow_table(_predictions()), change))
    assert loaded.header.model == ModelInfo(method_name="m", training_splits=None,
                                            early_stopping_splits=())
    assert loaded.header.model.label == "m"
    assert loaded.header.model.method_display_name is None
    assert loaded.header.model.shown_method_name == "m"


@pytest.mark.parametrize("method_display_name", [None, "Model M"])
def test_model_round_trips(tmp_path, method_display_name):
    model = ModelInfo(method_name="m", training_splits=("train", "unlabeled_val"),
                      early_stopping_splits=("val",), seed=3,
                      method_display_name=method_display_name, details={"lr": 0.1})
    predictions = dc.replace(_predictions(), header=dc.replace(HEADER, model=model))
    write_predictions(tmp_path / "m.predictions.parquet", predictions)
    loaded = read_predictions(tmp_path / "m.predictions.parquet")
    assert loaded.header.model == model
    assert loaded.header.model.details == model.details
    assert loaded.header.model.method_display_name == method_display_name
    assert loaded.header.model.label == "m/seed3"
    assert loaded.header.model.shown_method_name == (method_display_name or "m")


def test_model_identity_ignores_details_display_name_and_split_order():
    model = ModelInfo(method_name="m", training_splits=("train", "extra"),
                      early_stopping_splits=("val", "val2"), seed=1, details={"commit": "a"})
    same = ModelInfo(method_name="m", training_splits=["extra", "train"],
                     early_stopping_splits=("val2", "val"), seed=1, method_display_name="M",
                     details={"commit": "b"})
    assert model == same and hash(model) == hash(same)
    assert model.training_splits == ("extra", "train")
    assert model != dc.replace(model, early_stopping_splits=())


def test_split_use():
    model = ModelInfo(method_name="m", training_splits=("train",),
                      early_stopping_splits=("val",))
    assert [model.get_split_use(s) for s in ("train", "val", "test")] == \
           ["training", "early_stopping", "held_out"]
    unknown = dc.replace(model, training_splits=None)
    assert [unknown.get_split_use(s) for s in ("train", "val")] == ["unknown", "unknown"]


def test_check_split_names():
    model = ModelInfo(method_name="m", training_splits=("train",),
                      early_stopping_splits=("valid",))
    check_split_names(dc.replace(model, early_stopping_splits=()), {"train", "val"})
    with pytest.raises(PredictionFormatError, match=r"unknown splits \['valid'\]"):
        check_split_names(model, {"train", "val"})
