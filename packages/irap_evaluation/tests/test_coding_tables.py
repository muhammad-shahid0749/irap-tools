import codecs
import dataclasses as dc

import numpy as np
import pandas as pd
import pytest

from irap_data.attrs import IRAP_VIETNAM_ATTRS_ALL
from irap_evaluation.reports.coding_tables import (
    get_unfilled_columns,
    get_unmapped_attributes,
    load_coding_table_template,
    predictions_to_coding_table,
    predictions_to_confidence_table,
    write_coding_table,
)
from irap_evaluation.predictions import (
    PredictionFormatError,
    select_segments,
    to_irap_codes,
)

from synthetic_vietnam import ROAD_LENGTH, SPLIT, make_predictions, to_hard_predictions


def _without_attribute(predictions, attribute):
    attribute_to_irap_codes = {a: c for a, c in predictions.attribute_to_irap_codes.items()
                               if a != attribute}
    return dc.replace(predictions, attribute_to_irap_codes=attribute_to_irap_codes,
                      probs={a: predictions.probs[a] for a in attribute_to_irap_codes})


@pytest.fixture
def template():
    return load_coding_table_template()


def test_packaged_template(template):
    assert len(template.columns) == 69
    assert template.columns[:4] == ("Coder name", "Coding date", "Road survey date",
                                    "Image reference")
    # Every Vietnam attribute without a column has a documented reason (see the template).
    assert get_unmapped_attributes(template, IRAP_VIETNAM_ATTRS_ALL) == [
        "Service road", "Speed limit", "Intersecting road volume"]
    unfilled = get_unfilled_columns(template, ["Grade"])
    assert "Grade" not in unfilled and "Curvature" in unfilled and "Section" not in unfilled


def test_template_rejects_unknown_columns(template):
    with pytest.raises(ValueError, match="not in the template"):
        dc.replace(template, attribute_to_column={"Grade": "Gradient"})


def test_coding_table(metadata, template, tmp_path):
    segments = list(reversed(metadata.splits[SPLIT]))
    predictions = make_predictions(metadata.vocabulary, segments, name="m", seed=0)
    # Written as text, not as a formula.
    table = predictions_to_coding_table([predictions], metadata, template, coder_name="=1+1",
                                        coding_date="2026-09-28")
    assert tuple(table.columns) == template.columns
    # Rows follow the road sequences, whatever the order of the predictions.
    assert list(table["Image reference"][:ROAD_LENGTH]) == \
           metadata.road_id_to_segment_id_sequence["road0"]
    first, second = (metadata.segment_id_to_road_data[s] for s in table["Image reference"][:2])
    assert table["Latitude end"][0] == second["lat"]
    assert table["Distance"][1] == pytest.approx(second["distance_km"])
    assert pd.isna(table["Latitude end"][ROAD_LENGTH - 1])  # the last segment of a road
    codes = to_irap_codes(predictions)
    row_of = {sid: i for i, sid in enumerate(predictions.segment_ids)}
    rows = [row_of[s] for s in table["Image reference"]]
    np.testing.assert_array_equal(table["Carriageway"],
                                  codes["Carriageway label"][rows])
    assert table["Speed limit"].isna().all()

    confidence = predictions_to_confidence_table([predictions], metadata, template)
    assert list(confidence["Image reference"]) == list(table["Image reference"])
    np.testing.assert_allclose(confidence["Lane width"].astype(float),
                               predictions.probs["Lane width"][rows].max(1))
    paths = write_coding_table(tmp_path / "table.xlsx", table, confidence)
    sheets = pd.read_excel(paths[0], sheet_name=None, dtype=object)
    assert sheets.keys() == {"Coding table", "Confidence"}
    for stored, written in ((sheets["Coding table"], table),
                            (sheets["Confidence"], confidence)):
        pd.testing.assert_frame_equal(stored, written.astype(object).where(written.notna(), np.nan))
    csv_paths = write_coding_table(tmp_path / "table.csv", table, confidence)
    assert [p.name for p in csv_paths] == ["table.csv", "table.confidence.csv"]
    for path in csv_paths:
        assert path.read_bytes().startswith(codecs.BOM_UTF8)
    assert list(pd.read_csv(csv_paths[0]).columns) == list(template.columns)


def test_confidence_table_refuses_hard_predictions(metadata, template, predictions):
    hard = to_hard_predictions(predictions)
    with pytest.raises(ValueError, match="no probabilities"):
        predictions_to_confidence_table([predictions, hard], metadata, template)


def test_coding_table_of_several_predictions(metadata, template):
    segments = list(metadata.splits[SPLIT])
    whole = make_predictions(metadata.vocabulary, segments, name="m", seed=0)
    # Two parts of the segments, e.g. of two splits, the second without 'Lane width'.
    first = select_segments(whole, segments[1::2])
    second = _without_attribute(select_segments(whole, segments[::2]), "Lane width")

    def make_table(predictions_seq):
        return predictions_to_coding_table(predictions_seq, metadata, template, coder_name="m",
                                           coding_date="2026-09-28")

    table, whole_table = make_table([first, second]), make_table([whole])
    pd.testing.assert_frame_equal(table.drop(columns="Lane width"),
                                  whole_table.drop(columns="Lane width"))
    is_from_first = table["Image reference"].isin(first.segment_ids)
    pd.testing.assert_series_equal(table["Lane width"][is_from_first],
                                   whole_table["Lane width"][is_from_first])
    assert table["Lane width"][~is_from_first].isna().all()
    confidence = predictions_to_confidence_table([first, second], metadata, template)
    assert confidence["Lane width"][~is_from_first].isna().all()
    assert confidence["Grade"].isna().all()  # no attribute fills it

    with pytest.raises(PredictionFormatError, match="several of the predictions"):
        make_table([whole, first])
    other_model = dc.replace(first, header=dc.replace(
        first.header, model=dc.replace(first.header.model, method_name="other")))
    with pytest.raises(PredictionFormatError, match="one model"):
        make_table([other_model, second])
