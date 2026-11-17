import concurrent.futures
import datetime
import functools
import json
import multiprocessing

import irap_evaluation as ie
import pandas as pd
import pytest
from irap_data.metadata import MetaFiles
from irap_evaluation.reports import coding_tables
from run_helpers import OTHER_SPLIT, add_model, make_model, with_split
from synthetic_vietnam import SPLIT, make_predictions

from irap_evaluation_server.coding_table_export import (
    MAX_CODING_TABLE_FILE_NAME_BYTES,
    parse_coding_date,
    plan_model_coding_table,
    write_model_coding_table,
)
from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.datasets import load_dataset_contexts


@pytest.mark.parametrize("is_hard", [False, True])
def test_coding_table_equals_irap_evaluation(archive, dataset_contexts, tmp_path, is_hard):
    context = dataset_contexts["vietnam"]
    submission = add_model(archive, dataset_contexts,
                           make_model(dataset_contexts, is_hard=is_hard))
    prediction_paths = [archive.get_predictions_path(submission.id)]
    path = tmp_path / "table.xlsx"
    assert write_model_coding_table(prediction_paths, context.metadata, path, coder_name="Bo",
                                    coding_date="2026-10-03") == path
    predictions = ie.read_predictions(prediction_paths[0])
    template = coding_tables.load_coding_table_template()
    expected = coding_tables.predictions_to_coding_table(
        [predictions], context.metadata, template, coder_name="Bo", coding_date="2026-10-03")
    segment_column = template.field_to_column["segment_id"]
    sheets = pd.read_excel(path, sheet_name=None, dtype={segment_column: str})
    assert list(sheets) == (["Coding table"] if is_hard else ["Coding table", "Confidence"])
    stored = sheets["Coding table"]
    assert list(stored.columns) == list(expected.columns)
    assert len(stored) == len(expected) == predictions.num_segments
    assert list(stored[segment_column]) == list(expected[segment_column])

    csv_path = tmp_path / "table.csv"
    write_model_coding_table(prediction_paths, context.metadata, csv_path, coder_name="Bo",
                             coding_date="2026-10-03")
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".csv"] == ["table.csv"]
    with pytest.raises(ValueError, match="xlsx or .csv"):
        write_model_coding_table(prediction_paths, context.metadata, tmp_path / "t.json",
                                 coder_name="Bo", coding_date="2026-10-03")


@pytest.fixture
def disjoint_split_contexts(metadata_dir):
    """The dataset with the roads 0 and 1 in `SPLIT` and the others in `OTHER_SPLIT`."""
    splits_path = metadata_dir / MetaFiles.SPLITS
    road_path = metadata_dir / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE
    sequences = list(json.loads(road_path.read_text(encoding="utf-8")).values())
    splits = {SPLIT: [s for seq in sequences[:2] for s in seq],
              OTHER_SPLIT: [s for seq in sequences[2:] for s in seq]}
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    return load_dataset_contexts({"vietnam": DatasetConfig(
        dataset_dir=metadata_dir, metadata_dir=metadata_dir, analysis_splits=())})


@pytest.fixture
def two_split_submissions(archive, disjoint_split_contexts):
    """The submissions of one model of `SPLIT` and `OTHER_SPLIT`."""
    metadata = disjoint_split_contexts["vietnam"].metadata
    return tuple(
        add_model(archive, disjoint_split_contexts, with_split(make_predictions(
            metadata.vocabulary, list(metadata.splits[split]), name="m", seed=0, model_seed=0), split))
        for split in (SPLIT, OTHER_SPLIT))


def test_plan_model_coding_table(archive, disjoint_split_contexts, two_split_submissions,
                                 tmp_path):
    first, second = two_split_submissions

    def plan(splits, file_format="csv", coding_date="", coder_name=""):
        return plan_model_coding_table(archive, disjoint_split_contexts, first.model.id,
                                       splits, file_format, coder_name=coder_name,
                                       coding_date=coding_date)

    job = plan([OTHER_SPLIT, SPLIT])
    assert job.file_name == f"m_seed0.{OTHER_SPLIT}_{SPLIT}.coding_table.csv"
    assert job.prediction_paths == (archive.get_predictions_path(second.id),
                                    archive.get_predictions_path(first.id))
    assert job.coder_name == "m"  # The method name.
    path = write_model_coding_table(job.prediction_paths, job.metadata, tmp_path / "t.csv",
                                    coder_name=job.coder_name, coding_date=job.coding_date)
    stored = pd.read_csv(path)
    assert len(stored) == first.num_segments + second.num_segments
    coder_column = coding_tables.load_coding_table_template().field_to_column["coder_name"]
    assert set(stored[coder_column]) == {"m"}
    for splits, file_format, coding_date, message in [
            ([], "csv", "", "at least one split"),
            (["missing"], "csv", "", "no files of the splits"),
            ([SPLIT], "json", "", "Unknown format"),
            ([SPLIT], "csv", "03.10.2026", "YYYY-MM-DD")]:
        with pytest.raises(ValueError, match=message):
            plan(splits, file_format, coding_date)
    with pytest.raises(ValueError, match="formula"):
        plan([SPLIT], "csv", coder_name="=1+1")
    plan([SPLIT], "xlsx", coder_name="=1+1")  # As text.


def test_coding_table_is_written_in_a_spawned_process(archive, disjoint_split_contexts,
                                                      two_split_submissions, tmp_path):
    """Like the server with `run.cpu_bound`, which pickles the function and the job."""
    first, second = two_split_submissions
    job = plan_model_coding_table(archive, disjoint_split_contexts, first.model.id,
                                  [SPLIT, OTHER_SPLIT], "xlsx", coder_name="", coding_date="")
    with concurrent.futures.ProcessPoolExecutor(
            1, mp_context=multiprocessing.get_context("spawn")) as executor:
        path = executor.submit(functools.partial(
            write_model_coding_table, job.prediction_paths, job.metadata,
            tmp_path / "t.xlsx", coder_name=job.coder_name,
            coding_date=job.coding_date)).result()
    assert len(pd.read_excel(path)) == first.num_segments + second.num_segments


def test_plan_model_coding_table_of_a_long_method_name(archive, dataset_contexts):
    # E.g. the default method name of an ensemble of many methods.
    name = "+".join(f"method{i}_seeds0-4" for i in range(30))
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, name=name))
    job = plan_model_coding_table(archive, dataset_contexts, submission.model.id, [SPLIT],
                                  "xlsx", coder_name="Bo", coding_date="")
    assert len(job.file_name.encode()) <= MAX_CODING_TABLE_FILE_NAME_BYTES
    assert job.file_name.endswith(f".{SPLIT}.coding_table.xlsx")


def test_parse_coding_date():
    assert parse_coding_date("") == datetime.datetime.now(datetime.UTC).date().isoformat()
    assert parse_coding_date("2026-10-03") == "2026-10-03"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_coding_date("03.10.2026")
