import dataclasses as dc
import hashlib
import os
import time
import zipfile

import irap_evaluation as ie
import pytest
from run_helpers import (
    OTHER_SPLIT,
    make_model,
    with_context_offsets,
    with_split,
    write_upload,
)
from synthetic_vietnam import SPLIT

from irap_evaluation_server import uploads
from irap_evaluation_server.uploads import (
    apply_upload,
    is_archive_upload,
    plan_upload,
    read_upload_headers,
)


def make_model_files(dataset_contexts, splits=(SPLIT, OTHER_SPLIT), **kwargs):
    """Makes the predictions of the model on each split, by archive member name."""
    predictions = make_model(dataset_contexts, **kwargs)
    return {f"model/m.{split}.predictions.parquet": with_split(predictions, split)
            for split in splits}


def write_archive(path, member_name_to_contents, tmp_path, compression=zipfile.ZIP_STORED):
    """Writes a .zip archive of predictions, or of texts for other files."""
    with zipfile.ZipFile(path, "w", compression=compression) as zip_file:
        zip_file.writestr("model/", "")  # A directory entry, which is skipped.
        for member_name, contents in member_name_to_contents.items():
            if isinstance(contents, str):
                zip_file.writestr(member_name, contents)
                continue
            member_path = tmp_path / "member.parquet"
            ie.write_predictions(member_path, contents)
            zip_file.write(member_path, member_name)
    return path


def plan_archive(archive, dataset_contexts, member_name_to_contents, tmp_path, seed=None):
    path = write_archive(archive.make_upload_path("m.zip"), member_name_to_contents, tmp_path)
    return plan_upload(archive, dataset_contexts, path, file_name="m.zip", seed=seed,
                       description="all splits")


def read_member_sha256s(path):
    with zipfile.ZipFile(path) as zip_file:
        return {i.filename: hashlib.sha256(zip_file.read(i)).hexdigest()
                for i in zip_file.infolist() if not i.is_dir()}


def test_uploads_are_checked(archive, dataset_contexts):
    predictions = make_model(dataset_contexts)

    def plan(predictions):
        return plan_upload(archive, dataset_contexts, write_upload(archive, predictions),
                           file_name="m")

    other_dataset = dc.replace(predictions, header=dc.replace(predictions.header, dataset="bh"))
    with pytest.raises(ValueError, match="dataset 'bh'"):
        plan(other_dataset)
    incomplete = ie.select_segments(predictions, predictions.segment_ids[:-3])
    with pytest.raises(ValueError, match="No predictions for 3"):
        plan(incomplete)
    wrong_codes = dc.replace(predictions, attribute_to_irap_codes={
        **predictions.attribute_to_irap_codes, "Curvature": (1, 3)})
    with pytest.raises(ie.PredictionFormatError, match="Curvature"):
        plan(wrong_codes)
    assert archive.list_models() == []


def test_remove_uploads(archive):
    archive.make_upload_path("m.parquet").write_bytes(b"x")
    assert archive.remove_uploads() == 1
    assert not list(archive.uploads_dir.iterdir())

    old_path = archive.make_upload_path("old.parquet")
    new_path = archive.make_upload_path("new.parquet")
    old_path.write_bytes(b"x")
    new_path.write_bytes(b"x")
    old_time_s = time.time() - 3600
    os.utime(old_path, (old_time_s, old_time_s))
    assert archive.remove_uploads(older_than_s=600) == 1
    assert list(archive.uploads_dir.iterdir()) == [new_path]


def test_upload_paths(archive):
    file_path = archive.make_upload_path("model.parquet")
    archive_path = archive.make_upload_path("MODEL.ZIP")
    assert not is_archive_upload(file_path)
    assert is_archive_upload(archive_path)
    for path in (file_path, archive_path):
        assert path.parent == archive.uploads_dir
        assert archive.get_upload_path(path.name) == path
    for name in ("../archive.sqlite3", f"../{file_path.name}", "x.zip", file_path.stem):
        assert archive.get_upload_path(name) is None


def test_discarded_upload_keeps_only_the_uploaded_file(archive, dataset_contexts):
    path = write_upload(archive, make_model(dataset_contexts, model_seed=1))
    planned = plan_upload(archive, dataset_contexts, path, file_name="m", seed=2)
    assert len(list(archive.uploads_dir.iterdir())) == 2  # With the rewritten file.
    planned.discard()
    assert list(archive.uploads_dir.iterdir()) == [path]


def test_upload_of_an_archive(archive, two_split_contexts, tmp_path):
    planned = plan_archive(archive, two_split_contexts,
                           make_model_files(two_split_contexts, model_seed=1), tmp_path)
    member_sha256s = read_member_sha256s(planned.uploaded_path)
    submissions = apply_upload(archive, planned, submitter="Alice")

    assert not list(archive.uploads_dir.iterdir())
    assert [(s.split, s.label, s.model.description) for s in submissions] == [
        (SPLIT, "m/seed1", "all splits"), (OTHER_SPLIT, "m/seed1", "all splits")]
    assert archive.list_submissions() == submissions[::-1]
    actions = archive.list_actions()[::-1]
    assert [a.submission_id for a in actions] == [s.id for s in submissions]
    for action, submission, (member_name, sha256) in zip(
            actions, submissions, member_sha256s.items(), strict=True):
        assert action.details == {"file_name": "m.zip", "archive_member": member_name,
                                  "notes": []}
        assert submission.file_sha256 == sha256
        assert hashlib.sha256(archive.get_predictions_path(submission.id).read_bytes()
                              ).hexdigest() == sha256


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_read_upload_headers(archive, two_split_contexts, tmp_path, compression):
    files = make_model_files(two_split_contexts, model_seed=1)
    headers = [p.header for p in files.values()]
    path = write_archive(archive.make_upload_path("m.zip"), files, tmp_path, compression)
    assert read_upload_headers(path, file_name="m.zip") == headers
    path = write_upload(archive, next(iter(files.values())))
    assert read_upload_headers(path, file_name="m") == headers[:1]


def test_read_upload_headers_refuses_invalid_files(archive, two_split_contexts, tmp_path,
                                                   monkeypatch):
    files = make_model_files(two_split_contexts)
    path = write_archive(archive.make_upload_path("m.zip"),
                         {**files, "model/x.parquet": "not parquet"}, tmp_path)
    with pytest.raises(ValueError, match=r"^model/x\.parquet: "):
        read_upload_headers(path, file_name="m.zip")
    path = write_archive(archive.make_upload_path("m.zip"), files, tmp_path)
    monkeypatch.setattr(uploads, "MAX_UPLOAD_NUM_BYTES", 1000)
    with pytest.raises(ValueError, match="The files of the archive have more"):
        read_upload_headers(path, file_name="m.zip")


def test_archive_seed_replaces_the_seeds_of_all_files(archive, two_split_contexts, tmp_path):
    planned = plan_archive(archive, two_split_contexts,
                           make_model_files(two_split_contexts, model_seed=1), tmp_path, seed=5)
    submissions = apply_upload(archive, planned, submitter="Alice")
    assert [s.label for s in submissions] == ["m/seed5", "m/seed5"]
    for submission in submissions:
        stored = ie.read_predictions(archive.get_predictions_path(submission.id))
        assert stored.header.model.seed == 5
    assert all(a.details["original_seed"] == 1 for a in archive.list_actions())
    assert not list(archive.uploads_dir.iterdir())


def _make_files_of_two_models(dataset_contexts):
    files = make_model_files(dataset_contexts, splits=(SPLIT,), model_seed=1)
    other = make_model_files(dataset_contexts, splits=(OTHER_SPLIT,), model_seed=2)
    return files | other


def _make_files_of_two_offsets(dataset_contexts):
    files = make_model_files(dataset_contexts, splits=(SPLIT,))
    [(name, other)] = make_model_files(dataset_contexts, splits=(OTHER_SPLIT,)).items()
    return files | {name: with_context_offsets(other, (0,))}


def _make_one_split_twice(dataset_contexts):
    [predictions] = make_model_files(dataset_contexts, splits=(SPLIT,)).values()
    return {"a.parquet": predictions, "b.parquet": predictions}


@pytest.mark.parametrize("make_member_name_to_contents, message", [
    (_make_files_of_two_models,
     (r"test\.predictions\.parquet: It is of the model m/seed2 on vietnam, but"
      r" .*val\.predictions\.parquet is of m/seed1 on vietnam\.")),
    (_make_files_of_two_offsets,
     (r"test\.predictions\.parquet: It has the context offsets \(0,\), but"
      r" .*val\.predictions\.parquet has the context offsets \(0, -1\)\.")),
    (_make_one_split_twice, r"^b\.parquet: It is of the split 'val', like a\.parquet"),
    (lambda contexts: {**make_model_files(contexts), "model/notes.txt": "x"},
     r"only prediction files .* Other files: model/notes\.txt\.$"),
    (lambda contexts: {}, r"has no files"),
])
def test_archive_contents_are_checked(archive, two_split_contexts, tmp_path,
                                      make_member_name_to_contents, message):
    path = write_archive(archive.make_upload_path("m.zip"),
                         make_member_name_to_contents(two_split_contexts), tmp_path)
    with pytest.raises(ValueError, match=message):
        plan_upload(archive, two_split_contexts, path, file_name="m.zip")
    assert archive.list_models() == []
    assert list(archive.uploads_dir.iterdir()) == [path]


def test_upload_that_is_not_a_zip_archive_is_refused(archive, dataset_contexts):
    path = archive.make_upload_path("m.zip")
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match=r"m\.zip is not a \.zip archive"):
        plan_upload(archive, dataset_contexts, path, file_name="m.zip")
    assert path.exists()


def test_macos_metadata_in_an_archive_is_skipped(archive, two_split_contexts, tmp_path):
    files = make_model_files(two_split_contexts)
    planned = plan_archive(
        archive, two_split_contexts,
        {**files, **{f"__MACOSX/{name.replace('/', '/._')}": "x" for name in files}}, tmp_path)
    assert len(planned.update.new_submissions) == 2


def test_encrypted_archive_member_is_refused(archive, dataset_contexts):
    path = archive.make_upload_path("m.zip")
    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("m.parquet", b"x")
    data = bytearray(path.read_bytes())
    data[data.rfind(b"PK\x01\x02") + 8] |= 0x01  # The encryption flag of the central directory.
    path.write_bytes(data)
    with pytest.raises(ValueError, match=r"^m\.parquet: It cannot be extracted: .*encrypted"):
        plan_upload(archive, dataset_contexts, path, file_name="m.zip")
    assert list(archive.uploads_dir.iterdir()) == [path]


def test_archive_extraction_is_limited(archive, dataset_contexts, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_UPLOAD_NUM_BYTES", 1000)
    path = archive.make_upload_path("m.zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("m.parquet", bytes(10 ** 6))  # A zip bomb, compressed to 1 kB.
    with pytest.raises(ValueError, match=r"^m\.parquet: The files of the archive have more"):
        plan_upload(archive, dataset_contexts, path, file_name="m.zip")
    assert list(archive.uploads_dir.iterdir()) == [path]  # Without the extracted part.
