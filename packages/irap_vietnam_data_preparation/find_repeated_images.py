"""Find the segments with equal images, consecutive or anywhere in the dataset, and count them.

Usage:
    python find_repeated_images.py <data_dir> [--skip-missing-images]

Reads the image paths (`segment_id_to_data_paths_rel.json`) and the road sequences
(`road_id_to_segment_id_sequence.json`) that build_metadata.py writes. build_metadata.py
already leaves out images because of equal images (see
`repeated_images.find_images_to_exclude`), so this script reports the equal images that remain.

repeated_images.py defines equal images, their groups and the folder spans of a group. A
group of n segments has n - 1 repeated images. A run is a maximal stretch of consecutive
segments of a road sequence that are in the same group. A group is consecutive if one run
holds all its segments. The position of a run says where it lies in its road sequence: at the
start, in the middle, at the end, or it is the whole road sequence.

Folder spans show what road sequences can hide: a road sequence can end inside a run of equal
images of a folder, and one different image splits a run.

The script prints, and writes to `_work/repeated_images_report.json`:
- the road sequences with runs of 2 or more segments, with the first and last seg_id, the
  length and the position of each run,
- the groups that are not consecutive, with their runs,
- the folder spans of each group, with the first and last seg_id, the number of segments, the
  number of other segments and the position of each span.

Most files are not read whole (see repeated_images.py).

A missing image file is an error. With `--skip-missing-images`, it is left out of its road
sequence, so that the segments around it count as consecutive, and counted in the report.
"""

import argparse
import dataclasses
import itertools
import json
import sys
import typing as T
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import layout
from repeated_images import (
    RUN_POSITIONS, FolderSpan, find_equal_file_groups, find_folder_spans,
    get_folder_to_seg_ids, get_run_position, get_seg_id_to_group_index)


@dataclasses.dataclass(frozen=True)
class Run:
    """A maximal stretch of consecutive segments of a road sequence with equal images."""
    road_id: str
    group_index: int
    seg_ids: list[str]
    position: str  # in the road sequence, one of `RUN_POSITIONS`


def find_runs(road_to_seg_ids: T.Mapping[str, T.Sequence[str]], groups: list[list[str]]
              ) -> list[Run]:
    """Finds the runs of each road sequence, in the order of the road sequences.

    A segment that is in no group ends a run. A run can be a single segment, if its group has
    other segments elsewhere.
    """
    seg_id_to_group_index = get_seg_id_to_group_index(groups)
    runs = []
    for road_id, seg_ids in road_to_seg_ids.items():
        start = 0
        for group_index, run_seg_ids in itertools.groupby(
                seg_ids, key=seg_id_to_group_index.get):
            run_seg_ids = list(run_seg_ids)
            if group_index is not None:
                position = get_run_position(start, len(run_seg_ids), len(seg_ids))
                runs.append(Run(road_id, group_index, run_seg_ids, position))
            start += len(run_seg_ids)
    return runs


def find_repeated_images(road_to_seg_ids: T.Mapping[str, T.Sequence[str]],
                         seg_id_to_image_path: T.Mapping[str, Path], *,
                         skip_missing_images: bool
                         ) -> tuple[list[list[str]], list[Run], list[FolderSpan], list[str]]:
    """Returns the groups of equal images, their runs and folder spans, and the seg_ids
    without an image file."""
    segs_without_path = {s for seg_ids in road_to_seg_ids.values() for s in seg_ids
                         if s not in seg_id_to_image_path}
    if segs_without_path:
        raise ValueError(f"{len(segs_without_path)} segments of road sequences have no image"
                         f" path, e.g. {sorted(segs_without_path)[:5]}.")

    missing_seg_ids = [s for s, path in tqdm(seg_id_to_image_path.items(),
                                             desc="Checking image files")
                       if not path.is_file()]
    if missing_seg_ids and not skip_missing_images:
        seg_id = missing_seg_ids[0]
        raise FileNotFoundError(f"{len(missing_seg_ids)} images are missing, e.g. the image of"
                                f" segment {seg_id}: {seg_id_to_image_path[seg_id]}")
    missing = set(missing_seg_ids)

    seg_id_to_existing_image_path = {s: path for s, path in seg_id_to_image_path.items()
                                     if s not in missing}
    groups = find_equal_file_groups(seg_id_to_existing_image_path)
    runs = find_runs({road_id: [s for s in seg_ids if s not in missing]
                      for road_id, seg_ids in road_to_seg_ids.items()}, groups)
    folder_spans = find_folder_spans(get_folder_to_seg_ids(seg_id_to_existing_image_path),
                                     groups)
    return groups, runs, folder_spans, missing_seg_ids


def _get_num_repeated_images(equal_image_seg_ids: T.Iterable[T.Sequence[str]]) -> int:
    return sum(len(seg_ids) - 1 for seg_ids in equal_image_seg_ids)


def _count_by_position(items: T.Sequence[Run | FolderSpan]) -> dict[str, int]:
    return {position: sum(item.position == position for item in items)
            for position in RUN_POSITIONS}


def to_report(groups: list[list[str]], runs: list[Run], folder_spans: list[FolderSpan],
              road_to_seg_ids: T.Mapping[str, T.Sequence[str]], num_images: int,
              missing_seg_ids: list[str]) -> dict:
    group_index_to_runs = defaultdict(list)
    road_to_long_runs = defaultdict(list)  # runs of 2 or more segments
    for run in runs:
        group_index_to_runs[run.group_index].append(run)
        if len(run.seg_ids) > 1:
            road_to_long_runs[run.road_id].append(run)
    group_index_to_folder_spans = defaultdict(list)
    for span in folder_spans:
        group_index_to_folder_spans[span.group_index].append(span)

    group_reports = sorted(
        ({"seg_ids": seg_ids,
          "is_consecutive": any(len(run.seg_ids) == len(seg_ids)
                                for run in group_index_to_runs[i]),
          "runs": [{"road_id": run.road_id, "position": run.position, "seg_ids": run.seg_ids}
                   for run in group_index_to_runs[i]],
          "folder_spans": [{"folder": span.folder, "position": span.position,
                            "first_seg_id": span.seg_ids[0], "last_seg_id": span.seg_ids[-1],
                            "num_segments": len(span.seg_ids),
                            "num_other_segments": span.num_other_segments}
                           for span in group_index_to_folder_spans[i]]}
         for i, seg_ids in enumerate(groups)),
        key=lambda group: -len(group["seg_ids"]))
    road_reports = sorted(
        ((road_id, {"num_segments": len(road_to_seg_ids[road_id]),
                    "num_repeated_images": _get_num_repeated_images(
                        run.seg_ids for run in road_runs),
                    "runs": [{"position": run.position, "seg_ids": run.seg_ids}
                             for run in road_runs]})
         for road_id, road_runs in road_to_long_runs.items()),
        key=lambda item: -item[1]["num_repeated_images"])
    long_runs = [run for runs in road_to_long_runs.values() for run in runs]
    return {
        "num_images": num_images,
        "num_missing_images": len(missing_seg_ids),
        "missing_image_seg_ids": missing_seg_ids,
        "num_repeated_images": _get_num_repeated_images(groups),
        "num_equal_image_groups": len(groups),
        "num_non_consecutive_equal_image_groups":
            sum(not group["is_consecutive"] for group in group_reports),
        "num_road_sequences": len(road_to_seg_ids),
        "num_road_sequences_with_consecutive_equal_images": len(road_reports),
        "position_to_num_consecutive_runs": _count_by_position(long_runs),
        "position_to_num_folder_spans": _count_by_position(folder_spans),
        "road_id_to_consecutive_equal_images": dict(road_reports),
        "equal_image_groups": group_reports,
    }


def _format_run(run: dict) -> str:
    seg_ids = run["seg_ids"]
    if len(seg_ids) == 1:
        return f"{seg_ids[0]} ({run['position']})"
    return f"{seg_ids[0]}-{seg_ids[-1]} ({len(seg_ids)}, {run['position']})"


def _format_folder_span(span: dict) -> str:
    if span["num_segments"] == 1:
        return f"{span['folder']}: {span['first_seg_id']} ({span['position']})"
    return (f"{span['folder']}: {span['first_seg_id']}-{span['last_seg_id']}"
            f" ({span['num_segments']} segments, {span['num_other_segments']} other,"
            f" {span['position']})")


def print_report(report: dict) -> None:
    if report["road_id_to_consecutive_equal_images"]:
        print("Road sequences with equal images in consecutive segments:")
    for road_id, repeated in report["road_id_to_consecutive_equal_images"].items():
        runs = ", ".join(map(_format_run, repeated["runs"]))
        print(f"{road_id}: {repeated['num_repeated_images']} repeated of"
              f" {repeated['num_segments']} images, segments {runs}")
    positions = ", ".join(f"{num_runs} {position}" for position, num_runs
                          in report["position_to_num_consecutive_runs"].items())
    print(f"\n{report['num_road_sequences_with_consecutive_equal_images']} of"
          f" {report['num_road_sequences']} road sequences have consecutive equal images."
          f" Runs by position in their road sequence: {positions}.")

    non_consecutive_groups = [g for g in report["equal_image_groups"] if not g["is_consecutive"]]
    if non_consecutive_groups:
        print("\nGroups of equal images that are not consecutive segments of a road sequence:")
        for group in non_consecutive_groups:
            runs = "; ".join(f"{run['road_id']}: {_format_run(run)}" for run in group["runs"])
            print(f"{len(group['seg_ids'])} images: {runs}")

    if report["equal_image_groups"]:
        print("\nFolder spans of the groups of equal images (from the first to the last image"
              " of the group, in seg_id order):")
    for group in report["equal_image_groups"]:
        spans = "; ".join(map(_format_folder_span, group["folder_spans"]))
        print(f"{len(group['seg_ids'])} images: {spans}")
    positions = ", ".join(f"{num_spans} {position}" for position, num_spans
                          in report["position_to_num_folder_spans"].items())
    print(f"Folder spans by position in their folder: {positions}.")

    print(f"\n{report['num_equal_image_groups']} groups of equal images"
          f" ({len(non_consecutive_groups)} not consecutive):"
          f" {report['num_repeated_images']} repeated of {report['num_images']} images.")
    if report["num_missing_images"]:
        print(f"Skipped {report['num_missing_images']} missing images.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="iRAP-Vietnam dataset root.")
    parser.add_argument("--skip-missing-images", action="store_true",
                        help="Leave out segments whose image file is missing instead of failing.")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    road_to_seg_ids = json.loads(
        layout.road_id_to_segment_id_sequence_path(data_dir).read_text(encoding="utf-8"))
    seg_id_to_data_paths = json.loads(
        layout.segment_id_to_data_paths_rel_path(data_dir).read_text(encoding="utf-8"))
    seg_id_to_image_path = {s: data_dir / paths["rgb"]
                            for s, paths in seg_id_to_data_paths.items()}

    groups, runs, folder_spans, missing_seg_ids = find_repeated_images(
        road_to_seg_ids, seg_id_to_image_path, skip_missing_images=args.skip_missing_images)
    report = to_report(groups, runs, folder_spans, road_to_seg_ids,
                       num_images=len(seg_id_to_image_path) - len(missing_seg_ids),
                       missing_seg_ids=missing_seg_ids)
    print_report(report)

    report_path = layout.repeated_images_report_path(data_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {report_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
