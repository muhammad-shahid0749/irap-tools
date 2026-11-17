"""Build the iRAP-Vietnam metadata directory from parsed coding tables (per-segment iRAP attribute annotations).

Usage:
    python build_metadata.py <data_dir>

Reads (under <data_dir>):
    _raw/attribute_metadata.json
    _work/rows.parquet                    (from parse_coding_tables.py)
    FRAMES/<video_dir>/...                (nested dir from extract_images.py)

Writes (into <data_dir>/ directly):
    segment_id_to_data_paths_rel.json
    segment_id_to_road_data.json
    road_id_to_segment_id_sequence.json   (section and unlabeled road sequences, see below)
    attribute_metadata.json              (copy of _raw/attribute_metadata.json with int
                                           iRAP codes and without the misleading
                                           attribute_irap_number_to_class_idx)
    unlabeled_segment_ids.json
    unlabeled_sequence_id_to_data.json    (recording -> {segs, centroid} of its unlabeled
                                           segments, for the split editor)
    unlabeled_unlocated_segment_ids.json  (seg_ids from recordings with no
                                           labeled siblings, so no map coordinate
                                           is derivable; written only when
                                           non-empty, and deleted if a previous
                                           build left one behind)

Writes (into <data_dir>/_work/):
    build_report.json                     (counts, warnings)

Images that do not show their segment are left out as if their files were missing,
and the rows of their segments are dropped (``exclude_images``). They are the images
that ``repeated_images.find_images_to_exclude`` finds by their equal files. Each is
printed as a warning and listed in the build report.

Rows are matched to images **by seg_id**: every ``*.png`` under ``FRAMES/`` is
indexed by the integer in its ``_seg<N>.png`` suffix, and each parquet row is
joined to its seg_id. Exactly one image must match each seg_id; if multiple
images share a seg_id a ValueError is raised. Rows with no matching image are
dropped. The image's parent folder name is compared to the row's ``section``;
mismatches are recorded (``prefix_mismatch``) but not dropped – they happen
because the coding-table "Section" cell and the RAR-side video-folder name are
independently authored.

Within each section, the sequence is sorted by Distance ascending. The
adjacency invariant `distance_step_km == 0.01 × abs(seg_id_step)` is
enforced (each unit of seg_id ↔ 10 m). Violating transitions split the
section into multiple road_ids (`<section>__part0`, `<section>__part1`, …)
and are reported on stderr and in the build_report.

Unlabeled segments between two adjacent labeled segments of the same recording
are inserted into that section road sequence
(`interleave_unlabeled_into_road_sequences`). The other unlabeled segments get
unlabeled road sequences of their own, `unlabeled:<recording>:<first seg_id>`
(`make_unlabeled_road_sequences`), so every image is in a road sequence, which
the loader takes the context frames from. Only an unlabeled segment that two
road sequences span is in both.
"""

import argparse
import bisect
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import typing as T

import numpy as np
import pandas as pd

import layout
from parse_coding_tables import ordered_attribute_names
from repeated_images import find_equal_file_groups, find_images_to_exclude, get_folder_to_seg_ids


# Each unit of seg_id step corresponds to 10 m along a section. A 20 m labeled
# segment therefore has a seg_id step of 2; a missing labeled segment between
# two coded rows shows up as step 4, etc.
SEG_ID_STEP_TO_KM = 0.010
ADJACENCY_TOLERANCE_KM = 0.0025  # ±2.5 m
# A larger seg_id step between the images of a folder starts a new recording
# (see split_folders_into_recordings). 100 m allows for a few missing images;
# the recordings of the Vietnam "nan" folder are 89 or more seg_ids apart.
MAX_SEG_ID_STEP_WITHIN_RECORDING = 10

# Captures the integer N in any "..._segN.png" filename.
SEG_ID_FROM_FILENAME_RE = re.compile(r"_seg(\d+)\.png$", re.IGNORECASE)


def index_images_by_seg_id(images_dir: Path) -> dict[str, list[Path]]:
    """Walk ``images_dir`` recursively, indexing ``*.png`` files by seg_id.

    Keys are decimal seg_id strings (matching the parquet's ``seg_id`` column).
    Values are paths relative to ``images_dir.parent`` so they begin with
    ``FRAMES/<video_dir>/<basename>.png``.
    """
    index: dict[str, list[Path]] = defaultdict(list)
    data_root = images_dir.parent
    for path in images_dir.rglob("*.png"):
        m = SEG_ID_FROM_FILENAME_RE.search(path.name)
        if m is None:
            continue
        seg_id = str(int(m.group(1)))  # normalize away leading zeros
        index[seg_id].append(path.relative_to(data_root))
    return index


def choose_image_of_seg_id(image_paths: T.Iterable[Path]) -> Path:
    """Chooses the image of a seg_id that has several images (an unlabeled segment's)."""
    return min(image_paths)


def exclude_images(
    seg_id_to_image_paths: T.Mapping[str, list[Path]],
    excluded_folders: T.Collection[str],
    excluded_seg_ids: T.Collection[str],
) -> dict[str, list[Path]]:
    """Leaves out the images of the excluded folders and seg_ids.

    The build then treats them as if their files were missing.
    """
    kept = {seg_id: [path for path in paths if path.parent.name not in excluded_folders]
            for seg_id, paths in seg_id_to_image_paths.items()
            if seg_id not in excluded_seg_ids}
    return {seg_id: paths for seg_id, paths in kept.items() if paths}


def exclude_repeated_images(
    seg_id_to_image_paths: T.Mapping[str, list[Path]],
    root: Path,
) -> tuple[dict[str, list[Path]], dict[str, T.Any]]:
    """Leaves out images because of equal images (`repeated_images.find_images_to_exclude`),
    as `exclude_images`, and warns of them.

    Args:
        seg_id_to_image_paths: The images of each seg_id, relative to ``root``.

    Returns:
        The kept images of each seg_id, and a report of the left-out ones.
    """
    seg_id_to_image_path = {sid: root / choose_image_of_seg_id(paths)
                            for sid, paths in seg_id_to_image_paths.items()}
    excluded_folders, excluded_seg_ids = find_images_to_exclude(
        get_folder_to_seg_ids(seg_id_to_image_path), find_equal_file_groups(seg_id_to_image_path))
    kept = exclude_images(seg_id_to_image_paths, excluded_folders, excluded_seg_ids)
    folder_to_num_images = Counter(
        path.parent.name for paths in seg_id_to_image_paths.values() for path in paths)
    excluded_folder_to_num_images = {f: folder_to_num_images[f] for f in excluded_folders}
    num_excluded_images = folder_to_num_images.total() - sum(map(len, kept.values()))
    if num_excluded_images:
        print(f"WARNING: left out {num_excluded_images} images because of equal images "
              f"(see repeated_images.find_images_to_exclude):", file=sys.stderr)
        for folder, reason in excluded_folders.items():
            print(f"  folder {folder} ({excluded_folder_to_num_images[folder]} images): "
                  f"{reason}", file=sys.stderr)
        for sid, reason in excluded_seg_ids.items():
            print(f"  {sid}: {reason}", file=sys.stderr)
    return kept, {"num_images": num_excluded_images,
                  "folder_to_reason": excluded_folders,
                  "folder_to_num_images": excluded_folder_to_num_images,
                  "seg_id_to_reason": excluded_seg_ids}


def match_rows_to_images(
    df: pd.DataFrame,
    seg_id_to_image_paths: dict[str, list[Path]],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, T.Any]]:
    """Match each parquet row to an image by seg_id.

    Returns ``(kept_df, seg_id_to_relpath, stats)`` where ``stats`` carries
    per-row counters and example lists for the build report.

    Raises ValueError if more than one image shares a seg_id.
    """

    kept_indices: list[int] = []
    seg_to_path: dict[str, str] = {}
    no_image_examples: list[str] = []
    prefix_mismatch_examples: list[dict] = []
    num_no_image = 0
    num_prefix_mismatch = 0

    for idx, seg_id, section in zip(df.index, df["seg_id"].astype(str),
                                    df["section"].astype(str)):
        candidates = seg_id_to_image_paths.get(seg_id, [])
        if not candidates:
            num_no_image += 1
            if len(no_image_examples) < 20:
                no_image_examples.append(seg_id)
            continue
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple images found for segment ID {seg_id}: "
                f"{', '.join(str(c) for c in candidates)}"
            )
        chosen = candidates[0]

        prefix = chosen.parent.name
        if prefix != section:
            num_prefix_mismatch += 1
            if len(prefix_mismatch_examples) < 20:
                prefix_mismatch_examples.append({
                    "seg_id": seg_id,
                    "section_in_table": section,
                    "prefix_in_image": prefix,
                })

        seg_to_path[seg_id] = chosen.as_posix()
        kept_indices.append(idx)

    kept_df = df.loc[kept_indices].copy()
    stats = {
        "num_rows_no_image": num_no_image,
        "num_rows_prefix_mismatch": num_prefix_mismatch,
        "no_image_examples": no_image_examples,
        "prefix_mismatch_examples": prefix_mismatch_examples,
    }
    return kept_df, seg_to_path, stats


def build_segment_id_to_data_paths_rel(
    seg_to_path: T.Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Map ``seg_id -> {"rgb": "<rel/path/to/image.png>"}``."""
    return {sid: {"rgb": rel} for sid, rel in seg_to_path.items()}


def build_segment_id_to_road_data(
    df: pd.DataFrame, attribute_names: T.Sequence[str],
) -> dict[str, dict]:
    """Build ``segment_id -> {required_attributes, ..., comments, ...}``."""
    out: dict[str, dict] = {}
    for sid, section, distance, length, lat, lon, comments, attrs in zip(
        df["seg_id"], df["section"], df["distance"], df["length"],
        df["lat"], df["lon"], df["comments"],
        df[list(attribute_names)].to_dict("records"),
    ):
        out[sid] = {
            "section": section,
            "distance_km": float(distance),
            "length_km": float(length),
            "lat": float(lat),
            "lon": float(lon),
            "comments": comments,
            "required_attributes": {a: (None if pd.isna(v) else int(v))
                                    for a, v in attrs.items()},
        }
    return out


def build_road_sequences_and_validate(
    df: pd.DataFrame,
) -> tuple[dict[str, list[str]], list[dict], dict[str, int]]:
    """Group by section, sort by distance, validate adjacency, split on violation.

    The invariant enforced is

        distance_step_km  ==  abs(seg_id_step) * SEG_ID_STEP_TO_KM    (±1 m)

    i.e. each unit of seg_id corresponds to 10 m along the road. This catches
    coding-table errors where one row's seg_id was hyperlinked to a different
    recording session (huge seg_id jump, normal distance step), and also flags
    sections whose seg_id-step convention is inconsistent with the rest of the
    dataset.

    When the invariant fails between positions ``i-1`` and ``i``, the section's
    sequence is split there: the run ending at ``i-1`` is emitted as one
    sub-sequence, and a fresh run starts at ``i``. Sub-sequences are named
    ``"<section>"`` if there is only one for the section (i.e. the section was
    clean), or ``"<section>__part<N>"`` for ``N = 0, 1, 2, ...`` if it was
    split. Singleton sub-sequences are still emitted (the loader will skip
    them as context centres because no offset window fits).

    Returns:
        road_to_seq: mapping ``road_id -> [seg_id, ...]``.
        violations: one entry per failed transition (see code for keys).
        sections_split: ``{section: num_sub_sequences}`` for sections that
            produced more than one sub-sequence.
    """
    road_to_seq: dict[str, list[str]] = {}
    violations: list[dict] = []
    sections_split: dict[str, int] = {}

    def _emit(section: str, seg_ids: list[str], parts: list[tuple[int, int]]) -> None:
        if len(parts) == 1:
            lo, hi = parts[0]
            road_to_seq[section] = seg_ids[lo:hi]
            return
        sections_split[section] = len(parts)
        for n, (lo, hi) in enumerate(parts):
            road_to_seq[f"{section}__part{n}"] = seg_ids[lo:hi]

    for section, group in df.groupby("section", sort=True):
        sub = group.sort_values("distance", kind="stable").reset_index(drop=True)
        seg_ids = sub["seg_id"].astype(str).tolist()
        seg_ids_int = [int(s) for s in seg_ids]
        distances = sub["distance"].astype(float).tolist()
        source_files = sub["source_file"].astype(str).tolist()

        parts: list[tuple[int, int]] = []
        run_start = 0
        for i in range(1, len(seg_ids)):
            seg_step = seg_ids_int[i] - seg_ids_int[i - 1]
            dist_step = distances[i] - distances[i - 1]
            expected_dist_step = abs(seg_step) * SEG_ID_STEP_TO_KM
            ok = (
                seg_step != 0
                and abs(dist_step - expected_dist_step) <= ADJACENCY_TOLERANCE_KM
            )
            if ok:
                continue
            violations.append({
                "section": section,
                "prev_pos": i - 1,
                "next_pos": i,
                "prev_seg_id": seg_ids[i - 1],
                "next_seg_id": seg_ids[i],
                "prev_distance_km": float(distances[i - 1]),
                "next_distance_km": float(distances[i]),
                "distance_step_km": float(dist_step),
                "seg_id_step": int(seg_step),
                "expected_distance_step_km": float(expected_dist_step),
                "source_file_prev": source_files[i - 1],
                "source_file_next": source_files[i],
            })
            parts.append((run_start, i))
            run_start = i
        parts.append((run_start, len(seg_ids)))
        _emit(section, seg_ids, parts)

    return road_to_seq, violations, sections_split


def split_into_runs(sorted_ids: list[int], max_step: int) -> list[list[int]]:
    """Split non-empty sorted ids wherever consecutive ids are more than `max_step` apart."""
    runs = [[sorted_ids[0]]]
    for prev, curr in zip(sorted_ids, sorted_ids[1:]):
        if curr - prev > max_step:
            runs.append([])
        runs[-1].append(curr)
    return runs


def split_folders_into_recordings(seg_to_folder: dict[str, str]) -> dict[str, str]:
    """Map each seg_id to its recording: its image folder, or a run of it.

    seg_ids count up by 1 every 10 m within a recording and jump between
    recordings. An image folder usually holds one recording, but a folder whose
    video name was lost upstream (``FRAMES/nan/nan_seg*.png``) holds several,
    from different roads. Such a folder is split wherever consecutive seg_ids
    are more than ``MAX_SEG_ID_STEP_WITHIN_RECORDING`` apart, so that each
    recording is placed on the map and assigned to a split on its own.

    Returns:
        ``seg_id -> recording``, where the recording is the folder name for a
        folder with one run and ``"<folder>@<first seg_id of the run>"``
        otherwise.
    """
    folder_to_seg_ids: dict[str, list[int]] = defaultdict(list)
    for sid, folder in seg_to_folder.items():
        folder_to_seg_ids[folder].append(int(sid))
    seg_to_recording: dict[str, str] = {}
    for folder, seg_ids in folder_to_seg_ids.items():
        runs = split_into_runs(sorted(seg_ids), MAX_SEG_ID_STEP_WITHIN_RECORDING)
        for run in runs:
            recording = folder if len(runs) == 1 else f"{folder}@{run[0]}"
            seg_to_recording.update((str(sid), recording) for sid in run)
    return seg_to_recording


def interleave_unlabeled_into_road_sequences(
    road_to_seq: dict[str, list[str]],
    seg_to_recording: dict[str, str],
    recording_to_unlabeled_seg_ids: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Inject unlabeled siblings between adjacent labeled segments from the same recording.

    `build_road_sequences_and_validate` lists only labeled segments, so consecutive
    entries in a road sequence are typically 20 m apart (seg_id step 2). Cameras
    saved images every 10 m, so an unlabeled image usually sits between each pair.
    For the loader to use those as context frames (e.g. context_sequence=(0,-1,-2,…)),
    they must appear in the road sequence at the right position.

    We only inject between two labeled neighbours from the same recording (see
    `split_folders_into_recordings`), which is where the "1 seg_id = 10 m"
    invariant holds.

    Two road sequences span the same unlabeled seg_ids when both take labeled segments
    from the same stretch of a recording, e.g. two sections that code alternate frames,
    or two parts of a section split by the adjacency check. This is worth checking in the
    coding tables. Such a seg_id is injected into both, as it lies between labeled
    neighbours in each (see `irap_data.metadata.index_road_sequences` for its position as
    a context centre, and `find_shared_segments`).
    """
    recording_to_sorted_ids = {
        recording: sorted(int(s) for s in seg_ids)
        for recording, seg_ids in recording_to_unlabeled_seg_ids.items()
    }
    enriched: dict[str, list[str]] = {}
    for road_id, seq in road_to_seq.items():
        if not seq:
            enriched[road_id] = []
            continue
        out = [seq[0]]
        for prev, curr in zip(seq, seq[1:]):
            r_prev = seg_to_recording.get(prev)
            r_curr = seg_to_recording.get(curr)
            if r_prev is not None and r_prev == r_curr:
                p_int, c_int = int(prev), int(curr)
                lo, hi = min(p_int, c_int), max(p_int, c_int)
                ids = recording_to_sorted_ids.get(r_prev, [])
                in_between = ids[bisect.bisect_right(ids, lo):bisect.bisect_left(ids, hi)]
                if c_int < p_int:
                    in_between = in_between[::-1]
                out.extend(str(s) for s in in_between)
            out.append(curr)
        enriched[road_id] = out
    return enriched


def find_shared_segments(road_to_seq: dict[str, list[str]]) -> list[dict]:
    """Find the seg_ids that are in more than one road sequence.

    Returns:
        One entry per group of road sequences that share seg_ids, with their road IDs
        and the shared seg_ids.
    """
    seg_to_road_ids: dict[str, list[str]] = defaultdict(list)
    for road_id, seq in road_to_seq.items():
        for sid in seq:
            seg_to_road_ids[sid].append(road_id)
    road_ids_to_shared_seg_ids: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for sid, road_ids in seg_to_road_ids.items():
        if len(road_ids) > 1:
            road_ids_to_shared_seg_ids[tuple(road_ids)].append(sid)
    return [{"road_ids": list(road_ids), "seg_ids": seg_ids}
            for road_ids, seg_ids in road_ids_to_shared_seg_ids.items()]


def make_unlabeled_road_sequences(
    road_to_seq: dict[str, list[str]],
    recording_to_unlabeled_seg_ids: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Make road sequences of the unlabeled segments that are in none of `road_to_seq`.

    These are the unlabeled segments outside the labeled spans of their recording, and
    all segments of a recording without labeled segments (the unlocated ones). Each run
    of consecutive seg_ids (step 1, 10 m) of a recording becomes a road sequence, in
    seg_id order, so that the loader can take context frames for these segments. A run
    ends at a missing seg_id, which keeps the spacing in a road sequence uniform.

    Returns:
        ``"unlabeled:<recording>:<first seg_id>" -> [seg_id, ...]``.
    """
    seg_ids_in_roads = {sid for seq in road_to_seq.values() for sid in seq}
    unlabeled_road_to_seq: dict[str, list[str]] = {}
    for recording, seg_ids in recording_to_unlabeled_seg_ids.items():
        ids = sorted(int(s) for s in seg_ids if s not in seg_ids_in_roads)
        if not ids:
            continue
        for run in split_into_runs(ids, max_step=1):
            unlabeled_road_to_seq[f"unlabeled:{recording}:{run[0]}"] = [str(s) for s in run]
    return unlabeled_road_to_seq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="iRAP-Vietnam dataset root.")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    images = layout.images_dir(data_dir)
    attr_meta_in = layout.attr_meta_path(data_dir)
    rows_path = layout.rows_path(data_dir)
    if not images.is_dir():
        print(f"ERROR: {images} not found.", file=sys.stderr)
        return 1
    if not attr_meta_in.is_file():
        print(f"ERROR: {attr_meta_in} not found.", file=sys.stderr)
        return 1
    if not rows_path.is_file():
        print(f"ERROR: {rows_path} not found.", file=sys.stderr)
        return 1

    with open(attr_meta_in, "r", encoding="utf-8") as f:
        attr_meta = json.load(f)
    attribute_names = ordered_attribute_names(attr_meta)

    df = pd.read_parquet(rows_path)
    print(f"Loaded {len(df)} rows from {rows_path}.")

    # Some attributes may have been excluded during parsing (e.g. Vietnam
    # attributes with no ground truth). Only keep those present in the parquet.
    parquet_cols = set(df.columns)
    attribute_names = [a for a in attribute_names if a in parquet_cols]

    num_rows_in = len(df)
    print(f"Matching rows to images under {images} by seg_id...")
    seg_id_to_all_image_paths = index_images_by_seg_id(images)
    seg_id_to_image_paths, excluded_images_report = exclude_repeated_images(
        seg_id_to_all_image_paths, images.parent)
    is_excluded_row = df["seg_id"].astype(str).isin(
        seg_id_to_all_image_paths.keys() - seg_id_to_image_paths.keys())
    num_excluded_rows = int(is_excluded_row.sum())
    df = df[~is_excluded_row]
    print(f"  Dropped the {num_excluded_rows} rows of the segments whose images were left out.")
    num_images_total = len(seg_id_to_image_paths)
    df, seg_to_path, match_stats = match_rows_to_images(df, seg_id_to_image_paths)
    print(f"{len(df)}/{num_rows_in} rows matched to an image "
          f"({match_stats['num_rows_no_image']} dropped: no matching image).")
    if match_stats["num_rows_prefix_mismatch"]:
        print(f"    of which {match_stats['num_rows_prefix_mismatch']}/{len(df)} matched by seg_id "
              f"despite the coding-table Section disagreeing with the image folder name.")
    # Images with no matching parquet row (no labels).
    unlabeled_seg_ids = sorted(set(seg_id_to_image_paths.keys()) - set(seg_to_path.keys()), key=int)
    unlabeled_seg_to_path = {
        sid: choose_image_of_seg_id(seg_id_to_image_paths[sid]).as_posix()
        for sid in unlabeled_seg_ids
    }
    all_seg_to_path = {**seg_to_path, **unlabeled_seg_to_path}
    seg_to_folder = {sid: Path(rel).parent.name for sid, rel in all_seg_to_path.items()}
    seg_to_recording = split_folders_into_recordings(seg_to_folder)
    print(f"  Images under FRAMES: {num_images_total} total = "
          f"{len(seg_to_path)} labeled (matched to a parquet row) + "
          f"{len(unlabeled_seg_ids)} unlabeled (no parquet row).")
    folder_to_recordings: dict[str, set[str]] = defaultdict(set)
    for sid, recording in seg_to_recording.items():
        folder_to_recordings[seg_to_folder[sid]].add(recording)
    folders_split_into_recordings = {
        folder: sorted(recordings, key=lambda r: int(r.rsplit("@", 1)[1]))
        for folder, recordings in sorted(folder_to_recordings.items()) if len(recordings) > 1
    }
    if folders_split_into_recordings:
        print(f"  Split {len(folders_split_into_recordings)} image folder(s) holding several "
              f"recordings (seg_id steps > {MAX_SEG_ID_STEP_WITHIN_RECORDING}): "
              + ", ".join(f"{folder!r} into {len(recordings)}"
                          for folder, recordings in folders_split_into_recordings.items())
              + ".")

    # Group unlabeled seg_ids by recording, and compute a centroid for each
    # recording from labeled siblings in the same recording so they can be
    # placed on the map by the split editor.
    labeled_recording_to_seg_ids: dict[str, list[str]] = defaultdict(list)
    for sid in seg_to_path:
        labeled_recording_to_seg_ids[seg_to_recording[sid]].append(sid)
    labeled_seg_to_latlon = dict(zip(
        df["seg_id"].astype(str),
        zip(df["lat"].astype(float), df["lon"].astype(float)),
    ))
    unlabeled_recording_to_seg_ids: dict[str, list[str]] = defaultdict(list)
    for sid in unlabeled_seg_ids:
        unlabeled_recording_to_seg_ids[seg_to_recording[sid]].append(sid)

    unlabeled_sequence_id_to_data: dict[str, dict] = {}
    unplaceable_unlabeled_seg_ids: list[str] = []
    for recording, segs in unlabeled_recording_to_seg_ids.items():
        labeled_siblings = labeled_recording_to_seg_ids.get(recording, [])
        sibling_coords = [labeled_seg_to_latlon[s] for s in labeled_siblings]
        if not sibling_coords:
            unplaceable_unlabeled_seg_ids.extend(segs)
            continue
        lats = [c[0] for c in sibling_coords]
        lons = [c[1] for c in sibling_coords]
        unlabeled_sequence_id_to_data[recording] = {
            "segs": sorted(segs, key=int),
            "centroid": [float(np.median(lats)), float(np.median(lons))],
        }
    num_placeable_unlabeled = len(unlabeled_seg_ids) - len(unplaceable_unlabeled_seg_ids)
    print("  Unlabeled image partition (by recording):")
    print(f"    {num_placeable_unlabeled}/{len(unlabeled_seg_ids)} placeable on the map "
          f"(recording has ≥1 labeled sibling; centroid = median of sibling lat/lon; "
          f"selectable in split editor under unlabeled_train/val/test).")
    if unplaceable_unlabeled_seg_ids:
        print(f"    {len(unplaceable_unlabeled_seg_ids)}/{len(unlabeled_seg_ids)} unlocated "
              f"(recording has no labeled sibling, no map coordinate derivable; "
              f"auto-assigned to unlabeled_unlocated split).", file=sys.stderr)

    print("Building segment_id_to_data_paths_rel...")
    seg_to_paths = build_segment_id_to_data_paths_rel(all_seg_to_path)
    print("Building segment_id_to_road_data...")
    seg_to_road = build_segment_id_to_road_data(df, attribute_names)
    print("Building road_id_to_segment_id_sequence and validating adjacency...")
    road_to_seq, violations, sections_split = build_road_sequences_and_validate(df)
    num_labeled_in_sequences = sum(len(seq) for seq in road_to_seq.values())
    road_to_seq = interleave_unlabeled_into_road_sequences(
        road_to_seq, seg_to_recording, unlabeled_recording_to_seg_ids,
    )
    shared_unlabeled_segments = find_shared_segments(road_to_seq)
    shared_labeled_seg_ids = [sid for shared in shared_unlabeled_segments
                              for sid in shared["seg_ids"] if sid in seg_to_path]
    if shared_labeled_seg_ids:
        raise ValueError(f"{len(shared_labeled_seg_ids)} labeled seg_ids are in more than one "
                         f"road sequence, e.g. {shared_labeled_seg_ids[:10]}.")
    # Distinct seg_ids, as a shared unlabeled segment is in two road sequences.
    num_unlabeled_inserted_into_sequences = (
        len({sid for seq in road_to_seq.values() for sid in seq}) - num_labeled_in_sequences
    )
    print(f"  Interleaved {num_unlabeled_inserted_into_sequences}/{num_placeable_unlabeled} "
          f"placeable-unlabeled segments between adjacent labeled segments from the same "
          f"recording (the rest fall outside any labeled span).")
    if shared_unlabeled_segments:
        print(f"WARNING: {len(shared_unlabeled_segments)} group(s) of road sequences share "
              f"interleaved unlabeled segments, so the road sequences of a group take labeled "
              f"segments from the same stretch of a recording:", file=sys.stderr)
        for shared in shared_unlabeled_segments:
            print(f"  {' and '.join(map(repr, shared['road_ids']))}: seg_ids "
                  f"{', '.join(shared['seg_ids'])}", file=sys.stderr)
    unlabeled_road_to_seq = make_unlabeled_road_sequences(
        road_to_seq, unlabeled_recording_to_seg_ids)
    num_unlabeled_in_unlabeled_road_sequences = sum(
        len(seq) for seq in unlabeled_road_to_seq.values())
    print(f"  Added {len(unlabeled_road_to_seq)} unlabeled road sequences of the remaining "
          f"{num_unlabeled_in_unlabeled_road_sequences} unlabeled segments, including the "
          f"unlocated ones, one per run of consecutive seg_ids of a recording.")
    all_road_to_seq = {**road_to_seq, **unlabeled_road_to_seq}
    if len(all_road_to_seq) != len(road_to_seq) + len(unlabeled_road_to_seq):
        raise ValueError("A road ID of an unlabeled road sequence is also a section road ID.")
    # The unlabeled road sequences hold only seg_ids that are in no other road sequence.
    num_seg_ids_in_roads = len({sid for seq in all_road_to_seq.values() for sid in seq})
    if num_seg_ids_in_roads != len(all_seg_to_path):
        raise ValueError(f"{len(all_seg_to_path) - num_seg_ids_in_roads} of the "
                         f"{len(all_seg_to_path)} images are in no road sequence.")
    if violations:
        by_section: dict[str, list[dict]] = defaultdict(list)
        for v in violations:
            by_section[v["section"]].append(v)
        print(
            f"\nERROR: {len(violations)} segment-id / distance inconsistencies "
            f"in {len(by_section)} section(s).\n"
            f"  Fix these rows in the coding tables; for now the affected "
            f"sequences have been split so the loader will skip them as "
            f"context centres.\n",
            file=sys.stderr,
        )
        for section in sorted(by_section):
            entries = by_section[section]
            sources = sorted({e["source_file_prev"] for e in entries}
                             | {e["source_file_next"] for e in entries})
            print(f"section {section}  (source: {', '.join(repr(s) for s in sources)})",
                  file=sys.stderr)
            for v in entries:
                print(
                    f"  pos {v['prev_pos']}->{v['next_pos']}: "
                    f"seg {v['prev_seg_id']} -> {v['next_seg_id']}  "
                    f"(seg_id_step={v['seg_id_step']:+d}, "
                    f"distance_step={v['distance_step_km']:.3f} km, "
                    f"expected_step={v['expected_distance_step_km']:.3f} km)",
                    file=sys.stderr,
                )
            print(file=sys.stderr)

    def _dump(path: Path, obj: T.Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"Wrote {path}")

    _dump(layout.segment_id_to_data_paths_rel_path(data_dir), seg_to_paths)
    _dump(layout.segment_id_to_road_data_path(data_dir), seg_to_road)
    _dump(layout.road_id_to_segment_id_sequence_path(data_dir), all_road_to_seq)
    # Normalize iRAP codes to int when writing the metadata. The input JSON
    # encodes them as strings, but `required_attributes` in
    # segment_id_to_road_data.json uses ints (see build_segment_id_to_road_data).
    # `attribute_irap_number_to_class_idx` is dropped. Class indices are the value order of
    # `attribute_value_to_irap_number` (irap_data.ClassVocabulary, and the original irap_gaim
    # code, which deletes the field on loading), while this field numbers the classes in iRAP
    # code order. The two differ for the attributes whose values are not in code order
    # (both land-use sides and 'Pedestrian crossing - inspected road'), so decoding model
    # outputs with the field would give the wrong classes.
    attr_meta_normalized = {k: v for k, v in attr_meta.items()
                            if k != "attribute_irap_number_to_class_idx"}
    attr_meta_normalized["attribute_value_to_irap_number"] = {
        attr: {value: int(code) for value, code in mapping.items()}
        for attr, mapping in attr_meta["attribute_value_to_irap_number"].items()
    }
    _dump(layout.output_attr_meta_path(data_dir), attr_meta_normalized)
    _dump(layout.unlabeled_segment_ids_path(data_dir), unlabeled_seg_ids)
    _dump(layout.unlabeled_sequence_id_to_data_path(data_dir), unlabeled_sequence_id_to_data)

    unlocated_path = layout.unlabeled_unlocated_segment_ids_path(data_dir)
    if unplaceable_unlabeled_seg_ids:
        _dump(unlocated_path, sorted(unplaceable_unlabeled_seg_ids, key=int))
    elif unlocated_path.exists():
        unlocated_path.unlink()
        print(f"Removed stale {unlocated_path}")

    report = {
        "num_rows_in": num_rows_in,
        "num_rows_kept": len(df),
        "excluded_images": {**excluded_images_report, "num_rows": num_excluded_rows},
        **match_stats,
        "num_unlabeled_images": len(unlabeled_seg_ids),
        "unlabeled_examples": unlabeled_seg_ids[:20],
        "num_unlabeled_inserted_into_sequences": num_unlabeled_inserted_into_sequences,
        "shared_unlabeled_segments": shared_unlabeled_segments,
        "num_unlabeled_road_sequences": len(unlabeled_road_to_seq),
        "num_unlabeled_in_unlabeled_road_sequences": num_unlabeled_in_unlabeled_road_sequences,
        "num_unlabeled_sequences_placed": len(unlabeled_sequence_id_to_data),
        "folders_split_into_recordings": folders_split_into_recordings,
        "num_unplaceable_unlabeled_seg_ids": len(unplaceable_unlabeled_seg_ids),
        "unplaceable_unlabeled_examples": unplaceable_unlabeled_seg_ids[:20],
        "num_sections": df["section"].nunique() if not df.empty else 0,
        "num_road_sequences": len(road_to_seq),
        "section_lengths": {s: len(seq) for s, seq in road_to_seq.items()},
        "num_adjacency_violations": len(violations),
        "adjacency_violations": violations,
        "num_sections_split": len(sections_split),
        "sections_split": dict(sorted(sections_split.items())),
        "data_dir_name": data_dir.name,
    }
    _dump(layout.build_report_path(data_dir), report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
