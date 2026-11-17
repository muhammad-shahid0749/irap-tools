"""Compare the files that build_metadata.py writes in two builds, e.g. before and after a change.

Usage:
    python compare_metadata.py <old_dir> <new_dir>

Each directory is a data root or a copy of its generated files. `build_report.json` is looked
up in the directory itself and then in its `_work/` subdirectory. `splits.json` is not compared,
as build_metadata.py does not write it.

For each file, the script prints whether it is identical, and otherwise what differs: the keys
only in one build and the changed keys of a mapping, or the items only in one build of a list.
For the files of segment groups (road sequences and the unlabeled sequences of the split
editor), it also prints where the segments of each removed or changed group went.
"""

import argparse
import json
import sys
import typing as T
from collections import defaultdict
from pathlib import Path

import layout

#: The files of build_metadata.py, and whether each maps a group ID to its segments.
FILE_TO_IS_GROUPING: dict[str, bool] = {
    layout.SEGMENT_ID_TO_DATA_PATHS_REL_FILENAME: False,
    layout.SEGMENT_ID_TO_ROAD_DATA_FILENAME: False,
    layout.ROAD_ID_TO_SEGMENT_ID_SEQUENCE_FILENAME: True,
    layout.ATTR_META_FILENAME: False,
    layout.UNLABELED_SEGMENT_IDS_FILENAME: False,
    layout.UNLABELED_SEQUENCE_ID_TO_DATA_FILENAME: True,
    layout.UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME: False,
    layout.BUILD_REPORT: False,
}

MAX_EXAMPLES = 10


def find_file(directory: Path, name: str) -> Path | None:
    for path in (directory / name, directory / layout.WORK_SUBDIR / name):
        if path.is_file():
            return path
    return None


def get_group_segments(group: T.Any) -> list[str]:
    """The segments of a group: a road sequence is a list, an unlabeled sequence a dict."""
    return [str(s) for s in (group["segs"] if isinstance(group, dict) else group)]


def format_examples(items: T.Iterable[T.Any]) -> str:
    items = list(items)
    shown = ", ".join(map(str, items[:MAX_EXAMPLES]))
    return shown + (f", ... ({len(items)} in all)" if len(items) > MAX_EXAMPLES else "")


def describe_mapping_difference(old: dict, new: dict) -> list[str]:
    lines = []
    if only_old := [k for k in old if k not in new]:
        lines.append(f"only in old ({len(only_old)}): {format_examples(only_old)}")
    if only_new := [k for k in new if k not in old]:
        lines.append(f"only in new ({len(only_new)}): {format_examples(only_new)}")
    if changed := [k for k in old if k in new and old[k] != new[k]]:
        lines.append(f"changed ({len(changed)}): {format_examples(changed)}")
        lines += [f"    {k}: {old[k]!r} -> {new[k]!r}" for k in changed[:MAX_EXAMPLES]
                  if not isinstance(old[k], (dict, list)) and not isinstance(new[k], (dict, list))]
    return lines


def describe_regrouping(old: dict, new: dict) -> list[str]:
    """Where the segments of each old group that is not in the new build unchanged went."""
    new_segment_to_group = {s: g for g, group in new.items() for s in get_group_segments(group)}
    lines = []
    for group_id, group in old.items():
        if group_id in new and new[group_id] == group:
            continue
        destinations: dict[str | None, int] = defaultdict(int)
        for s in get_group_segments(group):
            destinations[new_segment_to_group.get(s)] += 1
        lines.append(f"{group_id!r} ({len(get_group_segments(group))} segments) -> "
                     + ", ".join(f"{'no group' if g is None else repr(g)} ({n})"
                                 for g, n in destinations.items()))
    return lines


def describe_difference(old: T.Any, new: T.Any, is_grouping: bool) -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        lines = describe_mapping_difference(old, new)
        if is_grouping:
            lines += describe_regrouping(old, new)
        return lines
    if isinstance(old, list) and isinstance(new, list):
        old_set, new_set = set(map(str, old)), set(map(str, new))
        if old_set == new_set:
            return ["same items in a different order"]
        return [f"only in old ({len(old_set - new_set)}): {format_examples(sorted(old_set - new_set))}",
                f"only in new ({len(new_set - old_set)}): {format_examples(sorted(new_set - old_set))}"]
    return [f"old: {old!r}", f"new: {new!r}"]


def compare(old_dir: Path, new_dir: Path) -> int:
    """Prints the comparison and returns the number of files that differ."""
    num_different = 0
    for name, is_grouping in FILE_TO_IS_GROUPING.items():
        old_path, new_path = find_file(old_dir, name), find_file(new_dir, name)
        if old_path is None and new_path is None:
            print(f"{name}: in neither build")
            continue
        if old_path is None or new_path is None:
            print(f"{name}: only in the {'new' if old_path is None else 'old'} build")
            num_different += 1
            continue
        old, new = (json.loads(p.read_text(encoding="utf-8")) for p in (old_path, new_path))
        if old == new:
            print(f"{name}: identical")
            continue
        num_different += 1
        print(f"{name}: different")
        for line in describe_difference(old, new, is_grouping):
            print(f"    {line}")
    return num_different


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    args = parser.parse_args(argv)
    num_different = compare(args.old_dir, args.new_dir)
    print(f"\n{num_different} of {len(FILE_TO_IS_GROUPING)} files differ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
