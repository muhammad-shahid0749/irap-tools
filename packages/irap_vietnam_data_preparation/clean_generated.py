"""Delete all pipeline outputs generated after the extraction step.

Removes the files produced by ``parse_coding_tables.py``,
``build_metadata.py`` and ``split_editor.py``, leaving the inputs to the
extraction step (``_raw/``) and its outputs (``FRAMES/`` and
``_work/FRAMES_duplicates/``) intact.

Usage:
    python irap_vietnam_data_preparation/clean_generated.py <data_dir>
                                                            [--dry-run] [--yes]
"""

import argparse
import sys
from pathlib import Path

import layout


def _targets(data_dir: Path) -> list[Path]:
    metadata = layout.metadata_dir(data_dir)
    work = layout.work_dir(data_dir)
    return [
        # parse_coding_tables.py
        layout.rows_path(data_dir),
        layout.parse_report_path(data_dir),
        # build_metadata.py
        layout.build_report_path(data_dir),
        metadata / "segment_id_to_data_paths_rel.json",
        metadata / "segment_id_to_road_data.json",
        metadata / "road_id_to_segment_id_sequence.json",
        metadata / layout.ATTR_META_FILENAME,
        layout.unlabeled_segment_ids_path(data_dir),
        layout.unlabeled_sequence_id_to_data_path(data_dir),
        layout.unlabeled_unlocated_segment_ids_path(data_dir),
        # split_editor.py
        metadata / "splits.json",
        # _work/ itself if it ends up empty (FRAMES_duplicates may keep it alive)
        work,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="IRAP_Vietnam dataset root.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be removed without deleting.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation prompt.")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} is not a directory.", file=sys.stderr)
        return 1

    candidates = _targets(data_dir)
    existing_files = [p for p in candidates if p.is_file()]
    work = layout.work_dir(data_dir)
    work_empty_after = (
        work.is_dir()
        and {c.name for c in work.iterdir()} <= {p.name for p in existing_files
                                                 if p.parent == work}
    )

    if not existing_files and not (work.is_dir() and work_empty_after):
        print("Nothing to remove.")
        return 0

    print("The following will be removed:")
    for p in existing_files:
        print(f"  {p}")
    if work.is_dir() and work_empty_after:
        print(f"  {work}/  (empty after cleanup)")

    if args.dry_run:
        return 0

    if not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    for p in existing_files:
        p.unlink()
        print(f"Removed {p}")

    if work.is_dir():
        try:
            work.rmdir()
            print(f"Removed empty {work}")
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
