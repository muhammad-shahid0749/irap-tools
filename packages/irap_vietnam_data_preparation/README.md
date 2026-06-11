# IRAP-Vietnam data preparation

End-to-end pipeline for turning the raw images and Excel tables into an IRAP-BH-compatible dataset.

For the rationale and decisions behind each step see [`vietnam_data_preparation.md`](vietnam_data_preparation.md).

## Prerequisites

- `unrar` (preferred) or `7z` on `PATH` for image extraction.
- Python packages: `uv pip install pandas openpyxl xlrd pyarrow tqdm numpy scikit-learn`. For the Streamlit-based manual split editor, also install `streamlit>=1.35 plotly`.
- The `IRAP_Vietnam/` dataset root, with `_raw/` populated (see "Layout" below).

## Layout

Every per-stage script takes a single positional `<data_dir>` argument – the path to
the dataset root – and derives all subpaths internally.

```
<data_dir>/                      # IRAP_Vietnam dataset root
  _raw/                          # populated manually + Stage 1
    coding-tables.zip            # (or unzipped coding-tables/ directory)
    attribute_metadata.json
    image_rars/                  # Stage 1 output (downloaded RAR archives)
  
  _work/                         # intermediate outputs, can be deleted after
    rows.parquet
    parse_report.json
    build_report.json
  
  images/                        # Stage 2 output (nested by video sequence)
    <video_dir>/
      <video_dir>_seg<N>.png
  segment_id_to_data_paths_rel.json  # Stage 3b outputs (directly in root)
  segment_id_to_road_data.json
  road_id_to_segment_id_sequence.json
  attribute_metadata.json        # copy of _raw/attribute_metadata.json
  splits.json                    # Stage 3c
```

`coding-tables.zip` contains the iRAP coding tables – Excel spreadsheets of
per-segment attribute annotations, where "coding" is iRAP's term for assigning
attribute codes to 20-m road segments. Before running anything, populate
`<data_dir>/_raw/` with at least `coding-tables.zip` (or an unzipped
`coding-tables/` directory) and `attribute_metadata.json`. Stage 1 fills
`<data_dir>/_raw/image_rars/`.

`make_vietnam_data()` searches for `IRAP_Vietnam/` in this order: `$IRAP_HOME`,
`$VIDLU_DATASETS`, `$VIDLU_DATA/datasets`, then ancestors of the package for
`data/datasets/IRAP_Vietnam`.

## End-to-end

```bash
bash prepare_dataset.sh <data_dir>
```

Skips downloading or extraction with `--skip-download` / `--skip-extract` if
already done. See `prepare_dataset.sh --help` for split-ratio / seed options.

## Per-stage commands

Set `DATA_DIR=/path/to/IRAP_Vietnam`.

### Stage 1 – Download images from Seafile

```bash
python download_images.py $DATA_DIR
```

Files are resumable (HTTP Range). Existing files with matching size are skipped. Total ≈ 110 GB across 7 `split*.rar` archives plus a small index.

### Stage 2 – Extract images grouped by source video

```bash
python extract_images.py $DATA_DIR
```

- Extracts with `unrar x` / `7z x`. The outer `splitN/` wrapper is stripped, leaving `images/<video_dir>/<video_dir>_seg<N>.png`.
- Checks for collisions across the `split*.rar` archives. Aborts on duplicates unless `--ignore-duplicates` is passed. All duplicate copies are extracted to `_work/images_duplicates/` for manual review.
- If `images/` exists, prompts to wipe before extraction (use `--yes` to wipe without prompting).
- `missing_segments*.rar` archives (corrected segments) are applied last: each `<video_dir>_seg<N>.png` in them replaces or adds `images/<video_dir>/<file>`. They are excluded from the collision check, and replaced vs. added counts are reported.
- No incremental mode: re-running wipes `images/`, re-extracts everything, then re-applies the missing_segments archives.

### Stage 3a – Parse coding tables

```bash
python parse_coding_tables.py $DATA_DIR
```

Auto-unzips `_raw/coding-tables.zip` if needed. Validates required column names
against `_raw/attribute_metadata.json` (errors on missing columns). Drops rows
with `Length != 0.02 km`, missing scalar/attribute fields, unparseable
`Image Reference FPZ`, unknown IRAP codes, or – when the optional
`offset_distance_m` column is present – an FPZ-to-annotation distance greater
than 10 m. Resolves duplicate `seg_id`s across files by keeping the row with
the most non-empty attribute cells.

### Stage 3b – Build BiH-compatible metadata

```bash
python build_metadata.py $DATA_DIR
```

Matches each parquet row to an image **by seg_id**, recursing into
`<data_dir>/images/<video_dir>/`. Rows with no image are dropped. Mismatches
between the coding-table `Section` cell and the image's `<video_dir>` name are
recorded as `prefix_mismatch` (not dropped). Validates the section adjacency
invariant (distance step ≈ 0.02 km between consecutive segments). The summary
is written to `_work/build_report.json`.

### Stage 3c – Assign train/val/test splits

#### Output format

Both tools write `splits.json` as `{<split_name>: [seg_id, ...]}` with segment ids as strings. Keys:

- `train`, `val`, `test` – labeled segments assigned to each split.
- `unlabeled_train`, `unlabeled_val`, `unlabeled_test` – unlabeled segments assigned the same way. Omitted if `unlabeled_sequence_id_to_data.json` is not present in the metadata directory.
- `unlabeled_unlocated` – unlabeled segments from image folders that have no labeled siblings, so no map coordinate is derivable. Auto-populated from `unlabeled_unlocated_segment_ids.json` and not user-editable in the map GUI. Omitted if the file is absent.

#### Automatic (K-Means)

```bash
python make_splits.py $DATA_DIR
```

Per-section deterministic allocation: whole sections go into one split,
giving geographically non-overlapping splits with no leakage.

#### Manual (map GUI)

```bash
streamlit run split_editor.py -- $DATA_DIR
```

Opens a browser-based map showing all road sections as coloured polylines.

1. Pick an active split in the sidebar (`train`, `val`, `test`, or `none`).
2. **Draw a rectangle** on the map – all sections whose centroid falls
   inside are assigned to the active split.
3. Use **Undo** to revert the last batch, **Reset** to clear all.
4. **Save** writes `splits.json` (see *Output format* above).

If `splits.json` already exists in the metadata directory it is used as the
starting assignment (useful for tweaking an automatic result).
Requires `streamlit >= 1.35`.

## Stage 4 – clean up

After verifying the output, you can delete the `_raw/` and `_work/` directories.