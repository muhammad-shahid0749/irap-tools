# iRAP-Vietnam data preparation

Version: 2026-10-07

End-to-end procedure for turning the raw iRAP-Vietnam release (images on Seafile
+ iRAP coding tables – per-segment attribute annotations – in
`coding-tables.zip`) into a dataset usable for training and evaluation through
`irap_data` (`make_vietnam_data`).

The goal is to mirror the **iRAP-BH** metadata layout closely enough that the
BH loader, `irap_data.IRAPDataset`, can load it without substantial changes.

## Decisions

1. Dataset name: **`iRAP-Vietnam`** (data and metadata dir: `IRAP_Vietnam`).
2. `coding-tables.zip` is downloaded manually from Google Drive for now;
   automated download is out of scope.
3. **Invariant: `distance[i+1] − distance[i] == 0.01 km × (seg_id[i+1] − seg_id[i])`**
   within ±2.5 m, across consecutive labeled segments of a section sorted by
   distance. Each unit of seg_id corresponds to 10 m along the road, so a 20 m
   labeled step is seg_id step 2, a single skipped label is seg_id step 4,
   etc. `build_metadata.py` enforces this: violating transitions split the
   section into multiple `road_id`s (`<section>__part0`, `<section>__part1`,
   …) and are reported on stderr and in `_work/build_report.json` so the
   underlying coding table can be corrected. `IRAPDataset` resolves context by
   position within `road_id_to_segment_id_sequence.json`, so any well-formed
   sub-sequence is directly usable; no loader variant needed.
4. The zip contains multiple XLS files; some have the data, some don't.
   Skip files missing the required columns; one sheet per file.
5. One section per file is expected; **warn** if a file mixes sections.
6. On duplicate `seg_id` rows, keep the row with **the most non-empty
   attribute cells**, and the first one in a tie. Report ties and
   conflicting codes as issues.
7. Column order is **not guaranteed**; identify columns by header name.
   **Error** if any required column (or any expected attribute column)
   is missing.
8. Text labels per iRAP code are provided via the supplied `attribute_metadata.json` (see Decision 14).
9. Drop rows with any missing required attribute (BH semantics).
10. `road_id` = section string, except for sections containing a continuity
    violation (Decision 3) which become `<section>__part0`, `<section>__part1`,
    …. `segment_id_to_road_data.json` keeps the original section string per
    segment, so splits keyed on the section string are unaffected.
11. Splits: per-section assignment; this is the last step.
12. RGB only.
13. Image filenames: `f"{section}_seg{seg_id}.png"`. Confirmed.
14. Attribute metadata: a pre-built `attribute_metadata.json` (with `attribute_to_idx` and `attribute_value_to_irap_number`) is supplied as input and copied into the metadata directory, with iRAP codes as ints and without `attribute_irap_number_to_class_idx`. That field numbers classes in iRAP-code order, which differs from the class indices (the value order of `attribute_value_to_irap_number`) for both land-use sides and 'Pedestrian crossing - inspected road', so it would decode model outputs wrongly. The pipeline validates that all Vietnam attribute columns and observed iRAP codes appear in it.
15. Code space: use the full code space from the supplied `attribute_metadata.json`, not restricted to codes observed in Vietnam. Class indices then match BH for shared attributes, so models can be evaluated across datasets without index remapping.

### Simplifications vs. BH

- Skip `seg_to_res/{train,val,test}.pickle` and the N-context filter; load
  with `use_ncontext_filter=False`.
- Store the section string directly as `road_id`; no separate id mapping.
- No `depth` modality.

---

## 1. Target dataset layout (BH-compatible)

BH keeps its JSON metadata in a sibling of the data root
(`IRAP_BIH_METADATA`), which is also the default of `IRAPDataset`.
`make_vietnam_data` instead reads the metadata from the data root itself:

```
<IRAP_HOME>/IRAP_Vietnam/    # data root: images and JSON metadata
```

The metadata directory must contain the following files (names match BH):

- **`splits.json`** – `{<split_name>: [seg_id, ...]}` with segment ids as
  strings. Labeled keys: `train`, `val`, `test`. Optional unlabeled keys
  (present when `unlabeled_sequence_id_to_data.json` exists): `unlabeled_train`,
  `unlabeled_val`, `unlabeled_test`. An additional `unlabeled_unlocated` key is
  auto-populated from `unlabeled_unlocated_segment_ids.json` (unlabeled segments
  from recordings with no labeled siblings, so no map coordinate is
  derivable). See "`splits.json` format" in
  [`README.md`](README.md) for details.
- **`segment_id_to_data_paths_rel.json`** – `{seg_id: {"rgb": "<rel/path.png>"}}`
  with paths relative to the data root. BH also has `"depth"`; we will likely
  omit it (RGB-only release).
- **`segment_id_to_road_data.json`** – `{seg_id: {"required_attributes":
  {attr_name: irap_code, ...}, ...}}`. The loader reads the labels from
  `required_attributes` (`irap_data.metadata.compute_segment_labels`) and the
  location from `section`, `distance_km`, `length_km`, `lat` and `lon`
  (`get_segment_location`). Other keys are kept but not read.
- **`attribute_metadata.json`** – two maps:
  - `"attribute_to_idx"`: `{attr_name: int}` – defines the canonical attribute
    order (and therefore the order of class indices in the loaded `target`).
  - `"attribute_value_to_irap_number"`: `{attr_name: {value_label: irap_code}}`.
    `irap_data.ClassVocabulary` numbers the classes in the order of each
    inner dict, so the **insertion order** of value labels defines the class
    index order. We will populate `value_label` with
    the human-readable iRAP label (see Q14 for the source of these labels).
- **`road_id_to_segment_id_sequence.json`** – `{road_id: [seg_id, seg_id, ...]}`,
  ordered along the road. Used for context-window construction (offsets like
  `(0, -1, -4)`) and for N-context filtering. Besides the section road
  sequences, it has unlabeled road sequences
  `unlabeled:<recording>:<first seg_id>` of unlabeled segments only (Step 3.5),
  so that every image is in a road sequence.
- (Optional) `seg_to_res/{train,val,test}.pickle` for N-context filtering. We
  can skip these initially and pass `use_ncontext_filter=False`.

`IRAPDataset` takes the context frames **by position in the road sequence**
(`irap_data.metadata.compute_segment_context_ids`), not by arithmetic on
segment ids. A window never crosses a road boundary, and a segment whose
window does not fit in its sequence is not a context centre. Because
`build_metadata.py` splits a section at every gap (Decision 3), a window never
spans a gap either.

---

## 2. Stage 1 – Download (already implemented)

Module: `irap_vietnam_data_preparation/download_images.py`

- Walks a Seafile public share link via the `share-links` REST API.
- Downloads every file (recurses into subfolders).
- Resumes via HTTP `Range`; skips files where local size == remote size.
- Writes to `<data_dir>/_raw/image_rars/`.

The share link is a required argument. The iRAP-Vietnam release contains 7 RAR archives
(`split1.rar` … `split7.rar`, ~110 GB total) plus an `.xlsx` index and
instructions. `coding-tables.zip` is downloaded **manually from Google
Drive** (Decision 2); we don't automate that.

---

## 3. Stage 2 – Extract images (already implemented)

Module: `irap_vietnam_data_preparation/extract_images.py`

- Extracts every `.rar` in `<data_dir>/_raw/image_rars/` into
  `<data_dir>/FRAMES/<video_dir>/`, without the `splitN/` directory that wraps
  the images in the archives. `FRAMES/` is emptied first.
- Tries `unrar` first, falls back to `7z`.
- Stops if two archives contain the same path, and writes all copies to
  `_work/FRAMES_duplicates/`. With `--ignore-duplicates`, it continues, keeps
  one copy of each path, and warns about the paths whose copies differ.
- Applies the patch archives last (name contains `missing_segments` or
  `new files`). Their images replace or add files.

Result: `FRAMES/<video_dir>/<video_dir>_seg<seg_id>.png`, e.g.
`FRAMES/VID_20241211_135439_00_011/VID_20241211_135439_00_011_seg1814508.png`.
The video directory is usually the section name.

---

## 4. Stage 3 – Build labels and metadata from `coding-tables.zip`

The zip contains XLS(X?) files with one row per labeled
segment. Each table has many columns; we use a fixed subset:

| Source column                        | Use                                                          |
| ------------------------------------ | ------------------------------------------------------------ |
| Section                              | Section / road name (matches the prefix in image filenames). |
| Distance                             | Distance from the start of the section.                      |
| Length                               | Must equal **0.02** (= 20 m). Drop rows where it isn't.       |
| Latitude start, Longitude start      | For geographic split assignment.                             |
| Image Reference FPZ                  | Markdown-link cell; extract the trailing integer = `seg_id`. |
| Comments                             | Kept for traceability (free-text).                            |
| `Carriageway label` … *up to but not including* `Additional comments` | iRAP attribute values (numeric iRAP codes). Some columns may be missing codes. |
| Additional comments                  | Ignored.                                                     |

### Step 3.1 – Collect labeled rows

For each `.xls`/`.xlsx` in the unzipped `coding-tables/` directory:

1. Read the (single) sheet via `openpyxl` / `pandas.read_excel`.
2. **Identify columns by header name**, not position. If any required column
   is missing (Section, Distance, Length, Latitude start, Longitude start,
   Image Reference FPZ, Comments, or any expected attribute column), **error
   out** with a clear message naming the file and missing columns.
3. **Skip files** whose headers don't include the labeling columns at all
   (= files in the zip that aren't coding tables).
4. Drop rows where any required field is missing or `Length != 0.02`.
5. Parse `Image Reference FPZ` (e.g. `[seg. no. 1814508](...)`) → `seg_id`
   via the trailing integer in the markdown link.
6. Verify the file's rows belong to a single `Section`; **warn** if mixed
   (Decision 5).
7. Resolve duplicate `seg_id` rows by keeping the row with the most
   non-empty cells in the attribute columns (Decision 6).
8. Build a row record:
   `(section, distance, length, lat, lon, seg_id, comments, attr_values)`.

### Step 3.2 – Match rows to images

Rows are matched to images by `seg_id`, the integer in the `_seg<seg_id>.png`
suffix of the images under `FRAMES/`.

- First, leave out images that do not show their segment, as if their files were
  missing, and drop the rows of their segments. The files in `FRAMES/` stay as
  extracted. Leaving out images, rather than setting the labels of their
  segments to −1, keeps the wrong images out of the context windows of other
  segments. A gap that a left-out segment leaves inside a section does not split
  its road sequence if the labeled segments around it are still continuous (as
  for a missing image), so the road sequence then has a 20 m step there. The
  images are those that `repeated_images.find_images_to_exclude` finds by equal
  files, printed as warnings and listed in the build report. They are found
  automatically, so that the rule, not a list, is the source of truth, and it
  applies to new images too. An image folder with a frozen frame at its start or
  end (at least `MIN_NUM_FROZEN_IMAGES` = 3 equal images) is left out whole: the
  GNSS coordinates used to sample its frames are time-shifted, so all its images
  are offset from the labels, whose coordinates are better aligned. Of the other
  equal images, the one with the lowest seg_id is kept: in a short run at the
  end of a folder the others repeat it, and elsewhere they were frames from the
  wrong time in the checked cases (2026-11-11). Near-copies of frames (e.g.
  1817093 of 1817061, and 1817124/1817131), which are not equal files, stay in:
  they are few, and finding them needs every image decoded.
- If no image matches, drop the row and log it (count + first N examples).
  Several images with the same `seg_id` are an error.
- If the video directory of the image differs from the row's `Section`, keep
  the row and report it (`prefix_mismatch` in `_work/build_report.json`). The
  two names are written independently.
- Build `segment_id_to_data_paths_rel.json` with `{seg_id: {"rgb":
  "FRAMES/<video_dir>/<filename>"}}`, relative to the data root.

### Step 3.3 – Attribute schema

`attribute_metadata.json` is **supplied as an input** alongside the coding
tables (Decision 14). We treat it as the canonical schema and only validate
the Vietnam data against it:

- **Canonical attribute set** = keys of the supplied
  `attribute_to_idx`. Each Vietnam table is checked to contain exactly the
  same set of attribute columns (any order). Missing columns ⇒ error
  (Decision 7); extra columns ⇒ error too (so we don't silently drop
  labels we should have kept).
- **Class indices** are assigned via insertion order of value labels in
  `attribute_value_to_irap_number[attr]`, matching `irap_data.ClassVocabulary`. Use the
  **full code space** from the supplied file (Decision 15) – codes that
  don't occur in Vietnam still get a class index, so the index space is
  shared with BH.
- For each row, look up each attribute's iRAP code (numeric in the XLS
  cell) and confirm it appears in
  `attribute_value_to_irap_number[attr].values()`. A cell with an unknown
  code, or with a non-integer value, becomes -1 like a blank cell, so that
  the row's other attributes are kept. The parser reports it as an issue.
- The validated file is **copied into the metadata directory**, normalized as
  in Decision 14.

### Step 3.4 – Build `segment_id_to_road_data.json`

For each kept segment:

```json
{
  "<seg_id>": {
    "section": "VID_20241211_135439_00_011",
    "distance_km": 1.234,
    "length_km": 0.02,
    "lat": 21.1234,
    "lon": 105.6789,
    "comments": "...",
    "required_attributes": {
      "Carriageway label": 1,
      "Upgrade cost": 3,
      ...
    }
  }
}
```

A blank or invalid attribute cell becomes -1, which the loader reads as a missing code
(`irap_data.metadata.compute_segment_labels`). With
`allow_missing_attributes=True`, the default of `make_vietnam_data`, a missing
code becomes the ignore label. Otherwise, as for BH, the segment is left out.
`parse_coding_tables.py --drop-rows-missing-attributes` drops such rows
already when parsing.

### Step 3.5 – Build `road_id_to_segment_id_sequence.json`

- Group rows by `Section`, sort each group's rows by `Distance` ascending.
- **Enforce the continuity invariant** (Decision 3): for every consecutive
  pair `(i-1, i)` in the sorted section,
  `abs(distance[i] − distance[i-1] − 0.01 × abs(seg_id[i] − seg_id[i-1])) ≤ 0.0025 km`.
  If the check fails, split the section at that point: the run ending at
  `i-1` becomes one sub-sequence, a fresh run starts at `i`. The road_id of
  a clean (un-split) section is the bare section string; a split section
  emits `<section>__part0`, `<section>__part1`, … in order.
- The pipeline **does not fail** on violations – it records every offending
  transition (`prev_seg_id`, `next_seg_id`, distances, `seg_id_step`,
  `expected_distance_step_km`, source XLS file) in `_work/build_report.json`
  and on stderr. These are typically data-entry errors in the coding tables
  (e.g. a row whose `Image Reference FPZ` hyperlinks to a seg_id from a
  different recording session); blocking the pipeline would make iteration
  on the rest of the dataset awkward. Loaders skip split-boundary segments
  as context centres automatically because no offset window fits.
- Insert each unlabeled segment that lies between two adjacent labeled
  segments of the same recording into that section road sequence. A segment
  that several road sequences span goes into each of them (reported as
  `shared_unlabeled_segments`).
- Add an unlabeled road sequence `unlabeled:<recording>:<first seg_id>` for
  each run of consecutive seg_ids of the remaining unlabeled segments of a
  recording.
  These include all unlocated segments.

### Step 3.6 – Build `splits.json` (per-section)

Goal: train/val/test should not share geography. Per-section assignment is
sufficient (Decision 11): `split_editor.py` assigns whole road sequences to
splits by drawing rectangles on a map. Done as the **last** step of the
pipeline so we can iterate on splits independently of the heavy parsing
work.

### Step 3.7 – Output layout

"Directory layout" in [`README.md`](README.md) gives the layout. The data root
is `<IRAP_HOME>/IRAP_Vietnam`, and the metadata files are in it (§1).

---

## 5. Stage 4 – Loading

`irap_data.make_vietnam_data` builds the splits with `make_irap_data` and
`IRAPDataset`, like `make_bh_data` does for BH. Its defaults come from
`irap_data.DATASET_PRESETS["vietnam"]`: no N-context filter
(`use_ncontext_filter=False`) and `allow_missing_attributes=True`. It reads the
metadata from the data root and leaves out an empty `test` split.

---

## 6. Suggested implementation order

All new modules go under
`irap_vietnam_data_preparation/`.

CLI: each step takes the dataset root `<data_dir>` as its only positional
argument, and reads and writes the paths of `layout.py` ("Directory layout"
in [`README.md`](README.md)). The inputs are `_raw/coding-tables.zip`,
`_raw/attribute_metadata.json` and, from step 2 on, the images in `FRAMES/`.
`parse_coding_tables.py` extracts the zip into `_work/coding-tables/` again
on every run.

The `Section` cell of every row is taken verbatim as the `road_id` (Step
3.5). A mismatch with the video directory of the matched image is reported,
not an error (Step 3.2).

Steps:

1. **`parse_coding_tables.py`** – load the supplied attribute metadata,
   iterate `.xls`/`.xlsx` files, validate headers against the canonical
   attribute set (Decision 7 + 14), drop non-coding files, parse rows,
   resolve duplicates (Decision 6), check single-section per file
   (Decision 5). Emit a normalized per-row Parquet/CSV +
   `parse_report.json` (counts, warnings, dropped rows).
2. **`build_metadata.py`** – consume the parsed table + the `FRAMES/`
   directory; emit `segment_id_to_data_paths_rel.json`,
   `segment_id_to_road_data.json`, `road_id_to_segment_id_sequence.json`
   (with consecutiveness validation, §3.5), and copy the supplied
   `attribute_metadata.json` (see Decision 14). No image reading required;
   idempotent.
3. **`split_editor.py`** – map GUI for assigning whole sequences to
   train/val/test by drawing boxes; writes `splits.json`. Run last so the
   splits can be redrawn without redoing parsing.
4. **Loader factory** – `make_vietnam_data` in
   `irap_data/irap_dataset.py` (a thin wrapper over `make_irap_data`). The
   dataset class, `IRAPDataset`, is shared with BH.
5. **Sanity script** – load all three splits, print shapes of first
   batches, attribute class counts, split sizes, and a few example records.
