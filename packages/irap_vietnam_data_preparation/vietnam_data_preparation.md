# IRAP-Vietnam data preparation

Version: 2026-05-07

End-to-end procedure for turning the raw IRAP-Vietnam release (images on Seafile
+ iRAP coding tables – per-segment attribute annotations – in
`coding-tables.zip`) into a dataset usable for training and evaluation through
`vidlu_irap_gaim/data/bih_dataset.py`.

The goal is to mirror the **IRAP-BiH** metadata layout closely enough that the
existing `BihSequence` dataset (or a thin sibling class) can load it without
substantial changes.

## Decisions

1. Dataset name: **`IRAP-Vietnam`** (data dir: `IRAP_VIETNAM`, metadata dir:
   `IRAP_VIETNAM_METADATA`).
2. `coding-tables.zip` is downloaded manually from Google Drive for now;
   automated download is out of scope.
3. **Invariant: `distance[i+1] − distance[i] == 0.01 km × (seg_id[i+1] − seg_id[i])`**
   within ±2.5 m, across consecutive labeled segments of a section sorted by
   distance. Each unit of seg_id corresponds to 10 m along the road, so a 20 m
   labeled step is seg_id step 2, a single skipped label is seg_id step 4,
   etc. `build_metadata.py` enforces this: violating transitions split the
   section into multiple `road_id`s (`<section>__part0`, `<section>__part1`,
   …) and are reported on stderr and in `_work/build_report.json` so the
   underlying coding table can be corrected. `BihSequence` resolves context by
   position within `road_id_to_segment_id_sequence.json`, so any well-formed
   sub-sequence is directly usable; no `BihSequence` variant needed.
4. The zip contains multiple XLS files; some have the data, some don't.
   Skip files missing the required columns; one sheet per file.
5. One section per file is expected; **warn** if a file mixes sections.
6. On duplicate `seg_id` rows, keep the row with **the most non-empty
   attribute cells**; warn on every duplicate.
7. Column order is **not guaranteed**; identify columns by header name.
   **Error** if any required column (or any expected attribute column)
   is missing.
8. Text labels per IRAP code are provided via the supplied `attribute_metadata.json` (see Decision 14).
9. Drop rows with any missing required attribute (BiH semantics).
10. `road_id` = section string, except for sections containing a continuity
    violation (Decision 3) which become `<section>__part0`, `<section>__part1`,
    …. `segment_id_to_road_data.json` keeps the original section string per
    segment, so splits keyed on the section string are unaffected.
11. Splits: per-section assignment; this is the last step.
12. RGB only.
13. Image filenames: `f"{section}_seg{seg_id}.png"`. Confirmed.
14. Attribute metadata: a pre-built `attribute_metadata.json` (with `attribute_to_idx` and `attribute_value_to_irap_number`) is supplied as input and copied verbatim into `IRAP_VIETNAM_METADATA/`. The pipeline validates that all Vietnam attribute columns and observed IRAP codes appear in it.
15. Code space: use the full code space from the supplied `attribute_metadata.json`, not restricted to codes observed in Vietnam. Class indices then match BiH for shared attributes, so models can be evaluated across datasets without index remapping.

### Simplifications vs. BiH

- Skip `seg_to_res/{train,val,test}.pickle` and the N-context filter; load
  with `use_ncontext_filter=False`.
- Store the section string directly as `road_id`; no separate id mapping.
- No `depth` modality.

---

## 1. Target dataset layout (BiH-compatible)

`bih_dataset.py:BihSequence` reads from two sibling directories:

```
<IRAP_HOME>/IRAP_VIETNAM/             # data root: images live here
<IRAP_HOME>/IRAP_VIETNAM_METADATA/    # JSON metadata (sibling of data root)
```

The metadata directory must contain the following files (names match BiH):

- **`splits.json`** – `{<split_name>: [seg_id, ...]}` with segment ids as
  strings. Labeled keys: `train`, `val`, `test`. Optional unlabeled keys
  (present when `unlabeled_sequence_id_to_data.json` exists): `unlabeled_train`,
  `unlabeled_val`, `unlabeled_test`. An additional `unlabeled_unlocated` key is
  auto-populated from `unlabeled_unlocated_segment_ids.json` (unlabeled segments
  from image folders with no labeled siblings, so no map coordinate is
  derivable). See the Stage 3c "Output format" section in
  [`README.md`](README.md) for details.
- **`segment_id_to_data_paths_rel.json`** – `{seg_id: {"rgb": "<rel/path.png>"}}`
  with paths relative to the data root. BiH also has `"depth"`; we will likely
  omit it (RGB-only release).
- **`segment_id_to_road_data.json`** – `{seg_id: {"required_attributes":
  {attr_name: irap_code, ...}, ...}}`. Only `required_attributes` is consumed
  by `BihSequence`; we can keep extra keys (e.g. lat/lon, distance).
- **`attribute_metadata.json`** – two maps:
  - `"attribute_to_idx"`: `{attr_name: int}` – defines the canonical attribute
    order (and therefore the order of class indices in the loaded `target`).
  - `"attribute_value_to_irap_number"`: `{attr_name: {value_label: irap_code}}`.
    `BihSequence` enumerates the keys of each inner dict to assign class
    indices (`enumerate(... .keys())`), so the **insertion order** of value
    labels defines the class index order. We will populate `value_label` with
    the human-readable IRAP label (see Q14 for the source of these labels).
- **`road_id_to_segment_id_sequence.json`** – `{road_id: [seg_id, seg_id, ...]}`,
  ordered along the road. Used for context-window construction (offsets like
  `(0, -1, -4)`) and for N-context filtering.
- (Optional) `seg_to_res/{train,val,test}.pickle` for N-context filtering. We
  can skip these initially and pass `use_ncontext_filter=False`.

`BihSequence` further needs context resolution to work with **integer
arithmetic on segment ids** (`int(sid) + offset`). The Vietnam segment ids
are pure integers and consecutive segments are 20 m apart with consecutive
ids (per Decision 3), so this works as-is. The table parser will verify
consecutiveness explicitly.

---

## 2. Stage 1 – Download (already implemented)

Module: `irap_vietnam_data_preparation/download_images.py`

- Walks a Seafile public share link via the `share-links` REST API.
- Downloads every file (recurses into subfolders).
- Resumes via HTTP `Range`; skips files where local size == remote size.
- CLI: `--share-url`, `--out`, `--password`, `--dry-run`.

Default share is the IRAP-Vietnam release, which contains 7 RAR archives
(`split1.rar` … `split7.rar`, ~110 GB total) plus an `.xlsx` index and
instructions. `coding-tables.zip` is downloaded **manually from Google
Drive** (Decision 2); we don't automate that.

---

## 3. Stage 2 – Extract images (already implemented)

Module: `irap_vietnam_data_preparation/extract_images.py`

- Extracts every `.rar` in `--in` into a single flat directory `--out`
  (default `<in>/images`).
- Tries `unrar` first, falls back to `7z`.
- One pass listing each archive (`unrar lt` / `7z l -slt`) recovers per-entry
  `(name, size, CRC32)`; this drives:
  1. A pre-extraction collision check across archives. With
     `--ignore-duplicates`, duplicates are classified into:
     - identical (same size+CRC),
     - ambiguous (same size, CRC missing),
     - differing content → printed as `WARN [content differs]` to stderr.
  2. A `tqdm` progress bar per archive driven by parsing extractor stdout.
- Resume: `unrar -o-` / `7z -aos` skip existing files; `--force` overwrites.

Result: `<out>/<segment_image_name>.png`, e.g.
`VID_20241211_135439_00_011_seg1814508.png`. Image basename encodes section
name + segment id.

---

## 4. Stage 3 – Build labels and metadata from `coding-tables.zip` (TODO)

This is the new work. The zip contains XLS(X?) files with one row per labeled
segment. Each table has many columns; we use a fixed subset:

| Source column                        | Use                                                          |
| ------------------------------------ | ------------------------------------------------------------ |
| Section                              | Section / road name (matches the prefix in image filenames). |
| Distance                             | Distance from the start of the section.                      |
| Length                               | Must equal **0.02** (= 20 m). Drop rows where it isn't.       |
| Latitude start, Longitude start      | For geographic split assignment.                             |
| Image Reference FPZ                  | Markdown-link cell; extract the trailing integer = `seg_id`. |
| Comments                             | Kept for traceability (free-text).                            |
| `Carriageway label` … *up to but not including* `Additional comments` | IRAP attribute values (numeric IRAP codes). Some columns may be missing codes. |
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
   non-empty cells in the attribute columns; warn on every duplicate
   (Decision 6).
8. Build a row record:
   `(section, distance, length, lat, lon, seg_id, comments, attr_values)`.

### Step 3.2 – Match rows to images

For each row, expected image filename:
`f"{section}_seg{seg_id}.png"` (assuming an underscore between section name
and `seg`, as in `VID_20241211_135439_00_011_seg1814508.png`).

- If the file is missing, drop the row and log it (count + first N examples).
- Build `segment_id_to_data_paths_rel.json` with `{seg_id: {"rgb":
  f"images/{filename}"}}` (the data root will be the directory **containing**
  `images/`).

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
  `attribute_value_to_irap_number[attr]`, matching `BihSequence`. Use the
  **full code space** from the supplied file (Decision 15) – codes that
  don't occur in Vietnam still get a class index, so the index space is
  shared with BiH.
- For each row, look up each attribute's IRAP code (numeric in the XLS
  cell) and confirm it appears in
  `attribute_value_to_irap_number[attr].values()`; warn-and-drop the row
  on unknown codes (matches Decision 9 / BiH semantics).
- The validated file is **copied into `IRAP_VIETNAM_METADATA/` verbatim**.

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

Rows where any required attribute is missing or `None`/blank get dropped
(matches BiH semantics in `bih_dataset.py:356-377`).

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

### Step 3.6 – Build `splits.json` (per-section)

Goal: train/val/test should not share geography. Per-section assignment is
sufficient (Decision 11): shuffle sections with a fixed seed, allocate to
splits by target ratio (e.g. 80/10/10). Done as the **last** step of the
pipeline so we can iterate on splits independently of the heavy parsing
work.

### Step 3.7 – Output layout

```
<IRAP_HOME>/IRAP_VIETNAM/
    images/
        VID_..._seg<seg_id>.png
        ...
<IRAP_HOME>/IRAP_VIETNAM_METADATA/
    splits.json
    segment_id_to_data_paths_rel.json
    segment_id_to_road_data.json
    attribute_metadata.json
    road_id_to_segment_id_sequence.json
```

The data root is `<IRAP_HOME>/IRAP_VIETNAM`; metadata sibling matches BiH's
`<root>_METADATA` convention used in `bih_dataset.py:308`.

---

## 5. Stage 4 – Wire up training/eval (TODO, after Stage 3)

Reuse `BihSequence` directly: the class derives the metadata dir from the
data dir name (`<root>_METADATA`), and integer-arithmetic context resolution
works because Vietnam segments are consecutive (Decision 3). Add a thin
factory `make_vietnam_data` mirroring `make_bih_data` but defaulting to
`use_ncontext_filter=False` and the Vietnam paths.

---

## 6. Suggested implementation order

All new modules go under
`irap_vietnam_data_preparation/`.

CLI: each step takes a single `--in <dir>` containing all inputs and a
single `--out <dir>` for outputs. Expected layout of `--in`:

```
<in>/
    coding-tables.zip            # OR an unzipped coding-tables/ directory
    attribute_metadata.json
    images/                      # flat directory from Stage 2 (only needed for step 2+)
```

`coding-tables.zip` is auto-unzipped (to a sibling `coding-tables/`) when
present; an existing `coding-tables/` directory takes precedence.

The `Section` cell of every row is taken verbatim and used both as
`road_id` and as the expected filename prefix in
`f"{section}_seg{seg_id}.png"`. Any mismatch between the `Section` cell
and the prefix of the matching image filename ⇒ error.

Steps:

1. **`parse_coding_tables.py`** – load the supplied attribute metadata,
   iterate `.xls`/`.xlsx` files, validate headers against the canonical
   attribute set (Decision 7 + 14), drop non-coding files, parse rows,
   resolve duplicates (Decision 6), check single-section per file
   (Decision 5). Emit a normalized per-row Parquet/CSV +
   `parse_report.json` (counts, warnings, dropped rows).
2. **`build_metadata.py`** – consume the parsed table + the `images/`
   directory; emit `segment_id_to_data_paths_rel.json`,
   `segment_id_to_road_data.json`, `road_id_to_segment_id_sequence.json`
   (with consecutiveness validation, §3.5), and copy the supplied
   `attribute_metadata.json` verbatim. No image reading required;
   idempotent.
3. **`split_editor.py`** – map GUI for assigning whole sequences to
   train/val/test by drawing boxes; writes `splits.json`. Run last so the
   splits can be redrawn without redoing parsing.
4. **Loader factory** – `make_vietnam_data` in
   `vidlu_irap_gaim/data/vietnam_dataset.py` (a thin wrapper over
   `BihSequence`). The class itself is reused unchanged.
5. **Sanity script** – load all three splits, print shapes of first
   batches, attribute class counts, split sizes, and a few example records.
