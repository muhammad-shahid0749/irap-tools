# irap_data

Data access and inspection for iRAP road-attribute datasets iRAP-BH and iRAP-Vietnam, in which each road segment has an image and labels for a set of road attributes.

The code is based on code by Marin Kačan (see [docs/differences-to-the-original-dataset-code.md](docs/differences-to-the-original-dataset-code.md)).

## Installation

```bash
uv pip install -e packages/irap_data           # from an irap-tools checkout
uv pip install "irap-data[torch] @ git+https://github.com/Ivan1248/irap-tools#subdirectory=packages/irap_data"
```

Without extras, the package needs only NumPy, and `import irap_data` does not import PyTorch. The extras add the modules marked in the [package layout](#package-layout): `torch` for the PyTorch datasets, `viewer` for the dataset viewer (includes `torch`), and `report` for the plots of the statistics report (Matplotlib).

## Data layout

Set `IRAP_HOME` (or `DATASETS_PATH`) to a directory that contains:

- `IRAP_BIH/` (images) and `IRAP_BIH_METADATA/` (metadata) for iRAP-BH,
- `IRAP_Vietnam/` (images and metadata) for iRAP-Vietnam.

A metadata directory contains `splits.json`, `segment_id_to_data_paths_rel.json`, `segment_id_to_road_data.json`, `attribute_metadata.json` and `road_id_to_segment_id_sequence.json`. The iRAP-BH one also contains `seg_to_res/{train,val,test}.pickle` for the [N-context filter](#segment-selection). An iRAP-Vietnam one can also contain `unlabeled_segment_id_to_location.json`, the coordinates of unlabeled segments.

## Usage

```python
from irap_data import make_bh_data, make_vietnam_data

splits = make_bh_data(context_offsets=(0, -1, -4))  # directories from $IRAP_HOME
train = splits["train"]
print(len(train), train.info.class_counts)

example = train[0]
# {"rgb": (S, 3, H, W) float tensor in [0, 1], "target": (A,) LongTensor (-1 = unlabeled),
#  "segment_id": str, "sequence_id": str}

# Missing attributes become -1, and an empty test split is left out.
vn = make_vietnam_data(dataset_dir="/data/IRAP_Vietnam")
```

Labels and segment selection without images or PyTorch:

```python
from irap_data import load_irap_metadata, select_preset_split_segments

metadata = load_irap_metadata("/data/IRAP_Vietnam")
val = select_preset_split_segments(metadata, "vietnam", "val", context_offsets=(0, -1, -4))
print(len(val.segment_ids), metadata.vocabulary.class_counts)
```

A folder of unlabeled images, ordered by file name:

```python
from irap_data import InferenceImageDataset

ds = InferenceImageDataset.from_folder(
    "/path/to/images", reference_dataset=splits["test"], context_offsets=(0, -1, -4))
```

### Segment selection

`irap_data.metadata.select_split_segments` selects the segments of a split for the datasets and the statistics. It keeps a segment if it has an image, labels, and a complete context window: every context offset falls inside its road sequence and has an image.

- **BH N-context filter.** `make_bh_data()` also restricts each labeled split to the segments in its `seg_to_res/<split>.pickle` whose 10 preceding and 10 following segments on the road are in the same file (`load_ncontext_segment_ids`). The selected segments are thus the same for any context offsets with `max(abs(context_offsets)) ≤ 10`. `use_ncontext_filter=False` disables the filter. iRAP-Vietnam has no such filter.
- **Reference set.** The reference set of a split is the segments with a complete context window for the `reference_context_offsets` of the release in `DATASET_PRESETS` (`select_preset_split_segments` with `context_offsets=None`). All models whose context offsets lie within this window can be scored on it.
- **Unlabeled splits.** `splits.json` can contain `unlabeled_train`, `unlabeled_val` and `unlabeled_test` (unlabeled segments from the same sequences or areas as the labeled split) and `unlabeled_unlocated` (segments with an unknown relationship to the other splits). The datasets return them with all targets `-1`. The context-window check applies to them, the N-context filter does not.
- **Segments in no road sequence.** A segment that is in no road sequence has no context window. `select_split_segments` warns about such segments, as they point to incomplete metadata.

### Conventions

- Targets use `-1` (`irap_data.metadata.IGNORE_LABEL_INDEX`) for a missing label, as `torch.nn.CrossEntropyLoss(ignore_index=-1)` expects.
- Class indices follow the value order of `attribute_value_to_irap_number` in `attribute_metadata.json` (see `ClassVocabulary`).
- `Dataset.info` is a `LazyDict`: `info["class_counts"]` and `info.class_counts` are equivalent, and `Lazy(...)` entries are computed on first access.
- The datasets do not normalize images. `info.pixel_stats` holds the mean and standard deviation to normalize with, by default `RGB_MEAN` and `RGB_STD` in `irap_data.irap_dataset`. By default, `make_irap_data` fits images to `INPUT_DIM`, (width, height, channels) = (384, 288, 3), by resizing them to the width 384 and padding or cropping the height (`fit_mode="fit_width"`). All releases then keep their full horizontal field of view. iRAP-BH images have exactly this size and pass through unchanged, and 1920×1080 iRAP-Vietnam images become 384×216 with 36 rows of the mean color above and below. `image_utils.fit_to_size` lists the other modes.

## Dataset viewer

```bash
IRAP_HOME=/path/to/IRAP_HOME irap-dataset-viewer
IRAP_HOME=/path/to/IRAP_HOME irap-dataset-viewer --server.port 8502  # options of `streamlit run`
```

The viewer finds the datasets under `$IRAP_HOME`, filters segments by attribute value, and shows the road sequence around each segment.

## Dataset statistics report

`irap-dataset-stats` computes the class frequencies and other statistics of a release from its metadata, without images or PyTorch.

```bash
irap-dataset-stats report vietnam -o reports/vietnam
irap-dataset-stats report bh --metadata-dir /data/IRAP_BIH_METADATA -o reports/bh --context-offsets 0,-1,-4
irap-dataset-stats plot vietnam --split train --sort-attributes imbalance --sort-classes count -o vietnam_skew.pdf
```

`report` writes one report for each selection of segments (see `SELECTION_DESCRIPTIONS` in `irap_data.reports.statistics`):

| Selection | Segments | Report |
|---|---|---|
| `labeled` | With an image and labels | `<dataset>_labeled_segments` |
| `with_context` | With a complete context window for the context offsets of the release (the [reference set](#segment-selection)) | `<dataset>_labeled_segments_with_context` |
| `model` | With a complete context window for `--context-offsets`. Written only if it differs from `with_context` | `<dataset>_labeled_segments_with_context_<offsets>` |

Each report is an HTML page with a class-frequency plot per labeled split (`--plot-format`, `--no-plots`), a segment map `<report>_map.html` (`--no-map`) and a `<report>_class_frequencies.csv` table. It covers split sizes, label coverage, class balance, rare classes (`--rare-fraction`), missing classes and the class frequencies of each attribute. Classes are counted in segments and in road sequences, since segments on the same road are strongly correlated. `<dataset>_statistics.json` holds all statistics.

The segment map shows the segments of all splits, colored by split, and fades those that the selection leaves out. Unlabeled segments are placed by `IRAPMetadata.segment_coordinates`. The page loads Leaflet and its basemap from the web.

`plot` draws the class frequencies of one split in one figure, in the format given by the extension of `-o`. A class without segments is drawn as a short red bar.
