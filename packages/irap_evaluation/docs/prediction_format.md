# iRAP prediction file format

A prediction file is an [Apache Parquet](https://parquet.apache.org/) file holding the predictions of a model for a split of an iRAP dataset. Files are conventionally named `<method>.<split>.predictions.parquet`, or `<method>_seed<seed>.<split>.predictions.parquet` for one of several models of a method. `make_prediction_file_name` makes such a name: it replaces characters that are not valid in a file name (`to_valid_file_name`), and it cuts a method name that would make the name longer than 255 bytes and adds a hash of it (`shorten_file_name`). The full method name is in the header.

Prediction file structure:
```
Parquet file
  key-value metadata:
    "irap_predictions": JSON header
      format_version: 1
      dataset: Literal["bh", "vietnam"]
      split: str
      context_offsets?: list[int] | None
      model:
        method_name: str
        method_display_name?: str | None
        training_splits: list[str] | None
        early_stopping_splits: list[str]
        seed?: int | None
        details?: dict | None
      attribute_to_irap_codes: dict[str, list[int]]
  columns:
    segment_id: string
    <attribute>: list<float32> or int64, nullable
    ... # one column per attribute
```

Header types are Python types of the parsed JSON. `?` marks optional fields. Column types are Arrow types.

`irap_evaluation.prediction_io` reads and writes the format (`read_predictions`, `write_predictions`, and `read_prediction_header` for the header alone) and names the files (`make_prediction_file_name`), represented in memory by `irap_evaluation.predictions.Predictions`.

## Columns

| Column | Type | Content |
|---|---|---|
| `segment_id` | `string \| large_string` | Segment identifier. |
| `<attribute>` | `list<float32>` or `int64`, nullable | Probability vector or class index, `null` for an [invalid attribute prediction](evaluation.md#invalid-attribute-predictions). |

There is one `<attribute>` column for each key of `attribute_to_irap_codes`, named as in `attribute_metadata.json`.
All attribute columns share the same output kind:

- **`probs`**: a predicted class distribution, a list with one value per class of the attribute. Values are $\ge 0$, and sum to 1 within $10^{-4}$. Writers should use `list<float32>` or `large_list<float32>` (e.g. from Polars). `float64` values are acceptable too.
- **`hard`**: the 0-based index of the predicted class, as `int64` or another integer type.

## Header

The header is a UTF-8 JSON object (without `NaN` or infinity) stored under the key `irap_predictions` in the file-level key-value metadata of the Parquet file.

In PyArrow, the file-level metadata is `pyarrow.parquet.read_metadata(path).metadata`. The schema metadata of a table read with `pyarrow.parquet.read_table` lacks the key in files written by some libraries, such as Polars.
Pandas cannot write key-value metadata. Pandas, Polars, and DuckDB drop it when reading and rewriting a file.

| Field | Type | Content |
|---|---|---|
| `format_version` | `int` | `1`. |
| `dataset` | `str` | Dataset identifier: `"bh"` or `"vietnam"`. |
| `split` | `str` | Segment split, e.g. `"val"`. |
| `context_offsets` | `list[int] \| None` | Relative positions of sequence segments read by the model, e.g. `[0, -1, -4]`. Absent or `null` if unknown. Optional. |
| `model` | `dict` | Information about the model instance (see [Model field](#model-field)). |
| `attribute_to_irap_codes` | `dict[str, list[int]]` | Attribute column name → iRAP codes defining class order for `probs` lists and `hard` class indices. |

Along with `dataset` and `split`, `context_offsets` defines the model's [evaluation set](evaluation.md#evaluation-sets).

### Model field

| Field | Type | Content |
|---|---|---|
| `method_name` | `str` | A string that identifies the method. Files with the same method name are predictions of the same learning method. |
| `method_display_name` | `str \| None` | Human-readable short name of the method, shown instead of `method_name`, e.g. `"ViT-L-DINOv3-supervised"`. Optional. |
| `training_splits` | `list[str] \| None` | Splits used to fit the model, `[]` for none (e.g. a zero-shot model), `null` if unknown. Required. |
| `early_stopping_splits` | `list[str]` | Splits used only for checkpoint selection, `[]` for none. Required. |
| `seed` | `int \| None` | Seed representing a training run. Optional. |
| `details` | `dict \| None` | Additional free-form JSON metadata about how the predictions were made, e.g. the code commit under `commit`, pre-training information, etc.. Optional. |

A model is identified by its dataset, method name and seed. All its files must have the same training and early stopping splits. The models of a method must have the same training and early stopping splits. A model trained on other splits, e.g. on `val` in addition to `train`, should have a different method name, e.g. `resnet-seq-trainval`.

#### Details field of an ensemble

An ensemble written by `irap_evaluation.ensemble_predictions` is a `probs` file whose `details["ensemble"]` has:

| Field | Type | Content |
|---|---|---|
| `segment_combination`, `attribute_combination` | `str` | Which segments and attributes of the members it combines: `same`, `intersection` or `union` (`MemberCombination`). |
| `members` | `list[dict]` | Per member: `model` (its [model field](#model-field)), `context_offsets`, `weight`, `attributes` and `num_segments` (the number of segments it predicts). |

## Examples

### Writing and reading with `irap_evaluation`

```python
import numpy as np
import irap_evaluation as ie

# Write predictions
header = ie.PredictionHeader(
  dataset="vietnam",
  split="val",
  context_offsets=(0, -1, -4),
  model=ie.ModelInfo(method_name="resnet-seq", training_splits=("train",),
                     early_stopping_splits=(), seed=3,
                     method_display_name="ResNet-seq", details={"commit": "4f2a9c1"})
)
predictions = ie.Predictions.from_logits(
  header,
  attribute_to_irap_codes = {"Lane width": (1, 2, 3)},
  segment_ids = ["<segment ID 1>", "<segment ID 2>"],
  logits = {"Lane width": np.array([[2.0, 0.1, -1.0], [0.3, 1.5, 0.2]])}
)
ie.write_predictions("resnet-seq_seed3.val.predictions.parquet", predictions)

# Read predictions
assert predictions == ie.read_predictions("resnet-seq_seed3.val.predictions.parquet")
```

For hard predictions (e.g. VLM answers), use `ie.Predictions.from_class_indices` with class indices or `ie.Predictions.from_irap_codes` with iRAP codes, using `irap_data.IGNORE_LABEL_INDEX` or `ie.INVALID_IRAP_CODE` (both −1) for missing answers.

### Writing without `irap_evaluation`

```python
import json
import polars as pl

header = dict(
  format_version=1,
  dataset="vietnam",
  split="val",
  model=dict(method_name="resnet-seq", training_splits=["train"], early_stopping_splits=[],
             seed=3, method_display_name="ResNet-seq", details={"commit": "4f2a9c1"}),
  context_offsets=[0, -1, -4],
  attribute_to_irap_codes={"Lane width": [1, 2, 3]}
)

# Write predictions with Polars
predictions = pl.DataFrame(
  {
    "segment_id": ["<segment ID 1>", "<segment ID 2>"],
    "Lane width": [[0.85, 0.12, 0.03], [0.20, 0.70, 0.10]]
  },
  schema={"segment_id": pl.String, "Lane width": pl.List(pl.Float32)}
)
metadata = {"irap_predictions": json.dumps(header)}
predictions.write_parquet("resnet-seq_seed3.val.predictions.parquet", metadata=metadata)
```

For hard predictions, attribute columns are `pl.Int64` class indices, with `None` for invalid attribute predictions.
