# iRAP evaluation

Evaluation of saved iRAP model predictions, as a library with a thin command-line tool, `irap-eval`:

- a [prediction file format](docs/prediction_format.md) for class distributions or hard predictions, with classes identified by iRAP code,
- the [iRAP metrics](docs/evaluation.md), equal to those of the training-time evaluation in `vidlu_irap_gaim`,
- the [manual review proportion](docs/evaluation.md#manual-review-proportion): how much of an attribute a reviewer has to code, starting with the least confident predictions, for its macro F1 to reach 0.9,
- bootstrap confidence intervals and method comparisons that resample road sequences and include the spread of a method's models (e.g. training seeds),
- ensembling, CSV and markdown tables, and export as iRAP coding tables.

It needs numpy, pandas, pyarrow and the torch-free core of [`irap_data`](../irap_data/), not torch.

## Installation

From the irap-tools checkout:

```bash
uv pip install -e packages/irap_data -e "packages/irap_evaluation[xlsx]"
```

From another project:

```bash
uv pip install "irap-data @ git+https://github.com/Ivan1248/irap-tools#subdirectory=packages/irap_data" \
               "irap-evaluation[xlsx] @ git+https://github.com/Ivan1248/irap-tools#subdirectory=packages/irap_evaluation"
```

The `xlsx` extra (xlsxwriter) is needed only for `.xlsx` coding tables.

## Command-line use

```bash
DATA=/data/IRAP_Vietnam   # the metadata directory of the release, e.g. IRAP_BIH_METADATA for BH

# Check files.
irap-eval validate resnet.val.predictions.parquet vit.val.predictions.parquet

# Ensemble two models (the weighted mean of their distributions). By default, they must predict the
# same segments and attributes; --segment-combination and --attribute-combination can choose their
# intersection or union.
irap-eval ensemble resnet.val.predictions.parquet vit.val.predictions.parquet \
    --name resnet+vit -o resnet+vit.val.predictions.parquet

# Score files, with 95 % bootstrap intervals from 1000 resamples.
irap-eval evaluate $DATA resnet.val.predictions.parquet vit.val.predictions.parquet \
    resnet+vit.val.predictions.parquet --out results/ --bootstrap 1000 --table-metrics mF1 MCC

# Compare two methods on the reference set. Each method has several models: files with the same
# model.method_name in the header and distinct model.seed values.
irap-eval compare $DATA resnet_seed*.val.predictions.parquet vit_seed*.val.predictions.parquet \
    --a resnet --b vit

# Write a model's predictions on two splits as one iRAP coding table, with the probability of each
# code on a second sheet.
irap-eval export-coding-table $DATA resnet+vit.val.predictions.parquet \
    resnet+vit.test.predictions.parquet -o resnet+vit.xlsx --confidence-table
```

Run `irap-eval <command> --help` for all options.

### Outputs of `evaluate`

`evaluate` scores each file on up to three [evaluation sets](docs/evaluation.md#evaluation-sets): `reference`, the same for all models, `full`, all segments with an image, for the models that predict all of them, and `compatible`, which matches the model's context offsets. Nulls in a file are scored according to `--null-policy` ([invalid attribute predictions](docs/evaluation.md#invalid-attribute-predictions)). It also takes the files of a model on several splits, which must agree in the model's training and early stopping splits.

Models are labeled `<method>` or `<method>/seed<seed>`. Every output records `split_use`: whether the model was trained on the scored split, used it for early stopping, may have been trained on it, or held it out ([terms](docs/evaluation.md#terms)). `evaluate` writes to `--out`:

| File | Content |
|---|---|
| `summary_long.csv` | One row per model, evaluation set, attribute (or `average`) and metric, with the method, seed, `split_use`, null policy and confidence bounds. `ci_num_undefined` counts the undefined bootstrap resamples, which the bounds leave out. |
| `averages_<dataset>_<split>_<set>.csv` | Model rows × `split_use` and attribute-average columns. |
| `method_averages_<dataset>_<split>_<set>.csv` | Only with `--null-policy first_class` and methods with several models: method rows × the number of models, the null policy, `split_use` and the means of the attribute averages over models. The confidence bounds include the spread of the models. |
| `per_attribute_<metric>_<dataset>_<split>_<set>.csv` | Attribute rows × model columns, with the average last. |
| `<model>/metrics_<dataset>_<split>_<set>.json` | All metrics and intervals, and for each attribute the iRAP codes of its classes, the per-class precision, recall, F1 and support, the confusion matrix (rows = ground truth), and for `probs` files the [review curve](docs/evaluation.md#manual-review-proportion). Undefined values are `null`. In the directory name, `/` is replaced by `_`. |

## Python use

See [`docs/prediction_format.md`](docs/prediction_format.md#examples) for writing prediction files. Scoring:

```python
from irap_data import load_irap_metadata
import irap_evaluation as ie

metadata = load_irap_metadata("/data/IRAP_Vietnam")
predictions = ie.read_predictions("resnet.predictions.parquet")
reference_set = ie.get_reference_set(metadata, "vietnam", "val")
result = ie.evaluate_predictions(predictions, reference_set)
intervals = ie.compute_bootstrap_intervals([result], num_resamples=1000)
print(result.metrics.averages["amF1"], intervals.averages["amF1"])
print(result.metrics.per_attribute["mF1"]["Lane width"])
class_metrics = ie.compute_class_metrics(result)

# Methods with several models: means over models, intervals and differences.
results = [ie.evaluate_predictions(ie.read_predictions(path), reference_set) for path in paths]
method_to_results = ie.group_models_by_method(results)
print(ie.compute_method_metrics(method_to_results["resnet"]).averages["amF1"],
      ie.compute_bootstrap_intervals(method_to_results["resnet"]).averages["amF1"])
differences = ie.compare_methods(method_to_results["resnet"], method_to_results["vit"], ["amF1"])
print(differences.averages["amF1"])  # MetricDifference(difference, interval)

# Other evaluation sets: the full set, the model-compatible set of some offsets, and a subset,
# e.g. one region.
full_set = ie.get_full_set(metadata, "vietnam", "val")
model_compatible_set = ie.get_model_compatible_set(metadata, "vietnam", "val", (0, -1, -4))
north_set = ie.select_evaluation_subset(reference_set, is_north, name="north")
```

Tables, JSON documents and the files of `irap-eval evaluate`:

```python
from irap_evaluation.reports import evaluation_report

long_table = evaluation_report.to_long_table([result])
print(evaluation_report.to_average_table(long_table))
evaluation_report.write_evaluation_reports([result], "results/", table_metrics=["mF1"])
```

## Coding tables

`export-coding-table` (`irap_evaluation.reports.coding_tables`) writes one row per predicted segment of the given files of one model, e.g. one file per split. The layout is that of `--template`, by default [`coding_table_template.json`](src/irap_evaluation/reports/data/coding_table_template.json):

- attribute columns: the iRAP code of the most probable class,
- location columns: from the dataset metadata, with the end coordinates from the next segment,
- other columns: blank, with the reason in the template's `_comment`.

A `.csv` file is UTF-8 with a byte order mark, so that Excel shows Vietnamese section names correctly.

## Layout

The layers import only downward:

```
irap_evaluation/
├─ predictions, prediction_io                        The Predictions datatype and the file format
├─ metric_names, metrics                             Metric names, statistics and computation
├─ evaluation_sets, evaluation                       Scoring a model on an evaluation set
├─ methods, bootstrap                                Means over models, intervals and comparisons
├─ ensembling                                        Combining predictions
├─ reports/                                          Tables and JSON documents of results
│                                                    (evaluation_report), coding tables
└─ tools/                                            The irap-eval command (cli)
```
