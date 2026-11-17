"""The `irap-eval` command-line interface.

Run `irap-eval <command> --help` for the options of each command.
"""

import argparse
import dataclasses as dc
import functools
import sys
import typing as T
from datetime import date
from pathlib import Path

import pandas as pd

from irap_data.metadata import load_irap_metadata

from ..bootstrap import BootstrapInterval, compare_methods, compute_bootstrap_intervals
from ..ensembling import MEMBER_COMBINATIONS, ensemble_predictions
from ..evaluation import (
    ATTRIBUTE_SELECTIONS,
    MISSING_ATTRIBUTE_POLICIES,
    NULL_POLICIES,
    AttributeSelection,
    EvaluationResult,
    MissingAttributePolicy,
    NullPolicy,
    evaluate_predictions,
    get_default_metric_names,
    select_evaluated_attributes,
)
from ..evaluation_sets import (
    EvaluationSet,
    get_model_compatible_set,
    get_reference_set,
    get_shared_evaluation_sets,
    select_evaluation_sets,
)
from ..methods import check_prediction_file_models
from ..metric_names import IRAP_MAIN_METRIC, parse_metric_name
from ..metrics import MetricValues, map_metric_values
from ..prediction_io import read_predictions, write_predictions
from ..predictions import (
    ModelInfo,
    Predictions,
    check_split_names,
    explain_split_use,
    get_common_dataset_and_split,
)
from ..reports import coding_tables, evaluation_report

#: The metrics that `--table-metrics` accepts: the per-attribute metrics that are computed by
#: default for probs files.
TABLE_METRICS = tuple(n for n in get_default_metric_names("probs")
                      if not parse_metric_name(n).is_averaged)


# Helpers ##########################################################################################

def _parse_context_offsets(text: str) -> tuple[int, ...]:
    return tuple(int(o) for o in text.split(","))


def _parse_method_context_offsets(items: T.Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Parses `NAME=0,-1,-4` items."""
    def parse(item: str) -> tuple[str, tuple[int, ...]]:
        name, separator, offsets = item.rpartition("=")
        if not separator or not name:
            raise ValueError(f"Expected NAME=OFFSETS, e.g. resnet=0,-1,-4, got {item!r}.")
        return name, _parse_context_offsets(offsets)

    return dict(map(parse, items))


def _read_prediction_files(paths: T.Sequence[Path]) -> list[Predictions]:
    """Reads prediction files of models that `check_prediction_file_models` accepts."""
    predictions_seq = [read_predictions(p) for p in paths]
    check_prediction_file_models([p.header for p in predictions_seq])
    return predictions_seq


def _explain_split_use(model: ModelInfo, split: str) -> str | None:
    """A note on the scores of `model` on `split` if it used the split, or may have used it."""
    return explain_split_use(model.get_split_use(split), split)


def _format_training_splits(model: ModelInfo) -> str:
    training = model.training_splits
    return "unknown splits" if training is None else str(list(training))


def _print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}\n\n{evaluation_report.to_markdown_table(table)}")


def _print_partially_undefined_intervals(subject: str, intervals: MetricValues[BootstrapInterval],
                                         num_resamples: int) -> None:
    """Prints the intervals that leave out some resamples, if there are any."""
    if label_to_count := evaluation_report.get_partially_undefined_intervals(intervals):
        counts = ", ".join(f"{label} {n}" for label, n in label_to_count.items())
        print(f"{subject}: undefined resamples (NaN values, e.g. inf - inf), of {num_resamples},"
              f" left out of the intervals: {counts}.")


# Commands #########################################################################################

def run_validate(args: argparse.Namespace) -> None:
    for path in args.files:
        header = (predictions := read_predictions(path)).header
        num_invalid = sum(int((~v).sum()) for v in predictions.is_valid.values())
        print(f"{path}: valid. Model {header.model.label!r}, {header.dataset}/{header.split},"
              f" trained on {_format_training_splits(header.model)},"
              f" early stopped on {list(header.model.early_stopping_splits)},"
              f" {predictions.output_kind}, {predictions.num_segments} segments,"
              f" {len(predictions.attributes)} attributes, {num_invalid} invalid attribute"
              f" predictions.")


def _score(predictions: Predictions, evaluation_set: EvaluationSet, *, null_policy: NullPolicy,
           attribute_selection: AttributeSelection,
           missing_attribute_policy: MissingAttributePolicy, num_resamples: int,
           confidence: float, seed: int) -> EvaluationResult:
    """Scores predictions, with bootstrap intervals of the model if `num_resamples` > 0."""
    result = evaluate_predictions(
        predictions, evaluation_set, null_policy=null_policy,
        missing_attribute_policy=missing_attribute_policy,
        attributes=select_evaluated_attributes(evaluation_set, attribute_selection))
    if result.missing_attributes:
        print(f"Model {result.model.label!r}, {evaluation_set.name} set: scored as invalid, since"
              f" not predicted: {', '.join(result.missing_attributes)}.")
    if not num_resamples:
        return result
    return dc.replace(result, intervals=compute_bootstrap_intervals(
        [result], num_resamples=num_resamples, confidence=confidence, seed=seed))


def _select_model_evaluation_sets(
    predictions: Predictions,
    context_offsets: tuple[int, ...] | None,
    shared_sets_f: T.Callable[[str, str], T.Mapping[str, EvaluationSet]],
    model_compatible_set_f: T.Callable[[str, str, tuple[int, ...]], EvaluationSet],
) -> list[EvaluationSet]:
    """`select_evaluation_sets` for a model, with its notes printed, and a note if it used the
    split (`_explain_split_use`)."""
    header = predictions.header
    name = header.model.label
    model_compatible_set = None if context_offsets is None else model_compatible_set_f(
        header.dataset, header.split, tuple(context_offsets))
    try:
        evaluation_sets, notes = select_evaluation_sets(
            predictions.segment_ids, shared_sets_f(header.dataset, header.split).values(),
            model_compatible_set)
    except ValueError as e:
        raise ValueError(f"Model {name!r}: {e}") from e
    if evaluation_sets and (split_use_note := _explain_split_use(header.model, header.split)):
        notes = [*notes, split_use_note]
    for note in notes:
        print(f"Model {name!r}: {note}")
    return evaluation_sets


def _compute_method_average_tables(
    results: T.Sequence[EvaluationResult], *, num_resamples: int, confidence: float, seed: int,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """`evaluation_report.to_method_average_table` of each evaluation set with methods of
    several models, with intervals if `num_resamples` > 0, whose undefined resamples are
    printed."""
    def to_table(key: tuple[str, str, str],
                 method_to_models: T.Mapping[str, T.Sequence[EvaluationResult]]) -> pd.DataFrame:
        intervals = None if not num_resamples else {
            name: compute_bootstrap_intervals(models, num_resamples=num_resamples,
                                              confidence=confidence, seed=seed)
            for name, models in method_to_models.items()}
        dataset, split, evaluation_set = key
        for name, method_intervals in (intervals or {}).items():
            _print_partially_undefined_intervals(
                f"Method {name!r}, {dataset}/{split}, {evaluation_set} set", method_intervals,
                num_resamples)
        return evaluation_report.to_method_average_table(method_to_models, intervals)

    return {key: to_table(key, method_to_models)
            for key, method_to_models
            in evaluation_report.group_multi_model_methods(results).items()}


def run_evaluate(args: argparse.Namespace) -> None:
    predictions_seq = _read_prediction_files(args.files)
    evaluation_report.check_model_dir_names(p.header.model.label for p in predictions_seq)
    method_context_offsets = _parse_method_context_offsets(args.context_offsets)
    method_names = {p.header.model.method_name for p in predictions_seq}
    if unknown := set(method_context_offsets) - method_names:
        raise ValueError(f"--context-offsets names unknown methods: {sorted(unknown)}.")
    metadata = load_irap_metadata(args.metadata_dir)
    for predictions in predictions_seq:
        check_split_names(predictions.header.model, metadata.splits)
    # Cached, since the shared sets are the same for all models of a release and split.
    shared_sets_f = functools.cache(functools.partial(get_shared_evaluation_sets, metadata))
    model_compatible_set_f = functools.cache(functools.partial(get_model_compatible_set, metadata))
    score = functools.partial(
        _score, null_policy=args.null_policy, attribute_selection=args.attributes,
        missing_attribute_policy=args.missing_attribute_policy, num_resamples=args.bootstrap,
        confidence=args.confidence, seed=args.seed)
    results = [
        score(predictions, evaluation_set)
        for predictions in predictions_seq
        for evaluation_set in _select_model_evaluation_sets(
            predictions,
            method_context_offsets.get(predictions.header.model.method_name,
                                       predictions.header.context_offsets),
            shared_sets_f, model_compatible_set_f)]
    if not results:
        raise ValueError("No model has an evaluation set with labels (see the notes above), so"
                         " nothing is scored.")
    for result in results:
        if result.intervals is not None:
            _print_partially_undefined_intervals(
                f"Model {result.model.label!r}, {result.evaluation_set.name} set",
                result.intervals, args.bootstrap)
    # Means over models need 'first_class' (NullPolicy).
    if args.null_policy == "first_class":
        method_average_tables = _compute_method_average_tables(
            results, num_resamples=args.bootstrap, confidence=args.confidence, seed=args.seed)
    else:
        method_average_tables = {}
        if evaluation_report.group_multi_model_methods(results):
            print(f"No means over the models of a method: the null policy {args.null_policy}"
                  f" scores each model on its own segments. Use --null-policy first_class for"
                  f" them.")

    evaluation_report.write_evaluation_reports(results, args.out, table_metrics=args.table_metrics,
                                               method_average_tables=method_average_tables)
    metric = args.table_metrics[0]
    evaluation_set_tables = evaluation_report.to_evaluation_set_tables(
        evaluation_report.to_long_table(results), [metric])
    for key, name_to_table in evaluation_set_tables.items():
        dataset, split, evaluation_set = key
        _print_table(f"Attribute averages per model, {dataset}/{split}, {evaluation_set} set",
                     name_to_table["averages"])
        if key in method_average_tables:
            _print_table(f"Attribute averages of the methods with several models (means over"
                         f" models), {dataset}/{split}, {evaluation_set} set",
                         method_average_tables[key])
        _print_table(f"{metric} per attribute, {dataset}/{split}, {evaluation_set} set",
                     name_to_table[f"per_attribute_{metric}"])
    print(f"\nWrote the results to {args.out}.")


def run_compare(args: argparse.Namespace) -> None:
    predictions_seq = _read_prediction_files(args.files)
    method_names = {p.header.model.method_name for p in predictions_seq}
    if missing := [n for n in (args.a, args.b) if n not in method_names]:
        raise ValueError(f"No files of the methods {missing}. The files are of"
                         f" {sorted(method_names)}.")
    if other := sorted(method_names - {args.a, args.b}):
        raise ValueError(f"Files of other methods than a and b: {other}.")
    dataset, split = get_common_dataset_and_split(p.header for p in predictions_seq)
    metadata = load_irap_metadata(args.metadata_dir)
    for predictions in predictions_seq:
        check_split_names(predictions.header.model, metadata.splits)
    reference_set = get_reference_set(metadata, dataset, split)
    attributes = select_evaluated_attributes(reference_set, args.attributes)

    def evaluate_models(method_name: str) -> list[EvaluationResult]:
        return [evaluate_predictions(p, reference_set, attributes=attributes,
                                     null_policy="first_class")
                for p in predictions_seq if p.header.model.method_name == method_name]

    models_a, models_b = evaluate_models(args.a), evaluate_models(args.b)
    differences = compare_methods(models_a, models_b, args.metrics, num_resamples=args.bootstrap,
                                  seed=args.seed, confidence=args.confidence)
    _print_table(f"a = {args.a} ({len(models_a)} models), b = {args.b} ({len(models_b)} models),"
                 f" {dataset}/{split}, reference set, {reference_set.num_segments} segments,"
                 f" {len(reference_set.sequence_ids)} road sequences, {args.bootstrap} resamples,"
                 f" {args.confidence:.0%} intervals",
                 evaluation_report.to_comparison_table(models_a, models_b, differences))
    for result in (models_a[0], models_b[0]):  # The models of a method use the same splits.
        if note := _explain_split_use(result.model, split):
            print(f"Method {result.model.method_name!r}: {note}")
    _print_partially_undefined_intervals(
        "a - b", map_metric_values(lambda d: d.interval, differences), args.bootstrap)


def run_ensemble(args: argparse.Namespace) -> None:
    predictions_seq = _read_prediction_files(args.files)
    ensemble = ensemble_predictions(
        predictions_seq, args.name, method_display_name=args.display_name, seed=args.model_seed,
        weights=args.weights, segment_combination=args.segment_combination,
        attribute_combination=args.attribute_combination)
    write_predictions(args.output, ensemble)
    print(f"Wrote the mean of {len(predictions_seq)} prediction files as the method"
          f" {ensemble.header.model.method_name!r}"
          f" ({ensemble.num_segments} segments, trained on"
          f" {_format_training_splits(ensemble.header.model)}) to {args.output}.")


def run_export_coding_table(args: argparse.Namespace) -> None:
    predictions_seq = [read_predictions(p) for p in args.files]
    metadata = load_irap_metadata(args.metadata_dir)
    for predictions in predictions_seq:
        check_split_names(predictions.header.model, metadata.splits)
    template = coding_tables.load_coding_table_template(args.template)
    table = coding_tables.predictions_to_coding_table(
        predictions_seq, metadata, template,
        coder_name=args.coder_name or predictions_seq[0].header.model.shown_method_name,
        coding_date=args.coding_date)
    confidence_table = (coding_tables.predictions_to_confidence_table(predictions_seq, metadata,
                                                                      template)
                        if args.confidence_table else None)
    for path in coding_tables.write_coding_table(args.output, table, confidence_table):
        print(f"Wrote {path}.")
    attributes = list(dict.fromkeys(a for p in predictions_seq for a in p.attributes))
    if unmapped := coding_tables.get_unmapped_attributes(template, attributes):
        print(f"Predicted attributes without a column (not exported): {unmapped}.")
    print(f"Blank columns: {coding_tables.get_unfilled_columns(template, attributes)}.")


# Argument parsing #################################################################################

def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attributes", choices=ATTRIBUTE_SELECTIONS, default="canonical",
                        help="canonical: the iRAP attribute subset."
                             " all: every attribute of the dataset. Attributes without a label"
                             " in the evaluation set are left out. Default: canonical.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed. Default: 0.")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level of the intervals. Default: 0.95.")


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irap-eval", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Check prediction files.")
    validate.add_argument("files", nargs="+", type=Path)
    validate.set_defaults(run=run_validate)

    evaluate = commands.add_parser(
        "evaluate", help="Score prediction files on the reference set, on the full set if they"
                         " predict all its segments, and, if the context offsets of a method are"
                         " known, on its model-compatible set. Methods with several models also"
                         " get the means over their models.")
    evaluate.add_argument("metadata_dir", type=Path, help="Metadata directory of the release.")
    evaluate.add_argument("files", nargs="+", type=Path, help="Prediction files.")
    evaluate.add_argument("--out", type=Path, required=True, help="Output directory.")
    evaluate.add_argument("--context-offsets", nargs="*", default=[], metavar="NAME=OFFSETS",
                          help="Context offsets of methods, e.g. resnet=0,-1,-4. They override"
                               " the offsets in the file headers.")
    evaluate.add_argument("--bootstrap", type=int, default=0, metavar="NUM_RESAMPLES",
                          help="Number of bootstrap resamples for confidence intervals. The"
                               " intervals of a model cover the road sequences, those of a method"
                               " with several models also the spread of its models."
                               " Default: 0 (no intervals).")
    evaluate.add_argument("--table-metrics", nargs="+", choices=TABLE_METRICS,
                          default=[IRAP_MAIN_METRIC], metavar="METRIC",
                          help=f"Metrics of the per-attribute tables, which end with the attribute"
                               f" average, from {', '.join(TABLE_METRICS)}. The metrics of"
                               f" distributions need probs files. Default: {IRAP_MAIN_METRIC}.")
    evaluate.add_argument("--missing-attribute-policy", choices=MISSING_ATTRIBUTE_POLICIES,
                          default="error",
                          help="How a selected attribute that a file does not predict is scored."
                               " error: not at all, the command fails. invalid: as if every"
                               " attribute prediction of it were invalid (see --null-policy)."
                               " Default: error.")
    evaluate.add_argument("--null-policy", choices=NULL_POLICIES, default="first_class",
                          help="How the results of the models score invalid attribute"
                               " predictions. first_class:"
                               " as class 0 in the metrics of the predicted classes and as the"
                               " uniform distribution in NLL and Brier. exclude: not scored, so"
                               " each model is scored on its own segments and no means over the"
                               " models of a method are written (see"
                               " irap_evaluation.evaluation.NullPolicy). Default: first_class.")
    _add_scoring_arguments(evaluate)
    evaluate.set_defaults(run=run_evaluate)

    compare = commands.add_parser(
        "compare", help="Compare two methods, each with one or more models, on the reference set,"
                        " with a bootstrap over road sequences and the spread of the models."
                        " Invalid attribute predictions are scored with the null policy"
                        " first_class.")
    compare.add_argument("metadata_dir", type=Path)
    compare.add_argument("files", nargs="+", type=Path,
                         help="The prediction files of the models of both methods.")
    compare.add_argument("--a", required=True, metavar="METHOD", help="Name of method a.")
    compare.add_argument("--b", required=True, metavar="METHOD", help="Name of method b.")
    compare.add_argument("--metrics", nargs="+", default=None,
                         help="Metrics to compare, e.g. amF1 or the per-attribute mF1."
                              " Default: all attribute averages.")
    compare.add_argument("--bootstrap", type=int, default=1000, metavar="NUM_RESAMPLES",
                         help="Number of bootstrap resamples. Default: 1000.")
    _add_scoring_arguments(compare)
    compare.set_defaults(run=run_compare)

    ensemble = commands.add_parser(
        "ensemble", help="Combine prediction files into the weighted mean of their distributions."
                         " The ensemble's training splits are the union of the members' training"
                         " splits.")
    ensemble.add_argument("files", nargs="+", type=Path)
    ensemble.add_argument("--name", default=None,
                          help="Method name of the ensemble. Default: the sorted method names of"
                               " the members with their seeds, e.g. resnet-seq_seeds0-4+vit-l.")
    ensemble.add_argument("--display-name", default=None,
                          help="Method display name of the ensemble. Default: none.")
    ensemble.add_argument("--model-seed", type=int, default=None,
                          help="Seed of the ensemble, if it is one model of a method with several"
                               " models. Default: none.")
    ensemble.add_argument("-o", "--output", type=Path, required=True)
    ensemble.add_argument("--weights", type=float, nargs="+", default=None,
                          help="A positive weight per file. Default: equal weights.")
    for noun in ("segment", "attribute"):
        ensemble.add_argument(f"--{noun}-combination", choices=MEMBER_COMBINATIONS,
                              default="same",
                              help=f"Which {noun}s to combine if the files predict different ones."
                                   f" same: none, the command fails. intersection: those that"
                                   f" every file predicts. union: those that any file predicts,"
                                   f" where a file that does not predict one counts as invalid."
                                   f" Default: same.")
    ensemble.set_defaults(run=run_ensemble)

    export = commands.add_parser("export-coding-table",
                                 help="Write the predictions of a model as an iRAP coding table.")
    export.add_argument("metadata_dir", type=Path)
    export.add_argument("files", nargs="+", type=Path,
                        help="Prediction files of one model on disjoint segments, e.g. one per"
                             " split, which go into one table.")
    export.add_argument("-o", "--output", type=Path, required=True, help="A .xlsx or .csv path.")
    export.add_argument("--coder-name", default=None,
                        help="Default: the method display name, or the method name.")
    export.add_argument("--coding-date", default=date.today().isoformat(),
                        help="YYYY-MM-DD. Default: today.")
    export.add_argument("--confidence-table", action="store_true",
                        help="Also write the probability of each exported code (probs files), as"
                             " a second sheet of a .xlsx file or as <stem>.confidence.csv.")
    export.add_argument("--template", type=Path, default=None,
                        help="A coding-table template JSON. Default: the packaged iRAP layout.")
    export.set_defaults(run=run_export_coding_table)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_argument_parser().parse_args(argv)
    try:
        args.run(args)
    except (ValueError, FileNotFoundError) as e:  # including PredictionFormatError
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
