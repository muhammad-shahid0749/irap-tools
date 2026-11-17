"""Saved predictions of iRAP road-attribute methods: file format, evaluation and ensembling.

The tables and JSON documents of results and the coding-table export are in
`irap_evaluation.reports`, and the `irap-eval` command is in `irap_evaluation.tools`. This package
does not import them.

Typical use::

    from irap_data import load_irap_metadata
    import irap_evaluation as ie

    predictions = ie.read_predictions("resnet.predictions.parquet")
    metadata = load_irap_metadata("/data/IRAP_Vietnam")
    reference_set = ie.get_reference_set(metadata, "vietnam", "val")
    result = ie.evaluate_predictions(predictions, reference_set)
    print(result.metrics.averages["amF1"])
"""

from .bootstrap import (
    BootstrapInterval,
    MetricDifference,
    combine_model_resamples,
    compare_methods,
    compute_bootstrap_intervals,
    compute_percentile_interval,
    compute_resampled_differences,
    resample_model_metrics,
)
from .ensembling import (
    MEMBER_COMBINATIONS,
    MemberCombination,
    ensemble_predictions,
    make_ensemble_method_name,
    make_ensemble_model_info,
)
from .evaluation import (
    AttributeSelection,
    ClassMetrics,
    EvaluationResult,
    MissingAttributePolicy,
    NullPolicy,
    align_to_evaluation_set,
    compute_class_metrics,
    compute_review_curves,
    evaluate_predictions,
    get_default_metric_names,
    select_evaluated_attributes,
    select_result_attributes,
)
from .evaluation_sets import (
    FULL_SET_NAME,
    MODEL_COMPATIBLE_SET_NAME,
    REFERENCE_SET_NAME,
    SHARED_EVALUATION_SET_NAMES,
    EvaluationSet,
    get_evaluation_set,
    get_full_set,
    get_model_compatible_set,
    get_reference_set,
    get_shared_evaluation_sets,
    select_evaluation_sets,
    select_evaluation_subset,
)
from .file_names import MAX_FILE_NAME_BYTES, shorten_file_name, to_valid_file_name
from .methods import (
    check_method_models,
    check_method_results,
    check_models_have_same_metrics,
    check_prediction_file_models,
    check_results_are_comparable,
    compute_mean_metrics,
    compute_method_metrics,
    group_models_by_method,
)
from .metric_names import (
    IRAP_ATTRIBUTE_METRIC_NAMES,
    IRAP_MAIN_METRIC,
    MANUAL_REVIEW_PROPORTION_METRIC,
    MANUAL_REVIEW_TARGET_MF1,
    ParsedMetricName,
    add_per_attribute_metric_names,
    describe_attribute_average,
    describe_metric,
    get_attribute_average_name,
    get_irap_metric_names,
    is_lower_better,
    parse_metric_name,
)
from .metrics import (
    NUM_CONFIDENCE_BINS,
    ClassificationStatistics,
    MetricValues,
    ReviewCurve,
    compute_classification_metrics,
    compute_classification_statistics,
    compute_grouped_metrics,
    compute_multi_attribute_metrics,
    compute_manual_review_proportion,
    compute_review_curve,
    map_metric_values,
    select_metric_attributes,
    select_metric_names,
    sum_statistics,
)
from .prediction_io import (
    make_prediction_file_name,
    read_prediction_header,
    read_predictions,
    write_predictions,
)
from .predictions import (
    INVALID_IRAP_CODE,
    ModelInfo,
    OutputKind,
    PredictionFormatError,
    PredictionHeader,
    Predictions,
    SplitUse,
    add_invalid_attributes,
    add_invalid_segments,
    align_classes,
    check_class_codes,
    check_split_names,
    explain_split_use,
    format_model_label,
    get_common_dataset_and_split,
    select_segments,
    to_class_indices,
    to_irap_codes,
)
