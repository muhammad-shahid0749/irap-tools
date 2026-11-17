"""iRAP road-attribute datasets.

The metadata API (`attrs`, `metadata`, `lazy_dict`) needs only numpy. The dataset classes and
image utilities need the `torch` extra and are imported on first access, so that importing
`irap_data` does not import torch. `reports` computes the dataset statistics and writes their
report, and `tools` holds the command-line tools and the dataset viewer. The library imports
neither.
"""

import importlib
import typing as T

from .attrs import (
    IRAP_BH_ATTRS_TO_INCLUDE,
    filter_labeled_attrs,
    get_attrs_to_include,
    map_attr_names_to_indices,
)
from .lazy_dict import Lazy, LazyDict
from .metadata import (
    DATASET_PRESETS,
    IGNORE_LABEL_INDEX,
    ClassVocabulary,
    DatasetPreset,
    IRAPMetadata,
    SegmentLocation,
    SplitSelection,
    compute_class_occurrence_counts,
    compute_label_matrix,
    compute_num_segments_per_class,
    compute_segment_labels,
    compute_split_labels,
    get_bh_class_counts,
    get_bih_class_counts,
    get_class_counts,
    get_dataset_preset,
    get_default_irap_paths,
    get_segment_location,
    is_unlabeled_split,
    load_attribute_metadata,
    load_class_vocabulary,
    load_irap_metadata,
    load_ncontext_segment_ids,
    resolve_irap_paths,
    select_preset_split_segments,
    select_split_segments,
)

# Name -> submodule, for the names that need torch.
_TORCH_NAME_TO_MODULE = {
    "Dataset": "dataset",
    "InferenceImageDataset": "inference_dataset",
    "IRAP_DATASET_FACTORIES": "irap_dataset",
    "IRAPDataset": "irap_dataset",
    "make_bh_data": "irap_dataset",
    "make_bih_data": "irap_dataset",
    "make_irap_data": "irap_dataset",
    "make_irap_data_by_name": "irap_dataset",
    "make_vietnam_data": "irap_dataset",
    "JITTER_STANDARD": "jitter",
    "JITTER_STRONG": "jitter",
    "make_sequence_color_jitter": "jitter",
}

# `from irap_data import *` imports the torch names too, and so needs the `torch` extra.
__all__ = [name for name in globals() if not name.startswith("_")
           and name not in ("importlib", "T", "attrs", "lazy_dict", "metadata")]
__all__ += list(_TORCH_NAME_TO_MODULE)


def __getattr__(name: str) -> T.Any:
    if name not in _TORCH_NAME_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = _TORCH_NAME_TO_MODULE[name]
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"irap_data.{name} needs {e.name!r}. Install the extra: pip install 'irap-data[torch]'."
        ) from e
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_TORCH_NAME_TO_MODULE])
