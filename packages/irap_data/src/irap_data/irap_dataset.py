from pathlib import Path
import typing as T
import warnings
from types import SimpleNamespace

import numpy as np
import torch

from .dataset import Dataset
from .lazy_dict import LazyDict
from .image_utils import (
    DEFAULT_FIT_MODE,
    FitMode,
    fit_to_size,
    float_rgb_to_uint8,
    hwc_to_chw_float_tensor,
    load_image_cv2,
)
from .metadata import (
    DATASET_PRESETS,
    IRAPMetadata,
    get_default_irap_paths,
    is_unlabeled_split,
    load_irap_metadata,
    load_ncontext_segment_ids,
    resolve_irap_paths,
    select_split_segments,
)

RGB_MEAN: tuple[float, float, float] = (0.53354913, 0.52727484, 0.48752149)
RGB_STD: tuple[float, float, float] = (0.20401913, 0.20417478, 0.25402164)
INPUT_DIM: tuple[int, int, int] = (384, 288, 3)


class IRAPDataset(Dataset):
    """iRAP road sequence dataset for multi-attribute classification.

    Loads image sequences and corresponding road attribute labels for segments in a split. The
    segments are chosen by `metadata.select_split_segments`.

    Args:
        root: Dataset root directory containing image and sensor files.
        subset: Split name to load (e.g. 'train', 'val', 'test', 'unlabeled_train').
        metadata_dir: Directory containing metadata JSON files. If None, inferred
            from `root`. Not used if `metadata` is given.
        metadata: The loaded metadata files, to share them between the splits of a release.
        context_offsets: Frame offsets for context images relative to the target segment.
        mean: Channel means for RGB normalization.
        std: Channel standard deviations for RGB normalization.
        transforms: Optional transform mapping per modality.
        ncontext_segment_id_subset: Optional segment ID filter set.
        allow_missing_attributes: If True, missing or unmappable attribute labels are
            assigned `IGNORE_LABEL_INDEX` (-1) instead of discarding the segment.
        dataset_name: The release name, stored in `info` (see `make_irap_data`).

    `info` also holds `split` (= `subset`) and `context_offsets`, which transformations of the
    dataset keep, so that predictions can be traced back to the segments (see
    `irap_evaluation.PredictionHeader`).
    """

    # Canonical labeled subsets. The constructor accepts any key present in
    # `splits.json` (including `unlabeled_*` ones produced by the Vietnam prep
    # pipeline), but this tuple names the splits expected to carry labels.
    subsets = ("train", "val", "test")

    def __init__(
        self,
        root: str | Path,
        subset: str = "train",
        *,
        metadata_dir: str | Path | None = None,
        metadata: IRAPMetadata | None = None,
        context_offsets: T.Sequence[int] = (0, -1, -4),
        mean: T.Sequence[float] = RGB_MEAN,
        std: T.Sequence[float] = RGB_STD,
        transforms: T.Mapping[str, T.Callable] | None = None,
        ncontext_segment_id_subset: set[str] | None = None,
        allow_missing_attributes: bool = False,
        dataset_name: str | None = None,
    ) -> None:
        self.transforms = transforms or {}
        self.root = Path(root)
        if metadata is None:
            metadata = load_irap_metadata(
                metadata_dir if metadata_dir is not None
                else self.root.parent / (self.root.name + "_METADATA")
            )
        self.metadata_dir = metadata.metadata_dir
        self.context_offsets = list(context_offsets)

        selection = select_split_segments(
            metadata,
            subset,
            context_offsets=self.context_offsets,
            allow_missing_attributes=allow_missing_attributes,
            ncontext_segment_id_subset=ncontext_segment_id_subset,
        )

        self.seg_to_paths = {
            sid: {k: (None if v == "NONE" else (self.root / v)) for k, v in d.items()}
            for sid, d in metadata.segment_id_to_data_paths_rel.items()
        }
        self.road_to_seq = metadata.road_id_to_segment_id_sequence
        self.seq_index = metadata.sequence_index
        self.segment_id_to_labels = selection.segment_id_to_labels
        self.segment_id_to_context_ids = selection.segment_id_to_context_ids
        self.segment_ids = list(selection.segment_ids)

        vocabulary = metadata.vocabulary
        super().__init__(
            subset=subset,
            info=dict(
                problem="multi_attribute_classification",
                class_counts=vocabulary.class_counts,
                pixel_stats=SimpleNamespace(mean=np.array(mean), std=np.array(std)),
                attr_to_value_to_class_idx=vocabulary.attr_to_value_to_class_idx,
                attr_to_num_labeled=dict(selection.attr_to_num_labeled),
                segment_id_to_labels=self.segment_id_to_labels,
                segment_id_to_context_ids=self.segment_id_to_context_ids,
                segment_ids=self.segment_ids,
                metadata_dir=str(self.metadata_dir),
                dataset_name=dataset_name,
                split=subset,
                context_offsets=tuple(self.context_offsets),
            ),
        )

    def __len__(self) -> int:
        return len(self.segment_ids)

    def _load_sequence(self, seq_ids: T.Sequence[str], kind: str) -> torch.Tensor:
        tfm = self.transforms.get(kind)
        frames: list[torch.Tensor] = []
        for sid in seq_ids:
            p = self.seg_to_paths[sid].get(kind)
            if p is None:
                continue
            arr = load_image_cv2(str(p))
            t = tfm(arr) if tfm is not None else hwc_to_chw_float_tensor(arr)
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(np.asarray(t))
            frames.append(t.float())
        if not frames:
            return torch.empty(0)
        return torch.stack(frames, dim=0)

    def load_frame(self, segment_id: str, kind: str = "rgb") -> torch.Tensor:
        """The single transformed `(C, H, W)` frame of a segment.

        The same decoding and transform an example's frames go through, for consumers
        that address frames by segment id rather than by example (feature extraction).

        Raises:
            KeyError: If the segment has no image of this kind.
        """
        frames = self._load_sequence([segment_id], kind)
        if len(frames) == 0:
            raise KeyError(f"Segment {segment_id!r} has no {kind!r} image.")
        return frames[0]

    def get_example(self, idx: int) -> dict:
        # `select_split_segments` selects only segments with a complete context window and an
        # entry in `segment_id_to_labels` (all `IGNORE_LABEL_INDEX` in unlabeled splits). The
        # context segments themselves need only an image.
        sid = self.segment_ids[idx]
        return dict(
            rgb=self._load_sequence(self.segment_id_to_context_ids[sid], "rgb"),
            target=torch.LongTensor(self.segment_id_to_labels[sid]),
            segment_id=sid,
            sequence_id=self.seq_index[sid][0],
        )


def make_irap_data(
    *,
    dataset_dir: str | Path,
    metadata_dir: str | Path,
    context_offsets: T.Sequence[int] = (0, -1, -4),
    mean: T.Sequence[float] = RGB_MEAN,
    std: T.Sequence[float] = RGB_STD,
    input_dim_rgb: T.Sequence[int] = INPUT_DIM,
    fit_mode: FitMode = DEFAULT_FIT_MODE,
    transforms: T.Mapping[str, T.Callable] | None = None,
    ncontext_segment_id_subset: set[str] | None = None,
    use_ncontext_filter: bool = False,
    seg_to_res_path: str | Path | None = None,
    allow_missing_attributes: bool = False,
    dataset_name: str | None = None,
):
    """Builds iRAP datasets for all splits in a dataset metadata directory.

    Discovers splits from `splits.json` in `metadata_dir` and constructs an
    `IRAPDataset` instance for each split.

    Args:
        dataset_dir: Directory containing image and sensor data files.
        metadata_dir: Directory containing metadata JSON files (`splits.json`, etc.).
        context_offsets: Offsets for temporal context frames relative to the target segment.
        mean: Channel means for RGB normalization.
        std: Channel standard deviations for RGB normalization.
        input_dim_rgb: Target image dimensions (width, height, channels).
        fit_mode: How the default transform fits an image to `input_dim_rgb` (see
            `image_utils.fit_to_size`). The padding of `"fit_width"` has the color `mean`, so it is
            0 after standardization. iRAP-BH images have exactly the default size, so they pass
            through unchanged with every option except `"resize_width"` with another width.
        transforms: Custom transforms mapping by split name or single transform dict. If given,
            `input_dim_rgb` and `fit_mode` are not used.
        ncontext_segment_id_subset: Optional explicit set of segment IDs to include.
        use_ncontext_filter: Whether to apply N-context filtering from precomputed result files.
        seg_to_res_path: Directory containing precomputed `<split>.pickle` files for N-context
            filtering. Defaults to `<metadata_dir>/seg_to_res`.
        allow_missing_attributes: If True, missing or unmappable attribute labels are mapped
            to `IGNORE_LABEL_INDEX` (-1) rather than dropping the segment.
        dataset_name: The release name, a key of `IRAP_DATASET_FACTORIES`, stored in the
            `info` of the datasets. None for a directory that is not a known release.

    Returns:
        LazyDict mapping split names to `IRAPDataset` instances.
    """
    dataset_dir = Path(dataset_dir)
    metadata = load_irap_metadata(metadata_dir)

    # Unlabeled subsets (`unlabeled_train`, `unlabeled_val`, ...) appear only when the prep
    # pipeline wrote them. Iterating the file rather than a hardcoded tuple makes those splits
    # available transparently.
    split_names = list(metadata.splits)

    # Load segment ID filter from precomputed results if requested.
    # The pickle files only cover labeled segments, so this filter is applied
    # to labeled subsets only. Unlabeled subsets skip it.
    if use_ncontext_filter and ncontext_segment_id_subset is None:
        ncontext_segment_id_subset = (
            metadata.ncontext_segment_ids
            if seg_to_res_path is None
            else load_ncontext_segment_ids(
                seg_to_res_path, metadata.road_id_to_segment_id_sequence
            )
        )

    if transforms is None:
        target_wh = (int(input_dim_rgb[0]), int(input_dim_rgb[1]))
        fill_rgb = float_rgb_to_uint8(mean)
        per_kind: dict[str, T.Callable] = {}
        # Photometric jittering is handled in TrainerConfig, so loading is deterministic.
        per_kind["rgb"] = lambda img: hwc_to_chw_float_tensor(
            fit_to_size(img, target_wh, fit_mode, fill_rgb)
        )
        transforms = {split: per_kind for split in split_names}

    out = {}
    for split in split_names:
        out[split] = IRAPDataset(
            dataset_dir,
            split,
            metadata=metadata,
            transforms=transforms.get(split)
            if isinstance(transforms, dict)
            else transforms,
            context_offsets=tuple(context_offsets),
            mean=tuple(float(x) for x in mean),
            std=tuple(float(x) for x in std),
            # Pickle-driven N-context filtering doesn't cover unlabeled segments.
            ncontext_segment_id_subset=(
                None if is_unlabeled_split(split) else ncontext_segment_id_subset
            ),
            allow_missing_attributes=allow_missing_attributes,
            dataset_name=dataset_name,
        )
    return LazyDict(
        out
    )  # Not (yet) lazy actually. Used only for attribute access syntax.


def make_bh_data(
    *,
    dataset_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    use_ncontext_filter: bool = DATASET_PRESETS["bh"].use_ncontext_filter,
    allow_missing_attributes: bool = DATASET_PRESETS["bh"].allow_missing_attributes,
    **kwargs,
):
    """Builds iRAP-BH dataset splits.

    Convenience wrapper around `make_irap_data` with default directories resolved
    from `IRAP_HOME` (`IRAP_BIH` and `IRAP_BIH_METADATA`) and the defaults of
    `DATASET_PRESETS["bh"]`.
    """
    dataset_dir, metadata_dir = resolve_irap_paths(
        dataset_dir=dataset_dir, metadata_dir=metadata_dir
    )
    return make_irap_data(
        dataset_dir=dataset_dir,
        metadata_dir=metadata_dir,
        use_ncontext_filter=use_ncontext_filter,
        allow_missing_attributes=allow_missing_attributes,
        dataset_name="bh",
        **kwargs,
    )


def make_bih_data(**kwargs):
    """Deprecated: use `make_bh_data`."""
    warnings.warn(
        "make_bih_data is deprecated. Use make_bh_data.",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_bh_data(**kwargs)


def make_vietnam_data(
    *,
    dataset_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    use_ncontext_filter: bool = DATASET_PRESETS["vietnam"].use_ncontext_filter,
    allow_missing_attributes: bool = DATASET_PRESETS[
        "vietnam"
    ].allow_missing_attributes,
    **kwargs,
):
    """Builds iRAP-Vietnam dataset splits.

    Convenience wrapper around `make_irap_data` with default directory resolved
    from `IRAP_HOME` (`IRAP_Vietnam`), colocated metadata, and the defaults of
    `DATASET_PRESETS["vietnam"]`.

    Args:
        dataset_dir: Dataset root directory. If None, resolved from environment.
        metadata_dir: None or `dataset_dir`, as iRAP-Vietnam keeps its metadata in the dataset
            directory. It is a parameter for the common interface of `IRAP_DATASET_FACTORIES`.
        use_ncontext_filter: Whether to apply N-context filtering.
        allow_missing_attributes: Whether to retain segments with missing attribute codes
            by assigning `IGNORE_LABEL_INDEX`.
        **kwargs: Additional keyword arguments forwarded to `make_irap_data`.

    Returns:
        LazyDict mapping split names to `IRAPDataset` instances.
    """
    if dataset_dir is None:
        dataset_dir = get_default_irap_paths("vietnam")[0]
    if metadata_dir is not None and Path(metadata_dir) != Path(dataset_dir):
        raise ValueError(
            "metadata_dir should not be passed explicitly or should be the same as dataset_dir"
        )

    data = make_irap_data(
        dataset_dir=dataset_dir,
        metadata_dir=dataset_dir,
        use_ncontext_filter=use_ncontext_filter,
        allow_missing_attributes=allow_missing_attributes,
        dataset_name="vietnam",
        **kwargs,
    )

    if "test" in data and len(data["test"]) == 0:
        print(
            "Warning: The 'test' split of the iRAP Vietnam dataset is empty and will be removed from the returned dataset dict."
        )
        del data["test"]

    return data


# Registry of iRAP release presets. Single source of truth for the dataset-name
# strings used by CLI tools (vlm_inference, agent_classify, ...).
IRAP_DATASET_FACTORIES = {
    "bh": make_bh_data,
    "vietnam": make_vietnam_data,
}


def make_irap_data_by_name(name: str, **kwargs):
    """Builds an iRAP dataset dict by release name (``"bh"`` / ``"vietnam"``).

    Thin dispatch over :data:`IRAP_DATASET_FACTORIES`; forwards ``kwargs`` to the
    selected preset. Raises ``ValueError`` for an unknown name.
    """
    try:
        factory = IRAP_DATASET_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown iRAP dataset {name!r}. Choose from {sorted(IRAP_DATASET_FACTORIES)}."
        ) from None
    return factory(**kwargs)
