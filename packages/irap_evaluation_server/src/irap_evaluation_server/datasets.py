"""The configured datasets: their metadata and the evaluation sets of their splits."""

import dataclasses as dc
import threading
import typing as T
from pathlib import Path

import irap_evaluation as ie
from irap_data.metadata import IRAPMetadata, load_irap_metadata

from .config import DatasetConfig

#: The names of the evaluation sets that a submission can be scored on.
EVALUATION_SET_NAMES = (*ie.SHARED_EVALUATION_SET_NAMES, ie.MODEL_COMPATIBLE_SET_NAME)

_CachedValue = T.TypeVar("_CachedValue")


@dc.dataclass(eq=False)
class DatasetContext:
    """A configured dataset with its metadata and cached evaluation sets.

    The cache is shared by threads, e.g. those that check uploads and the scoring thread.

    Attributes:
        name: The dataset name, a key of `irap_data.DATASET_PRESETS`.
    """

    name: str
    config: DatasetConfig
    metadata: IRAPMetadata
    #: split -> its shared sets by name (`irap_evaluation.get_shared_evaluation_sets`).
    _shared_set_cache: dict[str, dict[str, ie.EvaluationSet]] = dc.field(
        default_factory=dict, init=False, repr=False)
    #: (split, context offsets) -> the model-compatible set.
    _model_compatible_set_cache: dict[tuple[str, tuple[int, ...]], ie.EvaluationSet] = dc.field(
        default_factory=dict, init=False, repr=False)
    #: Held while a set is computed, so that concurrent first uses compute it once. Cached sets
    #: are read without it, so that a page does not wait for the computation of another set.
    _cache_lock: threading.Lock = dc.field(default_factory=threading.Lock, init=False,
                                           repr=False)

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(self.metadata.splits)

    def _get_cached(self, cache: dict[T.Any, _CachedValue], key: T.Any, split: str,
                    compute: T.Callable[[], _CachedValue]) -> _CachedValue:
        """Returns `cache[key]`, computed with `compute` on first use.

        Raises:
            ValueError: If the dataset has no such split.
        """
        if (value := cache.get(key)) is not None:
            return value
        with self._cache_lock:
            if key not in cache:
                if split not in self.metadata.splits:
                    raise ValueError(f"The dataset {self.name!r} has no split {split!r}. Its"
                                     f" splits: {', '.join(self.split_names)}.")
                cache[key] = compute()
            return cache[key]

    def get_shared_evaluation_sets(self, split: str) -> dict[str, ie.EvaluationSet]:
        """Returns `irap_evaluation.get_shared_evaluation_sets` of a split, computed on first use.

        Raises:
            ValueError: If the dataset has no such split.
        """
        return self._get_cached(
            self._shared_set_cache, split, split,
            lambda: ie.get_shared_evaluation_sets(self.metadata, self.name, split))

    def get_reference_set(self, split: str) -> ie.EvaluationSet:
        return self.get_shared_evaluation_sets(split)[ie.REFERENCE_SET_NAME]

    def count_shared_evaluation_set_segments(self, split: str) -> dict[str, int]:
        """Returns the number of segments of each shared evaluation set of a split, by name."""
        return {n: s.num_segments for n, s in self.get_shared_evaluation_sets(split).items()}

    def get_model_compatible_set(self, split: str,
                                 context_offsets: T.Sequence[int]) -> ie.EvaluationSet:
        """Returns `irap_evaluation.get_model_compatible_set` of a split, computed on first use.

        Raises:
            ValueError: If the dataset has no such split.
        """
        context_offsets = tuple(context_offsets)
        return self._get_cached(
            self._model_compatible_set_cache, (split, context_offsets), split,
            lambda: ie.get_model_compatible_set(self.metadata, self.name, split,
                                                context_offsets))

    def get_submission_evaluation_sets(
            self, split: str, context_offsets: T.Sequence[int] | None
    ) -> dict[str, ie.EvaluationSet]:
        """Returns the evaluation sets that a submission of a split can be scored on, by name:
        the shared sets, and the model-compatible set if the context offsets are known.

        Raises:
            ValueError: If the dataset has no such split.
        """
        name_to_set = dict(self.get_shared_evaluation_sets(split))
        if context_offsets is not None:
            name_to_set[ie.MODEL_COMPATIBLE_SET_NAME] = self.get_model_compatible_set(
                split, context_offsets)
        return name_to_set

    def select_evaluation_sets(
            self, predictions: ie.Predictions) -> tuple[list[ie.EvaluationSet], list[str]]:
        """Applies `irap_evaluation.select_evaluation_sets` to predictions, with the shared sets
        of their split and the model-compatible set of the context offsets in their header.

        Raises:
            ValueError: If the predictions are of another dataset or of an unknown split, or
                `irap_evaluation.select_evaluation_sets` refuses them.
        """
        header = predictions.header
        if header.dataset != self.name:
            raise ValueError(f"The predictions are of the dataset {header.dataset!r},"
                             f" not {self.name!r}.")
        offsets = header.context_offsets
        return ie.select_evaluation_sets(
            predictions.segment_ids, self.get_shared_evaluation_sets(header.split).values(),
            None if offsets is None else self.get_model_compatible_set(header.split, offsets))

    def check_new_predictions(self, predictions: ie.Predictions) -> list[str]:
        """Checks the contents of predictions that are to be stored: the split names of their
        model (`irap_evaluation.check_split_names`), their attributes and classes
        (`check_predicted_classes`), and their evaluation sets (`select_evaluation_sets`).

        Returns:
            The notes of `select_evaluation_sets`, e.g. that the model is not scored on the
            reference set.

        Raises:
            irap_evaluation.PredictionFormatError: For an unknown split name, and see
                `check_predicted_classes`.
            ValueError: See `select_evaluation_sets`.
        """
        ie.check_split_names(predictions.header.model, self.split_names)
        self.check_predicted_classes(predictions)
        _, notes = self.select_evaluation_sets(predictions)
        return notes

    def find_segment_image(self, segment_id: str) -> Path | None:
        """Finds the RGB image of a segment in `config.dataset_dir`, None if the segment has no
        image, the file does not exist, or its path in the metadata leads outside the
        directory."""
        dataset_dir = self.config.dataset_dir
        relative = self.metadata.segment_id_to_data_paths_rel.get(segment_id, {}).get("rgb")
        if relative in (None, "NONE"):
            return None
        path = (dataset_dir / relative).resolve()
        return path if path.is_relative_to(dataset_dir.resolve()) and path.is_file() else None

    def check_predicted_classes(self, predictions: ie.Predictions) -> None:
        """Checks that the predicted attributes and their iRAP codes are those of the dataset.

        Raises:
            irap_evaluation.PredictionFormatError: If an attribute is unknown or its codes differ.
        """
        attribute_to_irap_codes = self.metadata.vocabulary.attribute_to_irap_codes
        if unknown := [a for a in predictions.attributes if a not in attribute_to_irap_codes]:
            raise ie.PredictionFormatError(f"Attributes unknown in the dataset {self.name!r}:"
                                           f" {unknown}.")
        ie.check_class_codes(predictions,
                             {a: attribute_to_irap_codes[a] for a in predictions.attributes})


def load_dataset_contexts(
        datasets: T.Mapping[str, DatasetConfig]) -> dict[str, DatasetContext]:
    """Loads the metadata of each dataset.

    Raises:
        ValueError: If a dataset directory does not exist, or an analysis split is not in the
            metadata.
    """
    contexts = {}
    for name, config in datasets.items():
        if not config.dataset_dir.is_dir():
            raise ValueError(f"Dataset {name!r}: the dataset directory {config.dataset_dir} does"
                             f" not exist.")
        metadata = load_irap_metadata(config.metadata_dir)
        if unknown := [s for s in config.analysis_splits if s not in metadata.splits]:
            raise ValueError(f"Dataset {name!r}: the analysis splits {unknown} are not in"
                             f" {config.metadata_dir}. Its splits: {', '.join(metadata.splits)}.")
        contexts[name] = DatasetContext(name=name, config=config, metadata=metadata)
    return contexts
