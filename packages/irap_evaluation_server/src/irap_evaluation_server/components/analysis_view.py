"""The view of the Analysis page: what it shows, as in its URL, and how the view is resolved
against the scored models of its dataset split."""

import dataclasses as dc
import typing as T

import irap_evaluation as ie

from ..datasets import DatasetContext
from ..method_scores import ScoredModel
from .attribute_filter import AttributeSubset
from .routes import ANALYSIS_PATH, ATTRIBUTE_PARAMETER, make_query_path
from .view_queries import parse_dataset, parse_evaluation_set, select_default_split


#: The panels of the Analysis page, of which one is shown: its key in the URL -> its title.
ANALYSIS_PANELS = {"matrix": "Confusion matrix", "classes": "Class analysis",
                   "curve": "Review curve"}
DEFAULT_ANALYSIS_PANEL = "matrix"
#: The row sort (`row_sorting`) of the classes of the Class analysis panel
#: (`analysis_html.make_class_analysis_html`) by default: the classes that the model tells
#: least well from the other classes of their attribute first, by their one-vs-rest AUROC.
DEFAULT_CLASS_SORT = "auroc"


@dc.dataclass(frozen=True)
class AnalysisView:
    """What the Analysis page shows, as in its URL.

    Attributes:
        model_id: Model A, whose submission of the split the confusion matrix is of, or
            None for the default, which `resolve_view` chooses.
        compare_id: Model B, which A is compared with, or None.
        attribute_name: The analysed attribute, or '' for the default, which `resolve_view`
            chooses.
        cell: (label index, predicted index) of the selected cell of the confusion matrix, with
            the predicted index K for invalid predictions, or None. The predicted index None
            selects the missed segments of the class: its row without the correct predictions.
        segment_id: The selected segment, or ''.
        panel: The shown panel, a key of `ANALYSIS_PANELS`.
        class_sort: The row sort of the classes of the Class analysis panel.
    """

    dataset: str
    split: str
    evaluation_set: str = ie.REFERENCE_SET_NAME
    model_id: int | None = None
    compare_id: int | None = None
    attribute_name: str = ""
    cell: tuple[int, int | None] | None = None
    segment_id: str = ""
    panel: str = DEFAULT_ANALYSIS_PANEL
    class_sort: str = DEFAULT_CLASS_SORT

    def to_path(self, attributes: T.Sequence[str] = ()) -> str:
        """Makes the URL path of the view, without its default values.

        Args:
            attributes: The attribute subset (`AttributeSubset.to_query`).
        """
        return make_query_path(ANALYSIS_PATH, {
            "dataset": self.dataset, "split": self.split,
            "set": "" if self.evaluation_set == ie.REFERENCE_SET_NAME else self.evaluation_set,
            "model": "" if self.model_id is None else str(self.model_id),
            "compare": "" if self.compare_id is None else str(self.compare_id),
            "attr": self.attribute_name,
            "cell": "" if self.cell is None else format_cell(self.cell),
            "segment": self.segment_id,
            "panel": "" if self.panel == DEFAULT_ANALYSIS_PANEL else self.panel,
            "class_sort": "" if self.class_sort == DEFAULT_CLASS_SORT else self.class_sort,
            ATTRIBUTE_PARAMETER: attributes})

    @classmethod
    def from_query(cls, query: T.Mapping[str, str],
                   dataset_contexts: T.Mapping[str, DatasetContext]) -> T.Self:
        """Parses the query parameters of `to_path`, and fills in the default dataset and split
        (`view_queries.select_default_split` of the analysis splits).

        Raises:
            ValueError: For an unknown dataset, evaluation set or panel, a dataset without
                analysis splits, a split that is not an analysis split, or a malformed model
                number or cell.
        """
        dataset = parse_dataset(query, dataset_contexts,
                                default=next(iter(list_analysis_datasets(dataset_contexts)), None))
        analysis_splits = dataset_contexts[dataset].config.analysis_splits
        if not analysis_splits:
            raise ValueError(f"The dataset {dataset!r} has no analysis splits (analysis_splits"
                             f" in the server configuration).")
        split = query.get("split") or select_default_split(analysis_splits)
        if split not in analysis_splits:
            raise ValueError(f"{split!r} is not an analysis split of {dataset!r}. Its analysis"
                             f" splits: {', '.join(analysis_splits)}.")
        panel = query.get("panel") or DEFAULT_ANALYSIS_PANEL
        if panel not in ANALYSIS_PANELS:
            raise ValueError(f"Unknown panel {panel!r}. Panels: {', '.join(ANALYSIS_PANELS)}.")
        return cls(dataset=dataset, split=split, evaluation_set=parse_evaluation_set(query),
                   model_id=_parse_model_id(query.get("model", ""), "model"),
                   compare_id=_parse_model_id(query.get("compare", ""), "compared model"),
                   attribute_name=query.get("attr", ""), cell=parse_cell(query.get("cell", "")),
                   segment_id=query.get("segment", ""), panel=panel,
                   class_sort=query.get("class_sort") or DEFAULT_CLASS_SORT)


def _parse_model_id(text: str, name: str) -> int | None:
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"The {name} must be a model number, got {text!r}.") from None


#: The predicted index of `AnalysisView.cell` in the URL that selects the missed segments of a
#: class.
_MISSED_CELL_TEXT = "missed"


def format_cell(cell: tuple[int, int | None]) -> str:
    """Formats `AnalysisView.cell` as in the URL, e.g. '2,3', or '2,missed' for the missed
    segments of a class."""
    label_index, predicted_index = cell
    return f"{label_index},{_MISSED_CELL_TEXT if predicted_index is None else predicted_index}"


def parse_cell(text: str) -> tuple[int, int | None] | None:
    """Parses the cell of `format_cell`, and '' as None.

    Raises:
        ValueError: If it is not two integers, or an integer and 'missed'.
    """
    if not text:
        return None
    try:
        label_text, predicted_text = text.split(",")
        return (int(label_text),
                None if predicted_text == _MISSED_CELL_TEXT else int(predicted_text))
    except ValueError:
        raise ValueError(f"A matrix cell is two indices, e.g. '2,3', or an index and"
                         f" '{_MISSED_CELL_TEXT}', got {text!r}.") from None


def list_analysis_datasets(dataset_contexts: T.Mapping[str, DatasetContext]) -> list[str]:
    """Lists the datasets that have analysis splits."""
    return [name for name, c in dataset_contexts.items() if c.config.analysis_splits]


def list_compared_models(models: T.Iterable[ScoredModel],
                         scored_model: ScoredModel) -> list[ScoredModel]:
    """Lists the models that can be compared with `scored_model`: the others scored on the same
    evaluation set."""
    fingerprint = scored_model.scores.evaluation_set_fingerprint
    return [m for m in models if m.submission.id != scored_model.submission.id
            and m.scores.evaluation_set_fingerprint == fingerprint]


def select_attribute_options(subset: AttributeSubset, scored_model: ScoredModel) -> list[str]:
    """Selects the attributes that can be analysed: those of the subset that the model is scored
    on."""
    return [a for a in subset.selected if a in scored_model.scores.attributes]


@dc.dataclass(frozen=True)
class ResolvedView:
    """A view with its models.

    Attributes:
        view: With model A and the attribute filled in.
        evaluation_set: The evaluation set of model A (and B).
    """

    view: AnalysisView
    model_a: ScoredModel
    model_b: ScoredModel | None
    evaluation_set: ie.EvaluationSet


def resolve_view(view: AnalysisView, models: T.Sequence[ScoredModel], subset: AttributeSubset,
                 get_evaluation_set: T.Callable[[ScoredModel], ie.EvaluationSet]
                 ) -> ResolvedView:
    """Chooses the defaults of a view and checks it against the models.

    Args:
        models: The scored models of the dataset split on the evaluation set of the view, at
            least one. By default, model A is the first.
        subset: The attribute subset of the page. By default, the attribute is the first of it
            that model A is scored on.
        get_evaluation_set: The evaluation set of a model
            (`method_scores.get_model_evaluation_set`).

    Raises:
        ValueError: If model A or B is not among the models, B cannot be compared with A, the
            attribute is not scored or not in the subset, or the cell or the segment is not in
            the matrix or the evaluation set.
    """
    model_a = (models[0] if view.model_id is None
               else next((m for m in models if m.submission.model.id == view.model_id), None))
    if model_a is None:
        raise ValueError(f"Model {view.model_id} has no scored file of"
                         f" {view.dataset}/{view.split} on this evaluation set (it may be"
                         f" deleted).")
    model_b = None
    if view.compare_id is not None:
        model_b = next((m for m in list_compared_models(models, model_a)
                        if m.submission.model.id == view.compare_id), None)
        if model_b is None:
            raise ValueError(f"Model {view.compare_id} cannot be compared with model"
                             f" {model_a.submission.model.id}: it is the same model, it is not"
                             f" scored, or its evaluation set differs.")
    evaluation_set = get_evaluation_set(model_a)
    options = select_attribute_options(subset, model_a)
    attribute = view.attribute_name or next(iter(options), "")
    errors = []
    if view.attribute_name and view.attribute_name not in model_a.scores.attributes:
        errors.append(f"The model is not scored on the attribute {view.attribute_name!r}.")
    elif attribute not in options:
        errors.append(f"The attribute {view.attribute_name!r} is not among the selected"
                      f" attributes." if attribute else
                      "The model is scored on none of the selected attributes.")
    elif view.cell is not None:
        num_classes = len(evaluation_set.vocabulary.get_irap_codes(attribute))
        label_index, predicted_index = view.cell
        if not (0 <= label_index < num_classes
                and (predicted_index is None or 0 <= predicted_index <= num_classes)):
            errors.append(f"The matrix cell {view.cell} is outside the matrix of {attribute!r}"
                          f" ({num_classes} classes).")
    if view.segment_id and view.segment_id not in evaluation_set.segment_ids:
        errors.append(f"The segment {view.segment_id!r} is not in the evaluation set.")
    if errors:
        raise ValueError(" ".join(errors))
    return ResolvedView(
        view=dc.replace(view, model_id=model_a.submission.model.id, attribute_name=attribute),
        model_a=model_a, model_b=model_b, evaluation_set=evaluation_set)
