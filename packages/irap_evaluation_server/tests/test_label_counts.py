import irap_evaluation as ie
import numpy as np
from irap_data.metadata import IGNORE_LABEL_INDEX, ClassVocabulary
from run_helpers import (
    add_model,
    add_models_of_different_context_offsets,
    make_model,
    score_and_get_models,
)
from synthetic_vietnam import SPLIT

from irap_evaluation_server.components.attribute_filter import AttributeSubset
from irap_evaluation_server.label_counts import AttributeLabelCounts, count_attribute_labels
from irap_evaluation_server.method_ranking import ShownMetrics, collect_scored_attributes
from irap_evaluation_server.method_scores import find_common_evaluation_set
from irap_evaluation_server.pages.scores import (
    _compute_scores_content,
    _load_split_scorings,
    _ScoresView,
)


def test_count_attribute_labels():
    x = IGNORE_LABEL_INDEX
    class_indices = {"a": np.array([0, 2, 2, x, 2, 0, 2]),  # Class 1 does not occur.
                     "b": np.array([1, 1, 1, 1, 1, 1, x]),
                     "c": np.full(7, x)}
    vocabulary = ClassVocabulary({a: {f"{a} {code}": code for code in (1, 2, 3)}
                                  for a in class_indices})
    evaluation_set = ie.EvaluationSet(
        dataset="vietnam", split="val", name=ie.REFERENCE_SET_NAME, context_offsets=(0,),
        vocabulary=vocabulary, segment_ids=tuple(map(str, range(7))),
        class_indices=class_indices, segment_sequence_ids=("s",) * 7)
    assert count_attribute_labels(evaluation_set, ["b", "a", "c"]) == {
        "b": AttributeLabelCounts(num_labeled_segments=6, num_labeled_classes=1,
                                  num_minority_class_segments=0, num_smallest_class_segments=6),
        "a": AttributeLabelCounts(num_labeled_segments=6, num_labeled_classes=2,
                                  num_minority_class_segments=2, num_smallest_class_segments=2),
        "c": AttributeLabelCounts(num_labeled_segments=0, num_labeled_classes=0,
                                  num_minority_class_segments=0,
                                  num_smallest_class_segments=None)}


def test_label_counts_equal_those_of_the_score_reports(archive, dataset_contexts, store, worker):
    """The label counts of the evaluation set equal the numbers of examples (`n`) and the
    supports of the classes in the scores of every model, which count every labeled segment,
    since invalid predictions are scored as class 0 (`ScoringSettings.null_policy`)."""
    submissions = [add_model(archive, dataset_contexts,
                             make_model(dataset_contexts, name, random_seed=seed))
                   for name, seed in (("m", 1), ("n", 2))]
    models = score_and_get_models(worker, [s.id for s in submissions])
    evaluation_set = find_common_evaluation_set(dataset_contexts["vietnam"], models)
    assert evaluation_set is dataset_contexts["vietnam"].get_reference_set(SPLIT)
    attributes = models[0].scores.attributes
    attribute_to_counts = count_attribute_labels(evaluation_set, attributes)
    for model in models:
        report = store.get_score_report(model.submission.id, ie.REFERENCE_SET_NAME)
        for attribute, counts in attribute_to_counts.items():
            supports = [s for s in report["classes"][attribute]["support"] if s > 0]
            assert counts == AttributeLabelCounts(
                num_labeled_segments=model.scores.metrics.per_attribute["n"][attribute],
                num_labeled_classes=len(supports),
                num_minority_class_segments=sum(supports) - max(supports),
                num_smallest_class_segments=min(supports))


def _make_per_attribute_table_html(worker, dataset_contexts, evaluation_set, shows_class_counts):
    """Makes the per-attribute table of mF1 as the Scores page does."""
    view = _ScoresView(dataset="vietnam", split=SPLIT, evaluation_set=evaluation_set,
                       sort_metric="amF1", metric_name="mF1")
    split_scorings = _load_split_scorings(worker, view)
    subset = AttributeSubset.from_query([], collect_scored_attributes(split_scorings.models))
    content = _compute_scores_content(
        worker, dataset_contexts["vietnam"], ShownMetrics.from_hidden(()).average_names, view,
        split_scorings, subset, shows_class_counts)
    return content.table.table_html


def _get_attribute_row_html(table_html, attribute):
    start = table_html.index(f"<tr><td>{attribute}</td>")
    return table_html[start:table_html.index("</tr>", start)]


def test_scores_page_shows_the_label_counts(archive, dataset_contexts, worker):
    submissions = add_models_of_different_context_offsets(archive, dataset_contexts)
    score_and_get_models(worker, [s.id for s in submissions])
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    attribute = "Curvature"
    counts = count_attribute_labels(reference_set, [attribute])[attribute]
    count_cells = (f'<td class="number">{counts.num_labeled_classes}</td>'
                   f'<td class="number">{counts.num_labeled_segments}</td>'
                   f'<td class="number">{counts.num_minority_class_segments}</td>'
                   f'<td class="number">{counts.num_smallest_class_segments}</td>')

    table_html = _make_per_attribute_table_html(worker, dataset_contexts, ie.REFERENCE_SET_NAME,
                                                shows_class_counts=True)
    assert count_cells in _get_attribute_row_html(table_html, attribute)

    # The counts that reveal the class distribution are hidden.
    table_html = _make_per_attribute_table_html(worker, dataset_contexts, ie.REFERENCE_SET_NAME,
                                                shows_class_counts=False)
    hidden_cells = (f'<td class="number">–</td><td class="number">{counts.num_labeled_segments}'
                    f'</td><td class="number">–</td><td class="number">–</td>')
    assert hidden_cells in _get_attribute_row_html(table_html, attribute)

    # The model-compatible sets of the methods differ.
    table_html = _make_per_attribute_table_html(
        worker, dataset_contexts, ie.MODEL_COMPATIBLE_SET_NAME, shows_class_counts=True)
    assert '<td class="number">–</td>' * 4 in _get_attribute_row_html(table_html, attribute)
