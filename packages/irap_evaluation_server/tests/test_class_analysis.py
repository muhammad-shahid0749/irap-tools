import re
import types

import numpy as np
import pytest
from irap_data.metadata import IGNORE_LABEL_INDEX

from irap_evaluation_server.class_analysis import (
    ClassDiscrimination,
    Confusion,
    compute_class_discrimination,
    list_class_scores,
)
from irap_evaluation_server.components.analysis_html import make_class_analysis_html
from irap_evaluation_server.pages.analysis import _LoadedModel


def test_class_discrimination_of_probabilities_with_ties():
    labels = np.array([0, 0, 1, 1, IGNORE_LABEL_INDEX])
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.4, 0.6], [0.2, 0.8], [1.0, 0.0]])
    discrimination = compute_class_discrimination(labels, probs.argmax(1), probs)
    # Of the 4 pairs of a segment of the class and one of the other class, 3 are ordered and one
    # is tied. Each class has a recall of 1/2 or 1 and predicts 0 or 1 of the 2 other segments.
    # The unlabeled segment is left out.
    assert discrimination.auroc == pytest.approx([0.875, 0.875])
    assert discrimination.informedness == pytest.approx([0.5, 0.5])


def test_class_discrimination_of_predictions_that_ignore_the_input():
    rng = np.random.default_rng(0)
    labels = rng.choice(3, size=200, p=[0.8, 0.15, 0.05])
    # Always the same distribution, so always the first class, whatever the class sizes.
    probs = np.tile([0.7, 0.2, 0.1], (200, 1))
    discrimination = compute_class_discrimination(labels, np.zeros(200, int), probs)
    assert discrimination.auroc == pytest.approx([0.5] * 3)
    assert discrimination.informedness == pytest.approx([0] * 3)


def test_class_discrimination_of_invalid_and_hard_predictions():
    # The last prediction is invalid: the first class and the uniform distribution. Class 3 has
    # no labels.
    labels = np.array([0, 1, 2, 2])
    predicted_indices = np.array([0, 1, 2, IGNORE_LABEL_INDEX])
    probs = np.vstack([np.eye(4)[:3], np.full(4, np.nan)])
    discrimination = compute_class_discrimination(labels, predicted_indices, probs)
    np.testing.assert_allclose(discrimination.auroc, [1, 1, 1, np.nan])
    np.testing.assert_allclose(discrimination.informedness, [1 - 1 / 3, 1, 0.5, np.nan])

    # Without invalid predictions, the AUROC of hard predictions follows from the informedness.
    rng = np.random.default_rng(1)
    labels, predicted_indices = rng.integers(0, 3, 100), rng.integers(0, 3, 100)
    discrimination = compute_class_discrimination(labels, predicted_indices,
                                                  np.eye(3)[predicted_indices])
    assert discrimination.auroc == pytest.approx((1 + discrimination.informedness) / 2)

    # A class with all labeled segments has no other classes to be told from.
    only_class = compute_class_discrimination(np.zeros(3, int), np.zeros(3, int),
                                              np.tile([0.6, 0.4], (3, 1)))
    assert np.isnan(only_class.auroc).all() and np.isnan(only_class.informedness).all()


def make_class_report(confusion_matrix, f1):
    """Makes the class scores of an attribute with the classes 'c0', 'c1', … and the iRAP codes
    1, 2, …

    Args:
        confusion_matrix: Rows are labels, columns are predictions.
    """
    num_classes = len(confusion_matrix)
    return {"irap_codes": list(range(1, num_classes + 1)),
            "values": [f"c{k}" for k in range(num_classes)],
            "support": [sum(row) for row in confusion_matrix], "f1": f1,
            "confusion_matrix": confusion_matrix}


@pytest.fixture
def class_reports():
    return {
        "a": make_class_report([[80, 10, 0], [5, 2, 3], [0, 0, 0]], [0.9, 0.2, 0.0]),
        "b": make_class_report([[40, 10], [0, 50]], [0.8, 0.9]),
        "<c>": make_class_report([[30, 0, 0], [3, 3, 14], [10, 0, 40]], [0.7, 0.1, 0.6])}


@pytest.fixture
def attribute_to_discrimination():
    return {"a": ClassDiscrimination(np.array([0.9, 0.6, np.nan]), np.array([0.5, 0.1, np.nan])),
            "b": ClassDiscrimination(np.array([0.95, 0.95]), np.array([0.8, 0.8])),
            "<c>": ClassDiscrimination(np.array([0.8, 0.55, 0.7]), np.array([0.6, -0.05, 0.4]))}


def test_list_class_scores(class_reports, attribute_to_discrimination):
    class_scores = list_class_scores(class_reports, attribute_to_discrimination)
    # The classes with labels, in the order of the attributes.
    assert [(c.attribute, c.class_index) for c in class_scores] == [
        ("a", 0), ("a", 1), ("b", 0), ("b", 1), ("<c>", 0), ("<c>", 1), ("<c>", 2)]
    rare = class_scores[1]
    assert (rare.irap_code, rare.class_name, rare.support, rare.f1) == (2, "c1", 10, 0.2)
    assert (rare.auroc, rare.informedness) == (0.6, 0.1)
    # Mostly predicted as the first class, then as the unlabeled third one.
    assert (rare.num_missed, rare.largest_confusion) == (8, Confusion(1, "c0", 5))
    without_errors = class_scores[3]
    assert (without_errors.num_missed, without_errors.largest_confusion) == (0, None)


def test_unpredicted_attributes_are_not_listed(class_reports):
    labels = np.array([0, 0, 1])
    # Correct predictions with the number of classes of each attribute.
    attribute_to_probs = {a: np.eye(len(r["support"]))[labels] for a, r in class_reports.items()}
    attributes = tuple(attribute_to_probs)
    aligned = types.SimpleNamespace(
        evaluation_set=types.SimpleNamespace(class_indices=dict.fromkeys(attributes, labels)),
        predicted_indices=dict.fromkeys(attributes, labels),
        predictions=types.SimpleNamespace(probs=attribute_to_probs))
    scored = types.SimpleNamespace(scores=types.SimpleNamespace(missing_attributes=("b",)))
    loaded = _LoadedModel(scored, aligned, class_reports)
    assert {c.attribute for c in loaded.class_scores} == {"a", "<c>"}
    # The class tables of the confusion matrix show the values of every attribute.
    assert set(loaded.attribute_to_discrimination) == set(attributes)
    assert loaded.attribute_to_discrimination["b"].auroc == pytest.approx([1, 1])


def test_class_analysis_table_sorts_and_selects_the_classes(class_reports,
                                                            attribute_to_discrimination):
    class_scores = list_class_scores(class_reports, attribute_to_discrimination)

    def get_rows(class_sort, shown_indices=range(7), selected_index=None):
        table = make_class_analysis_html(class_scores, shown_indices, class_sort, selected_index)
        assert "<c>" not in table  # Escaped.
        return [(int(i), bool(selected))
                for i, selected in re.findall(r'<tr data-class-row="(\d+)"( class=selected)?>',
                                              table)]

    def get_order(class_sort, **kwargs):
        return [i for i, _ in get_rows(class_sort, **kwargs)]

    assert get_order("class") == list(range(7))
    assert get_order("-class") == list(range(6, -1, -1))
    assert get_order("f1") == [5, 1, 6, 4, 2, 0, 3]  # Worst first.
    assert get_order("support") == [0, 2, 3, 6, 4, 5, 1]  # Largest first, ties in their order.
    assert get_order("missed") == [5, 0, 2, 6, 1, 3, 4]
    assert get_order("auroc") == [5, 1, 6, 4, 0, 2, 3]  # Lowest first.
    assert get_order("informedness") == [5, 1, 6, 0, 4, 2, 3]
    # Only the shown classes, e.g. of the selected attributes, with the selected one marked.
    assert get_rows("f1", shown_indices=[0, 1, 5], selected_index=1) == [
        (5, False), (1, True), (0, False)]
