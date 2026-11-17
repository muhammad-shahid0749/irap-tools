"""The metrics equal those of vidlu's training-time evaluation.

Skipped unless `vidlu` and `vidlu_irap_gaim` are importable, e.g. with the vidlu checkout on
`PYTHONPATH`.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from irap_data.metadata import IGNORE_LABEL_INDEX
from irap_evaluation.metric_names import get_irap_metric_names
from irap_evaluation.metrics import (
    compute_classification_statistics,
    compute_multi_attribute_metrics,
)

torch = pytest.importorskip("torch")
vidlu_irap_metrics = pytest.importorskip("vidlu_irap_gaim.metrics")
from vidlu.utils.collections import NameDict  # noqa: E402

CLASS_COUNTS = {"Number of lanes": 6, "Lane width": 3, "Curvature": 4, "Intersection type": 16,
                "Median Type": 15}


def _make_data(seed: int, num_examples: int = 600):
    """Random targets with missing labels and absent classes, and random distributions."""
    rng = np.random.default_rng(seed)
    targets, probs = {}, {}
    for attr, num_classes in CLASS_COUNTS.items():
        present = rng.choice(num_classes, size=max(1, num_classes // 2), replace=False)
        targets[attr] = np.where(rng.random(num_examples) < 0.1, IGNORE_LABEL_INDEX,
                                 rng.choice(present, size=num_examples))
        logits = rng.normal(size=(num_examples, num_classes)) * rng.uniform(0.5, 4)
        probs[attr] = torch.softmax(torch.from_numpy(logits), 1).float().numpy()
    return targets, probs


def _compute_vidlu_metrics(targets, outputs, output_kind):
    info = SimpleNamespace(attr_to_value_to_class_idx={a: {} for a in CLASS_COUNTS},
                           class_counts=tuple(CLASS_COUNTS.values()),
                           attr_to_num_labeled={a: 1 for a in CLASS_COUNTS})
    metrics = vidlu_irap_metrics.get_irap_metrics(
        SimpleNamespace(info=info), attrs_to_include=tuple(CLASS_COUNTS),
        output_kind=output_kind)
    metrics.update(NameDict(
        out=tuple(torch.from_numpy(outputs[a]) for a in CLASS_COUNTS),
        target=torch.from_numpy(np.stack([targets[a] for a in CLASS_COUNTS], 1))))
    return metrics.compute()


def _assert_metrics_equal(ours, vidlu):
    ours = {**ours.averages, **ours.per_attribute}
    assert list(ours) == list(vidlu)
    for name, value in vidlu.items():
        if isinstance(value, dict):
            assert list(ours[name]) == list(value), name
            for attr, attr_value in value.items():
                np.testing.assert_allclose(ours[name][attr], attr_value, rtol=2e-6, atol=1e-7,
                                           equal_nan=True, err_msg=f"{name} {attr}")
        else:
            np.testing.assert_allclose(ours[name], value, rtol=2e-6, atol=1e-7, equal_nan=True,
                                       err_msg=name)


@pytest.mark.parametrize("output_kind", ["probs", "hard"])
def test_metric_names_match(output_kind):
    assert (get_irap_metric_names(output_kind=output_kind)
            == vidlu_irap_metrics.get_irap_metric_names(output_kind=output_kind))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_probabilistic_metrics_match(seed):
    targets, probs = _make_data(seed)
    names = get_irap_metric_names(output_kind="probs")
    ours = compute_multi_attribute_metrics(
        {a: compute_classification_statistics(targets[a], probs[a].argmax(1), num_classes,
                                              probs=probs[a])
         for a, num_classes in CLASS_COUNTS.items()},
        names)
    _assert_metrics_equal(ours, _compute_vidlu_metrics(targets, probs, "probs"))


def test_hard_metrics_match():
    targets, probs = _make_data(3)
    one_hot = {a: np.eye(p.shape[1], dtype=np.float32)[p.argmax(1)] for a, p in probs.items()}
    names = get_irap_metric_names(output_kind="hard")
    ours = compute_multi_attribute_metrics(
        {a: compute_classification_statistics(targets[a], probs[a].argmax(1), num_classes)
         for a, num_classes in CLASS_COUNTS.items()},
        names)
    _assert_metrics_equal(ours, _compute_vidlu_metrics(targets, one_hot, "hard"))
