import dataclasses as dc
import functools
import urllib.parse

import irap_evaluation as ie
import pytest
from run_helpers import add_model, make_model, score_and_get_models
from synthetic_vietnam import SPLIT

from irap_evaluation_server.components.analysis_view import AnalysisView, resolve_view
from irap_evaluation_server.components.attribute_filter import AttributeSubset
from irap_evaluation_server.method_scores import get_model_evaluation_set


@pytest.fixture
def models(archive, dataset_contexts, worker):
    """Two models scored on the reference set."""
    added = [add_model(archive, dataset_contexts, make_model(dataset_contexts, name))
             for name in ("a", "b")]
    return score_and_get_models(worker, [s.id for s in added])


def _resolve(view, models, dataset_contexts, attributes=()):
    subset = AttributeSubset.from_query(attributes, models[0].scores.attributes)
    return resolve_view(view, models, subset,
                        functools.partial(get_model_evaluation_set, dataset_contexts["vietnam"]))


def test_path_and_query_round_trip(dataset_contexts):
    view = AnalysisView("vietnam", SPLIT, ie.MODEL_COMPATIBLE_SET_NAME, model_id=3,
                        compare_id=4, attribute_name="Lane width", cell=(1, 3),
                        segment_id="0004", panel="classes", class_sort="-f1")
    path = view.to_path(["Lane width", "Curvature"])
    query = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
    assert query["attribute"] == ["Lane width", "Curvature"]
    assert AnalysisView.from_query({k: v[0] for k, v in query.items()}, dataset_contexts) == view
    missed_view = dc.replace(view, cell=(1, None))
    query = urllib.parse.parse_qs(urllib.parse.urlparse(missed_view.to_path()).query)
    assert (AnalysisView.from_query({k: v[0] for k, v in query.items()}, dataset_contexts)
            == missed_view)
    assert AnalysisView("vietnam", SPLIT).to_path() == f"/analysis?dataset=vietnam&split={SPLIT}"
    assert AnalysisView.from_query({}, dataset_contexts) == AnalysisView("vietnam", SPLIT)
    for query, message in [({"split": "test"}, "not an analysis split"),
                           ({"set": "x"}, "Unknown evaluation set"),
                           ({"model": "x"}, "model number"), ({"cell": "1"}, "two indices"),
                           ({"cell": "1,x"}, "two indices"),
                           ({"panel": "x"}, "Unknown panel")]:
        with pytest.raises(ValueError, match=message):
            AnalysisView.from_query(query, dataset_contexts)


def test_resolve_view_refuses_what_is_not_in_the_models(models, dataset_contexts):
    a, b = models[0].submission.model.id, models[1].submission.model.id
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    resolved = _resolve(AnalysisView("vietnam", SPLIT, model_id=a, compare_id=b, cell=(2, 3),
                                     attribute_name="Lane width",
                                     segment_id=reference_set.segment_ids[0]),
                        models, dataset_contexts)
    assert resolved.model_b is models[1]
    _resolve(dc.replace(resolved.view, cell=(2, None)), models, dataset_contexts)
    for changes, message in [
            (dict(model_id=999), "has no scored file"),
            (dict(compare_id=a), "cannot be compared"),
            (dict(attribute_name="Nope"), "not scored on the attribute"),
            (dict(cell=(3, 0)), "outside the matrix"),  # 'Lane width' has 3 classes.
            (dict(cell=(0, 4)), "outside the matrix"),
            (dict(cell=(3, None)), "outside the matrix"),
            (dict(segment_id="x"), "not in the evaluation set")]:
        with pytest.raises(ValueError, match=message):
            _resolve(dc.replace(resolved.view, **changes), models, dataset_contexts)
    with pytest.raises(ValueError, match="not among the selected attributes"):
        _resolve(resolved.view, models, dataset_contexts, attributes=["Curvature"])
