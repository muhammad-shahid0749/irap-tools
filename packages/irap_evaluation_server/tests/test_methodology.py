import dataclasses as dc

import pytest

from irap_evaluation_server.methodology import (
    COMPARISONS_SECTION,
    EVALUATION_SETS_SECTION,
    INTERVALS_SECTION,
    check_described_settings,
    compute_t_quantile,
    make_methodology_sections,
)
from irap_evaluation_server.method_ranking import ShownMetrics
from irap_evaluation_server.scoring import SCORING_SETTINGS


def test_t_quantiles_of_one_and_two_degrees_of_freedom():
    # Tables of Student's t-distribution, e.g. scipy.stats.t.ppf(0.975, df).
    assert compute_t_quantile(0.975, 1) == pytest.approx(12.7062, abs=1e-4)
    assert compute_t_quantile(0.975, 2) == pytest.approx(4.3027, abs=1e-4)
    assert compute_t_quantile(0.05, 2) == pytest.approx(-2.9200, abs=1e-4)


def test_linked_sections_exist():
    ids = [s.id for s in make_methodology_sections(SCORING_SETTINGS, ["vietnam", "bh"],
                                                   ShownMetrics.from_hidden(()))]
    assert len(set(ids)) == len(ids)
    assert {EVALUATION_SETS_SECTION, INTERVALS_SECTION, COMPARISONS_SECTION} <= set(ids)


def test_settings_that_the_text_does_not_describe_are_refused():
    with pytest.raises(ValueError, match="null policy"):
        check_described_settings(dc.replace(SCORING_SETTINGS, null_policy="exclude"))
