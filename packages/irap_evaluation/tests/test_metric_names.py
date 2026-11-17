import pytest

from irap_evaluation.metric_names import (
    add_per_attribute_metric_names,
    describe_metric,
    get_attribute_average_name,
    get_irap_metric_names,
    is_lower_better,
    parse_metric_name,
)


def test_describe_metric_depends_on_ignore_missing_classes_only_where_it_applies():
    def is_affected(name: str) -> bool:
        return (describe_metric(name, ignore_missing_classes=True)
                != describe_metric(name, ignore_missing_classes=False))

    assert is_affected("mF1") and is_affected("amIoU")
    assert not any(map(is_affected, ["mF1_supp10", "nc_supp5", "mNLL", "F1", "MCC"]))


def test_add_per_attribute_metric_names():
    assert add_per_attribute_metric_names(["amF1", "aNLL", "mF1", "amF1_supp5"]) == (
        "amF1", "aNLL", "mF1", "amF1_supp5", "NLL", "mF1_supp5")


def test_parse_metric_name():
    assert parse_metric_name("amF1_supp10") == (True, "mF1_supp10", "mF1", 10)
    assert parse_metric_name("A") == (False, "A", "A", None)
    assert parse_metric_name("aA").is_averaged
    for name in ["nc", "A_supp5", "mF1_suppx", "xyz"]:
        with pytest.raises(ValueError):
            parse_metric_name(name)


def test_get_attribute_average_name():
    assert get_attribute_average_name("mF1_supp10") == "amF1_supp10"
    for name in ["amF1", "xyz"]:
        with pytest.raises(ValueError):
            get_attribute_average_name(name)


def test_is_lower_better():
    assert [n for n in get_irap_metric_names() if is_lower_better(n)] == [
        "aNLL", "aBrier", "amNLL", "amNLL_supp5", "amNLL_supp10", "NLL", "Brier", "mNLL"]
    assert is_lower_better("aMRP90")


def test_manual_review_proportion_metric_is_named_after_its_target():
    assert parse_metric_name("aMRP90") == (True, "MRP90", "MRP90", None)
    assert "mF1 reaches 0.9" in describe_metric("MRP90")


def test_get_irap_metric_names_puts_averages_first():
    names = get_irap_metric_names()
    assert names[:5] == ("amF1", "amP", "amR", "aMCC", "aA")
    assert "aNLL" not in get_irap_metric_names(output_kind="hard")
