import pytest

from irap_evaluation.file_names import shorten_file_name, to_valid_file_name
from irap_evaluation.prediction_io import make_prediction_file_name

LONG_NAME = "trainer,epoch_count=3_model,encoder_f=partial(encoder,variant='large')" * 4


def test_to_valid_file_name_replaces_runs_of_other_characters():
    assert to_valid_file_name("Qwen/Qwen3-VL, v1.0") == "Qwen_Qwen3-VL_v1.0"


def test_shorten_file_name_keeps_a_name_that_fits():
    assert shorten_file_name("model", ".val.parquet") == "model.val.parquet"
    assert shorten_file_name("a" * 250, ".txt") == "a" * 250 + ".txt"


def test_shorten_file_name_cuts_the_name_and_keeps_the_suffix():
    shortened = shorten_file_name(LONG_NAME, ".unlabeled_train.parquet")
    assert len(shortened.encode()) == 255
    assert shortened.startswith(LONG_NAME[:100])
    assert shortened.endswith(".unlabeled_train.parquet")


def test_shorten_file_name_cuts_a_name_the_same_way_whatever_the_suffix():
    def get_name_part(suffix):
        return shorten_file_name(LONG_NAME, suffix, max_bytes=100).removesuffix(suffix)

    hash_part = get_name_part(".a").rpartition("-")[2]
    assert get_name_part(".longer_suffix").endswith("-" + hash_part)


def test_shorten_file_name_keeps_different_names_different():
    assert shorten_file_name(LONG_NAME + "a") != shorten_file_name(LONG_NAME + "b")


def test_shorten_file_name_does_not_split_a_character():
    shortened = shorten_file_name("č" * 200, ".txt")
    assert len(shortened.encode()) <= 255
    assert shortened.startswith("č")
    shortened.encode().decode()  # valid UTF-8


def test_shorten_file_name_refuses_a_suffix_without_room_for_the_name():
    with pytest.raises(ValueError):
        shorten_file_name("name", "s" * 252)  # 256 bytes with the name, 261 with a hash


def test_prediction_file_name_has_the_method_the_seed_and_the_split():
    assert make_prediction_file_name("Qwen/Qwen3-VL", None, "val") \
        == "Qwen_Qwen3-VL.val.predictions.parquet"
    assert make_prediction_file_name("Qwen/Qwen3-VL", 3, "val") \
        == "Qwen_Qwen3-VL_seed3.val.predictions.parquet"


@pytest.mark.parametrize("seed", [None, 3])
def test_a_long_prediction_file_name_keeps_the_seed_and_the_split(seed):
    name = make_prediction_file_name(LONG_NAME, seed, "unlabeled_unlocated")
    assert len(name.encode()) == 255
    assert name.startswith("trainer_epoch_count")
    assert name.endswith(("" if seed is None else f"_seed{seed}")
                         + ".unlabeled_unlocated.predictions.parquet")
