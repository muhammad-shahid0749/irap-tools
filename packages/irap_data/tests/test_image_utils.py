import numpy as np
import pytest
import torch

from irap_data.image_utils import (
    center_crop,
    fit_to_size,
    float_rgb_to_uint8,
    hwc_to_chw_float_tensor,
    pad_or_crop_height,
    resize_to_cover,
    resize_to_width,
)


def _make_hwc(h, w, c=3, dtype=np.uint8):
    return np.arange(h * w * c, dtype=dtype).reshape(h, w, c)


# --- center_crop ---

def test_center_crop_exact_size():
    img = _make_hwc(288, 384)
    out = center_crop(img, (384, 288))
    assert out.shape == (288, 384, 3)
    np.testing.assert_array_equal(out, img)


def test_center_crop_larger_source():
    img = _make_hwc(300, 400)
    out = center_crop(img, (384, 288))
    assert out.shape == (288, 384, 3)
    # center pixel should be preserved
    out_cy, out_cx = 288 // 2, 384 // 2
    src_cy, src_cx = (300 - 288) // 2 + out_cy, (400 - 384) // 2 + out_cx
    np.testing.assert_array_equal(out[out_cy, out_cx], img[src_cy, src_cx])


def test_center_crop_error_on_smaller_source_width():
    img = _make_hwc(288, 300)
    with pytest.raises(ValueError, match="smaller than target"):
        center_crop(img, (384, 288))


def test_center_crop_error_on_smaller_source_height():
    img = _make_hwc(200, 384)
    with pytest.raises(ValueError, match="smaller than target"):
        center_crop(img, (384, 288))


# --- hwc_to_chw_float_tensor ---

def test_hwc_to_chw_float_tensor_shape():
    img = _make_hwc(288, 384)
    t = hwc_to_chw_float_tensor(img)
    assert t.shape == (3, 288, 384)


def test_hwc_to_chw_float_tensor_dtype_range():
    img = np.array([[[0, 128, 255]]], dtype=np.uint8)
    t = hwc_to_chw_float_tensor(img)
    assert t.dtype == torch.float32
    assert float(t[0, 0, 0]) == pytest.approx(0.0)
    assert float(t[2, 0, 0]) == pytest.approx(1.0)
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def test_hwc_to_chw_float_tensor_layout():
    img = _make_hwc(4, 5)
    t = hwc_to_chw_float_tensor(img)
    # channel 0 of tensor should match [:, :, 0] of source
    np.testing.assert_allclose(
        t[0].numpy(), img[:, :, 0].astype(np.float32) / 255.0
    )


# --- resize_to_cover ---

def test_resize_to_cover_already_large_enough():
    img = _make_hwc(288, 384)
    out = resize_to_cover(img, (384, 288))
    assert out.shape == (288, 384, 3)
    # no resize → same pixels
    np.testing.assert_array_equal(out, img)


def test_resize_to_cover_small_source():
    img = _make_hwc(100, 200)
    out = resize_to_cover(img, (384, 288))
    h, w = out.shape[:2]
    assert w >= 384 and h >= 288


def test_resize_to_cover_preserves_aspect():
    # 200×100 → cover 384×288: scale = max(384/200, 288/100) = max(1.92, 2.88) = 2.88
    # new size = ceil(200*2.88)=576, ceil(100*2.88)=288
    img = _make_hwc(100, 200)
    out = resize_to_cover(img, (384, 288))
    assert out.shape[0] == 288
    assert out.shape[1] >= 384


# --- resize_to_width / pad_or_crop_height ---

def test_resize_to_width_same_width_is_unchanged():
    img = _make_hwc(288, 384)
    assert resize_to_width(img, 384) is img


def test_resize_to_width_preserves_aspect():
    out = resize_to_width(_make_hwc(108, 192), 384)  # 16:9
    assert out.shape == (216, 384, 3)


def test_pad_or_crop_height_pads_centered_with_fill():
    img = np.full((216, 384, 3), 7, dtype=np.uint8)
    out = pad_or_crop_height(img, 288, fill_rgb=(1, 2, 3))
    assert out.shape == (288, 384, 3)
    np.testing.assert_array_equal(out[36:252], img)
    np.testing.assert_array_equal(out[:36], np.broadcast_to([1, 2, 3], (36, 384, 3)))
    np.testing.assert_array_equal(out[252:], np.broadcast_to([1, 2, 3], (36, 384, 3)))


def test_pad_or_crop_height_crops_centered():
    img = _make_hwc(288, 384)
    np.testing.assert_array_equal(pad_or_crop_height(img, 216, fill_rgb=(0, 0, 0)), img[36:252])


# --- fit_to_size ---

@pytest.mark.parametrize("mode", ["fit_width", "resize_width", "cover", "center_crop"])
def test_fit_to_size_bh_size_is_unchanged(mode):
    img = _make_hwc(288, 384)
    np.testing.assert_array_equal(fit_to_size(img, (384, 288), mode), img)


def test_fit_to_size_full_hd():
    img = _make_hwc(1080, 1920)
    fill = (136, 134, 124)
    out = fit_to_size(img, (384, 288), "fit_width", fill)
    assert out.shape == (288, 384, 3)
    np.testing.assert_array_equal(out[:36], np.broadcast_to(fill, (36, 384, 3)))
    assert fit_to_size(img, (384, 288), "resize_width").shape == (216, 384, 3)
    assert fit_to_size(img, (384, 288), "cover").shape == (288, 384, 3)
    np.testing.assert_array_equal(fit_to_size(img, (384, 288), "center_crop"),
                                  img[396:684, 768:1152])


def test_fit_to_size_custom_size_crops_height():
    out = fit_to_size(_make_hwc(288, 384), (384, 216), "fit_width")
    assert out.shape == (216, 384, 3)


def test_fit_to_size_unknown_mode():
    with pytest.raises(ValueError):
        fit_to_size(_make_hwc(288, 384), (384, 288), "stretch")


def test_float_rgb_to_uint8():
    assert float_rgb_to_uint8((0.0, 0.5, 1.0)) == (0, 128, 255)
