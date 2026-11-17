"""Image loading and preprocessing helpers shared by data-loading code."""

import typing as T

import cv2
import numpy as np
import torch

#: How `fit_to_size` fits an image to a target size. See `fit_to_size`.
FitMode = T.Literal["fit_width", "resize_width", "cover", "center_crop"]
DEFAULT_FIT_MODE: FitMode = "fit_width"


def load_image_cv2(path: str) -> np.ndarray:
    """Loads an image as an HWC uint8 RGB ndarray."""
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def center_crop(img_hwc: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """Center-crop an HWC ndarray to target_wh without interpolation.

    Raises ValueError if the source is smaller than the target in any dimension.
    """
    h, w = img_hwc.shape[:2]
    target_w, target_h = target_wh
    if w < target_w or h < target_h:
        raise ValueError(
            f"Source image ({w}×{h}) is smaller than target ({target_w}×{target_h}). "
            f"Call resize_to_cover first if the source size is not guaranteed."
        )
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return img_hwc[top:top + target_h, left:left + target_w]


def resize_to_cover(img_hwc: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """Resize an HWC ndarray so both dimensions meet or exceed target_wh.

    Preserves aspect ratio (resize-to-cover). Uses Lanczos resampling via PIL.
    Call center_crop afterwards to get the exact target size.
    """
    import math
    from PIL import Image

    h, w = img_hwc.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {(w, h)}")
    target_w, target_h = target_wh
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, int(math.ceil(w * scale)))
    new_h = max(target_h, int(math.ceil(h * scale)))
    if (new_w, new_h) == (w, h):
        return img_hwc
    pil = Image.fromarray(img_hwc)
    pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(pil)


def resize_to_width(img_hwc: np.ndarray, target_w: int) -> np.ndarray:
    """Resizes an HWC ndarray to width `target_w`, preserving the aspect ratio.

    Uses Lanczos resampling via PIL. An image that already has the width is returned unchanged.
    """
    from PIL import Image

    h, w = img_hwc.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {(w, h)}")
    if w == target_w:
        return img_hwc
    new_h = max(1, round(h * target_w / w))
    pil = Image.fromarray(img_hwc).resize((target_w, new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(pil)


def pad_or_crop_height(
    img_hwc: np.ndarray, target_h: int, fill_rgb: T.Sequence[int]
) -> np.ndarray:
    """Centers an HWC ndarray vertically in `target_h` rows.

    A shorter image gets rows of `fill_rgb` above and below, a taller one is center-cropped.
    The extra row of an odd difference goes to the bottom, as in `center_crop`.
    """
    h = img_hwc.shape[0]
    if h >= target_h:
        top = (h - target_h) // 2
        return img_hwc[top:top + target_h]
    top = (target_h - h) // 2
    out = np.empty((target_h, *img_hwc.shape[1:]), dtype=img_hwc.dtype)
    out[...] = np.asarray(fill_rgb, dtype=img_hwc.dtype)
    out[top:top + h] = img_hwc
    return out


def fit_to_size(
    img_hwc: np.ndarray,
    target_wh: tuple[int, int],
    mode: FitMode = DEFAULT_FIT_MODE,
    fill_rgb: T.Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    """Fits an HWC ndarray to a target size.

    Args:
        img_hwc: The image.
        target_wh: The target size (width, height).
        mode: How the image is fitted:
            - `"fit_width"`: resized to the target width, preserving the aspect ratio, and then
              padded with `fill_rgb` or center-cropped to the target height. The horizontal field
              of view stays complete, so releases with different aspect ratios keep the same
              horizontal field of view and the same output size.
            - `"resize_width"`: resized to the target width, preserving the aspect ratio. The
              output height depends on the aspect ratio, and the target height is not used.
            - `"cover"`: resized to cover the target size (`resize_to_cover`), and then
              center-cropped to it.
            - `"center_crop"`: center-cropped to the target size without resizing
              (`center_crop`).
        fill_rgb: The padding color of `"fit_width"`.
    """
    if mode == "fit_width":
        return pad_or_crop_height(resize_to_width(img_hwc, target_wh[0]), target_wh[1], fill_rgb)
    if mode == "resize_width":
        return resize_to_width(img_hwc, target_wh[0])
    if mode == "cover":
        return center_crop(resize_to_cover(img_hwc, target_wh), target_wh)
    if mode == "center_crop":
        return center_crop(img_hwc, target_wh)
    raise ValueError(f"Unknown fit mode {mode!r}. Choose from {T.get_args(FitMode)}.")


def float_rgb_to_uint8(rgb: T.Sequence[float]) -> tuple[int, ...]:
    """Converts an RGB color with channels in `[0, 1]` to a uint8 RGB color."""
    return tuple(int(round(float(c) * 255)) for c in rgb)


def hwc_to_chw_float_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    """HWC uint8 ndarray → CHW float32 tensor in [0, 1]."""
    return torch.from_numpy(img_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0)
