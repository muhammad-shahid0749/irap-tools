"""Visualization helpers of the dataset viewer app.

They need numpy, torch and OpenCV, which the `torch` extra provides.
"""

import typing as T

import cv2
import numpy as np
import torch

from irap_data.metadata import IGNORE_LABEL_INDEX


# Palette with ~25 distinct colors for good contrast on dark/light backgrounds.
# Index 0 is gray (often represents 'None' or 'Unknown').
_CLASS_COLOR_PALETTE = (
    "#808080",  # 0: Gray (None/Unknown)
    "#E6194B",  # 1: Red
    "#3CB44B",  # 2: Green
    "#FFE119",  # 3: Yellow
    "#4363D8",  # 4: Blue
    "#F58231",  # 5: Orange
    "#911EB4",  # 6: Purple
    "#42D4F4",  # 7: Cyan
    "#F032E6",  # 8: Magenta
    "#BFEF45",  # 9: Lime
    "#FABED4",  # 10: Pink
    "#469990",  # 11: Teal
    "#DCBEFF",  # 12: Lavender
    "#9A6324",  # 13: Brown
    "#FFFAC8",  # 14: Beige
    "#800000",  # 15: Maroon
    "#AAFFC3",  # 16: Mint
    "#808000",  # 17: Olive
    "#FFD8B1",  # 18: Apricot
    "#000075",  # 19: Navy
    "#A9A9A9",  # 20: Dark Gray
    "#E6BEFF",  # 21: Light Purple
    "#AA6E28",  # 22: Tan
    "#00FA9A",  # 23: Spring Green
    "#FF6347",  # 24: Tomato
)


def get_index_color(idx: int) -> str:
    """Hex color code for a class index, gray for 0 and negative indices."""
    if idx <= 0:
        return _CLASS_COLOR_PALETTE[0]
    return _CLASS_COLOR_PALETTE[1 + (idx - 1) % (len(_CLASS_COLOR_PALETTE) - 1)]


def format_class_value(values: T.Sequence[str], class_idx: int) -> str:
    """'<value> (<class_idx>)', with '(unlabeled)' as the value for `IGNORE_LABEL_INDEX`.

    Args:
        values: The values of the attribute, in class-index order.

    Raises:
        IndexError: For a class index outside `values`.
    """
    if class_idx == IGNORE_LABEL_INDEX:
        return f"(unlabeled) ({class_idx})"
    if not 0 <= class_idx < len(values):
        raise IndexError(f"Class index {class_idx} is not in [0, {len(values)}).")
    return f"{values[class_idx]} ({class_idx})"


def tensor_image_to_uint8_np(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) or (S, C, H, W) float tensor in [0, 1] -> uint8 HWC / SHWC array."""
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t.permute(0, 2, 3, 1)
    elif t.ndim == 3:
        t = t.permute(1, 2, 0)
    return (np.clip(t.numpy(), 0, 1) * 255).astype(np.uint8)


def create_composite_view_strip(images: np.ndarray) -> np.ndarray | None:
    """The first frame on top of a strip of the other frames, scaled to the width of the first.

    Args:
        images: (S, H, W, C) frames.

    Returns:
        The composite image, or None for no frames.
    """
    if len(images) == 0:
        return None

    main_img = images[0]
    if len(images) == 1:
        return main_img

    ctx_strip = np.concatenate(images[1:], axis=1)
    w_main = main_img.shape[1]
    h_ctx, w_ctx = ctx_strip.shape[:2]
    scale = w_main / w_ctx
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    ctx_resized = cv2.resize(ctx_strip, (w_main, int(h_ctx * scale)), interpolation=interp)
    return np.concatenate([main_img, ctx_resized], axis=0)
