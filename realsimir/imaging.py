"""Image loading / dtype wrangling shared by the detectors and the cropper."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

__all__ = ["load_image", "to_uint8_rgb", "to_float01", "from_float01"]


def load_image(path: str | os.PathLike, gray: bool = True) -> np.ndarray:
    """Read an IR frame as uint8.  PIL, not cv2, because raysense paths are CJK."""
    with Image.open(path) as im:
        im = im.convert("L" if gray else "RGB")
        return np.asarray(im)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Whatever came in -> HxWx3 uint8.

    Most detectors pretrained on visible imagery (darknet included) want three
    8-bit channels, while IR frames arrive single-channel and sometimes 16-bit,
    so this is the standard adapter at the front of a `detect` implementation.
    """
    arr = image
    if arr.dtype != np.uint8:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        arr = np.zeros_like(arr, dtype=np.float32) if hi <= lo else (arr - lo) / (hi - lo) * 255.0
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr)


def to_float01(image: np.ndarray) -> np.ndarray:
    """Anything -> float32 nominally in [0, 1], by the *dtype's* full range.

    Integer frames are divided by their dtype maximum (255, 65535, ...) rather
    than by their own min/max: augmentation has to be able to say "raise the
    level by 0.05" and mean the same brightness step on every frame, which a
    per-image stretch would not give.  Float input is passed through untouched,
    on the assumption it is already in 0-1 -- see `from_float01` for the way
    back.
    """
    if np.issubdtype(image.dtype, np.floating):
        return np.asarray(image, dtype=np.float32)
    info = np.iinfo(image.dtype)
    return image.astype(np.float32) / float(info.max)


def from_float01(image: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    """Inverse of `to_float01`: back to `dtype`, clipped and rounded."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return image.astype(dtype)
    info = np.iinfo(dtype)
    return np.clip(np.rint(image * float(info.max)), info.min, info.max).astype(dtype)
