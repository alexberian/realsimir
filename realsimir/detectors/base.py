"""The detector contract.

Everything the rest of the pipeline knows about a bounding box model is in this
file.  A new back end -- yolov8, DETR, a fine-tuned IR detector, hand labels --
only has to subclass `ShipDetector` and implement `detect`; `ShipCropper`, the
augmentation stage and the training loop never learn its name.

The contract, in full:

    detect(image, path=None) -> list[BBox]

  * `image` is a HxW (or HxWxC) numpy array as `load_image` returns it: uint8
    grayscale for our IR data.  Use `to_uint8_rgb` if the model wants 3x8-bit.
  * `path` is the source path when there is one, for back ends that key off the
    filename (ground-truth metadata, cached detections) rather than the pixels.
    Pixel-based models ignore it.  Either argument may be None, never both.
  * boxes come back in *source-image pixels* -- not letterboxed, not normalised,
    not in the model's input resolution.  Max coordinates are exclusive.
  * the input array must not be modified in place.
  * no detections is an empty list, never None.

Run `_finalize` on the way out and clipping, degenerate-box removal, label
renaming and score ordering are handled for you.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from ..boxes import BBox

__all__ = ["ShipDetector", "FunctionDetector"]


class ShipDetector(ABC):
    """Anything that can put boxes on ships.

    keep_labels    if given, only boxes whose label is in this set survive.
    rename_labels  applied after `keep_labels`; use it to fold a model's own
                   vocabulary onto ours ({'boat': 'ship'}), so downstream code
                   sees the same label no matter which back end produced it.
    min_side       boxes thinner than this (source pixels) are dropped -- a
                   sub-pixel box has nothing to crop around.
    """

    def __init__(
        self,
        keep_labels: Iterable[str] | None = None,
        rename_labels: Mapping[str, str] | None = None,
        min_side: float = 1.0,
    ):
        self.keep_labels = set(keep_labels) if keep_labels is not None else None
        self.rename_labels = dict(rename_labels or {})
        self.min_side = float(min_side)

    @abstractmethod
    def detect(self, image: np.ndarray | None, path: str | os.PathLike | None = None) -> list[BBox]:
        """Boxes for one frame, in source-image pixels."""

    def detect_batch(
        self,
        images: Sequence[np.ndarray | None],
        paths: Sequence[str | os.PathLike | None] | None = None,
    ) -> list[list[BBox]]:
        """Boxes for many frames.

        The default is a loop.  Override it if the back end can batch its
        forward pass -- any per-call batch size belongs in `__init__`, because
        callers only ever pass (images, paths).
        """
        paths = list(paths) if paths is not None else [None] * len(images)
        if len(paths) != len(images):
            raise ValueError(f"got {len(images)} images but {len(paths)} paths")
        return [self.detect(im, p) for im, p in zip(images, paths)]

    # -- for subclasses ------------------------------------------------------ #

    def _finalize(self, boxes: Iterable[BBox], shape: tuple[int, ...] | None = None) -> list[BBox]:
        """Common exit path: clip, drop degenerate, filter/rename, sort."""
        out: list[BBox] = []
        for b in boxes:
            if self.keep_labels is not None and b.label not in self.keep_labels:
                continue
            if shape is not None:
                b = b.clipped(int(shape[1]), int(shape[0]))
            if b.width < self.min_side or b.height < self.min_side:
                continue
            if b.label in self.rename_labels:
                b = BBox(b.x_min, b.y_min, b.x_max, b.y_max, b.score, self.rename_labels[b.label])
            out.append(b)
        out.sort(key=lambda b: (b.score, b.area), reverse=True)
        return out

    def __repr__(self) -> str:  # keeps run logs readable when the model is swapped
        return f"{type(self).__name__}()"


class FunctionDetector(ShipDetector):
    """Adapter for a model you already have in hand.

    The quickest way to try a different bounding box model without writing a
    class for it: wrap any `fn(image, path) -> boxes`, where each box is a BBox,
    a dict, or an (x0, y0, x1, y1[, score[, label]]) sequence.

        from torchvision.models import detection
        m = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").eval()
        det = FunctionDetector(lambda im, p: my_forward(m, im), rename_labels={"boat": "ship"})

    Fine for experiments and notebooks; once a back end earns its keep, give it
    a real module under realsimir/detectors/ and register it by name.
    """

    def __init__(self, fn: Callable[..., Iterable], takes_path: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.takes_path = takes_path

    def detect(self, image: np.ndarray | None, path: str | os.PathLike | None = None) -> list[BBox]:
        raw = self.fn(image, path) if self.takes_path else self.fn(image)
        boxes = [b if isinstance(b, BBox) else _coerce_box(b) for b in (raw or [])]
        shape = image.shape if image is not None else None
        return self._finalize(boxes, shape)

    def __repr__(self) -> str:
        return f"FunctionDetector({getattr(self.fn, '__name__', self.fn)!r})"


def _coerce_box(b) -> BBox:
    if isinstance(b, Mapping):
        return BBox.from_dict(dict(b))
    vals = list(b)
    if len(vals) < 4:
        raise ValueError(f"cannot read a box out of {b!r}")
    x0, y0, x1, y1 = (float(v) for v in vals[:4])
    score = float(vals[4]) if len(vals) > 4 else 1.0
    label = str(vals[5]) if len(vals) > 5 else "ship"
    return BBox(x0, y0, x1, y1, score, label)
