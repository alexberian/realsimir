"""Ship cropping -- todo step 05.

This is the front of the pipeline described in idea.txt: find the ship in an IR
frame and cut a fixed-size window around it.  Two things come out of it, and the
augmentation stage (step 06) needs both:

  * the *paste box* -- the ship's bounding box grown by 10-20%, i.e. the chunk
    that later gets lifted out, doctored (offset / rotated / re-gained / noised)
    and pasted back down;
  * the *crop window* -- a fixed 256x256 window centred on the same ship, taken
    at native resolution, which is what the diffusion model actually sees.  The
    window is cut identically from the doctored and the clean frame so the pair
    stays registered.

Which bounding box model finds the ship is a constructor argument, by name:

    ShipCropper("arete")                       # gt boxes for the sim renders
    ShipCropper("yolov3", conf_thresh=0.3)     # COCO 'boat' on the real frames
    ShipCropper({"name": "precomputed", "boxes_file": "boxes.json"})
    ShipCropper("mypkg.detectors:Yolov8ShipDetector", weights="ir_ships.pt")

Nothing in this file knows what a detector is made of; see realsimir/detectors/
for the contract and for how to add one.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import cv2
import numpy as np

from .boxes import BBox, ShipCrop
from .detectors import ShipDetector, build_detector
from .imaging import load_image

__all__ = ["ShipCropper"]


class ShipCropper:
    """Detector + crop policy: frame in, registered ship windows out.

    detector       a ShipDetector, or anything build_detector understands: a
                   registered name, a {'name': ..., ...} config, a
                   'module:ClassName' path.  Leftover keyword arguments go to
                   its constructor -- if one of them collides with a crop-policy
                   name below, pass the detector as a config dict instead.
    out_size       side of the emitted square patch, in pixels.
    paste_context  how much bigger than the ship the cut/paste box is
                   (0.15 -> 15% larger, the "jitter the box" step of idea.txt).
    window_context extra slack folded into the crop window on top of the paste
                   box; only bites when the ship is larger than out_size.
    min_score      drop detections below this confidence.  Model-dependent --
                   retune it when the detector changes.
    max_crops      keep at most this many ships per frame, best score first.
    pad_mode       numpy pad mode used when the window runs off the frame.

    A window is never smaller than out_size, so small ships (most of the arete
    renders) get a native-resolution 256x256 view -- scale == 1.0.  Big ships
    (msc_danit fills 444x242 px up close) get a window that is downscaled to fit,
    scale < 1.0, which the augmentation stage needs to know about.
    """

    def __init__(
        self,
        detector: ShipDetector | str | dict | Any = "yolov3",
        out_size: int = 256,
        paste_context: float = 0.15,
        window_context: float = 0.0,
        min_score: float = 0.0,
        max_crops: int | None = None,
        pad_mode: str = "reflect",
        **detector_kwargs,
    ):
        self.detector = build_detector(detector, **detector_kwargs)
        self.out_size = out_size
        self.paste_context = paste_context
        self.window_context = window_context
        self.min_score = min_score
        self.max_crops = max_crops
        self.pad_mode = pad_mode

    # -- geometry ----------------------------------------------------------- #

    def paste_box(self, detection: BBox) -> BBox:
        """The ship box grown by paste_context -- what step 06 lifts and re-lays."""
        return detection.expanded(self.paste_context)

    def window_for(self, detection: BBox) -> BBox:
        """Square, integer-sided crop window centred on the ship.

        The side is snapped to a whole number of pixels rather than derived by
        rounding the corners outwards: that keeps it exactly out_size for every
        ship small enough to fit, so those crops come out at scale == 1.0 and no
        resampling touches them.
        """
        box = self.paste_box(detection).expanded(self.window_context)
        side = int(np.ceil(max(box.width, box.height, self.out_size)))
        cx, cy = box.center
        x0, y0 = round(cx - side / 2.0), round(cy - side / 2.0)
        return BBox(x0, y0, x0 + side, y0 + side, score=detection.score, label=detection.label)

    # -- extraction --------------------------------------------------------- #

    def _extract(self, image: np.ndarray, window: BBox) -> np.ndarray:
        """Slice `window` out of `image`, padding wherever it leaves the frame."""
        h, w = image.shape[:2]
        x0, y0, x1, y1 = window.as_int()
        pad_left, pad_top = max(0, -x0), max(0, -y0)
        pad_right, pad_bottom = max(0, x1 - w), max(0, y1 - h)

        patch = image[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)]
        if pad_left or pad_right or pad_top or pad_bottom:
            pads = [(pad_top, pad_bottom), (pad_left, pad_right)] + [(0, 0)] * (image.ndim - 2)
            mode = self.pad_mode
            # reflect needs something to reflect off; fall back when the slice is thin
            if mode in ("reflect", "symmetric") and min(patch.shape[:2]) < 2:
                mode = "edge"
            patch = np.pad(patch, pads, mode=mode)
        return patch

    def crop(self, image: np.ndarray, detection: BBox, path: str | os.PathLike | None = None) -> ShipCrop:
        window = self.window_for(detection)
        patch = self._extract(image, window)

        side = patch.shape[0]
        scale = 1.0
        if side != self.out_size:
            interp = cv2.INTER_AREA if side > self.out_size else cv2.INTER_LINEAR
            patch = cv2.resize(patch, (self.out_size, self.out_size), interpolation=interp)
            scale = self.out_size / side

        return ShipCrop(
            patch=patch,
            window=window,
            detection=detection,
            paste_box=self.paste_box(detection),
            scale=scale,
            path=str(path) if path is not None else None,
        )

    # -- driving ------------------------------------------------------------ #

    def _select(self, boxes: list[BBox]) -> list[BBox]:
        boxes = [b for b in boxes if b.score >= self.min_score]
        boxes.sort(key=lambda b: (b.score, b.area), reverse=True)
        return boxes[: self.max_crops] if self.max_crops else boxes

    def detect(self, image: np.ndarray | None, path: str | os.PathLike | None = None) -> list[BBox]:
        return self._select(self.detector.detect(image, path))

    def crop_all(self, image: np.ndarray, path: str | os.PathLike | None = None) -> list[ShipCrop]:
        return [self.crop(image, b, path) for b in self.detect(image, path)]

    def crop_batch(
        self,
        images: Sequence[np.ndarray],
        paths: Sequence[str | os.PathLike | None] | None = None,
    ) -> list[list[ShipCrop]]:
        """crop_all over many frames, letting the detector batch its forward pass."""
        paths = list(paths) if paths is not None else [None] * len(images)
        per_image = self.detector.detect_batch(images, paths)
        return [
            [self.crop(im, b, p) for b in self._select(boxes)]
            for im, p, boxes in zip(images, paths, per_image)
        ]

    def __call__(self, source: str | os.PathLike | np.ndarray, path=None) -> list[ShipCrop]:
        """Convenience: takes a path or an array."""
        if isinstance(source, (str, os.PathLike)):
            path, image = source, load_image(source)
        else:
            image = source
        return self.crop_all(image, path)

    def __repr__(self) -> str:
        return (
            f"ShipCropper({self.detector!r}, out_size={self.out_size}, "
            f"paste_context={self.paste_context})"
        )
