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

Two detector back ends are provided:

  YoloShipDetector       pytorch-yolo-v3 with COCO weights, filtered to the
                         'boat' class.  For the unlabelled real IR imagery in
                         /workspace/data/open_ir_images -- no labels needed,
                         which is the point.
  AreteMetadataDetector  the ground-truth boxes that ship alongside the synthetic
                         renders in /workspace/data/arete_ir_sim_data.  No
                         network, no failure modes; use it on the sim side.

Both satisfy the ShipDetector interface, so ShipCropper does not care which one
it is holding.
"""

from __future__ import annotations

import glob
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = REPO_ROOT / "pytorch-yolo-v3"

ARETE_ROOT = Path("/workspace/data/arete_ir_sim_data")


# --------------------------------------------------------------------------- #
# boxes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in source-image pixels.  Max coordinates are exclusive."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float = 1.0
    label: str = "ship"

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def expanded(self, frac: float) -> "BBox":
        """Grow each side by `frac` of its length (0.2 -> box 20% larger)."""
        dx, dy = self.width * frac / 2.0, self.height * frac / 2.0
        return replace(
            self,
            x_min=self.x_min - dx,
            y_min=self.y_min - dy,
            x_max=self.x_max + dx,
            y_max=self.y_max + dy,
        )

    def shifted(self, dx: float, dy: float) -> "BBox":
        return replace(
            self,
            x_min=self.x_min + dx,
            y_min=self.y_min + dy,
            x_max=self.x_max + dx,
            y_max=self.y_max + dy,
        )

    def scaled(self, s: float) -> "BBox":
        return replace(
            self, x_min=self.x_min * s, y_min=self.y_min * s, x_max=self.x_max * s, y_max=self.y_max * s
        )

    def clipped(self, width: int, height: int) -> "BBox":
        return replace(
            self,
            x_min=float(np.clip(self.x_min, 0, width)),
            y_min=float(np.clip(self.y_min, 0, height)),
            x_max=float(np.clip(self.x_max, 0, width)),
            y_max=float(np.clip(self.y_max, 0, height)),
        )

    def rounded(self) -> "BBox":
        return replace(
            self,
            x_min=float(np.floor(self.x_min)),
            y_min=float(np.floor(self.y_min)),
            x_max=float(np.ceil(self.x_max)),
            y_max=float(np.ceil(self.y_max)),
        )

    def as_int(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) suitable for slicing; x1/y1 exclusive."""
        b = self.rounded()
        return int(b.x_min), int(b.y_min), int(b.x_max), int(b.y_max)

    def iou(self, other: "BBox") -> float:
        ix = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        iy = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class ShipCrop:
    """One extracted window plus the geometry needed to put it back."""

    patch: np.ndarray  # (out_size, out_size) or (out_size, out_size, C)
    window: BBox  # region of the source frame `patch` covers
    detection: BBox  # the raw ship box, in source-frame pixels
    paste_box: BBox  # detection grown by paste_context, in source-frame pixels
    scale: float  # patch pixels per source pixel (1.0 == native resolution)
    path: str | None = None

    def detection_in_patch(self) -> BBox:
        """The ship box expressed in `patch` coordinates."""
        return self.detection.shifted(-self.window.x_min, -self.window.y_min).scaled(self.scale)

    def paste_box_in_patch(self) -> BBox:
        return self.paste_box.shifted(-self.window.x_min, -self.window.y_min).scaled(self.scale)


# --------------------------------------------------------------------------- #
# image io
# --------------------------------------------------------------------------- #


def load_image(path: str | os.PathLike, gray: bool = True) -> np.ndarray:
    """Read an IR frame as uint8.  PIL, not cv2, because raysense paths are CJK."""
    with Image.open(path) as im:
        im = im.convert("L" if gray else "RGB")
        return np.asarray(im)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Whatever came in -> HxWx3 uint8, which is what darknet wants."""
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


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #


class ShipDetector(ABC):
    """Anything that can put boxes on ships.

    `path` is passed through for detectors that key off the filename rather than
    the pixels (AreteMetadataDetector); pixel-based detectors ignore it.
    """

    @abstractmethod
    def detect(self, image: np.ndarray | None, path: str | os.PathLike | None = None) -> list[BBox]:
        ...

    def detect_batch(
        self,
        images: Sequence[np.ndarray | None],
        paths: Sequence[str | os.PathLike | None] | None = None,
    ) -> list[list[BBox]]:
        paths = paths if paths is not None else [None] * len(images)
        return [self.detect(im, p) for im, p in zip(images, paths)]


class YoloShipDetector(ShipDetector):
    """pytorch-yolo-v3 restricted to the COCO classes that read as 'ship'.

    The upstream `util.write_results` returns an uninitialised tensor when a
    batch produces no detections (util.py:121 -- `output` is allocated but never
    written), which surfaces as phantom boxes with garbage scores.  We only take
    `Darknet` from that repo and do the decode / NMS here instead.
    """

    def __init__(
        self,
        cfg_path: str | os.PathLike | None = None,
        weights_path: str | os.PathLike | None = None,
        yolo_dir: str | os.PathLike | None = None,
        input_size: int = 416,
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        class_names: Iterable[str] = ("boat",),
        device: str | torch.device | None = None,
    ):
        self.yolo_dir = Path(yolo_dir) if yolo_dir is not None else YOLO_DIR
        self.cfg_path = Path(cfg_path) if cfg_path else self.yolo_dir / "cfg" / "yolov3.cfg"
        self.weights_path = Path(weights_path) if weights_path else self.yolo_dir / "yolov3.weights"
        if input_size % 32 or input_size <= 32:
            raise ValueError(f"input_size must be a multiple of 32 and > 32, got {input_size}")
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        Darknet, all_classes = self._import_darknet()
        self.classes = all_classes
        wanted = set(class_names)
        unknown = wanted - set(all_classes)
        if unknown:
            raise ValueError(f"not COCO classes: {sorted(unknown)}")
        self.keep_class_ids = torch.tensor(
            sorted(all_classes.index(c) for c in wanted), device=self.device
        )

        self.model = Darknet(str(self.cfg_path))
        self.model.load_weights(str(self.weights_path))
        self.model.net_info["height"] = str(input_size)
        self.model.to(self.device).eval()

    def _import_darknet(self):
        """darknet.py does `from util import *`, so its directory must be importable."""
        d = str(self.yolo_dir)
        if d not in sys.path:
            sys.path.insert(0, d)
        from darknet import Darknet  # noqa: E402  (path must be set first)
        from util import load_classes  # noqa: E402

        return Darknet, load_classes(str(self.yolo_dir / "data" / "coco.names"))

    # -- letterboxing ------------------------------------------------------- #

    def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple[float, float, int, int]]:
        h, w = rgb.shape[:2]
        n = self.input_size
        scale = min(n / w, n / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((n, n, 3), 128, dtype=np.uint8)
        left, top = (n - new_w) // 2, (n - new_h) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        # int() truncation above means new_w/w is the scale actually applied
        return canvas, (new_w / w, new_h / h, left, top)

    def _undo_letterbox(self, xyxy: torch.Tensor, geom: tuple[float, float, int, int]) -> torch.Tensor:
        sx, sy, left, top = geom
        out = xyxy.clone()
        out[:, [0, 2]] = (out[:, [0, 2]] - left) / sx
        out[:, [1, 3]] = (out[:, [1, 3]] - top) / sy
        return out

    # -- inference ---------------------------------------------------------- #

    @torch.no_grad()
    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        paths: Sequence[str | os.PathLike | None] | None = None,
        batch_size: int = 8,
    ) -> list[list[BBox]]:
        results: list[list[BBox]] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            tensors, geoms, shapes = [], [], []
            for im in chunk:
                rgb = to_uint8_rgb(im)
                canvas, geom = self._letterbox(rgb)
                tensors.append(torch.from_numpy(canvas.transpose(2, 0, 1).copy()))
                geoms.append(geom)
                shapes.append(rgb.shape[:2])
            batch = torch.stack(tensors).to(self.device).float().div_(255.0)
            raw = self.model(batch, self.device.type == "cuda")
            for i in range(len(chunk)):
                results.append(self._decode(raw[i], geoms[i], shapes[i]))
        return results

    def detect(self, image: np.ndarray, path: str | os.PathLike | None = None) -> list[BBox]:
        return self.detect_batch([image])[0]

    def _decode(
        self, pred: torch.Tensor, geom: tuple[float, float, int, int], shape: tuple[int, int]
    ) -> list[BBox]:
        """pred: (N, 5 + num_classes) with cx, cy, w, h in letterboxed pixels."""
        obj = pred[:, 4]
        cls_scores = pred[:, 5:][:, self.keep_class_ids]
        best_score, best_local = cls_scores.max(dim=1)
        score = obj * best_score
        keep = score > self.conf_thresh
        if not bool(keep.any()):
            return []

        pred, score, best_local = pred[keep], score[keep], best_local[keep]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        xyxy = self._undo_letterbox(xyxy, geom)

        class_ids = self.keep_class_ids[best_local]
        order = batched_nms(xyxy, score, class_ids, self.nms_thresh)
        xyxy, score, class_ids = xyxy[order], score[order], class_ids[order]

        img_h, img_w = shape
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, img_w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, img_h)

        boxes = []
        for (x0, y0, x1, y1), s, c in zip(xyxy.tolist(), score.tolist(), class_ids.tolist()):
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            boxes.append(BBox(x0, y0, x1, y1, score=s, label=self.classes[int(c)]))
        return boxes


class AreteMetadataDetector(ShipDetector):
    """Ground-truth boxes for the synthetic renders, read from the metadata CSVs.

    Layout is  <root>/<split>/images/<scene>/<modality>/<stem>.png  with the boxes
    in <root>/<split>/metadata/<scene>/*.csv, keyed by `image_name` == stem.  The
    LWIR columns carry a `_lwir` suffix; the unsuffixed ones belong to the EO
    ('color') render, which has a different resolution -- so the modality has to
    come from the path, not be assumed.

    CSV maxima are inclusive pixel indices; BBox maxima are exclusive, hence +1.
    """

    def __init__(self, root: str | os.PathLike = ARETE_ROOT, box: str = "tight", require_visible: bool = True):
        if box not in ("tight", "loose"):
            raise ValueError(f"box must be 'tight' or 'loose', got {box!r}")
        self.root = Path(root)
        self.box = box
        self.require_visible = require_visible
        self._cache: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _parse_path(path: Path) -> tuple[str, str, str, str]:
        """.../<split>/images/<scene>/<modality>/<stem>.png"""
        parts = path.parts
        try:
            i = len(parts) - 1 - parts[::-1].index("images")
        except ValueError as exc:
            raise ValueError(f"not an arete image path (no 'images' component): {path}") from exc
        if len(parts) < i + 4:
            raise ValueError(f"not an arete image path: {path}")
        return parts[i - 1], parts[i + 1], parts[i + 2], path.stem  # split, scene, modality, stem

    def _table(self, split: str, scene: str) -> dict:
        key = (split, scene)
        if key not in self._cache:
            import pandas as pd

            csvs = sorted(glob.glob(str(self.root / split / "metadata" / scene / "*.csv")))
            if not csvs:
                raise FileNotFoundError(f"no metadata csv for {split}/{scene} under {self.root}")
            # a few scenes (trawler_generic_0000) are split across several csvs
            df = pd.concat([pd.read_csv(c, low_memory=False) for c in csvs], ignore_index=True)
            # image_name is not a key in the benchmark split: 72/432 atlant_0000 rows
            # are re-renders of the same case under a later simulation_cycle, and only
            # one file per name exists on disk.  The boxes agree (71/72 exactly), so
            # take the latest render and move on.
            df = df.drop_duplicates(subset="image_name", keep="last")
            self._cache[key] = df.set_index("image_name").to_dict("index")
        return self._cache[key]

    def detect(self, image: np.ndarray | None, path: str | os.PathLike | None = None) -> list[BBox]:
        if path is None:
            raise ValueError("AreteMetadataDetector needs the image path to look up its metadata")
        path = Path(path)
        split, scene, modality, stem = self._parse_path(path)
        row = self._table(split, scene).get(stem)
        if row is None:
            return []

        suffix = "_lwir" if modality == "lwir" else ""
        visible = row.get(f"is_target_visible{suffix}")
        if self.require_visible and visible is not None and not bool(visible):
            return []

        try:
            x0 = float(row[f"x_min_{self.box}{suffix}"])
            x1 = float(row[f"x_max_{self.box}{suffix}"]) + 1.0
            y0 = float(row[f"y_min_{self.box}{suffix}"])
            y1 = float(row[f"y_max_{self.box}{suffix}"]) + 1.0
        except KeyError:
            return []
        if x1 - x0 < 1 or y1 - y0 < 1:
            return []
        return [BBox(x0, y0, x1, y1, score=1.0, label=str(row.get("category", "ship")))]


# --------------------------------------------------------------------------- #
# the cropper
# --------------------------------------------------------------------------- #


class ShipCropper:
    """Detector + crop policy: frame in, registered ship windows out.

    out_size       side of the emitted square patch, in pixels.
    paste_context  how much bigger than the ship the cut/paste box is
                   (0.15 -> 15% larger, the "jitter the box" step of idea.txt).
    window_context extra slack folded into the crop window on top of the paste
                   box; only bites when the ship is larger than out_size.
    pad_mode       numpy pad mode used when the window runs off the frame.

    A window is never smaller than out_size, so small ships (most of the arete
    renders) get a native-resolution 256x256 view -- scale == 1.0.  Big ships
    (msc_danit fills 444x242 px up close) get a window that is downscaled to fit,
    scale < 1.0, which the augmentation stage needs to know about.
    """

    def __init__(
        self,
        detector: ShipDetector,
        out_size: int = 256,
        paste_context: float = 0.15,
        window_context: float = 0.0,
        min_score: float = 0.0,
        max_crops: int | None = None,
        pad_mode: str = "reflect",
    ):
        self.detector = detector
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
        **detector_kwargs,
    ) -> list[list[ShipCrop]]:
        """crop_all over many frames, letting the detector batch its forward pass."""
        paths = list(paths) if paths is not None else [None] * len(images)
        per_image = self.detector.detect_batch(images, paths, **detector_kwargs)
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
