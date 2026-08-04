"""pytorch-yolo-v3 (COCO weights) as a ship detector.

The default back end for the unlabelled real IR imagery in
/workspace/data/open_ir_images: COCO's 'boat' class fires on IR ship hulls often
enough to bootstrap training, and no labels are required, which is the point.

Only `Darknet` is taken from the submodule.  Its `util.write_results` returns an
uninitialised tensor when a batch produces no detections (util.py:121 -- `output`
is allocated but never written), which surfaces as phantom boxes with garbage
scores, so the decode / NMS happens here instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from ..boxes import BBox
from ..imaging import to_uint8_rgb
from ..paths import YOLO_DIR
from .base import ShipDetector
from .registry import register_detector

__all__ = ["YoloShipDetector"]


@register_detector("yolov3", "yolo")
class YoloShipDetector(ShipDetector):
    """COCO YOLOv3 restricted to the classes that read as 'ship'.

    class_names  which COCO classes count as a ship.  ('boat',) by default.
                 Note these are darknet's spellings from data/coco.names, not
                 torchvision's -- 'aeroplane', not 'airplane'.
    batch_size   frames per forward pass in `detect_batch`.  A constructor
                 argument, not a call argument, so the caller does not need to
                 know which model it is holding.
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
        batch_size: int = 8,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.yolo_dir = Path(yolo_dir) if yolo_dir is not None else YOLO_DIR
        self.cfg_path = Path(cfg_path) if cfg_path else self.yolo_dir / "cfg" / "yolov3.cfg"
        self.weights_path = Path(weights_path) if weights_path else self.yolo_dir / "yolov3.weights"
        if input_size % 32 or input_size <= 32:
            raise ValueError(f"input_size must be a multiple of 32 and > 32, got {input_size}")
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.batch_size = batch_size

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
    ) -> list[list[BBox]]:
        results: list[list[BBox]] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
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

        boxes = [
            BBox(x0, y0, x1, y1, score=s, label=self.classes[int(c)])
            for (x0, y0, x1, y1), s, c in zip(xyxy.tolist(), score.tolist(), class_ids.tolist())
        ]
        return self._finalize(boxes, shape)

    def __repr__(self) -> str:
        return (
            f"YoloShipDetector(input_size={self.input_size}, conf_thresh={self.conf_thresh}, "
            f"device={self.device})"
        )
