"""Bounding box back ends -- the swappable half of todo step 05.

YOLOv3-on-COCO is a placeholder: it was the model idea.txt happened to name, it
has never seen an IR image, and it will be replaced.  Everything here exists so
that replacing it is a contained change.

Pick one by name; nothing downstream needs editing:

    from realsimir import ShipCropper, build_detector

    cropper = ShipCropper("yolov3", conf_thresh=0.3)          # real IR frames
    cropper = ShipCropper("arete")                            # synthetic renders
    cropper = ShipCropper({"name": "precomputed", "boxes_file": "boxes.json"})

Registered back ends
--------------------
  yolov3       pytorch-yolo-v3 with COCO weights, filtered to the 'boat' class.
  arete        ground-truth boxes from the synthetic render metadata CSVs.
  precomputed  boxes from a JSON file -- any external model, or a cached pass.

`python -m realsimir.detectors list` prints the live registry.

Adding a new model
------------------
Three steps, none of which touch the cropper, the augmentation stage or the
training loop:

 1. Write `realsimir/detectors/<name>.py`::

        from .base import ShipDetector
        from .registry import register_detector
        from ..boxes import BBox
        from ..imaging import to_uint8_rgb

        @register_detector("yolov8", doc="ultralytics YOLOv8 fine-tuned on IR ships")
        class Yolov8ShipDetector(ShipDetector):
            def __init__(self, weights="yolov8n.pt", conf_thresh=0.25, **kwargs):
                super().__init__(**kwargs)          # keep_labels / rename_labels
                from ultralytics import YOLO
                self.model, self.conf_thresh = YOLO(weights), conf_thresh

            def detect(self, image, path=None):
                res = self.model(to_uint8_rgb(image), conf=self.conf_thresh, verbose=False)[0]
                boxes = [
                    BBox(*xyxy, score=float(c), label=res.names[int(k)])
                    for xyxy, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf, res.boxes.cls)
                ]
                return self._finalize(boxes, image.shape)   # clip / drop / rename / sort

    Boxes must come back in *source-image pixels* -- see base.py for the full
    contract.  Override `detect_batch` only if the model can batch; put the
    batch size in `__init__`, not in the call.

 2. Add a lazy registration line in this file (below), so the import cost only
    lands on people who ask for it.

 3. Verify it against the contract before training on it::

        python -m realsimir.detectors check yolov8 --image /workspace/data/.../frame.jpg
        python main.py --detector yolov8            # eyeball figures/01,02

If the new model cannot live in this conda environment -- a conflicting torch
pin, a separate repo, a licensed binary -- do not fight it: run it wherever it
works, dump its boxes with the schema in precomputed.py, and use
`--detector precomputed --detector-arg boxes_file=boxes.json`.  For a one-off
experiment on a model object you already have, `FunctionDetector` wraps a plain
callable and skips all of the above.
"""

from __future__ import annotations

from .base import FunctionDetector, ShipDetector
from .contract import check_detector, synthetic_frame
from .registry import (
    available_detectors,
    build_detector,
    describe_detectors,
    get_detector_class,
    register_detector,
    register_lazy,
)

# Built-ins, registered without importing them: `import realsimir` must not drag
# in torch just because yolov3 exists.  The module is imported on first build.
register_lazy(
    "yolov3",
    "realsimir.detectors.yolov3:YoloShipDetector",
    "yolo",
    doc="pytorch-yolo-v3 COCO weights, filtered to the 'boat' class",
)
register_lazy(
    "arete",
    "realsimir.detectors.arete:AreteMetadataDetector",
    "metadata",
    "gt",
    doc="ground-truth boxes from the arete sim metadata CSVs",
)
register_lazy(
    "precomputed",
    "realsimir.detectors.precomputed:PrecomputedDetector",
    "json",
    "cached",
    doc="boxes read from a JSON file produced by any model",
)

__all__ = [
    "ShipDetector",
    "FunctionDetector",
    "build_detector",
    "register_detector",
    "register_lazy",
    "available_detectors",
    "describe_detectors",
    "get_detector_class",
    "check_detector",
    "synthetic_frame",
]


def __getattr__(name: str):
    """Lazy re-export of the concrete classes, so `from realsimir.detectors import
    YoloShipDetector` still works without paying for torch at package import."""
    targets = {
        "YoloShipDetector": ".yolov3",
        "AreteMetadataDetector": ".arete",
        "PrecomputedDetector": ".precomputed",
        "dump_detections": ".precomputed",
    }
    if name in targets:
        import importlib

        return getattr(importlib.import_module(targets[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
