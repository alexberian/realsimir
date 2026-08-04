"""realsimir -- realistic IR imagery of ships from synthetic renders.

See idea.txt for the algorithm and the todo list this package is built against.

Layout:

    paths.py       where the data and the submodules live
    boxes.py       BBox / ShipCrop geometry -- the shared vocabulary
    imaging.py     image loading and dtype adapters
    detectors/     bounding box back ends, selected by name (step 05a)
    ship_cropper.py  ShipCropper: detector + crop policy (step 05b)

The bounding box model is a value, not a hard-coded class:

    from realsimir import ShipCropper
    crops = ShipCropper("yolov3", conf_thresh=0.3)("/path/to/frame.jpg")

`realsimir/detectors/__init__.py` documents how to add another one.
"""

from __future__ import annotations

from .boxes import BBox, ShipCrop
from .detectors import (
    FunctionDetector,
    ShipDetector,
    available_detectors,
    build_detector,
    check_detector,
    describe_detectors,
    register_detector,
)
from .imaging import load_image, to_uint8_rgb
from .paths import ARETE_ROOT, EDM2_DIR, OPEN_IR_ROOT, REPO_ROOT, YOLO_DIR
from .ship_cropper import ShipCropper

__all__ = [
    # geometry
    "BBox",
    "ShipCrop",
    # cropping
    "ShipCropper",
    # detectors
    "ShipDetector",
    "FunctionDetector",
    "build_detector",
    "register_detector",
    "available_detectors",
    "describe_detectors",
    "check_detector",
    "AreteMetadataDetector",
    "PrecomputedDetector",
    "YoloShipDetector",
    "dump_detections",
    # io / paths
    "load_image",
    "to_uint8_rgb",
    "ARETE_ROOT",
    "OPEN_IR_ROOT",
    "REPO_ROOT",
    "YOLO_DIR",
    "EDM2_DIR",
]

_LAZY = {
    "YoloShipDetector": ".detectors.yolov3",
    "AreteMetadataDetector": ".detectors.arete",
    "PrecomputedDetector": ".detectors.precomputed",
    "dump_detections": ".detectors.precomputed",
}


def __getattr__(name: str):
    """Concrete detector classes stay importable from the top level, but are only
    imported when actually named -- `import realsimir` should not cost a torch
    import for code that only needs boxes or the metadata detector."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
