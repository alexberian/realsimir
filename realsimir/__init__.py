"""realsimir -- realistic IR imagery of ships from synthetic renders.

See idea.txt for the algorithm and the todo list this package is built against.

Layout:

    paths.py       where the data and the submodules live
    boxes.py       BBox / ShipCrop geometry -- the shared vocabulary
    imaging.py     image loading and dtype adapters
    detectors/     bounding box back ends, selected by name (step 05a)
    ship_cropper.py  ShipCropper: detector + crop policy (step 05b)
    augment/       ShipAugmenter: clean/doctored training pairs (step 06)
    data/          ShipDataGenerator: the stream those two feed (step 07)

The bounding box model is a value, not a hard-coded class:

    from realsimir import ShipCropper
    crops = ShipCropper("yolov3", conf_thresh=0.3)("/path/to/frame.jpg")

`realsimir/detectors/__init__.py` documents how to add another one.

Front of the pipeline, end to end:

    from realsimir import ShipAugmenter, ShipCropper

    cropper, aug = ShipCropper("arete"), ShipAugmenter()
    for i, crop in enumerate(cropper(path)):
        pair = aug(crop, seed=i)        # pair.doctored -> model -> pair.clean

...which is what `realsimir.data` runs over a corpus, for as long as training
lasts:

    from realsimir.data import ShipDataGenerator

    gen = ShipDataGenerator("real", index_file="boxes/real.json")
    for pair in gen.stream():
        ...
"""

from __future__ import annotations

from .augment import AugmentedPair, ShipAugmenter, check_augmenter
from .boxes import BBox, ShipCrop
from .data import (
    Composite,
    CompositeGenerator,
    CropIndex,
    ShipDataGenerator,
    check_generator,
    resolve_paths,
)
from .detectors import (
    FunctionDetector,
    ShipDetector,
    available_detectors,
    build_detector,
    check_detector,
    describe_detectors,
    register_detector,
)
from .imaging import from_float01, load_image, to_float01, to_uint8_rgb
from .paths import ARETE_ROOT, EDM2_DIR, OPEN_IR_ROOT, REPO_ROOT, YOLO_DIR
from .ship_cropper import ShipCropper

__all__ = [
    # geometry
    "BBox",
    "ShipCrop",
    # cropping
    "ShipCropper",
    # augmentation
    "ShipAugmenter",
    "AugmentedPair",
    "check_augmenter",
    # the data stream
    "ShipDataGenerator",
    "CompositeGenerator",
    "Composite",
    "CropIndex",
    "check_generator",
    "resolve_paths",
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
    "to_float01",
    "from_float01",
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
