"""realsimir -- realistic IR imagery of ships from synthetic renders.

See idea.txt for the algorithm and the todo list this package is built against.
"""

from .ship_cropper import (
    ARETE_ROOT,
    AreteMetadataDetector,
    BBox,
    ShipCrop,
    ShipCropper,
    ShipDetector,
    YoloShipDetector,
    load_image,
    to_uint8_rgb,
)

__all__ = [
    "ARETE_ROOT",
    "AreteMetadataDetector",
    "BBox",
    "ShipCrop",
    "ShipCropper",
    "ShipDetector",
    "YoloShipDetector",
    "load_image",
    "to_uint8_rgb",
]
