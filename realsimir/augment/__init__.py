"""Data augmentation -- todo step 06.

Turns a detector box into a training pair: the frame as it is (the target) and
the frame with the ship's rectangle lifted, spoiled and laid back down (the
input).  `augmenter.py` holds the object and the argument for why it is shaped
this way; `ops.py` holds the individual corruptions and the registry that lets
a config name them.

    from realsimir import ShipAugmenter, ShipCropper

    cropper, aug = ShipCropper("arete"), ShipAugmenter()
    pair = aug(cropper(path)[0], seed=0)
    pair.doctored, pair.clean, pair.mask

    ShipAugmenter(ops={"shift": {"max_px": 4}, "blur": None})   # tune / drop
    ShipAugmenter(ops=[*DEFAULT_OPS, MyOwnCorruption()])        # extend

`python -m realsimir.augment list` prints the catalogue, `... check` runs the
invariants (clean and doctored agree outside the rectangle, a seed replays
exactly) against whichever ops are configured.  Figures: `python main.py`
writes 06/07/08 to ./figures.
"""

from __future__ import annotations

from .augmenter import AugmentedPair, ShipAugmenter
from .check import check_augmenter
from .ops import (
    DEFAULT_OPS,
    Blur,
    Corruption,
    Gain,
    Gamma,
    GeometricCorruption,
    Level,
    Noise,
    OpContext,
    PhotometricCorruption,
    Rescale,
    Rotate,
    Shift,
    Stripe,
    available_ops,
    build_op,
    build_ops,
    describe_ops,
    register_op,
)

__all__ = [
    "ShipAugmenter",
    "AugmentedPair",
    "check_augmenter",
    "Corruption",
    "GeometricCorruption",
    "PhotometricCorruption",
    "OpContext",
    "DEFAULT_OPS",
    "register_op",
    "build_op",
    "build_ops",
    "available_ops",
    "describe_ops",
    "Shift",
    "Rotate",
    "Rescale",
    "Gain",
    "Level",
    "Gamma",
    "Blur",
    "Noise",
    "Stripe",
]
