"""Online data generation -- todo step 07.

Steps 05 and 06 built the two halves of one example: `ShipCropper` finds a ship
and cuts a 256x256 window around it, `ShipAugmenter` spoils the ship's rectangle
and hands back the (clean, doctored) pair.  This package is what runs them over
a corpus, for as long as training lasts.

    from realsimir.data import ShipDataGenerator, check_generator

    gen = ShipDataGenerator("real", detector="yolov3", index_file="boxes/real.json")
    check_generator(gen)                 # before trusting it
    pair = gen[0]                        # .doctored -> model -> .clean
    for pair in gen.stream():            # infinite; a new corruption every epoch
        ...

Four objects, each with its own file and its own argument:

    pools.py       where the frames are -- named sets, and how to sample a
                   million-file pool without walking it
    index.py       CropIndex: which ship in which frame is example i, detected
                   once and cached, because that is the expensive part and the
                   part that must not change mid-run
    generator.py   ShipDataGenerator: the stream itself.  Boxes from the index,
                   pixels from disk, corruption drawn per (seed, epoch, index)
    composite.py   CompositeGenerator: the test streams -- synthetic ship chunks
                   pasted into real frames (what the model is for, no target),
                   and `.self_paste()`, the same ships put back into their own
                   renders by the training recipe, which does have one

plus `torch_dataset.py` (Dataset / DataLoader adapters, the only file here that
imports torch) and `check.py` (the acceptance test to run before training).

    python -m realsimir.data pools
    python -m realsimir.data index --source real --out boxes/real.json
    python -m realsimir.data check --source real --index-file boxes/real.json
"""

from __future__ import annotations

from .check import check_generator
from .composite import (
    DEFAULT_MAX_ROTATION_DEG,
    Composite,
    CompositeGenerator,
    limit_rotation,
)
from .generator import ShipDataGenerator
from .index import CropIndex, CropRef
from .pools import (
    FramePool,
    available_pools,
    describe_pools,
    get_pool,
    register_pool,
    resolve_paths,
)

__all__ = [
    "ShipDataGenerator",
    "CompositeGenerator",
    "Composite",
    "limit_rotation",
    "DEFAULT_MAX_ROTATION_DEG",
    "CropIndex",
    "CropRef",
    "check_generator",
    "FramePool",
    "register_pool",
    "get_pool",
    "available_pools",
    "describe_pools",
    "resolve_paths",
    # torch-only, imported on demand
    "ShipPairDataset",
    "CompositeDataset",
    "make_dataloader",
    "to_tensor",
]

_LAZY = {
    "ShipPairDataset": ".torch_dataset",
    "CompositeDataset": ".torch_dataset",
    "make_dataloader": ".torch_dataset",
    "to_tensor": ".torch_dataset",
}


def __getattr__(name: str):
    """The torch adapters are importable from here, but only cost torch when used."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
