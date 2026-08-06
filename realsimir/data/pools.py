"""Where the frames come from -- the pools the generator draws on.

idea.txt names two halves of the data and one rule about them: the real IR
frames under /workspace/data/open_ir_images are what the model is trained to
reproduce, the synthetic renders under /workspace/data/arete_ir_sim_data are
what it will be asked to fix at test time, and nothing may be written to
either.  A *pool* is a name for one of those sets, so a config -- step 09's
training config in particular -- can carry "raysense" instead of a glob nobody
can read:

    from realsimir.data import resolve_paths, describe_pools

    paths = resolve_paths("real")                    # every real IR frame
    paths = resolve_paths("arete", limit=4000, seed=0)
    paths = resolve_paths("/my/frames/*.png")        # or just a pattern

Size is the reason this file exists rather than a constant in the generator.
The real pools are small enough to enumerate on every run -- 9402 raysense +
2916 MassMIND frames -- but the synthetic one is not: full_ir holds 1,088,640
renders across 27 scenes, 240 GB.  A training run wants a *sample* of that, and
the sample has to be seeded rather than "the first N", because the renders are
ordered by scene and then by simulation time: the first N is one ship at one
range in one atmosphere.  `stride` is the other half of the same problem -- both
pools are frame-grabbed from continuous sequences, so consecutive files are near
duplicates and thinning them buys diversity per frame decoded.

Only the two maritime pools are registered as `real`.  open_ir_images also
carries kaist / llvip / m3fd / flir, which are street scenes: pedestrians and
cars, no ships, nothing for a ship detector to find.  They are left out
deliberately -- add one with `register_pool` if a use for it appears.
"""

from __future__ import annotations

import glob as globlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..paths import ARETE_ROOT, OPEN_IR_ROOT

__all__ = [
    "FramePool",
    "register_pool",
    "get_pool",
    "available_pools",
    "describe_pools",
    "resolve_paths",
    "IMAGE_SUFFIXES",
]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm")


@dataclass(frozen=True)
class FramePool:
    """A named set of frames on disk.

    patterns  glob patterns, expanded with `recursive=True` so `**` works.
    kind      'real' or 'synthetic' -- which side of idea.txt this is.  The
              generator uses it for nothing; a caller assembling both halves
              does.
    detector  the back end that suits this pool, used when the caller does not
              name one.  Ground truth exists for the synthetic renders, so
              'arete' is not a preference there but the right answer; the real
              pools have no labels, which is the whole premise, so they fall to
              whatever bounding box model is current.
    """

    name: str
    patterns: tuple[str, ...]
    kind: str = "real"
    detector: str | None = None
    doc: str = ""

    def paths(self) -> list[str]:
        """Every frame in the pool, sorted, memoised for the process.

        Sorted rather than as-globbed: the order is what a seeded sample and a
        stride are taken against, and glob order is filesystem order, which is
        not stable across machines.
        """
        if self.name not in _CACHE:
            found: list[str] = []
            for pattern in self.patterns:
                found.extend(globlib.glob(pattern, recursive=True))
            _CACHE[self.name] = sorted(dict.fromkeys(found))
        return _CACHE[self.name]

    def __len__(self) -> int:
        return len(self.paths())

    def __repr__(self) -> str:
        return f"FramePool({self.name!r}, kind={self.kind!r}, {len(self.patterns)} pattern(s))"


_POOLS: dict[str, FramePool] = {}
_CACHE: dict[str, list[str]] = {}


def register_pool(pool: FramePool) -> FramePool:
    """Add a pool to the registry, or replace one of the built-ins."""
    _POOLS[pool.name.strip().lower()] = pool
    _CACHE.pop(pool.name, None)
    return pool


def get_pool(name: str) -> FramePool:
    key = str(name).strip().lower()
    if key not in _POOLS:
        raise KeyError(f"unknown pool {name!r}; registered: {', '.join(available_pools())}")
    return _POOLS[key]


def available_pools() -> list[str]:
    return sorted(_POOLS)


def describe_pools() -> dict[str, str]:
    return {k: _POOLS[k].doc for k in available_pools()}


# --------------------------------------------------------------------------- #
# the built-ins
# --------------------------------------------------------------------------- #

_RAYSENSE = str(OPEN_IR_ROOT / "raysense" / "红外船舶数据库" / "dataset" / "images")
_MASSMIND = str(OPEN_IR_ROOT / "MassMIND" / "Images")

register_pool(
    FramePool(
        "raysense",
        (f"{_RAYSENSE}/train/*.jpg", f"{_RAYSENSE}/test/*.jpg"),
        kind="real",
        doc="9402 real IR ship frames (raysense 红外船舶数据库), 384x288, grabbed from video",
    )
)
register_pool(
    FramePool(
        "massmind",
        (f"{_MASSMIND}/*.png",),
        kind="real",
        doc="2916 real maritime LWIR frames (MassMIND), 640x512",
    )
)
register_pool(
    FramePool(
        "real",
        get_pool("raysense").patterns + get_pool("massmind").patterns,
        kind="real",
        doc="every real IR frame: raysense + MassMIND -- the training pool of idea.txt",
    )
)
register_pool(
    FramePool(
        "arete",
        (f"{ARETE_ROOT}/full_ir/images/*/lwir/*.png",),
        kind="synthetic",
        detector="arete",
        doc="1,088,640 synthetic LWIR renders, 27 scenes (arete full_ir) -- sample it, do not walk it",
    )
)
register_pool(
    FramePool(
        "arete_benchmark",
        (f"{ARETE_ROOT}/benchmark/images/*/lwir/*.png",),
        kind="synthetic",
        detector="arete",
        doc="1080 renders over 3 scenes (arete benchmark) -- small, fixed, good for figures",
    )
)


# --------------------------------------------------------------------------- #
# the front door
# --------------------------------------------------------------------------- #


def resolve_paths(
    spec: Any,
    limit: int | None = None,
    stride: int = 1,
    seed: int | None = 0,
    shuffle: bool = False,
) -> list[str]:
    """Turn whatever the caller configured into a concrete list of frame paths.

    spec accepts, in rough order of how a config would use it::

        "real"                        a registered pool name
        FramePool(...)                a pool object
        "/data/frames/*.png"          a glob pattern (** works)
        "/data/frames"                a directory, walked for image files
        "/data/one_frame.png"         a single file
        ["raysense", "/extra/*.png"]  any mixture of the above

    limit / stride / seed subset the result, in that order of application:
    `stride` first (keep every Nth, thinning near-duplicate video frames), then
    `limit` (a seeded sample without replacement, re-sorted afterwards so the
    list is stable).  `seed=None` with a limit takes the first N instead, which
    is only what you want when the pool is already unordered.

    shuffle returns the sample in seeded random order rather than sorted; the
    generator does its own per-epoch shuffling, so leave it off unless something
    downstream consumes the list in order and cannot.
    """
    paths = _expand(spec)
    if stride > 1:
        paths = paths[::stride]
    if limit is not None and len(paths) > limit:
        if seed is None:
            paths = paths[:limit]
        else:
            pick = np.random.default_rng(seed).choice(len(paths), size=limit, replace=False)
            paths = [paths[i] for i in sorted(pick)]
    if shuffle:
        order = np.random.default_rng(0 if seed is None else seed).permutation(len(paths))
        paths = [paths[i] for i in order]
    return paths


def _expand(spec: Any) -> list[str]:
    """spec -> sorted, de-duplicated paths, without any subsetting."""
    if isinstance(spec, FramePool):
        return list(spec.paths())
    if isinstance(spec, (str, os.PathLike)):
        return _expand_one(str(spec))
    if isinstance(spec, Iterable):
        out: list[str] = []
        for s in spec:
            out.extend(_expand(s))
        return sorted(dict.fromkeys(out))
    raise TypeError(f"cannot read a frame list out of {spec!r}")


def _expand_one(spec: str) -> list[str]:
    key = spec.strip().lower()
    if key in _POOLS:
        return list(_POOLS[key].paths())
    if any(c in spec for c in "*?["):
        return sorted(dict.fromkeys(globlib.glob(spec, recursive=True)))
    p = Path(spec)
    if p.is_dir():
        found = [str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES]
        return sorted(found)
    if p.exists():
        return [str(p)]
    raise FileNotFoundError(
        f"{spec!r} is not a registered pool ({', '.join(available_pools())}), "
        "a glob pattern, a directory or a file"
    )
