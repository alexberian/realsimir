"""The corruptions -- individual ways to make a pasted ship look wrong.

Each op is a small object with two halves, kept apart on purpose:

    sample(rng, ctx) -> params | None    draw this example's magnitudes
    matrix / apply                       do the deterministic work

Splitting them is what makes an example reproducible and reportable: the
augmenter records every `params` dict it drew, so a figure can be regenerated,
a bad-looking batch can be explained, and step 09 can log the distribution it
actually trained on rather than the one the config asked for.  `sample`
returning None means "not this time" -- that is how `p` is implemented.

Two families, because they compose differently:

  * `GeometricCorruption` contributes a 3x3 homogeneous forward map.  The
    augmenter multiplies them all together and resamples *once*, so stacking
    shift + rotate + rescale costs one bilinear interpolation, not three.
  * `PhotometricCorruption` maps float32 pixels in [0, 1] to float32 pixels in
    [0, 1].  These run in the order listed; the default order (gain, level,
    gamma, blur, noise) is the order a sensor would apply them.

Ops are named, like the detectors, so the training config can carry strings:

    ShipAugmenter(ops={"shift": {"max_px": 4}, "noise": {"sigma": (0, 0.05)}})

Adding one
----------
    from realsimir.augment import PhotometricCorruption, register_op

    @register_op("vignette", doc="cos^4 falloff, as if the paste came from a
                                 different part of the focal plane")
    class Vignette(PhotometricCorruption):
        def __init__(self, strength=(0.0, 0.25), **kwargs):
            super().__init__(**kwargs)
            self.strength = strength

        def sample(self, rng, ctx):
            return {"strength": _uniform(rng, self.strength)}

        def apply(self, image, params, rng):
            ...  # image is float32 in [0, 1], HxW or HxWxC
            return out

Then `ShipAugmenter(ops=[*DEFAULT_OPS, "vignette"])`, or pass the instance
directly.  No other file needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from ..boxes import BBox

__all__ = [
    "OpContext",
    "Corruption",
    "GeometricCorruption",
    "PhotometricCorruption",
    "Shift",
    "Rotate",
    "Rescale",
    "Gain",
    "Level",
    "Gamma",
    "Blur",
    "Noise",
    "Stripe",
    "DEFAULT_OPS",
    "register_op",
    "build_op",
    "build_ops",
    "available_ops",
    "describe_ops",
]

Range = float | tuple[float, float]


@dataclass(frozen=True)
class OpContext:
    """What an op is allowed to know about the example it is corrupting.

    `rect` is the paste rectangle in the coordinates the op works in, which is
    a cropped region of the frame, not the frame itself -- so an op must not
    assume rect.x_min is the frame offset.  Its *size* is the useful part: it
    is how big the ship is on screen, which is what a shift of "a few pixels"
    should scale against.
    """

    rect: BBox
    shape: tuple[int, int]  # (h, w) of the working image


def _uniform(rng: np.random.Generator, spec: Range) -> float:
    """(lo, hi) -> a draw; a bare number -> that number, every time."""
    if isinstance(spec, (int, float)):
        return float(spec)
    lo, hi = spec
    return float(rng.uniform(lo, hi))


def _about(center: tuple[float, float], a: np.ndarray) -> np.ndarray:
    """3x3 homogeneous map applying the 2x2 `a` about `center`."""
    c = np.asarray(center, dtype=np.float64)
    m = np.eye(3, dtype=np.float64)
    m[:2, :2] = a
    m[:2, 2] = c - a @ c
    return m


# --------------------------------------------------------------------------- #
# base classes
# --------------------------------------------------------------------------- #


class Corruption(ABC):
    """One named way to spoil a crop.

    p  probability of firing at all.  Below 1.0 the op is absent from some
       examples, which keeps the model from assuming every input carries every
       artefact.
    """

    def __init__(self, p: float = 1.0):
        self.p = float(p)

    def draw(self, rng: np.random.Generator, ctx: OpContext) -> dict | None:
        """`sample`, gated by p.  This is what the augmenter calls."""
        if self.p < 1.0 and rng.random() >= self.p:
            return None
        return self.sample(rng, ctx)

    @abstractmethod
    def sample(self, rng: np.random.Generator, ctx: OpContext) -> dict | None:
        """The magnitudes for one example; None to skip."""

    @property
    def name(self) -> str:
        return getattr(type(self), "op_name", type(self).__name__.lower())

    def __repr__(self) -> str:
        args = [f"{k}={v!r}" for k, v in vars(self).items() if k != "p"]
        if self.p != 1.0:
            args.append(f"p={self.p!r}")
        return f"{type(self).__name__}({', '.join(args)})"


class GeometricCorruption(Corruption):
    """Moves the pasted content around inside the rectangle.

    These are the "slightly offset or rotated" of idea.txt.  Keep them small:
    the target is the *original* frame, so a shift the model cannot possibly
    infer is a shift it can only learn to average over.  A few percent of the
    ship's size is enough to break pixel-exact alignment with the surroundings,
    which is the cue we actually want it to fix.
    """

    @abstractmethod
    def matrix(self, params: dict, ctx: OpContext) -> np.ndarray:
        """3x3 homogeneous forward map (source pixel -> destination pixel)."""


class PhotometricCorruption(Corruption):
    """Changes the pixel values without moving them.

    Gain, level, gamma, blur and noise between them cover the realistic ways a
    crop taken from one sensor/scene fails to match the frame it is dropped
    into.  Input and output are float32 in [0, 1], HxW or HxWxC; the augmenter
    clips once at the end, so an op may overshoot internally.
    """

    @abstractmethod
    def apply(self, image: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
        """Corrupted copy of `image`.  Must not modify the input in place."""


# --------------------------------------------------------------------------- #
# registry -- same conventions as realsimir.detectors.registry
# --------------------------------------------------------------------------- #

_OPS: dict[str, type] = {}
_ALIASES: dict[str, str] = {}
_DOC: dict[str, str] = {}


def register_op(name: str, *aliases: str, doc: str | None = None):
    """Class decorator: make `name` build this corruption."""

    def deco(cls: type) -> type:
        key = name.strip().lower()
        existing = _OPS.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(f"op {name!r} is already registered to {existing.__name__}")
        cls.op_name = key
        _OPS[key] = cls
        _DOC[key] = " ".join((doc or (cls.__doc__ or "")).strip().split("\n")[0].split())
        for a in aliases:
            _ALIASES.setdefault(a.strip().lower(), key)
        return cls

    return deco


def available_ops() -> list[str]:
    return sorted(_OPS)


def describe_ops() -> dict[str, str]:
    return {k: _DOC.get(k, "") for k in available_ops()}


def build_op(spec: Any, **overrides) -> Corruption:
    """name / {'name': ..., ...} / class / instance -> a live Corruption."""
    if isinstance(spec, Corruption):
        if overrides:
            raise TypeError("build_op got keyword overrides for an already-built op")
        return spec
    if isinstance(spec, Mapping):
        cfg = dict(spec)
        name = cfg.pop("name", None) or cfg.pop("type", None)
        if name is None:
            raise ValueError("op config needs a 'name' key")
        return build_op(name, **{**cfg, **overrides})
    if isinstance(spec, type):
        cls = spec
    else:
        key = str(spec).strip().lower()
        key = _ALIASES.get(key, key)
        if ":" in key:  # module:ClassName, for an op living outside this repo
            import importlib

            mod, _, attr = str(spec).partition(":")
            cls = getattr(importlib.import_module(mod), attr)
        elif key in _OPS:
            cls = _OPS[key]
        else:
            raise KeyError(f"unknown corruption {spec!r}; registered: {', '.join(available_ops())}")
    if not (isinstance(cls, type) and issubclass(cls, Corruption)):
        raise TypeError(f"{cls} is not a Corruption subclass")
    return cls(**overrides)


def build_ops(spec: Any = None) -> list[Corruption]:
    """Turn whatever the caller configured into a list of ops.

    Accepts, in rough order of how a config file would use it::

        None                                  the defaults below
        ["shift", "noise"]                    names, in the order they run
        {"shift": {"max_px": 4}, "blur": None}  names -> kwargs; None/False drops one
        [Shift(max_px=4), "noise"]            live objects mixed in

    A mapping is the form to reach for when overriding defaults from YAML, and
    the only one that can *remove* an op.
    """
    if spec is None:
        spec = DEFAULT_OPS
    if isinstance(spec, (str, Corruption, type)):
        spec = [spec]
    if isinstance(spec, Mapping):
        out = []
        for name, cfg in spec.items():
            if cfg is None or cfg is False:
                continue
            out.append(build_op(name, **(cfg if isinstance(cfg, Mapping) else {})))
        return out
    if not isinstance(spec, Iterable):
        raise TypeError(f"cannot read an op list out of {spec!r}")
    return [build_op(s) for s in spec]


# --------------------------------------------------------------------------- #
# geometric
# --------------------------------------------------------------------------- #


@register_op("shift", "offset", doc="translate the pasted content by a few pixels")
class Shift(GeometricCorruption):
    """Slide the content inside the rectangle.

    The amount is the smaller of `max_px` and `max_frac` of the rectangle's
    mean side, so a 20 px ship and a 400 px ship both get a shift that reads as
    "slightly misplaced" rather than "obliterated".
    """

    def __init__(self, max_px: float = 8.0, max_frac: float = 0.08, p: float = 1.0):
        super().__init__(p)
        self.max_px = float(max_px)
        self.max_frac = float(max_frac)

    def sample(self, rng, ctx):
        r = min(self.max_px, self.max_frac * (ctx.rect.width + ctx.rect.height) / 2.0)
        if r <= 0:
            return None
        return {"dx": float(rng.uniform(-r, r)), "dy": float(rng.uniform(-r, r))}

    def matrix(self, params, ctx):
        m = np.eye(3, dtype=np.float64)
        m[0, 2], m[1, 2] = params["dx"], params["dy"]
        return m


@register_op("rotate", "rotation", doc="rotate the pasted content about the box centre")
class Rotate(GeometricCorruption):
    """A few degrees of roll -- a box that was drawn on a ship that is not level."""

    def __init__(self, max_deg: float = 6.0, p: float = 1.0):
        super().__init__(p)
        self.max_deg = float(max_deg)

    def sample(self, rng, ctx):
        if self.max_deg <= 0:
            return None
        return {"deg": float(rng.uniform(-self.max_deg, self.max_deg))}

    def matrix(self, params, ctx):
        t = np.deg2rad(params["deg"])
        c, s = np.cos(t), np.sin(t)
        return _about(ctx.rect.center, np.array([[c, -s], [s, c]]))


@register_op("rescale", "zoom", "scale", doc="resize the pasted content in place")
class Rescale(GeometricCorruption):
    """The crop was taken at slightly the wrong range, or the box was too tight.

    Note this scales the *content*, not the rectangle: the seam stays where it
    was and the ship inside it grows or shrinks past it.
    """

    def __init__(self, factor: Range = (0.92, 1.08), anisotropic: bool = False, p: float = 1.0):
        super().__init__(p)
        self.factor = factor
        self.anisotropic = bool(anisotropic)

    def sample(self, rng, ctx):
        sx = _uniform(rng, self.factor)
        sy = _uniform(rng, self.factor) if self.anisotropic else sx
        return {"sx": sx, "sy": sy}

    def matrix(self, params, ctx):
        return _about(ctx.rect.center, np.diag([params["sx"], params["sy"]]).astype(np.float64))


# --------------------------------------------------------------------------- #
# photometric
# --------------------------------------------------------------------------- #


@register_op("gain", doc="multiplicative pixel gain -- the sensor scale is off")
class Gain(PhotometricCorruption):
    """`idea.txt`'s "pixel gain changed".

    Multiplicative about zero rather than about the local mean, because that is
    what a mismatched gain actually does: it moves the background level too,
    which is exactly the discontinuity the seam shows.
    """

    def __init__(self, factor: Range = (0.85, 1.18), p: float = 1.0):
        super().__init__(p)
        self.factor = factor

    def sample(self, rng, ctx):
        return {"factor": _uniform(rng, self.factor)}

    def apply(self, image, params, rng):
        return image * np.float32(params["factor"])


@register_op("level", "bias", "offset_level", doc="additive brightness offset")
class Level(PhotometricCorruption):
    """A DC step, in units of the full range (0.05 == 13 counts of an 8-bit frame).

    Together with `gain` this is the affine radiometric mismatch between a crop
    and the frame it lands in -- the single most visible paste artefact in LWIR,
    where the whole scene sits in a narrow band of counts.
    """

    def __init__(self, amount: float = 0.05, p: float = 1.0):
        super().__init__(p)
        self.amount = float(amount)

    def sample(self, rng, ctx):
        if self.amount <= 0:
            return None
        return {"delta": float(rng.uniform(-self.amount, self.amount))}

    def apply(self, image, params, rng):
        return image + np.float32(params["delta"])


@register_op("gamma", doc="nonlinear tone response")
class Gamma(PhotometricCorruption):
    """Different AGC curve on the donor than on the frame."""

    def __init__(self, factor: Range = (0.85, 1.2), p: float = 0.7):
        super().__init__(p)
        self.factor = factor

    def sample(self, rng, ctx):
        return {"gamma": _uniform(rng, self.factor)}

    def apply(self, image, params, rng):
        return np.clip(image, 0.0, 1.0) ** np.float32(params["gamma"])


@register_op("blur", doc="gaussian blur -- a softer donor than the frame around it")
class Blur(PhotometricCorruption):
    """Optics, range or an upsampled crop: the paste is less sharp than its host."""

    def __init__(self, sigma: Range = (0.0, 1.2), p: float = 0.7):
        super().__init__(p)
        self.sigma = sigma

    def sample(self, rng, ctx):
        s = _uniform(rng, self.sigma)
        return {"sigma": s} if s > 0.05 else None

    def apply(self, image, params, rng):
        s = float(params["sigma"])
        k = int(2 * round(3 * s) + 1)
        return cv2.GaussianBlur(image, (k, k), s, borderType=cv2.BORDER_REFLECT_101)


@register_op("noise", doc="additive gaussian sensor noise")
class Noise(PhotometricCorruption):
    """`sigma` is in units of the full range, so 0.02 is ~5 counts of 8-bit."""

    def __init__(self, sigma: Range = (0.0, 0.035), p: float = 1.0):
        super().__init__(p)
        self.sigma = sigma

    def sample(self, rng, ctx):
        s = _uniform(rng, self.sigma)
        return {"sigma": s} if s > 0 else None

    def apply(self, image, params, rng):
        return image + rng.normal(0.0, params["sigma"], image.shape).astype(np.float32)


@register_op("stripe", "fpn", doc="row/column fixed-pattern noise, the classic LWIR artefact")
class Stripe(PhotometricCorruption):
    """Per-column (and/or per-row) offsets, constant along the other axis.

    Not in DEFAULT_OPS: whether it belongs depends on the sensor behind the real
    frames, and switching it on when the data has none would teach the model to
    remove something it will never see.  Turn it on with
    `ops={**dict.fromkeys(DEFAULT_OPS, {}), "stripe": {"sigma": (0, 0.02)}}`
    once the real imagery is known to show it.
    """

    def __init__(self, sigma: Range = (0.0, 0.02), axis: str = "column", p: float = 0.5):
        super().__init__(p)
        self.sigma = sigma
        self.axis = axis

    def sample(self, rng, ctx):
        s = _uniform(rng, self.sigma)
        if s <= 0:
            return None
        axis = self.axis
        if axis == "both":
            axis = "column" if rng.random() < 0.5 else "row"
        return {"sigma": s, "axis": axis}

    def apply(self, image, params, rng):
        h, w = image.shape[:2]
        n = w if params["axis"] == "column" else h
        offs = rng.normal(0.0, params["sigma"], n).astype(np.float32)
        offs = offs[None, :] if params["axis"] == "column" else offs[:, None]
        if image.ndim == 3:
            offs = offs[..., None]
        return image + offs


# The set idea.txt asks for -- "slightly offset or rotated pixel gain changed or
# noised" -- plus the two that follow from the same argument (a mis-sized crop,
# a softer one).  Photometric ops run in this order, which is the order a sensor
# would impose them.
DEFAULT_OPS: tuple[str, ...] = ("shift", "rotate", "rescale", "gain", "level", "gamma", "blur", "noise")


def split_ops(ops: Sequence[Corruption]) -> tuple[list[GeometricCorruption], list[PhotometricCorruption]]:
    """Partition into the two families, rejecting anything that is neither."""
    geo: list[GeometricCorruption] = []
    photo: list[PhotometricCorruption] = []
    for op in ops:
        if isinstance(op, GeometricCorruption):
            geo.append(op)
        elif isinstance(op, PhotometricCorruption):
            photo.append(op)
        else:
            raise TypeError(
                f"{type(op).__name__} subclasses Corruption directly; it must subclass "
                "GeometricCorruption or PhotometricCorruption so the augmenter knows how to run it"
            )
    return geo, photo
