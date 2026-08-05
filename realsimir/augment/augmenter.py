"""ShipAugmenter -- todo step 06, the object that manufactures training pairs.

The middle of the algorithm in idea.txt.  Given a frame (or a crop of one) and
the ship box the detector found:

    1. jitter the box:  grow it by a random 10-20% and nudge it off centre, so
       the model never sees a rectangle that sits perfectly on the ship;
    2. lift the content inside that rectangle, doctor it -- offset, rotate,
       rescale, re-gain, re-level, blur, noise (realsimir/augment/ops.py) -- and
       lay it straight back down, hard-edged;
    3. hand back the doctored frame *and* the untouched one, pixel-registered.

The training signal is the difference between the two, which lives entirely
inside the rectangle: everything outside it is identical, so the model is being
told "these surroundings are ground truth, make the box agree with them".  That
is the same job it will be given at test time, when the box contains a
synthetic ship instead of a doctored real one -- which is why `augment` takes
an optional `source`: pass a patch and it is pasted through the identical
machinery, so the test-time composite cannot drift from the training-time one.

    from realsimir import ShipCropper, ShipAugmenter

    cropper, aug = ShipCropper("arete"), ShipAugmenter()
    for crop in cropper(path):
        pair = aug(crop, seed=index)        # pair.doctored -> model -> pair.clean

What it deliberately does not do: recover the exact offset.  A shift the model
cannot infer from the surroundings is one it can only learn to average over, so
the geometric ops are small on purpose.  The artefact worth fixing is the seam
and the mismatch in level, sharpness and noise across it, not the couple of
pixels the ship moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from ..boxes import BBox, ShipCrop
from ..imaging import from_float01, to_float01
from .ops import Corruption, OpContext, build_ops, split_ops

__all__ = ["ShipAugmenter", "AugmentedPair"]


@dataclass
class AugmentedPair:
    """One training example: what the model is shown, and what it must produce.

    clean     the untouched image -- the target.
    doctored  the same image with the ship's rectangle lifted and spoiled -- the
              input.  Identical to `clean` outside `rect`.
    mask      float32 alpha the paste was composited through, 1 inside the
              rectangle and 0 outside (soft in between when feathered).  Step 09
              can weight the loss with it; it is also what tells a diffusion
              sampler which pixels are up for grabs.
    ship      the detector's box, in this image's coordinates.
    rect      the jittered paste rectangle actually used.
    params    every magnitude drawn, for logging and for explaining a batch.
    """

    clean: np.ndarray
    doctored: np.ndarray
    mask: np.ndarray
    ship: BBox
    rect: BBox
    params: dict = field(default_factory=dict)
    seed: int | None = None
    crop: ShipCrop | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.clean.shape

    def difference(self) -> np.ndarray:
        """|doctored - clean| in [0, 1] -- what the model has to undo."""
        return np.abs(to_float01(self.doctored) - to_float01(self.clean))

    def to_record(self, prefix: str = "aug") -> dict:
        """Flat scalars for wandb: {'aug/shift/dx': -3.1, 'aug/gain/factor': 1.07, ...}."""
        out: dict[str, Any] = {f"{prefix}/seed": self.seed}
        for family in ("geometric", "photometric"):
            for op, params in self.params.get(family, {}).items():
                for k, v in params.items():
                    if isinstance(v, (int, float, bool)):
                        out[f"{prefix}/{op}/{k}"] = v
        out[f"{prefix}/rect_frac"] = float(self.rect.area / max(self.ship.area, 1e-6))
        return out


class ShipAugmenter:
    """Detector box in, (clean, doctored) pair out.

    ops           what may go wrong inside the rectangle: a list of names, a
                  {name: kwargs} mapping, or live `Corruption` objects.  See
                  realsimir/augment/ops.py for the catalogue and for how to add
                  one.  None means `DEFAULT_OPS`.
    context       how much bigger than the ship the paste rectangle is, drawn
                  per example and per axis -- idea.txt's "10 or 20% larger".
    offset_frac   how far the rectangle itself may sit off the ship's centre, as
                  a fraction of the ship's size.  Models a detector that is a
                  little wrong, which the real one will be.
    feather_px    gaussian softening of the rectangle's edge.  0 keeps the seam
                  hard, which is what a naive paste at test time will produce --
                  raise it only if the test-time compositor feathers too, or the
                  model will be trained to fix an artefact it is never shown.
    p_clean       fraction of examples returned untouched (doctored == clean).
                  Above 0 it teaches "leave a good crop alone"; the default 0
                  spends every example on the corruption.
    match_levels  only consulted when a foreign `source` is pasted: linearly
                  rescale the donor to the mean/std of the region it covers
                  before the ops run.  Off by default -- for training the two
                  already match, and at test time the level mismatch is part of
                  what the model is there to fix.
    dtype         output dtype; None matches the input.  Integer frames are
                  scaled by their dtype range internally; a *float* input is
                  taken to be in [0, 1] already (see imaging.to_float01) and is
                  clipped to it on the way out.
    seed          seeds the augmenter's own generator, used when a call does not
                  supply one.  Prefer per-example seeds (`aug(crop, seed=i)`):
                  they survive dataloader workers and make a figure replayable.
    """

    def __init__(
        self,
        ops: Any = None,
        context: tuple[float, float] = (0.10, 0.20),
        offset_frac: float = 0.04,
        feather_px: float = 0.0,
        p_clean: float = 0.0,
        match_levels: bool = False,
        dtype: np.dtype | type | None = None,
        seed: int | None = None,
        border_mode: int = cv2.BORDER_REFLECT_101,
    ):
        self.ops: list[Corruption] = build_ops(ops)
        self.geometric, self.photometric = split_ops(self.ops)
        self.context = context
        self.offset_frac = float(offset_frac)
        self.feather_px = float(feather_px)
        self.p_clean = float(p_clean)
        self.match_levels = bool(match_levels)
        self.dtype = dtype
        self.border_mode = border_mode
        self._rng = np.random.default_rng(seed)

    # -- geometry ----------------------------------------------------------- #

    def rect_for(self, ship: BBox, rng: np.random.Generator) -> tuple[BBox, dict]:
        """The jittered paste rectangle: ship box, grown and nudged."""
        lo, hi = self.context
        cx, cy = float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))
        rect = BBox(
            ship.x_min - ship.width * cx / 2.0,
            ship.y_min - ship.height * cy / 2.0,
            ship.x_max + ship.width * cx / 2.0,
            ship.y_max + ship.height * cy / 2.0,
            score=ship.score,
            label=ship.label,
        )
        ox = float(rng.uniform(-self.offset_frac, self.offset_frac)) * ship.width
        oy = float(rng.uniform(-self.offset_frac, self.offset_frac)) * ship.height
        return rect.shifted(ox, oy), {"context": (cx, cy), "offset": (ox, oy)}

    # -- the composite ------------------------------------------------------ #

    def _roi(self, rect: BBox, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Integer window around `rect` big enough to warp inside.

        Everything happens in here rather than on the whole frame: a 4k frame
        with six ships costs six small warps, not six full-frame ones, and the
        result is identical because only pixels inside the rectangle survive.
        """
        h, w = shape
        x0, y0, x1, y1 = rect.as_int()
        margin = int(np.ceil(0.5 * max(rect.width, rect.height))) + 16
        return (
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(w, x1 + margin),
            min(h, y1 + margin),
        )

    def _alpha(self, rect: BBox, shape: tuple[int, int]) -> np.ndarray:
        """Hard (or feathered) rectangular mask, float32 in [0, 1]."""
        h, w = shape
        a = np.zeros((h, w), dtype=np.float32)
        x0, y0, x1, y1 = rect.as_int()
        a[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = 1.0
        if self.feather_px > 0:
            k = int(2 * round(3 * self.feather_px) + 1)
            a = cv2.GaussianBlur(a, (k, k), self.feather_px, borderType=cv2.BORDER_REPLICATE)
        return a

    def _donor(
        self,
        sub: np.ndarray,
        rect: BBox,
        matrix: np.ndarray,
        source: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The content to lay into the rectangle, plus where it is defined.

        With no `source` the donor is the frame itself, warped -- the training
        case, where the ship gets re-laid on top of itself.  With a `source` it
        is that patch, fitted to the rectangle and then warped by the same
        matrix -- the test-time case, a synthetic ship dropped into a real
        frame.  One code path, so the two cannot diverge.
        """
        h, w = sub.shape[:2]
        if source is None:
            donor = cv2.warpAffine(
                sub, matrix[:2], (w, h), flags=cv2.INTER_LINEAR, borderMode=self.border_mode
            )
            return donor, None

        sh, sw = source.shape[:2]
        fit = np.array(
            [
                [rect.width / sw, 0.0, rect.x_min],
                [0.0, rect.height / sh, rect.y_min],
                [0.0, 0.0, 1.0],
            ]
        )
        total = (matrix @ fit)[:2]
        donor = cv2.warpAffine(
            source, total, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        valid = cv2.warpAffine(
            np.ones((sh, sw), np.float32), total, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
        )
        return donor, valid

    @staticmethod
    def _match(donor: np.ndarray, host: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Put the donor on the host's mean and std, inside the mask.

        The gain is clamped, and dropped entirely for a near-flat donor: a
        synthetic render that sits in a handful of counts must not be stretched
        to a real frame's contrast, because that is amplifying its quantisation,
        not matching its radiometry.
        """
        sel = alpha > 0.5
        if sel.sum() < 8:
            return donor
        d, h = donor[sel], host[sel]
        ds = float(d.std())
        gain = float(np.clip(float(h.std()) / ds, 0.2, 5.0)) if ds > 1e-3 else 1.0
        return (donor - float(d.mean())) * gain + float(h.mean())

    # -- the whole thing ---------------------------------------------------- #

    def augment(
        self,
        image: np.ndarray,
        ship: BBox,
        source: np.ndarray | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        match_levels: bool | None = None,
        crop: ShipCrop | None = None,
    ) -> AugmentedPair:
        """Doctor one ship in one image.

        `image` may be the whole frame or a crop of it, as long as `ship` is in
        the same coordinates -- `augment_crop` is the version for a ShipCrop.
        `source`, when given, is pasted into the rectangle instead of the
        image's own content (the synthetic-ship-into-real-frame path).
        """
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else self._rng
        out_dtype = np.dtype(self.dtype) if self.dtype is not None else image.dtype

        clean = (
            np.array(image, copy=True)  # the caller owns what comes back
            if image.dtype == out_dtype
            else from_float01(to_float01(image), out_dtype)
        )
        h, w = image.shape[:2]
        rect, jitter = self.rect_for(ship, rng)
        rect = rect.clipped(w, h)
        # the two families are always present, empty or not, so a consumer of
        # `params` never has to special-case an example where nothing fired
        params: dict = {
            "rect": rect.to_dict(),
            **jitter,
            "feather_px": self.feather_px,
            "geometric": {},
            "photometric": {},
        }

        skip = self.p_clean > 0 and rng.random() < self.p_clean
        if skip or rect.width < 2 or rect.height < 2:
            params["skipped"] = "p_clean" if skip else "degenerate rect"
            return AugmentedPair(clean, clean.copy(), self._alpha(rect, (h, w)), ship, rect, params, seed, crop)

        rx0, ry0, rx1, ry1 = self._roi(rect, (h, w))
        # copy, not a view: for a float32 frame `to_float01` passes the slice
        # straight through, and cv2 wants contiguous input anyway
        sub = np.array(to_float01(image[ry0:ry1, rx0:rx1]), dtype=np.float32, copy=True, order="C")
        local = rect.shifted(-rx0, -ry0)
        ctx = OpContext(local, sub.shape[:2])

        matrix = np.eye(3)
        for op in self.geometric:
            drawn = op.draw(rng, ctx)
            if drawn is None:
                continue
            matrix = op.matrix(drawn, ctx) @ matrix
            params["geometric"][_key(params["geometric"], op.name)] = drawn

        source_f = None if source is None else to_float01(_match_channels(source, sub))
        donor, valid = self._donor(sub, local, matrix, source_f)

        alpha = self._alpha(local, sub.shape[:2])
        if valid is not None:
            alpha = alpha * valid
        if source_f is not None and (self.match_levels if match_levels is None else match_levels):
            donor = self._match(donor, sub, alpha)

        for op in self.photometric:
            drawn = op.draw(rng, ctx)
            if drawn is None:
                continue
            donor = op.apply(donor, drawn, rng)
            params["photometric"][_key(params["photometric"], op.name)] = drawn

        donor = np.clip(donor, 0.0, 1.0)
        if sub.ndim == 3:
            alpha = alpha[..., None]
        blended = sub * (1.0 - alpha) + donor * alpha

        doctored = clean.copy()
        doctored[ry0:ry1, rx0:rx1] = from_float01(blended, out_dtype)

        full_alpha = np.zeros((h, w), np.float32)
        full_alpha[ry0:ry1, rx0:rx1] = alpha if alpha.ndim == 2 else alpha[..., 0]
        return AugmentedPair(clean, doctored, full_alpha, ship, rect, params, seed, crop)

    def augment_crop(self, crop: ShipCrop, **kwargs) -> AugmentedPair:
        """`augment` on a ShipCropper window, in patch coordinates.

        This is the training path: the pair comes out already 256x256 and
        registered, and the ops see the ship at the size the model will.  Note
        the augmenter re-derives its own paste rectangle from the detection
        rather than reusing `crop.paste_box` -- the cropper's context is fixed
        (it has a window to size), this one is drawn per example.
        """
        return self.augment(crop.patch, crop.detection_in_patch(), crop=crop, **kwargs)

    def __call__(self, subject: ShipCrop | np.ndarray, *args, **kwargs) -> AugmentedPair:
        if isinstance(subject, ShipCrop):
            return self.augment_crop(subject, *args, **kwargs)
        return self.augment(subject, *args, **kwargs)

    def batch(self, crops: Sequence[ShipCrop], seed: int | None = None, **kwargs) -> list[AugmentedPair]:
        """One pair per crop.  With `seed`, example i is reproducible as seed + i."""
        return [
            self.augment_crop(c, seed=None if seed is None else seed + i, **kwargs)
            for i, c in enumerate(crops)
        ]

    def __repr__(self) -> str:
        return f"ShipAugmenter(ops={[op.name for op in self.ops]}, context={self.context})"


def _key(seen: dict, name: str) -> str:
    """Unique key for the params record when an op appears more than once."""
    if name not in seen:
        return name
    i = 2
    while f"{name}#{i}" in seen:
        i += 1
    return f"{name}#{i}"


def _match_channels(source: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Make a donor patch's channel count agree with the frame it lands in."""
    if source.ndim == like.ndim and (source.ndim == 2 or source.shape[2] == like.shape[2]):
        return source
    if like.ndim == 2:
        return source.mean(axis=2).astype(source.dtype)
    c = like.shape[2]
    if source.ndim == 2:
        return np.repeat(source[:, :, None], c, axis=2)
    if source.shape[2] == 1:
        return np.repeat(source, c, axis=2)
    raise ValueError(f"cannot paste a {source.shape[2]}-channel patch into a {c}-channel frame")
