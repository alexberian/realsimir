"""CompositeGenerator -- the other half of step 07: the stream the model is *for*.

The training generator makes pairs out of real frames alone.  This one makes the
thing idea.txt is ultimately aiming at:

    "we can take croppings of the synthetic IR ship images and paste them into
     realworld images, then ask the model to do what it does best, clean
     poor-looking croppings"

So: a synthetic ship chunk, a host frame, one paste through exactly the
machinery step 06 uses for training, and a 256x256 window around the result.

Two kinds of test image, from the one object
--------------------------------------------
    from realsimir.data import CompositeGenerator, ShipDataGenerator

    hosts  = ShipDataGenerator("real",  index_file="boxes/real.json")
    donors = ShipDataGenerator("arete", index_file="boxes/arete.json", limit=4000)

    into_real = CompositeGenerator(hosts, donors)        # the deployment case
    into_self = CompositeGenerator.self_paste(donors)    # the measurable case

`into_real` is what the model is ultimately for, and it has no ground truth --
nobody ever photographed that ship in that harbour, so there is nothing to score
against.  You look at those.

`into_self` puts each synthetic ship back into its *own* render, by exactly the
recipe the training set is made with (step 06 on a synthetic crop instead of a
real one).  The original render is then the answer, so `c.target` is a real
target and the model can be scored -- MSE, LPIPS, whatever step 09 wants -- on
imagery of the same kind it will be given at test time.  It is the only place in
this project where "is the output right" is a question with an answer.

Rotation
--------
Both kinds cap how far a placed synthetic ship may be rolled (`max_rotation_deg`,
2 degrees by default) rather than taking the augmenter's training range.  A ship
sits level on the water: a few degrees of roll is a plausible sloppy crop, and
more is a hull tilted against its own wake and horizon, which is not an artefact
the model should be asked to invent its way out of.  The training corruption
keeps its wider range on purpose -- there it is teaching the model what
misalignment looks like, and the target still shows the ship level.

Placement
---------
Where a ship plausibly goes is a question about the scene, not about the paste,
and getting it wrong makes an image no model can save -- a hull in the sky, or
one the size of a house on the horizon line.  Two policies, both honest about
what they know:

  'detection'  put the synthetic ship where the host frame already has one, at
               that ship's size.  The host's own detector answers "where does a
               ship belong in this image, and how big is it at that range", which
               is a question we otherwise have no way to answer.  Needs a host
               generator whose index has boxes, which is the normal case.
  'lower'      uniform in a band of the lower frame, donor at its native size.
               A fallback for hosts with no detections; the composite is only as
               plausible as the guess.

Pass a callable for anything else: `fn(rng, host_shape, donor_shape) -> BBox`.
Self-paste needs no policy: the ship goes back where it came from.  Step 08 is
where all of these get looked at before any training happens.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

from ..augment import Rotate, ShipAugmenter
from ..boxes import BBox
from ..ship_cropper import ShipCropper
from .generator import ShipDataGenerator

__all__ = ["Composite", "CompositeGenerator", "limit_rotation", "DEFAULT_MAX_ROTATION_DEG"]

#: how far a *placed* synthetic ship may be rolled, in degrees.  Small on
#: purpose -- see the module docstring.
DEFAULT_MAX_ROTATION_DEG = 2.0


@dataclass
class Composite:
    """One test-time input: a synthetic ship living in a host frame.

    patch       the model's input, out_size x out_size, in the host's dtype.
    mask        float32 alpha over `patch`: 1 where the synthetic ship was laid
                down, 0 where the pixels are the host's own.  A diffusion
                sampler is free to change the first and must not change the
                second.
    host_patch  the same window of the host frame *before* the paste.  When the
                ship was pasted into a foreign frame this is not a target --
                there is no ship in it -- but it is what "the model must not
                disturb the surroundings" is checked against.  When the ship
                went back into its own render it *is* the target, and
                `has_target` says so.
    box         where the ship was placed, in the host frame's pixels.
                `box_in_patch()` is the same rectangle over `patch`.
    """

    patch: np.ndarray
    mask: np.ndarray
    host_patch: np.ndarray
    box: BBox
    host_path: str | None = None
    donor_path: str | None = None
    window: BBox | None = None
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    seed: int | None = None
    has_target: bool = False
    frame: np.ndarray | None = None  # the whole composited frame, when asked for

    @property
    def target(self) -> np.ndarray | None:
        """What the model should produce, when that is a thing that exists.

        None for a ship pasted into a foreign frame: the image being asked for
        has never been photographed.  A self-paste composite has one, which is
        what makes it scoreable.
        """
        return self.host_patch if self.has_target else None

    def box_in_patch(self) -> BBox:
        """`box` in patch coordinates -- mirrors ShipCrop.detection_in_patch."""
        if self.window is None:
            return self.box
        return self.box.shifted(-self.window.x_min, -self.window.y_min).scaled(self.scale)

    def to_record(self, prefix: str = "test") -> dict:
        """Flat scalars for wandb, matching AugmentedPair.to_record."""
        out: dict[str, Any] = {
            f"{prefix}/seed": self.seed,
            f"{prefix}/box_px": float(max(self.box.width, self.box.height)),
            f"{prefix}/mask_frac": float((self.mask > 0.5).mean()),
            f"{prefix}/has_target": self.has_target,
        }
        for family in ("geometric", "photometric"):
            for op, params in self.params.get(family, {}).items():
                for k, v in params.items():
                    if isinstance(v, (int, float, bool)):
                        out[f"{prefix}/{op}/{k}"] = v
        return out


class CompositeGenerator:
    """Synthetic ships into host frames -- the test / inference stream.

    hosts      a ShipDataGenerator over the real pool.  Its index supplies both
               the frames and, for 'detection' placement, the spots.
    donors     a ShipDataGenerator over the synthetic pool.  Only its *crops* are
               used, never its corruptions: the donor chunk is cut from the
               render, and everything that then happens to it is the augmenter's.
    into       'real' pastes the donor into a host frame from another sensor --
               the deployment case, no target.  'self' puts each synthetic ship
               back into its own render by the training recipe, which leaves the
               original as the target; `CompositeGenerator.self_paste(donors)`
               is the way to ask for it.
    max_rotation_deg
               cap, in degrees, on how far a placed ship may be rolled.  Applied
               to a *copy* of the augmenter, so the training stream that shares
               it is unaffected.  None uses the augmenter as given.
    placement  'detection', 'lower', or fn(rng, host_shape, donor_shape) -> BBox.
               Ignored when into='self': the ship goes back where it was.
    donor_box  which part of the synthetic crop to lift: 'paste' (the detection
               grown by the cropper's context, the default -- a little sea comes
               with the hull, as it would in a real crop) or 'tight'.
    match_levels
               rescale the donor to the host's local mean/std before the ops.
               Off by default: the radiometric mismatch between a render and a
               real sensor is a large part of what the model is being trained to
               fix, so removing it here would be answering the exam question.
    scale_jitter
               multiply the placed ship's size by a draw from this range, on top
               of the placement policy.  Detections are not a ground truth of
               scale either.
    keep_frame keep the whole composited frame on each Composite.  Off by
               default -- a 4k frame per example is not something to stream --
               but step 08's figures want it.  Nothing to keep when
               into='self': that paste happens inside the crop, so the composite
               *is* the patch.
    """

    def __init__(
        self,
        hosts: ShipDataGenerator,
        donors: ShipDataGenerator | None = None,
        augmenter: ShipAugmenter | None = None,
        cropper: ShipCropper | None = None,
        into: str = "real",
        max_rotation_deg: float | None = DEFAULT_MAX_ROTATION_DEG,
        placement: str | Callable = "detection",
        donor_box: str = "paste",
        match_levels: bool = False,
        scale_jitter: tuple[float, float] = (0.9, 1.1),
        lower_band: tuple[float, float] = (0.55, 0.85),
        seed: int = 0,
        keep_frame: bool = False,
        length: int | None = None,
    ):
        if donor_box not in ("paste", "tight"):
            raise ValueError(f"donor_box must be 'paste' or 'tight', got {donor_box!r}")
        if into not in ("real", "self"):
            raise ValueError(f"into must be 'real' or 'self', got {into!r}")
        self.hosts = hosts
        self.donors = donors if donors is not None else hosts
        self.into = into
        self.max_rotation_deg = max_rotation_deg
        self.augmenter = limit_rotation(augmenter or hosts.augmenter, max_rotation_deg)
        self.cropper = cropper or hosts.cropper
        self.placement = placement
        self.donor_box = donor_box
        self.match_levels = bool(match_levels)
        self.scale_jitter = scale_jitter
        self.lower_band = lower_band
        self.seed = int(seed)
        self.keep_frame = bool(keep_frame)
        self._length = length

    @classmethod
    def self_paste(cls, donors: ShipDataGenerator, **kwargs) -> "CompositeGenerator":
        """Each synthetic ship back into its own render -- the scoreable test set.

        The same recipe the training set is made with (step 06 on a crop), run
        over synthetic imagery instead of real: so every composite carries the
        render it came from as `c.target`, and a metric is possible.  The
        rotation cap applies here too -- these images stand in for the ones
        pasted into real frames, and should look like them.
        """
        return cls(donors, donors, into="self", **kwargs)

    # -- pairing ------------------------------------------------------------ #

    @property
    def has_targets(self) -> bool:
        """True when every composite carries the image the model should produce."""
        return self.into == "self"

    def __len__(self) -> int:
        """One composite per host example, unless `length` says otherwise."""
        return self._length if self._length is not None else len(self.hosts)

    def _donor_index(self, i: int, rng: np.random.Generator) -> int:
        """Which synthetic ship goes into composite i.

        Drawn rather than paired by position: the two pools have nothing to do
        with each other, and `i % len(donors)` would lock a host frame to one
        render for the life of the run.
        """
        return int(rng.integers(len(self.donors)))

    # -- placement ---------------------------------------------------------- #

    def _target(
        self, rng: np.random.Generator, host_i: int, host_shape: tuple[int, int], donor_shape: tuple[int, int]
    ) -> tuple[BBox, str]:
        """Where the synthetic ship goes, in host-frame pixels."""
        if callable(self.placement):
            return self.placement(rng, host_shape, donor_shape), "callable"

        dh, dw = donor_shape
        if self.placement == "detection":
            ship = self.hosts.index[host_i].box
            # the host's own ship sets the position *and* the scale: a hull that
            # far away is that many pixels across, whatever the render's size is
            aspect = dw / max(dh, 1)
            h = ship.height * float(rng.uniform(*self.scale_jitter))
            w = h * aspect
            cx, cy = ship.center
            return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, label="ship"), "detection"

        if self.placement == "lower":
            h, w = host_shape
            lo, hi = self.lower_band
            s = float(rng.uniform(*self.scale_jitter))
            bw, bh = dw * s, dh * s
            cx = float(rng.uniform(0.2, 0.8)) * w
            cy = float(rng.uniform(lo, hi)) * h
            return BBox(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, label="ship"), "lower"

        raise ValueError(f"unknown placement {self.placement!r}; use 'detection', 'lower' or a callable")

    # -- one composite ------------------------------------------------------ #

    def seed_for(self, i: int) -> int:
        return int(np.random.SeedSequence([self.seed, 0xC0FFEE, int(i)]).generate_state(1, dtype=np.uint32)[0])

    def donor_patch(self, donor_index: int) -> tuple[np.ndarray, str | None]:
        """The ship chunk cut out of a synthetic render."""
        crop = self.donors.crop(donor_index)
        box = crop.paste_box_in_patch() if self.donor_box == "paste" else crop.detection_in_patch()
        h, w = crop.patch.shape[:2]
        x0, y0, x1, y1 = box.clipped(w, h).as_int()
        return crop.patch[y0:y1, x0:x1], crop.path

    def composite(self, i: int) -> Composite:
        if self.into == "self":
            return self._self_composite(i)
        seed = self.seed_for(i)
        rng = np.random.default_rng(seed)
        host_i = i % len(self.hosts)
        ref = self.hosts.index[host_i]
        frame = self.hosts.frame(ref.path)

        donor_i = self._donor_index(i, rng)
        donor, donor_path = self.donor_patch(donor_i)
        if min(donor.shape[:2]) < 4:
            raise ValueError(f"donor {donor_i} is {donor.shape}, too small to paste")

        target, how = self._target(rng, host_i, frame.shape[:2], donor.shape[:2])
        pair = self.augmenter.augment(
            frame, target, source=donor, seed=seed, match_levels=self.match_levels
        )

        # the same window out of the composite, the host and the mask, so the
        # three are registered: cropper.window_for is a pure function of the box
        pasted = self.cropper.crop(pair.doctored, target, ref.path)
        host = self.cropper.crop(frame, target, ref.path)
        mask = self.cropper.crop(pair.mask, target).patch

        params = dict(pair.params, placement=how, donor_index=donor_i, host_index=host_i)
        return Composite(
            patch=np.ascontiguousarray(pasted.patch),
            mask=np.ascontiguousarray(mask),
            host_patch=np.ascontiguousarray(host.patch),
            box=target,
            host_path=ref.path,
            donor_path=donor_path,
            window=pasted.window,
            scale=pasted.scale,
            params=params,
            seed=seed,
            frame=pair.doctored if self.keep_frame else None,
        )

    def _self_composite(self, i: int) -> Composite:
        """A synthetic ship put back into its own render.

        This is `ShipAugmenter.augment_crop` -- the training recipe, unchanged --
        run over a synthetic crop instead of a real one, which is the point: the
        model is being asked the same question it was trained on, about imagery
        of the kind it will be given, and the render it came from is the answer.

        The work happens in the crop's coordinates rather than the render's, for
        the same reason training does: the ops then see the ship at the size the
        model will, and a big ship whose window was downscaled is not resampled
        twice.
        """
        seed = self.seed_for(i)
        crop = self.donors.crop(i % len(self.donors))
        pair = self.augmenter.augment_crop(crop, seed=seed)
        params = dict(pair.params, placement="self", donor_index=i, host_index=i)
        return Composite(
            patch=pair.doctored,
            mask=pair.mask,
            host_patch=pair.clean,  # the render as it was: the target
            box=crop.detection,
            host_path=crop.path,
            donor_path=crop.path,
            window=crop.window,
            scale=crop.scale,
            params=params,
            seed=seed,
            has_target=True,
        )

    def __getitem__(self, i: int) -> Composite:
        return self.composite(i)

    def __iter__(self) -> Iterator[Composite]:
        for i in range(len(self)):
            yield self.composite(i)

    def batches(self, batch_size: int) -> Iterator[list[Composite]]:
        batch: list[Composite] = []
        for c in self:
            batch.append(c)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __repr__(self) -> str:
        where = (
            f"{self.donors.source_name!r} into its own renders"
            if self.into == "self"
            else f"{self.donors.source_name!r} into {self.hosts.source_name!r}, "
            f"placement={self.placement!r}"
        )
        return (
            f"CompositeGenerator({len(self)} composites, {where}, "
            f"max_rotation={self.max_rotation_deg}deg)"
        )


def limit_rotation(augmenter: ShipAugmenter, max_deg: float | None) -> ShipAugmenter:
    """A copy of `augmenter` whose rotation ops are capped at `max_deg`.

    A copy, because the augmenter usually arrives borrowed from the training
    generator, and a test set quietly retuning the training corruption is the
    kind of bug that only shows up as a worse model.  `max_deg=None`, or an
    augmenter that already rotates less than that, hands back what it was given.
    """
    if max_deg is None:
        return augmenter
    rotations = [op for op in augmenter.ops if isinstance(op, Rotate)]
    if not rotations or all(op.max_deg <= max_deg for op in rotations):
        return augmenter
    out = copy.deepcopy(augmenter)
    # deepcopy keeps the ops shared between .ops and .geometric, so one write does both
    for op in out.ops:
        if isinstance(op, Rotate):
            op.max_deg = min(op.max_deg, float(max_deg))
    return out
