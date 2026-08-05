"""Acceptance test for an augmenter and its ops.

The augmentation stage has one property the training loop silently depends on:
`clean` and `doctored` are the same image everywhere except inside the paste
rectangle.  Break it -- an op that writes outside the mask, a blur that runs on
the whole frame, an in-place edit of the caller's array -- and the model quietly
learns to reconstruct pixels it was supposed to trust, with nothing in the loss
curve to say so.

Run this after adding or retuning an op:

    from realsimir.augment import ShipAugmenter, check_augmenter
    check_augmenter(ShipAugmenter(ops=[*DEFAULT_OPS, MyOp()]))

or `python -m realsimir.augment check --set feather_px=2`.
"""

from __future__ import annotations

import numpy as np

from ..boxes import BBox
from ..detectors import synthetic_frame

__all__ = ["check_augmenter"]


def check_augmenter(
    augmenter,
    image: np.ndarray | None = None,
    ship: BBox | None = None,
    trials: int = 8,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Check the invariants the training loop relies on.  [] means usable."""
    problems: list[str] = []
    image = synthetic_frame(240, 320) if image is None else image
    ship = BBox(139.0, 124.0, 183.0, 140.0, score=0.9) if ship is None else ship
    before = image.copy()

    pairs = [augmenter.augment(image, ship, seed=s) for s in range(trials)]

    if not np.array_equal(before, image):
        problems.append("augment modified the input image in place")

    for i, pair in enumerate(pairs):
        tag = f"seed {i}"
        if pair.doctored.shape != image.shape or pair.clean.shape != image.shape:
            problems.append(f"{tag}: shape changed, {image.shape} -> {pair.doctored.shape}")
            continue
        if pair.doctored.dtype != pair.clean.dtype:
            problems.append(f"{tag}: doctored is {pair.doctored.dtype}, clean is {pair.clean.dtype}")

        outside = pair.mask <= 0.0
        if not np.array_equal(pair.doctored[outside], pair.clean[outside]):
            n = int((pair.doctored[outside] != pair.clean[outside]).sum())
            problems.append(
                f"{tag}: {n} pixel(s) changed outside the paste mask -- an op is writing "
                "past the rectangle, so the target is no longer ground truth"
            )
        if not (pair.mask.min() >= 0.0 and pair.mask.max() <= 1.0):
            problems.append(f"{tag}: mask outside [0, 1]: [{pair.mask.min()}, {pair.mask.max()}]")
        if pair.rect.width < 1 or pair.rect.height < 1:
            problems.append(f"{tag}: degenerate paste rect {pair.rect}")

    # a single untouched draw is fine -- ops carry a probability, and p_clean is
    # a deliberate way to ask for some.  Never touching anything is not.
    if augmenter.p_clean < 1 and all(np.array_equal(p.doctored, p.clean) for p in pairs):
        problems.append(f"no corruption in {trials} draws; the ops never fire on this box")

    replay = augmenter.augment(image, ship, seed=0)
    if not np.array_equal(replay.doctored, pairs[0].doctored):
        problems.append(
            "the same seed produced a different result -- an op is drawing from somewhere "
            "other than the rng it is handed, so batches cannot be reproduced"
        )
    if trials > 1 and all(np.array_equal(p.doctored, pairs[0].doctored) for p in pairs[1:]):
        problems.append("every seed produced the same result -- the ops are not random")

    if verbose:
        name = f"{type(augmenter).__name__}({[op.name for op in augmenter.ops]})"
        if problems:
            print(f"FAIL {name}: {len(problems)} problem(s) over {trials} draw(s)")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"OK   {name}: {trials} draws, clean and doctored agree outside the rectangle")
    if strict and problems:
        raise AssertionError("augmenter check failed:\n  " + "\n  ".join(problems))
    return problems
