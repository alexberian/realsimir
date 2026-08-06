"""Acceptance test for a configured generator -- run it before a training run.

`check_augmenter` (step 06) proves the corruption behaves on a synthetic frame.
This proves the whole front end behaves on *your* data, with *your* boxes: that
the index actually matches the frames on disk, that an example is the same in a
worker as in the parent, and that epochs vary the input without moving the
target.  Everything it tests is something that fails silently -- a model trains
to a plausible-looking loss curve on data that is quietly wrong.

    from realsimir.data import ShipDataGenerator, check_generator
    check_generator(ShipDataGenerator("real", index_file="boxes/real.json"))

or `python -m realsimir.data check --source real --index-file boxes/real.json`.
"""

from __future__ import annotations

import pickle

import numpy as np

__all__ = ["check_generator"]


def check_generator(
    generator,
    trials: int = 8,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Check what the training loop assumes.  [] means the generator is usable."""
    problems: list[str] = []
    n = len(generator)
    if n == 0:
        problems.append("the index is empty -- no frame in this pool has a detection")
        return _finish(problems, generator, 0, strict, verbose)

    picks = _spread(n, trials)
    size = generator.cropper.out_size

    for i in picks:
        tag = f"example {i}"
        pair = generator.example(i, epoch=0)

        if pair.clean.shape[:2] != (size, size) or pair.doctored.shape != pair.clean.shape:
            problems.append(f"{tag}: {pair.clean.shape[:2]} patch, expected ({size}, {size})")
            continue
        if pair.doctored.dtype != pair.clean.dtype:
            problems.append(f"{tag}: doctored is {pair.doctored.dtype}, clean is {pair.clean.dtype}")

        outside = pair.mask <= 0.0
        if not np.array_equal(pair.doctored[outside], pair.clean[outside]):
            problems.append(
                f"{tag}: pixels differ outside the paste mask -- the surroundings are supposed "
                "to be ground truth (run check_augmenter on the ops)"
            )
        if not (pair.mask > 0.5).any():
            problems.append(f"{tag}: empty paste mask; the box is off the patch")
        if np.array_equal(pair.doctored, pair.clean) and not pair.params.get("skipped"):
            problems.append(f"{tag}: nothing was corrupted, and no reason was recorded")

        # a patch that is one flat value is a crop of empty sea or a dead frame:
        # a real example, but a whole pool of them means the boxes are wrong
        if float(np.std(pair.clean.astype(np.float32))) < 1e-3:
            problems.append(f"{tag}: the crop is featureless -- is the box on a ship? ({_where(generator, i)})")

    # 1. determinism: the same (index, epoch) is the same bytes, always
    a, b = generator.example(picks[0], epoch=0), generator.example(picks[0], epoch=0)
    if not np.array_equal(a.doctored, b.doctored):
        problems.append("the same (index, epoch) produced two different examples")

    # 2. the target is stable across epochs, the input is not.  This is the one
    #    that matters most: the clean patch is what the diffusion model is asked
    #    to produce, so it must not move underneath it between epochs.
    later = generator.example(picks[0], epoch=1)
    if not np.array_equal(later.clean, a.clean):
        problems.append("the clean target changed between epochs -- the crop is not stable")
    if np.array_equal(later.doctored, a.doctored) and len(generator.augmenter.ops):
        problems.append("epoch 1 corrupted example 0 exactly as epoch 0 did -- the seed ignores the epoch")

    # 3. an epoch visits every example exactly once
    order = generator.order(0)
    if sorted(order.tolist()) != list(range(n)):
        problems.append(f"order(0) is not a permutation of the {n} examples")
    if n > 1 and np.array_equal(order, generator.order(1)):
        problems.append("every epoch visits the examples in the same order")

    # 4. the frame cache must not change what comes out
    if generator.cache_frames > 0:
        first = generator.example(picks[0], epoch=0)
        for i in picks[1:]:  # evict, then come back
            generator.example(i, epoch=0)
        if not np.array_equal(generator.example(picks[0], epoch=0).doctored, first.doctored):
            problems.append("an example changed after the frame cache turned over")

    # 5. a dataloader worker gets the same data as the parent
    try:
        clone = pickle.loads(pickle.dumps(generator))
    except Exception as exc:
        problems.append(
            f"the generator cannot be pickled ({type(exc).__name__}: {exc}) -- a DataLoader "
            "with num_workers>0 on a spawn start method will fail.  Cache the boxes to an "
            "index file so the detector need not travel."
        )
    else:
        if not np.array_equal(clone.example(picks[0], epoch=0).doctored, a.doctored):
            problems.append("a pickled copy produced different data -- workers will disagree")

    return _finish(problems, generator, len(picks), strict, verbose)


def _spread(n: int, k: int) -> list[int]:
    """Indices spread over the whole index, not the first k -- an index is
    ordered by frame, so the first k examples are one scene."""
    k = min(k, n)
    return sorted({int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)})


def _where(generator, i: int) -> str:
    ref = generator.index[i]
    return f"{ref.key} {ref.box.width:.0f}x{ref.box.height:.0f}px score {ref.box.score:.2f}"


def _finish(problems: list[str], generator, n: int, strict: bool, verbose: bool) -> list[str]:
    if verbose:
        name = type(generator).__name__
        if problems:
            print(f"FAIL {name}: {len(problems)} problem(s) over {n} example(s)")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"OK   {name}: {n} examples checked, {len(generator)} in the index")
    if strict and problems:
        raise AssertionError("generator check failed:\n  " + "\n  ".join(problems))
    return problems
