"""Acceptance test for a bounding box back end.

Run this against a new detector before wiring it into training.  It catches the
mistakes that are otherwise invisible until the crops come out wrong: boxes left
in the model's input resolution, normalised 0-1 coordinates, (y, x) ordering, a
`detect_batch` that disagrees with `detect`, in-place edits of the caller's
frame.

    from realsimir import build_detector, check_detector
    check_detector(build_detector("yolov3"), samples=["/workspace/data/.../frame.jpg"])

or from the shell:

    python -m realsimir.detectors check yolov3 --image /path/to/frame.jpg
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..boxes import BBox
from .base import ShipDetector

__all__ = ["check_detector", "synthetic_frame"]

Sample = tuple[np.ndarray | None, str | None]


def synthetic_frame(height: int = 240, width: int = 320, seed: int = 0) -> np.ndarray:
    """A cheap stand-in IR frame: warm sky gradient, horizon, one bright blob."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0.35, 0.15, height, dtype=np.float32)[:, None]
    frame = np.repeat(y, width, axis=1)
    frame[int(height * 0.6) :] = 0.45
    cy, cx = int(height * 0.55), int(width * 0.5)
    frame[cy - 8 : cy + 8, cx - 22 : cx + 22] = 0.9
    frame += rng.normal(0, 0.01, frame.shape).astype(np.float32)
    return np.clip(frame * 255, 0, 255).astype(np.uint8)


def _as_samples(samples: Sequence[Any] | None) -> list[Sample]:
    if not samples:
        return [(synthetic_frame(seed=0), None), (synthetic_frame(seed=1), None)]
    from ..imaging import load_image

    out: list[Sample] = []
    for s in samples:
        if isinstance(s, tuple):
            out.append(s)
        elif isinstance(s, np.ndarray):
            out.append((s, None))
        else:
            out.append((load_image(s), str(s)))
    return out


def check_detector(
    detector: ShipDetector,
    samples: Sequence[Any] | None = None,
    strict: bool = False,
    verbose: bool = True,
    tol_px: float = 1.0,
) -> list[str]:
    """Check a detector against the contract in base.py.

    samples  image paths, arrays, or (array, path) pairs.  Path-keyed back ends
             (arete, precomputed) need real paths; pixel-based ones are happy
             with the synthetic frames used by default.
    strict   raise instead of returning the problem list.
    tol_px   how far detect_batch may drift from detect before it counts as a
             disagreement.  Not zero: a GPU picks different kernels for batch 1
             than for batch N, so the same YOLOv3 weights move a corner by a
             fraction of a pixel between the two paths.  Losing or gaining a
             box is always a failure regardless of this.

    Returns the list of problems -- empty means the detector is usable.
    """
    problems: list[str] = []

    def bad(msg: str) -> None:
        problems.append(msg)

    if not isinstance(detector, ShipDetector):
        bad(f"{type(detector).__name__} does not subclass ShipDetector")
        if not hasattr(detector, "detect"):
            return _finish(problems, detector, 0, strict, verbose)

    pairs = _as_samples(samples)
    singles: list[list[BBox]] = []

    for i, (image, path) in enumerate(pairs):
        tag = f"sample {i}" + (f" ({Path(path).name})" if path else "")
        before = None if image is None else image.copy()
        try:
            boxes = detector.detect(image, path)
        except Exception as exc:
            singles.append([])
            bad(f"{tag}: detect raised {type(exc).__name__}: {exc}")
            continue
        singles.append(boxes if isinstance(boxes, list) else [])

        if boxes is None:
            bad(f"{tag}: detect returned None; it must return [] when nothing is found")
            continue
        if not isinstance(boxes, list):
            bad(f"{tag}: detect returned {type(boxes).__name__}, expected list")
            continue
        if before is not None and not np.array_equal(before, image):
            bad(f"{tag}: detect modified the input image in place")

        h, w = (image.shape[:2] if image is not None else (None, None))
        for b in boxes:
            if not isinstance(b, BBox):
                bad(f"{tag}: got {type(b).__name__}, expected BBox")
                continue
            vals = (b.x_min, b.y_min, b.x_max, b.y_max)
            if not all(np.isfinite(v) for v in vals):
                bad(f"{tag}: non-finite box {vals}")
                continue
            if b.width <= 0 or b.height <= 0:
                bad(f"{tag}: degenerate box {vals} (is it (y0,x0,y1,x1) rather than (x0,y0,x1,y1)?)")
            if h is not None and (b.x_max > w + 1 or b.y_max > h + 1 or b.x_min < -1 or b.y_min < -1):
                bad(
                    f"{tag}: box {vals} falls outside the {w}x{h} frame -- boxes must be in "
                    "source-image pixels, clipped to the frame (call self._finalize)"
                )
            if h is not None and max(b.width, b.height) <= 1.5 and b.x_max <= 1.5:
                bad(f"{tag}: box {vals} looks normalised to 0-1; scale it back to pixels")
            if not isinstance(b.label, str):
                bad(f"{tag}: label {b.label!r} is {type(b.label).__name__}, expected str")

    # detect_batch must agree with detect, and must not reorder frames
    images = [im for im, _ in pairs]
    paths = [p for _, p in pairs]
    try:
        batched = detector.detect_batch(images, paths)
        if not isinstance(batched, list) or len(batched) != len(pairs):
            bad(
                f"detect_batch returned {len(batched) if hasattr(batched, '__len__') else '?'} "
                f"results for {len(pairs)} images"
            )
        else:
            for i, (a, b) in enumerate(zip(singles, batched)):
                bad_pair = _batch_disagreement(a, b, tol_px=tol_px)
                if bad_pair:
                    bad(f"sample {i}: detect_batch disagrees with detect -- {bad_pair}")
        if detector.detect_batch([], []) != []:
            bad("detect_batch([]) must return []")
    except Exception as exc:
        bad(f"detect_batch raised {type(exc).__name__}: {exc}")

    return _finish(problems, detector, len(pairs), strict, verbose)


def _batch_disagreement(single: list[BBox], batch: list[BBox], tol_px: float) -> str:
    """'' when the two agree to within tol_px, otherwise what differs."""
    if len(single) != len(batch):
        return f"{len(single)} boxes alone vs {len(batch)} batched"
    for a, b in zip(single, batch):
        if a.label != b.label:
            return f"label {a.label!r} vs {b.label!r}"
        drift = max(
            abs(a.x_min - b.x_min), abs(a.y_min - b.y_min), abs(a.x_max - b.x_max), abs(a.y_max - b.y_max)
        )
        if drift > tol_px:
            return f"box moved {drift:.2f} px between the single and batched call"
        if abs(a.score - b.score) > 0.05:
            return f"score {a.score:.3f} vs {b.score:.3f}"
    return ""


def _finish(problems: list[str], detector, n: int, strict: bool, verbose: bool) -> list[str]:
    if verbose:
        name = type(detector).__name__
        if problems:
            print(f"FAIL {name}: {len(problems)} problem(s) over {n} sample(s)")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"OK   {name}: satisfies the ShipDetector contract over {n} sample(s)")
    if strict and problems:
        raise AssertionError(f"{type(detector).__name__} violates the detector contract:\n  " + "\n  ".join(problems))
    return problems
