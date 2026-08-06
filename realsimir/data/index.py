"""The crop index -- which ship, in which frame, is training example number i.

A frame is not a training example: a frame with three ships is three examples,
and a frame with none is not an example at all.  The index is that enumeration,
made once and then held:

    index = CropIndex.build(resolve_paths("real"), build_detector("yolov3"))
    index.save("boxes/real_yolov3.json")
    ...
    index = CropIndex.load("boxes/real_yolov3.json")

Why it is a file and not a pass at the top of every run:

  * **Cost.**  Detection is the expensive part of the pipeline and the only part
    that does not change between epochs.  Cropping and augmenting a 256x256
    patch is microseconds; a YOLOv3 forward pass over 12k frames is minutes of
    GPU time, repeated for nothing every time training restarts.
  * **Reproducibility.**  A run is trained against a fixed set of boxes.  If the
    bounding box model is swapped -- and idea.txt says it will be, YOLOv3-on-COCO
    is a placeholder -- the change shows up as a new index file, not as a silent
    difference between two runs of the same script.
  * **len().**  A map-style dataset has to know how many examples it has before
    it has produced any of them.  That is exactly `len(index)`.

The file format is the one `realsimir.detectors.precomputed` already defines, on
purpose: an index file *is* a PrecomputedDetector file and vice versa, so boxes
dumped by a model that cannot live in this environment can be trained on
directly, and an index built here can be served back through the detector
interface for the figures.  Frames the detector found nothing in are stored with
an empty list -- "ran, found nothing" has to be distinguishable from "never ran",
or a resumed build re-scans the empty frames for ever.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..boxes import BBox
from ..detectors.base import ShipDetector

__all__ = ["CropRef", "CropIndex"]


@dataclass(frozen=True)
class CropRef:
    """One training example, before any pixels are touched.

    path  the frame it lives in.
    box   the ship, in that frame's pixels.
    n     which detection it was within the frame, so a reference is traceable
          back to a line of the index file even after filtering.
    """

    path: str
    box: BBox
    n: int = 0

    @property
    def key(self) -> str:
        """Stable identity for logging: '<stem>#<n>'."""
        return f"{Path(self.path).stem}#{self.n}"


class CropIndex:
    """The enumerated (frame, ship) pairs the generator draws from.

    entries  the examples, in the order they were found.
    scanned  every frame the detector actually ran on, including the ones with
             no ships.  Kept so `save` can record them and a later `load` knows
             not to look at them again.
    meta     free-form provenance -- which detector, which pool, when.  Written
             alongside the boxes under a reserved key, ignored on load by
             anything that only wants boxes.
    """

    #: reserved key in the JSON file; not a frame path, so it cannot collide
    META_KEY = "__meta__"

    def __init__(
        self,
        entries: Iterable[CropRef],
        scanned: Iterable[str] | None = None,
        meta: dict | None = None,
    ):
        self.entries: list[CropRef] = list(entries)
        self.scanned: list[str] = list(scanned) if scanned is not None else _unique_paths(self.entries)
        self.meta: dict = dict(meta or {})

    # -- building ----------------------------------------------------------- #

    @classmethod
    def build(
        cls,
        paths: Sequence[str | os.PathLike],
        detector: ShipDetector,
        min_score: float = 0.0,
        min_side: float = 8.0,
        max_per_frame: int | None = None,
        batch_size: int = 8,
        load: Callable | None = None,
        on_error: str = "warn",
        progress: bool | Callable[[str], None] = True,
        meta: dict | None = None,
    ) -> "CropIndex":
        """Run `detector` over `paths` once and enumerate what it found.

        min_side   drop boxes thinner than this many source pixels.  A 4 px ship
                   carries no detail to reconstruct and its 256x256 window is
                   almost entirely sea -- an example that teaches the model
                   nothing about ships.  Raise it rather than lower it.
        min_score  drop boxes below this confidence.  Detector-specific: retune
                   when the bounding box model changes.
        max_per_frame  keep at most this many ships per frame, best first, so a
                   harbour scene with forty hulls does not dominate an epoch.
        on_error   'warn' (default) skips a frame that fails to load or detect
                   and keeps going -- an hour into a pass over 12k frames is not
                   where you want one corrupt JPEG to end the job.  'raise' is
                   for wiring a new back end up.
        """
        from ..imaging import load_image

        load = load or load_image
        say = _printer(progress)
        paths = [str(p) for p in paths]
        entries: list[CropRef] = []
        scanned: list[str] = []
        failed: list[str] = []
        started = last = time.time()

        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            images, keep = [], []
            for p in chunk:
                try:
                    images.append(load(p))
                    keep.append(p)
                except Exception as exc:  # unreadable file, truncated download, ...
                    if on_error == "raise":
                        raise
                    failed.append(p)
                    say(f"  ! {p}: {type(exc).__name__}: {exc}")
            if not keep:
                continue
            try:
                per_frame = detector.detect_batch(images, keep)
            except Exception as exc:
                if on_error == "raise":
                    raise
                failed.extend(keep)
                say(f"  ! detect_batch failed on {len(keep)} frame(s): {type(exc).__name__}: {exc}")
                continue

            for p, boxes in zip(keep, per_frame):
                scanned.append(p)
                kept = _select(boxes, min_score, min_side, max_per_frame)
                entries.extend(CropRef(p, b, n) for n, b in enumerate(kept))

            now = time.time()
            if now - last > 2.0:
                done = len(scanned) + len(failed)
                rate = done / max(now - started, 1e-6)
                say(f"  {done}/{len(paths)} frames, {len(entries)} ships, {rate:.0f} frames/s")
                last = now

        say(
            f"  {len(scanned)} frames scanned, {len(entries)} ships, "
            f"{len(failed)} unreadable, {time.time() - started:.1f}s"
        )
        info = {
            "detector": repr(detector),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "min_score": min_score,
            "min_side": min_side,
            "max_per_frame": max_per_frame,
            **(meta or {}),
        }
        return cls(entries, scanned, info)

    # -- serialisation ------------------------------------------------------ #

    def save(self, file: str | os.PathLike) -> Path:
        """Write the precomputed-detector schema: {path: [box, ...]}, plus meta."""
        by_path: dict[str, list[dict]] = {p: [] for p in self.scanned}
        for e in self.entries:
            by_path.setdefault(e.path, []).append(e.box.to_dict())
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as fh:
            json.dump({self.META_KEY: self.meta, **by_path}, fh, indent=1)
        return file

    @classmethod
    def load(
        cls,
        file: str | os.PathLike,
        min_score: float = 0.0,
        min_side: float = 0.0,
        max_per_frame: int | None = None,
        keep: Iterable[str] | None = None,
    ) -> "CropIndex":
        """Read an index (or any PrecomputedDetector file) back.

        The filters are applied again on load, so a single index built with
        permissive thresholds can serve several runs with different ones --
        cheaper than re-detecting, and the file stays the record of what the
        detector actually said.

        keep  restrict to these frame paths, for training on a subset of a pool
              that was indexed whole.
        """
        with open(file) as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{file} must hold an object mapping frame path -> [box, ...]")
        meta = raw.pop(cls.META_KEY, {}) or {}
        wanted = None if keep is None else {str(k) for k in keep}

        entries: list[CropRef] = []
        scanned: list[str] = []
        for path, boxes in raw.items():
            if wanted is not None and path not in wanted:
                continue
            scanned.append(path)
            selected = _select(
                [BBox.from_dict(b) for b in boxes], min_score, min_side, max_per_frame
            )
            entries.extend(CropRef(path, b, n) for n, b in enumerate(selected))
        meta = dict(meta, source_file=str(file))
        return cls(entries, scanned, meta)

    # -- using -------------------------------------------------------------- #

    def filtered(
        self,
        min_score: float = 0.0,
        min_side: float = 0.0,
        max_per_frame: int | None = None,
        predicate: Callable[[CropRef], bool] | None = None,
    ) -> "CropIndex":
        """A copy with fewer examples; `scanned` is preserved, so provenance is."""
        by_path: dict[str, list[BBox]] = {}
        for e in self.entries:
            if predicate is None or predicate(e):
                by_path.setdefault(e.path, []).append(e.box)
        entries = [
            CropRef(p, b, n)
            for p in self.scanned
            if p in by_path
            for n, b in enumerate(_select(by_path[p], min_score, min_side, max_per_frame))
        ]
        return CropIndex(entries, self.scanned, self.meta)

    def frames(self) -> list[str]:
        """The frames that actually carry an example, in index order."""
        return _unique_paths(self.entries)

    def stats(self) -> dict:
        sides = [max(e.box.width, e.box.height) for e in self.entries]
        scores = [e.box.score for e in self.entries]
        n_frames = len(self.frames())
        return {
            "examples": len(self.entries),
            "frames_with_ships": n_frames,
            "frames_scanned": len(self.scanned),
            "hit_rate": n_frames / len(self.scanned) if self.scanned else 0.0,
            "ships_per_hit": len(self.entries) / n_frames if n_frames else 0.0,
            "box_side_px": _quartiles(sides),
            "score": _quartiles(scores),
            "labels": _counts(e.box.label for e in self.entries),
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return CropIndex(self.entries[i], self.scanned, self.meta)
        return self.entries[i]

    def __iter__(self):
        return iter(self.entries)

    def __repr__(self) -> str:
        det = self.meta.get("detector", "?")
        return f"CropIndex({len(self.entries)} examples in {len(self.frames())} frames, detector={det})"


# --------------------------------------------------------------------------- #


def _select(
    boxes: Sequence[BBox], min_score: float, min_side: float, max_per_frame: int | None
) -> list[BBox]:
    """One frame's boxes, thresholded and truncated -- best score first."""
    kept = [
        b
        for b in boxes
        if b.score >= min_score and min(b.width, b.height) >= min_side
    ]
    kept.sort(key=lambda b: (b.score, b.area), reverse=True)
    return kept[:max_per_frame] if max_per_frame else kept


def _unique_paths(entries: Sequence[CropRef]) -> list[str]:
    return list(dict.fromkeys(e.path for e in entries))


def _quartiles(values: Sequence[float]) -> dict:
    if not values:
        return {}
    import numpy as np

    q = np.percentile(values, [0, 25, 50, 75, 100])
    return {"min": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "max": float(q[4])}


def _counts(labels: Iterable[str]) -> dict:
    out: dict[str, int] = {}
    for label in labels:
        out[label] = out.get(label, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _printer(progress: bool | Callable[[str], None]) -> Callable[[str], None]:
    if progress is True:
        return print
    if progress is False or progress is None:
        return lambda _msg: None
    return progress
