"""Boxes read from a JSON file instead of a live model.

Two jobs, both of which make swapping the bounding box model cheaper:

  * **Any model, no integration.**  A newer detector that needs a conflicting
    environment (ultralytics, mmdet, a licensed SAM variant) never has to import
    into `realsimir`.  Run it wherever it lives, dump boxes to JSON in the schema
    below, and point the cropper at the file.
  * **Caching.**  Step 07's online generator will hit the same real frames every
    epoch; `dump_detections` freezes a YOLO pass so training does not pay for it
    repeatedly, and so a training run is reproducible against a fixed set of
    boxes even if the detector changes underneath it.

Schema -- an object mapping image key to a list of boxes:

    {
      "frame_0001.jpg": [
        {"x_min": 12.0, "y_min": 40.5, "x_max": 96.0, "y_max": 71.0,
         "score": 0.83, "label": "boat"}
      ],
      "frame_0002.jpg": []
    }

Cached boxes are not bit-identical to a fresh GPU pass -- batch composition
changes which cuDNN kernels run, which moved YOLOv3 corners by up to 0.024 px in
practice.  Well below the integer crop window, so the 256x256 patches come out
the same; do not expect exact float equality when comparing the two.

Keys may be absolute paths, basenames or stems; lookup tries all three, so a
file written on another machine still resolves.  `BBox.from_dict` also accepts
`bbox: [x0,y0,x1,y1]` / `confidence` / `class`, which covers most exporters
without a conversion step.  An empty list means "detector ran, found nothing" --
a missing key means "never ran", and raises unless `allow_missing`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..boxes import BBox
from .base import ShipDetector
from .registry import register_detector

__all__ = ["PrecomputedDetector", "dump_detections"]


@register_detector("precomputed", "json", "cached")
class PrecomputedDetector(ShipDetector):
    """Serve boxes from a JSON sidecar keyed by image path.

    boxes_file     the JSON described above.
    allow_missing  return [] for unknown images instead of raising.  Leave it
                   False while wiring a new model up -- a silent empty result
                   looks exactly like a detector that found no ships.
    """

    def __init__(
        self,
        boxes_file: str | os.PathLike,
        allow_missing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.boxes_file = Path(boxes_file)
        self.allow_missing = allow_missing
        with open(self.boxes_file) as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{self.boxes_file} must hold an object mapping image key -> [box, ...]")

        # index under every key form we might be asked for; first writer wins so
        # an exact path always beats a basename collision
        self._by_key: dict[str, list[BBox]] = {}
        for key, boxes in raw.items():
            parsed = [BBox.from_dict(b) for b in boxes]
            for k in self._key_forms(key):
                self._by_key.setdefault(k, parsed)

    @staticmethod
    def _key_forms(key: str | os.PathLike) -> list[str]:
        p = Path(str(key))
        forms = [str(key), p.name, p.stem]
        try:
            forms.append(str(p.resolve()))
        except OSError:  # pragma: no cover - resolve() on a weird path
            pass
        return forms

    def detect(self, image: np.ndarray | None, path: str | os.PathLike | None = None) -> list[BBox]:
        if path is None:
            raise ValueError("PrecomputedDetector needs the image path to look its boxes up")
        for k in self._key_forms(path):
            if k in self._by_key:
                return self._finalize(self._by_key[k], image.shape if image is not None else None)
        if self.allow_missing:
            return []
        raise KeyError(f"no detections stored for {path} in {self.boxes_file}")

    def __repr__(self) -> str:
        return f"PrecomputedDetector({self.boxes_file}, {len(self._by_key)} keys)"


def dump_detections(
    detector: ShipDetector,
    paths: Iterable[str | os.PathLike],
    out_file: str | os.PathLike,
    load: Callable | None = None,
    batch_size: int = 8,
    key: str = "path",
) -> Path:
    """Run `detector` over `paths` once and write a PrecomputedDetector file.

        dump_detections(build_detector("yolov3"), glob(...), "boxes.json")
        cropper = ShipCropper(build_detector("precomputed", boxes_file="boxes.json"))

    key: "path" (as given), "name" (basename) or "stem".
    """
    from ..imaging import load_image  # local: keeps PIL off the import path of the registry

    load = load or load_image
    paths = [str(p) for p in paths]
    keyer = {"path": lambda p: p, "name": lambda p: Path(p).name, "stem": lambda p: Path(p).stem}[key]

    out: dict[str, list[dict]] = {}
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        images = [load(p) for p in chunk]
        for p, boxes in zip(chunk, detector.detect_batch(images, chunk)):
            out[keyer(p)] = [b.to_dict() for b in boxes]

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fh:
        json.dump(out, fh, indent=1)
    return out_file
