"""Box and crop geometry -- the vocabulary every detector back end speaks.

Deliberately free of torch / cv2 / any model dependency: a new bounding box
model only has to produce `BBox` objects in source-image pixels, and everything
downstream (cropping, augmentation, pasting) keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["BBox", "ShipCrop"]


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in source-image pixels.  Max coordinates are exclusive."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float = 1.0
    label: str = "ship"

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def expanded(self, frac: float) -> "BBox":
        """Grow each side by `frac` of its length (0.2 -> box 20% larger)."""
        dx, dy = self.width * frac / 2.0, self.height * frac / 2.0
        return replace(
            self,
            x_min=self.x_min - dx,
            y_min=self.y_min - dy,
            x_max=self.x_max + dx,
            y_max=self.y_max + dy,
        )

    def shifted(self, dx: float, dy: float) -> "BBox":
        return replace(
            self,
            x_min=self.x_min + dx,
            y_min=self.y_min + dy,
            x_max=self.x_max + dx,
            y_max=self.y_max + dy,
        )

    def scaled(self, s: float) -> "BBox":
        return replace(
            self, x_min=self.x_min * s, y_min=self.y_min * s, x_max=self.x_max * s, y_max=self.y_max * s
        )

    def clipped(self, width: int, height: int) -> "BBox":
        return replace(
            self,
            x_min=float(np.clip(self.x_min, 0, width)),
            y_min=float(np.clip(self.y_min, 0, height)),
            x_max=float(np.clip(self.x_max, 0, width)),
            y_max=float(np.clip(self.y_max, 0, height)),
        )

    def rounded(self) -> "BBox":
        return replace(
            self,
            x_min=float(np.floor(self.x_min)),
            y_min=float(np.floor(self.y_min)),
            x_max=float(np.ceil(self.x_max)),
            y_max=float(np.ceil(self.y_max)),
        )

    def as_int(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) suitable for slicing; x1/y1 exclusive."""
        b = self.rounded()
        return int(b.x_min), int(b.y_min), int(b.x_max), int(b.y_max)

    def iou(self, other: "BBox") -> float:
        ix = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        iy = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "score": self.score,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BBox":
        """Accepts either x_min/.. or xyxy-style keys, so exports from other
        detectors can be read back without a bespoke converter."""
        if "x_min" in d:
            x0, y0, x1, y1 = d["x_min"], d["y_min"], d["x_max"], d["y_max"]
        elif "bbox" in d:  # [x0, y0, x1, y1]
            x0, y0, x1, y1 = d["bbox"]
        else:
            x0, y0, x1, y1 = d["x0"], d["y0"], d["x1"], d["y1"]
        return cls(
            float(x0),
            float(y0),
            float(x1),
            float(y1),
            score=float(d.get("score", d.get("confidence", 1.0))),
            label=str(d.get("label", d.get("class", "ship"))),
        )


@dataclass
class ShipCrop:
    """One extracted window plus the geometry needed to put it back."""

    patch: np.ndarray  # (out_size, out_size) or (out_size, out_size, C)
    window: BBox  # region of the source frame `patch` covers
    detection: BBox  # the raw ship box, in source-frame pixels
    paste_box: BBox  # detection grown by paste_context, in source-frame pixels
    scale: float  # patch pixels per source pixel (1.0 == native resolution)
    path: str | None = None

    def detection_in_patch(self) -> BBox:
        """The ship box expressed in `patch` coordinates."""
        return self.detection.shifted(-self.window.x_min, -self.window.y_min).scaled(self.scale)

    def paste_box_in_patch(self) -> BBox:
        return self.paste_box.shifted(-self.window.x_min, -self.window.y_min).scaled(self.scale)
