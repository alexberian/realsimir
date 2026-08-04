"""Ground-truth boxes for the synthetic arete renders, read from the metadata CSVs.

No network, no failure modes -- the right back end for the sim side, and the
reference the real-imagery detectors get compared against.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

from ..boxes import BBox
from ..paths import ARETE_ROOT
from .base import ShipDetector
from .registry import register_detector

__all__ = ["AreteMetadataDetector"]


@register_detector("arete", "metadata", "gt")
class AreteMetadataDetector(ShipDetector):
    """Boxes straight out of the render metadata.

    Layout is  <root>/<split>/images/<scene>/<modality>/<stem>.png  with the boxes
    in <root>/<split>/metadata/<scene>/*.csv, keyed by `image_name` == stem.  The
    LWIR columns carry a `_lwir` suffix; the unsuffixed ones belong to the EO
    ('color') render, which has a different resolution -- so the modality has to
    come from the path, not be assumed.

    CSV maxima are inclusive pixel indices; BBox maxima are exclusive, hence +1.
    """

    def __init__(
        self,
        root: str | os.PathLike = ARETE_ROOT,
        box: str = "tight",
        require_visible: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if box not in ("tight", "loose"):
            raise ValueError(f"box must be 'tight' or 'loose', got {box!r}")
        self.root = Path(root)
        self.box = box
        self.require_visible = require_visible
        self._cache: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _parse_path(path: Path) -> tuple[str, str, str, str]:
        """.../<split>/images/<scene>/<modality>/<stem>.png"""
        parts = path.parts
        try:
            i = len(parts) - 1 - parts[::-1].index("images")
        except ValueError as exc:
            raise ValueError(f"not an arete image path (no 'images' component): {path}") from exc
        if len(parts) < i + 4:
            raise ValueError(f"not an arete image path: {path}")
        return parts[i - 1], parts[i + 1], parts[i + 2], path.stem  # split, scene, modality, stem

    def _table(self, split: str, scene: str) -> dict:
        key = (split, scene)
        if key not in self._cache:
            import pandas as pd

            csvs = sorted(glob.glob(str(self.root / split / "metadata" / scene / "*.csv")))
            if not csvs:
                raise FileNotFoundError(f"no metadata csv for {split}/{scene} under {self.root}")
            # a few scenes (trawler_generic_0000) are split across several csvs
            df = pd.concat([pd.read_csv(c, low_memory=False) for c in csvs], ignore_index=True)
            # image_name is not a key in the benchmark split: 72/432 atlant_0000 rows
            # are re-renders of the same case under a later simulation_cycle, and only
            # one file per name exists on disk.  The boxes agree (71/72 exactly), so
            # take the latest render and move on.
            df = df.drop_duplicates(subset="image_name", keep="last")
            self._cache[key] = df.set_index("image_name").to_dict("index")
        return self._cache[key]

    def detect(self, image: np.ndarray | None, path: str | os.PathLike | None = None) -> list[BBox]:
        if path is None:
            raise ValueError("AreteMetadataDetector needs the image path to look up its metadata")
        path = Path(path)
        split, scene, modality, stem = self._parse_path(path)
        row = self._table(split, scene).get(stem)
        if row is None:
            return []

        suffix = "_lwir" if modality == "lwir" else ""
        visible = row.get(f"is_target_visible{suffix}")
        if self.require_visible and visible is not None and not bool(visible):
            return []

        try:
            x0 = float(row[f"x_min_{self.box}{suffix}"])
            x1 = float(row[f"x_max_{self.box}{suffix}"]) + 1.0
            y0 = float(row[f"y_min_{self.box}{suffix}"])
            y1 = float(row[f"y_max_{self.box}{suffix}"]) + 1.0
        except KeyError:
            return []
        box = BBox(x0, y0, x1, y1, score=1.0, label=str(row.get("category", "ship")))
        return self._finalize([box], image.shape if image is not None else None)

    def __repr__(self) -> str:
        return f"AreteMetadataDetector(box={self.box!r}, root={self.root})"
