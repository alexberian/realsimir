"""Tests for the swappable-detector machinery of step 05.

These are the guard rails for "drop in a different bounding box model": the
registry resolves names, the contract checker catches the classic mistakes, and
the cropper's geometry does not depend on which back end produced the box.

Nothing here needs torch, weights or /workspace/data -- run with

    conda run -n realsimir python -m pytest tests -q
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from realsimir import (
    BBox,
    FunctionDetector,
    ShipCropper,
    ShipDetector,
    available_detectors,
    build_detector,
    check_detector,
)
from realsimir.detectors import register_detector, synthetic_frame
from realsimir.detectors.precomputed import PrecomputedDetector, dump_detections

FRAME = synthetic_frame(240, 320)
SHIP = BBox(139.0, 124.0, 183.0, 140.0, score=0.9, label="boat")


# --------------------------------------------------------------------------- #
# stand-in back ends
# --------------------------------------------------------------------------- #


@register_detector("test_fixed")
class FixedDetector(ShipDetector):
    """Returns one box, always.  Stands in for a real model in these tests."""

    def __init__(self, box: BBox = SHIP, **kwargs):
        super().__init__(**kwargs)
        self.box = box

    def detect(self, image, path=None):
        return self._finalize([self.box], image.shape if image is not None else None)


class NormalisedDetector(ShipDetector):
    """A detector that forgot to scale 0-1 coordinates back to pixels."""

    def detect(self, image, path=None):
        return [BBox(0.42, 0.51, 0.57, 0.58)]


class InvertedDetector(ShipDetector):
    """Corners the wrong way round -- x_max < x_min, so there is nothing to crop."""

    def detect(self, image, path=None):
        return [BBox(SHIP.x_max, SHIP.y_max, SHIP.x_min, SHIP.y_min)]


class InputResolutionDetector(ShipDetector):
    """Boxes left in the model's 416x416 input frame instead of source pixels."""

    def detect(self, image, path=None):
        return [BBox(300, 300, 380, 340)]


class MutatingDetector(ShipDetector):
    def detect(self, image, path=None):
        image[:] = 0
        return []


class DesyncedBatchDetector(ShipDetector):
    def detect(self, image, path=None):
        return [SHIP]

    def detect_batch(self, images, paths=None):
        return [[] for _ in images]


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_builtins_are_registered_without_importing_them():
    for name in ("yolov3", "arete", "precomputed"):
        assert name in available_detectors()


def test_import_realsimir_does_not_import_torch():
    """The whole point of the lazy registry: listing models is cheap."""
    code = "import sys, realsimir; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", "importing realsimir pulled in torch"


def test_build_by_name_and_alias():
    assert isinstance(build_detector("test_fixed"), FixedDetector)
    assert isinstance(build_detector("TEST_FIXED"), FixedDetector)


def test_build_passes_constructor_arguments():
    box = BBox(1, 2, 30, 40)
    det = build_detector("test_fixed", box=box)
    assert det.detect(FRAME)[0].x_max == 30


def test_build_from_config_dict_with_overrides():
    cfg = {"name": "test_fixed", "rename_labels": {"boat": "ship"}}
    det = build_detector(cfg, min_side=2.0)
    assert det.min_side == 2.0
    assert det.detect(FRAME)[0].label == "ship"


def test_build_from_dotted_path(tmp_path, monkeypatch):
    """The escape hatch: a detector living entirely outside this repo."""
    (tmp_path / "outside_model.py").write_text(
        "from realsimir import BBox, ShipDetector\n"
        "class OutsideDetector(ShipDetector):\n"
        "    def __init__(self, tag='x', **kw):\n"
        "        super().__init__(**kw)\n"
        "        self.tag = tag\n"
        "    def detect(self, image, path=None):\n"
        "        return self._finalize([BBox(10, 10, 60, 40, 0.5, self.tag)], image.shape)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    det = build_detector("outside_model:OutsideDetector", tag="vessel")
    assert check_detector(det, verbose=False) == []
    assert det.detect(FRAME)[0].label == "vessel"


def test_dotted_path_must_name_a_ship_detector():
    with pytest.raises(TypeError):
        build_detector("pathlib:Path")


def test_build_accepts_an_instance_unchanged():
    det = FixedDetector()
    assert build_detector(det) is det
    with pytest.raises(TypeError):
        build_detector(det, min_side=3.0)


def test_unknown_name_lists_what_is_available():
    with pytest.raises(KeyError, match="yolov3"):
        build_detector("no_such_model")


def test_cannot_shadow_an_existing_name():
    with pytest.raises(ValueError, match="already registered"):

        @register_detector("test_fixed")
        class Clash(ShipDetector):
            def detect(self, image, path=None):
                return []


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #


def test_good_detector_passes():
    assert check_detector(FixedDetector(), verbose=False) == []


def test_function_detector_accepts_tuples_and_dicts():
    det = FunctionDetector(lambda im, p: [(10, 20, 60, 44, 0.7, "boat")])
    assert check_detector(det, verbose=False) == []
    assert det.detect(FRAME)[0].label == "boat"

    det = FunctionDetector(lambda im, p: [{"bbox": [10, 20, 60, 44], "confidence": 0.7}])
    assert det.detect(FRAME)[0].score == pytest.approx(0.7)


@pytest.mark.parametrize(
    "cls, expected",
    [
        (NormalisedDetector, "normalised"),
        (InvertedDetector, "degenerate"),
        (InputResolutionDetector, "outside the"),
        (MutatingDetector, "in place"),
        (DesyncedBatchDetector, "disagrees"),
    ],
)
def test_contract_catches_the_classic_mistakes(cls, expected):
    problems = check_detector(cls(), verbose=False)
    assert any(expected in p for p in problems), problems


def test_strict_raises():
    with pytest.raises(AssertionError):
        check_detector(NormalisedDetector(), strict=True, verbose=False)


def test_path_keyed_detector_reports_a_missing_path_instead_of_crashing():
    problems = check_detector(build_detector("arete"), verbose=False)
    assert any("detect raised" in p for p in problems)


# --------------------------------------------------------------------------- #
# base-class services every back end inherits
# --------------------------------------------------------------------------- #


def test_finalize_clips_drops_renames_and_sorts():
    class Messy(ShipDetector):
        def detect(self, image, path=None):
            boxes = [
                BBox(-20, -10, 100, 60, score=0.3, label="boat"),  # off the frame
                BBox(10, 10, 10.2, 10.2, score=0.99, label="boat"),  # sub-pixel
                BBox(5, 5, 90, 40, score=0.8, label="person"),  # wrong class
                BBox(50, 50, 120, 90, score=0.5, label="boat"),
            ]
            return self._finalize(boxes, image.shape)

    boxes = Messy(keep_labels={"boat"}, rename_labels={"boat": "ship"}).detect(FRAME)
    assert [b.score for b in boxes] == [0.5, 0.3]  # sorted, filtered
    assert {b.label for b in boxes} == {"ship"}
    assert boxes[-1].x_min == 0 and boxes[-1].y_min == 0  # clipped


def test_detect_batch_default_matches_detect():
    det = FixedDetector()
    frames = [synthetic_frame(seed=i) for i in range(3)]
    assert det.detect_batch(frames) == [det.detect(f) for f in frames]
    with pytest.raises(ValueError):
        det.detect_batch(frames, ["only-one-path"])


# --------------------------------------------------------------------------- #
# precomputed boxes -- the "model that cannot live in this env" path
# --------------------------------------------------------------------------- #


def test_dump_and_reload_round_trip(tmp_path):
    img_paths = []
    for i in range(3):
        p = tmp_path / f"frame_{i}.npy"
        np.save(p, FRAME)
        img_paths.append(p)

    out = dump_detections(FixedDetector(), img_paths, tmp_path / "boxes.json", load=np.load)
    stored = json.loads(out.read_text())
    assert len(stored) == 3

    det = build_detector("precomputed", boxes_file=out)
    assert det.detect(FRAME, img_paths[0])[0].to_dict() == SHIP.to_dict()
    # a key written elsewhere still resolves by basename
    assert det.detect(FRAME, "/somewhere/else/frame_1.npy")


def test_missing_key_is_loud_by_default(tmp_path):
    f = tmp_path / "boxes.json"
    f.write_text(json.dumps({"a.png": [SHIP.to_dict()]}))
    with pytest.raises(KeyError):
        PrecomputedDetector(f).detect(FRAME, "b.png")
    assert PrecomputedDetector(f, allow_missing=True).detect(FRAME, "b.png") == []


def test_reads_foreign_key_spellings(tmp_path):
    f = tmp_path / "boxes.json"
    f.write_text(json.dumps({"a.png": [{"bbox": [1, 2, 30, 40], "confidence": 0.5, "class": "vessel"}]}))
    box = PrecomputedDetector(f).detect(FRAME, "a.png")[0]
    assert (box.x_min, box.y_max, box.score, box.label) == (1, 40, 0.5, "vessel")


# --------------------------------------------------------------------------- #
# the cropper does not care which back end it holds
# --------------------------------------------------------------------------- #


def test_cropper_takes_a_detector_spec():
    cropper = ShipCropper("test_fixed", out_size=64)
    crops = cropper.crop_all(FRAME)
    assert len(crops) == 1 and crops[0].patch.shape == (64, 64)


def test_small_ship_is_cropped_at_native_resolution():
    crop = ShipCropper("test_fixed", out_size=256).crop_all(FRAME)[0]
    assert crop.scale == 1.0
    assert crop.window.width == 256
    # the ship lands where the geometry says it does
    box = crop.detection_in_patch()
    assert box.width == pytest.approx(SHIP.width)
    src_cx, src_cy = SHIP.center
    assert box.center[0] == pytest.approx(src_cx - crop.window.x_min, abs=1)
    assert box.center[1] == pytest.approx(src_cy - crop.window.y_min, abs=1)


def test_big_ship_is_downscaled_to_fit():
    big = BBox(10, 10, 310, 230, score=1.0)
    crop = ShipCropper(FixedDetector(box=big), out_size=256).crop_all(FRAME)[0]
    assert crop.scale < 1.0
    assert crop.patch.shape == (256, 256)
    assert crop.window.width >= big.expanded(0.15).width


def test_window_off_frame_is_padded_not_shifted():
    corner = BBox(0, 0, 20, 16, score=1.0)
    crop = ShipCropper(FixedDetector(box=corner), out_size=256).crop_all(FRAME)[0]
    assert crop.patch.shape == (256, 256)
    assert crop.window.x_min < 0 and crop.window.y_min < 0


def test_min_score_and_max_crops_are_cropper_policy_not_detector_policy():
    class Three(ShipDetector):
        def detect(self, image, path=None):
            return self._finalize(
                [
                    BBox(10, 10, 60, 40, score=0.9),
                    BBox(80, 80, 130, 110, score=0.5),
                    BBox(150, 120, 200, 150, score=0.1),
                ],
                image.shape,
            )

    assert len(ShipCropper(Three()).crop_all(FRAME)) == 3
    assert len(ShipCropper(Three(), min_score=0.4).crop_all(FRAME)) == 2
    assert len(ShipCropper(Three(), max_crops=1).crop_all(FRAME)) == 1
