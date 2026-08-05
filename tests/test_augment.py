"""Tests for the augmentation stage of step 06.

The invariant everything else rests on is narrow and easy to break: `clean` and
`doctored` must be the same image outside the paste rectangle, because that is
the promise the loss is built on -- "the surroundings are ground truth".  Most
of what follows is that property, approached from a different angle each time:
odd dtypes, boxes off the frame, a foreign patch pasted in, a custom op.

    conda run -n realsimir python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from realsimir import BBox, FunctionDetector, ShipAugmenter, ShipCropper, check_augmenter
from realsimir.augment import (
    DEFAULT_OPS,
    GeometricCorruption,
    Noise,
    PhotometricCorruption,
    Shift,
    available_ops,
    build_ops,
    register_op,
)
from realsimir.detectors import synthetic_frame

FRAME = synthetic_frame(240, 320)
SHIP = BBox(139.0, 124.0, 183.0, 140.0, score=0.9)


def outside(pair) -> tuple[np.ndarray, np.ndarray]:
    """The two images restricted to the pixels the paste never touched."""
    sel = pair.mask <= 0.0
    return pair.doctored[sel], pair.clean[sel]


# --------------------------------------------------------------------------- #
# stand-in ops
# --------------------------------------------------------------------------- #


@register_op("test_fixed_shift")
class FixedShift(GeometricCorruption):
    """Always the same translation, so the warp's direction can be asserted."""

    def __init__(self, dx: float = 5.0, dy: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dx, self.dy = dx, dy

    def sample(self, rng, ctx):
        return {"dx": self.dx, "dy": self.dy}

    def matrix(self, params, ctx):
        m = np.eye(3)
        m[0, 2], m[1, 2] = params["dx"], params["dy"]
        return m


@register_op("test_white")
class White(PhotometricCorruption):
    """Blows the donor out to 1.0 -- easy to see where it landed."""

    def sample(self, rng, ctx):
        return {}

    def apply(self, image, params, rng):
        return np.ones_like(image)


# --------------------------------------------------------------------------- #
# the invariant
# --------------------------------------------------------------------------- #


def test_default_augmenter_satisfies_its_contract():
    assert check_augmenter(ShipAugmenter(), verbose=False) == []


def test_nothing_changes_outside_the_rectangle():
    for seed in range(16):
        pair = ShipAugmenter().augment(FRAME, SHIP, seed=seed)
        got, want = outside(pair)
        assert np.array_equal(got, want)


def test_something_changes_inside_it():
    pair = ShipAugmenter().augment(FRAME, SHIP, seed=0)
    assert (pair.doctored != pair.clean).any()
    assert pair.difference()[pair.mask > 0.5].mean() > 0.0


def test_clean_is_the_input_and_the_input_is_not_touched():
    before = FRAME.copy()
    pair = ShipAugmenter().augment(FRAME, SHIP, seed=0)
    assert np.array_equal(pair.clean, FRAME)
    assert np.array_equal(before, FRAME)
    pair.doctored[:] = 0  # the caller owns what comes back
    assert np.array_equal(before, FRAME)


@pytest.mark.parametrize(
    "image",
    [
        FRAME,
        FRAME.astype(np.float32) / 255,
        np.repeat(FRAME[:, :, None], 3, axis=2),
        (FRAME.astype(np.uint16) * 257),
    ],
    ids=["uint8", "float32", "rgb", "uint16"],
)
def test_dtype_and_shape_survive(image):
    pair = ShipAugmenter().augment(image, SHIP, seed=1)
    assert pair.doctored.shape == image.shape == pair.clean.shape
    assert pair.doctored.dtype == image.dtype
    assert check_augmenter(ShipAugmenter(), image=image, verbose=False) == []


def test_dtype_argument_converts_the_pair():
    pair = ShipAugmenter(dtype=np.float32).augment(FRAME, SHIP, seed=1)
    assert pair.clean.dtype == pair.doctored.dtype == np.float32
    assert 0.0 <= pair.doctored.min() and pair.doctored.max() <= 1.0


# --------------------------------------------------------------------------- #
# reproducibility -- step 07 hands out one seed per example
# --------------------------------------------------------------------------- #


def test_a_seed_replays_exactly():
    aug = ShipAugmenter()
    a, b = aug.augment(FRAME, SHIP, seed=7), aug.augment(FRAME, SHIP, seed=7)
    assert np.array_equal(a.doctored, b.doctored)
    assert a.params["geometric"] == b.params["geometric"]


def test_different_seeds_differ():
    aug = ShipAugmenter()
    a, b = aug.augment(FRAME, SHIP, seed=1), aug.augment(FRAME, SHIP, seed=2)
    assert not np.array_equal(a.doctored, b.doctored)


def test_unseeded_calls_advance_the_augmenters_own_stream():
    aug = ShipAugmenter(seed=0)
    a, b = aug.augment(FRAME, SHIP), aug.augment(FRAME, SHIP)
    assert not np.array_equal(a.doctored, b.doctored)
    assert np.array_equal(ShipAugmenter(seed=0).augment(FRAME, SHIP).doctored, a.doctored)


def test_batch_seeds_are_consecutive():
    crops = ShipCropper(FunctionDetector(lambda im, p: [SHIP]), out_size=128).crop_all(FRAME) * 3
    aug = ShipAugmenter()
    pairs = aug.batch(crops, seed=100)
    assert [p.seed for p in pairs] == [100, 101, 102]
    assert np.array_equal(pairs[1].doctored, aug.augment_crop(crops[1], seed=101).doctored)


# --------------------------------------------------------------------------- #
# the paste rectangle -- idea.txt's "jitter the box"
# --------------------------------------------------------------------------- #


def test_rect_is_the_ship_grown_by_ten_to_twenty_percent():
    aug = ShipAugmenter(offset_frac=0.0)
    for seed in range(64):
        rect, _ = aug.rect_for(SHIP, np.random.default_rng(seed))
        assert 1.10 - 1e-9 <= rect.width / SHIP.width <= 1.20 + 1e-9
        assert 1.10 - 1e-9 <= rect.height / SHIP.height <= 1.20 + 1e-9


def test_rect_offset_still_leaves_the_ship_inside():
    """The box is a sloppy detection, not a different detection."""
    aug = ShipAugmenter()
    for seed in range(64):
        rect, params = aug.rect_for(SHIP, np.random.default_rng(seed))
        assert abs(params["offset"][0]) <= aug.offset_frac * SHIP.width
        assert rect.x_min <= SHIP.x_min and rect.x_max >= SHIP.x_max
        assert rect.y_min <= SHIP.y_min and rect.y_max >= SHIP.y_max


def test_mask_is_the_rectangle():
    pair = ShipAugmenter().augment(FRAME, SHIP, seed=0)
    x0, y0, x1, y1 = pair.rect.as_int()
    assert pair.mask[y0:y1, x0:x1].min() == 1.0
    assert pair.mask.sum() == pytest.approx((x1 - x0) * (y1 - y0))


def test_feather_softens_the_seam():
    pair = ShipAugmenter(feather_px=3.0).augment(FRAME, SHIP, seed=0)
    assert ((pair.mask > 0.01) & (pair.mask < 0.99)).any()
    got, want = outside(pair)  # the invariant holds against the *feathered* mask
    assert np.array_equal(got, want)


@pytest.mark.parametrize(
    "ship",
    [BBox(0, 0, 20, 16), BBox(300, 220, 340, 260), BBox(-12, -12, 14, 10)],
    ids=["corner", "off-the-right-edge", "mostly-outside"],
)
def test_boxes_at_the_frame_edge_are_clipped_not_crashed(ship):
    pair = ShipAugmenter().augment(FRAME, ship, seed=0)
    assert pair.doctored.shape == FRAME.shape
    got, want = outside(pair)
    assert np.array_equal(got, want)


def test_a_box_entirely_off_the_frame_returns_a_clean_pair():
    pair = ShipAugmenter().augment(FRAME, BBox(-80, -80, -40, -40), seed=0)
    assert np.array_equal(pair.doctored, pair.clean)
    assert "degenerate" in pair.params["skipped"]


# --------------------------------------------------------------------------- #
# ops: configuration and composition
# --------------------------------------------------------------------------- #


def test_ops_can_be_named_configured_or_handed_over_as_objects():
    assert [op.name for op in build_ops(["shift", "noise"])] == ["shift", "noise"]
    assert build_ops({"shift": {"max_px": 4}})[0].max_px == 4.0
    assert build_ops([Shift(max_px=2)])[0].max_px == 2.0
    assert [op.name for op in build_ops(None)] == list(DEFAULT_OPS)


def test_a_mapping_can_drop_a_default_op():
    ops = build_ops({name: {} for name in DEFAULT_OPS} | {"blur": None, "noise": False})
    assert "blur" not in [op.name for op in ops] and "noise" not in [op.name for op in ops]


def test_aliases_resolve_and_unknown_names_list_the_registry():
    assert build_ops(["offset"])[0].name == "shift"
    with pytest.raises(KeyError, match="rescale"):
        build_ops(["no_such_op"])


def test_registry_lists_the_builtins_and_rejects_a_clash():
    assert set(DEFAULT_OPS) <= set(available_ops())
    with pytest.raises(ValueError, match="already registered"):

        @register_op("shift")
        class Clash(PhotometricCorruption):
            def sample(self, rng, ctx):
                return {}

            def apply(self, image, params, rng):
                return image


def test_an_op_that_is_neither_family_is_refused():
    from realsimir.augment.ops import Corruption

    class Neither(Corruption):
        def sample(self, rng, ctx):
            return {}

    with pytest.raises(TypeError, match="GeometricCorruption"):
        ShipAugmenter(ops=[Neither()])


def test_probability_zero_means_never():
    aug = ShipAugmenter(ops=[Noise(sigma=0.2, p=0.0)])
    for seed in range(8):
        pair = aug.augment(FRAME, SHIP, seed=seed)
        assert pair.params["photometric"] == {}
        assert np.array_equal(pair.doctored, pair.clean)


def test_p_clean_returns_the_frame_untouched():
    pair = ShipAugmenter(p_clean=1.0).augment(FRAME, SHIP, seed=0)
    assert np.array_equal(pair.doctored, pair.clean)
    assert pair.params["skipped"] == "p_clean"


def test_no_ops_at_all_is_a_faithful_round_trip():
    """Warp + composite with nothing configured must not disturb the pixels."""
    pair = ShipAugmenter(ops=[]).augment(FRAME, SHIP, seed=0)
    assert np.array_equal(pair.doctored, pair.clean)


def test_geometric_ops_move_the_content_the_way_they_say():
    ramp = np.tile(np.linspace(0, 1, 320, dtype=np.float32), (240, 1))
    pair = ShipAugmenter(ops=[{"name": "test_fixed_shift", "dx": 5.0}]).augment(ramp, SHIP, seed=0)
    x0, y0, x1, y1 = pair.rect.as_int()
    inner = (slice(y0 + 2, y1 - 2), slice(x0 + 2, x1 - 2))
    shifted = pair.clean[y0 + 2 : y1 - 2, x0 + 2 - 5 : x1 - 2 - 5]
    assert np.allclose(pair.doctored[inner], shifted, atol=1e-5)


def test_photometric_ops_run_only_inside_the_rectangle():
    pair = ShipAugmenter(ops=["test_white"]).augment(FRAME, SHIP, seed=0)
    x0, y0, x1, y1 = pair.rect.as_int()
    assert (pair.doctored[y0:y1, x0:x1] == 255).all()
    got, want = outside(pair)
    assert np.array_equal(got, want)


def test_params_record_every_op_that_fired():
    pair = ShipAugmenter().augment(FRAME, SHIP, seed=0)
    fired = set(pair.params["geometric"]) | set(pair.params["photometric"])
    assert fired <= set(DEFAULT_OPS) and "shift" in fired
    record = pair.to_record()
    assert record["aug/shift/dx"] == pair.params["geometric"]["shift"]["dx"]
    assert all(isinstance(v, (int, float, bool, type(None))) for v in record.values())


def test_the_same_op_twice_is_recorded_twice():
    aug = ShipAugmenter(ops=[Shift(max_px=2), Shift(max_px=8)])
    params = aug.augment(FRAME, SHIP, seed=0).params["geometric"]
    assert set(params) == {"shift", "shift#2"}


# --------------------------------------------------------------------------- #
# pasting a foreign patch -- the test-time path, same machinery
# --------------------------------------------------------------------------- #


def test_a_source_patch_lands_in_the_rectangle():
    donor = np.full((40, 90), 255, np.uint8)
    pair = ShipAugmenter(ops=[]).augment(FRAME, SHIP, source=donor, seed=0)
    x0, y0, x1, y1 = pair.rect.as_int()
    assert pair.doctored[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2].min() > 250
    got, want = outside(pair)
    assert np.array_equal(got, want)


def test_a_source_patch_is_corrupted_by_the_same_ops():
    donor = np.full((40, 90), 128, np.uint8)
    aug = ShipAugmenter(ops=[{"name": "gain", "factor": 0.5}])
    pair = aug.augment(FRAME, SHIP, source=donor, seed=0)
    x0, y0, x1, y1 = pair.rect.as_int()
    assert pair.doctored[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4].mean() == pytest.approx(64, abs=2)


def test_match_levels_puts_the_donor_on_the_hosts_statistics():
    donor = np.full((40, 90), 10, np.uint8)
    plain = ShipAugmenter(ops=[]).augment(FRAME, SHIP, source=donor, seed=0)
    matched = ShipAugmenter(ops=[], match_levels=True).augment(FRAME, SHIP, source=donor, seed=0)
    sel = matched.mask > 0.5
    assert abs(int(matched.doctored[sel].mean()) - int(FRAME[sel].mean())) < 3
    assert plain.doctored[sel].mean() < 20


def test_a_colour_patch_can_be_pasted_into_a_grey_frame():
    donor = np.full((40, 90, 3), 200, np.uint8)
    pair = ShipAugmenter(ops=[]).augment(FRAME, SHIP, source=donor, seed=0)
    assert pair.doctored.shape == FRAME.shape
    x0, y0, x1, y1 = pair.rect.as_int()
    assert pair.doctored[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2].mean() == pytest.approx(200, abs=2)


# --------------------------------------------------------------------------- #
# with the cropper in front of it
# --------------------------------------------------------------------------- #


def test_augment_crop_works_in_patch_coordinates():
    crop = ShipCropper(FunctionDetector(lambda im, p: [SHIP]), out_size=256).crop_all(FRAME)[0]
    pair = ShipAugmenter()(crop, seed=0)
    assert pair.clean.shape == pair.doctored.shape == (256, 256)
    assert np.array_equal(pair.clean, crop.patch)
    assert pair.crop is crop
    # the rect tracks the ship where the *patch* puts it, not the frame
    assert pair.rect.iou(crop.detection_in_patch()) > 0.7
    got, want = outside(pair)
    assert np.array_equal(got, want)


def test_a_downscaled_window_is_augmented_at_the_size_the_model_sees():
    big = BBox(10, 10, 310, 230, score=1.0)
    crop = ShipCropper(FunctionDetector(lambda im, p: [big]), out_size=128).crop_all(FRAME)[0]
    assert crop.scale < 1.0
    pair = ShipAugmenter()(crop, seed=0)
    assert pair.clean.shape == (128, 128)
    assert pair.rect.width < 128 * 1.5


# --------------------------------------------------------------------------- #
# the checker itself
# --------------------------------------------------------------------------- #


def test_check_catches_a_leak_outside_the_mask():
    class Leaky(ShipAugmenter):
        def augment(self, image, ship, **kwargs):
            pair = super().augment(image, ship, **kwargs)
            pair.doctored[0, 0] = 255 - pair.doctored[0, 0]  # nowhere near the rect
            return pair

    problems = check_augmenter(Leaky(), verbose=False)
    assert any("outside the paste mask" in p for p in problems)
    with pytest.raises(AssertionError):
        check_augmenter(Leaky(), strict=True, verbose=False)


def test_check_catches_an_op_that_ignores_the_seed():
    @register_op("test_unseeded")
    class Unseeded(PhotometricCorruption):
        def sample(self, rng, ctx):
            return {"v": float(np.random.rand())}  # not the rng it was handed

        def apply(self, image, params, rng):
            return image * params["v"]

    problems = check_augmenter(ShipAugmenter(ops=["test_unseeded"]), verbose=False)
    assert any("same seed" in p for p in problems)
