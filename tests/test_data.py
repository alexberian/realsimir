"""Tests for the online data generator of step 07.

The stream has three properties the training loop cannot check for itself, and
most of what follows is those:

  * example i is the same ship in the same window in every epoch -- the clean
    patch is the diffusion target, so it must not move underneath the model;
  * the corruption is a pure function of (seed, epoch, index), so workers,
    batch size and visiting order cannot change the data;
  * the index is the enumeration of ships, not of frames, and survives a round
    trip through its file.

Everything here runs on frames written to a tmp directory by a stand-in
detector: no data on /workspace, no weights, no GPU.

    conda run -n realsimir python -m pytest tests -q
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
from PIL import Image

from realsimir import BBox, FunctionDetector, ShipAugmenter
from realsimir.data import (
    CompositeGenerator,
    CropIndex,
    FramePool,
    ShipDataGenerator,
    available_pools,
    check_generator,
    describe_pools,
    register_pool,
    resolve_paths,
)
from realsimir.detectors import build_detector, synthetic_frame

# two ships per frame, at fixed places, so an index has a known length
SHIPS = [BBox(139.0, 124.0, 183.0, 140.0, score=0.9), BBox(40.0, 150.0, 78.0, 166.0, score=0.6)]
N_FRAMES = 6


@pytest.fixture(scope="module")
def frames(tmp_path_factory) -> list[str]:
    """N_FRAMES synthetic IR frames on disk, each carrying the same two ships."""
    d = tmp_path_factory.mktemp("frames")
    paths = []
    for i in range(N_FRAMES):
        p = d / f"frame_{i:03d}.png"
        Image.fromarray(synthetic_frame(240, 320, seed=i)).save(p)
        paths.append(str(p))
    return paths


def _two_ships(image, path):
    """Module level, not a lambda: a generator is pickled to reach a dataloader
    worker, and a config that cannot cross that boundary is a real problem the
    checker is supposed to report."""
    return list(SHIPS)


@pytest.fixture(scope="module")
def detector():
    """A detector that finds both ships in every frame, and nothing else."""
    return FunctionDetector(_two_ships)


def make_generator(frames, detector, **kwargs) -> ShipDataGenerator:
    kwargs.setdefault("cropper", {"out_size": 128})
    return ShipDataGenerator(source=frames, detector=detector, min_side=4.0, **kwargs)


# --------------------------------------------------------------------------- #
# pools
# --------------------------------------------------------------------------- #


def test_the_builtin_pools_are_registered_and_described():
    assert {"real", "raysense", "massmind", "arete"} <= set(available_pools())
    assert all(describe_pools()[name] for name in available_pools())


def test_resolve_paths_takes_a_pattern_a_directory_or_a_list(frames, tmp_path):
    d = str(np.__file__)  # any existing file, to prove a file resolves to itself
    assert resolve_paths(frames) == sorted(frames)
    assert resolve_paths(f"{frames[0][:-7]}*.png") == sorted(frames)
    assert resolve_paths(str(tmp_path)) == []
    assert resolve_paths(d) == [d]
    with pytest.raises(FileNotFoundError):
        resolve_paths("no_such_pool")


def test_a_pool_can_be_registered_and_sampled(frames):
    pool = register_pool(FramePool("test_pool", (f"{frames[0][:-7]}*.png",), doc="the tmp frames"))
    assert len(pool) == N_FRAMES
    assert resolve_paths("test_pool") == sorted(frames)
    assert resolve_paths("test_pool", stride=2) == sorted(frames)[::2]
    # a seeded sample is a sample, not the first N, and it replays
    a = resolve_paths("test_pool", limit=3, seed=1)
    assert len(a) == 3 and a == resolve_paths("test_pool", limit=3, seed=1)
    assert resolve_paths("test_pool", limit=3, seed=None) == sorted(frames)[:3]


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #


def test_the_index_enumerates_ships_not_frames(frames, detector):
    index = CropIndex.build(frames, detector, min_side=4.0, progress=False)
    assert len(index) == N_FRAMES * len(SHIPS)
    assert len(index.frames()) == N_FRAMES
    assert index.stats()["ships_per_hit"] == pytest.approx(2.0)
    assert [e.n for e in index.entries[:2]] == [0, 1]


def test_the_index_round_trips_through_its_file(frames, detector, tmp_path):
    index = CropIndex.build(frames, detector, min_side=4.0, progress=False)
    f = index.save(tmp_path / "boxes.json")
    back = CropIndex.load(f)
    assert len(back) == len(index)
    assert [(e.path, e.box.as_int()) for e in back] == [(e.path, e.box.as_int()) for e in index]
    assert back.meta["detector"] == index.meta["detector"]


def test_the_index_file_is_a_precomputed_detector_file(frames, detector, tmp_path):
    """One format, both directions -- the claim index.py makes."""
    f = CropIndex.build(frames, detector, min_side=4.0, progress=False).save(tmp_path / "b.json")
    served = build_detector("precomputed", boxes_file=str(f))
    boxes = served.detect(None, frames[0])
    assert [b.as_int() for b in boxes] == [s.as_int() for s in SHIPS]


def test_frames_with_no_ships_are_recorded_as_scanned(frames, tmp_path):
    empty = FunctionDetector(lambda im, p: [] if p == frames[0] else list(SHIPS))
    index = CropIndex.build(frames, empty, min_side=4.0, progress=False)
    assert len(index.frames()) == N_FRAMES - 1
    assert len(index.scanned) == N_FRAMES
    raw = json.loads(index.save(tmp_path / "b.json").read_text())
    assert raw[frames[0]] == []  # "ran, found nothing", not "never ran"


def test_filters_apply_on_build_and_again_on_load(frames, detector, tmp_path):
    f = CropIndex.build(frames, detector, min_side=4.0, progress=False).save(tmp_path / "b.json")
    assert len(CropIndex.load(f, min_score=0.8)) == N_FRAMES  # only the 0.9 ship
    assert len(CropIndex.load(f, max_per_frame=1)) == N_FRAMES
    assert len(CropIndex.load(f, min_side=100)) == 0
    assert len(CropIndex.load(f, keep=frames[:2])) == 2 * len(SHIPS)


def test_an_unreadable_frame_does_not_end_the_pass(frames, detector, tmp_path):
    bad = str(tmp_path / "truncated.png")
    (tmp_path / "truncated.png").write_bytes(b"not a png")
    index = CropIndex.build(frames + [bad], detector, min_side=4.0, progress=False)
    assert len(index) == N_FRAMES * len(SHIPS)
    with pytest.raises(Exception):
        CropIndex.build([bad], detector, on_error="raise", progress=False)


# --------------------------------------------------------------------------- #
# the generator: the three properties
# --------------------------------------------------------------------------- #


def test_the_generator_satisfies_its_contract(frames, detector):
    assert check_generator(make_generator(frames, detector), verbose=False) == []


def test_length_is_the_number_of_ships(frames, detector):
    gen = make_generator(frames, detector)
    assert len(gen) == N_FRAMES * len(SHIPS)


def test_a_list_of_paths_is_summarised_not_recited(frames, detector):
    """`source` may be ten thousand paths; a repr, a wandb config and the index
    file's provenance all have to survive that."""
    gen = make_generator(frames, detector)
    assert gen.source_name == f"{N_FRAMES} paths"
    assert frames[0] not in repr(gen)
    assert gen.stats()["source"] == f"{N_FRAMES} paths"
    pattern = f"{frames[0][:-7]}*.png"
    assert make_generator(pattern, detector).source_name == pattern


def test_the_target_is_stable_across_epochs_and_the_input_is_not(frames, detector):
    gen = make_generator(frames, detector)
    a, b = gen.example(3, epoch=0), gen.example(3, epoch=1)
    assert np.array_equal(a.clean, b.clean)
    assert not np.array_equal(a.doctored, b.doctored)
    assert a.seed != b.seed


def test_an_example_is_a_pure_function_of_seed_epoch_index(frames, detector):
    one = make_generator(frames, detector, seed=7)
    two = make_generator(frames, detector, seed=7)
    for i in (0, 5, 11):
        assert np.array_equal(one.example(i, epoch=2).doctored, two.example(i, epoch=2).doctored)
    assert make_generator(frames, detector, seed=8).example(0).seed != one.example(0).seed


def test_the_seed_does_not_depend_on_the_size_of_the_index(frames, detector):
    """The reason seed_for hashes rather than adds: retuning the detector adds
    examples, and must not renumber the ones that were already there."""
    small = make_generator(frames[:2], detector)
    large = make_generator(frames, detector)
    assert small.seed_for(0, 3) == large.seed_for(0, 3)


def test_the_crop_is_the_same_pixels_every_time(frames, detector):
    gen = make_generator(frames, detector)
    assert np.array_equal(gen.crop(4).patch, gen.crop(4).patch)
    assert np.array_equal(gen.example(4, epoch=9).clean, gen.crop(4).patch)


def test_a_pickled_generator_produces_the_same_data(frames, detector):
    """Stand-in for a DataLoader worker: same config, same bytes, no cache in tow."""
    gen = make_generator(frames, detector)
    gen.example(0)  # warm the frame cache
    clone = pickle.loads(pickle.dumps(gen))
    assert clone._cache == {}
    assert np.array_equal(clone.example(2, epoch=1).doctored, gen.example(2, epoch=1).doctored)


def test_the_frame_cache_does_not_change_what_comes_out(frames, detector):
    cached = make_generator(frames, detector, cache_frames=4)
    plain = make_generator(frames, detector, cache_frames=0)
    for i in range(len(plain)):
        assert np.array_equal(cached.example(i).doctored, plain.example(i).doctored)


def test_a_caller_writing_to_a_patch_cannot_poison_the_cache(frames, detector):
    gen = make_generator(frames, detector)
    crop = gen.crop(0)
    crop.patch[:] = 0
    assert gen.crop(0).patch.max() > 0


# --------------------------------------------------------------------------- #
# epochs and streams
# --------------------------------------------------------------------------- #


def test_an_epoch_visits_every_example_once_in_a_seeded_order(frames, detector):
    gen = make_generator(frames, detector)
    order = gen.order(0)
    assert sorted(order.tolist()) == list(range(len(gen)))
    assert np.array_equal(order, gen.order(0))
    assert not np.array_equal(order, gen.order(1))


def test_iter_is_one_epoch_and_stream_keeps_going(frames, detector):
    gen = make_generator(frames, detector)
    assert len(list(gen)) == len(gen)
    assert gen.epoch == 0  # __iter__ does not advance it
    pairs = [p for _, p in zip(range(len(gen) * 2 + 3), gen.stream())]
    assert len(pairs) == len(gen) * 2 + 3
    assert gen.epoch == 2


def test_virtual_indices_walk_epochs(frames, detector):
    gen = make_generator(frames, detector)
    n = len(gen)
    assert np.array_equal(gen.example_at(0).doctored, gen.example(0, epoch=0).doctored)
    assert np.array_equal(gen.example_at(n + 1).doctored, gen.example(1, epoch=1).doctored)


def test_batches_are_the_size_asked_for(frames, detector):
    gen = make_generator(frames, detector)
    sizes = [len(b) for b in gen.batches(5, epochs=1)]
    assert sizes[:-1] == [5] * (len(sizes) - 1) and sum(sizes) == len(gen)
    assert all(len(b) == 5 for b in gen.batches(5, epochs=1, drop_last=True))


# --------------------------------------------------------------------------- #
# the index cache on disk
# --------------------------------------------------------------------------- #


def test_the_index_file_is_written_once_and_read_back(frames, detector, tmp_path):
    f = tmp_path / "cached.json"
    first = make_generator(frames, detector, index_file=f, progress=False)
    assert len(first) == N_FRAMES * len(SHIPS)
    assert f.exists()

    # a detector that would raise if it ran: the second generator must not detect
    never = FunctionDetector(lambda im, p: (_ for _ in ()).throw(AssertionError("re-detected")))
    second = make_generator(frames, detector=never, index_file=f, progress=False)
    assert len(second) == len(first)
    assert np.array_equal(second.example(0).clean, first.example(0).clean)


def test_a_limit_means_the_same_thing_whether_the_index_was_built_or_loaded(
    frames, detector, tmp_path
):
    f = tmp_path / "all.json"
    make_generator(frames, detector, index_file=f, progress=False).build_index()
    subset = make_generator(frames, detector, index_file=f, limit=2, progress=False)
    assert len(subset) == 2 * len(SHIPS)
    assert set(subset.index.frames()) <= set(frames)


# --------------------------------------------------------------------------- #
# the checker itself
# --------------------------------------------------------------------------- #


def test_check_catches_a_target_that_moves_between_epochs(frames, detector):
    class Wobbly(ShipDataGenerator):
        def crop(self, index):
            crop = super().crop(index)
            crop.patch = (crop.patch + np.uint8(self.epoch)) if self.epoch else crop.patch
            return crop

        def example(self, index, epoch=None, seed=None):
            self.epoch = self.epoch if epoch is None else epoch
            return super().example(index, epoch, seed)

    gen = Wobbly(source=frames, detector=detector, min_side=4.0, cropper={"out_size": 128})
    assert any("clean target changed" in p for p in check_generator(gen, verbose=False))


def test_check_catches_an_empty_index(frames):
    gen = make_generator(frames, FunctionDetector(lambda im, p: []))
    problems = check_generator(gen, verbose=False)
    assert any("index is empty" in p for p in problems)
    with pytest.raises(AssertionError):
        check_generator(gen, strict=True, verbose=False)


def test_check_catches_a_seed_that_ignores_the_epoch(frames, detector):
    class Frozen(ShipDataGenerator):
        def seed_for(self, index, epoch=None):
            return int(index)

    gen = Frozen(source=frames, detector=detector, min_side=4.0, cropper={"out_size": 128})
    assert any("ignores the epoch" in p for p in check_generator(gen, verbose=False))


# --------------------------------------------------------------------------- #
# composites -- the test-time stream
# --------------------------------------------------------------------------- #


def test_a_composite_puts_the_donor_where_the_host_had_a_ship(frames, detector):
    hosts = make_generator(frames, detector)
    donors = make_generator(frames, detector)
    comp = CompositeGenerator(hosts, donors, placement="detection", seed=0)
    c = comp[0]
    assert c.patch.shape == c.host_patch.shape == (128, 128)
    assert c.mask.shape == (128, 128)
    assert (c.mask > 0.5).any()
    assert c.box.iou(hosts.index[0].box) > 0.3  # placed on the host's own ship
    assert not np.array_equal(c.patch, c.host_patch)


def test_a_composite_leaves_the_host_alone_outside_the_mask(frames, detector):
    hosts = make_generator(frames, detector)
    comp = CompositeGenerator(hosts, hosts, seed=3)
    for i in range(4):
        c = comp[i]
        outside = c.mask <= 0.0
        assert np.array_equal(c.patch[outside], c.host_patch[outside])


def test_composites_replay_and_lower_placement_stays_in_the_frame(frames, detector):
    hosts = make_generator(frames, detector)
    comp = CompositeGenerator(hosts, hosts, placement="lower", seed=1)
    assert np.array_equal(comp[2].patch, comp[2].patch)
    assert len(list(comp)) == len(hosts)
    for c in comp:
        assert c.box.center[1] > 0.4 * 240
        assert c.params["placement"] == "lower"


def test_a_placement_callable_is_honoured(frames, detector):
    hosts = make_generator(frames, detector)
    where = BBox(10.0, 10.0, 60.0, 40.0, label="ship")
    comp = CompositeGenerator(hosts, hosts, placement=lambda rng, host, donor: where)
    assert comp[0].box == where


def test_the_box_can_be_expressed_over_the_patch(frames, detector):
    hosts = make_generator(frames, detector)
    c = CompositeGenerator(hosts, hosts, seed=0)[0]
    box = c.box_in_patch()
    # the mask is the paste rectangle, which is the box grown a little: the two
    # must at least agree about where in the patch the ship is
    ys, xs = np.nonzero(c.mask > 0.5)
    assert xs.min() - 1 <= box.center[0] <= xs.max() + 1
    assert ys.min() - 1 <= box.center[1] <= ys.max() + 1


# --------------------------------------------------------------------------- #
# how far a placed ship may be rolled
# --------------------------------------------------------------------------- #


def test_composites_cap_the_rotation_without_touching_the_training_augmenter(frames, detector):
    """A ship sits level on the water; the training corruption may still roll it
    further, because there the target shows the ship level anyway."""
    from realsimir.augment import Rotate

    hosts = make_generator(frames, detector)
    training_max = [op.max_deg for op in hosts.augmenter.ops if isinstance(op, Rotate)]
    comp = CompositeGenerator(hosts, hosts, max_rotation_deg=1.5)

    assert [op.max_deg for op in comp.augmenter.ops if isinstance(op, Rotate)] == [1.5]
    assert [op.max_deg for op in hosts.augmenter.ops if isinstance(op, Rotate)] == training_max
    assert training_max == [6.0] and hosts.augmenter is not comp.augmenter

    drawn = [
        abs(comp[i].params["geometric"].get("rotate", {}).get("deg", 0.0)) for i in range(len(hosts))
    ]
    assert max(drawn) <= 1.5 and max(drawn) > 0.0


def test_the_cap_is_a_default_and_can_be_switched_off(frames, detector):
    from realsimir.augment import Rotate
    from realsimir.data.composite import DEFAULT_MAX_ROTATION_DEG

    hosts = make_generator(frames, detector)
    capped = CompositeGenerator(hosts, hosts)
    assert [op.max_deg for op in capped.augmenter.ops if isinstance(op, Rotate)] == [
        DEFAULT_MAX_ROTATION_DEG
    ]
    uncapped = CompositeGenerator(hosts, hosts, max_rotation_deg=None)
    assert uncapped.augmenter is hosts.augmenter


def test_an_augmenter_that_already_rotates_little_is_left_alone(frames, detector):
    aug = ShipAugmenter(ops={"rotate": {"max_deg": 0.5}, "noise": {}})
    hosts = make_generator(frames, detector, augmenter=aug)
    assert CompositeGenerator(hosts, hosts, max_rotation_deg=2.0).augmenter is aug


# --------------------------------------------------------------------------- #
# self-paste -- the test images that have an answer
# --------------------------------------------------------------------------- #


def test_self_paste_puts_the_ship_back_in_its_own_frame_with_a_target(frames, detector):
    donors = make_generator(frames, detector)
    test = CompositeGenerator.self_paste(donors, seed=0)

    assert len(test) == len(donors) and test.has_targets
    c = test[0]
    assert c.has_target and c.target is not None
    assert np.array_equal(c.target, donors.crop(0).patch)  # the render, untouched
    assert c.host_path == c.donor_path == donors.index[0].path
    assert not np.array_equal(c.patch, c.target)
    outside = c.mask <= 0.0
    assert np.array_equal(c.patch[outside], c.target[outside])


def test_self_paste_is_the_training_recipe(frames, detector):
    """Same crop, same seed, same corruption -- the point of the mode is that
    the test images are made the way the training ones were."""
    gen = make_generator(frames, detector)
    test = CompositeGenerator.self_paste(gen, max_rotation_deg=None, seed=0)
    c = test[3]
    pair = gen.augmenter.augment_crop(gen.crop(3), seed=c.seed)
    assert np.array_equal(c.patch, pair.doctored)
    assert np.array_equal(c.target, pair.clean)


def test_self_paste_caps_rotation_too(frames, detector):
    donors = make_generator(frames, detector)
    test = CompositeGenerator.self_paste(donors, max_rotation_deg=1.0)
    drawn = [abs(test[i].params["geometric"].get("rotate", {}).get("deg", 0.0)) for i in range(8)]
    assert max(drawn) <= 1.0


def test_a_pasted_composite_has_no_target(frames, detector):
    hosts = make_generator(frames, detector)
    comp = CompositeGenerator(hosts, hosts)
    assert not comp.has_targets
    assert comp[0].has_target is False and comp[0].target is None


# --------------------------------------------------------------------------- #
# torch adapter
# --------------------------------------------------------------------------- #


def test_the_torch_dataset_emits_batchable_tensors(frames, detector):
    torch = pytest.importorskip("torch")
    from realsimir.data import ShipPairDataset, make_dataloader

    gen = make_generator(frames, detector)
    ds = ShipPairDataset(gen, repeats=3)
    assert len(ds) == len(gen) * 3
    item = ds[0]
    assert item["doctored"].shape == (1, 128, 128) and item["doctored"].dtype == torch.uint8
    assert item["mask"].shape == (1, 128, 128) and item["mask"].dtype == torch.float32

    loader = make_dataloader(gen, batch_size=4, num_workers=0, repeats=2)
    batch = next(iter(loader))
    assert batch["clean"].shape == (4, 1, 128, 128)
    assert batch["seed"].shape == (4,)


def test_the_dataset_ends(frames, detector):
    """It looks like a detail; it is the difference between `for x in dataset`
    terminating and hanging, because the generator wraps a virtual index into
    the next epoch rather than running out."""
    pytest.importorskip("torch")
    from realsimir.data import ShipPairDataset

    ds = ShipPairDataset(make_generator(frames, detector), repeats=2)
    assert len(list(ds)) == len(ds)
    assert ds[-1]["index"] == ds[len(ds) - 1]["index"]
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_the_composite_dataset_carries_a_target_only_when_there_is_one(frames, detector):
    pytest.importorskip("torch")
    from realsimir.data import CompositeDataset

    gen = make_generator(frames, detector)
    scoreable = CompositeDataset(CompositeGenerator.self_paste(gen))
    assert scoreable.has_targets and "clean" in scoreable[0]
    assert scoreable[0]["clean"].shape == scoreable[0]["doctored"].shape

    deployed = CompositeDataset(CompositeGenerator(gen, gen))
    assert not deployed.has_targets and "clean" not in deployed[0]


def test_normalize_puts_the_pair_in_minus_one_to_one(frames, detector):
    torch = pytest.importorskip("torch")
    from realsimir.data import ShipPairDataset

    item = ShipPairDataset(make_generator(frames, detector), normalize=True)[0]
    assert item["clean"].dtype == torch.float32
    assert -1.0 <= float(item["clean"].min()) and float(item["clean"].max()) <= 1.0


def test_workers_agree_with_the_parent(frames, detector):
    pytest.importorskip("torch")
    from realsimir.data import ShipPairDataset, make_dataloader

    gen = make_generator(frames, detector)
    single = {int(b["index"]): b["doctored"] for b in ShipPairDataset(gen)}
    loader = make_dataloader(gen, batch_size=2, num_workers=2, shuffle=False, drop_last=False)
    seen = 0
    for batch in loader:
        for j, index in enumerate(batch["index"].tolist()):
            assert (batch["doctored"][j] == single[index]).all()
            seen += 1
    assert seen == len(gen)
