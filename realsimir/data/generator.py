"""ShipDataGenerator -- todo step 07, the online data generator.

The object that turns a directory of unlabelled IR frames into an endless
stream of training pairs, and the last piece of the front end: steps 05 and 06
built the cropper and the augmenter, this is what feeds them.

    from realsimir.data import ShipDataGenerator

    gen = ShipDataGenerator("real", detector="yolov3", index_file="boxes/real.json")
    pair = gen[0]                    # AugmentedPair: .doctored -> model -> .clean
    for pair in gen.stream():        # infinite, a fresh corruption every epoch
        ...

Online, in the sense idea.txt means: nothing is written to disk, no corpus of
doctored frames is materialised, and the corruption is drawn at the moment the
example is asked for.  That matters more here than it usually does -- the point
of the algorithm is that the model sees a *distribution* of bad crops rather
than a fixed set, and a frozen dataset of, say, 40k doctored images would be
sampled thousands of times over a diffusion training run.

What is *not* online is the detection.  Boxes come from a `CropIndex` built once
and cached (index.py explains why), so an epoch costs a JPEG decode and a warp
per example rather than a forward pass of the bounding box model.

The three properties the training loop leans on
----------------------------------------------
 1. `len(gen)` is the number of ships found, not the number of frames.  One
    frame with three ships is three examples, and epochs are defined over those.
 2. Example i is *the same ship in the same window* in every epoch -- only the
    corruption changes.  The clean patch is the diffusion target, so it must not
    move underneath the model; what varies is the input it has to map back to it.
 3. Every example's seed is a pure function of (generator seed, epoch, index).
    Not of a global RNG, a worker id, or the order the examples came out in --
    so `num_workers` does not change the data, a batch can be reproduced from
    its indices alone, and a figure in a paper can be regenerated a year later.

Seeds are mixed with `np.random.SeedSequence` rather than added together for a
specific reason: the index grows when the detector is retuned, and an arithmetic
scheme keyed on `len(index)` would silently renumber every example's corruption
when it does.

Worker safety
-------------
The frame cache is dropped when the generator is pickled, so handing one to a
`DataLoader(num_workers=N)` copies the configuration and not a pile of decoded
frames.  Each worker then rebuilds its own cache over the examples it is given;
`torch_dataset.py` wires that up.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Iterator, Mapping

import numpy as np

from ..augment import AugmentedPair, ShipAugmenter
from ..boxes import ShipCrop
from ..detectors import ShipDetector, build_detector
from ..imaging import load_image
from ..ship_cropper import ShipCropper
from .index import CropIndex, CropRef
from .pools import get_pool, resolve_paths

__all__ = ["ShipDataGenerator"]

#: stands in for the example index when seeding an epoch's visiting order.
#: SeedSequence takes unsigned words only, so this is a value out of range
#: rather than the -1 that would read more naturally.
_ORDER = 0xFFFF_FFFF


class ShipDataGenerator:
    """Frames on disk in, (clean, doctored, mask) training pairs out.

    source        what to train on: a pool name ('real', 'raysense', 'arete'),
                  a glob, a directory, a list of paths, or a ready CropIndex.
                  See pools.py.
    detector      the bounding box model, by name or as an object -- only
                  consulted when an index has to be built.  None takes the
                  pool's own suggestion ('arete' has ground truth), falling back
                  to the project default.
    index_file    where the built index lives.  If the file exists it is loaded;
                  otherwise the detector is run once and the result written
                  there.  This is the flag that turns a five-minute startup into
                  a one-second one for every run after the first.
    cropper       ShipCropper or a dict of its arguments.  Its `out_size` is the
                  resolution the model trains at.
    augmenter     ShipAugmenter or a dict of its arguments (step 06).
    limit/stride/sample_seed
                  subset the pool before indexing -- see pools.resolve_paths.
                  The synthetic pool has a million renders; take a sample.
    min_score/min_side/max_per_frame
                  which detections count as an example.  Applied on load as well
                  as on build, so one permissive index can serve several runs.
    seed          the run's seed.  Together with (epoch, index) it determines
                  every corruption drawn.
    cache_frames  how many decoded frames to keep.  A frame with several ships
                  is decoded once; 0 disables.  Frames are big, so this is a
                  handful, not thousands.
    """

    def __init__(
        self,
        source: Any = "real",
        detector: Any = None,
        index_file: str | os.PathLike | None = None,
        index: CropIndex | str | os.PathLike | None = None,
        cropper: ShipCropper | Mapping | None = None,
        augmenter: ShipAugmenter | Mapping | None = None,
        limit: int | None = None,
        stride: int = 1,
        sample_seed: int = 0,
        min_score: float = 0.0,
        min_side: float = 8.0,
        max_per_frame: int | None = None,
        seed: int = 0,
        epoch: int = 0,
        cache_frames: int = 8,
        progress: bool = True,
    ):
        self.source = source
        self.detector_spec = detector if detector is not None else _suggested_detector(source)
        self.index_file = None if index_file is None else str(index_file)
        self.cropper = cropper if isinstance(cropper, ShipCropper) else ShipCropper(
            _NULL_DETECTOR, **dict(cropper or {})
        )
        self.augmenter = augmenter if isinstance(augmenter, ShipAugmenter) else ShipAugmenter(
            **dict(augmenter or {})
        )
        self.limit, self.stride, self.sample_seed = limit, stride, sample_seed
        self.min_score, self.min_side, self.max_per_frame = min_score, min_side, max_per_frame
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.cache_frames = int(cache_frames)
        self.progress = progress

        self._index: CropIndex | None = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        if index is not None:
            self._index = index if isinstance(index, CropIndex) else self._load_index(index)

    # -- the index ---------------------------------------------------------- #

    @property
    def index(self) -> CropIndex:
        """The (frame, ship) enumeration, built or loaded on first use.

        Deliberately lazy: constructing a generator is cheap enough to do in a
        config, while building an index is a pass of the detector over the pool.
        Call `build_index()` to force it at a moment of your choosing.
        """
        if self._index is None:
            self._index = self.build_index()
        return self._index

    def build_index(self, force: bool = False) -> CropIndex:
        """Load the cached index, or run the detector over the pool and cache it."""
        if self.index_file and not force and os.path.exists(self.index_file):
            self._index = self._load_index(self.index_file)
            return self._index

        paths = resolve_paths(
            self.source, limit=self.limit, stride=self.stride, seed=self.sample_seed
        )
        if not paths:
            raise FileNotFoundError(f"no frames matched {self.source!r}")
        detector = build_detector(self.detector_spec)
        if self.progress:
            print(f"indexing {len(paths)} frames with {detector!r}")
        self._index = CropIndex.build(
            paths,
            detector,
            min_score=self.min_score,
            min_side=self.min_side,
            max_per_frame=self.max_per_frame,
            progress=self.progress,
            meta={"source": self.source_name, "limit": self.limit, "stride": self.stride},
        )
        if self.index_file:
            self._index.save(self.index_file)
            if self.progress:
                print(f"  cached -> {self.index_file}")
        return self._index

    def _load_index(self, file: str | os.PathLike) -> CropIndex:
        """Cached boxes, re-filtered and restricted to the pool we were asked for.

        The `keep` restriction is what makes `limit=` mean the same thing whether
        the index was built for this run or inherited from a larger one.
        """
        keep = None
        if self.limit is not None or self.stride > 1:
            keep = resolve_paths(
                self.source, limit=self.limit, stride=self.stride, seed=self.sample_seed
            )
        index = CropIndex.load(
            file,
            min_score=self.min_score,
            min_side=self.min_side,
            max_per_frame=self.max_per_frame,
            keep=keep,
        )
        if not len(index):
            raise ValueError(f"{file} yielded no examples (filters too tight, or a different pool?)")
        return index

    # -- one example -------------------------------------------------------- #

    def seed_for(self, index: int, epoch: int | None = None) -> int:
        """The seed example `index` is corrupted with in `epoch`.

        A hash, not a sum: the index grows when the detector is retuned, and any
        scheme of the form `base + epoch * len(index) + i` would renumber the
        whole run when it does.
        """
        epoch = self.epoch if epoch is None else epoch
        state = np.random.SeedSequence([self.seed, int(epoch), int(index)])
        return int(state.generate_state(1, dtype=np.uint32)[0])

    def frame(self, path: str) -> np.ndarray:
        """The decoded frame, from the cache when it is there.

        Callers must not write to what comes back -- the cropper does not, and
        the augmenter copies before it touches anything.
        """
        if self.cache_frames <= 0:
            return load_image(path)
        cached = self._cache.get(path)
        if cached is None:
            cached = load_image(path)
            self._cache[path] = cached
            while len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(path)
        return cached

    def crop(self, index: int) -> ShipCrop:
        """Example `index`'s window -- the same pixels in every epoch."""
        ref: CropRef = self.index[index]
        crop = self.cropper.crop(self.frame(ref.path), ref.box, ref.path)
        if crop.patch.base is not None:
            # a window that needed no resize is a *view* into the cached frame;
            # hand out a copy so a caller writing to the patch cannot poison it
            crop.patch = crop.patch.copy()
        return crop

    def example(self, index: int, epoch: int | None = None, seed: int | None = None) -> AugmentedPair:
        """One training pair.  `seed` overrides the (epoch, index) derivation."""
        crop = self.crop(index)
        return self.augmenter.augment_crop(
            crop, seed=self.seed_for(index, epoch) if seed is None else seed
        )

    def __getitem__(self, index: int) -> AugmentedPair:
        return self.example(index)

    def __len__(self) -> int:
        return len(self.index)

    # -- streams ------------------------------------------------------------ #

    def set_epoch(self, epoch: int) -> "ShipDataGenerator":
        """Move to `epoch`: same crops, different corruptions, different order."""
        self.epoch = int(epoch)
        return self

    def order(self, epoch: int | None = None) -> np.ndarray:
        """The visiting order for an epoch -- a seeded permutation of every example.

        Drawn from the same three words as `seed_for`, with `_ORDER` where the
        example index goes: an index that large is not an example, so the order
        cannot accidentally share a stream with one of the corruptions.
        """
        epoch = self.epoch if epoch is None else epoch
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch), _ORDER]))
        return rng.permutation(len(self))

    def __iter__(self) -> Iterator[AugmentedPair]:
        """One epoch, shuffled.  Does not advance the epoch -- `stream` does."""
        for i in self.order():
            yield self.example(int(i))

    def stream(self, epochs: int | None = None) -> Iterator[AugmentedPair]:
        """Examples for `epochs` epochs, or for ever.  Advances `self.epoch`."""
        n = 0
        while epochs is None or n < epochs:
            for i in self.order():
                yield self.example(int(i))
            self.epoch += 1
            n += 1

    def batches(
        self, batch_size: int, epochs: int | None = 1, drop_last: bool = False
    ) -> Iterator[list[AugmentedPair]]:
        """`stream`, grouped.  The numpy-only path -- torch_dataset.py is the other."""
        batch: list[AugmentedPair] = []
        for pair in self.stream(epochs):
            batch.append(pair)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch and not drop_last:
            yield batch

    def example_at(self, j: int) -> AugmentedPair:
        """Example for a *virtual* index that runs past the end of the epoch.

        `j = epoch * len(gen) + i`.  This is the form an infinite sampler wants
        (edm2 trains for a number of images, not a number of epochs): a map-style
        dataset of length `len(gen) * repeats` whose every visit is a fresh
        corruption of a stable crop.
        """
        n = len(self)
        return self.example(j % n, epoch=j // n)

    # -- reporting ---------------------------------------------------------- #

    @property
    def source_name(self) -> str:
        """Short name for logs and reprs.

        `source` may be a pool name or a list of ten thousand paths; the second
        form must not end up in a repr, a wandb config or the index file's
        provenance, so it is summarised instead.
        """
        if isinstance(self.source, str):
            return self.source
        if isinstance(self.source, (list, tuple, set)):
            return f"{len(self.source)} paths"
        return str(self.source)

    def stats(self) -> dict:
        """Index statistics plus the configuration -- one dict to log at step 09."""
        return {
            **self.index.stats(),
            "source": self.source_name,
            "detector": self.index.meta.get("detector", str(self.detector_spec)),
            "out_size": self.cropper.out_size,
            "ops": [op.name for op in self.augmenter.ops],
            "seed": self.seed,
        }

    # -- pickling ----------------------------------------------------------- #

    def __getstate__(self) -> dict:
        """Cross a process boundary without carrying decoded frames."""
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def __repr__(self) -> str:
        n = "?" if self._index is None else len(self._index)
        return (
            f"ShipDataGenerator({self.source_name!r}, {n} examples, "
            f"{self.cropper.out_size}px, seed={self.seed}, epoch={self.epoch})"
        )


def _suggested_detector(source: Any) -> Any:
    """A named pool may know which bounding box model belongs to it."""
    if isinstance(source, str):
        try:
            pool = get_pool(source)
        except KeyError:
            return "yolov3"
        if pool.detector:
            return pool.detector
    return "yolov3"


class _NullDetector(ShipDetector):
    """Placeholder for the cropper's detector slot.

    The generator never detects -- boxes come from the index -- but ShipCropper
    is built around a detector.  Rather than let it construct the default
    (which would load YOLOv3 weights in every dataloader worker for nothing), it
    gets one that refuses to run and says why.
    """

    def detect(self, image, path=None):
        raise RuntimeError(
            "this cropper belongs to a ShipDataGenerator and has no detector -- "
            "boxes come from the CropIndex.  Build a ShipCropper of your own if "
            "you want to detect."
        )

    def __repr__(self) -> str:
        return "NullDetector(boxes come from the index)"


_NULL_DETECTOR = _NullDetector()
