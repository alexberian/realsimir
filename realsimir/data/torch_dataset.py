"""torch adapters -- the generator as a Dataset, ready for step 09's training loop.

Kept in its own module so that `import realsimir.data` costs no torch: the
detector registry, the pools and the index are useful from a notebook or a CPU
box that has no CUDA build installed.

    from realsimir.data import ShipDataGenerator, make_dataloader

    gen = ShipDataGenerator("real", index_file="boxes/real.json")
    loader = make_dataloader(gen, batch_size=32, num_workers=8)
    for batch in loader:
        batch["doctored"], batch["clean"], batch["mask"]

Conventions, chosen to drop into edm2 without a shim
----------------------------------------------------
`clean` and `doctored` come out as uint8 NCHW, which is what edm2's
`StandardRGBEncoder.encode_latents` expects -- it does `x / 127.5 - 1` itself, so
handing it floats in [0, 1] would train the model on half the range with no
error anywhere.  Pass `normalize=True` to get float32 in [-1, 1] instead, for a
loop that does its own encoding.  `mask` is always float32, because it is a
weight and not a picture.

Channels: our IR frames are single-channel, so C == 1.  Whatever the net's
`img_channels` is set to has to agree with that.

Epochs and workers
------------------
`repeats` is the number of times the dataset walks its examples, and the epoch
of a visit is folded into its virtual index (`j // len(gen)`), so a fresh
corruption arrives on every pass *without* anything having to call `set_epoch`
across a process boundary.  That is what makes `persistent_workers=True` safe
here, and it is why the loader is shuffled over the virtual range rather than
re-created every epoch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .composite import CompositeGenerator
from .generator import ShipDataGenerator

__all__ = ["ShipPairDataset", "CompositeDataset", "make_dataloader", "to_tensor"]


def _bounds(i: int, n: int) -> int:
    """Python-list indexing semantics, including a negative index."""
    i = int(i)
    if i < 0:
        i += n
    if not 0 <= i < n:
        raise IndexError(f"index {i} is out of range for a dataset of {n}")
    return i


def to_tensor(image: np.ndarray, normalize: bool = False) -> torch.Tensor:
    """HxW or HxWxC numpy -> CHW tensor.

    uint8 stays uint8 unless `normalize`, in which case it becomes float32 in
    [-1, 1] by the dtype's full range -- the same convention as
    `realsimir.imaging.to_float01`, so a 16-bit frame and an 8-bit one land in
    the same place.
    """
    arr = image if image.ndim == 3 else image[:, :, None]
    t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    if not normalize:
        return t
    if t.dtype == torch.uint8:
        return t.float() / 127.5 - 1.0
    if not torch.is_floating_point(t):
        info = np.iinfo(image.dtype)
        return t.float() / (float(info.max) / 2.0) - 1.0
    return t.float() * 2.0 - 1.0  # float input is [0, 1] by imaging.to_float01


class ShipPairDataset(torch.utils.data.Dataset):
    """Map-style view of a ShipDataGenerator.

    generator  the object from generator.py.
    repeats    how many passes over the index the dataset is long.  Each pass is
               a different epoch, so a different corruption of the same crops.
               Make it large for edm2-style "train for N images" runs; the only
               cost of a big number is that `len(loader)` is big.
    normalize  float32 in [-1, 1] instead of uint8 (see the module docstring).
    with_params  attach the drawn magnitudes to each item.  Off by default:
               they are a dict of dicts, which the default collate cannot batch.
    """

    def __init__(
        self,
        generator: ShipDataGenerator,
        repeats: int = 1,
        normalize: bool = False,
        with_params: bool = False,
    ):
        self.generator = generator
        self.repeats = int(repeats)
        self.normalize = bool(normalize)
        self.with_params = bool(with_params)
        self._len = len(generator) * self.repeats  # forces the index build here,
        # in the parent process, rather than once per worker

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, j: int) -> dict:
        # the generator wraps virtual indices round into later epochs, so it
        # cannot end an iteration by itself; a map-style dataset has to raise
        # IndexError past its length or `for x in dataset` never stops
        j = _bounds(j, self._len)
        pair = self.generator.example_at(j)
        item = {
            "doctored": to_tensor(pair.doctored, self.normalize),
            "clean": to_tensor(pair.clean, self.normalize),
            "mask": torch.from_numpy(np.ascontiguousarray(pair.mask))[None],
            "index": j % len(self.generator),
            "epoch": j // len(self.generator),
            "seed": int(pair.seed),
        }
        if self.with_params:
            item["params"] = pair.params
        return item

    def __repr__(self) -> str:
        return f"ShipPairDataset({self.generator!r}, repeats={self.repeats})"


class CompositeDataset(torch.utils.data.Dataset):
    """Map-style view of a CompositeGenerator -- the test stream.

    Items carry a `clean` target only when the generator has one to give (a
    self-paste set does, a ship dropped into a foreign frame does not).  Either
    way every item of one generator has the same keys, so the default collate
    batches them; a scoring loop can branch on `dataset.has_targets` once
    instead of per item.
    """

    def __init__(self, generator: CompositeGenerator, normalize: bool = False):
        self.generator = generator
        self.normalize = bool(normalize)

    @property
    def has_targets(self) -> bool:
        return self.generator.has_targets

    def __len__(self) -> int:
        return len(self.generator)

    def __getitem__(self, i: int) -> dict:
        i = _bounds(i, len(self.generator))
        c = self.generator[i]
        item = {
            "doctored": to_tensor(c.patch, self.normalize),
            "host": to_tensor(c.host_patch, self.normalize),
            "mask": torch.from_numpy(np.ascontiguousarray(c.mask))[None],
            "index": i,
            "seed": int(c.seed),
        }
        if c.has_target:
            item["clean"] = to_tensor(c.target, self.normalize)
        return item

    def __repr__(self) -> str:
        return f"CompositeDataset({self.generator!r})"


def make_dataloader(
    generator: ShipDataGenerator | CompositeGenerator,
    batch_size: int = 32,
    num_workers: int = 4,
    repeats: int = 1,
    shuffle: bool | None = None,
    normalize: bool = False,
    drop_last: bool = True,
    **kwargs: Any,
) -> torch.utils.data.DataLoader:
    """A DataLoader over either generator, with the defaults this project wants.

    No `worker_init_fn`: every corruption is seeded from (run seed, epoch,
    index), so the workers cannot disagree with each other or with a single
    process run -- which is the property that lets a batch be reproduced from
    its indices.  `persistent_workers` follows `num_workers` for the same
    reason; nothing needs to be reset between epochs.
    """
    if isinstance(generator, CompositeGenerator):
        dataset: torch.utils.data.Dataset = CompositeDataset(generator, normalize=normalize)
        shuffle = False if shuffle is None else shuffle  # a test set has an order
    else:
        dataset = ShipPairDataset(generator, repeats=repeats, normalize=normalize)
        shuffle = True if shuffle is None else shuffle
    options = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    options.update(kwargs)  # the caller's word beats ours, including on these
    return torch.utils.data.DataLoader(dataset, **options)
