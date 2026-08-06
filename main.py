"""Demo / sanity check for todo steps 05 (ship cropping) and 06 (augmentation).

Runs the pipeline over both halves of the data and dumps figures to ./figures:

    01_real_detections.png       real IR frames + detector boxes and windows
    02_real_crops.png            the 256x256 patches those windows produce
    03_synthetic_detections.png  arete renders + ground-truth boxes and windows
    04_synthetic_crops.png       the 256x256 patches those windows produce
    05_window_geometry.png       tiny vs. huge ship -- native crop vs. downscale
    06_augmented_pairs.png       clean | doctored | difference: the training pair
    07_corruption_ops.png        one corruption at a time, exaggerated
    08_synthetic_into_real.png   the same paste at test time: sim ship, real frame
    09_synthetic_into_self.png   sim ship back into its own render -- the test
                                 images that have a right answer

Usage:  conda run -n realsimir python main.py [--out figures] [--seed 0]

The real-imagery detector is chosen with --detector, so this doubles as the
eyeball test when the bounding box model is swapped:

    python main.py --detector yolov3 --detector-arg conf_thresh=0.35
    python main.py --detector precomputed --detector-arg boxes_file=boxes.json
    python main.py --detector mypkg.detectors:Yolov8ShipDetector --detector-arg weights=ir.pt
    python main.py --list-detectors
"""

from __future__ import annotations

import argparse
import ast
import glob
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from realsimir import (
    ARETE_ROOT,
    OPEN_IR_ROOT,
    BBox,
    CompositeGenerator,
    ShipAugmenter,
    ShipCropper,
    ShipDataGenerator,
    build_detector,
    check_augmenter,
    check_detector,
    describe_detectors,
    load_image,
    to_float01,
)
from realsimir.augment import DEFAULT_OPS

REAL_POOLS = {
    "raysense": OPEN_IR_ROOT / "raysense" / "红外船舶数据库" / "dataset" / "images" / "train" / "*.jpg",
    "MassMIND": OPEN_IR_ROOT / "MassMIND" / "Images" / "*.png",
}

DET_COLOR, PASTE_COLOR, WIN_COLOR = "#ff3b30", "#ffd60a", "#00d4ff"


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #


def stretch(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5, ref: np.ndarray | None = None) -> np.ndarray:
    """Percentile stretch -- LWIR renders are near-flat and invisible raw.

    `ref` fixes the limits from another image: a clean/doctored pair has to be
    stretched identically or a gain change re-normalises itself away and the
    figure shows nothing.
    """
    a = img.astype(np.float32)
    r = a if ref is None else ref.astype(np.float32)
    lo, hi = np.percentile(r, lo_pct), np.percentile(r, hi_pct)
    return np.zeros_like(a) if hi <= lo else np.clip((a - lo) / (hi - lo), 0, 1)


def draw_box(ax, box, color, label=None, lw=1.2, ls="-"):
    ax.add_patch(
        mpatches.Rectangle(
            (box.x_min, box.y_min), box.width, box.height, fill=False, edgecolor=color, lw=lw, ls=ls
        )
    )
    if label:
        ax.text(
            box.x_min + 1,
            box.y_min - 2,
            label,
            color=color,
            fontsize=6,
            va="bottom",
            ha="left",
            clip_on=True,
            bbox=dict(fc="black", ec="none", alpha=0.55, pad=0.8),
        )


def grid(n, ncols, figsize_per=3.0):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per, nrows * figsize_per))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    return fig, axes


def save(fig, path: Path, title: str, bottom: float = 0.0):
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_frames(samples, out_path, title, ncols=4, paste_context=0.15, out_size=256):
    """samples: list of (image, crops, caption)."""
    fig, axes = grid(len(samples), ncols)
    for ax, (image, crops, caption) in zip(axes, samples):
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(stretch(image), cmap="gray", vmin=0, vmax=1)
        for c in crops:
            draw_box(ax, c.window, WIN_COLOR, ls="--")
            draw_box(ax, c.paste_box, PASTE_COLOR, ls=":")
            draw_box(ax, c.detection, DET_COLOR, label=f"{c.detection.label} {c.detection.score:.2f}")
        ax.set_title(caption, fontsize=7)
    handles = [
        mpatches.Patch(color=DET_COLOR, label="detection"),
        mpatches.Patch(color=PASTE_COLOR, label=f"paste box (+{paste_context:.0%})"),
        mpatches.Patch(color=WIN_COLOR, label=f"{out_size} crop window"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    save(fig, out_path, title, bottom=0.045)


def plot_crops(crops, out_path, title, ncols=4):
    fig, axes = grid(len(crops), ncols)
    for ax, c in zip(axes, crops):
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(stretch(c.patch), cmap="gray", vmin=0, vmax=1)
        draw_box(ax, c.paste_box_in_patch(), PASTE_COLOR, ls=":")
        draw_box(ax, c.detection_in_patch(), DET_COLOR)
        name = Path(c.path).name if c.path else ""
        ax.set_title(
            f"{name[:34]}\n{c.patch.shape[1]}x{c.patch.shape[0]} px, scale={c.scale:.2f}", fontsize=7
        )
    save(fig, out_path, title)


# --------------------------------------------------------------------------- #
# real IR
# --------------------------------------------------------------------------- #


def run_real(cropper: ShipCropper, out_dir: Path, rng: random.Random, pool_size=48, n_show=8, tag="") -> list:
    frames, keep = [], []
    for name, pattern in REAL_POOLS.items():
        paths = sorted(glob.glob(str(pattern)))
        if not paths:
            print(f"  [{name}] no images found at {pattern}")
            continue
        sample = rng.sample(paths, min(pool_size, len(paths)))
        images = [load_image(p) for p in sample]
        hits = 0
        for p, im, crops in zip(sample, images, cropper.crop_batch(images, sample)):
            if not crops:
                continue
            hits += 1
            keep.append((im, crops, f"[{name}] {Path(p).name[:28]}  ({len(crops)} ship)"))
        print(f"  [{name}] {hits}/{len(sample)} frames with >=1 detection")

    if not keep:
        print("  no real detections at all -- skipping real figures")
        return []
    rng.shuffle(keep)
    frames = keep[:n_show]
    n = cropper.out_size
    plot_frames(
        frames,
        out_dir / "01_real_detections.png",
        f"Real IR: {tag} detections + crop windows",
        paste_context=cropper.paste_context,
        out_size=n,
    )
    crops = [c for _, cs, _ in frames for c in cs][:8]
    plot_crops(crops, out_dir / "02_real_crops.png", f"Real IR: {n}x{n} ship crops")
    return frames


# --------------------------------------------------------------------------- #
# synthetic IR
# --------------------------------------------------------------------------- #


def arete_samples(rng: random.Random, n=8, split="full_ir", min_width=24, max_width=200):
    """Pick renders whose ship is big enough to see but still crop-sized."""
    import pandas as pd

    scenes = sorted(p.name for p in (ARETE_ROOT / split / "images").iterdir() if p.is_dir())
    picks = []
    for scene in rng.sample(scenes, min(n, len(scenes))):
        csvs = sorted(glob.glob(str(ARETE_ROOT / split / "metadata" / scene / "*.csv")))
        df = pd.concat(
            [pd.read_csv(c, usecols=["image_name", "x_min_tight_lwir", "x_max_tight_lwir"]) for c in csvs],
            ignore_index=True,
        )
        w = df.x_max_tight_lwir - df.x_min_tight_lwir
        ok = df[(w >= min_width) & (w <= max_width)]
        if ok.empty:
            ok = df.iloc[(w - min_width).abs().argsort()[:1]]  # scene has no big ship; take the biggest
        row = ok.sample(1, random_state=rng.randrange(2**31)).iloc[0]
        picks.append(ARETE_ROOT / split / "images" / scene / "lwir" / f"{row.image_name}.png")
    return picks


def run_synthetic(cropper: ShipCropper, out_dir: Path, rng: random.Random, n_show=8) -> list:
    paths = arete_samples(rng, n=n_show)
    frames, crops = [], []
    for p in paths:
        im = load_image(p)
        cs = cropper.crop_all(im, p)
        if not cs:
            print(f"  no metadata box for {p.name}")
            continue
        scene = p.parents[1].name
        frames.append((im, cs, f"[{scene}] {im.shape[1]}x{im.shape[0]}  ship {cs[0].detection.width:.0f} px wide"))
        crops.extend(cs)
    if not frames:
        print("  no synthetic boxes -- skipping synthetic figures")
        return []
    n = cropper.out_size
    plot_frames(
        frames,
        out_dir / "03_synthetic_detections.png",
        "Synthetic IR (arete): ground-truth boxes + crop windows",
        paste_context=cropper.paste_context,
        out_size=n,
    )
    plot_crops(crops[:8], out_dir / "04_synthetic_crops.png", f"Synthetic IR (arete): {n}x{n} ship crops")
    return crops


# --------------------------------------------------------------------------- #
# augmentation (step 06)
# --------------------------------------------------------------------------- #

RECT_COLOR = "#ff2fd0"
DIFFERENCE = object()  # panel marker: plot this one on an absolute scale, not stretched


def plot_pairs(pairs, out_path, title):
    """clean | doctored | |difference|, one example per row."""
    fig, axes = plt.subplots(len(pairs), 3, figsize=(9.6, 4.0 * len(pairs)))
    axes = np.atleast_2d(axes)
    for (pair, name, drawn), row in zip(pairs, axes):
        diff = pair.difference()
        for ax, img in zip(row, (pair.clean, pair.doctored, diff)):
            if img is diff:  # a real scale, not a percentile stretch: the point
                ax.imshow(diff, cmap="magma", vmin=0, vmax=max(float(diff.max()), 1e-6))
            else:  # is how much changed, and where
                ax.imshow(stretch(img, ref=pair.clean), cmap="gray", vmin=0, vmax=1)
            draw_box(ax, pair.rect, RECT_COLOR)
            draw_box(ax, pair.ship, DET_COLOR, ls=":")
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_title("clean -- the target", fontsize=8)
        row[1].set_title("doctored -- the model's input", fontsize=8)
        row[2].set_title(f"|difference|, max {diff.max():.2f}", fontsize=8)
        row[0].set_ylabel(name, fontsize=7)
        row[1].set_xlabel(textwrap.fill(drawn, 78), fontsize=5.5, family="monospace")
    handles = [
        mpatches.Patch(color=DET_COLOR, label="detection"),
        mpatches.Patch(color=RECT_COLOR, label="paste rectangle (jittered +10-20%)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    save(fig, out_path, title, bottom=0.03)


def run_augment(aug: ShipAugmenter, crops, out_dir: Path, seed: int, n_show=4):
    """Figure 06: the training pairs.  Figure 07: what each corruption does."""
    if not crops:
        print("  no crops to augment -- skipping augmentation figures")
        return
    pairs = []
    for i, crop in enumerate(crops[:n_show]):
        pair = aug(crop, seed=seed + i)
        drawn = " ".join(
            f"{k}{_fmt(v)}" for fam in ("geometric", "photometric") for k, v in pair.params[fam].items()
        )
        name = Path(crop.path).name[:26] if crop.path else f"crop {i}"
        pairs.append((pair, name, f"seed {pair.seed}: {drawn}"))
    plot_pairs(
        pairs,
        out_dir / "06_augmented_pairs.png",
        f"Step 06: training pairs -- ops {list(DEFAULT_OPS)}",
    )

    # one crop, one op at a time, each turned up until it is visible on paper
    crop = crops[0]
    demos = [
        ("shift", {"max_px": 10, "max_frac": 1.0}),
        ("rotate", {"max_deg": 10}),
        ("rescale", {"factor": (1.15, 1.18)}),
        ("gain", {"factor": (1.3, 1.35)}),
        ("level", {"amount": 0.12}),
        ("gamma", {"factor": (0.55, 0.6)}),
        ("blur", {"sigma": (2.0, 2.2)}),
        ("noise", {"sigma": (0.06, 0.07)}),
        ("stripe", {"sigma": (0.04, 0.05)}),
    ]
    fig, axes = grid(len(demos) + 2, 4, figsize_per=2.8)
    clean = aug(crop, seed=seed).clean
    axes[0].set_axis_on()
    axes[0].imshow(stretch(clean), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("clean", fontsize=8)
    for ax, (name, kwargs) in zip(axes[1:], demos):
        # p=1 and a fixed rectangle: every panel differs only by its one op
        one = ShipAugmenter(ops={name: {**kwargs, "p": 1.0}}, context=(0.15, 0.15), offset_frac=0.0)
        pair = one(crop, seed=seed)
        ax.set_axis_on()
        ax.imshow(stretch(pair.doctored, ref=pair.clean), cmap="gray", vmin=0, vmax=1)
        draw_box(ax, pair.rect, RECT_COLOR)
        drawn = {**pair.params["geometric"], **pair.params["photometric"]}
        ax.set_title(f"{name}  {_fmt(next(iter(drawn.values()), {}))}", fontsize=8)
    ax = axes[len(demos) + 1]
    pair = ShipAugmenter()(crop, seed=seed)
    ax.set_axis_on()
    ax.imshow(stretch(pair.doctored, ref=pair.clean), cmap="gray", vmin=0, vmax=1)
    draw_box(ax, pair.rect, RECT_COLOR)
    ax.set_title("all defaults, stacked", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    save(
        fig,
        out_dir / "07_corruption_ops.png",
        "Step 06: one corruption at a time, exaggerated (magenta = paste rectangle)",
    )


def run_paste(aug: ShipAugmenter, cropper: ShipCropper, sim_crops, real_frames, out_dir: Path, seed: int, n_show=3):
    """Figure 08: the same machinery at test time -- a synthetic ship in a real frame.

    Not a training pair: the target does not exist.  It is here because the
    composite has to come out of the same code as the training one, and this is
    the figure that shows whether it does.
    """
    if not sim_crops or not real_frames:
        print("  need both halves of the data for the paste figure -- skipping")
        return
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(min(n_show, len(sim_crops), len(real_frames))):
        sim = sim_crops[i]
        frame = real_frames[i][0]
        # the ship chunk out of the synthetic crop, at its native pixel size
        x0, y0, x1, y1 = sim.paste_box_in_patch().clipped(*sim.patch.shape[:2][::-1]).as_int()
        donor = sim.patch[y0:y1, x0:x1]
        if min(donor.shape[:2]) < 4:
            continue
        # naive placement -- lower half, and far enough in that the display
        # window does not run off the frame.  *Where* a ship plausibly goes is a
        # question for step 08; this figure is about the composite, not the spot.
        h, w = frame.shape[:2]
        dh, dw = donor.shape[:2]
        half = cropper.out_size / 2
        cx = float(np.clip(rng.uniform(0.25, 0.75) * w, min(half, w / 2), max(w - half, w / 2)))
        cy = float(np.clip(rng.uniform(0.55, 0.85) * h, min(half, h / 2), max(h - half, h / 2)))
        target = BBox(cx - dw / 2, cy - dh / 2, cx + dw / 2, cy + dh / 2, label="ship")
        plain = aug.augment(frame, target, source=donor, seed=seed + i)
        matched = aug.augment(frame, target, source=donor, seed=seed + i, match_levels=True)
        rows.append((sim, donor, cropper.crop(frame, target), cropper.crop(plain.doctored, target),
                     cropper.crop(matched.doctored, target), plain))

    if not rows:
        print("  no usable synthetic donors -- skipping the paste figure")
        return
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, ((sim, donor, host, pasted, matched, pair), row) in enumerate(zip(rows, axes)):
        panels = [
            (donor, None, "synthetic ship chunk"),
            (host.patch, None, "real frame, before"),
            (pasted.patch, host.patch, "pasted (levels as rendered)"),
            (matched.patch, host.patch, "pasted, match_levels=True"),
        ]
        for ax, (img, ref, name) in zip(row, panels):
            ax.imshow(stretch(img, ref=ref), cmap="gray", vmin=0, vmax=1)
            if ref is not None:
                draw_box(ax, pair.rect.shifted(-pasted.window.x_min, -pasted.window.y_min), RECT_COLOR)
            if i == 0:
                ax.set_title(name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"{donor.shape[1]}x{donor.shape[0]} px", fontsize=7)
    save(
        fig,
        out_dir / "08_synthetic_into_real.png",
        "Step 06 at test time: synthetic ship crops pasted into real IR frames",
    )


def run_self_paste(aug, cropper: ShipCropper, sim_crops, out_dir: Path, seed: int, n_show=3):
    """Figure 09: the same test-time paste, pointed back at the renders themselves.

    Figure 08 is what the model is *for* and cannot be scored: nobody ever
    photographed that ship in that harbour, so there is no right answer to
    compare against.  This one puts each synthetic ship back into its own render
    by the recipe the training set is made with, which leaves the original as the
    answer -- the only test images in this project where "is the output correct"
    is a question with a number behind it.

    Built through CompositeGenerator rather than by hand, so what is on paper is
    what step 10 will actually be fed.
    """
    paths = list(dict.fromkeys(c.path for c in sim_crops if c.path))
    if not paths:
        print("  no synthetic crops -- skipping the self-paste figure")
        return
    # the cropper and its (warm) metadata detector are reused, so these rows are
    # the same windows as figures 03/04, and no CSV is read twice
    gen = ShipDataGenerator(
        paths,
        detector=cropper.detector,
        cropper=cropper,
        augmenter=aug,
        min_side=4.0,
        seed=seed,
        progress=False,
    )
    test = CompositeGenerator.self_paste(gen, seed=seed)
    print(f"  {test!r}")
    rows = [test[i] for i in range(min(n_show, len(test)))]

    fig, axes = plt.subplots(len(rows), 5, figsize=(16, 3.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (c, row) in enumerate(zip(rows, axes)):
        rect = mask_box(c.mask)
        zx0, zy0, zx1, zy1 = zoom_window(rect, c.patch.shape[:2]).as_int()
        zoom = (slice(zy0, zy1), slice(zx0, zx1))
        near = rect.shifted(-zx0, -zy0)  # the rectangle in the zoom's coordinates
        diff = np.abs(to_float01(c.patch) - to_float01(c.target))
        # the two close-ups share the target's stretch, or a gain change would
        # re-normalise itself away and the seam would vanish from the figure
        ship, near_ship = c.box_in_patch(), c.box_in_patch().shifted(-zx0, -zy0)
        panels = [
            (c.target[zoom], None, (near_ship,), "close up: the target"),
            (c.patch[zoom], c.target[zoom], (near, near_ship), "close up: what the model is given"),
            (c.target, None, (), "the render, before -- the target"),
            (c.patch, c.target, (rect, ship), "pasted back into itself -- the input"),
            (diff, DIFFERENCE, (rect, ship), f"|difference|, max {diff.max():.2f}"),
        ]
        for ax, (img, ref, boxes, name) in zip(row, panels):
            if ref is DIFFERENCE:  # a real scale, not a stretch: how much moved, and where
                ax.imshow(img, cmap="magma", vmin=0, vmax=max(float(img.max()), 1e-6))
            else:
                ax.imshow(stretch(img, ref=ref), cmap="gray", vmin=0, vmax=1)
            for box, color in zip(boxes, (RECT_COLOR, DET_COLOR) if len(boxes) > 1 else (DET_COLOR,)):
                draw_box(ax, box, color, ls="-" if color == RECT_COLOR else ":")
            ax.set_title(name if i == 0 else "", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        drawn = " ".join(
            f"{k}{_fmt(v)}" for fam in ("geometric", "photometric") for k, v in c.params[fam].items()
        )
        row[0].set_ylabel(
            f"{Path(c.donor_path).name[:22]}\nship {c.box.width:.0f}x{c.box.height:.0f} px", fontsize=6
        )
        row[3].set_xlabel(textwrap.fill(f"seed {c.seed}: {drawn}", 108), fontsize=5.5, family="monospace")
    handles = [
        mpatches.Patch(color=DET_COLOR, label="ship box (the metadata's, dotted)"),
        mpatches.Patch(color=RECT_COLOR, label="paste rectangle -- the only pixels the model may change"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    save(
        fig,
        out_dir / "09_synthetic_into_self.png",
        f"Step 07 test set: synthetic ships pasted back into their own renders "
        f"(roll capped at {test.max_rotation_deg}°)",
        bottom=0.03,
    )


def mask_box(mask: np.ndarray) -> BBox:
    """The rectangle the paste owns, read back off the alpha it was composited through."""
    ys, xs = np.nonzero(mask > 0.5)
    if not len(xs):
        return BBox(0, 0, 0, 0)
    return BBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def zoom_window(rect: BBox, shape: tuple[int, int], margin: float = 0.45) -> BBox:
    """`rect` plus surroundings, clipped to the patch.

    The artefact worth looking at is the seam, which is the *boundary* of the
    rectangle -- a close-up cropped to the rectangle itself would show the pasted
    content with nothing to compare it against.
    """
    h, w = shape
    pad = margin * max(rect.width, rect.height)
    return BBox(rect.x_min - pad, rect.y_min - pad, rect.x_max + pad, rect.y_max + pad).clipped(w, h)


def _fmt(params: dict) -> str:
    return "(" + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()) + ")"


# --------------------------------------------------------------------------- #
# window geometry
# --------------------------------------------------------------------------- #


def run_window_geometry(cropper: ShipCropper, out_dir: Path):
    """A tiny ship and a huge one, to show when the window stops being native-res."""
    import pandas as pd

    cases = []
    for scene, want in (("atlant_0000", "smallest"), ("msc_danit_0000", "largest")):
        csvs = sorted(glob.glob(str(ARETE_ROOT / "full_ir" / "metadata" / scene / "*.csv")))
        df = pd.concat(
            [
                pd.read_csv(c, usecols=["image_name", "x_min_tight_lwir", "x_max_tight_lwir", "is_target_visible_lwir"])
                for c in csvs
            ],
            ignore_index=True,
        )
        df = df[df.is_target_visible_lwir == 1]
        w = df.x_max_tight_lwir - df.x_min_tight_lwir
        row = df.loc[w.idxmin() if want == "smallest" else w.idxmax()]
        cases.append((scene, ARETE_ROOT / "full_ir" / "images" / scene / "lwir" / f"{row.image_name}.png"))

    fig, axes = plt.subplots(len(cases), 2, figsize=(8, 4 * len(cases)))
    axes = np.atleast_2d(axes)
    for (scene, path), (ax_full, ax_crop) in zip(cases, axes):
        im = load_image(path)
        crop = cropper.crop_all(im, path)[0]
        ax_full.imshow(stretch(im), cmap="gray", vmin=0, vmax=1)
        draw_box(ax_full, crop.window, WIN_COLOR, ls="--")
        draw_box(ax_full, crop.detection, DET_COLOR)
        ax_full.set_title(
            f"{scene}\nframe {im.shape[1]}x{im.shape[0]}, ship {crop.detection.width:.0f}x"
            f"{crop.detection.height:.0f} px, window {crop.window.width:.0f} px",
            fontsize=8,
        )
        ax_crop.imshow(stretch(crop.patch), cmap="gray", vmin=0, vmax=1)
        draw_box(ax_crop, crop.detection_in_patch(), DET_COLOR)
        ax_crop.set_title(
            f"patch {crop.patch.shape[1]}x{crop.patch.shape[0]}, scale={crop.scale:.3f}"
            f" ({'native res' if crop.scale == 1.0 else 'downscaled to fit'})",
            fontsize=8,
        )
        for ax in (ax_full, ax_crop):
            ax.set_xticks([])
            ax.set_yticks([])
    save(
        fig,
        out_dir / "05_window_geometry.png",
        f"Crop window: never below {cropper.out_size} px, downscaled only for big ships",
    )


# --------------------------------------------------------------------------- #


def detector_args(pairs: list[str]) -> dict:
    """--detector-arg conf_thresh=0.3 --detector-arg class_names=['boat','aeroplane']"""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--detector-arg needs key=value, got {p!r}")
        k, _, v = p.partition("=")
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-size", type=int, default=256)
    ap.add_argument("--paste-context", type=float, default=0.15)
    ap.add_argument("--max-crops", type=int, default=3, help="keep at most N ships per real frame")
    ap.add_argument(
        "--detector",
        default="yolov3",
        help="bounding box model for the real IR frames: a registered name, or 'module:ClassName'",
    )
    ap.add_argument(
        "--detector-arg",
        action="append",
        default=[],
        metavar="K=V",
        help="constructor argument for --detector; repeatable",
    )
    ap.add_argument("--conf", type=float, default=None, help="shorthand for --detector-arg conf_thresh=")
    ap.add_argument("--skip-real", action="store_true", help="synthetic figures only (no detector needed)")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--skip-augment", action="store_true", help="cropping figures only (steps 05)")
    ap.add_argument(
        "--feather", type=float, default=0.0, help="soften the paste seam by this many px (step 06)"
    )
    ap.add_argument("--check", action="store_true", help="run the contract check on --detector first")
    ap.add_argument("--list-detectors", action="store_true", help="print the registry and exit")
    args = ap.parse_args()

    if args.list_detectors:
        for name, doc in describe_detectors().items():
            print(f"  {name:<12}  {doc}")
        print("\n  ...or any 'module:ClassName' path to a ShipDetector subclass.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sim_cropper = ShipCropper("arete", out_size=args.out_size, paste_context=args.paste_context)
    sim_crops, real_frames = [], []

    if not args.skip_synthetic:
        print("synthetic IR (arete metadata boxes)")
        sim_crops = run_synthetic(sim_cropper, args.out, rng)
        run_window_geometry(sim_cropper, args.out)

    if not args.skip_real:
        kwargs = detector_args(args.detector_arg)
        if args.conf is not None:
            kwargs["conf_thresh"] = args.conf
        print(f"real IR ({args.detector})")
        detector = build_detector(args.detector, **kwargs)
        print(f"  {detector!r}")
        if args.check:
            pool = sorted(glob.glob(str(next(iter(REAL_POOLS.values())))))[:2]
            check_detector(detector, samples=pool or None, strict=True)

        real_cropper = ShipCropper(
            detector, out_size=args.out_size, paste_context=args.paste_context, max_crops=args.max_crops
        )
        real_frames = run_real(real_cropper, args.out, rng, tag=args.detector)

    if not args.skip_augment:
        print("augmentation (step 06)")
        aug = ShipAugmenter(feather_px=args.feather)
        print(f"  {aug!r}")
        check_augmenter(aug, strict=True)
        # prefer real crops: the corruption has to look plausible on the imagery
        # the model is actually trained on
        run_augment(aug, [c for _, cs, _ in real_frames for c in cs] or sim_crops, args.out, args.seed)
        run_paste(aug, sim_cropper, sim_crops, real_frames, args.out, args.seed)
        run_self_paste(aug, sim_cropper, sim_crops, args.out, args.seed)

    print(f"done -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
