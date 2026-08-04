"""Demo / sanity check for todo step 05 (ship cropping object).

Runs the cropper over both halves of the data and dumps figures to ./figures:

    01_real_detections.png       real IR frames + YOLO 'boat' boxes and windows
    02_real_crops.png            the 256x256 patches those windows produce
    03_synthetic_detections.png  arete renders + ground-truth boxes and windows
    04_synthetic_crops.png       the 256x256 patches those windows produce
    05_window_geometry.png       tiny vs. huge ship -- native crop vs. downscale

Usage:  conda run -n realsimir python main.py [--out figures] [--seed 0]
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from realsimir import ARETE_ROOT, AreteMetadataDetector, ShipCropper, YoloShipDetector, load_image

REAL_ROOT = Path("/workspace/data/open_ir_images")
REAL_POOLS = {
    "raysense": REAL_ROOT / "raysense" / "红外船舶数据库" / "dataset" / "images" / "train" / "*.jpg",
    "MassMIND": REAL_ROOT / "MassMIND" / "Images" / "*.png",
}

DET_COLOR, PASTE_COLOR, WIN_COLOR = "#ff3b30", "#ffd60a", "#00d4ff"


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #


def stretch(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    """Percentile stretch -- LWIR renders are near-flat and invisible raw."""
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
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


def plot_frames(samples, out_path, title, ncols=4):
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
        mpatches.Patch(color=PASTE_COLOR, label="paste box (+15%)"),
        mpatches.Patch(color=WIN_COLOR, label="256 crop window"),
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


def run_real(cropper: ShipCropper, out_dir: Path, rng: random.Random, pool_size=48, n_show=8):
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
        print(f"  [{name}] {hits}/{len(sample)} frames with >=1 boat detection")

    if not keep:
        print("  no real detections at all -- skipping real figures")
        return
    rng.shuffle(keep)
    frames = keep[:n_show]
    plot_frames(frames, out_dir / "01_real_detections.png", "Real IR: YOLOv3 'boat' detections + crop windows")
    crops = [c for _, cs, _ in frames for c in cs][:8]
    plot_crops(crops, out_dir / "02_real_crops.png", "Real IR: 256x256 ship crops")


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


def run_synthetic(cropper: ShipCropper, out_dir: Path, rng: random.Random, n_show=8):
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
        return
    plot_frames(
        frames, out_dir / "03_synthetic_detections.png", "Synthetic IR (arete): ground-truth boxes + crop windows"
    )
    plot_crops(crops[:8], out_dir / "04_synthetic_crops.png", "Synthetic IR (arete): 256x256 ship crops")


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
    save(fig, out_dir / "05_window_geometry.png", "Crop window: never below 256 px, downscaled only for big ships")


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-size", type=int, default=256)
    ap.add_argument("--paste-context", type=float, default=0.15)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO objectness*class threshold")
    ap.add_argument("--max-crops", type=int, default=3, help="keep at most N ships per real frame")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("synthetic IR (arete metadata boxes)")
    sim_cropper = ShipCropper(
        AreteMetadataDetector(), out_size=args.out_size, paste_context=args.paste_context
    )
    run_synthetic(sim_cropper, args.out, rng)
    run_window_geometry(sim_cropper, args.out)

    print("real IR (yolov3 'boat')")
    yolo = YoloShipDetector(conf_thresh=args.conf)
    print(f"  device={yolo.device}, input={yolo.input_size}, conf>={args.conf}")
    real_cropper = ShipCropper(
        yolo, out_size=args.out_size, paste_context=args.paste_context, max_crops=args.max_crops
    )
    run_real(real_cropper, args.out, rng)

    print(f"done -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
