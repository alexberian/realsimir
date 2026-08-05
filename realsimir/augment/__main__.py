"""Inspect and validate the augmentation stage from the shell.

    python -m realsimir.augment list
    python -m realsimir.augment check
    python -m realsimir.augment check --set feather_px=2 --set p_clean=0.1
    python -m realsimir.augment check --ops shift --ops noise --image frame.jpg

`list` prints the registered corruptions and which of them are on by default.
`check` runs the invariants in check.py -- the gate to pass after writing an op.
"""

from __future__ import annotations

import argparse
import ast
import sys

from . import DEFAULT_OPS, ShipAugmenter, available_ops, check_augmenter, describe_ops


def _kv(pairs: list[str]) -> dict:
    """--set feather_px=2 --set context=(0.1,0.3)"""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--set needs key=value, got {p!r}")
        k, _, v = p.partition("=")
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m realsimir.augment",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show the registered corruptions")

    c = sub.add_parser("check", help="run the augmenter against its invariants")
    c.add_argument("--ops", action="append", default=[], help="use only these ops; repeatable")
    c.add_argument("--set", action="append", default=[], metavar="K=V", help="ShipAugmenter argument")
    c.add_argument("--image", default=None, help="sample frame; a synthetic one is used otherwise")
    c.add_argument("--ship", default=None, metavar="X0,Y0,X1,Y1", help="ship box in --image")
    c.add_argument("--trials", type=int, default=8)

    args = ap.parse_args(argv)

    if args.cmd == "list":
        width = max(len(n) for n in available_ops())
        for name, doc in describe_ops().items():
            mark = "*" if name in DEFAULT_OPS else " "
            print(f" {mark} {name:<{width}}  {doc}")
        print(f"\n  * = in DEFAULT_OPS, in the order they run: {', '.join(DEFAULT_OPS)}")
        print("  ...or any 'module:ClassName' path to a Corruption subclass.")
        return 0

    aug = ShipAugmenter(ops=args.ops or None, **_kv(args.set))
    print(f"built {aug!r}")

    image = ship = None
    if args.image:
        from ..imaging import load_image

        image = load_image(args.image)
    if args.ship:
        from ..boxes import BBox

        ship = BBox(*(float(v) for v in args.ship.split(",")))

    return 1 if check_augmenter(aug, image=image, ship=ship, trials=args.trials) else 0


if __name__ == "__main__":
    sys.exit(main())
