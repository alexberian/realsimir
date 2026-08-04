"""Inspect and validate bounding box back ends from the shell.

    python -m realsimir.detectors list
    python -m realsimir.detectors check arete --image /workspace/data/arete_ir_sim_data/.../x.png
    python -m realsimir.detectors check yolov3 --image frame.jpg --set conf_thresh=0.3
    python -m realsimir.detectors dump yolov3 --glob '/workspace/data/open_ir_images/**/*.jpg' \
        --out boxes.json

`check` is the gate to run when swapping in a new model: it exercises the
contract in base.py and prints what, if anything, is wrong.  `dump` freezes a
detector's output into a file the `precomputed` back end can serve.
"""

from __future__ import annotations

import argparse
import ast
import glob as globlib
import sys

from . import available_detectors, build_detector, check_detector, describe_detectors


def _kv(pairs: list[str]) -> dict:
    """--set conf_thresh=0.3 --set class_names=['boat','aeroplane']"""
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
    ap = argparse.ArgumentParser(prog="python -m realsimir.detectors", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show the registered detectors")

    c = sub.add_parser("check", help="run a detector against the ShipDetector contract")
    c.add_argument("detector")
    c.add_argument("--image", action="append", default=[], help="sample frame; repeatable")
    c.add_argument("--set", action="append", default=[], metavar="K=V", help="constructor argument")

    d = sub.add_parser("dump", help="freeze a detector's boxes into a precomputed JSON file")
    d.add_argument("detector")
    d.add_argument("--glob", required=True, help="quote it -- expanded here, not by the shell")
    d.add_argument("--out", required=True)
    d.add_argument("--set", action="append", default=[], metavar="K=V")
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--key", default="path", choices=("path", "name", "stem"))

    args = ap.parse_args(argv)

    if args.cmd == "list":
        width = max(len(n) for n in available_detectors())
        for name, doc in describe_detectors().items():
            print(f"  {name:<{width}}  {doc}")
        print("\n  ...or any 'module:ClassName' path to a ShipDetector subclass.")
        return 0

    detector = build_detector(args.detector, **_kv(args.set))
    print(f"built {detector!r}")

    if args.cmd == "check":
        return 1 if check_detector(detector, samples=args.image or None) else 0

    from .precomputed import dump_detections

    paths = sorted(globlib.glob(args.glob, recursive=True))[: args.limit]
    if not paths:
        raise SystemExit(f"no files matched {args.glob!r}")
    out = dump_detections(detector, paths, args.out, key=args.key)
    print(f"wrote {out} ({len(paths)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
