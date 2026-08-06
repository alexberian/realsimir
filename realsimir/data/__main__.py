"""Build, inspect and validate the data stream from the shell.

    python -m realsimir.data pools
    python -m realsimir.data index --source real   --out boxes/real.json
    python -m realsimir.data index --source arete  --out boxes/arete.json --limit 4000
    python -m realsimir.data check --source real   --index-file boxes/real.json
    python -m realsimir.data stats --source real   --index-file boxes/real.json

`index` is the one long-running command in this project's front end -- a pass of
the bounding box model over the pool -- and the reason it is a separate command
is that it should be run once, on purpose, and not implicitly at the start of a
training run.  `check` is the gate before step 09: it exercises the invariants in
check.py against the real index, and exits non-zero if any of them fails.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys

from . import (
    CropIndex,
    ShipDataGenerator,
    available_pools,
    check_generator,
    describe_pools,
    get_pool,
    resolve_paths,
)


def _kv(pairs: list[str]) -> dict:
    """--set out_size=128 --set paste_context=0.2"""
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


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="real", help="pool name, glob, directory or file")
    p.add_argument("--detector", default=None, help="bounding box model; default is the pool's own")
    p.add_argument("--limit", type=int, default=None, help="sample this many frames from the pool")
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame (video pools)")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--min-side", type=float, default=8.0, help="drop ships thinner than this, px")
    p.add_argument("--max-per-frame", type=int, default=None)
    p.add_argument("--out-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="K=V", help="ShipCropper argument")


def _generator(args) -> ShipDataGenerator:
    return ShipDataGenerator(
        source=args.source,
        detector=args.detector,
        index_file=getattr(args, "index_file", None),
        limit=args.limit,
        stride=args.stride,
        sample_seed=args.sample_seed,
        min_score=args.min_score,
        min_side=args.min_side,
        max_per_frame=args.max_per_frame,
        cropper={"out_size": args.out_size, **_kv(args.set)},
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m realsimir.data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pools", help="list the registered frame pools")
    p.add_argument("--count", action="store_true", help="also count the files (globs the disk)")

    p = sub.add_parser("index", help="run the detector over a pool and cache its boxes")
    _common(p)
    p.add_argument("--out", required=True, help="where to write the index (a precomputed-boxes JSON)")
    p.add_argument("--force", action="store_true", help="re-detect even if --out exists")

    p = sub.add_parser("check", help="run the generator against its invariants")
    _common(p)
    p.add_argument("--index-file", default=None)
    p.add_argument("--trials", type=int, default=8)

    p = sub.add_parser("stats", help="print what the index and the generator contain")
    _common(p)
    p.add_argument("--index-file", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "pools":
        width = max(len(n) for n in available_pools())
        for name, doc in describe_pools().items():
            pool = get_pool(name)
            count = f"  [{len(pool)} files]" if args.count else ""
            print(f"  {name:<{width}}  {doc}{count}")
        print("\n  ...or any glob, directory or list of paths.")
        return 0

    if args.cmd == "index":
        import os

        if os.path.exists(args.out) and not args.force:
            index = CropIndex.load(args.out)
            print(f"{args.out} already exists: {index!r} (--force to rebuild)")
            return 0
        from ..detectors import build_detector

        paths = resolve_paths(args.source, limit=args.limit, stride=args.stride, seed=args.sample_seed)
        if not paths:
            raise SystemExit(f"no frames matched {args.source!r}")
        detector = build_detector(args.detector or _default_detector(args.source))
        print(f"indexing {len(paths)} frames from {args.source!r} with {detector!r}")
        index = CropIndex.build(
            paths,
            detector,
            min_score=args.min_score,
            min_side=args.min_side,
            max_per_frame=args.max_per_frame,
            meta={"source": str(args.source), "limit": args.limit, "stride": args.stride},
        )
        index.save(args.out)
        print(f"wrote {args.out}: {index!r}")
        return 0

    gen = _generator(args)
    if args.cmd == "stats":
        print(f"{gen!r}")
        print(json.dumps(gen.stats(), indent=2, default=str))
        return 0

    print(f"built {gen!r}")
    return 1 if check_generator(gen, trials=args.trials) else 0


def _default_detector(source: str):
    from .generator import _suggested_detector

    return _suggested_detector(source)


if __name__ == "__main__":
    sys.exit(main())
