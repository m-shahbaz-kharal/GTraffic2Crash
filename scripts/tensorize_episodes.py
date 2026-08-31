#!/usr/bin/env python
"""Rasterise recorded episodes into the sparse tensors the model trains on."""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c.grid import TOWN05_PADDED  # noqa: E402
from g2c.tensorize import tensorise_episode  # noqa: E402


def one(job):
    src, dst, stride = job
    if os.path.exists(dst):
        return dst, None
    meta = tensorise_episode(src, dst, TOWN05_PADDED, stride)
    return dst, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=5,
                    help="recorded frames per stored frame; 5 turns 10 Hz into 2 Hz")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    jobs = []
    for src in sorted(glob.glob(os.path.join(args.raw, "*.npz"))):
        jobs.append((src, os.path.join(args.out, os.path.basename(src)), args.stride))
    print(f"{len(jobs)} episodes -> {args.out} (stride {args.stride})", flush=True)

    done, frames, colls = 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for dst, meta in ex.map(one, jobs):
            done += 1
            if meta:
                frames += meta["n_frames"]; colls += meta["n_collisions"]
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  frames {frames}  collisions {colls}", flush=True)
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(args.out, "*.npz")))
    print(f"done: {frames} frames, {colls} collisions, {size/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
