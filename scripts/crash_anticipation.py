#!/usr/bin/env python
"""Does the forecast make an imminent collision visible before it happens?

This is the question the whole project exists to answer, and it is deliberately
measured rather than trained for -- no term in the loss singles collisions out.

For every held-out window whose future contains a real collision, the impact cell
and the step it happens at are known.  Two things are then measured:

  lead time   at each step before impact, how much occupancy does the forecast
              place at the impact site?  A model that sees the collision coming
              should raise occupancy there ahead of time.
  contrast    the same quantity at a matched control site -- a random occupied
              cell in the same frame that is not the impact.  Without this the
              first number means nothing: a model that simply predicts high
              occupancy everywhere would score well.

The headline number is the ratio between them.  Above one means the forecast
concentrates mass at the place the crash actually happens, before it happens.
"""
import argparse
import json
import os
import time
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c.data import EpisodeStore, OccupancyWindows, StaticMap, split_episodes  # noqa: E402
from g2c.train import Forecaster  # noqa: E402


def patch(field, r, c, k=2):
    """Mean value in a (2k+1) square around (r, c), clipped to the grid."""
    H, W = field.shape[-2:]
    r0, r1 = max(0, r - k), min(H, r + k + 1)
    c0, c1 = max(0, c - k), min(W, c + k + 1)
    if r1 <= r0 or c1 <= c0:
        return float("nan")
    return float(field[..., r0:r1, c0:c1].mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="/lustre/fs1/home/mu290328/datasets/GTraffic2Crash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=1500)
    ap.add_argument("--min-speed", type=float, default=3.0)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Forecaster.load_from_checkpoint(args.checkpoint, map_location=dev).eval().to(dev)
    cfg = model.cfg
    static = StaticMap(os.path.join(args.data, "meta", "town05_static.npz"))
    grid = static.grid
    store = EpisodeStore(os.path.join(args.data, "tensor"), grid)

    sp = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "episode_split.json")
    if os.path.exists(sp):
        val_ids = set(json.load(open(sp))["validation_episodes"])
        rederived = False
    else:
        _, val_ids = split_episodes(store.meta, args.val_fraction, 0)
        rederived = True
        print("!" * 72)
        print("!! This run recorded no episode split, so it is being re-derived from")
        print("!! the episodes now on disk.  If the run trained on a different set,")
        print("!! these windows overlap its training data and the numbers below are")
        print("!! indicative only, not a held-out measurement.")
        print("!" * 72)
    val = OccupancyWindows(store, cfg.t_in, cfg.t_out, 3, episodes=val_ids)
    idx = [i for i in range(len(val))
           if val.windows[i].has_collision
           and val.windows[i].collision_speed >= args.min_speed
           and val.windows[i].collision_step >= 2]
    idx = idx[:args.max_windows]
    print(f"[crash] {len(idx)} held-out windows with a collision at or above "
          f"{args.min_speed} m/s, at least 1 s into the horizon")
    if not idx:
        return 1

    sub = torch.utils.data.Subset(val, idx)
    loader = DataLoader(sub, batch_size=args.batch_size, num_workers=6)
    ch = static.channels.to(dev)
    rng = np.random.default_rng(0)

    lead = defaultdict(list)      # steps-before-impact -> occupancy at impact site
    ctrl = defaultdict(list)      # the same, at a matched control site
    truth = defaultdict(list)
    t0 = time.time()
    for n, batch in enumerate(loader):
        # A rate line, because the first attempt at this hit its wall clock with
        # nothing in the log to say whether it was slow or stuck.
        if n and n % 20 == 0:
            done = n * args.batch_size
            rate = done / (time.time() - t0)
            print(f"[crash] {done}/{len(idx)} windows, {rate:.1f}/s, "
                  f"{(len(idx) - done) / max(rate, 1e-6) / 60:.1f} min left",
                  flush=True)
        b = {k: v.to(dev) for k, v in batch.items()}
        B = b["past_occupancy"].shape[0]
        prob = torch.sigmoid(model.net(b["past_occupancy"],
                                       ch.unsqueeze(0).expand(B, -1, -1, -1))
                             ["occupancy_logits"]).sum(2).cpu().numpy()
        true = b["future_occupancy"].sum(2).cpu().numpy()
        for i in range(B):
            x, y = batch["collision_xy"][i].tolist()
            if not np.isfinite(x):
                continue
            r = int((y - grid.origin_y) / grid.cell)
            c = int((x - grid.origin_x) / grid.cell)
            if not (0 <= r < grid.rows and 0 <= c < grid.cols):
                continue
            k = int(batch["collision_step"][i])
            # A control site: a cell that is occupied in the last observed frame
            # and well away from the impact, so the comparison is like for like.
            obs = b["past_occupancy"][i, -1].sum(0).cpu().numpy()
            cand = np.argwhere(obs > 0.1)
            cand = [p for p in cand if abs(p[0] - r) + abs(p[1] - c) > 12]
            if not cand:
                continue
            cr, cc = cand[rng.integers(len(cand))]
            for step in range(k + 1):
                before = k - step                       # steps before impact
                lead[before].append(patch(prob[i, step], r, c))
                ctrl[before].append(patch(prob[i, step], cr, cc))
                truth[before].append(patch(true[i, step], r, c))

    rows = []
    for before in sorted(lead):
        if len(lead[before]) < 20:
            continue
        L, C, T = np.mean(lead[before]), np.mean(ctrl[before]), np.mean(truth[before])
        rows.append({"steps_before_impact": before,
                     "seconds_before_impact": before * store.frame_dt_s,
                     "n": len(lead[before]),
                     "forecast_at_impact_site": round(float(L), 4),
                     "forecast_at_control_site": round(float(C), 4),
                     "truth_at_impact_site": round(float(T), 4),
                     "contrast_ratio": round(float(L / max(C, 1e-6)), 3)})

    print(f"\n{'before impact':>14} {'n':>6} {'forecast @ impact':>18} "
          f"{'@ control':>11} {'truth @ impact':>15} {'contrast':>9}")
    for r_ in rows:
        print(f"{r_['seconds_before_impact']:>12.1f} s {r_['n']:>6} "
              f"{r_['forecast_at_impact_site']:>18.4f} "
              f"{r_['forecast_at_control_site']:>11.4f} "
              f"{r_['truth_at_impact_site']:>15.4f} {r_['contrast_ratio']:>9.2f}x")
    json.dump({"held_out": not rederived, "rows": rows},
              open(os.path.join(args.out, "crash_anticipation.json"), "w"), indent=2)
    print(f"\n[crash] wrote {args.out}/crash_anticipation.json")
    print("[crash] contrast above 1 means the forecast concentrates occupancy at "
          "the place the crash happens, before it happens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
