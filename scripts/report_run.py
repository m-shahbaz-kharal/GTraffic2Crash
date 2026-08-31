#!/usr/bin/env python
"""Compact progress report for a training run, from its validation log.

Weights & Biases runs offline on this cluster, so this is the readable view of
how a run is going: the gap over each baseline, how quality decays with forecast
horizon, and whether BIAS is drifting.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--root", default="/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/runs")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    path = os.path.join(args.root, args.run, "reports", "validation.jsonl")
    if not os.path.exists(path):
        print(f"no validation log yet at {path}")
        return 1
    recs = load(path)
    print(f"run '{args.run}': {len(recs)} validation points, "
          f"last at epoch {recs[-1]['epoch']} step {recs[-1]['step']}, "
          f"{recs[-1]['wall_seconds']/3600:.2f} h wall")

    print(f"\n{'epoch':>5} {'step':>8} {'soft_iou':>9} {'+0.5s':>7} {'+5.0s':>7} "
          f"{'bias':>6} {'f1@0.1':>7} {'count_err':>10} {'flow m/s':>9}")
    for r in recs:
        h = r["headline"]
        print(f"{r['epoch']:>5} {r['step']:>8} {h['soft_iou_mean']:>9.4f} "
              f"{h['soft_iou_first_step']:>7.4f} {h['soft_iou_last_step']:>7.4f} "
              f"{h['bias_mean']:>6.3f} {h['f1@0.1_mean']:>7.4f} "
              f"{h.get('vehicle_count_relative_error_mean', float('nan')):>10.4f} "
              f"{h.get('flow_endpoint_error_m_s_mean', float('nan')):>9.3f}")

    last = recs[-1]
    print(f"\nagainst the baselines (mean soft intersection-over-union):")
    m = np.mean(last["by_horizon"]["soft_iou"])
    print(f"   {'model':<36} {m:>7.4f}")
    for name, bm in last["baselines"].items():
        b = np.mean(bm["soft_iou"])
        print(f"   {name:<36} {b:>7.4f}   gap {m-b:+.4f}")

    print(f"\nby forecast horizon (model):")
    dt = last.get("frame_dt_s", 0.5)
    hor = last["by_horizon"]
    print("   " + "".join(f"{(i+1)*0.5:>8.1f}s" for i in range(len(hor['soft_iou']))))
    for key in ("soft_iou", "bias", "f1@0.1"):
        if key in hor:
            print(f"   {key:<9}" + "".join(f"{v:>9.3f}" for v in hor[key]))

    if last.get("by_scenario"):
        print(f"\nby scenario (mean soft intersection-over-union):")
        for name, h in sorted(last["by_scenario"].items()):
            print(f"   {name:<24} {h['soft_iou_mean']:>7.4f}   "
                  f"last step {h['soft_iou_last_step']:>6.3f}   bias {h['bias_mean']:>5.3f}")

    if args.plot and len(recs) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        step = [r["step"] for r in recs]
        ax[0].plot(step, [np.mean(r["by_horizon"]["soft_iou"]) for r in recs],
                   label="model", lw=2)
        for name in recs[-1]["baselines"]:
            ax[0].axhline(np.mean(recs[-1]["baselines"][name]["soft_iou"]),
                          ls="--", lw=1, label=name.replace("_", " "))
        ax[0].set_xlabel("step"); ax[0].set_ylabel("mean soft IoU")
        ax[0].set_title("forecast quality vs baselines"); ax[0].legend(fontsize=7)
        ax[1].plot(step, [r["headline"]["bias_mean"] for r in recs], lw=2)
        ax[1].axhline(1.0, ls="--", c="k", lw=1)
        ax[1].set_xlabel("step"); ax[1].set_title("bias (1.0 = unbiased)")
        h = recs[-1]["by_horizon"]["soft_iou"]
        ax[2].plot([(i+1)*dt for i in range(len(h))], h, "o-", label="model")
        for name, bm in recs[-1]["baselines"].items():
            ax[2].plot([(i+1)*dt for i in range(len(bm["soft_iou"]))], bm["soft_iou"],
                       "--", lw=1, label=name.replace("_", " "))
        ax[2].set_xlabel("forecast horizon (s)"); ax[2].set_title("decay with horizon")
        ax[2].legend(fontsize=7)
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(args.root, args.run, "reports", "progress.png")
        fig.savefig(out, dpi=110)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
