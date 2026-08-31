#!/usr/bin/env python
"""Final evaluation: the model against the baselines, per horizon and per scenario.

Everything reported here comes from held-out episodes -- never held-out windows
from a training episode, which would share vehicles and overlap in time.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c import viz  # noqa: E402
from g2c.baselines import all_baselines  # noqa: E402
from g2c.data import EpisodeStore, OccupancyWindows, StaticMap, split_episodes  # noqa: E402
from g2c.metrics import Accumulator, table  # noqa: E402
from g2c.tensorize import FLOW_SCALE, GROUP_NAMES, GROUP_RGB  # noqa: E402
from g2c.train import Forecaster  # noqa: E402


def scenario_of(batch, i):
    if bool(batch["has_collision"][i]):
        return "collision" if float(batch["collision_speed"][i]) >= 3.0 else "low_speed_contact"
    return "incident_traffic" if int(batch["n_bad_actors"][i]) > 0 else "nominal"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="/lustre/fs1/home/mu290328/datasets/GTraffic2Crash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-stride", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--allow-rederived-split", action="store_true",
                    help="proceed when the run recorded no split; the result is "
                         "then not a valid held-out evaluation")
    ap.add_argument("--val-fraction", type=float, default=0.15,
                    help="must match the value training used, or the split differs "
                         "and the held-out set is no longer held out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Forecaster.load_from_checkpoint(args.checkpoint, map_location=dev)
    model.eval().to(dev)
    cfg = model.cfg
    static = StaticMap(os.path.join(args.data, "meta", "town05_static.npz"))
    store = EpisodeStore(os.path.join(args.data, "tensor"), static.grid)
    # Prefer the split the run actually used.  Deriving it again from whatever
    # episodes happen to be on disk would silently use a different held-out set
    # if any episodes were added since.
    sp = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)),
                      "episode_split.json")
    if os.path.exists(sp):
        rec = json.load(open(sp))
        val_ids = set(rec["validation_episodes"])
        print(f"[eval] using the run's recorded split "
              f"({len(val_ids)} held-out episodes of {rec['n_episodes_on_disk']})")
    else:
        _, val_ids = split_episodes(store.meta, args.val_fraction, 0)
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)),
                                "config.json")
        saw = None
        if os.path.exists(cfg_path):
            saw = json.load(open(cfg_path)).get("max_episodes")
        print(f"!! WARNING: this run recorded no episode split, so it is being "
              f"re-derived from the {len(store.meta)} episodes now on disk.")
        print(f"!! If the run trained on a different number of episodes, the "
              f"'held-out' set below overlaps its training data and these numbers "
              f"are NOT a valid held-out evaluation. Use the run's own in-training "
              f"validation instead.")
        if not args.allow_rederived_split:
            print("!! Refusing. Pass --allow-rederived-split to proceed anyway "
                  "(as a code-path check, not as a result).")
            return 1
    val = OccupancyWindows(store, cfg.t_in, cfg.t_out, args.val_stride, episodes=val_ids)
    print(f"[eval] {len(val):,} held-out windows from {len(val_ids)} episodes")
    print(f"[eval] scenarios: {dict(Counter(val.scenario(i) for i in range(len(val))))}")
    loader = DataLoader(val, batch_size=args.batch_size, num_workers=8)

    mva = torch.as_tensor(store.mean_vehicle_area(), device=dev)
    cell_area = static.grid.cell ** 2
    valid = static.valid.to(dev)
    accs = {"model": Accumulator(cfg.t_out)}
    per_scenario = defaultdict(lambda: defaultdict(lambda: Accumulator(cfg.t_out)))
    kept = []

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        batch = {k: v.to(dev) for k, v in batch.items()}
        B = batch["past_occupancy"].shape[0]
        pred = model.net(batch["past_occupancy"],
                         static.channels.to(dev).unsqueeze(0).expand(B, -1, -1, -1))
        prob = torch.sigmoid(pred["occupancy_logits"])
        forecasts = {"model": prob}
        forecasts.update(all_baselines(batch, model.cells_per_frame_step, cfg.t_out))
        common = dict(valid=valid, cell_area=cell_area, mean_vehicle_area=mva)
        for name, p in forecasts.items():
            accs.setdefault(name, Accumulator(cfg.t_out)).update(
                p, batch["future_occupancy"],
                flow_pred=pred["flow"] if name == "model" else None,
                flow_true=batch["future_flow"] if name == "model" else None, **common)
            for i in range(B):
                per_scenario[scenario_of(batch, i)][name].update(
                    p[i:i+1], batch["future_occupancy"][i:i+1], **common)
        for i in range(B):
            if scenario_of(batch, i) == "collision" and len(kept) < args.videos:
                kept.append(({k: v[i:i+1].cpu() for k, v in batch.items()},
                             prob[i:i+1].cpu()))

    results = {name: {k: list(map(float, v)) for k, v in a.compute(FLOW_SCALE).items()}
               for name, a in accs.items()}
    headline = {name: a.headline(FLOW_SCALE) for name, a in accs.items()}
    computed = {name: a.compute(FLOW_SCALE) for name, a in accs.items()}

    dt = store.frame_dt_s
    lines = ["SOFT INTERSECTION-OVER-UNION BY FORECAST HORIZON", "",
             table(computed, "soft_iou", dt), "",
             "BIAS  (predicted mass / true mass; 1.0 is unbiased, below 1 is hedging)", "",
             table(computed, "bias", dt), "",
             "F1 AT THRESHOLD 0.1", "", table(computed, "f1@0.1", dt), "",
             "F1 AT THRESHOLD 0.1, POOLED 4x4  (how much of the skill survives blur)", "",
             table(computed, "f1@0.1_pool4", dt), ""]
    lines.append("BY SCENARIO  (mean soft intersection-over-union over the horizon)")
    names = list(accs)
    lines.append(f"{'':<22}" + "".join(f"{n[:20]:>22}" for n in names))
    for sc, d in sorted(per_scenario.items()):
        row = f"{sc:<22}"
        for n in names:
            row += f"{np.mean(d[n].compute()['soft_iou']):>22.3f}" if n in d else f"{'-':>22}"
        lines.append(row)
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(args.out, "evaluation.txt"), "w").write(report)
    json.dump({"headline": headline, "by_horizon": results,
               "by_scenario": {sc: {n: a.headline(FLOW_SCALE) for n, a in d.items()}
                               for sc, d in per_scenario.items()},
               "frame_dt_s": dt, "windows": len(val)},
              open(os.path.join(args.out, "evaluation.json"), "w"), indent=2)

    road = static.channels[0].numpy() > 0.05
    for j, (b, p) in enumerate(kept):
        frames = []
        for k in range(cfg.t_in):
            im = viz.render_occupancy(b["past_occupancy"][0, k].numpy(), road=road, scale=4)
            frames.append(viz.hstack([viz.caption(im, [f"observed t={(k-cfg.t_in+1)*dt:+.1f}s"]),
                                      im * 0 + 18]))
        for k in range(cfg.t_out):
            t = (k + 1) * dt
            a = viz.caption(viz.render_occupancy(b["future_occupancy"][0, k].numpy(),
                                                 road=road, scale=4),
                            [f"ground truth t={t:+.1f}s", "collision window"])
            c = viz.caption(viz.render_occupancy(p[0, k].numpy(), road=road, scale=4),
                            [f"forecast t={t:+.1f}s"])
            frames.append(viz.hstack([a, c]))
        viz.write_video(os.path.join(args.out, f"collision_forecast_{j}.mp4"),
                        frames + [frames[-1]] * 4, fps=4)
    print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
