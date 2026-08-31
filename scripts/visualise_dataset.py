#!/usr/bin/env python
"""Look at the data before training on it.

Produces: the static map channels, a video of one training window with the
past/future split marked, and a statistics summary.  Every number the model will
be judged on depends on these tensors being right, and the cheapest way to catch
a wrong sign or a transposed axis is to look.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c import viz  # noqa: E402
from g2c.data import EpisodeStore, OccupancyWindows, StaticMap  # noqa: E402
from g2c.tensorize import FLOW_SCALE, GROUP_NAMES, GROUP_RGB  # noqa: E402


def static_panel(static: StaticMap, out):
    """One picture per constant map channel, so a wrong sign is visible."""
    tiles = []
    for i, name in enumerate(static.names):
        ch = static.channels[i].numpy()
        lo, hi = float(ch.min()), float(ch.max())
        norm = (ch - lo) / max(hi - lo, 1e-6)
        # Diverging channels (the lane direction sines) read better centred.
        if lo < -0.1:
            img = np.stack([np.clip(-ch, 0, 1), np.zeros_like(ch), np.clip(ch, 0, 1)], -1)
        else:
            img = np.stack([norm, norm, norm], -1)
        img = (np.repeat(np.repeat(img, 3, 0), 3, 1) * 255).astype(np.uint8)
        tiles.append(viz.caption(img, [f"{name}   [{lo:.2f}, {hi:.2f}]"]))
    v = static.valid.numpy()
    img = (np.repeat(np.repeat(np.stack([v] * 3, -1), 3, 0), 3, 1) * 255).astype(np.uint8)
    tiles.append(viz.caption(img, [f"valid   {100*v.mean():.0f}% of cells"]))
    grid = viz.vstack([viz.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    viz.write_png(os.path.join(out, "static_map.png"), grid)
    return grid.shape


def window_video(store, windows, static, index, out, name, fps=4):
    """One training window as a video: ten observed frames then ten to predict."""
    w = windows.windows[index]
    road = static.channels[0].numpy() > 0.05
    dt = store.frame_dt_s
    frames = []
    span = windows.t_in + windows.t_out
    for k in range(span):
        cov, flow = store.densify(w.episode, w.start + k)
        future = k >= windows.t_in
        # Flow is stored in units of FLOW_SCALE m/s; convert to cells per frame
        # so the drawn segment length means "how far this moves in one step".
        cells = flow * FLOW_SCALE * dt / store.grid.cell
        img = viz.render_occupancy(cov, road=road, scale=5, flow=cells)
        tag = "TO PREDICT" if future else "OBSERVED"
        t_rel = (k - windows.t_in + 1) * dt if future else (k - windows.t_in + 1) * dt
        img = viz.caption(img, [
            f"episode {store.meta[w.episode]['episode']}  window {index}  "
            f"frame {k+1}/{span}   {tag}",
            f"t = {t_rel:+.1f} s   {int(cov.sum() > 0)} "
            f"occupied {100*(cov.sum(0)>0).mean():.2f}%   "
            f"scenario: {windows.scenario(index)}"])
        if future:
            img[:4, :] = [255, 90, 60]
            img[-4:, :] = [255, 90, 60]
            img[:, :4] = [255, 90, 60]
            img[:, -4:] = [255, 90, 60]
        img = viz.legend(img, [(g.replace('_', ' '), GROUP_RGB[g]) for g in GROUP_NAMES]
                              + [("motion (1 step)", (235, 245, 255))])
        frames.append(img)
    # Hold the last frame briefly so the end of the horizon is readable.
    frames += [frames[-1]] * 3
    return viz.write_video(os.path.join(out, name), frames, fps=fps)


def statistics(store, windows):
    rows = []
    occ_rate, cov_max, counts, speeds = [], [], [], []
    for e in range(len(store.episodes)):
        z = store.episodes[e]
        n = store.n_frames(e)
        for f in range(0, n, max(1, n // 30)):
            cov, flow = store.densify(e, f)
            occ = cov.sum(0)
            occ_rate.append((occ > 0).mean())
            cov_max.append(occ.max())
            speeds.append(np.abs(flow).max() * FLOW_SCALE)
        counts.append(z["vehicle_count"].sum(1).mean())
    from collections import Counter
    sc = Counter(windows.scenario(i) for i in range(len(windows)))
    return {
        "episodes": len(store.episodes),
        "frames": int(sum(store.n_frames(e) for e in range(len(store.episodes)))),
        "frame_dt_s": store.frame_dt_s,
        "windows": len(windows),
        "occupied_cell_fraction_mean": float(np.mean(occ_rate)),
        "max_cell_coverage": float(np.max(cov_max)),
        "vehicles_per_frame_mean": float(np.mean(counts)),
        "max_speed_m_s": float(np.max(speeds)),
        "scenario_counts": dict(sc),
        "mean_vehicle_area_m2": store.mean_vehicle_area().tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/lustre/fs1/home/mu290328/datasets/GTraffic2Crash")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--t-in", type=int, default=10)
    ap.add_argument("--t-out", type=int, default=10)
    ap.add_argument("--n-videos", type=int, default=2)
    args = ap.parse_args()

    static = StaticMap(os.path.join(args.data, "meta", "town05_static.npz"))
    print(static)
    print("static map panel:", static_panel(static, args.out))

    store = EpisodeStore(os.path.join(args.data, "tensor"), static.grid)
    windows = OccupancyWindows(store, args.t_in, args.t_out, stride=1)
    stats = statistics(store, windows)
    print(json.dumps(stats, indent=2))
    json.dump(stats, open(os.path.join(args.out, "dataset_statistics.json"), "w"), indent=2)

    # Prefer windows that contain something worth looking at.
    interesting = [i for i in range(len(windows)) if windows.scenario(i) != "nominal"]
    picks = (interesting[:args.n_videos] if interesting
             else list(range(0, len(windows), max(1, len(windows) // args.n_videos)))[:args.n_videos])
    for j, i in enumerate(picks):
        p = window_video(store, windows, static, i, args.out,
                         f"window_{windows.scenario(i)}_{j}.mp4")
        print("wrote", p)


if __name__ == "__main__":
    main()
