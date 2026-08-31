#!/usr/bin/env python
"""Rasterise Town05's road network onto the model's grid, once.

The network is given the map as constant input channels.  This is free
information that would otherwise cost capacity to rediscover: 28 % of cells
touch road and hold about 80 % of all vehicle mass, and lane direction says
which way traffic on a given cell is allowed to move -- a strong prior on the
flow field the model has to predict.

Channels produced:
    road           fraction of the cell covered by drivable lane
    lane_sin/cos   area-weighted mean lane heading, as a unit vector.  Opposing
                   carriageways in one cell partly cancel, which is honest: that
                   cell genuinely carries traffic both ways.
    lane_coherence magnitude of that mean before normalising, i.e. how
                   single-directional the cell is.  1.0 = one-way, 0 = balanced.
    junction       fraction of the cell inside a junction
    speed_limit    area-weighted speed limit, in m/s

Every lane is drawn with the same exact polygon-intersection rasteriser used for
vehicles, so road area and vehicle area are measured on identical terms.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c.grid import Grid, TOWN05_PADDED, _overlap_areas  # noqa: E402

STEP = 0.5  # metres between sampled waypoints along each lane


def accumulate(grid, cx, cy, yaw, length, width, values, chunk=4000):
    """Area-weighted scatter of per-waypoint values into grid cells.

    Returns (area, weighted_sums) where weighted_sums[i] is sum(area * values[i]).
    """
    R, C = grid.rows, grid.cols
    area_acc = np.zeros(R * C)
    val_acc = [np.zeros(R * C) for _ in values]
    for s in range(0, len(cx), chunk):
        e = slice(s, s + chunk)
        rr, cc, ar = _overlap_areas(grid, cx[e], cy[e], yaw[e], length[e], width[e])
        keep = (rr >= 0) & (rr < R) & (cc >= 0) & (cc < C) & (ar > 0)
        flat = np.where(keep, rr * C + cc, 0)
        ar = np.where(keep, ar, 0.0)
        area_acc += np.bincount(flat.reshape(-1), weights=ar.reshape(-1), minlength=R * C)
        for i, v in enumerate(values):
            val_acc[i] += np.bincount(flat.reshape(-1),
                                      weights=(ar * np.asarray(v)[e][:, None]).reshape(-1),
                                      minlength=R * C)
    return area_acc.reshape(R, C), [v.reshape(R, C) for v in val_acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import carla
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(180.0)
    world = client.get_world()
    if args.town not in world.get_map().name:
        world = client.load_world(args.town)
    cmap = world.get_map()

    wps = cmap.generate_waypoints(STEP)
    print(f"[map] {len(wps)} waypoints at {STEP} m spacing on {cmap.name}", flush=True)

    cx, cy, yaw, wid, junc, lim = [], [], [], [], [], []
    for w in wps:
        t = w.transform
        cx.append(t.location.x); cy.append(t.location.y)
        yaw.append(t.rotation.yaw); wid.append(max(w.lane_width, 1.0))
        junc.append(1.0 if w.is_junction else 0.0)
    cx = np.asarray(cx); cy = np.asarray(cy); yaw = np.asarray(yaw)
    wid = np.asarray(wid); junc = np.asarray(junc)
    length = np.full(len(cx), STEP)

    # Speed limits live on landmarks, not waypoints; fall back to the town default.
    lim = np.full(len(cx), 30.0 / 3.6)
    for w, i in zip(wps, range(len(wps))):
        try:
            signs = w.get_landmarks_of_type(60.0, "274")
            if signs:
                lim[i] = float(signs[0].value) / 3.6
        except Exception:
            pass

    grid = TOWN05_PADDED
    s, c = np.sin(np.radians(yaw)), np.cos(np.radians(yaw))
    area, (sin_w, cos_w, junc_w, lim_w) = accumulate(
        grid, cx, cy, yaw, length, wid, [s, c, junc, lim])

    cell_area = grid.cell ** 2
    road = np.clip(area / cell_area, 0, 1).astype(np.float32)
    safe = np.where(area > 1e-9, area, 1.0)
    mean_sin, mean_cos = sin_w / safe, cos_w / safe
    coherence = np.hypot(mean_sin, mean_cos)
    norm = np.where(coherence > 1e-6, coherence, 1.0)
    out = {
        "road": road,
        "lane_sin": (mean_sin / norm).astype(np.float32),
        "lane_cos": (mean_cos / norm).astype(np.float32),
        "lane_coherence": coherence.astype(np.float32),
        "junction": np.clip(junc_w / safe, 0, 1).astype(np.float32),
        "speed_limit_m_s": (lim_w / safe).astype(np.float32),
    }
    for k in out:
        out[k][area <= 1e-9] = 0.0

    # Validity: cells inside the town's own bounding box.  The padded grid extends
    # past the map, and those cells must not count towards any metric.
    xs = np.array([w.transform.location.x for w in wps])
    ys = np.array([w.transform.location.y for w in wps])
    rows = np.arange(grid.rows)[:, None] * grid.cell + grid.origin_y
    cols = np.arange(grid.cols)[None, :] * grid.cell + grid.origin_x
    margin = 30.0
    valid = ((cols >= xs.min() - margin) & (cols <= xs.max() + margin) &
             (rows >= ys.min() - margin) & (rows <= ys.max() + margin))
    out["valid"] = np.broadcast_to(valid, (grid.rows, grid.cols)).astype(np.float32)

    meta = {"town": cmap.name, "grid": grid.as_dict(), "waypoint_step_m": STEP,
            "n_waypoints": len(wps),
            "road_cell_fraction": float((road > 0.05).mean()),
            "valid_cell_fraction": float(out["valid"].mean()),
            "world_bounds_xy": [float(xs.min()), float(ys.min()),
                                float(xs.max()), float(ys.max())],
            "channels": list(out)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, meta=json.dumps(meta), **out)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[map] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
