"""Smoke-test the raster + rendering path on real Town05 traffic.

The upstream UCFTCam2TWorld dataset holds 16 episodes of Town05 vehicle states.
It is only 1 Hz, so it is no substitute for the data this project will generate,
but it is real traffic in the right town and it exercises every step between a
list of vehicles and a video -- which is worth checking before a single CARLA
episode is generated.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c.grid import CLASS_ID_TO_INDEX, TOWN05_PADDED, rasterise  # noqa: E402
from g2c import viz  # noqa: E402
from g2c.tensorize import CLASS_TO_GROUP, GROUP_NAMES, GROUP_RGB, N_GROUPS  # noqa: E402

ROOT = "/lustre/fs1/home/mu290328/datasets/UCFTCam2TWorld/dataset/episodes"
ROAD_PNG = "/lustre/fs1/home/mu290328/datasets/UCFTCam2TWorld/dataset/meta/road_bev.png"


def road_mask(grid):
    """Coarse drivable mask on this project's grid, from the upstream raster.

    A stand-in until the map is regenerated from CARLA with lane directions; the
    fraction of each cell that is road is kept rather than a hard flag, since the
    network is given it as a continuous prior channel.
    """
    im = np.array(Image.open(ROAD_PNG)).astype(np.float32)
    sub = int(round(grid.cell / 0.5))
    pad = np.zeros((grid.rows * sub, grid.cols * sub), np.float32)
    pad[:im.shape[0], :im.shape[1]] = im
    return pad.reshape(grid.rows, sub, grid.cols, sub).mean(axis=(1, 3))


def load(path):
    rec = json.load(open(os.path.join(path, "frame.json")))["vehicles"]
    arr = lambda k, i=None: np.array([v[k] if i is None else v[k][i] for v in rec])
    cls = np.array([CLASS_ID_TO_INDEX[v["category_id"]] for v in rec])
    return {"id": np.array([v["actor_id"] for v in rec]),
            "x": arr("center_world", 0), "y": arr("center_world", 1),
            "yaw": arr("yaw_deg"), "l": arr("size_lwh", 0), "w": arr("size_lwh", 1),
            "group": CLASS_TO_GROUP[cls]}


def main():
    ep = sys.argv[1] if len(sys.argv) > 1 else "ep_003"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    out = "docs/figures"
    grid = TOWN05_PADDED
    road = road_mask(grid)
    print(f"grid {grid.rows}x{grid.cols} at {grid.cell} m = "
          f"{grid.rows*grid.cell:.0f} x {grid.cols*grid.cell:.0f} m; "
          f"road covers {100*(road>0.05).mean():.1f}% of cells")

    frames = sorted(os.listdir(os.path.join(ROOT, ep)))[:n + 1]
    frames = [f for f in frames if f.startswith("frame_")]
    prev = None
    imgs, stats = [], []
    for k, f in enumerate(frames):
        v = load(os.path.join(ROOT, ep, f))
        # Velocity by finite difference on matched actor ids (1 s apart here).
        vx = np.zeros(len(v["x"])); vy = np.zeros(len(v["x"]))
        if prev is not None:
            lut = {i: j for j, i in enumerate(prev["id"])}
            for j, i in enumerate(v["id"]):
                if i in lut:
                    vx[j] = v["x"][j] - prev["x"][lut[i]]
                    vy[j] = v["y"][j] - prev["y"][lut[i]]
        out_r = rasterise(grid, v["x"], v["y"], v["yaw"], v["l"], v["w"], v["group"],
                          vx=vx, vy=vy, n_classes=N_GROUPS)
        cov = out_r["coverage"]
        occ = cov.sum(0)
        stats.append((float(out_r["mass"].sum()), float((occ > 0).mean()),
                      float(occ.max()), float(np.abs(vx).max())))
        # Flow is drawn in cells per frame, so divide the metres by the cell size.
        img = viz.render_occupancy(cov, road=road > 0.05, scale=5,
                                   flow=out_r["flow"] / grid.cell if prev is not None else None)
        img = viz.caption(img, [f"CARLA Town05  {ep}  frame {k:03d}  (upstream data, 1 Hz)",
                                f"{len(v['x'])} vehicles   occupied cells "
                                f"{100*(occ>0).mean():.2f}%"])
        img = viz.legend(img, [(g.replace('_', ' '), GROUP_RGB[g]) for g in GROUP_NAMES] +
                              [("motion (1 s)", (235, 245, 255))])
        imgs.append(img)
        prev = v
        if k == len(frames) // 2:
            viz.write_png(f"{out}/upstream_frame.png", img)

    viz.write_video(f"{out}/upstream_town05.mp4", imgs, fps=5)
    m = np.array(stats)
    print(f"rendered {len(imgs)} frames -> {out}/upstream_town05.mp4  ({imgs[0].shape})")
    print(f"vehicles {m[:,0].mean():.0f} (min {m[:,0].min():.0f} max {m[:,0].max():.0f})")
    print(f"occupied cells {100*m[:,1].mean():.2f}%   max cell coverage {m[:,2].max():.3f}")


if __name__ == "__main__":
    main()
