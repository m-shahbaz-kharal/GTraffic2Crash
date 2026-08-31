"""Turn recorded vehicle state into the sparse tensors the model trains on.

Roughly 98 % of the town grid is empty at any instant, so the rasterised tensors
are stored sparsely -- flat cell index plus an 8-bit value -- and densified in
the data loader.  A dense float16 store of the same content would be about 25x
larger for no extra information.

Each frame yields:
    coverage  per class, fraction of the cell covered, quantised to 1/255
    flow      coverage-weighted mean velocity, on cells that hold something
    counts    per-class vehicle mass, i.e. the number of vehicles
plus per-frame event labels (collision, speeding present, ...) so that training
windows can be sampled and scored by what happens in them.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .grid import CLASS_NAMES, Grid, rasterise

# Six CARLA classes collapse to three groups.  Six channels would leave
# motorcycles at ~0.05 % density, too sparse for a stable per-channel
# distribution; three keeps every channel above ~0.3 % while preserving the
# distinctions the dynamics actually depend on.
GROUPS = {"car": 0, "van": 1, "truck": 1, "bus": 1, "motorcycle": 2, "bicycle": 2}
GROUP_NAMES = ["car", "large_vehicle", "two_wheeler"]
GROUP_RGB = {"car": (0, 120, 255), "large_vehicle": (255, 120, 0),
             "two_wheeler": (255, 220, 0)}
N_GROUPS = len(GROUP_NAMES)
CLASS_TO_GROUP = np.array([GROUPS[c] for c in CLASS_NAMES], dtype=np.int64)

# Velocities are divided by this before being stored or fed to a network, so
# that a typical urban speed lands near 1.
FLOW_SCALE = 15.0

STOPPED_SPEED = 0.5      # m/s below which a vehicle counts as stopped
SPEEDING_SPEED = 16.0    # m/s (~58 km/h); Town05 limits are 30-60 km/h


def load_episode(path):
    """Read one recorded episode into plain arrays."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z, meta


def frame_vehicles(z, f):
    """Per-vehicle state at recorded frame `f`."""
    lo, hi = z["frame_offset"][f], z["frame_offset"][f + 1]
    v = z["record_vehicle"][lo:hi]
    return {
        "index": v,
        "xy": z["record_xy"][lo:hi],
        "yaw": z["record_yaw"][lo:hi],
        "velocity": z["record_velocity"][lo:hi],
        "extent": z["vehicle_extent"][v],
        "group": CLASS_TO_GROUP[z["vehicle_class"][v]],
        "behaviour": z["vehicle_behaviour"][v],
    }


def tensorise_episode(src, dst, grid: Grid, stride: int):
    """Rasterise one episode, keeping every `stride`-th recorded frame."""
    z, meta = load_episode(src)
    n_src = int(meta["n_frames"])
    keep = list(range(0, n_src, stride))
    H, W = grid.rows, grid.cols

    coll_frame = z["collision_frame"]
    coll_a, coll_b = z["collision_a"], z["collision_b"]

    cov_off, cov_idx, cov_val, cov_chan = [0], [], [], []
    flo_off, flo_idx, flo_vec = [0], [], []
    counts, n_veh, n_moving, max_speed, n_speeding = [], [], [], [], []
    has_coll, coll_xy, times = [], [], []
    behaviour_present = []

    for out_f, f in enumerate(keep):
        v = frame_vehicles(z, f)
        speed = np.hypot(v["velocity"][:, 0], v["velocity"][:, 1])
        out = rasterise(grid, v["xy"][:, 0], v["xy"][:, 1], v["yaw"],
                        v["extent"][:, 0], v["extent"][:, 1], v["group"],
                        vx=v["velocity"][:, 0], vy=v["velocity"][:, 1],
                        n_classes=N_GROUPS)
        cov = out["coverage"]
        c, r, cc = np.nonzero(cov)
        cov_chan.append(c.astype(np.uint8))
        cov_idx.append((r * W + cc).astype(np.int32))
        cov_val.append(np.round(cov[c, r, cc] * 255.0).astype(np.uint8))
        cov_off.append(cov_off[-1] + len(c))

        occ = cov.sum(0) > 0
        fr, fc = np.nonzero(occ)
        flo_idx.append((fr * W + fc).astype(np.int32))
        flo_vec.append((out["flow"][:, fr, fc].T / FLOW_SCALE).astype(np.float16))
        flo_off.append(flo_off[-1] + len(fr))

        # Per-group vehicle count, taken from the vehicle list rather than the
        # raster: exact, and unaffected by footprints clipped at the grid edge.
        inside = ((v["xy"][:, 0] >= grid.origin_x) &
                  (v["xy"][:, 0] < grid.origin_x + W * grid.cell) &
                  (v["xy"][:, 1] >= grid.origin_y) &
                  (v["xy"][:, 1] < grid.origin_y + H * grid.cell))
        counts.append(np.bincount(v["group"][inside], minlength=N_GROUPS)[:N_GROUPS]
                      .astype(np.float32))

        n_veh.append(len(v["index"]))
        n_moving.append(int((speed > STOPPED_SPEED).sum()))
        max_speed.append(float(speed.max()) if len(speed) else 0.0)
        n_speeding.append(int((speed > SPEEDING_SPEED).sum()))
        behaviour_present.append(np.bincount(v["behaviour"], minlength=8)[:8])
        times.append(float(z["frame_time"][f]))

        # A collision recorded at source frame `f` is attributed to the kept
        # frame that covers it.
        window = (coll_frame >= f) & (coll_frame < f + stride)
        has_coll.append(bool(window.any()))
        if window.any():
            k = np.argmax(window)
            ai, bi = coll_a[k], coll_b[k]
            pos = []
            for actor in (ai, bi):
                hit = np.nonzero(v["index"] == actor)[0]
                if len(hit):
                    pos.append(v["xy"][hit[0]])
            coll_xy.append(np.mean(pos, axis=0) if pos else [np.nan, np.nan])
        else:
            coll_xy.append([np.nan, np.nan])

    out_meta = dict(meta)
    out_meta.update({
        "source": os.path.basename(src),
        "grid": grid.as_dict(),
        "frame_dt_s": meta["frame_dt_s"] * stride,
        "tensor_stride": stride,
        "n_frames": len(keep),
        "groups": GROUP_NAMES,
        "flow_scale_m_s": FLOW_SCALE,
        "coverage_quantisation": "uint8, value/255",
    })
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    np.savez_compressed(
        dst,
        meta=json.dumps(out_meta),
        frame_time=np.asarray(times, np.float32),
        coverage_offset=np.asarray(cov_off, np.int64),
        coverage_channel=np.concatenate(cov_chan) if cov_chan else np.zeros(0, np.uint8),
        coverage_index=np.concatenate(cov_idx) if cov_idx else np.zeros(0, np.int32),
        coverage_value=np.concatenate(cov_val) if cov_val else np.zeros(0, np.uint8),
        flow_offset=np.asarray(flo_off, np.int64),
        flow_index=np.concatenate(flo_idx) if flo_idx else np.zeros(0, np.int32),
        flow_vector=np.concatenate(flo_vec) if flo_vec else np.zeros((0, 2), np.float16),
        vehicle_count=np.asarray(counts, np.float32),
        n_vehicles=np.asarray(n_veh, np.int32),
        n_moving=np.asarray(n_moving, np.int32),
        max_speed=np.asarray(max_speed, np.float32),
        n_speeding=np.asarray(n_speeding, np.int32),
        behaviour_present=np.asarray(behaviour_present, np.int32),
        has_collision=np.asarray(has_coll, np.bool_),
        collision_xy=np.asarray(coll_xy, np.float32),
    )
    return out_meta


def densify(z, f, grid: Grid):
    """Rebuild the dense (coverage, flow) tensors for one stored frame."""
    H, W = grid.rows, grid.cols
    lo, hi = z["coverage_offset"][f], z["coverage_offset"][f + 1]
    cov = np.zeros((N_GROUPS, H * W), np.float32)
    cov[z["coverage_channel"][lo:hi], z["coverage_index"][lo:hi]] = \
        z["coverage_value"][lo:hi].astype(np.float32) / 255.0
    lo, hi = z["flow_offset"][f], z["flow_offset"][f + 1]
    flow = np.zeros((2, H * W), np.float32)
    flow[:, z["flow_index"][lo:hi]] = z["flow_vector"][lo:hi].astype(np.float32).T
    return cov.reshape(N_GROUPS, H, W), flow.reshape(2, H, W)
