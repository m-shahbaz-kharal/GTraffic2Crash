"""Windows of past and future occupancy, and the static map they sit on.

The whole tensorised dataset is small enough to hold in memory in sparse form
(about 6 KB per frame, so a few gigabytes for hundreds of episodes), so episodes
are loaded once and windows are densified on demand.  That avoids both a large
dense store on disk and per-window file I/O.

Windows carry labels for what happens in them -- a collision, a speeding
vehicle, a red-light runner -- so that sampling can be balanced and, more
importantly, so that forecast quality can be reported per scenario type.  An
aggregate score dominated by quiet road would hide exactly the failure that
matters.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .grid import Grid
from .tensorize import GROUP_NAMES, N_GROUPS

STATIC_CHANNELS = ["road", "lane_sin", "lane_cos", "lane_coherence",
                   "junction", "speed_limit_m_s"]


class StaticMap:
    """Constant map channels and the validity mask, on the model's grid."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.grid = Grid.from_dict(self.meta["grid"])
        ch = [z[k].astype(np.float32) for k in STATIC_CHANNELS]
        # Speed limit is in m/s; scale it to the same range as the rest.
        ch[-1] = ch[-1] / 20.0
        self.channels = torch.from_numpy(np.stack(ch))
        self.valid = torch.from_numpy(z["valid"].astype(np.float32))
        self.names = list(STATIC_CHANNELS)

    def __repr__(self):
        return (f"StaticMap({self.grid.rows}x{self.grid.cols}, "
                f"{len(self.names)} channels, "
                f"{100*float(self.valid.mean()):.0f}% valid)")


@dataclass
class Window:
    episode: int
    start: int
    has_collision: bool
    collision_speed: float
    collision_step: int          # index within the future window, -1 if none
    collision_xy: tuple
    n_speeding: int
    n_bad_actors: int
    n_vehicles: int


class EpisodeStore:
    """All tensorised episodes, held sparsely in memory."""

    def __init__(self, root, grid: Grid, episodes=None):
        self.grid = grid
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if episodes is not None:
            keep = set(episodes)
            self.files = [f for f in self.files
                          if int(os.path.basename(f).split("_")[-1].split(".")[0]) in keep]
        self.episodes, self.meta = [], []
        for f in self.files:
            z = np.load(f, allow_pickle=False)
            self.episodes.append({k: z[k] for k in z.files if k != "meta"})
            self.meta.append(json.loads(str(z["meta"])))
        if not self.episodes:
            raise FileNotFoundError(f"no tensorised episodes under {root}")
        self.frame_dt_s = self.meta[0]["frame_dt_s"]

    def n_frames(self, e):
        return len(self.episodes[e]["frame_time"])

    def densify(self, e, f):
        """Dense (coverage, flow) for one frame of one episode."""
        H, W = self.grid.rows, self.grid.cols
        z = self.episodes[e]
        lo, hi = z["coverage_offset"][f], z["coverage_offset"][f + 1]
        cov = np.zeros((N_GROUPS, H * W), np.float32)
        cov[z["coverage_channel"][lo:hi], z["coverage_index"][lo:hi]] = \
            z["coverage_value"][lo:hi].astype(np.float32) * (1.0 / 255.0)
        lo, hi = z["flow_offset"][f], z["flow_offset"][f + 1]
        flow = np.zeros((2, H * W), np.float32)
        flow[:, z["flow_index"][lo:hi]] = z["flow_vector"][lo:hi].astype(np.float32).T
        return cov.reshape(N_GROUPS, H, W), flow.reshape(2, H, W)

    def mean_vehicle_area(self):
        """Mean footprint area per class group, needed to convert coverage to a
        vehicle count.  Averaged over every frame of every episode."""
        # Coverage integrates to footprint area; the count is that area divided
        # by the mean area of one vehicle of the group.
        areas, cnt = np.zeros(N_GROUPS), np.zeros(N_GROUPS)
        cell_area = self.grid.cell ** 2
        for e, z in enumerate(self.episodes):
            n = self.n_frames(e)
            for f in range(0, n, max(1, n // 20)):
                cov, _ = self.densify(e, f)
                areas += cov.sum(axis=(1, 2)) * cell_area
                cnt += z["vehicle_count"][f]
        return np.where(cnt > 0, areas / np.maximum(cnt, 1e-6), 10.0).astype(np.float32)


class OccupancyWindows(Dataset):
    """Sliding windows of `t_in` observed frames followed by `t_out` future ones."""

    # A contact at walking pace in a stopped queue is not a crash.  The
    # calibration probe put meaningful impacts at above roughly 3 m/s.
    CRASH_SPEED_M_S = 3.0

    def __init__(self, store: EpisodeStore, t_in=10, t_out=10, stride=1,
                 episodes=None):
        self.store, self.t_in, self.t_out = store, t_in, t_out
        self.windows = []
        span = t_in + t_out
        for e in range(len(store.episodes)):
            if episodes is not None and store.meta[e]["episode"] not in episodes:
                continue
            z = store.episodes[e]
            n = store.n_frames(e)
            beh = z["behaviour_present"]
            for s in range(0, n - span + 1, stride):
                fut = slice(s + t_in, s + span)
                cs = (z["collision_speed"][fut] if "collision_speed" in z
                      else np.zeros(t_out, np.float32))
                hit = z["has_collision"][fut]
                k = int(np.argmax(np.where(hit, cs, -1.0))) if hit.any() else -1
                xy = (tuple(z["collision_xy"][s + t_in + k]) if k >= 0
                      else (float("nan"), float("nan")))
                self.windows.append(Window(
                    episode=e, start=s,
                    has_collision=bool(hit.any()),
                    collision_speed=float(cs.max()),
                    collision_step=k, collision_xy=xy,
                    n_speeding=int(z["n_speeding"][fut].max()),
                    n_bad_actors=int(beh[s + t_in, 1:].sum()) if beh.shape[1] > 1 else 0,
                    n_vehicles=int(z["n_vehicles"][s + t_in - 1]),
                ))

    def __len__(self):
        return len(self.windows)

    def scenario(self, i):
        """Coarse label used to report metrics per kind of situation.

        Deliberately based on behaviour tags and recorded collisions rather than
        on inferred speed.  The grid covers the whole town, so "is someone
        speeding somewhere" is true almost always in an incident episode and
        discriminates nothing; whether a collision actually happens in the frames
        being predicted is the label that matters.
        """
        w = self.windows[i]
        if w.has_collision and w.collision_speed >= self.CRASH_SPEED_M_S:
            return "collision"
        if w.has_collision:
            return "low_speed_contact"
        if w.n_bad_actors > 0:
            return "incident_traffic"
        return "nominal"

    def __getitem__(self, i):
        w = self.windows[i]
        past_c, fut_c, fut_f = [], [], []
        last_flow = None
        for k in range(self.t_in):
            c, fl = self.store.densify(w.episode, w.start + k)
            past_c.append(c)
            last_flow = fl                      # ground-truth flow at t = 0
        for k in range(self.t_in, self.t_in + self.t_out):
            c, fl = self.store.densify(w.episode, w.start + k)
            fut_c.append(c); fut_f.append(fl)
        past = np.stack(past_c); fut = np.stack(fut_c); flow = np.stack(fut_f)
        last_flow = np.asarray(last_flow)

        return {
            "past_occupancy": torch.from_numpy(past),
            "future_occupancy": torch.from_numpy(fut),
            "future_flow": torch.from_numpy(flow),
            "last_observed_flow": torch.from_numpy(last_flow),
            "episode": torch.tensor(w.episode),
            "start": torch.tensor(w.start),
            "has_collision": torch.tensor(w.has_collision),
            "collision_speed": torch.tensor(w.collision_speed),
            "collision_step": torch.tensor(w.collision_step),
            "collision_xy": torch.tensor(w.collision_xy, dtype=torch.float32),
            "n_speeding": torch.tensor(w.n_speeding),
            "n_bad_actors": torch.tensor(w.n_bad_actors),
        }


def split_episodes(meta_list, val_fraction=0.15, seed=0):
    """Split by episode, never by window.

    Windows from one episode overlap heavily and share the same vehicles; mixing
    them across the split would leak the answer and report a validation score
    that means nothing.
    """
    ids = sorted({m["episode"] for m in meta_list})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_val = max(1, int(round(val_fraction * len(ids))))
    val = {ids[i] for i in order[:n_val]}
    train = {ids[i] for i in order[n_val:]}
    return train, val
