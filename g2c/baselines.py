"""The bar the model has to clear.

Two baselines, and the second is deliberately given an unfair advantage.

  persistence          repeat the last observed frame.  The floor.  On a sparse
                       grid at short horizons it is a surprisingly strong floor,
                       which is exactly why it must be measured rather than
                       assumed away.

  constant velocity    transport the last observed frame along a velocity field,
                       held fixed over the horizon.  Run twice:
                         * with the *ground-truth* velocity field, which no
                           deployed system could have.  This answers "even given
                           perfect knowledge of every vehicle's current motion,
                           how far does pure extrapolation get?"  Everything the
                           model gains over this is knowledge of roads, signals
                           and interaction -- which is the whole reason to train it.
                         * with velocity estimated from the last two observed
                           grids by dense optical flow, which is the deployable
                           version and the fairer head-to-head.

Both use the same mass-conserving forward warp as the training objective, so any
difference in score is a difference in prediction and not in resampling.
"""
from __future__ import annotations

import numpy as np
import torch

from .losses import warp_forward


def persistence(past_occupancy, t_out):
    """Repeat the last observed frame."""
    return past_occupancy[:, -1:].expand(-1, t_out, -1, -1, -1).clone()


def constant_velocity(past_occupancy, flow_cells_per_step, t_out):
    """Transport the last observed frame along a fixed velocity field.

    The field is applied cumulatively: after k steps the mass has moved k times,
    which is what "constant velocity" means.  Warping the accumulated field each
    step (rather than the original by k*flow) also lets mass spread, which is the
    honest behaviour of repeated resampling.
    """
    B, _, C, H, W = past_occupancy.shape
    cur = past_occupancy[:, -1]
    out = []
    for _ in range(t_out):
        cur = warp_forward(cur, flow_cells_per_step)
        out.append(cur)
    return torch.stack(out, dim=1).clamp(0, 1)


def estimate_flow(prev_occupancy, curr_occupancy, max_speed_cells=4.0):
    """Dense optical flow between two occupancy grids, in cells per step.

    Farneback on the class-summed occupancy.  Deliberately a standard,
    off-the-shelf estimator rather than anything tuned: the point of this
    baseline is to represent what is obtainable without learning the scene.
    """
    import cv2
    B = prev_occupancy.shape[0]
    out = np.zeros((B, 2, *prev_occupancy.shape[-2:]), np.float32)
    a = (prev_occupancy.sum(1).clamp(0, 1) * 255).byte().cpu().numpy()
    b = (curr_occupancy.sum(1).clamp(0, 1) * 255).byte().cpu().numpy()
    for i in range(B):
        f = cv2.calcOpticalFlowFarneback(a[i], b[i], None, 0.5, 4, 15, 3, 5, 1.2, 0)
        out[i] = np.clip(f.transpose(2, 0, 1), -max_speed_cells, max_speed_cells)
    return torch.from_numpy(out).to(prev_occupancy.device)


def all_baselines(batch, cells_per_frame_step, t_out):
    """Every baseline's forecast for one batch, keyed by name."""
    past = batch["past_occupancy"]
    preds = {"persistence": persistence(past, t_out)}
    if "last_observed_flow" in batch:
        oracle = batch["last_observed_flow"] * cells_per_frame_step
        preds["constant_velocity_oracle_flow"] = constant_velocity(past, oracle, t_out)
    est = estimate_flow(past[:, -2], past[:, -1])
    preds["constant_velocity_estimated_flow"] = constant_velocity(past, est, t_out)
    return preds
