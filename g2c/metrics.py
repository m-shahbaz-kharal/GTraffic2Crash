"""Scoring a forecast, per horizon step and per kind of situation.

Two choices here are deliberate.

Nothing is averaged over the horizon.  A single number hides the thing worth
knowing -- how fast quality decays with lead time -- and every model in this
literature decays.  Reporting per step makes the decay curve the headline.

BIAS is reported alongside every overlap score.  A deterministic forecaster can
win a thresholded overlap metric by hedging: predicting less area than really
exists, so that misses and false alarms trade favourably.  PreDiff measured
exactly this on a strong baseline -- 0.24, meaning it predicted a quarter of the
area it should have.  Overlap without BIAS cannot tell a good forecast from a
timid one.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

THRESHOLDS = (0.05, 0.10, 0.20, 0.40)
POOLS = (1, 2, 4)


class Accumulator:
    """Running sums for every metric, kept per future timestep."""

    def __init__(self, t_out, thresholds=THRESHOLDS, pools=POOLS):
        self.t_out, self.thresholds, self.pools = t_out, thresholds, pools
        self.s = defaultdict(lambda: np.zeros(t_out, dtype=np.float64))
        self.n = 0

    @torch.no_grad()
    def update(self, prob, target, valid, flow_pred=None, flow_true=None,
               cell_area=16.0, mean_vehicle_area=None):
        """prob/target: (B, T, C, H, W).  valid: (B, H, W) or (H, W)."""
        B, T, C, H, W = prob.shape
        v = valid.reshape(-1, 1, 1, H, W) if valid.dim() == 3 else valid.view(1, 1, 1, H, W)
        p = (prob * v).float()
        y = (target * v).float()
        po, yo = p.sum(2).clamp(0, 1), y.sum(2).clamp(0, 1)   # any-class occupancy

        add = lambda k, t: self.s.__setitem__(k, self.s[k] + t.double().cpu().numpy())
        dims = (0, 2, 3)                                       # over batch and space

        # Threshold-free: soft intersection over union, and mass bias.
        add("soft_intersection", torch.minimum(po, yo).sum(dims))
        add("soft_union", torch.maximum(po, yo).sum(dims))
        add("predicted_mass", po.sum(dims))
        add("true_mass", yo.sum(dims))
        add("absolute_error", (po - yo).abs().sum(dims))

        for tau in self.thresholds:
            pb, yb = (po > tau).float(), (yo > tau).float()
            add(f"true_positive@{tau}", (pb * yb).sum(dims))
            add(f"false_positive@{tau}", (pb * (1 - yb)).sum(dims))
            add(f"false_negative@{tau}", ((1 - pb) * yb).sum(dims))
            for k in self.pools:
                if k == 1:
                    continue
                pp = torch.nn.functional.max_pool2d(pb.reshape(B * T, 1, H, W), k)
                yp = torch.nn.functional.max_pool2d(yb.reshape(B * T, 1, H, W), k)
                pp = pp.reshape(B, T, -1); yp = yp.reshape(B, T, -1)
                add(f"true_positive@{tau}_pool{k}", (pp * yp).sum((0, 2)))
                add(f"false_positive@{tau}_pool{k}", (pp * (1 - yp)).sum((0, 2)))
                add(f"false_negative@{tau}_pool{k}", ((1 - pp) * yp).sum((0, 2)))

        if mean_vehicle_area is not None:
            a = torch.as_tensor(mean_vehicle_area, device=p.device).view(1, 1, C)
            pn = p.sum((-1, -2)) * cell_area / a
            yn = y.sum((-1, -2)) * cell_area / a
            add("count_absolute_error", (pn - yn).abs().sum((0, 2)))
            add("count_true", yn.sum((0, 2)))

        if flow_pred is not None and flow_true is not None:
            m = (y.sum(2, keepdim=True) > 1e-3).float()
            epe = ((flow_pred - flow_true) ** 2).sum(2, keepdim=True).sqrt() * m
            add("flow_endpoint_error", epe.sum(dims))
            add("flow_cells", m.sum(dims))
        self.n += B

    def compute(self, flow_scale=1.0):
        s, out = self.s, {}
        eps = 1e-9
        out["soft_iou"] = s["soft_intersection"] / (s["soft_union"] + eps)
        out["bias"] = s["predicted_mass"] / (s["true_mass"] + eps)
        out["mean_absolute_error"] = s["absolute_error"] / max(self.n, 1)
        for tau in self.thresholds:
            for k in self.pools:
                suf = "" if k == 1 else f"_pool{k}"
                tp, fp, fn = (s[f"true_positive@{tau}{suf}"],
                              s[f"false_positive@{tau}{suf}"],
                              s[f"false_negative@{tau}{suf}"])
                out[f"f1@{tau}{suf}"] = 2 * tp / (2 * tp + fp + fn + eps)
                # Critical success index, the nowcasting convention.
                out[f"critical_success_index@{tau}{suf}"] = tp / (tp + fp + fn + eps)
        if "count_true" in s:
            out["vehicle_count_absolute_error"] = s["count_absolute_error"] / max(self.n, 1)
            out["vehicle_count_relative_error"] = \
                s["count_absolute_error"] / (s["count_true"] + eps)
        if "flow_cells" in s:
            out["flow_endpoint_error_m_s"] = \
                s["flow_endpoint_error"] / (s["flow_cells"] + eps) * flow_scale
        return out

    def headline(self, flow_scale=1.0):
        """The few numbers worth putting in a log line."""
        m = self.compute(flow_scale)
        pick = lambda k, i: float(m[k][i]) if k in m else float("nan")
        last = self.t_out - 1
        return {
            "soft_iou_mean": float(np.mean(m["soft_iou"])),
            "soft_iou_first_step": pick("soft_iou", 0),
            "soft_iou_last_step": pick("soft_iou", last),
            "f1@0.1_mean": float(np.mean(m["f1@0.1"])),
            "f1@0.1_last_step": pick("f1@0.1", last),
            "bias_mean": float(np.mean(m["bias"])),
            "bias_last_step": pick("bias", last),
            "vehicle_count_relative_error_mean":
                float(np.mean(m.get("vehicle_count_relative_error", [np.nan]))),
            "flow_endpoint_error_m_s_mean":
                float(np.mean(m.get("flow_endpoint_error_m_s", [np.nan]))),
        }


def table(named_metrics, key="soft_iou", dt=0.5):
    """A readable per-horizon comparison of several models."""
    lines = []
    t = next(iter(named_metrics.values()))[key]
    head = "  ".join(f"{(i+1)*dt:>5.1f}s" for i in range(len(t)))
    lines.append(f"{'':<34}{head}")
    for name, m in named_metrics.items():
        lines.append(f"{name:<34}" + "  ".join(f"{v:6.3f}" for v in m[key]))
    return "\n".join(lines)
