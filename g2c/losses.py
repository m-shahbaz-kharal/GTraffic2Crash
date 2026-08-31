"""The training objective.

Most of the conventional choices in the occupancy literature are wrong for this
target, and the reasons are specific enough to be worth stating in the code
rather than only in a paper.

The target is *soft* -- a cell holds the fraction of its area covered by a
vehicle, so a value of 0.4 is not a weak positive but an exact measurement.

  * Soft Dice / soft IoU / Tversky are provably not minimised at the truth for
    fractional targets: with y = 0.5, soft Dice is minimised at x = 1 (Wang et
    al., MICCAI 2023, arXiv 2303.16296).  They would inflate every partially
    covered cell.  Separately they carry a volume bias in exactly the
    high-uncertainty regime forecasting lives in (Bertels et al., MedIA 2021),
    and being ratios over the whole image their gradient scale swings with the
    number of vehicles in the frame.
  * Lovasz-softmax is the convex closure of a set function over *misclassified
    pixel sets*, which is undefined when the ground truth is 0.4.
  * Focal loss is not strictly proper for any gamma > 0 (Charoenphakdee et al.,
    arXiv 2011.09172); its minimiser is not the true posterior and the bias
    direction is under-confidence, which here means quietly losing vehicles.

So the core term is a plain soft cross-entropy, which *is* minimised at the
truth and is defined for fractional targets without modification.  The usual
objection -- that 98 % empty cells swamp the gradient -- is handled where it
actually bites, at initialisation, by setting the output bias to the prior (see
`model.py`); after that, an empty cell the network already calls empty has
gradient `p - y` which is near zero.

Three further terms carry the parts a per-cell loss cannot see:
  flow       what the motion is, not just where the mass is
  trace      that mass *moves* rather than teleporting or evaporating
  count      that the number of vehicles is right, correcting the drift that
             per-cell reweighting necessarily introduces
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    occupancy: float = 1.0
    flow: float = 0.5
    trace: float = 0.5
    count: float = 0.5
    # Cells at or beside a vehicle count this many times a cell of open ground.
    # A fixed, stated reweighting of the sampling distribution, not a distortion
    # of the link function, so the bias it introduces is known rather than
    # emergent.  DGMR and SimCast both show intensity weighting of this kind buys
    # more skill than far more expensive machinery.
    positive_weight: float = 9.0
    dilation: int = 3


def occupancy_weight(target, positive_weight: float, dilation: int, valid):
    """Per-cell weight: high at and around anything occupied, 1 elsewhere."""
    occ = target.sum(dim=2, keepdim=True)
    B, T = occ.shape[:2]
    near = F.max_pool2d(occ.reshape(B * T, 1, *occ.shape[-2:]),
                        kernel_size=dilation, stride=1, padding=dilation // 2)
    near = (near > 1e-3).float().reshape(B, T, 1, *occ.shape[-2:])
    return (1.0 + positive_weight * near) * valid


def soft_bce(logits, target, weight):
    """Cross-entropy for a target in [0, 1], minimised exactly at p = target."""
    # `weight` broadcasts over the class axis, so the ratio is already the mean
    # over cells of the summed-over-classes cross-entropy.
    per_cell = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per_cell * weight).sum() / weight.sum().clamp_min(1.0)


def warp_forward(field, flow):
    """Move `field` one step along `flow`, conserving total mass.

    Bilinear *scatter* rather than the usual `grid_sample` gather: a gather asks
    "where did this cell come from", which needs backward flow, while the model
    predicts where mass at a cell is going.  Scattering the mass forward keeps
    the sum exactly, which is the property the trace term is there to enforce.
    """
    B, C, H, W = field.shape
    dy, dx = flow[:, 1], flow[:, 0]
    yy, xx = torch.meshgrid(torch.arange(H, device=field.device, dtype=field.dtype),
                            torch.arange(W, device=field.device, dtype=field.dtype),
                            indexing="ij")
    ty = (yy[None] + dy).clamp(0, H - 1)
    tx = (xx[None] + dx).clamp(0, W - 1)
    y0, x0 = ty.floor().long(), tx.floor().long()
    y1, x1 = (y0 + 1).clamp(max=H - 1), (x0 + 1).clamp(max=W - 1)
    fy, fx = ty - y0.to(ty.dtype), tx - x0.to(tx.dtype)

    out = torch.zeros_like(field)
    for (yi, xi, w) in ((y0, x0, (1 - fy) * (1 - fx)), (y0, x1, (1 - fy) * fx),
                        (y1, x0, fy * (1 - fx)), (y1, x1, fy * fx)):
        idx = (yi * W + xi).unsqueeze(1).expand(B, C, H, W).reshape(B, C, -1)
        out = out.reshape(B, C, -1).scatter_add(
            2, idx, (field * w.unsqueeze(1)).reshape(B, C, -1)).reshape(B, C, H, W)
    return out


def forecasting_loss(pred, batch, w: LossWeights, cells_per_frame_step: float):
    """Full objective.  Returns (total, per-term dict) with every term logged."""
    logits = pred["occupancy_logits"]                 # B,T,C,H,W
    target = batch["future_occupancy"]
    valid = batch["valid"][:, None, None]             # B,1,1,H,W
    weight = occupancy_weight(target, w.positive_weight, w.dilation, valid)

    terms = {}
    terms["occupancy_cross_entropy"] = soft_bce(logits, target, weight)
    prob = torch.sigmoid(logits)

    if pred.get("flow") is not None and "future_flow" in batch:
        occ = target.sum(dim=2, keepdim=True)
        m = (occ > 1e-3).float() * valid
        terms["flow_error"] = ((pred["flow"] - batch["future_flow"]).abs() * m).sum() / \
            m.sum().clamp_min(1.0) / 2.0

        # Flow-traced consistency: transporting the occupancy at t along the
        # predicted flow must land on the occupancy at t+1.  This is the term
        # that carries the physics -- mass moves, it does not teleport -- and it
        # is the answer to blur that does not require an adversary.
        B, T, C, H, W = prob.shape
        src = torch.cat([batch["past_occupancy"][:, -1:], prob[:, :-1]], dim=1)
        fl = pred["flow"][:, :-1] if T > 1 else pred["flow"]
        fl = torch.cat([pred["flow"][:, :1], fl], dim=1)[:, :T]
        moved = warp_forward(src.reshape(B * T, C, H, W),
                             (fl * cells_per_frame_step).reshape(B * T, 2, H, W))
        moved = moved.reshape(B, T, C, H, W).clamp(1e-6, 1 - 1e-6)
        terms["flow_trace"] = (F.binary_cross_entropy(moved, target, reduction="none")
                               * weight).sum() / weight.sum().clamp_min(1.0)

    # Count consistency, per frame and per class, in units of vehicles.  Weighted
    # cross-entropy is no longer proper, so it drifts; this measures the drift in
    # the one summary that is physically meaningful and corrects it.
    area = batch["cell_area_m2"]
    veh_area = batch["mean_vehicle_area_m2"][:, None, :]
    pred_n = (prob * valid).sum(dim=(-1, -2)) * area / veh_area
    true_n = (target * valid).sum(dim=(-1, -2)) * area / veh_area
    # Relative, so the term is dimensionless and sits on the same scale as the
    # cross-entropy.  In absolute vehicles it would be ~100x larger than every
    # other term and would dominate the gradient outright.  The +1 keeps it
    # finite for a class with no vehicles in the frame.
    terms["vehicle_count_relative_error"] = \
        ((pred_n - true_n).abs() / (true_n + 1.0)).mean()
    terms["vehicle_count_absolute_error"] = (pred_n - true_n).abs().mean()

    total = (w.occupancy * terms["occupancy_cross_entropy"]
             + w.flow * terms.get("flow_error", 0.0)
             + w.trace * terms.get("flow_trace", 0.0)
             + w.count * terms["vehicle_count_relative_error"])
    terms = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in terms.items()}
    terms["total"] = total.detach()
    return total, terms
