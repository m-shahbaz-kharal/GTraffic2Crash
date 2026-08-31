#!/usr/bin/env python
"""Why does the occupancy term not move?  Test the candidates side by side.

The occupancy cross-entropy sat at exactly 0.1737 for hundreds of steps while
every other loss term trained.  Rather than guess, this runs the same batch under
several variants and reports which one unfreezes it, plus the gradient actually
reaching the occupancy head.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c.losses import LossWeights, forecasting_loss, occupancy_weight, soft_bce  # noqa: E402
from g2c.model import OccupancyForecaster  # noqa: E402
from g2c.tensorize import FLOW_SCALE  # noqa: E402
from g2c.train import G2CDataModule  # noqa: E402


def run(name, batch, static, valid, dm, steps=120, lr=1e-3, occupancy_only=False,
        residual=True, autocast=True, clip=1.0, normalise_by_positives=False,
        weights=None):
    dev = static.device
    B = batch["past_occupancy"].shape[0]
    torch.manual_seed(0)
    cfg = OccupancyForecaster.preset("small", rows=dm.grid.rows, cols=dm.grid.cols,
                                     n_static=static.shape[0], checkpointing=False,
                                     residual_from_last=residual)
    net = OccupancyForecaster(cfg).to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    cells = FLOW_SCALE * dm.frame_dt_s / dm.grid.cell
    lb = dict(batch)
    lb["valid"] = valid.unsqueeze(0).expand(B, -1, -1)
    lb["cell_area_m2"] = torch.tensor(dm.grid.cell ** 2, device=dev)
    lb["mean_vehicle_area_m2"] = torch.as_tensor(dm.mean_vehicle_area, device=dev) \
        .unsqueeze(0).expand(B, -1)
    st = static.unsqueeze(0).expand(B, -1, -1, -1)
    first, last, gnorm = None, None, 0.0
    for i in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            pred = net(batch["past_occupancy"], st)
            if occupancy_only:
                w = occupancy_weight(lb["future_occupancy"], 9.0, 3,
                                     valid[None, None, None])
                if normalise_by_positives:
                    w = w / w.mean()
                ce = soft_bce(pred["occupancy_logits"], lb["future_occupancy"], w)
                loss, terms = ce, {"occupancy_cross_entropy": ce.detach()}
            else:
                loss, terms = forecasting_loss(pred, lb, weights or LossWeights(), cells)
        v = float(terms["occupancy_cross_entropy"])
        if i == 0:
            first = v
        last = v
        opt.zero_grad(); loss.backward()
        if i == 0:
            g = net.head.occupancy.weight.grad
            gnorm = float(g.abs().mean()) if g is not None else float("nan")
        if clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()
    delta = 100 * (first - last) / max(first, 1e-9)
    print(f"{name:<46} {first:>9.5f} -> {last:>9.5f}   {delta:>7.2f}%   "
          f"grad {gnorm:.2e}", flush=True)
    del net, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/lustre/fs1/home/mu290328/datasets/GTraffic2Crash")
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()
    dev = torch.device("cuda")
    dm = G2CDataModule(args.data, 10, 10, 4, 0, val_fraction=0.2)
    dm.setup()
    batch = {k: v.to(dev) for k, v in next(iter(dm.train_dataloader())).items()}
    static, valid = dm.static.channels.to(dev), dm.static.valid.to(dev)
    print(f"\n{'configuration':<46} {'occ_ce first':>9}    {'last':>9}   {'change':>7}")
    W = LossWeights
    run("occupancy term alone (the ceiling)", batch, static, valid, dm, args.steps,
        occupancy_only=True)
    run("full loss, separate heads, count 1e-3", batch, static, valid, dm, args.steps)
    run("full loss, separate heads, count 0", batch, static, valid, dm, args.steps,
        weights=W(count=0.0))
    run("full loss at the training learning rate 3e-4", batch, static, valid, dm,
        args.steps, lr=3e-4)


if __name__ == "__main__":
    main()
