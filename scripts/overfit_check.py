#!/usr/bin/env python
"""Can the network fit a single batch it sees over and over?

This is the plumbing test, and it has to be measured on the *same* batch it is
trained on -- validating on held-out data instead answers a different question
and would look like failure even when everything works.

If the loss does not fall and soft-IoU on that one batch does not approach 1,
the fault is in the model, the loss or the data path, and no amount of data or
training time will fix it.  If it does, the pipeline is sound and any later
failure to generalise is a real result rather than a bug.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c.losses import LossWeights, forecasting_loss  # noqa: E402
from g2c.metrics import Accumulator  # noqa: E402
from g2c.model import OccupancyForecaster  # noqa: E402
from g2c.train import G2CDataModule  # noqa: E402
from g2c.tensorize import FLOW_SCALE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/lustre/fs1/home/mu290328/datasets/GTraffic2Crash")
    ap.add_argument("--size", default="small")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--checkpointing", action="store_true",
                    help="needed for the base model: without it a batch of 4 does "
                         "not fit in 80 GiB")
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    dm = G2CDataModule(args.data, 10, 10, args.batch_size, 0, val_fraction=0.2)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    batch = {k: v.to(dev) for k, v in batch.items()}
    static = dm.static.channels.to(dev)
    valid = dm.static.valid.to(dev)
    B = batch["past_occupancy"].shape[0]

    cfg = OccupancyForecaster.preset(args.size, rows=dm.grid.rows, cols=dm.grid.cols,
                                     n_static=static.shape[0],
                                     checkpointing=args.checkpointing)
    net = OccupancyForecaster(cfg).to(dev)
    n, _ = net.parameter_count()
    opt = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.0)
    cells = FLOW_SCALE * dm.frame_dt_s / dm.grid.cell
    mva = torch.as_tensor(dm.mean_vehicle_area, device=dev)
    lb = dict(batch)
    lb["valid"] = valid.unsqueeze(0).expand(B, -1, -1)
    lb["cell_area_m2"] = torch.tensor(dm.grid.cell ** 2, device=dev)
    lb["mean_vehicle_area_m2"] = mva.unsqueeze(0).expand(B, -1)

    def score(prob):
        a = Accumulator(cfg.t_out)
        a.update(prob, batch["future_occupancy"], valid=valid,
                 cell_area=dm.grid.cell ** 2, mean_vehicle_area=mva)
        h = a.headline()
        return h["soft_iou_mean"], h["bias_mean"]

    # Reference points on this exact batch.
    with torch.no_grad():
        pers = batch["past_occupancy"][:, -1:].expand(-1, cfg.t_out, -1, -1, -1)
        p_iou, _ = score(pers)
    print(f"[overfit] {args.size}: {n/1e6:.1f} M parameters, batch {B}, "
          f"{args.steps} steps at lr {args.learning_rate}")
    print(f"[overfit] persistence on this batch: soft_iou {p_iou:.4f}")
    print(f"{'step':>6} {'total':>8} {'occ_ce':>8} {'flow':>8} {'trace':>8} "
          f"{'count':>8} {'soft_iou':>9} {'bias':>7} {'sec':>6}")

    t0 = time.time()
    net.train()
    for i in range(args.steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(batch["past_occupancy"], static.unsqueeze(0).expand(B, -1, -1, -1))
            loss, terms = forecasting_loss(pred, lb, LossWeights(), cells)
        if i % max(1, args.steps // 20) == 0:
            with torch.no_grad():
                iou, bias = score(torch.sigmoid(pred["occupancy_logits"]).float())
            print(f"{i:>6} {float(terms['total']):>8.4f} "
                  f"{float(terms['occupancy_cross_entropy']):>8.4f} "
                  f"{float(terms.get('flow_error', 0)):>8.4f} "
                  f"{float(terms.get('flow_trace', 0)):>8.4f} "
                  f"{float(terms['vehicle_count_relative_error']):>8.4f} "
                  f"{iou:>9.4f} {bias:>7.3f} {time.time()-t0:>6.0f}", flush=True)
        if i < args.steps:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)

    print(f"\n[overfit] persistence {p_iou:.4f} -> model {iou:.4f} on the same batch")
    print("[overfit] " + ("PASS: the network can fit what it is shown"
                          if iou > max(0.9, p_iou + 0.25) else
                          "FAIL: the network cannot fit a single batch"))


if __name__ == "__main__":
    main()
