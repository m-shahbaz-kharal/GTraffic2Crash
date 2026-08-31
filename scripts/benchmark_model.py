#!/usr/bin/env python
"""Measure the model's real step time, because the FLOP estimate is only an estimate.

The whole training plan rests on how many samples per second one H100 delivers.
That number decides the model size, the batch size and how much data is worth
generating, so it is measured rather than assumed.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c.losses import LossWeights, forecasting_loss  # noqa: E402
from g2c.model import ModelConfig, OccupancyForecaster  # noqa: E402


def bench(size, batch, steps=12, checkpointing=True, rows=144, cols=160, t=10):
    dev = torch.device("cuda")
    cfg = OccupancyForecaster.preset(size, rows=rows, cols=cols, t_in=t, t_out=t,
                                     checkpointing=checkpointing)
    net = OccupancyForecaster(cfg).to(dev)
    n, _ = net.parameter_count()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    past = torch.rand(batch, t, 3, rows, cols, device=dev) * 0.05
    static = torch.rand(batch, cfg.n_static, rows, cols, device=dev)
    fut = (torch.rand(batch, t, 3, rows, cols, device=dev) < 0.02).float()
    flow = torch.randn(batch, t, 2, rows, cols, device=dev) * 0.1
    b = {"future_occupancy": fut, "past_occupancy": past, "future_flow": flow,
         "valid": torch.ones(batch, rows, cols, device=dev),
         "cell_area_m2": torch.tensor(16.0, device=dev),
         "mean_vehicle_area_m2": torch.full((batch, 3), 10.0, device=dev)}
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(past, static)
            loss, _ = forecasting_loss(pred, b, LossWeights(), 1.0)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 4:
            times.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    dt = sum(times) / len(times)
    del net, opt
    torch.cuda.empty_cache()
    return {"size": size, "parameters_millions": n / 1e6, "batch": batch,
            "checkpointing": checkpointing, "seconds_per_step": round(dt, 3),
            "samples_per_second": round(batch / dt, 2),
            "peak_memory_gib": round(peak, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=375_000)
    args = ap.parse_args()
    print(torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")
    rows = []
    for size, batch, ckpt in [("small", 16, False), ("base", 4, True), ("base", 8, True),
                              ("base", 8, False), ("base", 16, True), ("large", 8, True)]:
        try:
            r = bench(size, batch, checkpointing=ckpt)
            rows.append(r)
            hours = args.windows / r["samples_per_second"] / 3600
            print(f"{r['size']:>6} batch {r['batch']:>2} "
                  f"checkpointing={str(r['checkpointing']):>5}: "
                  f"{r['parameters_millions']:6.1f} M  {r['seconds_per_step']:6.3f} s/step  "
                  f"{r['samples_per_second']:7.2f} samples/s  "
                  f"{r['peak_memory_gib']:5.1f} GiB  "
                  f"-> {hours:5.2f} h/epoch on 1 GPU, {hours/2:5.2f} h on 2")
        except RuntimeError as e:
            print(f"{size} batch {batch} checkpointing={ckpt}: {str(e)[:90]}")
            torch.cuda.empty_cache()
