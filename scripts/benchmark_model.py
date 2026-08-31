#!/usr/bin/env python
"""Measure the model's real step time, because the FLOP estimate is only an estimate.

The whole training plan rests on how many samples per second one H100 delivers.
That number decides the model size, the batch size and how much data is worth
generating, so it is measured rather than assumed.
"""
import argparse
import gc
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
    print(f"  [{size}/{batch}] building...", flush=True)
    net = OccupancyForecaster(cfg).to(dev)
    n, _ = net.parameter_count()
    print(f"  [{size}/{batch}] {n/1e6:.0f} M params on device", flush=True)
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
        if i < 3:
            print(f"  [{size}/{batch}] step {i}", flush=True)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(past, static)
            if i == 0:
                torch.cuda.synchronize(); print("    forward ok", flush=True)
            loss, _ = forecasting_loss(pred, b, LossWeights(), 1.0)
            if i == 0:
                torch.cuda.synchronize(); print("    loss ok", flush=True)
        loss.backward()
        if i == 0:
            torch.cuda.synchronize(); print("    backward ok", flush=True)
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 4:
            times.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    dt = sum(times) / len(times)
    # Free everything before returning: a Python frame that still references a
    # tensor keeps its whole graph alive, and the next configuration then OOMs
    # for reasons that have nothing to do with its own size.
    del net, opt, past, static, fut, flow, b, pred, loss
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
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
    for size, batch, ckpt in [("small", 8, True), ("base", 2, True), ("base", 4, True),
                              ("base", 6, True), ("base", 8, True), ("large", 4, True)]:
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
            print(f"{size:>6} batch {batch:>2} checkpointing={str(ckpt):>5}: "
                  f"{str(e).splitlines()[0][:80]}")
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
