#!/usr/bin/env python
"""Train the occupancy forecaster."""
import argparse
import json
import os
import sys

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c.losses import LossWeights  # noqa: E402
from g2c.model import ModelConfig, OccupancyForecaster  # noqa: E402
from g2c.train import Forecaster, G2CDataModule  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/lustre/fs1/home/mu290328/datasets/GTraffic2Crash")
    p.add_argument("--run", default="dev")
    p.add_argument("--output-root", default="/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/runs")
    p.add_argument("--size", default="base", choices=["small", "base", "large"])
    p.add_argument("--t-in", type=int, default=10)
    p.add_argument("--t-out", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--val-stride", type=int, default=5)
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--limit-train-batches", type=float, default=1.0)
    p.add_argument("--limit-val-batches", type=float, default=200,
               help="validation batches per epoch; ~1600 windows is plenty for "
                    "stable metrics and keeps every epoch's validation quick")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--no-checkpointing", action="store_true",
                   help="disable activation checkpointing (faster, more memory)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--overfit-batches", type=int, default=0,
                   help="train and validate on this many batches only; the plumbing "
                        "test -- a large model must be able to memorise a handful "
                        "of windows, and if it cannot, no amount of data will help")
    p.add_argument("--wandb-project", default="GTraffic2Crash")
    p.add_argument("--resume", default="")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    L.seed_everything(0, workers=True)
    out = os.path.join(args.output_root, args.run)
    os.makedirs(out, exist_ok=True)

    dm = G2CDataModule(args.data, args.t_in, args.t_out, args.batch_size, args.workers,
                       train_stride=args.train_stride, val_stride=args.val_stride,
                       max_episodes=args.max_episodes or None)
    dm.setup()

    steps_per_epoch = max(1, len(dm.train_set) // (args.batch_size * max(args.devices, 1)
                                                   * args.accumulate))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs

    cfg = OccupancyForecaster.preset(
        args.size, rows=dm.grid.rows, cols=dm.grid.cols,
        t_in=args.t_in, t_out=args.t_out,
        n_static=dm.static.channels.shape[0],
        checkpointing=not args.no_checkpointing)

    model = Forecaster(cfg, LossWeights(), dm.static.channels, dm.static.valid,
                       dm.mean_vehicle_area, dm.frame_dt_s, dm.grid.cell,
                       learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                       warmup_steps=args.warmup_steps, total_steps=total_steps,
                       augment=not args.no_augment, output_dir=out)
    n, parts = model.net.parameter_count()
    print(f"[model] {args.size}: {n/1e6:.1f} M parameters "
          f"({ {k: round(v/1e6,1) for k,v in parts.items()} })")
    print(f"[train] {steps_per_epoch:,} steps/epoch, {total_steps:,} total")

    # Weights & Biases runs offline here; `scripts/sync_wandb.sh` uploads later.
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", args.output_root)
    from lightning.pytorch.loggers import WandbLogger
    logger = WandbLogger(project=args.wandb_project, name=args.run, save_dir=args.output_root,
                         config={**vars(args), "model": vars(cfg),
                                 "parameters": n, "steps_per_epoch": steps_per_epoch})

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # Two checkpoints: the best by validation quality, and the latest, so a
        # preempted or timed-out job resumes without losing an epoch.
        # The monitored metric name contains a slash, which is a hazard inside a
        # checkpoint filename, so the filename carries only epoch and step.
        ModelCheckpoint(dirpath=os.path.join(out, "checkpoints"),
                        filename="best-epoch{epoch:03d}-step{step:07d}",
                        monitor="validation/soft_iou_mean", mode="max", save_top_k=2,
                        save_last=True, auto_insert_metric_name=False),
    ]
    trainer = L.Trainer(
        max_epochs=args.epochs, max_steps=args.max_steps if args.max_steps > 0 else -1,
        accelerator="gpu", devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        precision=args.precision, logger=logger, callbacks=callbacks,
        accumulate_grad_batches=args.accumulate, gradient_clip_val=1.0,
        log_every_n_steps=25, default_root_dir=out,
        limit_train_batches=args.overfit_batches or args.limit_train_batches,
        limit_val_batches=args.overfit_batches or args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=0)
    json.dump({**vars(args), "model": vars(cfg), "parameters": n},
              open(os.path.join(out, "config.json"), "w"), indent=2)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume or None)


if __name__ == "__main__":
    main()
