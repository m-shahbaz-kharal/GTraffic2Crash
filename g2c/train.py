"""Training: a Lightning module, the data plumbing, and honest logging.

Two things this module treats as first-class rather than as afterthoughts.

**Baselines are evaluated by the same code, on the same batches, at the same
time as the model.**  Persistence and constant-velocity extrapolation are scored
every validation epoch and logged next to the model, so the only number that
ever gets reported is the gap.  A forecast that cannot beat "copy the last
frame" is not a forecast, and finding that out at the end would be a waste.

**Every loss term is logged separately.**  A single total tells you nothing about
which of four terms is driving the gradient or which one has quietly gone to
zero.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import viz
from .baselines import all_baselines
from .data import EpisodeStore, OccupancyWindows, StaticMap, split_episodes
from .losses import LossWeights, forecasting_loss
from .metrics import Accumulator, table
from .model import ModelConfig, OccupancyForecaster
from .tensorize import FLOW_SCALE, GROUP_NAMES, GROUP_RGB

# Lightning stores the model and loss configurations in the checkpoint's
# hyper-parameters, and torch.load has defaulted to weights_only=True since
# PyTorch 2.6 -- which refuses to unpickle them.  Without this, both resuming a
# preempted run and loading a checkpoint for evaluation fail.
torch.serialization.add_safe_globals([ModelConfig, LossWeights])


class G2CDataModule(L.LightningDataModule):
    def __init__(self, data_root, t_in=10, t_out=10, batch_size=8, workers=8,
                 val_fraction=0.15, train_stride=1, val_stride=5, seed=0,
                 max_episodes=None):
        super().__init__()
        self.save_hyperparameters()
        self.static = StaticMap(os.path.join(data_root, "meta", "town05_static.npz"))
        self.grid = self.static.grid

    def setup(self, stage=None):
        h = self.hparams
        self.store = EpisodeStore(os.path.join(h.data_root, "tensor"), self.grid)
        if h.max_episodes:
            self.store.episodes = self.store.episodes[:h.max_episodes]
            self.store.meta = self.store.meta[:h.max_episodes]
        train_ids, val_ids = split_episodes(self.store.meta, h.val_fraction, h.seed)
        self.train_set = OccupancyWindows(self.store, h.t_in, h.t_out,
                                          h.train_stride, episodes=train_ids)
        self.val_set = OccupancyWindows(self.store, h.t_in, h.t_out,
                                        h.val_stride, episodes=val_ids)
        self.mean_vehicle_area = self.store.mean_vehicle_area()
        self.frame_dt_s = self.store.frame_dt_s
        print(f"[data] {len(self.store.episodes)} episodes: "
              f"{len(train_ids)} train / {len(val_ids)} validation (split by episode)")
        print(f"[data] {len(self.train_set):,} training windows, "
              f"{len(self.val_set):,} validation windows")
        from collections import Counter
        print(f"[data] validation scenarios: "
              f"{dict(Counter(self.val_set.scenario(i) for i in range(len(self.val_set))))}")
        # Record the exact split.  It is derived from the list of episodes present
        # on disk, so adding episodes later silently changes it -- and an
        # evaluation run against a different split is not an evaluation.
        self.split = {"train_episodes": sorted(train_ids),
                      "validation_episodes": sorted(val_ids),
                      "n_episodes_on_disk": len(self.store.episodes),
                      "val_fraction": h.val_fraction, "seed": h.seed}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.workers, pin_memory=True,
                          drop_last=True, persistent_workers=self.hparams.workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.workers, pin_memory=True,
                          persistent_workers=self.hparams.workers > 0)


class Forecaster(L.LightningModule):
    def __init__(self, model_config: ModelConfig, loss_weights: LossWeights,
                 static_channels=None, valid_mask=None, mean_vehicle_area=None,
                 frame_dt_s=0.5, cell_size_m=4.0, learning_rate=3e-4,
                 weight_decay=0.05, warmup_steps=500, total_steps=100_000,
                 augment=True, video_every_n_epochs=1, output_dir="runs/dev"):
        super().__init__()
        self.save_hyperparameters(ignore=["static_channels", "valid_mask"])
        self.net = OccupancyForecaster(model_config)
        self.cfg = model_config
        self.weights = loss_weights
        # The map and the validity mask are buffers, so they travel in the
        # state dict and do not need to be supplied when a checkpoint is
        # reloaded -- placeholders of the right shape are enough, and are
        # immediately overwritten by the saved values.
        if static_channels is None:
            static_channels = torch.zeros(model_config.n_static,
                                          model_config.rows, model_config.cols)
        if valid_mask is None:
            valid_mask = torch.ones(model_config.rows, model_config.cols)
        if mean_vehicle_area is None:
            mean_vehicle_area = torch.ones(model_config.n_groups) * 10.0
        self.register_buffer("static", static_channels)
        self.register_buffer("valid", valid_mask)
        self.register_buffer("mean_vehicle_area", torch.as_tensor(mean_vehicle_area))
        self.cell_area = cell_size_m ** 2
        # Flow is stored in units of FLOW_SCALE m/s; the warp needs cells per step.
        self.cells_per_frame_step = FLOW_SCALE * frame_dt_s / cell_size_m
        self.output_dir = output_dir
        self._val_acc = None
        self._baseline_acc = {}
        self._scenario_acc = {}
        self._t0 = time.time()
        self._last_print = time.time()
        self._last_step = 0

    # ------------------------------------------------------------- helpers --
    def _augment(self, batch):
        """Mirror the whole batch, map included, so the world stays consistent.

        Done at batch level rather than per sample because the static map is
        shared: flipping one sample's occupancy without flipping its map would
        put traffic on the wrong side of the road.
        """
        static, valid = self.static, self.valid
        if torch.rand(()) < 0.5:                      # mirror east-west
            for k in ("past_occupancy", "future_occupancy", "future_flow",
                      "last_observed_flow"):
                batch[k] = torch.flip(batch[k], dims=[-1])
            batch["future_flow"][:, :, 0] *= -1
            batch["last_observed_flow"][:, 0] *= -1
            static = torch.flip(static, dims=[-1]).clone()
            static[2] *= -1                            # lane_cos is the x component
            valid = torch.flip(valid, dims=[-1])
        if torch.rand(()) < 0.5:                      # mirror north-south
            for k in ("past_occupancy", "future_occupancy", "future_flow",
                      "last_observed_flow"):
                batch[k] = torch.flip(batch[k], dims=[-2])
            batch["future_flow"][:, :, 1] *= -1
            batch["last_observed_flow"][:, 1] *= -1
            static = torch.flip(static, dims=[-2]).clone()
            static[1] *= -1                            # lane_sin is the y component
            valid = torch.flip(valid, dims=[-2])
        return batch, static, valid

    def _loss_inputs(self, batch, valid):
        b = dict(batch)
        b["valid"] = valid.unsqueeze(0).expand(batch["past_occupancy"].shape[0], -1, -1)
        b["cell_area_m2"] = torch.tensor(self.cell_area, device=self.device)
        b["mean_vehicle_area_m2"] = self.mean_vehicle_area.unsqueeze(0).expand(
            batch["past_occupancy"].shape[0], -1)
        return b

    # -------------------------------------------------------------- steps --
    def training_step(self, batch, idx):
        if self.hparams.augment:
            batch, static, valid = self._augment(batch)
        else:
            static, valid = self.static, self.valid
        B = batch["past_occupancy"].shape[0]
        pred = self.net(batch["past_occupancy"], static.unsqueeze(0).expand(B, -1, -1, -1))
        loss, terms = forecasting_loss(pred, self._loss_inputs(batch, valid),
                                       self.weights, self.cells_per_frame_step)
        for k, v in terms.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=(k == "total"))

        # An explicit heartbeat.  Lightning's progress bar writes carriage
        # returns that never reach a redirected log file, so without this a run
        # is completely opaque until its first validation -- which for the
        # headline run is over an hour in.
        if self.trainer.is_global_zero and self.global_step % 100 == 0:
            now = time.time()
            dt = now - self._last_print
            done = self.global_step - self._last_step
            rate = done / dt if dt > 0 else 0.0
            total = self.trainer.max_steps if self.trainer.max_steps > 0 else \
                self.trainer.estimated_stepping_batches
            eta = (total - self.global_step) / rate / 3600 if rate > 0 else float("nan")
            print(f"[step {self.global_step:>7,}/{total:,}] "
                  f"loss {float(terms['total']):.4f}  "
                  f"occupancy {float(terms['occupancy_cross_entropy']):.4f}  "
                  f"flow {float(terms.get('flow_error', 0)):.4f}  "
                  f"trace {float(terms.get('flow_trace', 0)):.4f}  "
                  f"count {float(terms['vehicle_count_relative_error']):.4f}  "
                  f"{rate:.2f} steps/s  eta {eta:.1f} h", flush=True)
            self._last_print, self._last_step = now, self.global_step
        return loss

    def on_validation_epoch_start(self):
        t = self.cfg.t_out
        self._val_acc = Accumulator(t)
        self._baseline_acc = {}
        self._scenario_acc = {}
        self._video_batch = None

    def validation_step(self, batch, idx):
        B = batch["past_occupancy"].shape[0]
        static = self.static.unsqueeze(0).expand(B, -1, -1, -1)
        pred = self.net(batch["past_occupancy"], static)
        b = self._loss_inputs(batch, self.valid)
        loss, terms = forecasting_loss(pred, b, self.weights, self.cells_per_frame_step)
        for k, v in terms.items():
            self.log(f"validation/{k}", v, on_step=False, on_epoch=True, sync_dist=True)

        prob = torch.sigmoid(pred["occupancy_logits"])
        common = dict(valid=self.valid, cell_area=self.cell_area,
                      mean_vehicle_area=self.mean_vehicle_area)
        self._val_acc.update(prob, batch["future_occupancy"],
                             flow_pred=pred["flow"], flow_true=batch["future_flow"], **common)

        # Per-scenario, so a score dominated by quiet road cannot hide a failure
        # on the windows that matter.
        for name in ("nominal", "incident_traffic", "collision", "low_speed_contact"):
            if name == "collision":
                m = batch["has_collision"] & (batch["collision_speed"] >= 3.0)
            elif name == "low_speed_contact":
                m = batch["has_collision"] & (batch["collision_speed"] < 3.0)
            elif name == "incident_traffic":
                m = (~batch["has_collision"]) & (batch["n_bad_actors"] > 0)
            else:
                m = (~batch["has_collision"]) & (batch["n_bad_actors"] == 0)
            if m.any():
                self._scenario_acc.setdefault(name, Accumulator(self.cfg.t_out)).update(
                    prob[m], batch["future_occupancy"][m], **common)

        # Baselines on *every* validation batch, so the comparison is like for
        # like.  Scoring them on a subset while scoring the model on everything
        # would make the gap depend on which batches happened to come first.
        for name, p in all_baselines(batch, self.cells_per_frame_step,
                                     self.cfg.t_out).items():
            self._baseline_acc.setdefault(name, Accumulator(self.cfg.t_out)).update(
                p, batch["future_occupancy"], **common)
        if idx == 0:
            self._video_batch = ({k: v[:1].detach().cpu() for k, v in batch.items()},
                                 prob[:1].detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        head = self._val_acc.headline(FLOW_SCALE)
        for k, v in head.items():
            self.log(f"validation/{k}", v, sync_dist=True)
        m = self._val_acc.compute(FLOW_SCALE)
        for i in range(self.cfg.t_out):
            self.log(f"validation_by_horizon/soft_iou_at_{(i+1)*self.hparams.frame_dt_s:.1f}s",
                     float(m["soft_iou"][i]), sync_dist=True)
            self.log(f"validation_by_horizon/bias_at_{(i+1)*self.hparams.frame_dt_s:.1f}s",
                     float(m["bias"][i]), sync_dist=True)
        for name, acc in self._scenario_acc.items():
            h = acc.headline(FLOW_SCALE)
            for k in ("soft_iou_mean", "soft_iou_last_step", "bias_mean"):
                self.log(f"validation_by_scenario/{name}_{k}", h[k], sync_dist=True)
        named = {"model": m}
        for name, acc in self._baseline_acc.items():
            bm = acc.compute(FLOW_SCALE)
            named[name] = bm
            h = acc.headline(FLOW_SCALE)
            for k in ("soft_iou_mean", "soft_iou_last_step", "bias_mean", "f1@0.1_mean"):
                self.log(f"baseline_{name}/{k}", h[k], sync_dist=True)
            self.log(f"gap_over_baseline/{name}_soft_iou_mean",
                     head["soft_iou_mean"] - h["soft_iou_mean"], sync_dist=True)

        if self.trainer.is_global_zero:
            print(f"\n[epoch {self.current_epoch}] soft intersection-over-union "
                  f"by horizon ({self.hparams.frame_dt_s:.1f} s steps)")
            print(table(named, "soft_iou", self.hparams.frame_dt_s))
            print("\nbias (predicted mass / true mass; 1.0 is unbiased)")
            print(table(named, "bias", self.hparams.frame_dt_s))
            self._write_report(named, head)
            if (self.current_epoch % self.hparams.video_every_n_epochs == 0
                    and self._video_batch is not None):
                self._write_video()

    # ---------------------------------------------------------- reporting --
    def _write_report(self, named, head):
        d = os.path.join(self.output_dir, "reports")
        os.makedirs(d, exist_ok=True)
        rec = {"epoch": self.current_epoch, "step": self.global_step,
               "wall_seconds": round(time.time() - self._t0, 1),
               "frame_dt_s": self.hparams.frame_dt_s, "headline": head,
               "by_horizon": {k: list(map(float, v)) for k, v in named["model"].items()},
               "baselines": {n: {k: list(map(float, v)) for k, v in m.items()}
                             for n, m in named.items() if n != "model"},
               "by_scenario": {n: a.headline(FLOW_SCALE)
                               for n, a in self._scenario_acc.items()}}
        with open(os.path.join(d, "validation.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _write_video(self):
        batch, prob = self._video_batch
        road = self.static[0].detach().cpu().numpy() > 0.05
        past = batch["past_occupancy"][0].numpy()
        true = batch["future_occupancy"][0].numpy()
        pred = prob[0].numpy()
        frames = []
        for k in range(past.shape[0]):
            img = viz.render_occupancy(past[k], road=road, scale=4)
            img = viz.caption(img, [f"observed  t = {(k - past.shape[0] + 1)*self.hparams.frame_dt_s:+.1f} s"])
            frames.append(viz.hstack([img, img * 0 + 20]))
        for k in range(true.shape[0]):
            t = (k + 1) * self.hparams.frame_dt_s
            a = viz.caption(viz.render_occupancy(true[k], road=road, scale=4),
                            [f"ground truth  t = {t:+.1f} s"])
            b = viz.caption(viz.render_occupancy(pred[k], road=road, scale=4),
                            [f"prediction    t = {t:+.1f} s",
                             f"predicted mass {pred[k].sum():.0f} vs true {true[k].sum():.0f}"])
            frames.append(viz.hstack([a, b]))
        d = os.path.join(self.output_dir, "videos")
        path = viz.write_video(os.path.join(d, f"epoch_{self.current_epoch:04d}.mp4"),
                               frames, fps=4)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"validation/forecast_video": wandb.Video(path, format="mp4")},
                          step=self.global_step)
        except Exception:
            pass

    # -------------------------------------------------------- optimisation --
    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or "pos_" in n or n.endswith("globals")
             else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.learning_rate, betas=(0.9, 0.95))

        warm, total = self.hparams.warmup_steps, self.hparams.total_steps
        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            t = min(1.0, (step - warm) / max(1, total - warm))
            return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * t))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
