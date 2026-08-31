# GTraffic2Crash

Forecasting town-scale traffic from a bird's-eye occupancy grid, so that an
imminent collision is visible in the forecast before it happens.

The upstream project ([LCam2GlobalTraffic](../LCam2GlobalTraffic)) turns 141
fixed traffic-camera images into one town-wide vehicle-occupancy grid over CARLA
Town05. This project takes that grid as its input format and answers the next
question: **given the last five seconds of the town, what do the next five
seconds look like?**

Crash anticipation is then a read-out of the forecast rather than a separate
model. A network that can say where every vehicle will be in three seconds has
necessarily learned what a crash predictor needs; a network trained on a
one-bit, very rare crash label learns a shortcut instead.

Read [`docs/DESIGN.md`](docs/DESIGN.md) for the decisions and why they were made.

## What is here

```
g2c/grid.py        grid geometry; exact rotated-rectangle / cell intersection
g2c/tensorize.py   recorded vehicle state -> sparse per-class coverage + flow
g2c/data.py        windows of past and future, split by episode, labelled by scenario
g2c/model.py       axial space-time transformer, ~440 M parameters
g2c/losses.py      soft cross-entropy + flow + flow-trace + count consistency
g2c/metrics.py     per-horizon soft-IoU, F1 at several pooling scales, and BIAS
g2c/baselines.py   persistence and constant velocity (oracle and estimated flow)
g2c/train.py       Lightning module; baselines scored beside the model every epoch
g2c/viz.py         renders grids, videos and comparison panels
scripts/           generation, tensorisation, training, evaluation, probes
slurm/             cluster jobs
```

## The data

CARLA 0.9.15, Town05, 400 episodes of 7.5 minutes -- 50 hours of traffic.
Half the episodes are nominal; half carry 6-14 % bad actors (speeding,
red-light running, erratic weaving, tailgating).

Recorded as **raw vehicle state** at 10 Hz, then rasterised to a **144 x 160
grid of 4 m cells** at 2 Hz. Keeping the raw layer means grid resolution, time
step and channel layout can all be revised for the cost of a CPU pass instead of
re-running the simulator.

The target is per-class **area coverage** in [0, 1] -- the fraction of each cell
covered by a vehicle footprint. Soft rather than binary, which is what lets a
4 m grid still resolve sub-metre position and therefore velocity.

## What the probes established

Nothing about the generation plan was assumed; each of these was measured first.

| question | answer |
|---|---|
| Town05 capacity | 302 spawn points |
| throughput | 4.4x realtime at 60 vehicles, 1.1x at 200, 0.8x at 302 |
| does turning rendering off help? | barely -- 1.3x. CARLA is CPU-bound, so generation runs on the idle V100 nodes and leaves the H100s for training |
| is the traffic gridlocked? | no. The moving fraction oscillates 0.25-0.83 on the traffic-light cycle |
| do Traffic-Manager vehicles crash? | never with default settings (0 per minute) |
| ... with bad actors? | 126 per simulated minute across 172 of 250 vehicles when they ignore other vehicles -- carnage. Rewritten to keep collision avoidance on and break the *rules* instead, giving roughly 1 meaningful crash per minute |

## Running it

```bash
uv sync

# probes (already run; results in logs/)
sbatch slurm/probe_carla.sbatch
sbatch slurm/probe_density.sbatch
sbatch slurm/probe_crash.sbatch

# data
sbatch slurm/build_data_smoke.sbatch      # static map + two check episodes
sbatch slurm/generate_dataset.sbatch      # 400 episodes across 40 CARLA servers
.venv/bin/python scripts/tensorize_episodes.py --raw <raw> --out <tensor>
.venv/bin/python scripts/visualise_dataset.py

# training
sbatch slurm/smoke_train.sbatch                    # throughput + overfit check
RUN=base_v1 SIZE=base sbatch slurm/train.sbatch

# evaluation
.venv/bin/python scripts/evaluate.py --checkpoint runs/base_v1/checkpoints/last.ckpt \
    --out runs/base_v1/evaluation
```

Weights & Biases runs **offline** on this cluster. After `wandb login`,
`scripts/sync_wandb.sh` uploads every run. Training also writes
`runs/<run>/reports/validation.jsonl` and a forecast video per epoch, so
progress is readable without W&B.

## Cluster notes

* Request GPUs with untyped `--gres=gpu:1` plus `--constraint=h100`; a typed
  gres allocates but does not expose the device.
* The compute nodes ship no `nvidia_icd.json`, so CARLA's Vulkan loader cannot
  find the GPU until `~/software/carla-scripts/vulkan_setup.sh` is sourced.
* Nodes `evc46`-`evc50` are excluded; their GPUs have been unreliable.
