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

CARLA 0.9.15, Town05, 400 episodes of 7.5 minutes -- **50 hours of traffic,
360,480 frames, roughly 356,000 training windows**, of which about 15 % contain a
real collision. Half the episodes are nominal; half carry 6-14 % bad actors
(speeding, red-light running, erratic weaving, tailgating). Forty CARLA servers
generated it in under two hours on the idle V100 nodes.

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

## Results so far

Held-out validation, split by episode. The bar is constant-velocity extrapolation
given the *ground-truth* velocity field -- an advantage no deployed system has.

| forecast | mean soft-IoU | +0.5 s | +5.0 s | BIAS |
|---|---|---|---|---|
| **model** (26 M, trained) | **0.762** | **0.904** | **0.627** | 1.00 - 1.06 |
| persistence | 0.423 | 0.547 | 0.312 | 1.00 |
| constant velocity, oracle flow | 0.479 | 0.819 | 0.307 | 0.98 |
| constant velocity, estimated flow | 0.361 | 0.604 | 0.200 | 1.00 |

The model beats the oracle baseline at *every* horizon including half a second
-- where the oracle is handed the true velocity of every vehicle -- and at five
seconds it reaches **exactly double** the best baseline (0.623 against 0.312).
BIAS stays between 1.00 and 1.06 across the whole horizon, having started at
1.55 -- so the overlap is not bought by hedging, which is the failure mode that
metric exists to catch. It does not fail where it matters either: collision
windows score 0.758 against nominal traffic's 0.764.

### Does the forecast show a crash coming?

Measured, not trained for -- no term in the loss singles collisions out. For each
held-out window containing a real collision, the forecast's occupancy at the
impact site is compared with a matched control site (a cell occupied in the last
observed frame, well away from the impact).

| lead time | forecast at impact | at control | contrast |
|---|---|---|---|
| at impact | 0.0904 | 0.0566 | **1.60x** |
| 1.0 s before | 0.0859 | 0.0588 | 1.46x |
| 2.0 s before | 0.0800 | 0.0592 | 1.35x |
| 3.0 s before | 0.0746 | 0.0602 | 1.24x |
| 4.5 s before | 0.0709 | 0.0712 | 1.00x |

The forecast concentrates **1.6x more occupancy where the crash actually
happens**, with the signal decaying to nothing by four and a half seconds -- an
honest three-to-four second warning horizon.

A 439 M model is training on the full 44-hour dataset.

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

## Checking on a run

```bash
./scripts/status.sh                        # everything in flight, one line each
.venv/bin/python scripts/report_run.py base_v1 --plot
```

`report_run.py` prints the gap over every baseline, the decay across the horizon,
BIAS drift and the per-scenario split, and writes `runs/<run>/reports/progress.png`.
Training also writes a forecast video per epoch to `runs/<run>/videos/` -- ten
observed frames, then ten pairs of ground truth beside the prediction.

The headline run needs about 39 hours against a 36-hour job limit, so SLURM will
requeue it once; the batch script picks up `last.ckpt` automatically and the
resume path is tested (`slurm/resume_test.sbatch`). If it needs restarting by
hand:

```bash
RUN=base_v1 SIZE=base BATCH=8 EPOCHS=6 LR=2e-4 \
EXTRA="--warmup-steps 1000 --max-steps 32000 --val-check-interval 0.05 --limit-val-batches 150" \
  sbatch slurm/train.sbatch
```

When it finishes:

Do the evaluation **before** merging the backfilled episodes. The split is a
function of which episodes are on disk; `runs/base_v1/episode_split.json` pins it,
but keeping `tensor/` unchanged until after evaluation keeps everything simple.

```bash
RUN=base_v1 sbatch slurm/evaluate.sbatch          # uses the run's recorded split
RUN=base_v1 sbatch slurm/crash_analysis.sbatch    # the crash-anticipation numbers
mv $DATA/tensor_backfill/*.npz $DATA/tensor/      # 47 more episodes, for the next run
```

Weights & Biases is offline; `scripts/sync_wandb.sh` uploads every run after
`wandb login`.

## Cluster notes

* Request GPUs with untyped `--gres=gpu:1` plus `--constraint=h100`; a typed
  gres allocates but does not expose the device.
* The compute nodes ship no `nvidia_icd.json`, so CARLA's Vulkan loader cannot
  find the GPU until `~/software/carla-scripts/vulkan_setup.sh` is sourced.
* Nodes `evc46`-`evc50` are excluded; their GPUs have been unreliable.
