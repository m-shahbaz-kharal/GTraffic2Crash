# GTraffic2Crash — design

Forecast the next few seconds of town-wide traffic from a bird's-eye occupancy
grid, well enough that an imminent collision is visible in the forecast.

This document records the decisions and why they were made, so that a result —
good or bad — can be traced to a choice rather than to an accident.

---

## 1. The problem, stated precisely

An upstream model (LCam2GlobalTraffic) already turns 141 fixed traffic-camera
images into one town-wide vehicle-occupancy grid over CARLA Town05: 136 x 153
cells of 4 m, covering 544 m x 612 m, scored at 0.808 occupied-cell F1 against
ground truth. This project takes that grid as its *input format* and asks the
next question:

> given the last T_in grids, predict the next T_out grids.

Everything downstream — crash anticipation, congestion, violation detection —
depends on that forecast being genuinely good first. So the deliverable is a
forecaster and an honest measurement of it, not a crash classifier.

### Why the forecast is the right target rather than a crash label

A crash label is one bit at one instant. It is rare, it arrives too late to be
useful, and a model trained on it learns a shortcut ("two blobs are close")
rather than dynamics. A forecast is dense supervision — every cell at every
future step — and a model that can say *where every vehicle will be in three
seconds* has necessarily learned the thing a crash predictor needs. Crash
anticipation then falls out as a read-out of the forecast, and can be evaluated
separately without retraining.

---

## 2. What the data is

### 2.1 Two layers, deliberately

**Raw layer — per-tick vehicle state.** For every simulator tick: each vehicle's
id, class, centre, yaw, extent, velocity, angular velocity, plus traffic-light
states, plus a per-vehicle behaviour tag (normal / speeding / red-light-runner /
erratic) and collision events. About 8 KB per frame at 200 vehicles.

**Derived layer — rasterised tensors** built from the raw layer by
`g2c/grid.py`, stored sparsely.

The split matters. Grid resolution, time step, channel layout and class grouping
are all *decisions I might get wrong*. Keeping the raw states means any of them
can be revised for the cost of a CPU pass, without re-running CARLA. Roughly
50 hours of simulated traffic is ~2 GB of raw state and several days of compute
to regenerate — that asymmetry is worth the extra layer.

### 2.2 Grid

Same origin, same 4 m cell as upstream, padded to **144 x 160** so the shape
divides cleanly by 2, 4, 8 and 16. The 11 % of added cells sit outside the town
and are always empty; they cost nothing and remove every awkward remainder in
the network.

4 m is coarse relative to a car — but the representation is *soft coverage*, not
a binary flag, so sub-cell position survives. A car straddling a cell boundary
splits its area between the two cells in proportion; the centroid is recoverable
to well under a metre. This is the single most important property of the
representation and it is why the coarse grid is acceptable: **velocity is still
observable across frames even when a vehicle moves less than one cell**. A
binary grid at 4 m would destroy that and force a much finer, much more
expensive raster.

### 2.3 Channels

Per frame, the target tensor is per-class **area coverage** in [0, 1] — the
fraction of each cell covered by that class's footprints:

| ch | class | why separate |
|---|---|---|
| 0 | car | the bulk of traffic |
| 1 | large (truck, van, bus) | different size, acceleration and turning radius |
| 2 | two-wheeler (motorcycle, bicycle) | much slower, much smaller, different lanes |

Six raw classes are grouped into three. Six channels would put motorcycles at
~0.05 % density, which is too sparse to learn a stable per-channel distribution;
three groups keep every channel above ~0.3 % while preserving the distinction the
dynamics actually depend on. The raw layer keeps all six, so this is revisable.

Coverage rather than the upstream *mass* convention (each vehicle contributes a
total of 1.0). Three reasons: coverage is bounded in [0, 1], which is what makes
a probabilistic loss well posed; coverage draws a bus as a large solid blob
rather than a faint one, which is what "a picture of the traffic" should mean;
and overlapping footprints — i.e. a collision — show up directly as saturated
coverage. Mass is still computed and stored, because it is exactly the quantity
the count-consistency loss term needs and it is the upstream convention. The two
are related by a per-class constant, so nothing is lost.

**Static conditioning channels** (constant in time, concatenated to every frame):
drivable-road mask, lane-direction field (sin/cos of the lane heading), junction
mask. The road prior is worth a great deal: 28 % of cells touch road and hold
80 % of all vehicle mass. Telling the model where a vehicle *can* be is free
information that it would otherwise spend capacity rediscovering.

**Flow channels** (targets, and inputs for past frames): coverage-weighted mean
velocity (v_x, v_y) per cell, in m/s. See §4.3 for why these earn their place.

Traffic-light state is recorded in the raw layer but **not** fed to the model.
The task is to predict traffic from traffic; feeding the signal phase would let
the model learn "cars stop when the light is red" without learning that cars
decelerate, and the upstream camera model would not supply it at inference.

### 2.4 Time step and horizon

**dt = 0.5 s (2 Hz), T_in = 10 (5 s history), T_out = 10 (5 s horizon).**

The simulator runs at 20 Hz for physics fidelity; every 10th tick is kept.

The choice of dt is a real trade-off and worth spelling out. A vehicle at 10 m/s
moves 5 m per frame at 2 Hz — 1.25 cells. Too fast (dt large) and frame-to-frame
correspondence is ambiguous; too slow (dt small) and consecutive frames are
nearly identical, so copying the last frame becomes a near-perfect solution and
the model learns nothing. 1.25 cells per frame sits where motion is unmistakable
but continuous. 5 s is also the horizon that matters: the interval between "this
looks wrong" and impact is typically 1–3 s.

Because the raw layer is stored at 20 Hz, dt is a preprocessing constant, not a
commitment.

---

## 3. What the model is

### 3.1 The evidence

Three findings from the literature drove this, and they point the same way.

**Recurrent-free wins on this kind of data.** The OpenSTL benchmark (NeurIPS
2023 D&B) evaluates 24 models on a common footing. On TaxiBJ — a 32x32 traffic
flow grid, the closest public analogue to this problem — the best recurrent-free
model reaches MSE 0.3108 at 2.49 GFLOPs and 1268 FPS, while the best recurrent
model (MIM) reaches 0.3110 at 64.10 GFLOPs and 275 FPS: identical accuracy, 26x
the compute. On WeatherBench the recurrent-free models win outright. Their
stated explanation is that recurrent models carry a *local* temporal inductive
bias that pays off on high-frequency data and wastes compute on low-frequency
data. Town traffic at 2 Hz is low-frequency. Predicting all future frames in one
parallel pass also sidesteps autoregressive drift entirely.

**Axial factorisation is the best-validated space-time attention.** Earthformer
(NeurIPS 2022) ablates seven attention factorisations under identical conditions.
Axial attention plus global vectors wins on every metric on both benchmarks
(MovingMNIST MSE 41.79, N-body 14.82) *and* is near-cheapest (7.6 M params,
34 GFLOPs). Video-Swin and divided space-time are strictly dominated; the largest
joint space-time cuboid tested is the worst pattern of all.

**Attention is not automatically the answer.** In OpenSTL's translator sweep — same
encoder, same decoder, only the temporal module changed — plain ViT is the worst
or near-worst variant on every dataset, while gated convolutional translators
(MogaNet, VAN, ConvNeXt, gSTA, TAU) lead. The lesson is that the
encoder–translator–decoder *scaffold* does most of the work, and that a
transformer earns its place only if it is factorised sensibly.

### 3.2 The architecture

```
                    144 x 160 x (3 coverage + 2 flow + 4 static)  x 10 frames
                                        |
   conv stem        3 residual conv blocks, stride 2 overall      per frame, shared
                                        |
                         72 x 80 tokens x d, x 10 frames         = 57,600 tokens
                                        |
   encoder          N_enc axial blocks:  attend over T
                                         attend over H            + P global vectors
                                         attend over W
                                        |
                                    memory
                                        |
   decoder          N_dec axial blocks, each also cross-attending to memory
                    queries = 10 learned future-frame embeddings
                                        |
   conv head        transposed conv back to 144 x 160
                                        |
              occupancy logits (3 x 10 frames)   +   flow (2 x 10 frames)
```

Every attention is over one axis at a time, so the cost of the whole stack is
linear in the number of tokens rather than quadratic. `P = 8` global vectors
(Earthformer's value) restore the long-range mixing that pure axial attention
lacks — a congestion wave or a signal phase change is a genuinely global effect,
and the ablation shows global vectors help every pattern tested.

The conv stem and head matter as much as the transformer. Attention over 8 m
patches cannot place a vehicle to sub-cell precision; the convolutional decoder
is what turns a coarse token field back into a sharp raster. This is the
scaffold that OpenSTL's ablation says does most of the work.

### 3.3 Size

Per axial block: 3 attention sub-blocks (4d^2 each) + 3 feed-forward blocks
(8d^2 each) = 36d^2. With d = 768 and 12 encoder + 8 decoder blocks (the decoder
blocks carry an extra cross-attention, 4d^2, so 40d^2):

    encoder  12 x 36 x 768^2  = 255 M
    decoder   8 x 40 x 768^2  = 189 M
    stem, head, embeddings                ~ 15 M
                                  total  ~ 460 M

That is deliberately larger than anything in the surveyed occupancy literature
except the diffusion models (DOME 444 M, COME 693 M), and about 30x Earthformer.
The reasoning is the instruction to prove the idea at full scale first: a model
that fails at 460 M has failed for a reason worth knowing, whereas a model that
fails at 15 M might simply be too small. Activations at batch 4 in bf16 with
checkpointing are ~20 GB, which fits one H100 with room to spare.

If the 460 M model overfits, that is a *measurable* outcome and the response is
more data, which §5 shows is cheap. Starting small and scaling up would confound
the two.

---

## 4. The loss

This is where most of the thinking went, because the naive choices all fail in
known ways and several popular choices are provably wrong for this target.

### 4.1 What is ruled out, and why

**Soft Dice, soft IoU, Tversky.** Wang et al. (MICCAI 2023, arXiv 2303.16296)
prove that for a fractional target these are *not* minimised at the truth: with
y = 0.5, soft Dice is minimised at x = 1. Every non-zero cell in this grid holds
a fractional value, so soft Dice would push every partially-covered cell toward
1.0 and inflate the vehicle count. Separately, Bertels et al. (Medical Image
Analysis 2021) show soft Dice introduces a volume bias precisely in tasks with
high inherent uncertainty — and forecasting is definitionally that. Ruled out.

**Lovasz-softmax.** It is the convex closure of a set function over
*misclassified pixel sets*. That set is undefined when the ground truth is 0.4.
Applying it would require thresholding the target, destroying the sub-cell
information that §2.2 identifies as the whole point of soft coverage. Ruled out,
despite its ubiquity in the occupancy literature — which inherits it from LiDAR
*semantic segmentation*, where labels are hard.

**Focal loss as the primary term.** Charoenphakdee et al. (arXiv 2011.09172)
prove focal loss is not strictly proper for any gamma > 0: its minimiser is not
the true posterior, and the bias direction is under-confidence. AdaFocal
(arXiv 2211.11838) measures the damage — on ImageNet, focal is 5.7x worse
calibrated than plain cross-entropy (16.77 % vs 2.93 % ECE). Under-confidence
here means systematically under-predicting occupancy, i.e. losing vehicles. The
diagnostic exists in the forecasting literature too: PreDiff measures
Earthformer's BIAS at the highest threshold as **0.24** — it predicts a quarter
of the area it should, and wins the threshold metric by hedging.

### 4.2 The core term: prior-initialised soft cross-entropy

    L_occ = - w . [ y log p + (1 - y) log(1 - p) ]

with y the target coverage in [0, 1] and p the predicted coverage.

This is minimised exactly at p = E[y]. It is proper, it is decomposable over
cells (so the gradient scale does not swing with the number of vehicles in the
frame, unlike every Dice-family loss), and it is defined for fractional targets
without modification.

The usual objection is that 98 % of cells are empty and will swamp the gradient.
That objection is weaker than it looks. The gradient with respect to the logit
is `p - y`; an empty cell the model already predicts as empty contributes almost
nothing. The imbalance bites at *initialisation*, where p = 0.5 everywhere and
20,000 empty cells outvote 400 occupied ones. The fix is the one Lin et al.
prescribed for exactly this and which they report was necessary to keep training
from diverging: **initialise the output bias to the prior**, here
`-log((1 - 0.02) / 0.02) = -3.9`. Cheap, principled, and it removes the reason
to reach for focal loss at all.

`w` is a per-cell weight, not a focal modulator: `w = 1 + 9 * dilate(y > 0)`.
Cells at or near a vehicle count ten times a cell in the middle of a field. This
is the DGMR/SimCast intensity-weighting idea, and SimCast's ablation shows a
weighted loss alone buys more skill (+4.5 % CSI) than an entire diffusion stage —
at zero inference cost. It is a stated, fixed reweighting of the sampling
distribution rather than a distortion of the link function, so the bias it
introduces is known rather than emergent.

### 4.3 Flow, and flow-traced consistency

The model also predicts a per-cell velocity field, supervised by L1 against the
rasterised ground-truth flow on occupied cells. Then:

    L_trace = crossentropy( warp(occupancy_t, flow_t) , occupancy_{t+1} )

Warping the predicted occupancy at t along the predicted flow must reproduce the
predicted occupancy at t+1. This is Mahjourian et al.'s flow-trace loss (Waymo,
RA-L 2022), and it is the term that carries the physics: mass moves, it does not
teleport or evaporate. It is also the honest answer to the "L2 gives blur"
problem — rather than bolting on an adversary, it constrains *how* probability
mass may be redistributed between frames.

### 4.4 Count consistency

    L_count = | sum(predicted mass) - sum(true mass) |   per frame, per class

The one term that directly attacks the BIAS-0.24 failure. Weighted
cross-entropy is no longer proper, so it will drift; this measures the drift in
the one summary statistic that is physically meaningful (the vehicle count) and
corrects it. This is the counting term of DM-Count (NeurIPS 2020), whose whole
design is for "a sparse density map whose total mass is a count" — verbatim this
problem. DM-Count's ablation (103.1 -> 94.9 -> 85.6 MAE) shows the mass term
alone is insufficient, which is why it is one term of four here and not the
loss.

DM-Count's optimal-transport term is deliberately *not* included in version one.
Sinkhorn over 23,000 cells at ~25 ms per frame per sample is a real cost, and
the flow-trace term already supplies a spatial-transport prior grounded in the
actual physics rather than in a generic ground metric. It is the first thing to
add if the forecast turns out to be spatially diffuse in a way flow does not fix.

### 4.5 The whole loss

    L = L_occ  +  0.5 * L_flow  +  0.5 * L_trace  +  0.05 * L_count

Weights are a starting point, to be checked by watching each term separately in
the logs — every term is logged individually for exactly this reason.

---

## 5. How much data, and how it is made

### 5.1 The budget

Anchors from the literature: Earthformer trains 15 M params on 35,718 SEVIR
sequences; Cam4DOcc trains on 23,930; OpenSTL's MovingMNIST results use 10,000;
the Waymo occupancy challenge has 487,000. A 460 M model on a *single town* —
much lower scene diversity than nuScenes — should sit at the high end of that
range in sample count while accepting that its intrinsic diversity is lower.

Target: **~375,000 training windows** from **~50 hours** of simulated traffic.

    50 h x 3600 s x 2 Hz  = 360,000 frames
    stride-1 windows of 20 frames ~= 359,000 windows

Stride-1 windows overlap heavily, so the count overstates independence; the real
diversity comes from episode count. So: **400 episodes x 7.5 minutes**, each with
its own seed, vehicle count sampled in [120, 320], and bad-actor configuration.
Many short episodes beat few long ones for diversity, at the cost of repeated
spawn overhead.

Storage: raw state ~2 GB; rasterised tensors stored **sparsely** (only the ~2–6 %
non-zero cells, as int32 index + uint8 value) ~2 GB, densified in the data loader
in microseconds. A dense float16 store would be 52 GB for the same content.

### 5.2 Making the traffic, and making it crash

Normal traffic comes from CARLA's Traffic Manager in synchronous mode with a
fixed seed. Bad actors are the same vehicles with the Traffic Manager's per-agent
knobs pushed: `ignore_lights_percentage`, `vehicle_percentage_speed_difference`
(negative is faster), `ignore_vehicles_percentage`, `distance_to_leading_vehicle`
set to zero, and randomised lane changes.

**Whether that actually produces collisions is an empirical question, not an
assumption.** A probe job (`slurm/probe_carla.sbatch`) measures it directly,
along with achievable ticks per second at 200/250/350 vehicles, with and without
rendering, and at two timesteps. Nothing about the generation plan is fixed until
those numbers land. If Traffic Manager manipulation alone does not crash reliably,
the fallback is scripted conflict: pick two vehicles on conflicting paths at a
junction, take one off autopilot, and drive it through with direct
`VehicleControl`.

Episodes are split roughly half nominal traffic and half incident traffic, and
every window is **indexed by what happens in it** — nominal, speeding present,
red-light violation, near-miss, collision — so that sampling can be balanced and,
more importantly, so that **forecast quality can be reported per scenario type**.
An aggregate number that is dominated by empty road at 3 a.m. would hide exactly
the failure that matters.

Parked vehicles are included (they are real, and the upstream camera model
outputs them) but the primary metrics are computed on **dynamic vehicles only**,
with the all-vehicle number reported alongside. Parked cars are trivially
predictable and would otherwise inflate every score.

---

## 6. How it is judged

Two baselines must be beaten before any claim is made, and both are implemented
before the model is:

1. **Persistence** — repeat the last observed frame. At 5 s this should be poor,
   but it is the floor, and on sparse grids it is a surprisingly strong floor.
2. **Constant velocity** — warp the last frame by the observed flow field. This
   is the honest baseline: it captures everything obtainable without learning
   anything about roads, signals or interaction. **The model's entire value is
   the gap between it and this.**

Metrics, per future timestep (not averaged into one number, which hides
degradation):

| metric | what it catches |
|---|---|
| Soft-IoU and AUC-PR on occupancy | overall spatial agreement, threshold-free |
| **BIAS** = predicted area / true area | the hedging failure; must be near 1.0 |
| Vehicle-count MAE | mass conservation |
| Flow end-point error on occupied cells | whether motion is right, not just position |
| Occupied-cell F1 at several pooling scales (1, 2, 4) | how much of the skill is real placement versus blur |
| All of the above, **split by scenario type** | whether it works when it matters |

The pooling-scale sweep is taken from the nowcasting literature, where it is the
standard diagnostic separating genuine placement skill from blur: a blurry model's
score improves sharply with pooling, a sharp one's does not.

A specific, separately reported question: **on windows containing a collision,
does the predicted occupancy show the two vehicles converging?** Measured as
forecast quality restricted to a radius around the impact point, versus the same
model's quality on nominal windows.

---

## 7. Order of work

Each step produces something checkable before the next begins.

1. ~~Grid geometry and exact rasteriser, validated against upstream.~~ **done**
2. CARLA probe: throughput and collision rates. **queued**
3. Generator: episodes to raw state. Visual check of a single episode.
4. Preprocessing: raw state to sparse tensors. Videos of sample windows.
5. Baselines: persistence and constant velocity, scored. This fixes the bar.
6. Model, on a deliberately tiny subset, overfitted to ~zero loss. If a 460 M
   model cannot memorise 100 windows, something is wrong with the plumbing and
   no amount of data will fix it.
7. Full training, with per-term loss logging and validation forecasts rendered
   as video every epoch.
8. Evaluation against the baselines, per timestep and per scenario type.

---

## 8. Known risks

- **Traffic Manager may not produce collisions.** Step 2 answers this. Fallback
  in §5.2.
- **Persistence may be hard to beat at short horizons.** Expected and fine; the
  test is the 3–5 s range, and the per-timestep reporting will show exactly where
  the model starts to add value.
- **The forecast may be diffuse.** The literature says to expect it from any
  deterministic model. The response is to *measure* it (BIAS, pooling sweep)
  rather than hide it, and the documented next step is a cascade — a
  deterministic backbone with a generative refinement stage, which CasCast shows
  beats both pure approaches — not a switch to pure diffusion, which CasCast's
  own ablation shows collapses (CSI 0.3210 vs 0.4310 for the deterministic model
  alone).
- **A single town limits generalisation.** True and unaddressed in version one.
  Town05 is the town the upstream model was built for, so it is the right first
  target; other towns are a scaling axis, not a correctness issue.
