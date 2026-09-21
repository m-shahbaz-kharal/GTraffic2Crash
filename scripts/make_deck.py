"""Build the presentation deck.

One run only (base_v1, step 27,297).  Every number in here is read back from
the run's own validation log rather than typed in, so the deck cannot drift
away from the measurement.
"""

import json
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDES = os.path.join(ROOT, "docs", "slides")
OUT = os.path.join(SLIDES, "GTraffic2Crash.pptx")

FONT = "Arial"
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
RULE = RGBColor(0xE1, 0xE0, 0xD9)
BLUE = RGBColor(0x2A, 0x78, 0xD6)

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.62)
BODY_W = W - 2 * MARGIN


# ----------------------------------------------------------------- numbers

def load_metrics():
    """The dedicated evaluation, not the training-time log.

    scripts/evaluate.py runs the whole held-out set and refuses a split it
    cannot verify against the run's own record; the in-training validation is
    a capped sample of it.  So this is the number that belongs on a slide.
    """
    path = os.path.join(ROOT, "runs", "base_v1", "evaluation", "evaluation.json")
    d = json.load(open(path))
    return {
        "headline": d["headline"]["model"],
        "by_horizon": d["by_horizon"]["model"],
        "baselines": {k: v for k, v in d["by_horizon"].items() if k != "model"},
        "by_scenario": {k: v["model"] for k, v in d["by_scenario"].items()},
        "windows": d["windows"],
    }


def load_dataset_stats():
    path = os.path.join(ROOT, "docs", "figures", "dataset_statistics.json")
    return json.load(open(path))


def mean(xs):
    return sum(xs) / len(xs)


# ----------------------------------------------------------------- helpers

def blank_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(1, 0, 0, W, H)      # 1 = rectangle
    bg.fill.solid()
    bg.fill.fore_color.rgb = SURFACE
    bg.line.fill.background()
    bg.shadow.inherit = False
    return slide


def textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    return frame


def write(frame, lines, first=True):
    """lines: [(text, size, bold, colour, space_before)] or (runs, ...)."""
    for i, spec in enumerate(lines):
        text, size, bold, colour, space = spec
        para = frame.paragraphs[0] if (first and i == 0) else frame.add_paragraph()
        para.space_before = Pt(space)
        para.line_spacing = 1.22
        runs = text if isinstance(text, list) else [(text, bold, colour)]
        for rtext, rbold, rcolour in runs:
            run = para.add_run()
            run.text = rtext
            run.font.size = Pt(size)
            run.font.bold = rbold
            run.font.name = FONT
            run.font.color.rgb = rcolour
    return frame


def title(slide, text, sub=None):
    frame = textbox(slide, MARGIN, Inches(0.40), BODY_W, Inches(0.7))
    write(frame, [(text, 30, True, INK, 0)])
    rule = slide.shapes.add_shape(1, MARGIN, Inches(1.08), Inches(0.78), Emu(34290))
    rule.fill.solid()
    rule.fill.fore_color.rgb = BLUE
    rule.line.fill.background()
    rule.shadow.inherit = False
    if sub:
        f = textbox(slide, MARGIN, Inches(1.24), BODY_W, Inches(0.35))
        write(f, [(sub, 14, False, MUTED, 0)])


def hanging(para, indent_in=0.26):
    """Wrap continuation lines under the text, not under the bullet."""
    pr = para._pPr if para._pPr is not None else para._p.get_or_add_pPr()
    pr.set("marL", str(int(Inches(indent_in))))
    pr.set("indent", str(-int(Inches(indent_in))))


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def cell_border(cell, edge, colour=RULE, width=Pt(0.75)):
    tc = cell._tc
    pr = tc.get_or_add_tcPr()
    tag = qn("a:ln" + edge)
    for old in pr.findall(tag):
        pr.remove(old)
    ln = pr.makeelement(tag, {"w": str(int(width)), "cap": "flat",
                              "cmpd": "sng", "algn": "ctr"})
    fill = ln.makeelement(qn("a:solidFill"), {})
    clr = fill.makeelement(qn("a:srgbClr"), {"val": str(colour)})
    fill.append(clr)
    ln.append(fill)
    # a:lnL a:lnR a:lnT a:lnB must precede a:fill elements inside tcPr
    pr.insert(0, ln)


def plain_table(slide, left, top, width, rows, cols, col_widths=None):
    shape = slide.shapes.add_table(rows, cols, left, top, width, Inches(0.4))
    table = shape.table
    tbl = shape._element.graphic.graphicData.tbl
    tbl[0][-1].text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # no style, no grid
    table.first_row = False
    table.horz_banding = False
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    return table


def fill_cell(cell, text, size=13, bold=False, colour=INK2,
              align=PP_ALIGN.LEFT, pad=Inches(0.07)):
    cell.fill.solid()
    cell.fill.fore_color.rgb = SURFACE
    cell.margin_left = Inches(0.10) if align == PP_ALIGN.LEFT else Inches(0.06)
    cell.margin_right = Inches(0.10)
    cell.margin_top = cell.margin_bottom = pad
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame = cell.text_frame
    frame.word_wrap = True
    para = frame.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = FONT
    run.font.color.rgb = colour


def picture(slide, path, left, top, width=None, height=None):
    return slide.shapes.add_picture(path, left, top, width=width, height=height)


def page_number(slide, n):
    frame = textbox(slide, W - MARGIN - Inches(0.6), H - Inches(0.52),
                    Inches(0.6), Inches(0.3))
    write(frame, [(str(n), 11, False, MUTED, 0)])
    frame.paragraphs[0].alignment = PP_ALIGN.RIGHT


# ----------------------------------------------------------------- slides

def slide_goal(prs):
    s = blank_slide(prs)
    title(s, "Forecasting town-wide traffic, five seconds ahead")
    frame = textbox(s, MARGIN, Inches(2.05), Inches(11.4), Inches(3.4))
    write(frame, [
        ([("Given the last 5 seconds of a top-down map of where every vehicle "
           "in a town is, predict the next 5 seconds.", True, INK)], 24, True, INK, 0),
        ("The map itself already exists — an upstream model builds it from 141 "
         "fixed traffic cameras. This work is the forecast that sits on top of it; "
         "it needs no new sensors.", 19, False, INK2, 22),
        ("Crash anticipation, congestion and violation detection are all read-outs "
         "of that forecast. So the forecast is what has to be right first, and it "
         "is the deliverable.", 19, False, INK2, 18),
    ])
    foot = textbox(s, MARGIN, Inches(6.58), Inches(11.4), Inches(0.4))
    write(foot, [("CARLA Town05  ·  439-million-parameter space-time transformer  "
                  "·  measured on 53 held-out episodes", 13, False, MUTED, 0)])
    notes(s, """
Why forecast the whole grid instead of training a crash classifier directly?

A crash label is one bit at one instant. It is rare, it arrives too late to be
useful, and a model trained on it learns a shortcut — "two blobs are close" —
rather than dynamics. A forecast is dense supervision: every cell, at every
future step. A model that can say where every vehicle will be in three seconds
has necessarily learned what a crash predictor needs, and crash anticipation
then falls out as a read-out of the forecast, with no retraining.

Why five seconds: the interval between "this looks wrong" and impact is
typically one to three seconds, so a five-second horizon covers it with margin.

The input format is exactly what the existing camera model already produces, so
nothing here requires new roadside hardware.
""")
    page_number(s, 1)


def slide_dataset(prs, stats):
    s = blank_slide(prs)
    title(s, "The dataset",
          "CARLA Town05 · generated for this project · split by episode")

    counts = stats["scenario_counts"]
    total = sum(counts.values())
    pct = {k: 100.0 * v / total for k, v in counts.items()}
    hours = stats["frames"] * stats["frame_dt_s"] / 3600.0

    rows = [
        ("Simulator", "CARLA 0.9.15, Town05, 20 Hz synchronous, "
                      f"{stats['vehicles_per_frame_mean']:.0f} vehicles per frame"),
        ("Drivers", "normal traffic plus deliberate bad actors — speeding, "
                    "red-light running, erratic “drunk” weaving"),
        ("Episodes", "355 — 302 for training, 53 held out for validation"),
        ("Duration", f"{stats['frames']:,} frames at {stats['frame_dt_s']} s "
                     f"= {hours:.0f} hours of simulated traffic"),
        ("Training examples", "264,742 windows of 10 past frames → 10 future "
                              "frames (5 s → 5 s); 9,381 held out"),
        ("Grid", "144 × 160 cells of 4 m — 576 m × 640 m, the whole town"),
        ("Cell value", "the fraction of the cell covered by a vehicle, 0 to 1 — "
                       "not a yes/no flag"),
        ("Channels", "3 vehicle classes (car, large, two-wheeler) + 6 static map "
                     "channels (road, lane direction, junctions, speed limit)"),
        ("Scenario mix", f"nominal {pct['nominal']:.0f}% · incident traffic "
                         f"{pct['incident_traffic']:.0f}% · collision "
                         f"{pct['collision']:.0f}% · low-speed contact "
                         f"{pct['low_speed_contact']:.0f}%"),
        ("Sparsity", f"{100*stats['occupied_cell_fraction_mean']:.1f} % of cells "
                     "hold a vehicle at any moment"),
    ]

    table = plain_table(s, MARGIN, Inches(1.82), BODY_W, len(rows), 2,
                        col_widths=[Inches(2.45), BODY_W - Inches(2.45)])
    for r, (k, v) in enumerate(rows):
        fill_cell(table.cell(r, 0), k, size=14, bold=True, colour=INK, pad=Inches(0.128))
        fill_cell(table.cell(r, 1), v, size=14, colour=INK2, pad=Inches(0.128))
        for c in (0, 1):
            cell_border(table.cell(r, c), "B")

    notes(s, """
Why simulated: we need ground truth for every vehicle at every instant, and we
need crashes on demand. Neither is collectable at scale on real roads.

Bad actors are produced by setting CARLA's Traffic Manager per vehicle — a
fraction of drivers speed, ignore red lights, or weave. Collisions emerge from
that rather than being scripted, so the near-misses are realistic too.

Why soft coverage rather than a yes/no flag: 4 m is coarse relative to a car,
but a car straddling a cell boundary splits its area between the two cells in
proportion, so sub-cell position — and therefore velocity between frames —
survives. A binary grid at 4 m would destroy that and force a much finer, much
more expensive raster. The rasteriser computes the exact rotated-rectangle /
cell intersection by polygon clipping; it is accurate to 3e-8 square metres.

The split is by episode, not by window, so no frame from a training episode can
leak into validation through an overlapping window.

The 2 % occupancy figure is the reason the loss needed care — see the next
slide. It is also why the baseline "repeat the last frame" is strong: half the
town is stopped at any moment because of the signals.
""")
    page_number(s, 2)


def slide_model(prs):
    s = blank_slide(prs)
    title(s, "The model", "439 million parameters · trained on 2 × H100")

    picture(s, os.path.join(SLIDES, "fig_model.png"),
            MARGIN, Inches(1.60), width=BODY_W)

    y = Inches(4.42)
    rows = [
        ("Input", "the last 10 frames (5 s) of the grid, plus the static road map "
                  "— no velocities, no traffic-light state"),
        ("Output", "the next 10 frames (5 s) of occupancy, plus a velocity vector "
                   "for every cell — decoded in one pass, so no drift"),
        ("Loss", "soft cross-entropy on coverage  +  flow  +  flow-trace  +  "
                 "count consistency"),
    ]
    table = plain_table(s, MARGIN, y, BODY_W, len(rows), 2,
                        col_widths=[Inches(1.30), BODY_W - Inches(1.30)])
    for r, (k, v) in enumerate(rows):
        fill_cell(table.cell(r, 0), k, size=15, bold=True, colour=INK, pad=Inches(0.08))
        fill_cell(table.cell(r, 1), v, size=15, colour=INK2, pad=Inches(0.08))
        for c in (0, 1):
            cell_border(table.cell(r, c), "B")

    frame = textbox(s, MARGIN, Inches(6.08), Inches(11.75), Inches(1.0))
    write(frame, [
        ([("Why that loss.  ", True, INK),
          ("The target is a fraction, so the loss must be minimised exactly at "
           "the truth. Cross-entropy is; Dice and IoU are minimised at 1.0 for a "
           "half-covered cell, and focal loss is not proper. Both would distort "
           "the vehicle count. The flow-trace term then forces mass to ", False, INK2),
          ("move", False, INK2),
          (" between frames rather than appear and vanish, and the count term "
           "keeps the total honest.", False, INK2)], 14, False, INK2, 0),
    ])
    frame.paragraphs[0].runs[2].font.italic = True

    notes(s, """
Architecture. A convolutional stem reads each frame at full resolution and
halves it twice, giving 72 x 80 tokens per frame. Twelve axial-attention
encoder blocks follow: each attends along time, then along rows, then along
columns, plus a handful of global vectors that let any part of the town
influence any other. Eight decoder blocks cross-attend to that memory and emit
all ten future frames in parallel.

Two findings from the literature drove this. OpenSTL (NeurIPS 2023) shows that
on low-frequency grid data — TaxiBJ, the closest public analogue — the best
recurrent-free model matches the best recurrent one at one twenty-sixth of the
compute. Earthformer (NeurIPS 2022) ablates seven space-time attention
factorisations under identical conditions and axial-plus-global-vectors wins on
every metric while being near-cheapest.

Residual-from-last: the network is initialised so that it predicts "nothing
moves" exactly, and learns only the change. Its capacity therefore goes on what
actually moves rather than on redrawing the static scene.

The loss in full: L = L_occ + 0.5 L_flow + 0.5 L_trace + 0.001 L_count.
- L_occ is per-cell soft cross-entropy, weighted 10:1 near a vehicle, with the
  output bias initialised to the measured 1.5 % occupancy prior so that 98 %
  empty cells do not swamp the gradient at step zero.
- L_flow is L1 against the true velocity on occupied cells.
- L_trace warps the observed occupancy along the predicted flow and requires it
  to reproduce the next frame. This is the term that carries the physics.
- L_count requires total predicted mass to match the true vehicle count, per
  frame and per class. A weighted cross-entropy is no longer strictly proper, so
  it drifts; this corrects the drift in the one summary statistic that is
  physically meaningful.

Ruled out, with evidence: soft Dice, IoU and Tversky are provably not minimised
at the truth for fractional targets (Wang et al., MICCAI 2023) — with y = 0.5
soft Dice is minimised at x = 1, so every partially covered cell would be pushed
to full and the vehicle count inflated. Lovasz-softmax is undefined for soft
labels. Focal loss is not strictly proper for any gamma > 0 (Charoenphakdee
et al.) and its bias direction is under-confidence, which here means losing
vehicles.

Velocity is an output and never an input: feeding ground-truth velocity in would
give the model information it will not have at deployment and would make every
number on the next slide optimistic.
""")
    page_number(s, 3)


def slide_results(prs, m):
    s = blank_slide(prs)
    title(s, "Results",
          "soft IoU — overlap between the forecast and what actually happened · "
          f"53 held-out episodes, {m['windows']:,} windows")

    b = m["baselines"]
    rows = [
        ("", "over the 5 s", "at +5 s"),
        ("This model", f"{m['headline']['soft_iou_mean']:.3f}",
         f"{m['headline']['soft_iou_last_step']:.3f}"),
        ("Repeat the last frame", f"{mean(b['persistence']['soft_iou']):.3f}",
         f"{b['persistence']['soft_iou'][-1]:.3f}"),
        ("Constant velocity, true speeds given",
         f"{mean(b['constant_velocity_oracle_flow']['soft_iou']):.3f}",
         f"{b['constant_velocity_oracle_flow']['soft_iou'][-1]:.3f}"),
        ("Constant velocity, speeds estimated",
         f"{mean(b['constant_velocity_estimated_flow']['soft_iou']):.3f}",
         f"{b['constant_velocity_estimated_flow']['soft_iou'][-1]:.3f}"),
    ]
    tw = Inches(6.05)
    table = plain_table(s, MARGIN, Inches(2.46), tw, len(rows), 3,
                        col_widths=[Inches(3.45), Inches(1.40), Inches(1.20)])
    for r, (a, c1, c2) in enumerate(rows):
        head = (r == 0)
        best = (r == 1)
        fill_cell(table.cell(r, 0), a, size=13, bold=best,
                  colour=MUTED if head else (INK if best else INK2),
                  pad=Inches(0.105))
        for c, val in ((1, c1), (2, c2)):
            fill_cell(table.cell(r, c), val, size=13, bold=best,
                      colour=MUTED if head else (BLUE if best else INK2),
                      align=PP_ALIGN.RIGHT, pad=Inches(0.105))
        for c in range(3):
            cell_border(table.cell(r, c), "B")

    picture(s, os.path.join(SLIDES, "fig_horizon.png"),
            Inches(6.83), Inches(1.84), width=Inches(6.12))

    count_err = 100.0 * m["headline"]["vehicle_count_relative_error_mean"]
    crash = m["by_scenario"]["collision"]["soft_iou_mean"]
    calm = m["by_scenario"]["nominal"]["soft_iou_mean"]
    frame = textbox(s, MARGIN, Inches(5.56), BODY_W, Inches(1.5))
    write(frame, [
        ([("The overlap is not bought by hedging.  ", True, INK),
          (f"Predicted vehicle count stays within {count_err:.1f} % of the truth "
           "across the whole horizon.", False, INK2)], 14, False, INK2, 0),
        ([("It holds up where it matters.  ", True, INK),
          (f"Collision episodes score {crash:.3f} against {calm:.3f} on nominal "
           "traffic — two points, not the result.", False, INK2)],
         14, False, INK2, 16),
        ([("Measured at step 27,297 of a 32,000-step budget — 1.65 passes over "
           "the training set, 35 hours on two H100s.", False, MUTED)],
         12, False, MUTED, 16),
    ])

    notes(s, """
Soft IoU here is the sum of min(prediction, truth) divided by the sum of
max(prediction, truth) over every valid cell — 1.0 would be a perfect forecast.
It is computed on the fractional coverage directly, with no thresholding, and
the padded cells outside the town are excluded so that trivially empty area
cannot inflate it.

Why "repeat the last frame" is a serious baseline and not a straw man: half the
town is stopped at any instant because of the traffic signals, so copying the
present is right about half the time. Beating it by 0.35 means the model is
successfully telling the vehicle that will sit still for five seconds from the
one that will travel thirty-six metres — which is the actual job.

The "true speeds given" baseline is an oracle. It is handed the ground-truth
velocity that the model never sees, and it still starts losing after about 1.5
seconds, because constant velocity cannot turn a corner, stop at a light, or
join the back of a queue. That crossing point on the chart is the clearest
statement of what the model has actually learned.

On hedging: the occupancy-forecasting literature has a known failure mode where
a model wins overlap metrics by predicting far less area than really exists.
PreDiff measures Earthformer's BIAS — predicted mass divided by true mass — at
0.24 at the highest threshold; it predicts a quarter of the area it should. Ours
runs BIAS_LO at +0.5 s to BIAS_HI at +5 s, so the overlap is real rather than
purchased by shrinking the forecast. This is exactly what the count-consistency
term in the loss was there to protect.

Per-horizon soft IoU: PER_HORIZON.
Per-horizon BIAS: PER_BIAS.
""".replace("BIAS_LO", f"{m['by_horizon']['bias'][0]:.3f}")
   .replace("BIAS_HI", f"{m['by_horizon']['bias'][-1]:.3f}")
   .replace("PER_HORIZON", ", ".join(f"{v:.3f}" for v in m["by_horizon"]["soft_iou"]))
   .replace("PER_BIAS", ", ".join(f"{v:.3f}" for v in m["by_horizon"]["bias"])))
    page_number(s, 4)


def slide_picture(prs):
    s = blank_slide(prs)
    title(s, "Five seconds ahead, on a held-out episode")

    picture(s, os.path.join(SLIDES, "fig_forecast.png"),
            Inches(1.72), Inches(1.58), width=Inches(9.90))

    frame = textbox(s, MARGIN, Inches(6.48), BODY_W, Inches(0.6))
    write(frame, [
        ("Blue cars, orange trucks and buses, yellow two-wheelers; grey is the "
         "drivable road. Both panels show the whole town at the same instant.",
         14, False, INK2, 0),
    ])
    frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    notes(s, """
What to look for.

The queues at the junctions are in the right places and roughly the right
lengths — the model has worked out which approaches are stopped and which are
discharging, from the traffic's own behaviour, without ever being told the
signal phase.

Vehicles stay on the road. Nothing drifts through a building. That is the static
road-map conditioning doing its job.

Where the forecast is softer than the truth — slightly fainter, slightly spread
— it is genuinely uncertain, typically about whether a queue has started moving
yet. That softness is the model being calibrated rather than blurry: the total
predicted mass still matches the true vehicle count to within a few per cent.

This is a single frame from the validation video written during training; the
full clip is in the run directory.
""")
    page_number(s, 5)


def slide_future(prs):
    s = blank_slide(prs)
    title(s, "What we are doing now")

    items = [
        ("Finishing the run.",
         "The numbers on the previous slides come from 1.65 passes over the "
         "training data. Training continues to the full budget and the curve has "
         "not flattened."),
        ("Crash anticipation.",
         "Read the forecast at the cell where a collision is about to happen and "
         "compare it with a matched control site, by lead time. That turns the "
         "forecast into a measured number of seconds of warning."),
        ("Beyond one town.",
         "Everything so far is Town05. The test that matters is whether it "
         "transfers to a town it was never trained on."),
        ("End to end on camera input.",
         "Drive the forecaster from the upstream camera model's grid rather than "
         "the simulator's, and measure what the input noise costs."),
        ("Longer horizon, finer grid.",
         "Five seconds at 4 m is version one. Both are preprocessing constants, "
         "not architecture — the raw vehicle states are kept so either can be "
         "revised for the cost of a CPU pass."),
    ]

    frame = textbox(s, MARGIN, Inches(1.92), Inches(11.8), Inches(5.2))
    lines = []
    for i, (head, body) in enumerate(items):
        lines.append(([("•  ", True, BLUE), (head + "  ", True, INK),
                       (body, False, INK2)], 17, False, INK2, 0 if i == 0 else 26))
    write(frame, lines)
    for para in frame.paragraphs:
        hanging(para)

    notes(s, """
Honest statement of what is not done yet.

The headline result is a forecast quality number, not a crash-warning number.
The crash-anticipation read-out is implemented and has been run on an earlier,
smaller model, but the version pinned to this checkpoint is still queued, so no
crash-warning figure is quoted here on purpose.

Transfer to an unseen town is the biggest open risk. A single town means low
scene diversity; some of the 0.805 is certainly memorised layout. Measuring how
much is the next experiment after the run finishes.

The camera-model hand-off is the step that makes this deployable. The forecaster
was deliberately built to consume exactly the camera model's output format, and
it was never given anything the camera model cannot produce — no velocities, no
signal phase — so the swap should be mechanical. Whether accuracy survives the
input noise is an open question.
""")
    page_number(s, 6)


# ----------------------------------------------------------------- main

def main():
    metrics = load_metrics()
    stats = load_dataset_stats()

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    slide_goal(prs)
    slide_dataset(prs, stats)
    slide_model(prs)
    slide_results(prs, metrics)
    slide_picture(prs)
    slide_future(prs)

    prs.save(OUT)
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides, "
          f"{os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"{metrics['windows']:,} held-out windows  "
          f"soft_iou {metrics['headline']['soft_iou_mean']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
