"""Figures for the written report.

Plain on purpose: square boxes, grey fills, one navy accent, and Times-metric
type so the figures sit inside the document rather than beside it.

Two things are automatic, because doing them by hand produces figures with an
empty half.  Boxes size themselves to their text, and each sheet crops itself
to whatever was actually drawn on it.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "doc")
os.makedirs(OUT, exist_ok=True)

for _f in ("NimbusRoman-Regular.otf", "NimbusRoman-Bold.otf"):
    _p = f"/usr/share/fonts/urw-base35/{_f}"
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)

INK = "#1a1a1a"
ACCENT = "#1f3864"
GREY1 = "#565656"
GREY2 = "#8c8c8c"
GREY3 = "#c8c8c8"
LIGHT = "#f1f1f1"
TINT = "#dde4ee"
EDGE = "#3f3f3f"

plt.rcParams.update({
    "font.family": "Nimbus Roman",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": GREY1,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

UNITS_PER_INCH = 8.0
TOP = 100.0


class Sheet:
    """A drawing measured in units, where 8 units is one inch of paper."""

    def __init__(self, w_in, x_units=100.0):
        self.w_in = w_in
        self.uy = UNITS_PER_INCH
        self.fig, self.ax = plt.subplots(figsize=(w_in, 8.0))
        self.ax.set_xlim(0, x_units)
        self.ax.set_ylim(0, TOP)
        self.ax.axis("off")
        self.lo, self.hi = TOP, 0.0

    def pt(self, points):
        return points / 72.0 * self.uy

    def mark(self, y0, y1):
        self.lo = min(self.lo, y0, y1)
        self.hi = max(self.hi, y0, y1)

    def height_for(self, title, lines, tsize=8.0, ssize=6.8, pad_pt=6.0):
        tl = title.split("\n")
        return (2 * self.pt(pad_pt)
                + len(tl) * self.pt(tsize * 1.25)
                + (self.pt(ssize * 0.6) if lines else 0.0)
                + len(lines) * self.pt(ssize * 1.32))

    def block(self, x, y, w, h, title, lines=(), fill=LIGHT, edge=EDGE,
              tsize=8.0, ssize=6.8, tcolor=INK, scolor=GREY1, lw=0.9):
        tl = title.split("\n")
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                                    linewidth=lw, zorder=2))
        total = self.height_for(title, lines, tsize, ssize, pad_pt=0.0)
        cursor = y + h / 2 + total / 2
        for line in tl:
            cursor -= self.pt(tsize * 1.25)
            self.ax.text(x + w / 2, cursor + self.pt(tsize * 0.38), line,
                         ha="center", va="center", fontsize=tsize,
                         fontweight="bold", color=tcolor, zorder=3)
        if lines:
            cursor -= self.pt(ssize * 0.6)
        for line in lines:
            cursor -= self.pt(ssize * 1.32)
            self.ax.text(x + w / 2, cursor + self.pt(ssize * 0.42), line,
                         ha="center", va="center", fontsize=ssize,
                         color=scolor, zorder=3)
        self.mark(y, y + h)

    def arrow(self, x0, y0, x1, y1, color=EDGE, lw=1.2, rad=0.0, scale=11):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                          mutation_scale=scale, linewidth=lw,
                                          color=color, zorder=1,
                                          connectionstyle=f"arc3,rad={rad}"))
        self.mark(min(y0, y1), max(y0, y1))

    def line(self, xs, ys, color=EDGE, lw=1.2):
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=1,
                     solid_capstyle="butt")
        self.mark(min(ys), max(ys))

    def note(self, x, y, text, size=6.8, color=GREY1, ha="center", va="top"):
        self.ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
                     zorder=3)
        self.mark(y - self.pt(size * 1.3), y + self.pt(size * 0.4))

    def save(self, name, pad=0.7):
        y0, y1 = self.lo - pad, self.hi + pad
        self.ax.set_ylim(y0, y1)
        self.fig.set_size_inches(self.w_in, (y1 - y0) / self.uy)
        self.fig.subplots_adjust(left=0.004, right=0.996, top=0.997, bottom=0.003)
        self.fig.savefig(os.path.join(OUT, name), dpi=220)
        plt.close(self.fig)
        print(name)


# ------------------------------------------------------------ 1. data pipeline
def fig_pipeline():
    s = Sheet(6.4)
    boxes = [
        (0.5, 17.2, "Simulate", ["CARLA Town05, 20 Hz", "100 to 260 vehicles",
                                 "450 seconds per episode"]),
        (20.7, 17.6, "Record", ["every tenth tick, so 0.5 s",
                                "position, heading, size,",
                                "speed, class, behaviour"]),
        (41.3, 17.6, "Rasterise", ["exact area of every vehicle",
                                   "inside every 4 m cell",
                                   "three vehicle classes"]),
        (61.9, 16.0, "Stack", ["one array per episode",
                               "144 by 160 cells",
                               "stored sparsely"]),
        (80.9, 18.6, "Cut windows", ["ten past frames and ten future",
                                     "frames, sliding forward",
                                     "one frame at a time"]),
    ]
    h = max(s.height_for(t, l) for _, _, t, l in boxes)
    y = TOP - h
    for x, w, t, lines in boxes:
        s.block(x, y, w, h, t, lines)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][1]
        s.arrow(x0 + 0.6, y + h / 2, boxes[i + 1][0] - 0.6, y + h / 2, scale=7)

    sm = ["road mask, lane direction, junction mask, speed limit"]
    sh = s.height_for("Static town map, built once", sm, 7.6)
    s.block(46.0, y - sh - 3.4, 33.0, sh, "Static town map, built once", sm,
            fill=TINT, tsize=7.6)
    s.arrow(66.0, y - 3.4, 70.0, y - 0.4, rad=-0.2, scale=7)
    s.save("fig_pipeline.png")


# ---------------------------------------------------------- 2. soft coverage
def fig_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.0))
    panels = [(7.0, "A car centred 7.0 m east", (0.386, 0.148)),
              (9.0, "The same car 2 m further east", (0.148, 0.386))]
    for ax, (cx, label, vals) in zip(axes, panels):
        ax.set_xlim(-0.3, 16.3); ax.set_ylim(-0.3, 12.3)
        ax.set_aspect("equal"); ax.axis("off")
        for gx in range(0, 20, 4):
            ax.plot([gx, gx], [0, 12], color=GREY3, lw=0.8, zorder=1)
        for gy in range(0, 16, 4):
            ax.plot([0, 16], [gy, gy], color=GREY3, lw=0.8, zorder=1)
        ax.add_patch(Rectangle((cx - 2.25, 5.05), 4.5, 1.9, facecolor=TINT,
                               edgecolor=ACCENT, linewidth=1.2, alpha=0.75,
                               zorder=2))
        for col in range(4):
            for row in range(3):
                v = vals[col - 1] if (row == 1 and col in (1, 2)) else 0.0
                ax.text(col * 4 + 0.3, row * 4 + 0.25,
                        f"{v:.3f}" if v else "0", ha="left", va="bottom",
                        fontsize=7.6, fontweight="bold" if v else "normal",
                        color=ACCENT if v else GREY2, zorder=4)
        ax.set_title(label, fontsize=8.5, color=INK, pad=6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.08)
    fig.savefig(os.path.join(OUT, "fig_coverage.png"), dpi=220)
    plt.close(fig)
    print("fig_coverage.png")


# ------------------------------------------------------- 3. training diagram
def fig_training():
    s = Sheet(6.5)
    gl = ["read and updated by every block,", "so any part of the town",
          "can reach any other"]
    gh = s.height_for("8 global vectors", gl, 7.6)
    gy = TOP - gh
    s.block(37.0, gy, 26.0, gh, "8 global vectors", gl, fill="#ffffff", tsize=7.6)

    chain = [
        (0.0, 14.0, "Input", ["10 past frames", "3 vehicle channels",
                              "6 map channels", "144 by 160 cells"], LIGHT),
        (16.9, 13.0, "Convolution\nstem", ["reads each frame", "at full size, then",
                                           "steps down to a", "36 by 40 grid"], LIGHT),
        (32.8, 16.0, "Axial encoder", ["12 blocks", "attention over time,",
                                       "then rows,", "then columns"], TINT),
        (51.7, 16.0, "Axial decoder", ["8 blocks", "all 10 future frames",
                                       "produced in one pass"], TINT),
        (70.6, 13.0, "Convolution\nhead", ["back up to", "144 by 160 cells"], LIGHT),
        (86.5, 13.5, "Output", ["10 future frames", "3 vehicle channels",
                                "2 speed channels"], LIGHT),
    ]
    h = max(s.height_for(t, l) for _, _, t, l, _ in chain)
    y = gy - 4.4 - h
    for x, w, t, lines, fill in chain:
        s.block(x, y, w, h, t, lines, fill=fill)
    for i in range(len(chain) - 1):
        x0 = chain[i][0] + chain[i][1]
        s.arrow(x0 + 0.6, y + h / 2, chain[i + 1][0] - 0.6, y + h / 2, scale=7)
    s.arrow(44.5, gy - 0.3, 42.5, y + h + 0.4, color=GREY1, lw=0.8, scale=6)
    s.arrow(55.5, gy - 0.3, 58.5, y + h + 0.4, color=GREY1, lw=0.8, scale=6)

    s.ax.annotate("", xy=(77.5, y), xytext=(23.5, y),
                  arrowprops=dict(arrowstyle="-|>", color=GREY1, lw=0.8,
                                  mutation_scale=7,
                                  connectionstyle="arc3,rad=0.13"))
    s.note(50.0, y - 3.4, "the last observed frame, kept at full resolution")

    losses = [
        (0.0, 22.8, "Occupancy loss", ["is the right fraction of",
                                       "each cell covered?", "weight 1.0"]),
        (25.7, 22.8, "Speed loss", ["is the speed in each",
                                    "occupied cell right?", "weight 0.5"]),
        (51.4, 22.8, "Movement loss", ["does the traffic slide along",
                                       "the speeds it predicted?", "weight 0.5"]),
        (77.2, 22.8, "Count loss", ["is the total number of",
                                    "vehicles right?", "weight 0.001"]),
    ]
    lh = max(s.height_for(t, l, 7.6) for _, _, t, l in losses)
    bus = y - 6.6
    ly = bus - 2.8 - lh
    s.line([88.6, 88.6], [y, bus], lw=0.9)
    s.line([11.4, 88.6], [bus, bus], lw=0.9)
    for x, w, t, lines in losses:
        s.block(x, ly, w, lh, t, lines, fill="#ffffff", tsize=7.6, tcolor=ACCENT)
        s.arrow(x + w / 2, bus, x + w / 2, ly + lh + 0.4, scale=6)
    for x, w, _, _ in losses[:-1]:
        s.ax.text(x + w + 1.45, ly + lh / 2, "+", ha="center", va="center",
                  fontsize=10, color=GREY1)
    s.save("fig_training.png")


# ------------------------------------------------------------- 4. the result
def fig_horizon():
    ev = json.load(open(os.path.join(ROOT, "runs", "base_v1", "evaluation",
                                     "evaluation.json")))
    bh = ev["by_horizon"]
    dt = ev.get("frame_dt_s", 0.5)
    H = [(i + 1) * dt for i in range(len(bh["model"]["soft_iou"]))]

    fig, ax = plt.subplots(figsize=(5.9, 3.2))
    series = [
        ("Our model", bh["model"]["soft_iou"], ACCENT, "-", 1.8, 4.2, 0),
        ("Constant speed,\ntrue speeds given",
         bh["constant_velocity_oracle_flow"]["soft_iou"], GREY1, "--", 1.2, 0, 16),
        ("Repeat the last frame",
         bh["persistence"]["soft_iou"], GREY1, "-.", 1.2, 0, -9),
        ("Constant speed,\nspeeds estimated",
         bh["constant_velocity_estimated_flow"]["soft_iou"], GREY2, ":", 1.5, 0, -5),
    ]
    for name, vals, c, ls, lw, ms, dy in series:
        ax.plot(H, vals, color=c, ls=ls, lw=lw, marker="o" if ms else None,
                markersize=ms, markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=0.9, clip_on=False, zorder=3 if ms else 2)
        ax.annotate(name, xy=(H[-1], vals[-1]), xytext=(10, dy),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=7.6, color=c,
                    fontweight="bold" if ms else "normal", linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.7,
                                    shrinkA=0, shrinkB=3) if abs(dy) > 7 else None)
    ax.set_xlim(0.4, 5.1); ax.set_ylim(0.2, 1.0)
    ax.set_xticks([0.5, 1, 2, 3, 4, 5])
    ax.set_xlabel("how far ahead the forecast reaches (seconds)", labelpad=5,
                  fontsize=8.5)
    ax.set_ylabel("agreement with what happened", labelpad=5, fontsize=8.5)
    ax.tick_params(labelsize=8.0)
    ax.grid(axis="y", color="#e8e8e8", lw=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.115, right=0.735, top=0.97, bottom=0.155)
    fig.savefig(os.path.join(OUT, "fig_horizon.png"), dpi=220)
    plt.close(fig)
    print("fig_horizon.png")


def main():
    fig_pipeline()
    fig_coverage()
    fig_training()
    fig_horizon()
    fig_forecast()


# -------------------------------------------------------- 5. the forecast pair
def fig_forecast():
    """Re-tone the dark validation panel for a printed page.

    Neutral pixels (background, road) are inverted to light greys; coloured
    pixels keep their hue but move to a single dark value so the vehicles read
    on white.  Nothing is moved, added or removed.
    """
    import numpy as np
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
    from PIL import Image

    src = np.array(Image.open(os.path.join(ROOT, "docs", "slides",
                                           "fig_forecast.png")).convert("RGB"))
    src = src.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(src)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Colour is judged by absolute chroma, not saturation: the near-black
    # background is (14, 15, 20), which is a *saturated* dark blue by ratio but
    # carries almost no chroma, and must stay neutral.
    chroma = src.max(axis=2) - src.min(axis=2)
    neutral = np.repeat((1.0 - 0.80 * val)[..., None], 3, axis=2)
    vehicle = hsv_to_rgb(np.stack([h, np.full_like(sat, 0.85),
                                   np.full_like(val, 0.52)], axis=-1))
    a = np.clip((chroma - 0.10) / 0.20, 0.0, 1.0)[..., None]
    out = (1 - a) * neutral + a * vehicle

    rows = slice(55, 484)
    left = out[rows, 16:508]
    right = out[rows, 554:1046]

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))
    for ax, panel, title in ((axes[0], left, "What actually happened"),
                             (axes[1], right, "What the model predicted")):
        ax.imshow(np.clip(panel, 0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GREY3); sp.set_linewidth(0.8); sp.set_visible(True)
        ax.set_title(title, fontsize=9.0, color=INK, pad=5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05)
    fig.savefig(os.path.join(OUT, "fig_forecast.png"), dpi=260)
    plt.close(fig)
    print("fig_forecast.png")


if __name__ == "__main__":
    main()
