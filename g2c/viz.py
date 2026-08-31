"""Pictures of the data and of what the model does with it.

Everything here writes plain RGB arrays rather than going through a plotting
library, because these are rendered per frame for videos of hundreds of frames
and matplotlib is far too slow for that.
"""
from __future__ import annotations

import os

import numpy as np

from .grid import Grid
from .tensorize import GROUP_NAMES, GROUP_RGB

BACKGROUND = (14, 16, 20)
ROAD = (52, 56, 64)
GRID_LINE = (26, 29, 35)


def group_colours() -> np.ndarray:
    return np.array([GROUP_RGB[g] for g in GROUP_NAMES], dtype=np.float32)


def render_occupancy(coverage, road=None, scale=5, flow=None, flow_every=4,
                     marks=None, gamma=0.65):
    """One frame of the town as an RGB image.

    `coverage` is (n_groups, rows, cols) in [0, 1]; each group is drawn in its own
    colour with the coverage value as opacity, so a lightly covered cell reads as
    a faint tint and a fully covered one as solid.  `gamma` lifts small coverage
    values into visibility -- without it a motorcycle occupying 6 % of a cell is
    invisible next to a bus.
    """
    C, H, W = coverage.shape
    img = np.zeros((H, W, 3), np.float32)
    img[:] = np.array(BACKGROUND, np.float32)
    if road is not None:
        img[road > 0] = np.array(ROAD, np.float32)

    cols = group_colours()
    alpha = np.clip(coverage, 0, 1) ** gamma
    # Painter's order: the largest vehicles last, so a bus is not hidden by a car.
    for g in (2, 0, 1):
        a = alpha[g][..., None]
        img = img * (1 - a) + cols[g][None, None, :] * a

    img = np.repeat(np.repeat(img, scale, 0), scale, 1)

    if scale >= 4:
        img[::scale, :, :] = img[::scale, :, :] * 0.82 + np.array(GRID_LINE, np.float32) * 0.18
        img[:, ::scale, :] = img[:, ::scale, :] * 0.82 + np.array(GRID_LINE, np.float32) * 0.18

    if flow is not None:
        img = _draw_flow(img, coverage.sum(0), flow, scale, flow_every)
    if marks:
        for (r, c, colour, radius) in marks:
            img = _draw_ring(img, r * scale + scale // 2, c * scale + scale // 2,
                             radius * scale, colour)
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_flow(img, occ, flow, scale, every):
    """Short line segments showing where each occupied cell is heading."""
    H, W = occ.shape
    rr, cc = np.nonzero(occ > 0.05)
    keep = ((rr % every == 0) & (cc % every == 0)) | (occ[rr, cc] > 0.5)
    rr, cc = rr[keep], cc[keep]
    for r, c in zip(rr, cc):
        vx, vy = flow[0, r, c], flow[1, r, c]
        n = np.hypot(vx, vy)
        if n < 1e-3:
            continue
        y0, x0 = r * scale + scale // 2, c * scale + scale // 2
        # Flow is in grid units per frame after scaling; draw a fixed-length
        # segment whose direction is the heading and whose length grows with speed.
        length = min(3.0 * scale, 0.8 * scale + 2.0 * scale * min(n, 1.5))
        y1, x1 = y0 + vy / n * length, x0 + vx / n * length
        for t in np.linspace(0, 1, int(length) + 2):
            y, x = int(y0 + (y1 - y0) * t), int(x0 + (x1 - x0) * t)
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                img[y, x] = img[y, x] * 0.25 + np.array([235, 245, 255], np.float32) * 0.75
    return img


def _draw_ring(img, cy, cx, radius, colour):
    th = np.linspace(0, 2 * np.pi, 180)
    for rad in (radius, radius - 1):
        ys = (cy + rad * np.sin(th)).astype(int)
        xs = (cx + rad * np.cos(th)).astype(int)
        ok = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
        img[ys[ok], xs[ok]] = np.array(colour, np.float32)
    return img


# ------------------------------------------------------------------ captions --
_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        from PIL import ImageFont
        for path in ("/usr/share/fonts/dejavu/DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(path):
                _FONT = ImageFont.truetype(path, 15)
                break
        else:
            _FONT = ImageFont.load_default()
    return _FONT


def caption(img, lines, xy=(8, 6), colour=(235, 240, 250)):
    """Burn a few lines of text into an RGB array."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = _font()
    y = xy[1]
    for line in (lines if isinstance(lines, (list, tuple)) else [lines]):
        d.text((xy[0] + 1, y + 1), line, font=f, fill=(0, 0, 0))
        d.text((xy[0], y), line, font=f, fill=colour)
        y += 18
    return np.array(im)


def legend(img, entries, xy=(8, None)):
    """Colour key: [(label, rgb), ...]."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = _font()
    x = xy[0]
    y = (img.shape[0] - 24 * len(entries) - 8) if xy[1] is None else xy[1]
    for label, rgb in entries:
        d.rectangle([x, y + 3, x + 12, y + 15], fill=tuple(int(v) for v in rgb))
        d.text((x + 19, y + 1), label, font=f, fill=(0, 0, 0))
        d.text((x + 18, y), label, font=f, fill=(230, 235, 245))
        y += 22
    return np.array(im)


def hstack(images, gap=10, colour=BACKGROUND):
    h = max(i.shape[0] for i in images)
    parts = []
    for k, im in enumerate(images):
        if im.shape[0] < h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)
            pad[:] = colour
            im = np.concatenate([im, pad], 0)
        parts.append(im)
        if k < len(images) - 1:
            sep = np.zeros((h, gap, 3), np.uint8)
            sep[:] = colour
            parts.append(sep)
    return np.concatenate(parts, 1)


def vstack(images, gap=10, colour=BACKGROUND):
    w = max(i.shape[1] for i in images)
    parts = []
    for k, im in enumerate(images):
        if im.shape[1] < w:
            pad = np.zeros((im.shape[0], w - im.shape[1], 3), np.uint8)
            pad[:] = colour
            im = np.concatenate([im, pad], 1)
        parts.append(im)
        if k < len(images) - 1:
            sep = np.zeros((gap, w, 3), np.uint8)
            sep[:] = colour
            parts.append(sep)
    return np.concatenate(parts, 0)


def write_video(path, frames, fps=6):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Even dimensions are required by the H.264 encoder.
    fixed = []
    for f in frames:
        h, w = f.shape[:2]
        fixed.append(f[:h - h % 2, :w - w % 2])
    imageio.mimwrite(path, fixed, fps=fps, codec="libx264",
                     quality=8, macro_block_size=1)
    return path


def write_png(path, img):
    from PIL import Image
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(img).save(path)
    return path
