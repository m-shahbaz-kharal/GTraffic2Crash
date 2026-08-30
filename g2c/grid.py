"""Town-scale bird's-eye-view grid geometry and vehicle rasterisation.

Coordinate conventions (inherited from CARLA and from the upstream
UCFTCam2TWorld dataset so that the two are interchangeable):

  world   left-handed, +x east, +y south, +z up; yaw measured about +z in
          degrees, 0 deg pointing along +x, increasing towards +y.
  grid    row = (y - origin_y) / cell,  col = (x - origin_x) / cell.
          Row therefore increases southwards, column eastwards, so an image
          drawn with row as the vertical axis is a conventional north-up map.

Two rasterisations of the same vehicle set are produced, and the difference
between them matters:

  coverage  per class, the fraction of a cell's area covered by that class's
            footprints, clipped into [0, 1].  This is a picture of the traffic:
            a bus is a large solid blob, a motorcycle a faint speck.  It is the
            quantity the forecasting model predicts, and its [0, 1] range is
            what makes a probabilistic loss well posed.
  mass      each vehicle contributes a total of exactly 1.0, split across cells
            in proportion to footprint overlap, so the grid sums to the vehicle
            count.  This is the upstream convention; it is kept for
            count-consistency terms and for compatibility with the camera model
            that will eventually supply the inputs.

Both come from the same supersampling pass, so they are consistent by
construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Vehicle classes, matching the upstream dataset's numbering exactly.
CLASS_IDS = {"car": 10, "truck": 11, "van": 12, "bus": 13, "motorcycle": 14, "bicycle": 15}
CLASS_NAMES = list(CLASS_IDS)                       # index in the channel dimension
CLASS_ID_TO_INDEX = {v: i for i, v in enumerate(CLASS_IDS.values())}
N_CLASSES = len(CLASS_NAMES)

# Drawing colours, matching the upstream palette so figures from the two
# projects can be read side by side.
CLASS_RGB = {
    "car": (0, 120, 255),
    "truck": (255, 120, 0),
    "van": (0, 200, 120),
    "bus": (200, 0, 200),
    "motorcycle": (255, 220, 0),
    "bicycle": (255, 60, 60),
}


@dataclass(frozen=True)
class Grid:
    """An axis-aligned, uniformly spaced bird's-eye-view grid over the town."""

    origin_x: float
    origin_y: float
    cell: float
    rows: int
    cols: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def extent_m(self) -> tuple[float, float]:
        """(north-south, east-west) size of the covered area, in metres."""
        return (self.rows * self.cell, self.cols * self.cell)

    def refine(self, factor: int) -> "Grid":
        """The same footprint at `factor` times the linear resolution."""
        return Grid(self.origin_x, self.origin_y, self.cell / factor,
                    self.rows * factor, self.cols * factor)

    def world_to_rc(self, x, y):
        """Continuous (row, col) for world (x, y).  Cell centres sit at *.5."""
        return ((np.asarray(y) - self.origin_y) / self.cell,
                (np.asarray(x) - self.origin_x) / self.cell)

    def rc_to_world(self, row, col):
        """World (x, y) at the centre of cell (row, col)."""
        return (self.origin_x + (np.asarray(col) + 0.5) * self.cell,
                self.origin_y + (np.asarray(row) + 0.5) * self.cell)

    def as_dict(self) -> dict:
        return {"origin_world_xy": [self.origin_x, self.origin_y], "cell_size_m": self.cell,
                "shape_rows_cols": [self.rows, self.cols]}

    @staticmethod
    def from_dict(d: dict) -> "Grid":
        return Grid(d["origin_world_xy"][0], d["origin_world_xy"][1],
                    d["cell_size_m"], d["shape_rows_cols"][0], d["shape_rows_cols"][1])


# The upstream Town05 grid: 136 x 153 cells of 4 m, i.e. 544 m x 612 m.
TOWN05_4M = Grid(-337.7955061581277, -269.6272170726676, 4.0, 136, 153)


# --------------------------------------------------------------- rasterising --
# The intersection of a rotated vehicle rectangle with an axis-aligned cell is a
# convex polygon, so its area can be computed exactly rather than sampled.  For
# each (vehicle, nearby cell) pair the vertices of that polygon are exactly:
#   * the cell corners lying inside the rectangle,
#   * the rectangle corners lying inside the cell,
#   * the crossings of the cell's edges with the rectangle's four edge lines.
# Those candidates are sorted by angle about their centroid -- valid because the
# intersection is convex -- and the shoelace formula gives the area.  Everything
# is vectorised over (vehicle, cell) pairs.
#
# Exactness matters here more than it might seem.  Town05 is a grid town, so most
# vehicles sit very nearly axis-aligned, which is precisely the configuration in
# which a supersampling rasteriser has its largest error: a footprint edge that
# falls between two samples shifts a whole strip of area into the wrong cell.
# Measured against this exact code, a 64x24 sample lattice was off by up to
# 0.43 m^2 in a 16 m^2 cell.  The exact version is also about ten times faster.

_CAND = 24  # 4 cell corners + 4 rectangle corners + 4 cell edges x 4 edge lines


def _overlap_areas(grid: Grid, cx, cy, yaw_deg, length, width):
    """Exact vehicle-rectangle / cell-square intersection areas, in m^2.

    Returns (row, col, area), each of shape (n_vehicles, n_cells_in_window).
    Rows and columns may fall outside the grid and are filtered by the caller.
    """
    cx = np.asarray(cx, dtype=np.float64); cy = np.asarray(cy, dtype=np.float64)
    L = np.asarray(length, dtype=np.float64); W = np.asarray(width, dtype=np.float64)
    n = len(cx)
    k = int(np.ceil(0.5 * np.hypot(L, W).max() / grid.cell)) + 1
    win = 2 * k + 1
    r0 = np.floor((cy - grid.origin_y) / grid.cell).astype(np.int64) - k
    c0 = np.floor((cx - grid.origin_x) / grid.cell).astype(np.int64) - k
    dr, dc = np.meshgrid(np.arange(win), np.arange(win), indexing="ij")
    rr = r0[:, None] + dr.reshape(-1)[None, :]
    cc = c0[:, None] + dc.reshape(-1)[None, :]
    P = rr.shape[1]

    xlo = grid.origin_x + cc * grid.cell; xhi = xlo + grid.cell
    ylo = grid.origin_y + rr * grid.cell; yhi = ylo + grid.cell
    SX = np.stack([xlo, xhi, xhi, xlo], axis=-1)          # cell corners, ccw
    SY = np.stack([ylo, ylo, yhi, yhi], axis=-1)

    a = np.radians(np.asarray(yaw_deg, dtype=np.float64))[:, None]
    ca, sa = np.cos(a), np.sin(a)
    hL, hW = (L / 2)[:, None], (W / 2)[:, None]
    dx = SX - cx[:, None, None]; dy = SY - cy[:, None, None]
    U = dx * ca[..., None] + dy * sa[..., None]           # cell corners, vehicle frame
    V = -dx * sa[..., None] + dy * ca[..., None]

    cu, cv, ck = [], [], []
    in_rect = (np.abs(U) <= hL[..., None]) & (np.abs(V) <= hW[..., None])
    for j in range(4):
        cu.append(U[..., j]); cv.append(V[..., j]); ck.append(in_rect[..., j])
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        u, v = su * hL, sv * hW
        wx = cx[:, None] + u * ca - v * sa
        wy = cy[:, None] + u * sa + v * ca
        cu.append(np.broadcast_to(u, (n, P)).copy())
        cv.append(np.broadcast_to(v, (n, P)).copy())
        ck.append((wx >= xlo) & (wx <= xhi) & (wy >= ylo) & (wy <= yhi))
    for j in range(4):
        pu, pv = U[..., j], V[..., j]
        qu, qv = U[..., (j + 1) % 4], V[..., (j + 1) % 4]
        for axis, lim in (("u", hL), ("u", -hL), ("v", hW), ("v", -hW)):
            if axis == "u":
                den = qu - pu; live = np.abs(den) > 1e-12
                t = np.where(live, (lim - pu) / np.where(live, den, 1.0), -1.0)
                iu = np.broadcast_to(lim, (n, P)); iv = pv + t * (qv - pv)
                good = live & (t >= 0) & (t <= 1) & (np.abs(iv) <= hW)
            else:
                den = qv - pv; live = np.abs(den) > 1e-12
                t = np.where(live, (lim - pv) / np.where(live, den, 1.0), -1.0)
                iv = np.broadcast_to(lim, (n, P)); iu = pu + t * (qu - pu)
                good = live & (t >= 0) & (t <= 1) & (np.abs(iu) <= hL)
            cu.append(np.broadcast_to(iu, (n, P)).copy())
            cv.append(np.broadcast_to(iv, (n, P)).copy())
            ck.append(good)

    CU = np.stack(cu, -1); CV = np.stack(cv, -1); OK = np.stack(ck, -1)
    cnt = OK.sum(-1)
    m = OK.astype(np.float64)
    mu = (CU * m).sum(-1) / np.maximum(cnt, 1)
    mv = (CV * m).sum(-1) / np.maximum(cnt, 1)
    ang = np.where(OK, np.arctan2(CV - mv[..., None], CU - mu[..., None]), np.inf)
    o = np.argsort(ang, -1)
    pu_ = np.take_along_axis(CU, o, -1); pv_ = np.take_along_axis(CV, o, -1)
    ok_ = np.take_along_axis(OK, o, -1)
    # Unused slots collapse onto the first vertex; the degenerate edges they
    # create contribute nothing to the shoelace sum.
    pu_ = np.where(ok_, pu_, pu_[..., :1]); pv_ = np.where(ok_, pv_, pv_[..., :1])
    area = 0.5 * np.abs((pu_ * np.roll(pv_, -1, -1) - np.roll(pu_, -1, -1) * pv_).sum(-1))
    return rr, cc, np.where(cnt >= 3, area, 0.0)


def rasterise(grid: Grid, cx, cy, yaw_deg, length, width, class_index,
              vx=None, vy=None, n_classes: int = N_CLASSES):
    """Rasterise a set of vehicles into per-class coverage, mass and flow grids.

    Every argument but `grid` is a 1-D array over vehicles.  `class_index` holds
    channel indices in [0, n_classes).  `vx`, `vy` are world-frame velocities in
    m/s; when given, a flow grid is returned alongside.

    Returns a dict with
        coverage (n_classes, rows, cols) float32, in [0, 1]
        mass     (rows, cols)            float32, summing to the vehicle count
        flow     (2, rows, cols)         float32, area-weighted mean velocity
                                         (only when vx/vy are supplied)
    """
    R, C = grid.rows, grid.cols
    cell_area = grid.cell * grid.cell
    n = len(cx)
    out = {"coverage": np.zeros((n_classes, R, C), np.float32),
           "mass": np.zeros((R, C), np.float32)}
    if vx is not None:
        out["flow"] = np.zeros((2, R, C), np.float32)
    if n == 0:
        return out

    rr, cc, area = _overlap_areas(grid, cx, cy, yaw_deg, length, width)
    keep = (rr >= 0) & (rr < R) & (cc >= 0) & (cc < C) & (area > 0)
    flat = np.where(keep, rr * C + cc, 0)
    area = np.where(keep, area, 0.0)

    veh_area = np.asarray(length, np.float64) * np.asarray(width, np.float64)
    ci = np.asarray(class_index, dtype=np.int64)

    mass = np.bincount(flat.reshape(-1), weights=(area / veh_area[:, None]).reshape(-1),
                       minlength=R * C)
    chan = (ci[:, None] * (R * C) + flat).reshape(-1)
    cov = np.bincount(chan, weights=(area / cell_area).reshape(-1),
                      minlength=n_classes * R * C)

    out["coverage"] = np.clip(cov.reshape(n_classes, R, C), 0.0, 1.0).astype(np.float32)
    out["mass"] = mass.reshape(R, C).astype(np.float32)

    if vx is not None:
        den = np.bincount(flat.reshape(-1), weights=area.reshape(-1), minlength=R * C)
        fx = np.bincount(flat.reshape(-1),
                         weights=(area * np.asarray(vx, np.float64)[:, None]).reshape(-1),
                         minlength=R * C)
        fy = np.bincount(flat.reshape(-1),
                         weights=(area * np.asarray(vy, np.float64)[:, None]).reshape(-1),
                         minlength=R * C)
        safe = np.where(den > 1e-12, den, 1.0)
        flow = np.stack([fx / safe, fy / safe]).reshape(2, R, C)
        out["flow"] = np.where(den.reshape(1, R, C) > 1e-12, flow, 0.0).astype(np.float32)
    return out


def occupancy_from_coverage(coverage: np.ndarray) -> np.ndarray:
    """Collapse per-class coverage to a single occupancy probability in [0, 1]."""
    return np.clip(coverage.sum(axis=-3), 0.0, 1.0)
