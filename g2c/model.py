"""Axial space-time transformer for town-scale occupancy forecasting.

The shape of this network follows two measured results rather than taste.

Earthformer (NeurIPS 2022) ablates seven ways of factorising attention over a
space-time grid under identical conditions; attending to one axis at a time
(time, then height, then width) plus a handful of global vectors wins on every
metric on both of its benchmarks *and* is among the cheapest.  Joint space-time
cuboids and Video-Swin windows are strictly dominated.

The OpenSTL benchmark (NeurIPS 2023) shows that on low-frequency grid data --
traffic flow and weather, the closest public analogues to this problem --
predicting every future frame in one parallel pass matches or beats recurrent
rollout at a fraction of the compute, and avoids autoregressive drift entirely.
So there is no recurrence here: the decoder emits all future frames at once.

The convolutional stem and head are not incidental.  Attention over 8 m patches
cannot place a vehicle to sub-cell precision; the convolutional decoder is what
turns a coarse token field back into a sharp raster.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    rows: int = 144
    cols: int = 160
    n_groups: int = 3            # occupancy channels (car / large / two-wheeler)
    n_static: int = 6            # road, lane sin/cos, coherence, junction, limit
    t_in: int = 10
    t_out: int = 10
    patch: int = 4               # cells per token side; 4 cells = 16 m
    dim: int = 640
    heads: int = 10
    enc_depth: int = 12
    dec_depth: int = 8
    n_global: int = 8            # Earthformer's value
    mlp_ratio: float = 4.0
    stem_width: int = 128
    dropout: float = 0.0
    predict_flow: bool = True
    # Predict the change from the last observed frame rather than the frame
    # itself.  Persistence is a strong baseline on a sparse grid, so making it
    # the zero of the output parameterisation means the network spends its
    # capacity on what actually moves.  T3Former and IR-WM both report large
    # gains from exactly this reformulation.
    residual_from_last: bool = True
    occupancy_prior: list = field(default_factory=lambda: [0.015, 0.004, 0.002])
    checkpointing: bool = True

    @property
    def th(self) -> int:
        return self.rows // self.patch

    @property
    def tw(self) -> int:
        return self.cols // self.patch


class Mlp(nn.Module):
    def __init__(self, dim, ratio, dropout):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Attention(nn.Module):
    """Multi-head attention with an optional extra set of keys/values.

    The extra keys are the global vectors: every token attends to its own axis
    *and* to a small shared summary of the whole scene, which restores the
    long-range mixing that pure axial attention gives up.
    """

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = dropout

    def forward(self, x, context=None, extra=None):
        B, N, D = x.shape
        src = x if context is None else context
        if extra is not None:
            src = torch.cat([src, extra], dim=1)
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(src).view(B, src.shape[1], 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


def _to_axis(x, axis):
    """Fold a (B, T, H, W, D) tensor so that `axis` becomes the sequence dim."""
    B, T, H, W, D = x.shape
    if axis == "t":
        return x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D), (B, T, H, W, D)
    if axis == "h":
        return x.permute(0, 1, 3, 2, 4).reshape(B * T * W, H, D), (B, T, H, W, D)
    return x.reshape(B * T * H, W, D), (B, T, H, W, D)


def _from_axis(y, axis, shape):
    B, T, H, W, D = shape
    if axis == "t":
        return y.view(B, H, W, T, D).permute(0, 3, 1, 2, 4)
    if axis == "h":
        return y.view(B, T, W, H, D).permute(0, 1, 3, 2, 4)
    return y.view(B, T, H, W, D)


class AxialBlock(nn.Module):
    """One pass of attention along time, then height, then width.

    Each axial attention is followed by its own feed-forward block, in pre-norm
    form.  Global vectors are read by every axis and updated once per block.
    """

    def __init__(self, cfg: ModelConfig, cross: bool = False):
        super().__init__()
        d, h = cfg.dim, cfg.heads
        self.axes = ("t", "h", "w")
        self.norm_a = nn.ModuleList(nn.LayerNorm(d) for _ in self.axes)
        self.attn = nn.ModuleList(Attention(d, h, cfg.dropout) for _ in self.axes)
        self.norm_m = nn.ModuleList(nn.LayerNorm(d) for _ in self.axes)
        self.mlp = nn.ModuleList(Mlp(d, cfg.mlp_ratio, cfg.dropout) for _ in self.axes)
        self.n_global = cfg.n_global
        if cfg.n_global:
            self.norm_g = nn.LayerNorm(d)
            self.norm_gx = nn.LayerNorm(d)
            self.attn_g = Attention(d, h, cfg.dropout)
            self.norm_gm = nn.LayerNorm(d)
            self.mlp_g = Mlp(d, cfg.mlp_ratio, cfg.dropout)
        self.cross = cross
        if cross:
            self.norm_c = nn.LayerNorm(d)
            self.norm_ctx = nn.LayerNorm(d)
            self.attn_c = Attention(d, h, cfg.dropout)
            self.norm_cm = nn.LayerNorm(d)
            self.mlp_c = Mlp(d, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x, g=None, memory=None, update_globals=True):
        B = x.shape[0]
        for i, axis in enumerate(self.axes):
            h = self.norm_a[i](x)
            seq, shape = _to_axis(h, axis)
            extra = None
            if g is not None:
                rep = seq.shape[0] // B
                extra = g.repeat_interleave(rep, dim=0)
            seq = self.attn[i](seq, extra=extra)
            x = x + _from_axis(seq, axis, shape)
            x = x + self.mlp[i](self.norm_m[i](x))

        if g is not None and update_globals:
            flat = self.norm_gx(x).reshape(B, -1, x.shape[-1])
            g = g + self.attn_g(self.norm_g(g), context=flat)
            g = g + self.mlp_g(self.norm_gm(g))

        if self.cross and memory is not None:
            B_, T, H, W, D = x.shape
            q = self.norm_c(x).reshape(B_, T * H * W, D)
            kv = self.norm_ctx(memory).reshape(B_, -1, D)
            x = x + self.attn_c(q, context=kv).view(B_, T, H, W, D)
            x = x + self.mlp_c(self.norm_cm(x))
        return x, g


class ConvStem(nn.Module):
    """Per-frame convolutional encoder, from the full grid down to tokens.

    Returns both the token grid and the full-resolution features, because the
    head needs the latter: a 16 m token cannot say which 4 m cell a car is in,
    but the stem saw the cell.
    """

    def __init__(self, in_ch, width, dim, patch):
        super().__init__()
        assert patch in (1, 2, 4, 8)
        self.fine = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU())
        layers, ch = [], width
        for _ in range(int(math.log2(patch))):
            layers += [nn.Conv2d(ch, min(ch * 2, dim), 3, stride=2, padding=1),
                       nn.GroupNorm(8, min(ch * 2, dim)), nn.SiLU()]
            ch = min(ch * 2, dim)
        layers += [nn.Conv2d(ch, dim, 1)]
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        fine = self.fine(x)
        return self.down(fine), fine


class ConvHead(nn.Module):
    """Token grid back to full-resolution logits and flow.

    The skip carries the last *observed* frame's full-resolution features into
    every predicted frame.  That gives the head a sharp reference of where things
    are now to transport from, instead of asking it to hallucinate 4 m detail out
    of 16 m tokens.  Future frames have no stem features of their own, so the
    last observed frame is the only sharp thing available -- and it is exactly
    the right thing, since forecasting is mostly transport.
    """

    def __init__(self, dim, width, patch, n_groups, predict_flow):
        super().__init__()
        layers, ch = [], dim
        for _ in range(int(math.log2(patch))):
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(ch, max(ch // 2, width), 3, padding=1),
                       nn.GroupNorm(8, max(ch // 2, width)), nn.SiLU()]
            ch = max(ch // 2, width)
        self.up = nn.Sequential(*layers)

        # Occupancy and flow get their own fusion bodies rather than sharing one.
        # Sharing costs nothing in parameters but a great deal in optimisation:
        # with a shared body the flow terms slowed occupancy learning by 3x
        # (a 47 % drop in occupancy cross-entropy over 120 steps became 16.5 %),
        # because the flow gradient dominates the features both heads read from.
        # The transformer trunk is still shared, which is where the useful
        # multi-task transfer actually happens; two extra 3x3 convolutions are a
        # cheap price for letting each head shape its own last few features.
        def fuse():
            return nn.Sequential(
                nn.Conv2d(ch + width, width, 3, padding=1),
                nn.GroupNorm(8, width), nn.SiLU(),
                nn.Conv2d(width, width, 3, padding=1),
                nn.GroupNorm(8, width), nn.SiLU())

        self.fuse_occupancy = fuse()
        self.occupancy = nn.Conv2d(width, n_groups, 1)
        self.fuse_flow = fuse() if predict_flow else None
        self.flow = nn.Conv2d(width, 2, 1) if predict_flow else None

    def forward(self, x, skip):
        h = torch.cat([self.up(x), skip], dim=1)
        occ = self.occupancy(self.fuse_occupancy(h))
        flow = self.flow(self.fuse_flow(h)) if self.flow is not None else None
        return occ, flow


class OccupancyForecaster(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = cfg.n_groups + cfg.n_static
        self.stem = ConvStem(in_ch, cfg.stem_width, cfg.dim, cfg.patch)
        self.head = ConvHead(cfg.dim, cfg.stem_width, cfg.patch, cfg.n_groups,
                             cfg.predict_flow)

        self.pos_space = nn.Parameter(torch.zeros(1, 1, cfg.th, cfg.tw, cfg.dim))
        self.pos_in_time = nn.Parameter(torch.zeros(1, cfg.t_in, 1, 1, cfg.dim))
        self.pos_out_time = nn.Parameter(torch.zeros(1, cfg.t_out, 1, 1, cfg.dim))
        nn.init.trunc_normal_(self.pos_space, std=0.02)
        nn.init.trunc_normal_(self.pos_in_time, std=0.02)
        nn.init.trunc_normal_(self.pos_out_time, std=0.02)

        self.globals = nn.Parameter(torch.zeros(1, cfg.n_global, cfg.dim)) \
            if cfg.n_global else None
        if self.globals is not None:
            nn.init.trunc_normal_(self.globals, std=0.02)

        self.encoder = nn.ModuleList(AxialBlock(cfg) for _ in range(cfg.enc_depth))
        self.decoder = nn.ModuleList(AxialBlock(cfg, cross=True) for _ in range(cfg.dec_depth))
        self.norm_out = nn.LayerNorm(cfg.dim)

        # At ~1.5 % occupancy a network initialised to predict 0.5 everywhere sees
        # 20,000 empty cells outvote 400 occupied ones and collapses to zero.
        # Lin et al. prescribe exactly this prior initialisation for that regime,
        # and report that training diverged without it.
        with torch.no_grad():
            self.head.occupancy.weight.mul_(0.01)
            if cfg.residual_from_last:
                # The offset already comes from the last observed frame, so the
                # network starts as exact persistence and learns the change.
                self.head.occupancy.bias.zero_()
            else:
                prior = torch.tensor(cfg.occupancy_prior[:cfg.n_groups], dtype=torch.float32)
                self.head.occupancy.bias.copy_(torch.log(prior / (1 - prior)))

    def _run(self, blocks, x, g, memory=None, last_updates_globals=True):
        n = len(blocks)
        for i, blk in enumerate(blocks):
            # Every block *reads* the global vectors -- that is what restores the
            # long-range mixing axial attention gives up.  Only the last block's
            # *update* of them feeds nothing downstream, so that one is skipped.
            upd = last_updates_globals or i < n - 1
            if self.cfg.checkpointing and self.training:
                x, g = checkpoint(blk, x, g, memory, upd, use_reentrant=False)
            else:
                x, g = blk(x, g, memory, upd)
        return x, g

    def forward(self, past, static):
        """past: (B, T_in, n_groups, H, W).  static: (B, n_static, H, W)."""
        cfg = self.cfg
        B, T, C, H, W = past.shape
        s = static.unsqueeze(1).expand(B, T, static.shape[1], H, W)
        x = torch.cat([past, s], dim=2).reshape(B * T, -1, H, W)
        if cfg.checkpointing and self.training:
            tokens, fine = checkpoint(self.stem, x, use_reentrant=False)
        else:
            tokens, fine = self.stem(x)
        # Only the last observed frame's fine features are kept, as the head's
        # sharp reference; the rest would be 10x the memory for no extra use.
        skip = fine.view(B, T, -1, H, W)[:, -1]
        x = tokens.reshape(B, T, cfg.dim, cfg.th, cfg.tw).permute(0, 1, 3, 4, 2)
        x = x + self.pos_space + self.pos_in_time

        g = self.globals.expand(B, -1, -1) if self.globals is not None else None
        memory, g = self._run(self.encoder, x, g)

        # Future frames start from the *last* observed frame rather than a
        # time-average of all of them: that is the state being extrapolated from,
        # and it makes persistence the easy function to represent.  The learned
        # per-step embedding is what distinguishes step 1 from step 10 before any
        # attention runs.
        q = memory[:, -1:].expand(B, cfg.t_out, -1, -1, -1)
        q = q + self.pos_space + self.pos_out_time
        y, g = self._run(self.decoder, q, g, memory, last_updates_globals=False)

        y = self.norm_out(y).permute(0, 1, 4, 2, 3).reshape(B * cfg.t_out, cfg.dim,
                                                            cfg.th, cfg.tw)
        skip_rep = skip.unsqueeze(1).expand(B, cfg.t_out, -1, H, W).reshape(
            B * cfg.t_out, -1, H, W)
        if cfg.checkpointing and self.training:
            occ, flow = checkpoint(self.head, y, skip_rep, use_reentrant=False)
        else:
            occ, flow = self.head(y, skip_rep)
        # Occupancy logits are produced in float32, deliberately, even under
        # autocast.  The residual base is log(1e-5 / (1 - 1e-5)) = -11.5, and
        # bfloat16's resolution at that magnitude is 0.044 -- so in bf16 every
        # correction the network tries to make smaller than that is rounded
        # away, the occupancy term freezes at its initial value, and the model
        # sits at persistence forever while every other loss term trains
        # normally.  That is exactly what it did before this cast.
        occ = occ.float().view(B, cfg.t_out, cfg.n_groups, H, W)
        if cfg.residual_from_last:
            # A tighter clamp keeps the floor this puts under every empty cell
            # negligible: at 1e-5 the spurious mass is 0.6% of a typical frame's
            # total, against 6% at 1e-4.
            last = past[:, -1:].float().clamp(1e-5, 1 - 1e-5)
            occ = occ + (torch.log(last) - torch.log1p(-last))
        flow = flow.float().view(B, cfg.t_out, 2, H, W) if flow is not None else None
        return {"occupancy_logits": occ, "flow": flow}

    @staticmethod
    def preset(name: str, **overrides) -> ModelConfig:
        """Three sizes: `small` for plumbing checks, `base` for the headline run.

        `base` is set by the compute budget, not by taste: at patch 4 it is about
        29 training TFLOPs per sample, which is roughly 2.5 passes over 375,000
        windows on two H100s in fifteen hours.  `large` needs proportionally more
        data or time.
        """
        table = {
            "small": dict(dim=256, heads=8, enc_depth=4, dec_depth=3, stem_width=64),
            "base": dict(dim=640, heads=10, enc_depth=12, dec_depth=8, stem_width=128),
            "large": dict(dim=768, heads=12, enc_depth=16, dec_depth=10, stem_width=160),
        }
        cfg = dict(table[name]); cfg.update(overrides)
        return ModelConfig(**cfg)

    def parameter_count(self):
        total = sum(p.numel() for p in self.parameters())
        parts = {
            "stem": sum(p.numel() for p in self.stem.parameters()),
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }
        parts["embeddings"] = total - sum(parts.values())
        return total, parts
