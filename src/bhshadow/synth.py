"""Synthetic data for injection-recovery (PREREGISTRATION 7.4).

Synthetic 10-s visibilities on the real uv points with the real thermal sigma,
station gains (amplitude lognormal 10 %, LMT 30 %; phase uniform; constant within
a scan), then the same processing as the real data.
"""
from __future__ import annotations

import zlib

import numpy as np

from . import models as M
from .uvdata import VisTable, assign_scans

FLUX = 0.6  # Jy, total compact flux of every truth model

# name -> (callable(u, v) -> unit-flux visibility, reference diameter or None, kind)
TRUTHS = {
    "ring38": (lambda u, v: M.blurred_ring(u, v, 38.0, 12.0), 38.0, "ring"),
    "ring42": (lambda u, v: M.blurred_ring(u, v, 42.0, 12.0), 42.0, "ring"),
    "ring46": (lambda u, v: M.blurred_ring(u, v, 46.0, 12.0), 46.0, "ring"),
    "cres38": (lambda u, v: M.mring1(u, v, 38.0, 12.0, 0.25, 200.0), 38.0, "ring"),
    "cres42": (lambda u, v: M.mring1(u, v, 42.0, 12.0, 0.25, 200.0), 42.0, "ring"),
    "cres46": (lambda u, v: M.mring1(u, v, 46.0, 12.0, 0.25, 200.0), 46.0, "ring"),
    "xscres43": (lambda u, v: M.offset_crescent(u, v, 23.5, 19.5, 2.5, 200.0, 10.0), 43.0, "ring"),
    "disk40": (lambda u, v: M.disk(u, v, 40.0, 10.0), None, "disk"),
    "disk50": (lambda u, v: M.disk(u, v, 50.0, 10.0), None, "disk"),
    "gauss40": (lambda u, v: M.gaussian(u, v, 40.0), None, "gaussian"),
}


def seed_for(*keys) -> int:
    return zlib.crc32("|".join(map(str, keys)).encode())


def corrupt(raw: VisTable, vis_model: np.ndarray, seed: int, gain_amp=0.10, gain_amp_lmt=0.30,
            thermal=True, gains=True) -> VisTable:
    rng = np.random.default_rng(seed)
    scan = assign_scans(raw.time)
    stations = sorted(set(raw.t1) | set(raw.t2))
    g = {}
    for s in np.unique(scan):
        for st in stations:
            sa = gain_amp_lmt if st == "LM" else gain_amp
            g[(s, st)] = rng.lognormal(0.0, sa) * np.exp(1j * rng.uniform(0, 2 * np.pi)) if gains else 1.0
    gi = np.array([g[(s, a)] for s, a in zip(scan, raw.t1)])
    gj = np.array([g[(s, b)] for s, b in zip(scan, raw.t2)])
    vis = gi * np.conj(gj) * vis_model
    if thermal:
        vis = vis + raw.sigma * (rng.standard_normal(len(vis)) + 1j * rng.standard_normal(len(vis)))
    meta = dict(raw.meta)
    meta.update({"synthetic": True, "seed": int(seed)})
    return VisTable(raw.time.copy(), raw.t1.copy(), raw.t2.copy(), raw.u.copy(), raw.v.copy(), vis,
                    raw.sigma.copy(), meta=meta)


def make(raw: VisTable, truth: str, rep: int, coverage: str, **kw) -> VisTable:
    fn = TRUTHS[truth][0]
    return corrupt(raw, FLUX * fn(raw.u, raw.v), seed_for(truth, coverage, rep), **kw)
