"""Robustness variant (PREREGISTRATION 4.9a): closure quantities from 10-s data,
averaged within each scan (closure phase: weighted circular mean, log closure
amplitude: weighted mean). Model points are the scan-averaged uv positions.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .closure import ClosureSet, _greedy_independent, closure_phases, log_closure_amplitudes
from .dataset import Dataset
from .uvdata import VisTable, add_systematic, assign_scans, debiased_amp, scan_average


def _tenseq_prepared(raw: VisTable) -> VisTable:
    t = add_systematic(raw)
    keep = (t.snr >= 2.0) & ~t.is_intrasite()
    amp = debiased_amp(t)
    t.vis = amp * np.exp(1j * np.angle(t.vis))
    t = t.select(keep)
    t.scan = None
    return t


def _bl_index(tab: VisTable):
    m = {}
    for i, (s, a, b) in enumerate(zip(tab.scan, tab.t1, tab.t2)):
        m[(s, a, b)] = (i, 1)
        m[(s, b, a)] = (i, -1)
    return m


def incoherent_dataset(ds: Dataset) -> Dataset:
    raw = ds.raw
    scan_of_time = dict(zip(raw.time, assign_scans(raw.time)))
    avg = scan_average(raw)
    avg = avg.select(~avg.is_intrasite())
    bl = _bl_index(avg)
    t10 = _tenseq_prepared(raw)
    cp10, lca10 = closure_phases(t10), log_closure_amplitudes(t10)

    def group(cs: ClosureSet, kind: str):
        acc = defaultdict(list)
        for lab, tt, v, s in zip(cs.label, cs.time, cs.obs, cs.sigma):
            acc[(scan_of_time[tt], lab)].append((v, s))
        idx, sgn, obs, sig, tim, labs = [], [], [], [], [], []
        for (sc, lab), vals in acc.items():
            v = np.array([x[0] for x in vals]); s = np.array([x[1] for x in vals])
            w = 1 / s**2
            if kind == "cp":
                a, b, c = lab.split("-")
                edges = [(a, b), (b, c), (c, a)]
                ids = [bl.get((sc, x, y)) for x, y in edges]
                if any(e is None for e in ids):
                    continue
                o = np.angle(np.sum(w * np.exp(1j * v)))
                idx.append([e[0] for e in ids]); sgn.append([e[1] for e in ids])
            else:
                num, den = lab.split("/")
                bls = [tuple(p.split("-")) for p in num.split("*") + den.split("*")]
                ids = [bl.get((sc, x, y)) for x, y in bls]
                if any(e is None for e in ids):
                    continue
                o = np.sum(w * v) / np.sum(w)
                idx.append([e[0] for e in ids]); sgn.append([1, 1, -1, -1])
            obs.append(o); sig.append(1 / np.sqrt(w.sum())); tim.append(sc); labs.append(lab)
        k = 3 if kind == "cp" else 4
        # keep a minimal (non-redundant) subset per scan, lowest sigma first
        keep = []
        tim_a = np.array(tim)
        for sc in np.unique(tim_a):
            rows = np.flatnonzero(tim_a == sc)
            vecs = []
            for r in rows:
                vec = np.zeros(len(avg))
                for i_, s_ in zip(idx[r], sgn[r]):
                    vec[i_] += s_
                vecs.append(vec)
            keep += [rows[j] for j in _greedy_independent(vecs, np.array(sig)[rows])]
        keep = sorted(keep)
        idx = [idx[i] for i in keep]; sgn = [sgn[i] for i in keep]
        obs = [obs[i] for i in keep]; sig = [sig[i] for i in keep]
        tim = [tim[i] for i in keep]; labs = [labs[i] for i in keep]
        return ClosureSet(np.array(idx, int).reshape(-1, k), np.array(sgn, float).reshape(-1, k),
                          np.array(obs), np.array(sig), np.array(tim, float), labs)

    return Dataset(ds.name + " (10s closures)", ds.doy, ds.band, raw, avg, group(cp10, "cp"), group(lca10, "lca"))
