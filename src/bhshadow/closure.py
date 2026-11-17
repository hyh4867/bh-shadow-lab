"""Closure phases and log closure amplitudes with a greedy SNR-ordered minimal set.

Each closure quantity is stored as indices into the visibility table plus signs,
so a model closure quantity is a signed sum of model phases (or log amplitudes).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from . import INTRA_SITE
from .uvdata import VisTable


@dataclass
class ClosureSet:
    """idx: (N, k) visibility indices, sgn: (N, k) signs (+1/-1), obs, sigma, time, label."""

    idx: np.ndarray
    sgn: np.ndarray
    obs: np.ndarray
    sigma: np.ndarray
    time: np.ndarray
    label: list

    def __len__(self) -> int:
        return len(self.obs)


def _greedy_independent(vectors: list[np.ndarray], sigmas: np.ndarray) -> list[int]:
    """Indices of a maximal linearly independent subset, lowest sigma first."""
    chosen: list[int] = []
    basis = np.zeros((0, len(vectors[0]))) if vectors else None
    rank = 0
    for i in np.argsort(sigmas, kind="stable"):
        trial = np.vstack([basis, vectors[i]])
        r = np.linalg.matrix_rank(trial)
        if r > rank:
            basis, rank = trial, r
            chosen.append(int(i))
    return chosen


def _scan_groups(tab: VisTable):
    key = tab.scan if tab.scan is not None else tab.time
    for s in np.unique(key):
        yield np.flatnonzero(key == s)


def closure_phases(tab: VisTable) -> ClosureSet:
    """psi_ijk = phi_ij + phi_jk + phi_ki (radians); sigma^2 = sum (sigma/A)^2."""
    amp, ph = np.abs(tab.vis), np.angle(tab.vis)
    rel = tab.sigma / np.where(amp > 0, amp, np.inf)
    idx_all, sgn_all, lab, tt = [], [], [], []
    for rows in _scan_groups(tab):
        bl = {}
        for r in rows:
            if amp[r] > 0:
                bl[(tab.t1[r], tab.t2[r])] = r
        stations = sorted({s for k in bl for s in k})
        blist = sorted(bl)
        bpos = {b: i for i, b in enumerate(blist)}

        def edge(a, b):
            if (a, b) in bl:
                return bl[(a, b)], 1, (a, b)
            if (b, a) in bl:
                return bl[(b, a)], -1, (b, a)
            return None

        cands, vecs, sigs = [], [], []
        for a, b, c in combinations(stations, 3):
            es = [edge(a, b), edge(b, c), edge(c, a)]
            if any(e is None for e in es):
                continue
            vec = np.zeros(len(blist))
            for r, s, key in es:
                vec[bpos[key]] += s
            cands.append(([e[0] for e in es], [e[1] for e in es], f"{a}-{b}-{c}"))
            vecs.append(vec)
            sigs.append(np.sqrt(sum(rel[e[0]] ** 2 for e in es)))
        if not cands:
            continue
        for i in _greedy_independent(vecs, np.array(sigs)):
            idx_all.append(cands[i][0]); sgn_all.append(cands[i][1]); lab.append(cands[i][2])
            tt.append(tab.time[rows].mean())
    idx = np.array(idx_all, int).reshape(-1, 3)
    sgn = np.array(sgn_all, float).reshape(-1, 3)
    obs = np.angle(np.exp(1j * (ph[idx] * sgn).sum(1)))
    sig = np.sqrt((rel[idx] ** 2).sum(1))
    return ClosureSet(idx, sgn, obs, sig, np.array(tt), lab)


def log_closure_amplitudes(tab: VisTable, drop_colocated: bool = False) -> ClosureSet:
    """ln(A_ij A_kl / (A_ik A_jl)); sigma^2 = sum (sigma/A)^2.

    drop_colocated (stage 6, PREREGISTRATION 11.4): leave out every quadrangle whose four stations
    include both members of a co-located pair (INTRA_SITE). Such quadrangles are close to 0 for any
    source structure. The greedy minimal set is then chosen from the remaining candidates.
    """
    amp = np.abs(tab.vis)
    rel = tab.sigma / np.where(amp > 0, amp, np.inf)
    idx_all, sgn_all, lab, tt = [], [], [], []
    for rows in _scan_groups(tab):
        bl = {}
        for r in rows:
            if amp[r] > 0:
                bl[frozenset((tab.t1[r], tab.t2[r]))] = r
        stations = sorted({s for k in bl for s in k})
        blist = sorted(bl, key=sorted)
        bpos = {b: i for i, b in enumerate(blist)}
        cands, vecs, sigs = [], [], []
        for a, b, c, d in combinations(stations, 4):
            if drop_colocated and any(pr <= {a, b, c, d} for pr in INTRA_SITE):
                continue
            pairs = [(frozenset((a, b)), frozenset((c, d))),
                     (frozenset((a, c)), frozenset((b, d))),
                     (frozenset((a, d)), frozenset((b, c)))]
            for (n1, n2), (d1, d2) in [(pairs[0], pairs[1]), (pairs[0], pairs[2]), (pairs[1], pairs[2])]:
                if not all(x in bl for x in (n1, n2, d1, d2)):
                    continue
                vec = np.zeros(len(blist))
                vec[bpos[n1]] += 1; vec[bpos[n2]] += 1; vec[bpos[d1]] -= 1; vec[bpos[d2]] -= 1
                rows4 = [bl[n1], bl[n2], bl[d1], bl[d2]]
                cands.append((rows4, [1, 1, -1, -1],
                              f"{'-'.join(sorted(n1))}*{'-'.join(sorted(n2))}/{'-'.join(sorted(d1))}*{'-'.join(sorted(d2))}"))
                vecs.append(vec)
                sigs.append(np.sqrt(sum(rel[r] ** 2 for r in rows4)))
        if not cands:
            continue
        for i in _greedy_independent(vecs, np.array(sigs)):
            idx_all.append(cands[i][0]); sgn_all.append(cands[i][1]); lab.append(cands[i][2])
            tt.append(tab.time[rows].mean())
    idx = np.array(idx_all, int).reshape(-1, 4)
    sgn = np.array(sgn_all, float).reshape(-1, 4)
    obs = (np.log(amp[idx]) * sgn).sum(1)
    sig = np.sqrt((rel[idx] ** 2).sum(1))
    return ClosureSet(idx, sgn, obs, sig, np.array(tt), lab)


def model_closure_phase(V: np.ndarray, cp: ClosureSet) -> np.ndarray:
    """V: (..., Nvis) complex model visibilities -> (..., Ncp) closure phases."""
    return (np.angle(V)[..., cp.idx] * cp.sgn).sum(-1)


def model_log_camp(V: np.ndarray, lca: ClosureSet) -> np.ndarray:
    return (np.log(np.abs(V) + 1e-300)[..., lca.idx] * lca.sgn).sum(-1)
