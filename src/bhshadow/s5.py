"""Shared helpers for the stage 5 analyses (PREREGISTRATION section 10)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .fitting import Problem, n_walkers, run_mcmc
from .uvdata import VisTable, assign_scans

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FITS = RES / "stage5_fits"  # one small JSON per fit (summary only), so long runs can resume
CHAINS = RES / "chains"
SEED = 20170411

SHARED = {  # joint fits: parameters common to all data sets
    "M3": ("d", "w"),
    "E1": ("d", "w", "fg", "xg", "yg", "fwhmg"),
    "E2": ("d", "w"),
}


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def sym(s: dict) -> float:
    return 0.5 * (s["minus"] + s["plus"])


def day_groups(dsets) -> list[int]:
    """Contiguous group index per data set: one group per observing day."""
    days = []
    for d in dsets:
        if d.doy not in days:
            days.append(d.doy)
    return [days.index(d.doy) for d in dsets]


def joint_problem(model: str, dsets) -> Problem:
    return Problem(model, dsets, groups=day_groups(dsets), shared=SHARED[model])


def cached_fit(tag: str, prob: Problem, seed: int, save_chain: bool = False, force: bool = False,
               derived: dict | None = None) -> dict:
    """Run the pre-registered sampler (walkers per PREREGISTRATION 10) or load a cached summary.

    derived: name -> function(chain, names) -> samples; median and 16/84 % are stored in out["derived"].
    """
    FITS.mkdir(parents=True, exist_ok=True)
    path = FITS / f"{tag}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    t0 = time.time()
    r = run_mcmc(prob, seed, nwalkers=n_walkers(prob.ndim))
    r["runtime_s"] = time.time() - t0
    r["tag"] = tag
    if derived:
        r["derived"] = {}
        for name, fn in derived.items():
            x = fn(r["_chain"], prob.names)
            q16, m, q84 = np.percentile(x, [16, 50, 84])
            r["derived"][name] = {"median": float(m), "minus": float(m - q16), "plus": float(q84 - m)}
    if save_chain:
        CHAINS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CHAINS / f"s5_{tag}.npz", chain=r["_chain"], names=np.array(prob.names))
    out = strip(r)
    path.write_text(json.dumps(out, indent=1))
    print(f"[fit] {tag}: chi2nu={out['chi2_nu']:.3f} k={out['k']} conv={out['converged']} "
          f"retried={out['retried']} {out['runtime_s']:.0f}s", flush=True)
    return out


def d_of(fit: dict) -> tuple[float, float]:
    """(median, symmetric 1-sigma) of the diameter parameter d."""
    s = fit["summary"]["d"]
    return s["median"], sym(s)


# ---------------------------------------------------------------------------
# Synthetic data with extra corruptions (PREREGISTRATION 10.3 and 10.7)
# ---------------------------------------------------------------------------

def corrupt_ext(raw: VisTable, vis_model: np.ndarray, seed: int, gain_amp: float = 0.10,
                gain_amp_lmt: float = 0.30, rw_step: float = 0.0, rw_lmt_only: bool = False,
                amp10_lmt: float = 0.0, nonclosing_amp: float = 0.0, nonclosing_phase_deg: float = 0.0,
                thermal: bool = True) -> VisTable:
    """Station gains per scan (as stage 3) plus optional:

    rw_step       phase random walk per station, N(0, rw_step) rad per 10-s point, from 0 at scan start
    rw_lmt_only   apply the random walk to LMT only
    amp10_lmt     extra LMT amplitude gain, lognormal per 10-s point
    nonclosing_*  baseline-based complex factor per baseline and scan (amp lognormal, phase normal)
    """
    rng = np.random.default_rng(seed)
    scan = assign_scans(raw.time)
    stations = sorted(set(raw.t1) | set(raw.t2))
    g = {}
    for s in np.unique(scan):
        for st in stations:
            sa = gain_amp_lmt if st == "LM" else gain_amp
            g[(s, st)] = rng.lognormal(0.0, sa) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    gi = np.array([g[(s, a)] for s, a in zip(scan, raw.t1)])
    gj = np.array([g[(s, b)] for s, b in zip(scan, raw.t2)])
    # per-10-s station terms
    ut = np.unique(raw.time)
    tindex = {t: i for i, t in enumerate(ut)}
    scan_of_t = assign_scans(ut)
    term = {st: np.ones(len(ut), complex) for st in stations}
    for st in stations:
        if rw_step > 0 and (not rw_lmt_only or st == "LM"):
            ph = np.zeros(len(ut))
            for s in np.unique(scan_of_t):
                idx = np.flatnonzero(scan_of_t == s)
                steps = rng.normal(0.0, rw_step, len(idx))
                steps[0] = 0.0
                ph[idx] = np.cumsum(steps)
            term[st] = term[st] * np.exp(1j * ph)
        if amp10_lmt > 0 and st == "LM":
            term[st] = term[st] * rng.lognormal(0.0, amp10_lmt, len(ut))
    ti = np.array([tindex[t] for t in raw.time])
    hi = np.array([term[a][k] for a, k in zip(raw.t1, ti)])
    hj = np.array([term[b][k] for b, k in zip(raw.t2, ti)])
    vis = gi * np.conj(gj) * hi * np.conj(hj) * vis_model
    if nonclosing_amp > 0 or nonclosing_phase_deg > 0:
        nc = {}
        for s, a, b in zip(scan, raw.t1, raw.t2):
            if (s, a, b) not in nc:
                nc[(s, a, b)] = rng.lognormal(0.0, nonclosing_amp) * np.exp(
                    1j * np.deg2rad(rng.normal(0.0, nonclosing_phase_deg)))
        vis = vis * np.array([nc[(s, a, b)] for s, a, b in zip(scan, raw.t1, raw.t2)])
    if thermal:
        vis = vis + raw.sigma * (rng.standard_normal(len(vis)) + 1j * rng.standard_normal(len(vis)))
    meta = dict(raw.meta)
    meta.update({"synthetic": True, "seed": int(seed)})
    return VisTable(raw.time.copy(), raw.t1.copy(), raw.t2.copy(), raw.u.copy(), raw.v.copy(), vis,
                    raw.sigma.copy(), meta=meta)


def xs_dhat(chain, names):
    """Paper VI eq. 20: d_hat = R_out + R_in = R_out (2 - psi)."""
    return chain[:, names.index("r_out")] * (2 - chain[:, names.index("psi")])


def xs_fw(chain, names):
    """Paper VI eq. 21: f_w = (R_out - R_in + 2 sqrt(2 ln 2) sigma) / d_hat."""
    ro, ps, sg = (chain[:, names.index(k)] for k in ("r_out", "psi", "sigma"))
    return (ro * ps + 2 * np.sqrt(2 * np.log(2)) * sg) / (ro * (2 - ps))
