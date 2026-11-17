"""Closure-only likelihood and emcee sampling (PREREGISTRATION sections 5-6)."""
from __future__ import annotations

from dataclasses import dataclass, field

import emcee
import numpy as np
from scipy.optimize import differential_evolution

from .closure import model_closure_phase, model_log_camp
from .dataset import Dataset
from .models import CONSTRAINTS, MODELS

PERIODIC = {"beta_pa", "beta2_arg", "pa", "th1", "th2", "th3"}


def _closure_chi2(V, d):
    """Closure-phase and log-closure-amplitude chi2 of model visibilities V (..., Nvis) for data set d."""
    c_cp = c_la = 0.0
    if len(d.cp):
        r = d.cp.obs - model_closure_phase(V, d.cp)
        c_cp = (2.0 * (1.0 - np.cos(r)) / d.cp.sigma**2).sum(-1)
    if len(d.lca):
        r = d.lca.obs - model_log_camp(V, d.lca)
        c_la = (r**2 / d.lca.sigma**2).sum(-1)
    return c_cp, c_la


@dataclass
class Problem:
    """A model fitted to one or more data sets.

    For a joint fit, `shared` parameters are common to all data sets and the
    remaining parameters are repeated per group (groups[i] = group of dataset i).
    """

    model: str
    datasets: list[Dataset]
    groups: list[int] | None = None
    shared: tuple[str, ...] = ()
    names: list[str] = field(default_factory=list)
    bounds: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        fn, pnames, pbounds = MODELS[self.model]
        self.fn, self.pnames = fn, pnames
        if self.groups is None:
            self.groups = [0] * len(self.datasets)
        ng = max(self.groups) + 1
        self.slot = {}  # (param, group) -> column
        for p, b in zip(pnames, pbounds):
            if p in self.shared or ng == 1:
                self.slot.update({(p, g): len(self.names) for g in range(ng)})
                self.names.append(p); self.bounds.append(b)
            else:
                for g in range(ng):
                    self.slot[(p, g)] = len(self.names)
                    self.names.append(f"{p}_{g}"); self.bounds.append(b)
        self.lo = np.array([b[0] for b in self.bounds])
        self.hi = np.array([b[1] for b in self.bounds])
        self.periodic = np.array([any(n == p or n.startswith(p + "_") for p in PERIODIC) for n in self.names])
        self.n_data = sum(d.n_data for d in self.datasets)
        self._eg = None

    @property
    def ndim(self) -> int:
        return len(self.names)

    @property
    def _evalgroups(self):
        """Data sets of one group share every parameter, so their model visibilities are computed in one
        call on the concatenated (u, v) (stage 7 speed-up; same numbers as one call per data set)."""
        if self._eg is None:
            self._eg = []
            for g in sorted(set(self.groups)):
                members = [d for d, gg in zip(self.datasets, self.groups) if gg == g]
                offs = np.cumsum([0] + [len(d.tab.u) for d in members])
                self._eg.append((g, np.concatenate([d.tab.u for d in members]),
                                 np.concatenate([d.tab.v for d in members]),
                                 [(d, int(a), int(b)) for d, a, b in zip(members, offs[:-1], offs[1:])]))
        return self._eg

    def chi2_parts(self, theta: np.ndarray):
        """theta (n, ndim) -> chi2_cp (n,), chi2_lca (n,)."""
        theta = np.atleast_2d(theta)
        c_cp = np.zeros(len(theta)); c_la = np.zeros(len(theta))
        for g, U, Vv, members in self._evalgroups:
            args = [theta[:, self.slot[(p, g)]] for p in self.pnames]
            Vall = self.fn(U, Vv, *args)
            for d, a, b in members:
                c1, c2 = _closure_chi2(Vall[..., a:b], d)
                c_cp += c1; c_la += c2
        return c_cp, c_la

    def chi2_parts_per_dataset(self, theta: np.ndarray):
        """Reference path (stages 2-6): one model call per data set. Used in tests."""
        theta = np.atleast_2d(theta)
        c_cp = np.zeros(len(theta)); c_la = np.zeros(len(theta))
        for d, g in zip(self.datasets, self.groups):
            args = [theta[:, self.slot[(p, g)]] for p in self.pnames]
            c1, c2 = _closure_chi2(self.fn(d.tab.u, d.tab.v, *args), d)
            c_cp += c1; c_la += c2
        return c_cp, c_la

    def chi2(self, theta):
        a, b = self.chi2_parts(theta)
        return a + b

    def in_prior(self, theta):
        theta = np.atleast_2d(theta)
        ok = np.ones(len(theta), bool)
        np_ = ~self.periodic
        ok &= np.all((theta[:, np_] >= self.lo[np_]) & (theta[:, np_] <= self.hi[np_]), axis=1)
        # Periodic angles: allowed range [lo - 360, hi + 360] (three copies of the circle).
        # The prior is uniform, so its projection onto the circle is still U(0, 360);
        # the bound only stops walkers drifting to overflow when the angle is unconstrained.
        pp = self.periodic
        if pp.any():
            ok &= np.all((theta[:, pp] >= self.lo[pp] - 360) & (theta[:, pp] <= self.hi[pp] + 360), axis=1)
        if self.model in CONSTRAINTS:
            for g in sorted(set(self.groups)):
                pdict = {p: theta[:, self.slot[(p, g)]] for p in self.pnames}
                ok &= CONSTRAINTS[self.model](pdict)
        return ok

    def log_prob(self, theta):
        theta = np.atleast_2d(theta).copy()
        lp = np.full(len(theta), -np.inf)
        ok = self.in_prior(theta)
        if ok.any():
            lp[ok] = -0.5 * self.chi2(theta[ok])
        return lp


def n_walkers(ndim: int) -> int:
    """PREREGISTRATION 10: max(32, 2k + 4), rounded up to an even number."""
    n = max(32, 2 * ndim + 4)
    return n + (n % 2)


def optimise(prob: Problem, seed: int, maxiter: int = 300) -> tuple[np.ndarray, float]:
    def f(t):
        if not prob.in_prior(t)[0]:
            return 1e30
        return float(prob.chi2(t)[0])

    sol = differential_evolution(f, list(zip(prob.lo, prob.hi)),
                                 seed=seed, maxiter=maxiter, tol=1e-7, polish=True,
                                 vectorized=False, updating="immediate")
    return sol.x, float(sol.fun)


def circ_stats(x_deg: np.ndarray):
    a = np.deg2rad(x_deg)
    mean = np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360
    dev = (x_deg - mean + 180) % 360 - 180
    lo, med, hi = np.percentile(dev, [16, 50, 84])
    return (mean + med) % 360, med - lo, hi - med


def run_mcmc(prob: Problem, seed: int, nwalkers: int = 32, nburn: int = 2000, nprod: int = 4000,
             x0: np.ndarray | None = None, retry: bool = True) -> dict:
    """Pre-registered sampler: emcee, 32 walkers, 2000 burn-in, 4000 production.

    Walkers start in a small ball around the differential-evolution optimum.
    If tau > nprod/50 the run is repeated once with double length.
    """
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0, _ = optimise(prob, seed)
    scale = np.maximum(1e-3 * (prob.hi - prob.lo), 1e-4)
    p0 = x0 + scale * rng.standard_normal((nwalkers, prob.ndim))
    np_ = ~prob.periodic
    p0[:, np_] = np.clip(p0[:, np_], prob.lo[np_] + 1e-9, prob.hi[np_] - 1e-9)
    # walkers that violate an extra constraint (e.g. E2 positivity) are pulled back towards x0
    for _ in range(50):
        bad = ~prob.in_prior(p0)
        if not bad.any():
            break
        p0[bad] = x0 + 0.5 * (p0[bad] - x0)
    sampler = emcee.EnsembleSampler(nwalkers, prob.ndim, prob.log_prob, vectorize=True)
    sampler.random_state = np.random.RandomState(int(seed) % (2**32)).get_state()
    state = sampler.run_mcmc(p0, nburn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, nprod, progress=False)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:  # noqa: BLE001
        tau = np.full(prob.ndim, np.nan)
    converged = bool(np.all(np.isfinite(tau)) and np.all(tau < nprod / 50))
    if not converged and retry:
        out = run_mcmc(prob, seed, nwalkers, 2 * nburn, 2 * nprod, x0=x0, retry=False)
        out["retried"] = True
        return out
    chain = sampler.get_chain(flat=True)
    lp = sampler.get_log_prob(flat=True)
    best = chain[np.argmax(lp)]
    chi2_min = float(min(-2 * lp.max(), prob.chi2(x0)[0]))
    if prob.chi2(x0)[0] < -2 * lp.max():
        best = x0
    k, n = prob.ndim, prob.n_data
    summ = {}
    for i, name in enumerate(prob.names):
        if prob.periodic[i]:
            m, lo, hi = circ_stats(chain[:, i])
        else:
            q16, m, q84 = np.percentile(chain[:, i], [16, 50, 84])
            lo, hi = m - q16, q84 - m
        summ[name] = {"median": float(m), "minus": float(lo), "plus": float(hi)}
    nu = n - k
    c_cp, c_la = prob.chi2_parts(best)
    return {
        "model": prob.model, "names": prob.names, "summary": summ, "best": best.tolist(),
        "chi2_min": chi2_min, "chi2_cp": float(c_cp[0]), "chi2_lca": float(c_la[0]),
        "n_data": n, "k": k, "nu": nu,
        "chi2_nu": chi2_min / nu if nu > 0 else np.nan,
        "bic": chi2_min + k * np.log(n), "tau": [float(t) for t in tau],
        "converged": converged, "retried": False, "nprod": nprod, "nburn": nburn,
        "acceptance": float(np.mean(sampler.acceptance_fraction)), "seed": int(seed),
        "_chain": chain,
    }
