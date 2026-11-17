"""Stage 6 sampling (PREREGISTRATION section 11): dynesty nested sampling, several independent
runs per model, between-run R-hat, and an emcee follow-up run started from the nested posterior
for the autocorrelation time tau.

The likelihood is the same closure-only likelihood as in stages 2-5 (fitting.Problem).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import emcee
import numpy as np

from .fitting import Problem, circ_stats, n_walkers

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FITS6 = RES / "stage6_fits"      # one JSON summary per run (resumable)
CKPT6 = RES / "stage6_ckpt"      # dynesty checkpoints (gitignored)
CHAINS6 = RES / "chains6"        # equal-weight posterior samples (gitignored)


class Like:
    """Picklable log-likelihood -chi2/2 for one point."""

    def __init__(self, prob: Problem):
        self.prob = prob

    def __call__(self, theta):
        return float(-0.5 * self.prob.chi2(np.asarray(theta))[0])


class PriorTransform:
    """Unit cube -> parameters. Uniform priors within the bounds, with two exceptions:

    * Gaussian x positions (shared parameters x1..xn) are sorted, which gives the uniform prior on
      the ordered region x1 <= x2 <= ... (normalised; removes label switching).
    * For models with an m=2 term, (beta, beta2) of each group are folded into the triangle
      beta + beta2 <= 0.5 (uniform on the triangle, normalised; non-negative brightness).
    """

    def __init__(self, prob: Problem):
        self.lo, self.hi = prob.lo.copy(), prob.hi.copy()
        xs = [i for i, n in enumerate(prob.names) if re.fullmatch(r"x\d", n)]
        self.order = sorted(xs, key=lambda i: prob.names[i]) if len(xs) > 1 else []
        self.folds = []
        for g in sorted(set(prob.groups)):
            if ("beta2", g) in prob.slot:
                self.folds.append((prob.slot[("beta", g)], prob.slot[("beta2", g)]))

    def __call__(self, u):
        u = np.array(u, float)
        for i, j in self.folds:
            if u[i] + u[j] > 1.0:
                u[i], u[j] = 1.0 - u[i], 1.0 - u[j]
        x = self.lo + u * (self.hi - self.lo)
        if self.order:
            x[self.order] = np.sort(x[self.order])
        return x

    def batch(self, U: np.ndarray) -> np.ndarray:
        """Vectorised version for an (n, ndim) array of unit-cube points (same mapping as __call__)."""
        U = np.array(U, float, ndmin=2)
        for i, j in self.folds:
            f = U[:, i] + U[:, j] > 1.0
            U[f, i], U[f, j] = 1.0 - U[f, i], 1.0 - U[f, j]
        X = self.lo + U * (self.hi - self.lo)
        if self.order:
            X[:, self.order] = np.sort(X[:, self.order], axis=1)
        return X


def summarise(prob: Problem, samples: np.ndarray, logl_max: float, best: np.ndarray) -> dict:
    summ = {}
    for i, name in enumerate(prob.names):
        if prob.periodic[i]:
            m, lo, hi = circ_stats(samples[:, i])
        else:
            q16, m, q84 = np.percentile(samples[:, i], [16, 50, 84])
            lo, hi = m - q16, q84 - m
        summ[name] = {"median": float(m), "minus": float(lo), "plus": float(hi)}
    chi2_min = -2.0 * logl_max
    k, n = prob.ndim, prob.n_data
    c_cp, c_la = prob.chi2_parts(best)
    return {"model": prob.model, "names": prob.names, "summary": summ, "best": best.tolist(),
            "chi2_min": chi2_min, "chi2_cp": float(c_cp[0]), "chi2_lca": float(c_la[0]),
            "n_data": n, "k": k, "nu": n - k, "chi2_nu": chi2_min / (n - k) if n > k else float("nan"),
            "bic": chi2_min + k * np.log(n)}


def run_nested(tag: str, prob: Problem, seed: int, nlive: int, njobs: int = 4, dlogz: float = 0.1,
               force: bool = False, derived: dict | None = None, fits: Path = FITS6, ckpts: Path = CKPT6,
               chains: Path = CHAINS6, slices: int | None = None) -> dict:
    """One dynesty run (static, multi-ellipsoid bounds, random slice sampling), checkpointed."""
    import dynesty
    from dynesty.pool import Pool

    fits.mkdir(parents=True, exist_ok=True); ckpts.mkdir(parents=True, exist_ok=True)
    chains.mkdir(parents=True, exist_ok=True)
    out_path = fits / f"{tag}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    ckpt = ckpts / f"{tag}.save"
    like, pt = Like(prob), PriorTransform(prob)
    periodic = [int(i) for i in np.flatnonzero(prob.periodic)] or None
    t0 = time.time()
    with Pool(njobs, like, pt) as pool:
        if ckpt.exists() and not force:
            sampler = dynesty.NestedSampler.restore(str(ckpt), pool=pool)
            sampler.run_nested(dlogz=dlogz, checkpoint_file=str(ckpt), checkpoint_every=120,
                               resume=True, print_progress=False)
        else:
            sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, prob.ndim, nlive=nlive,
                                            bound="multi", sample="rslice", periodic=periodic,
                                            rstate=np.random.default_rng(seed), pool=pool,
                                            queue_size=njobs, **({"slices": slices} if slices else {}))
            sampler.run_nested(dlogz=dlogz, checkpoint_file=str(ckpt), checkpoint_every=120,
                               print_progress=False)
    res = sampler.results
    samples = res.samples_equal(rstate=np.random.default_rng(seed + 1))
    imax = int(np.argmax(res.logl))
    out = summarise(prob, samples, float(res.logl[imax]), np.asarray(res.samples[imax]))
    out.update({"tag": tag, "seed": int(seed), "nlive": int(nlive), "dlogz": dlogz,
                "logz": float(res.logz[-1]), "logzerr": float(res.logzerr[-1]),
                "information": float(res.information[-1]), "niter": int(res.niter),
                "ncall": int(np.sum(res.ncall)), "n_samples_equal": int(len(samples)),
                "runtime_s": time.time() - t0, "sampler": f"dynesty {dynesty.__version__} rslice multi",
                "slices": slices if slices else 3 + prob.ndim})
    if derived:
        out["derived"] = {}
        for name, fn in derived.items():
            x = fn(samples, prob.names)
            q16, m, q84 = np.percentile(x, [16, 50, 84])
            out["derived"][name] = {"median": float(m), "minus": float(m - q16), "plus": float(q84 - m)}
    np.savez_compressed(chains / f"{tag}.npz", samples=samples, names=np.array(prob.names))
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[nested] {tag}: logZ={out['logz']:.2f}±{out['logzerr']:.2f} chi2nu={out['chi2_nu']:.3f} "
          f"k={out['k']} ncall={out['ncall']} {out['runtime_s']:.0f}s", flush=True)
    return out


def load_samples(tag: str, chains: Path = CHAINS6) -> np.ndarray:
    return np.load(chains / f"{tag}.npz")["samples"]


def _unwrap(x: np.ndarray, period: float) -> np.ndarray:
    a = 2 * np.pi * x / period
    c = np.arctan2(np.sin(a).mean(), np.cos(a).mean()) * period / (2 * np.pi)
    return c + (x - c + period / 2) % period - period / 2


def rhat(chains: list[np.ndarray], periodic: np.ndarray | None = None, periods=None) -> np.ndarray:
    """Gelman-Rubin R-hat treating each independent run's equal-weight samples as one chain.

    chains: list of (n_i, ndim). Periodic parameters are unwrapped around the pooled circular mean.
    """
    n = min(len(c) for c in chains)
    rng = np.random.default_rng(0)
    X = np.stack([c[rng.choice(len(c), n, replace=False)] for c in chains])  # (m, n, ndim)
    if periodic is not None:
        for i in np.flatnonzero(periodic):
            per = periods[i] if periods is not None else 360.0
            X[:, :, i] = _unwrap(X[:, :, i].ravel(), per).reshape(X.shape[:2])
    m = X.shape[0]
    means = X.mean(1)
    W = X.var(1, ddof=1).mean(0)
    B = n * means.var(0, ddof=1)
    var = (n - 1) / n * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(var / W)
    return np.where(W > 0, r, np.nan) if m > 1 else np.full(X.shape[2], np.nan)


def periods_of(prob: Problem) -> np.ndarray:
    return np.array([180.0 if re.fullmatch(r"th\d", n) else 360.0 for n in prob.names])


def emcee_check(tag: str, prob: Problem, samples: np.ndarray, seed: int, nburn: int = 2000,
                nprod: int = 8000, force: bool = False, fits: Path = FITS6, thin_by: int = 1) -> dict:
    """emcee run started from draws of the nested posterior; tau < nprod/50 is the criterion.

    thin_by > 1 stores every thin_by-th step only (memory); tau is converted back to steps.
    """
    fits.mkdir(parents=True, exist_ok=True)
    out_path = fits / f"{tag}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    rng = np.random.default_rng(seed)
    nw = n_walkers(prob.ndim)
    p0 = samples[rng.choice(len(samples), nw, replace=len(samples) < nw)]
    t0 = time.time()
    sampler = emcee.EnsembleSampler(nw, prob.ndim, prob.log_prob, vectorize=True)
    sampler.random_state = np.random.RandomState(int(seed) % (2**32)).get_state()
    state = sampler.run_mcmc(p0, nburn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, nprod // thin_by, progress=False, thin_by=thin_by)
    tau = sampler.get_autocorr_time(quiet=True) * thin_by  # stored steps -> steps
    chain = sampler.get_chain(flat=True)
    lp = sampler.get_log_prob(flat=True)
    best = chain[int(np.argmax(lp))]
    out = summarise(prob, chain, float(lp.max()), best)
    out.update({"tag": tag, "tau": [float(t) for t in tau], "tau_max": float(np.nanmax(tau)),
                "tau_limit": nprod / 50, "converged_tau": bool(np.all(np.isfinite(tau)) and np.all(tau < nprod / 50)),
                "nwalkers": nw, "nburn": nburn, "nprod": nprod, "thin_by": thin_by, "acceptance": float(np.mean(sampler.acceptance_fraction)),
                "runtime_s": time.time() - t0, "seed": int(seed)})
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[emcee] {tag}: tau_max={out['tau_max']:.0f} (limit {nprod / 50:.0f}) {out['runtime_s']:.0f}s", flush=True)
    return out
