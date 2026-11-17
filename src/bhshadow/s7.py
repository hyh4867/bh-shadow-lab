"""Shared helpers for the stage 7 analyses (PREREGISTRATION section 12): convergence of the large models.

Same likelihood, priors and dynesty settings as stage 6 (section 11.1) except:
* nlive = c * k (c = 50, or 150 in the G1 escalation step, 12.2),
* tau test: C3' (32000 steps, tau < 640) and, only if that fails, C3'' (128000 steps, tau < 2560),
* C4 mode check against the differential-evolution mode map (12.1).
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import differential_evolution

from .fitting import Problem
from .nested import RES, PriorTransform, emcee_check, load_samples, periods_of, rhat, run_nested, summarise

SEED7 = 20170411 + 7000
FITS7 = RES / "stage7_fits"
CKPT7 = RES / "stage7_ckpt"     # gitignored
CHAINS7 = RES / "chains7"       # gitignored
NRUNS = 3
C4_TOL = 20.0                   # chi2 units (12.3)
DIRS = {"fits": FITS7, "ckpts": CKPT7, "chains": CHAINS7}


def nlive_for(prob: Problem, c: int) -> int:
    return int(c * prob.ndim)


def multi_run7(tag: str, prob: Problem, seed: int, nlive: int, njobs: int = 4, nruns: int = NRUNS,
               slices: int | None = None) -> dict:
    """NRUNS dynesty runs + R-hat (C1) + lnZ spread (C2) + emcee tau test (C3', C3''). C4 is added later."""
    out_path = FITS7 / f"{tag}_multi.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    runs = [run_nested(f"{tag}_r{r}", prob, seed + r, nlive, njobs=njobs, slices=slices, **DIRS) for r in range(nruns)]
    samples = [load_samples(f"{tag}_r{r}", CHAINS7) for r in range(nruns)]
    pooled = np.vstack(samples)
    rh = rhat(samples, periodic=prob.periodic, periods=periods_of(prob))
    logz = [r["logz"] for r in runs]
    out = {"tag": tag, "model": prob.model, "k": prob.ndim, "n_data": prob.n_data, "names": prob.names,
           "nlive": nlive, "slices": slices if slices else 3 + prob.ndim, "nruns": nruns,
           "runs": [{k: r[k] for k in ("seed", "logz", "logzerr", "chi2_min", "chi2_nu", "ncall", "niter",
                                         "runtime_s", "information", "best")} | {"d": r["summary"].get("d")}
                    for r in runs],
           "logz_mean": float(np.mean(logz)), "logz_range": float(np.ptp(logz)),
           "rhat": {n: float(x) for n, x in zip(prob.names, rh)}}
    best_run = min(runs, key=lambda r: r["chi2_min"])
    out.update({k: best_run[k] for k in ("chi2_min", "chi2_cp", "chi2_lca", "nu", "chi2_nu", "bic", "best")})
    out["summary"] = summarise(prob, pooled, -0.5 * best_run["chi2_min"], np.asarray(best_run["best"]))["summary"]
    rd = out["rhat"].get("d", np.nan)
    out["C1_rhat"] = bool(np.isfinite(rd) and rd < 1.05 and np.nanmax(rh) < 1.10)
    out["C2_logz"] = bool(out["logz_range"] <= 2.0)
    out.update(tau_test(tag, prob, pooled, seed))
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[multi7] {tag}: nlive={nlive} lnZ={out['logz_mean']:.2f} range={out['logz_range']:.2f} "
          f"Rhat_d={rd:.3f} Rhat_max={np.nanmax(rh):.3f} chi2_runs={[round(r['chi2_min'], 1) for r in runs]} "
          f"C3p={out['C3p_tau']} C3pp={out['C3pp_tau']}", flush=True)
    return out


def tau_test(tag: str, prob: Problem, pooled: np.ndarray, seed: int) -> dict:
    """C3' (32000 steps, tau < 640); only if it fails, C3'' (128000 steps stored every 8th, tau < 2560)."""
    em = emcee_check(f"{tag}_emcee32k", prob, pooled, seed + 60, nburn=2000, nprod=32000, fits=FITS7)
    out = {"C3p_tau": bool(em["converged_tau"]),
           "emcee32k": {"tau_max": em["tau_max"], "tau_limit": em["tau_limit"], "d": em["summary"].get("d"),
                        "tau": dict(zip(prob.names, em["tau"])), "runtime_s": em["runtime_s"]},
           "C3pp_tau": None}
    if not out["C3p_tau"]:
        em2 = emcee_check(f"{tag}_emcee128k", prob, pooled, seed + 70, nburn=2000, nprod=128000, fits=FITS7,
                          thin_by=8)
        out["C3pp_tau"] = bool(em2["converged_tau"])
        out["emcee128k"] = {"tau_max": em2["tau_max"], "tau_limit": em2["tau_limit"], "d": em2["summary"].get("d"),
                            "tau": dict(zip(prob.names, em2["tau"])), "runtime_s": em2["runtime_s"]}
    out["C3_tau7"] = bool(out["C3p_tau"] or out["C3pp_tau"])
    return out


def de_fit(prob: Problem, seed: int, maxiter: int = 2000, popsize: int = 15) -> dict:
    """Differential evolution on the unit cube with the nested-sampling prior transform (12.1).

    Same parameterisation as the nested runs (x-ordering and the beta triangle fold), so every point of
    the cube maps into the prior support. Vectorised, L-BFGS-B polish at the end.
    """
    pt = PriorTransform(prob)

    def f(U):  # scipy vectorised: U (ndim, S)
        return prob.chi2(pt.batch(np.asarray(U).T))

    t0 = time.time()
    sol = differential_evolution(f, [(0.0, 1.0)] * prob.ndim, seed=seed, maxiter=maxiter, popsize=popsize,
                                 tol=1e-6, polish=True, vectorized=True, updating="deferred",
                                 init="latinhypercube")
    x = pt.batch(sol.x)[0]
    chi2 = float(prob.chi2(x)[0])
    return {"seed": int(seed), "chi2": chi2, "x": x.tolist(), "names": prob.names, "nfev": int(sol.nfev),
            "nit": int(sol.nit), "success": bool(sol.success), "message": str(sol.message),
            "d": float(x[prob.names.index("d")]), "runtime_s": time.time() - t0}


def distinct_modes(fits: list[dict], dchi2: float = 10.0, dd: float = 0.5) -> list[dict]:
    """Group optimisation results into modes: a new mode if chi2 differs by > dchi2 or d by > dd
    from every mode found so far (reporting only)."""
    modes: list[dict] = []
    for r in sorted(fits, key=lambda r: r["chi2"]):
        for m in modes:
            if abs(r["chi2"] - m["chi2"]) <= dchi2 and abs(r["d"] - m["d"]) <= dd:
                m["count"] += 1
                break
        else:
            modes.append({"chi2": r["chi2"], "d": r["d"], "count": 1, "seed": r["seed"]})
    return modes


def c4(run_chi2: list[float], global_best: float, tol: float = C4_TOL) -> bool:
    """C4: every run reaches the best mode found by any method (chi2_min <= global best + tol)."""
    return bool(all(c <= global_best + tol for c in run_chi2))
