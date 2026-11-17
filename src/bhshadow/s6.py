"""Shared helpers for the stage 6 analyses (PREREGISTRATION section 11)."""
from __future__ import annotations

import json

import numpy as np

from . import models as M
from .fitting import Problem
from .nested import FITS6, emcee_check, load_samples, periods_of, rhat, run_nested
from .s5 import day_groups

SEED6 = 20170411 + 6000
NLIVE = 300
NRUNS = 3


def shared_of(model: str) -> tuple[str, ...]:
    if model in ("M3", "E2"):
        return ("d", "w")
    if model == "E1":
        return ("d", "w", "fg", "xg", "yg", "fwhmg")
    if model[0] == "G" or model.startswith("R2G"):
        return ("d", "w") + tuple(M._gauss_names(int(model[-1])))
    raise KeyError(model)


def joint6(model: str, dsets) -> Problem:
    return Problem(model, dsets, groups=day_groups(dsets), shared=shared_of(model))


def multi_run(tag: str, prob: Problem, seed: int, nruns: int = NRUNS, emcee: bool = True,
              derived: dict | None = None) -> dict:
    """NRUNS independent dynesty runs + R-hat + lnZ spread + emcee tau check (11.1)."""
    runs = [run_nested(f"{tag}_r{r}", prob, seed + r, NLIVE, derived=derived) for r in range(nruns)]
    samples = [load_samples(f"{tag}_r{r}") for r in range(nruns)]
    pooled = np.vstack(samples)
    rh = rhat(samples, periodic=prob.periodic, periods=periods_of(prob))
    logz = [r["logz"] for r in runs]
    out = {"tag": tag, "model": prob.model, "k": prob.ndim, "n_data": prob.n_data, "names": prob.names,
           "runs": [{k: r[k] for k in ("seed", "logz", "logzerr", "chi2_min", "chi2_nu", "ncall", "niter",
                                         "runtime_s", "information")} | {"d": r["summary"].get("d")} for r in runs],
           "logz_mean": float(np.mean(logz)), "logz_range": float(np.ptp(logz)),
           "rhat": {n: float(x) for n, x in zip(prob.names, rh)}}
    best_run = min(runs, key=lambda r: r["chi2_min"])
    out.update({k: best_run[k] for k in ("chi2_min", "chi2_cp", "chi2_lca", "nu", "chi2_nu", "bic", "best")})
    from .nested import summarise
    s = summarise(prob, pooled, -0.5 * best_run["chi2_min"], np.asarray(best_run["best"]))
    out["summary"] = s["summary"]
    if derived:
        out["derived"] = {}
        for name, fn in derived.items():
            x = fn(pooled, prob.names)
            q16, m, q84 = np.percentile(x, [16, 50, 84])
            out["derived"][name] = {"median": float(m), "minus": float(m - q16), "plus": float(q84 - m)}
    rd = out["rhat"].get("d", np.nan)
    c1 = bool(np.isfinite(rd) and rd < 1.05 and np.nanmax(rh) < 1.10)
    c2 = bool(out["logz_range"] <= 2.0)
    out["C1_rhat"], out["C2_logz"] = c1, c2
    if emcee:
        em = emcee_check(f"{tag}_emcee", prob, pooled, seed + 50)
        out["emcee"] = {k: em[k] for k in ("tau_max", "tau_limit", "converged_tau", "nwalkers", "acceptance",
                                           "runtime_s")}
        out["emcee"]["tau"] = dict(zip(prob.names, em["tau"]))
        out["emcee"]["d"] = em["summary"].get("d")
        out["C3_tau"] = bool(em["converged_tau"])
        out.update(c3_long(tag, prob, pooled, seed))
    else:
        out["C3_tau"] = None
        out["C3p_tau"] = None
    out["converged"] = bool(c1 and c2 and (out["C3_tau"] is not False))
    out["converged_p"] = bool(c1 and c2 and (out["C3p_tau"] is not False))
    (FITS6 / f"{tag}_multi.json").write_text(json.dumps(out, indent=1))
    print(f"[multi] {tag}: lnZ={out['logz_mean']:.2f} range={out['logz_range']:.2f} Rhat_d={rd:.3f} "
          f"Rhat_max={np.nanmax(rh):.3f} tau_ok={out['C3_tau']} tau32k_ok={out['C3p_tau']} "
          f"conv={out['converged']} conv_p={out['converged_p']}", flush=True)
    return out


def c3_long(tag: str, prob: Problem, pooled: np.ndarray, seed: int) -> dict:
    """C3' (PREREGISTRATION 11.10): emcee from the nested posterior, burn-in 2000, production 32000, tau < 640."""
    em = emcee_check(f"{tag}_emcee32k", prob, pooled, seed + 60, nburn=2000, nprod=32000)
    return {"C3p_tau": bool(em["converged_tau"]),
            "emcee32k": {"tau_max": em["tau_max"], "tau_limit": em["tau_limit"], "d": em["summary"].get("d"),
                         "tau": dict(zip(prob.names, em["tau"])), "runtime_s": em["runtime_s"]}}


def single(tag: str, prob: Problem, seed: int, derived: dict | None = None) -> dict:
    return run_nested(tag, prob, seed, NLIVE, derived=derived)


def sym(s: dict) -> float:
    return 0.5 * (s["minus"] + s["plus"])
