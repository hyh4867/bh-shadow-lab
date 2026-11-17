"""Stage 6, 11.1-11.3: main models with nested sampling (3 runs each), convergence table, central model.

Output: results/stage6_models.json (per-run summaries in results/stage6_fits/).
Resumable: finished runs are read from results/stage6_fits/, dynesty checkpoints from results/stage6_ckpt/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.s6 import SEED6, joint6, multi_run, sym  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
ORDER = ["M3", "E1", "E2", "G1", "G2", "G3"]


def select(res: dict) -> dict:
    conv = [m for m in res if res[m]["converged"]]
    ext = [m for m in conv if m != "M3"]
    note = ""
    pool = conv if conv else list(res)
    if not conv:
        note = "no model converged; central = max lnZ overall (flagged)"
    elif not ext:
        note = "no extended model converged; central = best converged model (11.3 step 4)"
    best = max(pool, key=lambda m: res[m]["logz_mean"])
    simpler = [m for m in pool if res[m]["k"] < res[best]["k"] and res[m]["logz_mean"] >= res[best]["logz_mean"] - 5]
    central = min(simpler, key=lambda m: res[m]["k"]) if simpler else best
    bic_best = min(res, key=lambda m: res[m]["bic"])
    return {"central": central, "max_lnZ_model": best, "candidates": pool, "converged_models": conv,
            "bic_best": bic_best, "bic_agrees": bic_best == central, "note": note}


def main():
    ds = DS.load_all()
    res = {}
    for i, m in enumerate(ORDER):
        res[m] = multi_run(f"m6_{m}", joint6(m, ds), SEED6 + 100 * i)
    nstar = max((1, 2, 3), key=lambda n: res[f"G{n}"]["logz_mean"])
    r2 = f"R2G{nstar}"
    res[r2] = multi_run(f"m6_{r2}", joint6(r2, ds), SEED6 + 600)
    sel = select(res)
    C = sel["central"]
    rc = res[C]
    d_c, s_raw = rc["summary"]["d"]["median"], sym(rc["summary"]["d"])
    stat = s_raw * max(1.0, np.sqrt(rc["chi2_nu"]))
    table = []
    for m, r in res.items():
        table.append({
            "model": m, "k": r["k"], "d": r["summary"]["d"]["median"], "d_sigma_raw": sym(r["summary"]["d"]),
            "chi2_nu": r["chi2_nu"], "bic": r["bic"], "lnZ": r["logz_mean"], "lnZ_range": r["logz_range"],
            "dlnZ_vs_C": r["logz_mean"] - rc["logz_mean"],
            "dlnZ_over_chi2nuC": (r["logz_mean"] - rc["logz_mean"]) / rc["chi2_nu"],
            "rhat_d": r["rhat"]["d"], "rhat_max": max(r["rhat"].values()),
            "tau_max": r["emcee"]["tau_max"], "emcee_d": r["emcee"]["d"]["median"],
            "C1": r["C1_rhat"], "C2": r["C2_logz"], "C3": r["C3_tau"], "converged": r["converged"],
            "d_runs": [x["d"]["median"] for x in r["runs"]], "lnZ_runs": [x["logz"] for x in r["runs"]],
            "runtime_s": sum(x["runtime_s"] for x in r["runs"]),
        })
    # stage 5 emcee tau for the models that did not converge there
    s5 = {}
    for tag in ("ext_E1_all", "ext_E2_all"):
        p = RES / "stage5_fits" / f"{tag}.json"
        if p.exists():
            f = json.loads(p.read_text())
            s5[tag] = {"tau_max": max(f["tau"]), "tau_limit": f["nprod"] / 50, "converged": f["converged"],
                       "d": f["summary"]["d"]["median"]}
    out = {"note": "stage 6 additional analysis (PREREGISTRATION 11.1-11.3)", "table": table, "selection": sel,
           "nstar": nstar, "central": {"model": C, "d": d_c, "sigma_raw": s_raw, "chi2_nu": rc["chi2_nu"],
                                       "stat": stat, "summary": rc["summary"]},
           "stage5_emcee_reference": s5}
    (RES / "stage6_models.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"central": C, "d": d_c, "stat": stat, "sel": sel}, indent=1))
    for t in table:
        print(f"{t['model']:5s} k={t['k']:2d} d={t['d']:.2f} chi2nu={t['chi2_nu']:.3f} lnZ={t['lnZ']:.1f} "
              f"rng={t['lnZ_range']:.2f} Rd={t['rhat_d']:.3f} Rmax={t['rhat_max']:.3f} tau={t['tau_max']:.0f} conv={t['converged']}")


if __name__ == "__main__":
    main()
