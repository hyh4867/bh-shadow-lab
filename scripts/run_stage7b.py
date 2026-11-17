"""Stage 7, 12.7 ladder B: slices = 5 (3 + k) (five times the dynesty rslice default).

1. G1, nlive 300, 5 runs -> if converged: G3, G2 (nlive 300, 3 runs) -> R2G{n*} (nlive 300, 3 runs)
2. else G1, nlive 800, 5 runs (G2/G3 not run even if it converges)
Needs results/stage7_modemap.json. Output: results/stage7b_models.json. Resumable (results/stage7_fits/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.s6 import joint6  # noqa: E402
from bhshadow.s7 import C4_TOL, FITS7, SEED7, c4, multi_run7  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
SEEDS = {("G1", 300): 2100, ("G1", 800): 2200, ("G3", 300): 2300, ("G2", 300): 2400,
         ("R2G1", 300): 2600, ("R2G2", 300): 2620, ("R2G3", 300): 2640}


def global_best(model: str, extra: list[float]) -> float:
    mm = json.loads((RES / "stage7_modemap.json").read_text())["models"]
    vals = list(extra) + [json.loads(p.read_text())["chi2_min"] for p in FITS7.glob(f"m7*_{model}_*_r[0-9].json")]
    if model in mm:
        vals += [mm[model]["de_best_chi2"]] + mm[model]["stage6_run_chi2"]
    return min(vals)


def run(model: str, nlive: int, nruns: int, ds) -> dict:
    prob = joint6(model, ds)
    r = multi_run7(f"m7b_{model}_n{nlive}", prob, SEED7 + SEEDS[(model, nlive)], nlive, nruns=nruns,
                   slices=5 * (3 + prob.ndim))
    chi = [x["chi2_min"] for x in r["runs"]]
    gb = global_best(model, chi)
    r["global_best_chi2"] = gb
    r["C4_mode"] = c4(chi, gb)
    r["converged7"] = bool(r["C1_rhat"] and r["C2_logz"] and r["C3_tau7"] and r["C4_mode"])
    print(f"[conv7b] {model} nlive={nlive}: C1={r['C1_rhat']} C2={r['C2_logz']} C3'={r['C3p_tau']} "
          f"C3''={r['C3pp_tau']} C4={r['C4_mode']} (best {gb:.1f}, runs {[round(x, 1) for x in chi]}) "
          f"-> {r['converged7']}", flush=True)
    return r


def main():
    ds = DS.load_all()
    res, log = {}, []
    res[("G1", 300)] = run("G1", 300, 5, ds)
    if res[("G1", 300)]["converged7"]:
        log.append("G1 converged at nlive 300 with 5x slices")
        for m in ("G3", "G2"):
            res[(m, 300)] = run(m, 300, 3, ds)
        conv_g = [m for m in ("G1", "G2", "G3") if res.get((m, 300), {}).get("converged7")]
        if conv_g:
            nstar = max(conv_g, key=lambda m: res[(m, 300)]["logz_mean"])[-1]
            res[(f"R2G{nstar}", 300)] = run(f"R2G{nstar}", 300, 3, ds)
            log.append(f"n* = {nstar}")
    else:
        res[("G1", 800)] = run("G1", 800, 5, ds)
        log.append("G1 nlive 800 5x slices: " + ("converged (G2/G3 deferred, 12.7)" if res[("G1", 800)]["converged7"]
                                                  else "not converged -> mixing hypothesis rejected"))
    table = []
    for (m, n), r in res.items():
        table.append({"model": m, "c": f"B{n}", "nlive": n, "slices": r["slices"], "nruns": r["nruns"], "k": r["k"],
                      "d": r["summary"]["d"]["median"],
                      "d_sigma_raw": 0.5 * (r["summary"]["d"]["minus"] + r["summary"]["d"]["plus"]),
                      "chi2_nu": r["chi2_nu"], "bic": r["bic"], "lnZ": r["logz_mean"], "lnZ_range": r["logz_range"],
                      "lnZ_runs": [x["logz"] for x in r["runs"]], "d_runs": [x["d"]["median"] for x in r["runs"]],
                      "chi2_runs": [x["chi2_min"] for x in r["runs"]], "global_best_chi2": r["global_best_chi2"],
                      "rhat_d": r["rhat"]["d"], "rhat_max": max(r["rhat"].values()),
                      "tau32k": r["emcee32k"]["tau_max"], "tau128k": r.get("emcee128k", {}).get("tau_max"),
                      "emcee_d": (r.get("emcee128k") or r["emcee32k"])["d"]["median"],
                      "C1": r["C1_rhat"], "C2": r["C2_logz"], "C3p": r["C3p_tau"], "C3pp": r["C3pp_tau"],
                      "C4": r["C4_mode"], "converged7": r["converged7"],
                      "runtime_h": sum(x["runtime_s"] for x in r["runs"]) / 3600})
    out = {"note": "stage 7 additional analysis, ladder B (PREREGISTRATION 12.7)", "c4_tol": C4_TOL, "log": log,
           "table": table}
    (RES / "stage7b_models.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
