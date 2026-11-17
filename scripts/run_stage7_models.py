"""Stage 7, 12.2-12.3: nested sampling of the large models with nlive = c * k, convergence C1-C4.

Ladder (PREREGISTRATION 12.2): G1 (c = 50) -> if not converged, G1 (c = 150) -> c* = smallest c at which
G1 converged (stop if none) -> G3, G2 at c* -> R2G{n*} at c* (only if c* = 50 and G{n*} converged).
Needs results/stage7_modemap.json (run_stage7_modemap.py). Output: results/stage7_models.json.
Resumable (results/stage7_fits/, dynesty checkpoints in results/stage7_ckpt/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.s6 import joint6  # noqa: E402
from bhshadow.s7 import C4_TOL, FITS7, SEED7, c4, multi_run7, nlive_for  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
SEEDS = {("G1", 50): 100, ("G1", 150): 150, ("G2", 50): 200, ("G2", 150): 250, ("G3", 50): 300, ("G3", 150): 350,
         ("R2G1", 50): 600, ("R2G2", 50): 620, ("R2G3", 50): 640}


def global_best(model: str, extra: list[float]) -> float:
    mm = json.loads((RES / "stage7_modemap.json").read_text())["models"]
    vals = list(extra) + [json.loads(p.read_text())["chi2_min"] for p in FITS7.glob(f"m7_{model}_c*_r[0-9].json")]
    if model in mm:
        vals += [mm[model]["de_best_chi2"]] + mm[model]["stage6_run_chi2"]
    return min(vals)


def run(model: str, c: int, ds) -> dict:
    prob = joint6(model, ds)
    r = multi_run7(f"m7_{model}_c{c}", prob, SEED7 + SEEDS[(model, c)], nlive_for(prob, c))
    chi = [x["chi2_min"] for x in r["runs"]]
    gb = global_best(model, chi)
    r["global_best_chi2"] = gb
    r["C4_mode"] = c4(chi, gb)
    r["converged7"] = bool(r["C1_rhat"] and r["C2_logz"] and r["C3_tau7"] and r["C4_mode"])
    print(f"[conv7] {model} c={c}: C1={r['C1_rhat']} C2={r['C2_logz']} C3'={r['C3p_tau']} C3''={r['C3pp_tau']} "
          f"C4={r['C4_mode']} (best {gb:.1f}, runs {[round(x, 1) for x in chi]}) -> {r['converged7']}", flush=True)
    return r


def main():
    ds = DS.load_all()
    res, log = {}, []
    res[("G1", 50)] = run("G1", 50, ds)
    cstar = 50 if res[("G1", 50)]["converged7"] else None
    if cstar is None:
        res[("G1", 150)] = run("G1", 150, ds)
        cstar = 150 if res[("G1", 150)]["converged7"] else None
    log.append(f"c* = {cstar}")
    if cstar is not None:
        for m in ("G3", "G2"):
            res[(m, cstar)] = run(m, cstar, ds)
        conv_g = [m for m in ("G1", "G2", "G3") if res.get((m, cstar), {}).get("converged7")]
        if cstar == 50 and conv_g:
            nstar = max(conv_g, key=lambda m: res[(m, cstar)]["logz_mean"])[-1]
            res[(f"R2G{nstar}", 50)] = run(f"R2G{nstar}", 50, ds)
            log.append(f"n* = {nstar}")
        else:
            log.append("R2G skipped (12.2: c* = 150 or no G model converged)")
    else:
        log.append("G1 did not converge at c = 50 or 150: ladder stopped (12.2), G2/G3/R2G not rerun")
    table = []
    for (m, c), r in res.items():
        table.append({"model": m, "c": c, "nlive": r["nlive"], "k": r["k"], "d": r["summary"]["d"]["median"],
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
    out = {"note": "stage 7 additional analysis (PREREGISTRATION 12.2-12.3)", "c4_tol": C4_TOL, "cstar": cstar,
           "log": log, "table": table}
    (RES / "stage7_models.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
