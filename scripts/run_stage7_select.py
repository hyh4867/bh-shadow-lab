"""Stage 7, 12.4: model table under the stage 7 convergence rule (C1, C2, C3' or C3'', C4), central model by
the 11.3 rule, sigma_model by the 11.4 rule, budget (provisional if the central model is not E1) and mass.

Needs results/stage7_modemap.json and results/stage7_models.json. Output: results/stage7_selection.json.
No new fits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import literature as L  # noqa: E402
from bhshadow.mass import compare, mass_mc  # noqa: E402
from bhshadow.s7 import FITS7, c4  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
NUISANCE = {"E1", "G1", "G2", "G3", "R2G1", "R2G2", "R2G3"}


def main():
    mm = json.loads((RES / "stage7_modemap.json").read_text())["models"]
    s7 = json.loads((RES / "stage7_models.json").read_text())
    s6 = {t["model"]: t for t in json.loads((RES / "stage6_models.json").read_text())["table"]}
    rows = {}
    # stage 6 runs of the models that converged there (C3' rule), with the C4 check added
    for m in ("M3", "E1", "E2"):
        t = s6[m]
        j = json.loads((RES / "stage6_fits" / f"m6_{m}_multi.json").read_text())
        chi = [r["chi2_min"] for r in j["runs"]]
        gb = min([mm[m]["de_best_chi2"]] + chi)
        rows[m] = {"model": m, "source": "stage 6 (nlive 300)", "k": t["k"], "d": t["d"],
                   "d_sigma_raw": t["d_sigma_raw"], "chi2_nu": t["chi2_nu"], "bic": t["bic"], "lnZ": t["lnZ"],
                   "lnZ_range": t["lnZ_range"], "chi2_runs": chi, "global_best_chi2": gb, "C1": t["C1"],
                   "C2": t["C2"], "C3_tau7": t["C3p"], "C4": c4(chi, gb)}
    # stage 7 runs: the largest c per model, then ladder B (12.7) overrides
    s7b_path = RES / "stage7b_models.json"
    tabs = sorted(s7["table"], key=lambda t: t["c"])
    if s7b_path.exists():
        tabs += json.loads(s7b_path.read_text())["table"]
    for t in tabs:
        src = (f"stage 7 B (nlive {t['nlive']}, slices {t['slices']})" if str(t["c"]).startswith("B")
               else f"stage 7 (nlive {t['nlive']}, c = {t['c']})")
        rows[t["model"]] = {"model": t["model"], "source": src,
                            "k": t["k"], "d": t["d"], "d_sigma_raw": t["d_sigma_raw"], "chi2_nu": t["chi2_nu"],
                            "bic": t["bic"], "lnZ": t["lnZ"], "lnZ_range": t["lnZ_range"], "chi2_runs": t["chi2_runs"],
                            "global_best_chi2": t["global_best_chi2"], "C1": t["C1"], "C2": t["C2"],
                            "C3_tau7": bool(t["C3p"] or t["C3pp"]), "C4": t["C4"]}
    # C4 against the best chi2 of every run so far (12.7)
    for m, r in rows.items():
        allc = [json.loads(p.read_text())["chi2_min"] for p in FITS7.glob(f"m7*_{m}_*_r[0-9].json")]
        r["global_best_chi2"] = min([r["global_best_chi2"]] + allc)
        r["C4"] = c4(r["chi2_runs"], r["global_best_chi2"])
        r["converged7"] = bool(r["C1"] and r["C2"] and r["C3_tau7"] and r["C4"])
    # models not rerun in stage 7 keep their stage 6 status (not converged)
    for m in ("G1", "G2", "G3", "R2G3"):
        if m not in rows:
            t = s6[m]
            rows[m] = {"model": m, "source": "stage 6 (not rerun)", "k": t["k"], "d": t["d"], "lnZ": t["lnZ"],
                       "lnZ_range": t["lnZ_range"], "bic": t["bic"], "chi2_nu": t["chi2_nu"], "converged7": False}
    conv = [m for m, r in rows.items() if r["converged7"]]
    best = max(conv, key=lambda m: rows[m]["lnZ"])
    simpler = [m for m in conv if rows[m]["k"] < rows[best]["k"] and rows[m]["lnZ"] >= rows[best]["lnZ"] - 5]
    C = min(simpler, key=lambda m: rows[m]["k"]) if simpler else best
    rc = rows[C]
    stat = rc["d_sigma_raw"] * max(1.0, np.sqrt(rc["chi2_nu"]))
    bic_best = min(rows, key=lambda m: rows[m]["bic"])
    sel = {"central": C, "max_lnZ_model": best, "converged_models": conv, "bic_best": bic_best,
           "bic_best_converged": bic_best in conv, "d": rc["d"], "stat": stat}
    # sigma_model (11.4; overlap test of stage 6 = '겹침' -> converged nuisance models only)
    S6 = json.loads((RES / "stage6_systematics.json").read_text())
    overlap = S6["overlap_test"]["verdict"] == "겹침"
    pool = [m for m in conv if m != C and (m in NUISANCE or not overlap)]
    sig_model = max([abs(rows[m]["d"] - rc["d"]) for m in pool], default=0.0)
    comp = dict(S6["budget"]["sys_components"])
    comp["model"] = sig_model
    provisional = C != S6["central_model"]
    sys_ = float(np.sqrt(sum(v**2 for v in comp.values())))
    tot = float(np.hypot(stat, sys_))
    comb = float(np.hypot(tot, L.D_EHT_ERR))
    budget = {"d": rc["d"], "stat": stat, "sys_components": comp, "sys": sys_, "total": tot,
              "provisional": provisional,
              "provisional_note": ("jackknife, proc and set terms are the stage 6 values for "
                                   f"{S6['central_model']}, not for {C} (12.4)") if provisional else "",
              "model_term": {"overlap": overlap, "compared": pool, "diffs": {m: rows[m]["d"] - rc["d"] for m in pool}},
              "verdict_vs_42pm3": {"diff": rc["d"] - L.D_EHT, "combined_sigma": comb, "z": (rc["d"] - L.D_EHT) / comb,
                                   "result": "재현" if abs(rc["d"] - L.D_EHT) <= comb else "불일치"}}
    mres = {"GC_alpha": mass_mc(rc["d"], tot, L.ALPHA_GC, L.ALPHA_GC * L.ALPHA_GC_FRAC, L.DIST_MPC, L.DIST_ERR),
            "photon_ring_alpha": mass_mc(rc["d"], tot, L.ALPHA_PR, L.ALPHA_PR_ERR, L.DIST_MPC, L.DIST_ERR)}
    for key in ("GC_alpha", "photon_ring_alpha"):
        mres[f"comparison_{key}"] = {k: {**compare(mres[key]["mass_1e9"], v[:3]), "lit": v} for k, v in L.MASSES.items()}
    out = {"note": "stage 7 additional analysis (PREREGISTRATION 12.4)", "rows": rows, "selection": sel,
           "budget": budget, "mass": mres, "stage6_final": S6["budget"]}
    (RES / "stage7_selection.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    for m, r in rows.items():
        print(f"{m:5s} {r['source']:28s} d={r['d']:.2f} lnZ={r['lnZ']:.1f} conv7={r['converged7']}")
    print(json.dumps(sel), json.dumps(budget, default=float, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
