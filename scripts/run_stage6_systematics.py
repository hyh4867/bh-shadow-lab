"""Stage 6, 11.4: jackknife with the central model C, sigma_proc, co-located cut, overlap test,
final budget and mass.

Needs results/stage6_models.json (run_stage6_models.py). Output: results/stage6_systematics.json.
Resumable (results/stage6_fits/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow import literature as L  # noqa: E402
from bhshadow.mass import compare, mass_mc  # noqa: E402
from bhshadow.robust import incoherent_dataset  # noqa: E402
from bhshadow.s6 import SEED6, joint6, single, sym  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
STATIONS = ["AA", "AP", "AZ", "JC", "LM", "PV", "SM"]
NUISANCE = {"E1", "G1", "G2", "G3", "R2G1", "R2G2", "R2G3"}


def jk_sigma(values):
    v = np.asarray(values, float)
    n = len(v)
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


def colocated_count(ds) -> int:
    from bhshadow import INTRA_SITE
    n = 0
    for lab in ds.lca.label:
        st = set(lab.replace("*", "-").replace("/", "-").split("-"))
        n += any(pr <= st for pr in INTRA_SITE)
    return n


def main():
    M6 = json.loads((RES / "stage6_models.json").read_text())
    C = M6["central"]["model"]
    d_c, stat = M6["central"]["d"], M6["central"]["stat"]
    table = {t["model"]: t for t in M6["table"]}
    out = {"note": "stage 6 additional analysis (PREREGISTRATION 11.4)", "central_model": C, "d_C": d_c, "stat": stat}
    S = SEED6 + 1000
    dC = lambda f: f["summary"]["d"]["median"]  # noqa: E731

    # ---- jackknife with C ----
    jk = {"station": {}, "day": {}, "band": {}}
    for i, st in enumerate(STATIONS):
        jk["station"][st] = dC(single(f"j6_{C}_station_{st}", joint6(C, DS.load_all(drop_stations=(st,))), S + i))
    for i, doy in enumerate(DS.DAYS):
        sub = [DS.load(d, b) for d in DS.DAYS if d != doy for b in ("lo", "hi")]
        jk["day"][doy] = dC(single(f"j6_{C}_day_{doy}", joint6(C, sub), S + 20 + i))
    for i, band in enumerate(("lo", "hi")):
        sub = [DS.load(d, band) for d in DS.DAYS]
        jk["band"][band] = dC(single(f"j6_{C}_band_{band}", joint6(C, sub), S + 30 + i))
    sig = {"station": jk_sigma(list(jk["station"].values())), "day": jk_sigma(list(jk["day"].values())),
           "band": abs(jk["band"]["lo"] - jk["band"]["hi"]) / 2}
    out["jackknife"] = jk
    print("JK", json.dumps(jk), json.dumps(sig), flush=True)

    # ---- sigma_proc: 10-s closure variant ----
    ds = DS.load_all()
    inc = [incoherent_dataset(d) for d in ds]
    d_proc = dC(single(f"p6_{C}_tenseq", joint6(C, inc), S + 40))
    sig["proc"] = abs(d_proc - d_c)
    out["proc"] = {"d": d_proc}

    # ---- co-located cut ----
    cut = DS.load_all(drop_colocated=True)
    counts = [{"dataset": a.name, "n_lca_full": len(a.lca), "n_lca_cut": len(b.lca),
               "n_colocated_in_full": colocated_count(a), "n_cp": len(a.cp)} for a, b in zip(ds, cut)]
    fC = single(f"s6_{C}_colocut", joint6(C, cut), S + 50)
    fM = single("s6_M3_colocut", joint6("M3", cut), S + 51)
    sig["set"] = abs(dC(fC) - d_c)
    out["colocated_cut"] = {"counts": counts, "d_C_cut": dC(fC), "chi2_nu_C_cut": fC["chi2_nu"],
                            "d_M3_cut": dC(fM), "chi2_nu_M3_cut": fM["chi2_nu"], "d_M3_full": table["M3"]["d"],
                            "M3_shift": dC(fM) - table["M3"]["d"]}
    print("CUT", json.dumps(out["colocated_cut"]["counts"][:2]), dC(fC), dC(fM), flush=True)

    # ---- overlap test (stage 5 double counting) ----
    d_m3_full = table["M3"]["d"]
    m3_jk = {st: json.loads((RES / "stage5_fits" / f"jk_station_{st}.json").read_text())["summary"]["d"]["median"]
             for st in STATIONS}
    delta_full = d_c - d_m3_full
    deltas = {st: jk["station"][st] - m3_jk[st] for st in STATIONS}
    short = float(np.mean([abs(deltas["AZ"]), abs(deltas["LM"])]))
    overlap = short <= 0.5 * abs(delta_full)
    out["overlap_test"] = {"delta_full": delta_full, "delta_station": deltas, "mean_abs_delta_noAZ_noLM": short,
                           "threshold": 0.5 * abs(delta_full), "M3_jackknife_stage5": m3_jk,
                           "verdict": "겹침" if overlap else "겹치지 않음"}
    print("OVERLAP", json.dumps(out["overlap_test"]), flush=True)

    # ---- sigma_model ----
    conv = [m for m, t in table.items() if t["converged_p"] and m != C]  # changed rule C3p (11.10)
    pool = [m for m in conv if m in NUISANCE] if overlap else conv
    sig["model"] = max([abs(table[m]["d"] - d_c) for m in pool], default=0.0)
    out["model_term"] = {"compared_models": pool, "diffs": {m: table[m]["d"] - d_c for m in pool},
                         "rule": "overlap -> converged nuisance models only; else all converged models"}

    # ---- budget ----
    sys_ = float(np.sqrt(sum(v**2 for v in sig.values())))
    tot = float(np.hypot(stat, sys_))
    comb = float(np.hypot(tot, L.D_EHT_ERR))
    s5 = json.loads((RES / "stage5_systematics.json").read_text())["budget"]
    s5_style = float(np.sqrt(stat**2 + s5["sys_components"]["station"] ** 2 + s5["sys_components"]["model"] ** 2
                             + sig["day"] ** 2 + sig["band"] ** 2 + sig["proc"] ** 2 + sig["set"] ** 2))
    out["budget"] = {"d": d_c, "stat": stat, "sys_components": sig, "sys": sys_, "total": tot,
                     "verdict_vs_42pm3": {"diff": d_c - L.D_EHT, "combined_sigma": comb, "z": (d_c - L.D_EHT) / comb,
                                          "result": "재현" if abs(d_c - L.D_EHT) <= comb else "불일치"},
                     "reference_stage5_style_total": s5_style}
    mres = {"GC_alpha": mass_mc(d_c, tot, L.ALPHA_GC, L.ALPHA_GC * L.ALPHA_GC_FRAC, L.DIST_MPC, L.DIST_ERR),
            "photon_ring_alpha": mass_mc(d_c, tot, L.ALPHA_PR, L.ALPHA_PR_ERR, L.DIST_MPC, L.DIST_ERR)}
    for key in ("GC_alpha", "photon_ring_alpha"):
        mres[f"comparison_{key}"] = {k: {**compare(mres[key]["mass_1e9"], v[:3]), "lit": v} for k, v in L.MASSES.items()}
    out["mass"] = mres
    (RES / "stage6_systematics.json").write_text(json.dumps(out, indent=1, default=float))
    print("BUDGET", json.dumps(out["budget"], default=float), flush=True)
    print("MASS", mres["GC_alpha"]["mass_1e9"], mres["photon_ring_alpha"]["mass_1e9"], flush=True)


if __name__ == "__main__":
    main()
