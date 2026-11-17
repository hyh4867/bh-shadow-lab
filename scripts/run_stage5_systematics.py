"""Stage 5 (additional analysis): jackknife, model extensions, xs-ring, error budget and mass.

PREREGISTRATION 10.2, 10.4, 10.5, 10.6 (committed a4f7d8f before running).
Outputs: results/stage5_systematics.json (+ one JSON per fit in results/stage5_fits/).
Resumable: fits already in results/stage5_fits/ are loaded, not rerun.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import literature as L  # noqa: E402
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.closure import model_closure_phase, model_log_camp  # noqa: E402
from bhshadow.fitting import Problem  # noqa: E402
from bhshadow.mass import compare, mass_mc  # noqa: E402
from bhshadow.robust import incoherent_dataset  # noqa: E402
from bhshadow.s5 import RES, SEED, cached_fit, joint_problem, sym, xs_dhat, xs_fw  # noqa: E402

STATIONS = ["AA", "AP", "AZ", "JC", "LM", "PV", "SM"]
DELTA_OBS = None  # filled from stage 2


def jk_sigma(values):
    v = np.asarray(values, float)
    n = len(v)
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


def closure_chi2_by_lmt(prob: Problem, best: np.ndarray) -> dict:
    out = {"cp_LMT": [0.0, 0], "cp_other": [0.0, 0], "lca_LMT": [0.0, 0], "lca_other": [0.0, 0]}
    th = np.atleast_2d(best)
    for d, g in zip(prob.datasets, prob.groups):
        V = prob.fn(d.tab.u, d.tab.v, *[th[:, prob.slot[(p, g)]] for p in prob.pnames])
        r = d.cp.obs - model_closure_phase(V, d.cp)[0]
        c = 2 * (1 - np.cos(r)) / d.cp.sigma**2
        has = np.array(["LM" in lab for lab in d.cp.label])
        out["cp_LMT"][0] += c[has].sum(); out["cp_LMT"][1] += int(has.sum())
        out["cp_other"][0] += c[~has].sum(); out["cp_other"][1] += int((~has).sum())
        r = d.lca.obs - model_log_camp(V, d.lca)[0]
        c = r**2 / d.lca.sigma**2
        has = np.array(["LM" in lab for lab in d.lca.label])
        out["lca_LMT"][0] += c[has].sum(); out["lca_LMT"][1] += int(has.sum())
        out["lca_other"][0] += c[~has].sum(); out["lca_other"][1] += int((~has).sum())
    return {k: {"chi2": float(v[0]), "n": v[1], "chi2_per_closure": float(v[0] / v[1]) if v[1] else None}
            for k, v in out.items()}


def main():
    st2 = json.loads((RES / "stage2_joint.json").read_text())
    rob2 = json.loads((RES / "stage2_robustness.json").read_text())
    fits2 = json.loads((RES / "stage2_fits.json").read_text())
    d_m3, d_m3_raw = st2["d_median"], st2["d_sigma_raw"]
    out = {"note": "추가 분석 (PREREGISTRATION 10). Stage 1-4 verdicts unchanged.",
           "reference_M3_joint": {"d": d_m3, "sigma_raw": d_m3_raw, "chi2_nu": st2["chi2_nu"],
                                  "bic": st2["fit"]["bic"], "source": "results/stage2_joint.json"}}
    dsets = DS.load_all()

    # ---- 10.2 jackknife ----
    jk = {"station": {}, "day": {}, "band": {}}
    for i, st in enumerate(STATIONS):
        ds = DS.load_all(drop_stations=(st,))
        f = cached_fit(f"jk_station_{st}", joint_problem("M3", ds), SEED + 1000 + i)
        jk["station"][st] = {"d": f["summary"]["d"], "w": f["summary"]["w"], "chi2_nu": f["chi2_nu"],
                             "n_data": f["n_data"], "converged": f["converged"]}
    for i, doy in enumerate(DS.DAYS):
        ds = [d for d in dsets if d.doy != doy]
        f = cached_fit(f"jk_day_minus_{doy}", joint_problem("M3", ds), SEED + 1100 + i)
        jk["day"][DS.DAYS[doy]] = {"d": f["summary"]["d"], "chi2_nu": f["chi2_nu"], "n_data": f["n_data"],
                                   "converged": f["converged"]}
    for i, band in enumerate(("lo", "hi")):
        ds = [d for d in dsets if d.band == band]
        f = cached_fit(f"jk_band_{band}_only", joint_problem("M3", ds), SEED + 1200 + i)
        jk["band"][band] = {"d": f["summary"]["d"], "chi2_nu": f["chi2_nu"], "n_data": f["n_data"],
                            "converged": f["converged"]}
    sig_station = jk_sigma([v["d"]["median"] for v in jk["station"].values()])
    sig_day = jk_sigma([v["d"]["median"] for v in jk["day"].values()])
    sig_band = abs(jk["band"]["lo"]["d"]["median"] - jk["band"]["hi"]["d"]["median"]) / 2
    sig_proc = abs(rob2["closures_from_10s_then_scan_mean"]["d"]["median"] - rob2["primary"]["d"]["median"])
    jk["sigma_station"], jk["sigma_day"], jk["sigma_band"], jk["sigma_proc"] = sig_station, sig_day, sig_band, sig_proc
    out["jackknife"] = jk
    print("JK sigmas", sig_station, sig_day, sig_band, sig_proc, flush=True)

    # ---- 10.4 model extensions (joint) ----
    ext = {"M3": {"d": st2["fit"]["summary"]["d"], "chi2_nu": st2["chi2_nu"], "bic": st2["fit"]["bic"],
                  "k": st2["fit"]["k"], "n_data": st2["fit"]["n_data"], "source": "stage2_joint.json"}}
    nolmt = DS.load_all(drop_stations=("LM",))
    ext_fit = {}
    for j, m in enumerate(("E1", "E2")):
        f = cached_fit(f"ext_{m}_all", joint_problem(m, dsets), SEED + 1300 + j, save_chain=True)
        ext_fit[m] = f
        ext[m] = {"d": f["summary"]["d"], "w": f["summary"]["w"], "chi2_nu": f["chi2_nu"], "bic": f["bic"],
                  "k": f["k"], "n_data": f["n_data"], "converged": f["converged"], "retried": f["retried"],
                  "delta_bic_vs_M3": f["bic"] - st2["fit"]["bic"], "summary": f["summary"]}
        g = cached_fit(f"ext_{m}_noLMT", joint_problem(m, nolmt), SEED + 1310 + j)
        ext[m]["noLMT"] = {"d": g["summary"]["d"], "chi2_nu": g["chi2_nu"], "converged": g["converged"]}
    bics = {m: ext[m]["bic"] for m in ("M3", "E1", "E2")}
    best = min(bics, key=bics.get)
    central = best if bics["M3"] - bics[best] > 10 else "M3"
    ext["central_rule"] = "lowest joint BIC among M3/E1/E2 if more than 10 below M3, else M3"
    ext["central_model"] = central
    d_c = ext[central]["d"]["median"]
    sig_model = max(abs(ext[m]["d"]["median"] - d_c) for m in ("M3", "E1", "E2") if m != central)
    ext["sigma_model"] = sig_model
    out["model_extensions"] = ext
    print("EXT", {m: (ext[m]["d"]["median"], ext[m]["bic"]) for m in ("M3", "E1", "E2")}, "central", central, flush=True)

    # ---- 10.3 references on real data (no verdict) ----
    raw_nolmt = []
    for d in dsets:
        keep = (d.raw.t1 != "LM") & (d.raw.t2 != "LM")
        raw_nolmt.append(incoherent_dataset(DS.build(d.raw.select(keep), d.name, d.doy, d.band)))
    f = cached_fit("tenseq_noLMT", joint_problem("M3", raw_nolmt), SEED + 1400)
    prim_prob = joint_problem("M3", dsets)
    out["lmt_reference"] = {
        "tenseq_all_d": rob2["closures_from_10s_then_scan_mean"]["d"]["median"],
        "tenseq_noLMT_d": f["summary"]["d"]["median"], "tenseq_noLMT_chi2_nu": f["chi2_nu"],
        "scanavg_all_d": d_m3, "scanavg_noLMT_d": rob2["no_LMT"]["d"]["median"],
        "closure_chi2_M3_joint_best": closure_chi2_by_lmt(prim_prob, np.array(st2["fit"]["best"])),
    }

    # ---- 10.5 xs-ring per data set ----
    xs = []
    for i, d in enumerate(dsets):
        row = {"dataset": d.name}
        for j, m in enumerate(("X0", "X1")):
            f = cached_fit(f"xs_{m}_{d.doy}_{d.band}", Problem(m, [d]), SEED + 1500 + 10 * i + j,
                           derived={"d_hat": xs_dhat, "f_w": xs_fw})
            dh = f["derived"]["d_hat"]
            row[m] = {"d_hat": dh, "f_w": f["derived"]["f_w"], "pa": f["summary"]["pa"],
                      "chi2_nu": f["chi2_nu"], "bic": f["bic"], "k": f["k"], "converged": f["converged"],
                      "retried": f["retried"], "d_hat_sigma_scaled": sym(dh) * max(1.0, np.sqrt(f["chi2_nu"]))}
        m3 = fits2[d.name]["M3"]
        row["M3"] = {"d": m3["summary"]["d"]["median"], "chi2_nu": m3["chi2_nu"], "bic": m3["bic"],
                     "sigma_scaled": sym(m3["summary"]["d"]) * max(1.0, np.sqrt(m3["chi2_nu"]))}
        e, em, ep = L.TABLE3_XSRING[d.name]
        row["EHT_table3"] = {"d_hat": e, "minus": em, "plus": ep, "pa": L.TABLE3_XSRING_PA[d.name]}
        x1 = row["X1"]["d_hat"]["median"]
        es = ep if x1 > e else em
        tol = float(np.sqrt(row["X1"]["d_hat_sigma_scaled"] ** 2 + es**2 + (L.TABLE3_OBS_FRAC * e) ** 2))
        row["X1_vs_EHT"] = {"diff": x1 - e, "tolerance_7_2": tol, "within": bool(abs(x1 - e) <= tol)}
        row["delta_tot"] = row["M3"]["d"] - e
        row["delta_def"] = row["M3"]["d"] - x1
        row["delta_impl"] = x1 - e
        xs.append(row)
    dtot = float(np.mean([r["delta_tot"] for r in xs]))
    ddef = float(np.mean([r["delta_def"] for r in xs]))
    dimp = float(np.mean([r["delta_impl"] for r in xs]))
    if abs(dimp) <= abs(dtot) / 3:
        v = "정의 차이"
    elif abs(ddef) <= abs(dtot) / 3:
        v = "구현 차이"
    else:
        v = "둘 다"
    x0_mean = float(np.mean([r["delta_tot"] - (r["M3"]["d"] - r["X0"]["d_hat"]["median"]) for r in xs]))
    out["xsring"] = {"per_dataset": xs, "mean_delta_tot": dtot, "mean_delta_def": ddef, "mean_delta_impl": dimp,
                     "verdict": v, "n_X1_within_tolerance": int(sum(r["X1_vs_EHT"]["within"] for r in xs)),
                     "mean_delta_impl_X0": x0_mean,
                     "rule": "|impl| <= |tot|/3 -> definition; |def| <= |tot|/3 -> implementation; else both"}
    print("XS", dtot, ddef, dimp, v, flush=True)

    # ---- 10.6 budget and mass ----
    if central == "M3":
        stat = d_m3_raw * max(1.0, np.sqrt(st2["chi2_nu"]))
    else:
        stat = sym(ext[central]["d"]) * max(1.0, np.sqrt(ext[central]["chi2_nu"]))
    comps = {"station": sig_station, "day": sig_day, "band": sig_band, "model": sig_model, "proc": sig_proc}
    sys_ = float(np.sqrt(sum(c**2 for c in comps.values())))
    tot = float(np.hypot(stat, sys_))
    comb = float(np.hypot(tot, L.D_EHT_ERR))
    budget = {"central_model": central, "d": d_c, "stat": stat, "sys_components": comps, "sys": sys_, "total": tot,
              "verdict_vs_42pm3": {"diff": d_c - L.D_EHT, "combined_sigma": comb, "z": (d_c - L.D_EHT) / comb,
                                   "result": "재현" if abs(d_c - L.D_EHT) <= comb else "불일치"}}
    out["budget"] = budget
    mres = {"GC_alpha": mass_mc(d_c, tot, L.ALPHA_GC, L.ALPHA_GC * L.ALPHA_GC_FRAC, L.DIST_MPC, L.DIST_ERR),
            "photon_ring_alpha": mass_mc(d_c, tot, L.ALPHA_PR, L.ALPHA_PR_ERR, L.DIST_MPC, L.DIST_ERR)}
    for key in ("GC_alpha", "photon_ring_alpha"):
        mres[f"comparison_{key}"] = {k: {**compare(mres[key]["mass_1e9"], v[:3]), "lit": v} for k, v in L.MASSES.items()}
    x1d = np.array([r["X1"]["d_hat"]["median"] for r in xs])
    x1s = np.array([sym(r["X1"]["d_hat"]) for r in xs])
    xd, xerr = float(x1d.mean()), float(np.hypot(x1d.std(ddof=1), x1s.mean()))
    mres["xsring_dhat_GC_alpha"] = mass_mc(xd, xerr, L.ALPHA_GC, L.ALPHA_GC * L.ALPHA_GC_FRAC, L.DIST_MPC, L.DIST_ERR)
    mres["xsring_dhat_GC_alpha"]["d_hat_mean_X1"] = xd
    mres["comparison_xsring_dhat_GC_alpha"] = {k: {**compare(mres["xsring_dhat_GC_alpha"]["mass_1e9"], v[:3]),
                                                   "lit": v} for k, v in L.MASSES.items()}
    out["mass"] = mres
    print("BUDGET", json.dumps(budget, default=float), flush=True)
    print("MASS", mres["GC_alpha"]["mass_1e9"], mres["photon_ring_alpha"]["mass_1e9"],
          mres["xsring_dhat_GC_alpha"]["mass_1e9"], flush=True)
    (RES / "stage5_systematics.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
