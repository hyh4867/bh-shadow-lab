"""Stage 6, 11.6: 2018 follow-up. (a) why 4/25 b1, b2 are small (H1-H3); (b) 2018 budget with the
central model C; (c) 2018 vs 2017; (d) 2018 vs EHT 2018.

Needs results/stage6_models.json and results/stage6_systematics.json. Output: results/stage6_2018.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow import models as M  # noqa: E402
from bhshadow import synth  # noqa: E402
from bhshadow.fitting import Problem, run_mcmc  # noqa: E402
from bhshadow.nested import FITS6  # noqa: E402
from bhshadow.s5 import strip  # noqa: E402
from bhshadow.s6 import SEED6, joint6, single, sym  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
EHT18 = (43.3, 3.1, 1.5)  # value, minus, plus (A&A 681 A79, Table 12)
ST18 = ["AA", "AX", "GL", "LM", "MG", "MM", "PV", "SW"]


def emcee_fit(tag: str, prob: Problem, seed: int) -> dict:
    """Stage 2 sampler (emcee, 32 walkers), cached in results/stage6_fits/."""
    FITS6.mkdir(parents=True, exist_ok=True)
    p = FITS6 / f"{tag}.json"
    if p.exists():
        return json.loads(p.read_text())
    r = strip(run_mcmc(prob, seed))
    p.write_text(json.dumps(r, indent=1))
    print(f"[emcee] {tag}: d={r['summary']['d']['median']:.2f} chi2nu={r['chi2_nu']:.2f} conv={r['converged']}", flush=True)
    return r


def dmed(f):
    return f["summary"]["d"]["median"]


def jk_sigma(values):
    v = np.asarray(values, float)
    n = len(v)
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


def main():
    s5 = json.loads((RES / "stage5_2018.json").read_text())
    per5 = {r["dataset"]: r for r in s5["per_dataset"]}
    M6 = json.loads((RES / "stage6_models.json").read_text())
    S6 = json.loads((RES / "stage6_systematics.json").read_text())
    C = M6["central"]["model"]
    S = SEED6 + 3000
    out = {"note": "stage 6 additional analysis (PREREGISTRATION 11.6)", "central_model": C}

    # ---- (a) 4/25 b1, b2 ----
    d = {b: per5[f"2018-04-25 {b}"]["d"] for b in DS.BANDS2018}
    dobs = (d["b1"] + d["b2"]) / 2 - (d["b3"] + d["b4"]) / 2
    thr = dobs / 2
    a = {"d_stage5": d, "delta_obs": dobs, "threshold": thr}
    # coverage facts
    a["coverage"] = []
    for b in DS.BANDS2018:
        ds = DS.load_2018("115", b)
        snr = np.abs(ds.tab.vis) / ds.tab.sigma
        a["coverage"].append({"band": b, "stations": sorted(set(ds.tab.t1) | set(ds.tab.t2)), "n_cp": len(ds.cp),
                              "n_lca": len(ds.lca), "n_scanavg": len(ds.tab), "median_snr": float(np.median(snr)),
                              "rho_max_glambda": float(ds.tab.rho.max() / 1e9),
                              "n_scans": int(len(np.unique(ds.tab.scan)))})
    # H1: drop GL from b3, b4
    h1 = {b: dmed(emcee_fit(f"y6_H1_115_{b}_noGL", Problem("M3", [DS.load_2018("115", b, drop_stations=("GL",))]),
                            S + i)) for i, b in enumerate(("b3", "b4"))}
    shift = (h1["b3"] + h1["b4"]) / 2 - (d["b3"] + d["b4"]) / 2
    a["H1"] = {"d_noGL": h1, "shift": shift, "verdict": "지지" if shift <= thr else "기각"}
    # H2: synthetic on b1, b2 coverage, truth = stage 5 4/25 b3 M3 best fit
    fb3 = json.loads((RES / "stage5_fits" / "y2018_M3_115_b3.json").read_text())
    tb = dict(zip(fb3["names"], fb3["best"]))
    h2 = {}
    for j, b in enumerate(("b1", "b2")):
        raw = DS.load_2018("115", b).raw
        vals = []
        for rep in range(10):
            vis = synth.FLUX * M.mring1(raw.u, raw.v, tb["d"], tb["w"], tb["beta"], tb["beta_pa"])
            syn = synth.corrupt(raw, vis, synth.seed_for("y6H2", b, rep))
            dd = DS.build(syn, f"syn 115 {b} {rep}", "115", b)
            vals.append(dmed(emcee_fit(f"y6_H2_115_{b}_rep{rep}", Problem("M3", [dd]), S + 100 + 10 * j + rep)))
        h2[b] = {"d": vals, "bias": float(np.mean(vals) - tb["d"]), "spread": float(np.std(vals, ddof=1))}
    bias = (h2["b1"]["bias"] + h2["b2"]["bias"]) / 2
    spread = (h2["b1"]["spread"] + h2["b2"]["spread"]) / 2
    if bias <= thr:
        v2 = "coverage 편향 지지"
    elif abs(dobs) <= 2 * spread:
        v2 = "통계 요동으로 설명 가능"
    else:
        v2 = "기각"
    a["H2"] = {"truth": tb, "per_band": h2, "mean_bias": bias, "mean_spread": spread, "verdict": v2}
    # H3: drop one station from b1, b2
    h3 = {}
    for j, b in enumerate(("b1", "b2")):
        st_here = sorted(set(DS.load_2018("115", b).tab.t1) | set(DS.load_2018("115", b).tab.t2))
        h3[b] = {}
        for k, st in enumerate(st_here):
            dd = DS.load_2018("115", b, drop_stations=(st,))
            if dd.n_data <= 4:
                h3[b][st] = None
                continue
            h3[b][st] = dmed(emcee_fit(f"y6_H3_115_{b}_no{st}", Problem("M3", [dd]), S + 200 + 10 * j + k))
    hits = [str(st) for st in h3["b1"] if h3["b1"].get(st) is not None and h3["b2"].get(st) is not None
            and h3["b1"][st] - d["b1"] >= -thr and h3["b2"][st] - d["b2"] >= -thr]
    a["H3"] = {"d_drop": h3, "stations_raising_both": hits, "verdict": f"해당 국 영향: {hits}" if hits else "기각"}
    out["a_0425"] = a
    print("A", json.dumps({k: a[k]["verdict"] for k in ("H1", "H2", "H3")}, ensure_ascii=False), flush=True)

    # ---- (b) 2018 budget with C ----
    prim = [DS.load_2018(doy, b) for doy in ("111", "115") for b in DS.BANDS2018]
    fC = single(f"y6_{C}_joint", joint6(C, prim), S + 300)
    fM = single("y6_M3_joint", joint6("M3", prim), S + 301)
    d18, s_raw = dmed(fC), sym(fC["summary"]["d"])
    stat = s_raw * max(1.0, np.sqrt(fC["chi2_nu"]))
    jk = {"station": {}, "day": {}, "band": {}}
    for i, st in enumerate(ST18):
        sub = [DS.load_2018(doy, b, drop_stations=(st,)) for doy in ("111", "115") for b in DS.BANDS2018]
        jk["station"][st] = dmed(single(f"y6_{C}_jk_station_{st}", joint6(C, sub), S + 310 + i))
    for i, doy in enumerate(("111", "115")):
        sub = [DS.load_2018(doy, b) for b in DS.BANDS2018]
        jk["day"][doy] = dmed(single(f"y6_{C}_jk_day_{doy}", joint6(C, sub), S + 330 + i))
    for i, bands in enumerate((("b1", "b2"), ("b3", "b4"))):
        sub = [DS.load_2018(doy, b) for doy in ("111", "115") for b in bands]
        jk["band"]["+".join(bands)] = dmed(single(f"y6_{C}_jk_band_{'_'.join(bands)}", joint6(C, sub), S + 340 + i))
    sig = {"station": jk_sigma(list(jk["station"].values())), "day": jk_sigma(list(jk["day"].values())),
           "band": abs(jk["band"]["b1+b2"] - jk["band"]["b3+b4"]) / 2}
    overlap = S6["overlap_test"]["verdict"] == "겹침"
    m3_diff = abs(dmed(fM) - d18)
    sig["model"] = S6["budget"]["sys_components"]["model"] if overlap else m3_diff
    cut = [DS.load_2018(doy, b, drop_colocated=True) for doy in ("111", "115") for b in DS.BANDS2018]
    fCc = single(f"y6_{C}_colocut", joint6(C, cut), S + 350)
    sig["set"] = abs(dmed(fCc) - d18)
    sys_ = float(np.sqrt(sum(v**2 for v in sig.values())))
    tot = float(np.hypot(stat, sys_))
    out["b_budget"] = {"d": d18, "sigma_raw": s_raw, "chi2_nu": fC["chi2_nu"], "stat": stat, "jackknife": jk,
                       "sys_components": sig, "sys": sys_, "total": tot, "d_M3_joint": dmed(fM),
                       "abs_M3_minus_C": m3_diff, "d_C_colocut": dmed(fCc),
                       "model_term_rule": "2017 overlap -> 2017 sigma_model transferred" if overlap
                       else "|d_M3 - d_C| (2018)", "lnZ_C": fC["logz"], "lnZ_M3": fM["logz"]}
    print("B", json.dumps(out["b_budget"], default=float), flush=True)

    # ---- (c) 2018 vs 2017 ----
    b17 = S6["budget"]
    z = (d18 - b17["d"]) / np.hypot(tot, b17["total"])
    zs = (d18 - b17["d"]) / np.hypot(stat, b17["stat"])
    verdict = "같음" if abs(z) <= 1 else ("약한 차이" if abs(z) <= 2 else "다름")
    out["c_vs_2017"] = {"d2017": b17["d"], "tot2017": b17["total"], "z": z, "verdict": verdict, "z_stat_only": zs}
    # ---- (d) vs EHT 2018 ----
    diff = d18 - EHT18[0]
    se = EHT18[2] if diff > 0 else EHT18[1]
    comb = float(np.hypot(tot, se))
    out["d_vs_EHT2018"] = {"diff": diff, "combined_sigma": comb, "z": diff / comb,
                           "verdict": "재현" if abs(diff) <= comb else "불일치"}
    (RES / "stage6_2018.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    print("CD", json.dumps({"c": out["c_vs_2017"], "d": out["d_vs_EHT2018"]}, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
