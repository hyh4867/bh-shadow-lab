"""Stage 5 (additional analysis): EHT 2018 M87 data with the same pipeline. PREREGISTRATION 10.8.

Primary: 2018-04-21 and 2018-04-25, bands b1-b4, EHT-HOPS netcal files (8 data sets).
Supplementary: 2018-04-22 (per-data-set fits only).
Outputs: results/stage5_2018.json, results/fig_stage5_2018.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import plotstyle  # noqa: E402
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.fitting import Problem, circ_stats  # noqa: E402
from bhshadow.s5 import RES, SEED, cached_fit, joint_problem, sym  # noqa: E402
from bhshadow.uvdata import read_uvfits  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EHT2018 = (43.3, 3.1, 1.5)  # A&A 681, A79: abstract p.1, Table 12 p.30 (median, minus, plus)
EHT2018_PA = (208.0, 5.0)  # Table 12 p.30
# Table 11 p.30, THEMIS xs-ringauss (median, minus, plus) and PA
TABLE11 = {
    "2018-04-21 b1": (44.3, 0.88, 0.88, 208.3), "2018-04-21 b2": (45.2, 0.48, 0.89, 206.2),
    "2018-04-21 b3": (43.6, 0.38, 2.72, 203.3), "2018-04-21 b4": (44.4, 0.56, 2.02, 198.6),
    "2018-04-25 b1": (43.8, 1.11, 1.35, 191.5), "2018-04-25 b2": (43.7, 0.72, 1.12, 194.4),
    "2018-04-25 b3": (44.6, 0.60, 0.65, 207.7), "2018-04-25 b4": (44.5, 0.55, 0.45, 199.8),
}


def reader_check():
    p = DS.uvfits_path_2018("111", "b3")
    raw = read_uvfits(p, conjugate=False)
    c = np.genfromtxt(ROOT / "data/raw/csv2018" / (p.stem + ".csv"), delimiter=",", comments="#", dtype=None,
                      encoding=None)
    cu, cv = np.array([r[3] for r in c], float), np.array([r[4] for r in c], float)
    ca, cph, cs = (np.array([r[i] for r in c], float) for i in (5, 6, 7))
    n_ok = len(ca) == len(raw)
    return {"dataset": "2018-04-21 b3", "n_uvfits": len(raw), "n_csv": len(ca), "rr_equals_ll": raw.meta["rr_eq_ll"],
            "max_rel_diff_u": float(np.max(np.abs(raw.u - cu) / np.maximum(np.abs(cu), 1))) if n_ok else None,
            "max_rel_diff_amp": float(np.max(np.abs(np.abs(raw.vis) - ca) / ca)) if n_ok else None,
            "max_rel_diff_sigma": float(np.max(np.abs(raw.sigma - cs) / cs)) if n_ok else None,
            "max_abs_diff_phase_deg": float(np.max(np.abs((np.rad2deg(np.angle(raw.vis)) - cph + 180) % 360 - 180)))
            if n_ok else None}


def circ_mean_std(deg):
    a = np.deg2rad(np.asarray(deg))
    R = np.hypot(np.sin(a).mean(), np.cos(a).mean())
    return float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360), float(
        np.rad2deg(np.sqrt(-2 * np.log(min(R, 1.0)))))


def main():
    out = {"note": "추가 분석 (PREREGISTRATION 10.8)", "reader_check": reader_check()}
    prim = [DS.load_2018(doy, b) for doy in ("111", "115") for b in DS.BANDS2018]
    supp = [DS.load_2018("112", b) for b in DS.BANDS2018]
    out["data"] = DS.stack_info(prim + supp)
    per = []
    for i, d in enumerate(prim + supp):
        f = cached_fit(f"y2018_M3_{d.doy}_{d.band}", Problem("M3", [d]), SEED + 3000 + i)
        sc = max(1.0, np.sqrt(f["chi2_nu"]))
        row = {"dataset": d.name, "primary": d.doy != "112", "d": f["summary"]["d"]["median"],
               "sigma_scaled": sym(f["summary"]["d"]) * sc, "w": f["summary"]["w"]["median"],
               "beta": f["summary"]["beta"]["median"], "pa": f["summary"]["beta_pa"]["median"],
               "pa_sigma_scaled": sym(f["summary"]["beta_pa"]) * sc, "chi2_nu": f["chi2_nu"],
               "n_data": f["n_data"], "converged": f["converged"]}
        if d.name in TABLE11:
            row["EHT_table11"] = TABLE11[d.name]
        per.append(row)
    out["per_dataset"] = per
    J = cached_fit("y2018_M3_joint", joint_problem("M3", prim), SEED + 3100, save_chain=True)
    Jn = cached_fit("y2018_M3_joint_noLMT", joint_problem("M3", [DS.load_2018(doy, b, drop_stations=("LM",))
                                                                 for doy in ("111", "115") for b in DS.BANDS2018]),
                    SEED + 3101)
    d18, raw18 = J["summary"]["d"]["median"], sym(J["summary"]["d"])
    scaled = raw18 * max(1.0, np.sqrt(J["chi2_nu"]))
    s8 = float(np.std([r["d"] for r in per if r["primary"]], ddof=1))
    sig18 = float(np.hypot(scaled, s8))
    st2 = json.loads((RES / "stage2_joint.json").read_text())
    d17, sig17 = st2["d_median"], st2["sigma_mine"]
    z = (d18 - d17) / np.hypot(sig18, sig17)
    ver_a = "같음" if abs(z) <= 1 else ("약한 차이" if abs(z) <= 2 else "다름")
    syspath = RES / "stage5_systematics.json"
    full17 = json.loads(syspath.read_text())["budget"] if syspath.exists() else None
    e, em, ep = EHT2018
    es = ep if d18 > e else em
    comb_b = float(np.hypot(sig18, es))
    ver_b = "재현" if abs(d18 - e) <= comb_b else "불일치"
    # position angles per day
    pa18 = [J["summary"][f"beta_pa_{g}"] for g in (0, 1)]
    pa17 = [st2["fit"]["summary"][f"beta_pa_{g}"] for g in range(4)]
    m18, s18 = circ_mean_std([p["median"] for p in pa18])
    m17, s17 = circ_mean_std([p["median"] for p in pa17])
    floor18 = float(np.median([sym(p) for p in pa18]) * np.sqrt(J["chi2_nu"]))
    floor17 = float(np.median([sym(p) for p in pa17]) * np.sqrt(st2["chi2_nu"]))
    s18e, s17e = max(s18, floor18), max(s17, floor17)
    dpa = (m18 - m17 + 180) % 360 - 180
    spa = float(np.sqrt(s17e**2 / 4 + s18e**2 / 2))
    out["joint"] = {"fit": J, "d": d18, "sigma_raw": raw18, "chi2_nu": J["chi2_nu"], "sigma_scaled": scaled,
                    "s_per_dataset_std": s8, "sigma_2018": sig18,
                    "noLMT": {"d": Jn["summary"]["d"], "chi2_nu": Jn["chi2_nu"], "converged": Jn["converged"]}}
    out["verdicts"] = {
        "a_2018_vs_2017": {"d2018": d18, "sigma2018": sig18, "d2017": d17, "sigma2017": sig17, "z": float(z),
                           "result": ver_a,
                           "with_2017_full_budget": None if full17 is None else {
                               "sigma2017_total": full17["total"],
                               "z": float((d18 - full17["d"]) / np.hypot(sig18, full17["total"]))}},
        "b_2018_vs_EHT2018": {"diff": d18 - e, "combined_sigma": comb_b, "z": float((d18 - e) / comb_b),
                              "result": ver_b},
        "d_position_angle": {"pa2018_days": pa18, "pa2017_days": pa17, "mean2018": m18, "mean2017": m17,
                             "s2018": s18e, "s2017": s17e, "s2018_between_days": s18, "s2017_between_days": s17,
                             "delta": dpa, "sigma": spa, "z": dpa / spa,
                             "result": "PA 변화 있음" if abs(dpa) > 2 * spa else "변화 확인 못 함",
                             "EHT": {"2017": "150-200 deg", "2018": EHT2018_PA, "shift": "about 30 deg (abstract p.1)"}},
    }
    (RES / "stage5_2018.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    print(json.dumps(out["verdicts"], indent=1, default=float, ensure_ascii=False))

    plotstyle.apply()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ax = axs[0]
    pr = [r for r in per if r["primary"]]
    x = np.arange(len(pr))
    ax.axhspan(e - em, e + ep, color=plotstyle.GRID, label="EHT 2018: 43.3 (+1.5 −3.1) μas")
    ax.axhspan(d18 - sig18, d18 + sig18, color=plotstyle.SERIES[0], alpha=0.15,
               label=f"this work 2018 joint: {d18:.1f} ± {sig18:.1f}")
    ax.axhline(d17, color=plotstyle.SERIES[2], ls="--", lw=1.5, label=f"this work 2017 joint: {d17:.1f}")
    ax.errorbar(x - 0.1, [r["d"] for r in pr], [r["sigma_scaled"] for r in pr], fmt="o", color=plotstyle.SERIES[0],
                label="this work, M3 per data set")
    ax.errorbar(x + 0.1, [TABLE11[r["dataset"]][0] for r in pr],
                np.array([[TABLE11[r["dataset"]][1], TABLE11[r["dataset"]][2]] for r in pr]).T, fmt="s",
                color=plotstyle.SERIES[1], label="EHT THEMIS xs-ringauss (Table 11)")
    ax.set_xticks(x, [r["dataset"].replace("2018-", "") for r in pr], rotation=30)
    ax.set_ylabel("ring diameter (μas)"); ax.set_title("2018 ring diameter"); ax.legend(fontsize=7)
    ax = axs[1]
    for yr, pas, c in (("2017", pa17, plotstyle.SERIES[2]), ("2018", pa18, plotstyle.SERIES[0])):
        ax.errorbar(np.arange(len(pas)) + (0 if yr == "2017" else 4.5), [p["median"] for p in pas],
                    [sym(p) for p in pas], fmt="o", color=c, label=f"this work {yr} (joint fit, per day)")
    ax.axhspan(150, 200, color=plotstyle.GRID, alpha=0.6, label="EHT 2017: 150–200°")
    ax.axhspan(203, 213, color=plotstyle.SERIES[1], alpha=0.2, label="EHT 2018: 208 ± 5°")
    ax.set_xticks([0, 1, 2, 3, 4.5, 5.5], ["17-04-05", "04-06", "04-10", "04-11", "18-04-21", "04-25"], rotation=30)
    ax.set_ylabel("bright-side position angle (deg E of N)"); ax.set_title("Brightness position angle")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RES / "fig_stage5_2018.png"); plt.close(fig)


if __name__ == "__main__":
    main()
