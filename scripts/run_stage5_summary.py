"""Stage 5: one summary figure (wide 1440 px and narrow 390 px layouts). PREREGISTRATION 10.10.

Panels: (a) visibility amplitude vs baseline with the joint model, (b) ring diameter:
MCMC posterior (statistical only) with the stage 5 total uncertainty and literature,
(c) mass comparison. Reads results/*.json written by the other stage 5 scripts.
Outputs: results/fig_summary_wide.png (1440 px), results/fig_summary_narrow.png (390 px at 2x = 780 px)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import literature as L, plotstyle  # noqa: E402
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.s5 import joint_problem  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
C0, C1, C2 = plotstyle.SERIES[0], plotstyle.SERIES[1], plotstyle.SERIES[2]


def load():
    st2 = json.loads((RES / "stage2_joint.json").read_text())
    sysj = json.loads((RES / "stage5_systematics.json").read_text())
    y18 = json.loads((RES / "stage5_2018.json").read_text()) if (RES / "stage5_2018.json").exists() else None
    central = sysj["budget"]["central_model"]
    cfit = json.loads((RES / "stage5_fits" / f"ext_{central}_all.json").read_text()) if central != "M3" else st2["fit"]
    prob = joint_problem(central, DS.load_all())
    best = np.array(cfit["best"])
    rho, amp, sig, mod, lmt = [], [], [], [], []
    for d, g in zip(prob.datasets, prob.groups):
        t = d.tab
        V = prob.fn(t.u, t.v, *[best[prob.slot[(p, g)]] for p in prob.pnames])
        rho.append(t.rho / 1e9); amp.append(np.abs(t.vis)); sig.append(t.sigma); mod.append(np.abs(V))
        lmt.append((t.t1 == "LM") | (t.t2 == "LM"))
    rho, amp, sig, mod, lmt = map(np.concatenate, (rho, amp, sig, mod, lmt))
    # one flux factor for display only (closure fits carry no flux scale), non-LMT points
    w = 1 / sig[~lmt] ** 2
    F = float(np.sum(w * amp[~lmt] * mod[~lmt]) / np.sum(w * mod[~lmt] ** 2))
    chains = {"M3": np.load(RES / "chains" / "joint_primary.npz")["chain"][:, 0]}
    if central != "M3":
        chains[central] = np.load(RES / "chains" / f"s5_ext_{central}_all.npz")["chain"][:, 0]
    return st2, sysj, y18, central, (rho, amp, mod * F, lmt, F), chains


def panel_amp(ax, central, data, small):
    rho, amp, modF, lmt, F = data
    ax.scatter(rho[~lmt], amp[~lmt], s=6 if small else 8, color=plotstyle.TEXT2, alpha=0.45, lw=0,
               label="data (scan-averaged)")
    ax.scatter(rho[lmt], amp[lmt], s=6 if small else 8, color=C1, alpha=0.6, lw=0, label="data, LMT baselines")
    ax.scatter(rho, modF, s=4, color=C0, lw=0, label=f"joint model ({central}) at data points")
    ax.set_yscale("log"); ax.set_ylim(2e-3, 1.5)
    ax.set_xlabel("baseline length (Gλ)"); ax.set_ylabel("visibility amplitude (Jy)")
    ax.set_title("(a) Amplitude vs baseline, 2017" if small else "(a) Amplitude vs baseline, 2017 (8 data sets)",
                 loc="left")
    ax.legend(fontsize=7.8 if small else 8, loc="lower left", markerscale=2)
    ax.text(0.98, 0.97, "model scaled by one flux\nfactor, for display only", transform=ax.transAxes, ha="right",
            va="top", fontsize=7.2 if small else 7.5, color=plotstyle.TEXT2)


def panel_d(ax, st2, sysj, y18, chains, central, small):
    b = sysj["budget"]
    d, stat, tot = b["d"], b["stat"], b["total"]
    # posteriors (statistical only) as spikes in the top band
    for m, c in (("M3", plotstyle.TEXT2), (central, C0)):
        if m not in chains:
            continue
        h, edges = np.histogram(chains[m], bins=40, density=True)
        ax.fill_between(0.5 * (edges[1:] + edges[:-1]), 5.2, 5.2 + 0.75 * h / h.max(), color=c, step="mid")
        ax.text(np.median(chains[m]) + 0.4, 5.75, m, fontsize=7.8 if small else 8, color=c, va="center")
    ax.text(46.9, 5.6, "stat-only\nposteriors", fontsize=7.6 if small else 7.8, color=plotstyle.TEXT2,
            va="center")
    rows = [(4.3, f"this work 2017, {central}: {d:.1f} ± {tot:.1f}", d, tot, tot, "o", C0),
            (3.5, f"this work 2017, M3: {st2['d_median']:.1f} ± {st2['sigma_mine']:.1f}", st2["d_median"],
             st2["sigma_mine"], st2["sigma_mine"], "o", plotstyle.TEXT2)]
    xs = sysj["xsring"]["per_dataset"]
    xd = np.array([r["X1"]["d_hat"]["median"] for r in xs])
    rows.append((2.7, f"this work 2017, xs-ring d̂: {xd.mean():.1f} ± {xd.std(ddof=1):.1f}", xd.mean(),
                 xd.std(ddof=1), xd.std(ddof=1), "D", C2))
    rows.append((1.9, "EHT 2017: 42 ± 3", 42.0, 3.0, 3.0, "^", plotstyle.TEXT))
    if y18:
        j = y18["joint"]
        rows.append((1.1, f"this work 2018, M3: {j['d']:.1f} ± {j['sigma_2018']:.1f}", j["d"], j["sigma_2018"],
                     j["sigma_2018"], "s", C1))
        rows.append((0.3, "EHT 2018: 43.3 (+1.5 −3.1)", 43.3, 3.1, 1.5, "^", plotstyle.TEXT))
    for y, lab, v, lo, hi, mk, c in rows:
        ax.errorbar([v], [y], xerr=[[lo], [hi]], fmt=mk, color=c, ms=5, capsize=0)
        ax.text(35.3, y + 0.28, lab, fontsize=7.6 if small else 7.8, color=plotstyle.TEXT, va="center")
    ax.set_xlim(35, 52); ax.set_ylim(-0.2, 6.2); ax.set_yticks([])
    ax.set_xlabel("ring diameter (μas)")
    ax.set_title(f"(b) d = {d:.1f} ± {stat:.2f} (stat) ± {b['sys']:.1f} (sys) μas" if not small else
                 f"(b) d = {d:.1f} ± {stat:.2f} ± {b['sys']:.1f} μas", loc="left")


def panel_mass(ax, sysj, small):
    m = sysj["mass"]
    rows = [("this work, α = 11.56", m["GC_alpha"]["mass_1e9"], C0),
            ("this work, α = 10.0", m["photon_ring_alpha"]["mass_1e9"], C0)]
    rows += [(k.replace(" (2019)", ""), v[:3], plotstyle.TEXT2) for k, v in L.MASSES.items()]
    for i, (lab, (v, lo, hi), c) in enumerate(rows[::-1]):
        ax.errorbar(v, i, xerr=[[lo], [hi]], fmt="o", color=c, ms=5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=7.8 if small else 8)
    ax.set_xlabel("M (10⁹ M☉), D = 16.8 Mpc")
    ax.set_title("(c) Mass vs literature", loc="left")


def fig_systematics(sysj):
    jk, ext, xs = sysj["jackknife"], sysj["model_extensions"], sysj["xsring"]["per_dataset"]
    ref = sysj["reference_M3_joint"]["d"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 0.8, 1.3]})
    ax = axs[0]
    labs, vals, cols = [], [], []
    for k, v in jk["station"].items():
        labs.append(f"−{k}"); vals.append(v["d"]["median"]); cols.append(C0)
    for k, v in jk["day"].items():
        labs.append(f"−{k[5:]}"); vals.append(v["d"]["median"]); cols.append(C2)
    for k, v in jk["band"].items():
        labs.append(f"{k} only"); vals.append(v["d"]["median"]); cols.append(C1)
    x = np.arange(len(vals))
    ax.axhline(ref, color=plotstyle.TEXT2, lw=1, ls="--", label=f"all data (M3): {ref:.2f}")
    ax.scatter(x, vals, c=cols, s=36, zorder=3)
    ax.set_xticks(x, labs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("joint M3 diameter d (μas)")
    ax.set_title(f"Jackknife (M3): σ_station = {jk['sigma_station']:.2f}, σ_day = {jk['sigma_day']:.2f}", fontsize=9)
    ax.legend(fontsize=8)
    ax = axs[1]
    ms = ["M3", "E1", "E2"]
    for i, m in enumerate(ms):
        e = ext[m]
        ax.errorbar(i - 0.1, e["d"]["median"], yerr=[[e["d"]["minus"]], [e["d"]["plus"]]], fmt="o", color=C0,
                    label="all stations" if i == 0 else None)
        if "noLMT" in e:
            ax.plot(i + 0.1, e["noLMT"]["d"]["median"], "s", color=C1, label="without LMT" if i == 1 else None)
        ax.text(i, 37.6, f"χ²ν {e['chi2_nu']:.2f}\nΔBIC {e['bic'] - ext['M3']['bic']:+.0f}", ha="center",
                fontsize=7.5, color=plotstyle.TEXT2)
    rob = json.loads((RES / "stage2_robustness.json").read_text())
    ax.plot(0.1, rob["no_LMT"]["d"]["median"], "s", color=C1)
    ax.set_xticks(range(3), ["M3", "E1 (+Gauss)", "E2 (+m=2)"]); ax.set_ylim(37, 47.5)
    ax.set_ylabel("joint diameter d (μas)"); ax.set_title(f"Model extensions (central: {ext['central_model']})", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax = axs[2]
    x = np.arange(len(xs))
    ax.plot(x - 0.15, [r["M3"]["d"] for r in xs], "o", color=C0, label="this work, m-ring d (M3)")
    ax.errorbar(x, [r["X1"]["d_hat"]["median"] for r in xs], [r["X1"]["d_hat_sigma_scaled"] for r in xs], fmt="D",
                color=C2, label="this work, xs-ring d̂ (X1)")
    ax.errorbar(x + 0.15, [r["EHT_table3"]["d_hat"] for r in xs],
                np.array([[r["EHT_table3"]["minus"], r["EHT_table3"]["plus"]] for r in xs]).T, fmt="s", color=C1,
                label="EHT xs-ring d̂ (Paper VI Table 3)")
    ax.set_xticks(x, [r["dataset"][5:] for r in xs], rotation=30, fontsize=8)
    ax.set_ylabel("diameter (μas)")
    ax.set_title(f"m-ring vs xs-ring: verdict '{ {'정의 차이': 'definition', '구현 차이': 'implementation', '둘 다': 'both'}[sysj['xsring']['verdict']]}' (Δdef {sysj['xsring']['mean_delta_def']:.1f}, "
                 f"Δimpl {sysj['xsring']['mean_delta_impl']:+.1f})", fontsize=9)
    ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(RES / "fig_stage5_systematics.png"); plt.close(fig)


def main():
    st2, sysj, y18, central, data, chains = load()
    plotstyle.apply()
    fig_systematics(sysj)
    plotstyle.apply()
    # wide: 1440 px = 14.4 in at 100 dpi
    fig = plt.figure(figsize=(14.4, 4.6), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.1, 0.95], wspace=0.45)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    panel_amp(fig.add_subplot(gs[0]), central, data, False)
    panel_d(fig.add_subplot(gs[1]), st2, sysj, y18, chains, central, False)
    panel_mass(fig.add_subplot(gs[2]), sysj, False)
    fig.suptitle("M87* ring from EHT public data (2017, 2018): closure-only fits", x=0.01, ha="left",
                 fontsize=12, color=plotstyle.TEXT)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.13)
    fig.savefig(RES / "fig_summary_wide.png", dpi=100); plt.close(fig)
    # narrow: 390 px css width, saved at 2x (780 px)
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9})
    fig, axs = plt.subplots(3, 1, figsize=(3.9, 10.6), dpi=200, gridspec_kw={"height_ratios": [1, 0.95, 0.95]})
    panel_amp(axs[0], central, data, True)
    panel_d(axs[1], st2, sysj, y18, chains, central, True)
    panel_mass(axs[2], sysj, True)
    fig.suptitle("M87* ring, EHT public data", x=0.03, ha="left", fontsize=10, color=plotstyle.TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(RES / "fig_summary_narrow.png", dpi=200); plt.close(fig)
    print("flux scale for display:", data[4])


if __name__ == "__main__":
    main()
