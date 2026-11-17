"""Stage 6: one summary figure (wide 1440 px and narrow 390 px layouts). PREREGISTRATION 11.8.

Panels: (a) visibility amplitude vs baseline with the stage 6 central model C, (b) ring diameter:
nested-sampling posterior (statistical only), the stage 6 total uncertainty, models that did not
converge (reference only, hollow) and literature, (c) mass comparison.
Reads results/stage6_*.json. Outputs: results/fig_summary6_wide.png (1440 px),
results/fig_summary6_narrow.png (390 px at 2x = 780 px).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import literature as L, plotstyle  # noqa: E402
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.nested import load_samples  # noqa: E402
from bhshadow.s6 import NRUNS, joint6  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
C0, C1, C2 = plotstyle.SERIES[0], plotstyle.SERIES[1], plotstyle.SERIES[2]


def jl(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def load():
    M6, S6, X6, Y6 = jl("stage6_models.json"), jl("stage6_systematics.json"), jl("stage6_xsring.json"), jl("stage6_2018.json")
    C = M6["central"]["model"]
    mj = json.loads((RES / "stage6_fits" / f"m6_{C}_multi.json").read_text())
    prob = joint6(C, DS.load_all())
    best = np.array(mj["best"])
    rho, amp, sig, mod, lmt = [], [], [], [], []
    for d, g in zip(prob.datasets, prob.groups):
        t = d.tab
        V = prob.fn(t.u, t.v, *[best[prob.slot[(p, g)]] for p in prob.pnames])
        rho.append(t.rho / 1e9); amp.append(np.abs(t.vis)); sig.append(t.sigma); mod.append(np.abs(V))
        lmt.append((t.t1 == "LM") | (t.t2 == "LM"))
    rho, amp, sig, mod, lmt = map(np.concatenate, (rho, amp, sig, mod, lmt))
    w = 1 / sig[~lmt] ** 2  # one flux factor for display only (closure fits carry no flux scale)
    F = float(np.sum(w * amp[~lmt] * mod[~lmt]) / np.sum(w * mod[~lmt] ** 2))
    post = {}
    for m in dict.fromkeys(("M3", C)):
        pm = joint6(m, DS.load_all())
        post[m] = np.vstack([load_samples(f"m6_{m}_r{r}") for r in range(NRUNS)])[:, pm.names.index("d")]
    return M6, S6, X6, Y6, C, (rho, amp, mod * F, lmt, F), post


def panel_amp(ax, C, data, small):
    rho, amp, modF, lmt, F = data
    ax.scatter(rho[~lmt], amp[~lmt], s=6 if small else 8, color=plotstyle.TEXT2, alpha=0.45, lw=0,
               label="data (scan-averaged)")
    ax.scatter(rho[lmt], amp[lmt], s=6 if small else 8, color=C1, alpha=0.6, lw=0, label="data, LMT baselines")
    ax.scatter(rho, modF, s=4, color=C0, lw=0, label=f"central model ({C}) at data points")
    ax.set_yscale("log"); ax.set_ylim(2e-3, 1.5)
    ax.set_xlabel("baseline length (Gλ)"); ax.set_ylabel("visibility amplitude (Jy)")
    ax.set_title("(a) Amplitude vs baseline, 2017" if small else "(a) Amplitude vs baseline, 2017 (8 data sets)",
                 loc="left")
    ax.legend(fontsize=7.8 if small else 8, loc="lower left", markerscale=2)
    ax.text(0.98, 0.97, "model scaled by one flux\nfactor, for display only", transform=ax.transAxes, ha="right",
            va="top", fontsize=7.2 if small else 7.5, color=plotstyle.TEXT2)


def panel_d(ax, M6, S6, X6, Y6, C, post, small):
    b = S6["budget"]
    d, stat, tot = b["d"], b["stat"], b["total"]
    fs = 7.6 if small else 7.8
    for m, c in (("M3", plotstyle.TEXT2), (C, C0)):
        if m not in post:
            continue
        h, edges = np.histogram(post[m], bins=40, density=True)
        ax.fill_between(0.5 * (edges[1:] + edges[:-1]), 6.0, 6.0 + 0.75 * h / h.max(), color=c, step="mid")
        ax.text(np.median(post[m]) + 0.4, 6.55, m, fontsize=fs, color=c, va="center")
    ax.text(48.2, 6.4, "stat-only\nposteriors", fontsize=fs, color=plotstyle.TEXT2, va="center")
    table = {t["model"]: t for t in M6["table"]}
    m3 = table["M3"]
    rows = [(5.1, f"this work 2017, {C}: {d:.1f} ± {tot:.1f}", d, tot, tot, "o", C0, True),
            (4.3, f"this work 2017, M3: {m3['d']:.1f} (stat ± {m3['d_sigma_raw']:.2f})", m3["d"],
             m3["d_sigma_raw"], m3["d_sigma_raw"], "o", plotstyle.TEXT2, True)]
    nc = [t for t in M6["table"] if not t.get("converged_p", t["converged"])]
    if nc:
        vals = []
        for t in nc:
            j = int(np.argmax(t["lnZ_runs"]))
            vals.append((t["model"], t["d_runs"][j]))
        lo_, hi_ = min(v for _, v in vals), max(v for _, v in vals)
        lab = (f"not converged ({len(vals)} models): {lo_:.1f}–{hi_:.1f}" if small else
               "not converged, max-lnZ run: " + ", ".join(f"{m} {v:.1f}" for m, v in vals))
        for _, v in vals:
            ax.plot([v], [3.5], "o", mfc="none", mec=plotstyle.TEXT2, ms=5)
        ax.text(35.3, 3.78, lab, fontsize=fs, color=plotstyle.TEXT, va="center")
    if X6:
        xs = np.array([r["S4"]["d_hat"] for r in X6["per_dataset"]])
        rows.append((2.7, f"{'' if small else 'this work 2017, '}xs-ring d̂ (S4): {xs.mean():.1f}, spread {xs.std(ddof=1):.1f}", xs.mean(),
                     xs.std(ddof=1), xs.std(ddof=1), "D", C2, True))
    rows.append((1.9, "EHT 2017: 42 ± 3", 42.0, 3.0, 3.0, "^", plotstyle.TEXT, True))
    if Y6:
        yb = Y6["b_budget"]
        rows.append((1.1, f"this work 2018, {C}: {yb['d']:.1f} ± {yb['total']:.1f}", yb["d"], yb["total"],
                     yb["total"], "s", C1, True))
    rows.append((0.3, "EHT 2018: 43.3 (+1.5 −3.1)", 43.3, 3.1, 1.5, "^", plotstyle.TEXT, True))
    for y, lab, v, lo, hi, mk, c, _ in rows:
        ax.errorbar([v], [y], xerr=[[lo], [hi]], fmt=mk, color=c, ms=5, capsize=0)
        ax.text(35.3, y + 0.28, lab, fontsize=fs, color=plotstyle.TEXT, va="center")
    ax.set_xlim(35, 53); ax.set_ylim(-0.2, 7.0); ax.set_yticks([])
    ax.set_xlabel("ring diameter (μas)")
    ax.set_title(f"(b) d = {d:.1f} ± {stat:.2f} (stat) ± {b['sys']:.1f} (sys) μas" if not small else
                 f"(b) d = {d:.1f} ± {stat:.2f} ± {b['sys']:.1f} μas", loc="left")


def panel_mass(ax, S6, small):
    m = S6["mass"]
    rows = [("this work, α = 11.56", m["GC_alpha"]["mass_1e9"], C0),
            ("this work, α = 10.0", m["photon_ring_alpha"]["mass_1e9"], C0)]
    rows += [(k.replace(" (2019)", ""), v[:3], plotstyle.TEXT2) for k, v in L.MASSES.items()]
    for i, (lab, (v, lo, hi), c) in enumerate(rows[::-1]):
        ax.errorbar(v, i, xerr=[[lo], [hi]], fmt="o", color=c, ms=5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=7.8 if small else 8)
    ax.set_xlabel("M (10⁹ M☉), D = 16.8 Mpc")
    ax.set_title("(c) Mass vs literature", loc="left")


def main():
    M6, S6, X6, Y6, C, data, post = load()
    plotstyle.apply()
    fig = plt.figure(figsize=(14.4, 4.8), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 0.9], wspace=0.45)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    panel_amp(fig.add_subplot(gs[0]), C, data, False)
    panel_d(fig.add_subplot(gs[1]), M6, S6, X6, Y6, C, post, False)
    panel_mass(fig.add_subplot(gs[2]), S6, False)
    fig.suptitle("M87* ring from EHT public data (2017, 2018): closure-only fits, nested sampling (stage 6)",
                 x=0.01, ha="left", fontsize=12, color=plotstyle.TEXT)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.12)
    fig.savefig(RES / "fig_summary6_wide.png", dpi=100); plt.close(fig)
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9})
    fig, axs = plt.subplots(3, 1, figsize=(3.9, 11.2), dpi=200, gridspec_kw={"height_ratios": [1, 1.1, 0.95]})
    panel_amp(axs[0], C, data, True)
    panel_d(axs[1], M6, S6, X6, Y6, C, post, True)
    panel_mass(axs[2], S6, True)
    fig.suptitle("M87* ring, EHT public data", x=0.03, ha="left", fontsize=10, color=plotstyle.TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(RES / "fig_summary6_narrow.png", dpi=200); plt.close(fig)
    print("flux scale for display:", data[4])


if __name__ == "__main__":
    main()
