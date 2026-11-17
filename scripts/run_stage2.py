"""Stage 2: closure-only model fits, joint diameter, verdict, and mass.

Outputs (results/):
  stage2_fits.json        per-dataset fits for M1, M2, M3, D, G (BIC table)
  stage2_joint.json       joint M3 fit, sigma budget, pre-registered verdict, per-day comparison
  stage2_robustness.json  10-s closure variant and no-LMT variant (joint M3)
  stage2_mass.json        theta_g and mass, comparison with literature masses
  fig_stage2_diameters.png, fig_stage2_joint_posterior.png, fig_stage2_closure_fit.png,
  fig_mass_comparison.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import literature as L, plotstyle  # noqa: E402
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.closure import model_closure_phase  # noqa: E402
from bhshadow.fitting import Problem, run_mcmc  # noqa: E402
from bhshadow.mass import compare, mass_mc  # noqa: E402
from bhshadow.models import mring1  # noqa: E402
from bhshadow.robust import incoherent_dataset  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "chains").mkdir(parents=True, exist_ok=True)
plotstyle.apply()
SEED = 20170411
GROUPS = [0, 0, 1, 1, 2, 2, 3, 3]  # day index for (lo, hi) of each day


def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def sym(s):
    return 0.5 * (s["minus"] + s["plus"])


def joint_fit(dsets, seed, tag):
    prob = Problem("M3", dsets, groups=GROUPS, shared=("d", "w"))
    t0 = time.time()
    r = run_mcmc(prob, seed)
    r["runtime_s"] = time.time() - t0
    np.savez_compressed(RES / "chains" / f"joint_{tag}.npz", chain=r["_chain"], names=np.array(prob.names))
    return prob, r


def main():
    dsets = DS.load_all()
    fits = {}
    t0 = time.time()
    for i, d in enumerate(dsets):
        fits[d.name] = {}
        for j, m in enumerate(["M1", "M2", "M3", "D", "G"]):
            r = run_mcmc(Problem(m, [d]), SEED + 100 * i + j)
            fits[d.name][m] = strip(r)
            if m == "M3":
                np.savez_compressed(RES / "chains" / f"M3_{d.doy}_{d.band}.npz", chain=r["_chain"])
        print(d.name, {m: round(fits[d.name][m]["chi2_nu"], 2) for m in fits[d.name]},
              "M3 d =", round(fits[d.name]["M3"]["summary"]["d"]["median"], 2), f"{time.time() - t0:.0f}s", flush=True)
    for name in fits:
        bics = {m: fits[name][m]["bic"] for m in fits[name]}
        best = min(bics, key=bics.get)
        fits[name]["_bic_rank"] = sorted(bics, key=bics.get)
        fits[name]["_delta_bic_vs_best"] = {m: bics[m] - bics[best] for m in bics}
    (RES / "stage2_fits.json").write_text(json.dumps(fits, indent=1))

    # joint M3 fit (pre-registered primary)
    prob, J = joint_fit(dsets, SEED + 7, "primary")
    s = J["summary"]
    d_med, d_raw = s["d"]["median"], sym(s["d"])
    scale = max(1.0, np.sqrt(J["chi2_nu"]))
    d_scaled = d_raw * scale
    per = np.array([fits[d.name]["M3"]["summary"]["d"]["median"] for d in dsets])
    s8 = float(np.std(per, ddof=1))
    sig_mine = float(np.hypot(d_scaled, s8))
    comb = float(np.hypot(sig_mine, L.D_EHT_ERR))
    verdict = "재현" if abs(d_med - L.D_EHT) <= comb else "불일치"
    w_med = s["w"]["median"]
    d_img = d_med * (1 - w_med**2 / (4 * np.log(2) * d_med**2))
    perday = []
    n_pass = 0
    for d in dsets:
        f = fits[d.name]["M3"]
        sc = max(1.0, np.sqrt(f["chi2_nu"]))
        mine, mine_s = f["summary"]["d"]["median"], sym(f["summary"]["d"]) * sc
        e, em, ep = L.TABLE3_XSRING[d.name]
        es = ep if mine > e else em
        tol = float(np.sqrt(mine_s**2 + es**2 + (L.TABLE3_OBS_FRAC * e) ** 2))
        ok = abs(mine - e) <= tol
        n_pass += ok
        perday.append({"dataset": d.name, "d_M3": mine, "sigma_scaled": mine_s, "chi2_nu": f["chi2_nu"],
                       "d_EHT_xsring": e, "diff": mine - e, "tolerance": tol, "within": bool(ok),
                       "beta_pa_M3": f["summary"]["beta_pa"]["median"], "pa_EHT_xsring": L.TABLE3_XSRING_PA[d.name]})
    joint_out = {
        "fit": strip(J), "d_median": d_med, "d_sigma_raw": d_raw, "chi2_nu": J["chi2_nu"],
        "sigma_scale": scale, "d_sigma_scaled": d_scaled, "s8_per_dataset_std": s8,
        "per_dataset_d_M3": per.tolist(), "sigma_mine": sig_mine,
        "w_median": w_med, "d_image_domain_eq30": d_img,
        "verdict": {"criterion": "|d - 42| <= sqrt(sigma_mine^2 + 3^2)", "diff": d_med - L.D_EHT,
                    "combined_sigma": comb, "z": (d_med - L.D_EHT) / comb, "result": verdict,
                    "within_2sigma": bool(abs(d_med - L.D_EHT) <= 2 * comb)},
        "per_day_vs_table3": perday, "per_day_n_within": int(n_pass),
    }
    (RES / "stage2_joint.json").write_text(json.dumps(joint_out, indent=1))
    print("JOINT d =", d_med, "+-", sig_mine, verdict, "chi2nu", J["chi2_nu"], "per-day pass", n_pass, flush=True)

    # robustness
    rob = {}
    inc = [incoherent_dataset(d) for d in dsets]
    _, R1 = joint_fit(inc, SEED + 8, "tenseq")
    rob["closures_from_10s_then_scan_mean"] = {
        "d": R1["summary"]["d"], "w": R1["summary"]["w"], "chi2_nu": R1["chi2_nu"],
        "n_data": R1["n_data"], "converged": R1["converged"]}
    nolmt = DS.load_all(drop_stations=("LM",))
    _, R2 = joint_fit(nolmt, SEED + 9, "nolmt")
    rob["no_LMT"] = {"d": R2["summary"]["d"], "w": R2["summary"]["w"], "chi2_nu": R2["chi2_nu"],
                     "n_data": R2["n_data"], "converged": R2["converged"]}
    rob["primary"] = {"d": s["d"], "w": s["w"], "chi2_nu": J["chi2_nu"], "n_data": J["n_data"]}
    (RES / "stage2_robustness.json").write_text(json.dumps(rob, indent=1))
    print("ROBUST", json.dumps({k: v["d"] for k, v in rob.items()}), flush=True)

    # mass
    mres = {"d_used": d_med, "d_err_used": sig_mine}
    mres["GC_alpha"] = mass_mc(d_med, sig_mine, L.ALPHA_GC, L.ALPHA_GC * L.ALPHA_GC_FRAC, L.DIST_MPC, L.DIST_ERR)
    mres["photon_ring_alpha"] = mass_mc(d_med, sig_mine, L.ALPHA_PR, L.ALPHA_PR_ERR, L.DIST_MPC, L.DIST_ERR)
    mres["schwarzschild_alpha_point"] = {"alpha": L.ALPHA_SCHWARZSCHILD,
                                         "theta_g": d_med / L.ALPHA_SCHWARZSCHILD}
    mres["comparison_GC_alpha"] = {k: {**compare(mres["GC_alpha"]["mass_1e9"], v[:3]), "lit": v}
                                   for k, v in L.MASSES.items()}
    mres["comparison_photon_ring_alpha"] = {k: {**compare(mres["photon_ring_alpha"]["mass_1e9"], v[:3]), "lit": v}
                                            for k, v in L.MASSES.items()}
    (RES / "stage2_mass.json").write_text(json.dumps(mres, indent=1))
    print("MASS GC", mres["GC_alpha"]["mass_1e9"], "PR", mres["photon_ring_alpha"]["mass_1e9"], flush=True)

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(dsets))
    ax.axhspan(L.D_EHT - L.D_EHT_ERR, L.D_EHT + L.D_EHT_ERR, color=plotstyle.GRID, label="EHT Paper VI 42 ± 3 μas")
    ax.axhspan(d_med - sig_mine, d_med + sig_mine, color=plotstyle.SERIES[0], alpha=0.15,
               label=f"this work, joint M3: {d_med:.1f} ± {sig_mine:.1f} μas")
    ax.errorbar(x - 0.1, [p["d_M3"] for p in perday], [p["sigma_scaled"] for p in perday], fmt="o", ms=6,
                color=plotstyle.SERIES[0], label="this work, M3 per data set (σ × √χ²ν)")
    ax.errorbar(x + 0.1, [p["d_EHT_xsring"] for p in perday],
                np.array([[L.TABLE3_XSRING[d.name][1], L.TABLE3_XSRING[d.name][2]] for d in dsets]).T,
                fmt="s", ms=6, color=plotstyle.SERIES[1], label="EHT xs-ring (Paper VI Table 3)")
    ax.set_xticks(x, [d.name.replace("2017-", "") for d in dsets], rotation=30)
    ax.set_ylabel("ring diameter d (μas)")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Ring diameter from closure phases + log closure amplitudes")
    fig.tight_layout(); fig.savefig(RES / "fig_stage2_diameters.png"); plt.close(fig)

    ch = J["_chain"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    axs[0].hist(ch[:, 0], bins=80, color=plotstyle.SERIES[0])
    axs[0].set_xlabel("d (μas)"); axs[0].set_ylabel("samples"); axs[0].set_title("joint M3: diameter")
    axs[1].hist2d(ch[:, 0], ch[:, 1], bins=70, cmap="Blues")
    axs[1].set_xlabel("d (μas)"); axs[1].set_ylabel("blur FWHM w (μas)"); axs[1].set_title("d–w correlation")
    fig.tight_layout(); fig.savefig(RES / "fig_stage2_joint_posterior.png"); plt.close(fig)

    d = dsets[-1]
    g = GROUPS[-1]
    best = np.array(J["best"])
    V = mring1(d.tab.u, d.tab.v, best[prob.slot[("d", g)]], best[prob.slot[("w", g)]],
               best[prob.slot[("beta", g)]], best[prob.slot[("beta_pa", g)]])
    mod = model_closure_phase(V[None, :], d.cp)[0]
    labs = sorted(set(d.cp.label), key=lambda q: -d.cp.label.count(q))[:4]
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.2))
    for ax, lab in zip(axs[:4], labs):
        sel = np.array([q == lab for q in d.cp.label])
        ax.errorbar(d.cp.time[sel], np.rad2deg(d.cp.obs[sel]), np.rad2deg(d.cp.sigma[sel]), fmt="o", ms=3,
                    lw=0.8, color=plotstyle.SERIES[0], label="data")
        ax.plot(d.cp.time[sel], np.rad2deg(np.angle(np.exp(1j * mod[sel]))), "s", ms=4, mfc="none",
                color=plotstyle.SERIES[1], label="joint M3")
        ax.set_title(lab); ax.set_xlabel("UTC (h)")
    axs[0].set_ylabel("closure phase (deg)"); axs[0].legend(fontsize=7)
    res = np.angle(np.exp(1j * (d.cp.obs - mod))) / d.cp.sigma
    axs[4].hist(res, bins=30, color=plotstyle.SERIES[0])
    axs[4].set_title("normalised residuals (all triangles)"); axs[4].set_xlabel("(data − model)/σ")
    fig.suptitle("2017-04-11 hi: closure phases vs the joint M3 model", color=plotstyle.TEXT)
    fig.tight_layout(); fig.savefig(RES / "fig_stage2_closure_fit.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    rows = [("this work (α = 11.56 ± 1.32)", mres["GC_alpha"]["mass_1e9"], plotstyle.SERIES[0]),
            ("this work (α = 10.0 ± 0.4)", mres["photon_ring_alpha"]["mass_1e9"], plotstyle.SERIES[0])]
    rows += [(k, v[:3], plotstyle.TEXT2) for k, v in L.MASSES.items()]
    for i, (lab, (v, lo, hi), c) in enumerate(rows[::-1]):
        ax.errorbar(v, i, xerr=[[lo], [hi]], fmt="o", color=c, ms=6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlabel("M (10⁹ M☉), D = 16.8 Mpc")
    ax.set_title("M87* mass: this work vs literature")
    fig.tight_layout(); fig.savefig(RES / "fig_mass_comparison.png"); plt.close(fig)


if __name__ == "__main__":
    main()
