"""Stage 1: data, uv coverage, amplitude vs baseline, first null, closure quantities.

Outputs (results/):
  stage1_reader_check.json   uvfits reader vs the release csv files
  stage1_data_summary.json   counts per data set
  stage1_null.json           first-null fits (PREREGISTRATION 7.1)
  closure_<doy>_<band>.csv   closure phases and log closure amplitudes
  fig_uv_coverage.png, fig_amp_vs_baseline.png, fig_null_fit.png, fig_closure_phases.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import DAYS, plotstyle  # noqa: E402
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.models import blurred_ring  # noqa: E402
from bhshadow.nullfit import null_fit  # noqa: E402
from bhshadow.uvdata import prepare, read_uvfits  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
plotstyle.apply()
BANDS = ("lo", "hi")


def reader_check():
    rows = []
    for doy in DAYS:
        for band in BANDS:
            raw = read_uvfits(DS.uvfits_path(doy, band), conjugate=False)
            csvp = ROOT / "data/raw/csv" / f"SR1_M87_2017_{doy}_{band}_hops_netcal_StokesI.csv"
            c = np.genfromtxt(csvp, delimiter=",", comments="#", dtype=None, encoding=None)
            ct = np.array([r[0] for r in c], float)
            cu, cv = np.array([r[3] for r in c], float), np.array([r[4] for r in c], float)
            ca, cph, cs = (np.array([r[i] for r in c], float) for i in (5, 6, 7))
            same_len = len(ct) == len(raw)
            rows.append({
                "dataset": f"{DAYS[doy]} {band}", "n_uvfits": len(raw), "n_csv": len(ct),
                "rr_equals_ll": raw.meta["rr_eq_ll"],
                "max_rel_diff_u": float(np.max(np.abs(raw.u - cu) / np.maximum(np.abs(cu), 1))) if same_len else None,
                "max_rel_diff_v": float(np.max(np.abs(raw.v - cv) / np.maximum(np.abs(cv), 1))) if same_len else None,
                "max_rel_diff_amp": float(np.max(np.abs(np.abs(raw.vis) - ca) / ca)) if same_len else None,
                "max_rel_diff_sigma": float(np.max(np.abs(raw.sigma - cs) / cs)) if same_len else None,
                "max_abs_diff_phase_rad": float(np.max(np.abs(np.angle(np.exp(1j * (np.angle(raw.vis) - np.deg2rad(cph))))))) if same_len else None,
                "max_abs_diff_time_s": float(np.max(np.abs(raw.time - ct)) * 3600) if same_len else None,
            })
    ok = all(r["n_uvfits"] == r["n_csv"] and max(r["max_rel_diff_u"], r["max_rel_diff_v"], r["max_rel_diff_amp"],
                                                  r["max_rel_diff_sigma"]) <= 1e-4 for r in rows)
    out = {"criterion": "relative difference <= 1e-4 (PREREGISTRATION 3)", "pass": ok, "rows": rows}
    (RES / "stage1_reader_check.json").write_text(json.dumps(out, indent=1))
    return out


def ehtim_crosscheck(ds):
    """Compare 10-s closure phases with eht-imaging on one data set (if installed)."""
    try:
        import ehtim as eh  # noqa: F401
    except ImportError:
        return {"status": "ehtim not installed"}
    from bhshadow.closure import closure_phases
    from bhshadow.uvdata import VisTable
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        obs = eh.obsdata.load_uvfits(str(DS.uvfits_path(ds.doy, ds.band)))
        cps = obs.c_phases(mode="all", count="max", vtype="vis", ang_unit="deg")
    raw = ds.raw
    t = VisTable(raw.time, raw.t1, raw.t2, raw.u, raw.v, raw.vis, raw.sigma, scan=raw.time)
    keep = ~t.is_intrasite()
    mine = closure_phases(t.select(keep))
    # index ehtim closure phases by (time, sorted triangle) with orientation sign
    ref = {}
    for r in cps:
        tri = (r["t1"], r["t2"], r["t3"])
        ref[(round(float(r["time"]), 6), tri)] = float(r["cphase"])
    diffs = []
    for lab, tt, val in zip(mine.label, mine.time, mine.obs):
        a, b, c = lab.split("-")
        for tri, s in (((a, b, c), 1), ((b, c, a), 1), ((c, a, b), 1), ((a, c, b), -1), ((c, b, a), -1), ((b, a, c), -1)):
            key = (round(float(tt), 6), tri)
            if key in ref:
                # ours = conj(file) -> our closure phase = -(ehtim closure phase)
                d = np.angle(np.exp(1j * (val + s * np.deg2rad(ref[key]))))
                diffs.append(abs(d))
                break
    diffs = np.array(diffs)
    return {"status": "ok", "dataset": ds.name, "n_matched": int(len(diffs)), "n_mine": len(mine),
            "max_abs_diff_deg": float(np.rad2deg(diffs.max())) if len(diffs) else None,
            "median_abs_diff_deg": float(np.rad2deg(np.median(diffs))) if len(diffs) else None}


def main():
    rc = reader_check()
    print("reader check pass:", rc["pass"])
    dsets = DS.load_all()
    summary = DS.stack_info(dsets)
    summary_out = {"datasets": summary, "ehtim_closure_phase_check": ehtim_crosscheck(dsets[-1])}
    (RES / "stage1_data_summary.json").write_text(json.dumps(summary_out, indent=1))
    print(json.dumps(summary_out["ehtim_closure_phase_check"]))

    # closure tables
    for d in dsets:
        lines = ["kind,time_h,label,value,sigma"]
        lines += [f"cphase_rad,{t:.5f},{lab},{v:.6f},{s:.6f}" for t, lab, v, s in zip(d.cp.time, d.cp.label, d.cp.obs, d.cp.sigma)]
        lines += [f"logcamp,{t:.5f},{lab},{v:.6f},{s:.6f}" for t, lab, v, s in zip(d.lca.time, d.lca.label, d.lca.obs, d.lca.sigma)]
        (RES / f"closure_{d.doy}_{d.band}.csv").write_text("\n".join(lines) + "\n")

    # uv coverage: one panel per day, lo and hi share uv points (nearly identical)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True, sharey=True)
    for ax, doy in zip(axs, DAYS):
        for d in dsets:
            if d.doy != doy or d.band != "hi":
                continue
            r = d.raw
            for sgn in (1, -1):
                ax.plot(sgn * r.u / 1e9, sgn * r.v / 1e9, ".", ms=1.2, color=plotstyle.SERIES[0])
            ax.set_title(f"{DAYS[doy]} (hi band, 10 s points)")
        ax.set_aspect("equal")
        ax.set_xlim(8.5, -8.5)
        ax.set_xlabel("u (Gλ)  ← east")
    axs[0].set_ylabel("v (Gλ)")
    fig.suptitle("EHT 2017 M87 uv-coverage (points and their conjugates)", color=plotstyle.TEXT)
    fig.tight_layout()
    fig.savefig(RES / "fig_uv_coverage.png")
    plt.close(fig)

    # null fits: per data set, combined, combined without LMT (all non-intra-site, no SNR cut)
    null = {"method": "PREREGISTRATION 7.1", "per_dataset": {}, "combined": None, "combined_no_LMT": None}
    pts = []
    for doy in DAYS:
        for band in BANDS:
            raw = read_uvfits(DS.uvfits_path(doy, band))
            t = prepare(raw, snr_min=0.0)
            amp = np.abs(t.vis)
            sid = np.array([f"{doy}{band}{s}" for s in t.scan])
            null["per_dataset"][f"{DAYS[doy]} {band}"] = null_fit(t.rho, amp, t.sigma, sid)
            pts.append((doy, band, t, amp, sid))
    allr = np.concatenate([p[2].rho for p in pts]); alla = np.concatenate([p[3] for p in pts])
    alls = np.concatenate([p[2].sigma for p in pts]); allid = np.concatenate([p[4] for p in pts])
    null["combined"] = null_fit(allr, alla, alls, allid)
    nolmt = np.concatenate([(p[2].t1 != "LM") & (p[2].t2 != "LM") for p in pts])
    null["combined_no_LMT"] = null_fit(allr[nolmt], alla[nolmt], alls[nolmt], allid[nolmt])
    b1 = null["combined"]["b1_glambda"]
    null["verdict_b1"] = "재현" if 3.2 <= b1 <= 3.6 else "불일치"
    dn, dns = null["combined"]["d_null_muas"], null["combined"]["d_null_boot_std"]
    null["d_null_vs_42pm3"] = {"diff_muas": dn - 42.0, "combined_sigma": float(np.hypot(dns, 3.0)),
                               "within_1sigma": bool(abs(dn - 42.0) <= np.hypot(dns, 3.0))}
    (RES / "stage1_null.json").write_text(json.dumps(null, indent=1))
    print("null combined:", json.dumps(null["combined"]), null["verdict_b1"])

    # amplitude vs baseline, one panel per day
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    rr = np.linspace(0.01, 9, 600) * 1e9
    for ax, doy in zip(axs.ravel(), DAYS):
        for k, band in enumerate(BANDS):
            p = [q for q in pts if q[0] == doy and q[1] == band][0]
            t, amp = p[2], p[3]
            ax.errorbar(t.rho / 1e9, amp, t.sigma, fmt=plotstyle.MARKERS[k], ms=3.2, lw=0.6,
                        color=plotstyle.SERIES[k], label=f"{band} band, scan avg.", alpha=0.85)
        ax.axvline(b1, color=plotstyle.TEXT2, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(2e-3, 1.5)
        ax.set_title(DAYS[doy])
    for ax in axs[1]:
        ax.set_xlabel("baseline length (Gλ)")
    for ax in axs[:, 0]:
        ax.set_ylabel("debiased amplitude (Jy)")
    scale = np.median(np.concatenate([q[3][(q[2].rho > 0.5e9) & (q[2].rho < 1.5e9)] for q in pts]))
    ring = np.abs(blurred_ring(rr, 0 * rr, dn, 0.0))
    norm = scale / np.interp(1e9, rr, ring)
    for ax in axs.ravel():
        ax.plot(rr / 1e9, norm * ring, color=plotstyle.TEXT2, lw=1.2, label=f"thin ring |J0|, d = {dn:.1f} μas")
    axs[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Visibility amplitude vs baseline (intra-site excluded); dashed: fitted first null {b1:.2f} Gλ",
                 color=plotstyle.TEXT)
    fig.tight_layout()
    fig.savefig(RES / "fig_amp_vs_baseline.png")
    plt.close(fig)

    # null-fit zoom
    fig, ax = plt.subplots(figsize=(6, 4))
    m = (allr >= 2.5e9) & (allr <= 4.5e9)
    ax.errorbar(allr[m] / 1e9, alla[m], alls[m], fmt="o", ms=3, lw=0.6, color=plotstyle.SERIES[0],
                label="all 8 data sets, 2.5–4.5 Gλ")
    c = null["combined"]
    xx = np.linspace(2.5, 4.5, 200)
    ax.plot(xx, np.sqrt((c["k"] * (xx - c["b1_glambda"])) ** 2 + c["m"] ** 2), color=plotstyle.SERIES[1],
            label=f"fit: b₁ = {c['b1_glambda']:.3f} ± {c['b1_boot_std']:.3f} Gλ")
    ax.set_xlabel("baseline length (Gλ)"); ax.set_ylabel("debiased amplitude (Jy)")
    ax.legend(fontsize=8)
    ax.set_title("First-null fit A = √(k²(ρ−b₁)² + m²)")
    fig.tight_layout(); fig.savefig(RES / "fig_null_fit.png"); plt.close(fig)

    # closure phases vs time for April 11 hi, a few triangles with ALMA
    d = [x for x in dsets if x.doy == "101" and x.band == "hi"][0]
    labs = sorted(set(d.cp.label), key=lambda s: -d.cp.label.count(s))[:4]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, lab in zip(axs, labs):
        sel = np.array([x == lab for x in d.cp.label])
        ax.errorbar(d.cp.time[sel], np.rad2deg(d.cp.obs[sel]), np.rad2deg(d.cp.sigma[sel]), fmt="o", ms=3,
                    lw=0.8, color=plotstyle.SERIES[0])
        ax.axhline(0, color=plotstyle.TEXT2, lw=0.8)
        ax.set_title(lab); ax.set_xlabel("UTC (h)")
    axs[0].set_ylabel("closure phase (deg)")
    fig.suptitle("2017-04-11 hi: scan-averaged closure phases (a symmetric source would give 0 or ±180°)",
                 color=plotstyle.TEXT)
    fig.tight_layout(); fig.savefig(RES / "fig_closure_phases.png"); plt.close(fig)


if __name__ == "__main__":
    main()
