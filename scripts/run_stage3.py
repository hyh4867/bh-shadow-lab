"""Stage 3: injection-recovery on the real 2017-04-06 hi and 2017-04-10 hi coverage.

Outputs (results/):
  stage3_injection.csv    one row per synthetic data set
  stage3_summary.json     bias, scatter, interval coverage, ring/disk classification, verdicts
  fig_stage3_recovery.png, fig_stage3_dbic.png
Resumable: rows already in the csv are skipped.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import plotstyle  # noqa: E402
from bhshadow import dataset as DS  # noqa: E402
from bhshadow import synth  # noqa: E402
from bhshadow.fitting import Problem, run_mcmc  # noqa: E402
from bhshadow.uvdata import read_uvfits  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "stage3_injection.csv"
COVER = {"2017-04-06 hi": ("096", "hi"), "2017-04-10 hi": ("100", "hi")}
NREP = 10
FIELDS = ["truth", "kind", "coverage", "rep", "d_true", "d_M3", "d_M3_minus", "d_M3_plus", "w_M3", "chi2nu_M3",
          "bic_M3", "D_D", "chi2nu_D", "bic_D", "delta_bic", "class", "n_data", "converged_M3", "converged_D"]


def classify(dbic):
    return "ring" if dbic > 10 else ("disk" if dbic < -10 else "undetermined")


def run():
    done = set()
    if OUT.exists():
        with OUT.open() as f:
            done = {(r["truth"], r["coverage"], int(r["rep"])) for r in csv.DictReader(f)}
    new = not OUT.exists()
    f = OUT.open("a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    raws = {c: read_uvfits(DS.uvfits_path(*v)) for c, v in COVER.items()}
    t0 = time.time()
    n = 0
    for cov, raw in raws.items():
        for truth, (_, dref, kind) in synth.TRUTHS.items():
            for rep in range(NREP):
                if (truth, cov, rep) in done:
                    continue
                syn = synth.make(raw, truth, rep, cov)
                ds = DS.build(syn, f"{truth} {cov} {rep}")
                seed = synth.seed_for("fit", truth, cov, rep)
                r3 = run_mcmc(Problem("M3", [ds]), seed)
                rd = run_mcmc(Problem("D", [ds]), seed + 1)
                s = r3["summary"]
                dbic = rd["bic"] - r3["bic"]
                w.writerow({
                    "truth": truth, "kind": kind, "coverage": cov, "rep": rep, "d_true": dref if dref else "",
                    "d_M3": s["d"]["median"], "d_M3_minus": s["d"]["minus"], "d_M3_plus": s["d"]["plus"],
                    "w_M3": s["w"]["median"], "chi2nu_M3": r3["chi2_nu"], "bic_M3": r3["bic"],
                    "D_D": rd["summary"]["D"]["median"], "chi2nu_D": rd["chi2_nu"], "bic_D": rd["bic"],
                    "delta_bic": dbic, "class": classify(dbic), "n_data": r3["n_data"],
                    "converged_M3": r3["converged"], "converged_D": rd["converged"]})
                f.flush()
                n += 1
                print(f"{truth} {cov} {rep}: d={s['d']['median']:.2f} dBIC={dbic:.0f} "
                      f"chi2nu={r3['chi2_nu']:.2f} [{time.time() - t0:.0f}s, {n} new]", flush=True)
    f.close()


def summarise():
    with OUT.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("d_M3", "d_M3_minus", "d_M3_plus", "chi2nu_M3", "delta_bic", "w_M3", "D_D"):
            r[k] = float(r[k])
    out = {"per_truth": {}, "criteria": {
        "recovery_good": "|bias| <= 1 muas and scatter <= 2 muas (ring and crescent truths)",
        "classification": "dBIC = BIC_D - BIC_M3 > 10 ring, < -10 disk; distinguishable if >= 90% correct for both"}}
    for cov in COVER:
        for truth, (_, dref, kind) in synth.TRUTHS.items():
            rr = [r for r in rows if r["truth"] == truth and r["coverage"] == cov]
            if not rr:
                continue
            d = np.array([r["d_M3"] for r in rr])
            sig = np.array([0.5 * (r["d_M3_minus"] + r["d_M3_plus"]) for r in rr])
            sc = np.array([max(1.0, np.sqrt(r["chi2nu_M3"])) for r in rr])
            cls = [r["class"] for r in rr]
            e = {"n": len(rr), "d_mean": float(d.mean()), "scatter": float(d.std(ddof=1)) if len(d) > 1 else None,
                 "median_chi2nu_M3": float(np.median([r["chi2nu_M3"] for r in rr])),
                 "class_counts": {c: cls.count(c) for c in ("ring", "disk", "undetermined")},
                 "median_delta_bic": float(np.median([r["delta_bic"] for r in rr]))}
            if dref is not None:
                e["d_true"] = dref
                e["bias"] = float(d.mean() - dref)
                e["coverage_raw"] = float(np.mean(np.abs(d - dref) <= sig))
                e["coverage_scaled"] = float(np.mean(np.abs(d - dref) <= sig * sc))
                if truth != "xscres43":
                    e["recovery_good"] = bool(abs(e["bias"]) <= 1.0 and e["scatter"] <= 2.0)
            out["per_truth"][f"{truth} | {cov}"] = e
    for cov in COVER:
        ringlike = [r for r in rows if r["coverage"] == cov and r["kind"] == "ring"]
        disks = [r for r in rows if r["coverage"] == cov and r["kind"] == "disk"]
        gauss = [r for r in rows if r["coverage"] == cov and r["kind"] == "gaussian"]
        fr = np.mean([r["class"] == "ring" for r in ringlike]) if ringlike else np.nan
        fd = np.mean([r["class"] == "disk" for r in disks]) if disks else np.nan
        out[f"classification | {cov}"] = {
            "ring_truths_classified_ring": float(fr), "disk_truths_classified_disk": float(fd),
            "gaussian_classified_ring": float(np.mean([r["class"] == "ring" for r in gauss])) if gauss else None,
            "distinguishable": bool(fr >= 0.9 and fd >= 0.9)}
        good = [v["recovery_good"] for k, v in out["per_truth"].items() if k.endswith(cov) and "recovery_good" in v]
        out[f"recovery_all_good | {cov}"] = bool(all(good)) if good else None
    (RES / "stage3_summary.json").write_text(json.dumps(out, indent=1))

    plotstyle.apply()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cov in zip(axs, COVER):
        for k, (truth, (_, dref, kind)) in enumerate(t for t in synth.TRUTHS.items() if t[1][1] is not None):
            rr = [r for r in rows if r["truth"] == truth and r["coverage"] == cov]
            fam = 0 if truth.startswith("ring") else (1 if truth.startswith("cres") else 2)
            x = dref + (fam - 1) * 0.5 + np.linspace(-0.15, 0.15, len(rr))
            ax.errorbar(x, [r["d_M3"] - dref for r in rr], [0.5 * (r["d_M3_minus"] + r["d_M3_plus"]) for r in rr],
                        fmt=plotstyle.MARKERS[fam], ms=4, lw=0.7, color=plotstyle.SERIES[fam],
                        label=["symmetric ring", "m=1 crescent", "offset-disk crescent (d̂)"][fam]
                        if (truth in ("ring38", "cres38", "xscres43")) else None)
        ax.axhline(0, color=plotstyle.TEXT2, lw=1)
        ax.set_title(f"coverage: {cov}")
        ax.set_xlabel("true diameter (μas)")
    axs[0].set_ylabel("recovered − true d (μas), M3 fit")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "fig_stage3_recovery.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    names = list(synth.TRUTHS)
    for ax, cov in zip(axs, COVER):
        for i, truth in enumerate(names):
            rr = [r for r in rows if r["truth"] == truth and r["coverage"] == cov]
            v = np.array([r["delta_bic"] for r in rr])
            ax.plot(np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), np.sign(v) * np.log10(1 + np.abs(v)), "o",
                    ms=4, color=plotstyle.SERIES[0 if synth.TRUTHS[truth][2] == "ring" else
                                                 (1 if synth.TRUTHS[truth][2] == "disk" else 2)])
        for yv in (1, -1):
            ax.axhline(yv * np.log10(11), color=plotstyle.TEXT2, lw=0.8, ls="--")
        ax.set_xticks(range(len(names)), names, rotation=45)
        ax.set_title(f"coverage: {cov}")
    axs[0].set_ylabel("sign(ΔBIC)·log10(1+|ΔBIC|),  ΔBIC = BIC_disk − BIC_M3")
    fig.suptitle("Ring vs disk: dashed lines at ΔBIC = ±10", color=plotstyle.TEXT)
    fig.tight_layout(); fig.savefig(RES / "fig_stage3_dbic.png"); plt.close(fig)
    return out


if __name__ == "__main__":
    if "--summary-only" not in sys.argv:
        run()
    print(json.dumps(summarise(), indent=1)[:3000])
