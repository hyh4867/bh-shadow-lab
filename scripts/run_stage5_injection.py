"""Stage 5 (additional analysis): more realistic injection-recovery. PREREGISTRATION 10.7.

Truths (ring component 0.6 Jy) on the real 2017-04-06 hi and 2017-04-10 hi coverage:
  RI1  M3 ring d = 42, w = 12, |b1| = 0.25, eta = 200
  RI2  RI1 + m=2 (|b2| = 0.10, arg 60) + jet-like Gaussian (15 %, FWHM 40, 35 muas toward PA 290)
  RI3  two-disk crescent (d_hat = 43) + the same Gaussian
All truths also get: broad Gaussian (20 %, FWHM 150, 60 muas toward PA 290), a 0.6 Jy 1-mas
Gaussian, station gains as stage 3, phase random walk N(0, 0.1 rad) per 10 s per station,
baseline-based non-closing factors per scan (amp lognormal 3 %, phase N(0, 3 deg)), thermal noise.
Outputs: results/stage5_injection.csv (resumable), results/stage5_injection_summary.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow import models as M  # noqa: E402
from bhshadow import synth  # noqa: E402
from bhshadow.fitting import Problem, run_mcmc  # noqa: E402
from bhshadow.s5 import RES, corrupt_ext  # noqa: E402
from bhshadow.uvdata import read_uvfits  # noqa: E402

OUT = RES / "stage5_injection.csv"
COVER = {"2017-04-06 hi": ("096", "hi"), "2017-04-10 hi": ("100", "hi")}
NREP = 20
PA_JET = np.deg2rad(290.0)


def _off(r):
    return r * np.sin(PA_JET), r * np.cos(PA_JET)


def jet(u, v):
    return 0.15 * M.gauss_offset(u, v, 40.0, *_off(35.0))


def extras(u, v):
    return 0.20 * M.gauss_offset(u, v, 150.0, *_off(60.0)) + 1.0 * M.gaussian(u, v, 1000.0)


TRUTHS = {
    "RI1": (lambda u, v: M.mring1(u, v, 42.0, 12.0, 0.25, 200.0), 42.0),
    "RI2": (lambda u, v: M.mring2(u, v, 42.0, 12.0, 0.25, 200.0, 0.10, 60.0) + jet(u, v), 42.0),
    "RI3": (lambda u, v: M.offset_crescent(u, v, 23.5, 19.5, 2.5, 200.0, 10.0) + jet(u, v), 43.0),
}
KW = dict(rw_step=0.1, nonclosing_amp=0.03, nonclosing_phase_deg=3.0)
FIELDS = ["truth", "coverage", "rep", "d_ref", "d", "d_minus", "d_plus", "w", "chi2_nu", "sigma_scaled",
          "n_data", "converged", "retried"]


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
    t0 = time.time()
    for cov, (doy, band) in COVER.items():
        raw = read_uvfits(DS.uvfits_path(doy, band))
        for truth, (fn, dref) in TRUTHS.items():
            vis = synth.FLUX * (fn(raw.u, raw.v) + extras(raw.u, raw.v))
            for rep in range(NREP):
                if (truth, cov, rep) in done:
                    continue
                syn = corrupt_ext(raw, vis, synth.seed_for("real", truth, cov, rep), **KW)
                ds = DS.build(syn, f"{truth} {cov} {rep}")
                r = run_mcmc(Problem("M3", [ds]), synth.seed_for("realfit", truth, cov, rep))
                s = r["summary"]["d"]
                sc = 0.5 * (s["minus"] + s["plus"]) * max(1.0, np.sqrt(r["chi2_nu"]))
                w.writerow({"truth": truth, "coverage": cov, "rep": rep, "d_ref": dref, "d": s["median"],
                            "d_minus": s["minus"], "d_plus": s["plus"], "w": r["summary"]["w"]["median"],
                            "chi2_nu": r["chi2_nu"], "sigma_scaled": sc, "n_data": r["n_data"],
                            "converged": r["converged"], "retried": r["retried"]})
                f.flush()
                print(f"{truth} {cov} {rep}: d={s['median']:.2f} chi2nu={r['chi2_nu']:.2f} [{time.time() - t0:.0f}s]",
                      flush=True)
    f.close()


def summarise():
    rows = list(csv.DictReader(OUT.open()))
    fits2 = json.loads((RES / "stage2_fits.json").read_text())
    out = {"note": "추가 분석 (PREREGISTRATION 10.7)", "corruption": KW, "per": {}}
    for cov in COVER:
        for truth in TRUTHS:
            rr = [r for r in rows if r["truth"] == truth and r["coverage"] == cov]
            if not rr:
                continue
            d = np.array([float(r["d"]) for r in rr]); ref = float(rr[0]["d_ref"])
            sc = np.array([float(r["sigma_scaled"]) for r in rr])
            chi = np.array([float(r["chi2_nu"]) for r in rr])
            bias, scat = float(d.mean() - ref), float(d.std(ddof=1))
            cover = float(np.mean(np.abs(d - ref) <= sc))
            e = {"n": len(rr), "d_ref": ref, "bias": bias, "scatter": scat, "coverage_sigma_scaled": cover,
                 "sigma_scaled_median": float(np.median(sc)), "chi2_nu_median": float(np.median(chi)),
                 "chi2_nu_range": [float(chi.min()), float(chi.max())],
                 "n_converged": int(sum(r["converged"] == "True" for r in rr))}
            if truth in ("RI1", "RI2"):
                e["recovery"] = "양호" if abs(bias) <= 1 and scat <= 2 else "불량"
            e["coverage_verdict"] = "적절" if 0.5 <= cover <= 0.85 else ("과소평가" if cover < 0.5 else "과대평가")
            out["per"][f"{truth} {cov}"] = e
        out[f"real_chi2_nu {cov}"] = fits2[cov]["M3"]["chi2_nu"]
    (RES / "stage5_injection_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    run()
    summarise()
