"""Stage 5 (additional analysis): what causes the LMT shift? PREREGISTRATION 10.3.

Synthetic data on the eight real coverages, five realisations per test, joint M3 fits
with and without LMT. Shift = d(no LMT) - d(all).
  T_B1: truth = stage 2 joint best fit (all stations), standard stage 3 corruption
  T_B2: truth = joint best fit without LMT (d ~ 45.6), standard corruption
  T_G : truth as T_B1, plus LMT-only per-10-s amplitude gain (lognormal 50 %) and
        phase random walk (N(0, 0.3 rad) per 10 s, from 0 at each scan start)
Model-inadequacy test uses the E1/E2 fits from run_stage5_systematics.py.
Output: results/stage5_lmt.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow import synth  # noqa: E402
from bhshadow.models import mring1  # noqa: E402
from bhshadow.s5 import FITS, RES, SEED, cached_fit, corrupt_ext, joint_problem  # noqa: E402

NREP = 5
HALF = 1.95  # half of the observed shift, PREREGISTRATION 10.3


def truth_vis(dsets, best, names):
    groups = DS_groups(dsets)
    out = []
    for d, g in zip(dsets, groups):
        p = dict(zip(names, best))
        out.append(mring1(d.raw.u, d.raw.v, p["d"], p["w"], p[f"beta_{g}"], p[f"beta_pa_{g}"]))
    return out


def DS_groups(dsets):
    from bhshadow.s5 import day_groups
    return day_groups(dsets)


def main():
    st2 = json.loads((RES / "stage2_joint.json").read_text())
    rob2 = json.loads((RES / "stage2_robustness.json").read_text())
    delta_obs = rob2["no_LMT"]["d"]["median"] - rob2["primary"]["d"]["median"]
    dsets = DS.load_all()
    nolmt = DS.load_all(drop_stations=("LM",))
    ref_all = (st2["fit"]["best"], st2["fit"]["names"])
    f_nolmt = cached_fit("lmt_ref_noLMT", joint_problem("M3", nolmt), SEED + 9)  # same seed as stage 2
    ref_nolmt = (f_nolmt["best"], f_nolmt["names"])
    tests = {
        "T_B1": dict(truth=ref_all, kw={}),
        "T_B2": dict(truth=ref_nolmt, kw={}),
        "T_G": dict(truth=ref_all, kw=dict(amp10_lmt=0.5, rw_step=0.3, rw_lmt_only=True)),
    }
    res = {"delta_obs": delta_obs, "ref_noLMT_refit_d": f_nolmt["summary"]["d"], "tests": {}}
    for ti, (tname, spec) in enumerate(tests.items()):
        vis = truth_vis(dsets, *spec["truth"])
        rows = []
        for rep in range(NREP):
            syn_all, syn_nol = [], []
            for d, v in zip(dsets, vis):
                seed = synth.seed_for("lmt", tname, rep, d.doy, d.band)
                raw = corrupt_ext(d.raw, synth.FLUX * v, seed, **spec["kw"])
                syn_all.append(DS.build(raw, d.name, d.doy, d.band))
                syn_nol.append(DS.build(raw, d.name, d.doy, d.band, drop_stations=("LM",)))
            fa = cached_fit(f"lmt_{tname}_{rep}_all", joint_problem("M3", syn_all), SEED + 2000 + 100 * ti + 2 * rep)
            fn = cached_fit(f"lmt_{tname}_{rep}_noLMT", joint_problem("M3", syn_nol), SEED + 2001 + 100 * ti + 2 * rep)
            da, dn = fa["summary"]["d"]["median"], fn["summary"]["d"]["median"]
            rows.append({"rep": rep, "d_all": da, "d_noLMT": dn, "shift": dn - da,
                         "chi2nu_all": fa["chi2_nu"], "chi2nu_noLMT": fn["chi2_nu"],
                         "converged": [fa["converged"], fn["converged"]]})
            print(tname, rep, round(da, 3), round(dn, 3), flush=True)
        sh = np.array([r["shift"] for r in rows])
        res["tests"][tname] = {"rows": rows, "mean_shift": float(sh.mean()), "std_shift": float(sh.std(ddof=1)),
                               "truth_d": float(dict(zip(spec["truth"][1], spec["truth"][0]))["d"])}
    t = res["tests"]
    gain = "가능" if t["T_G"]["mean_shift"] >= HALF else "기각"
    base = "가능" if max(t["T_B1"]["mean_shift"], t["T_B2"]["mean_shift"]) >= HALF else "기각"
    # model inadequacy: lowest-BIC extension, all vs no-LMT
    ext = {m: (json.loads((FITS / f"ext_{m}_all.json").read_text()), json.loads((FITS / f"ext_{m}_noLMT.json").read_text()))
           for m in ("E1", "E2")}
    best = min(ext, key=lambda m: ext[m][0]["bic"])
    ext_shift = ext[best][1]["summary"]["d"]["median"] - ext[best][0]["summary"]["d"]["median"]
    model = "지지" if abs(ext_shift) <= HALF else "확장 모형으로도 설명 안 됨"
    res["verdicts"] = {
        "gain": {"result": gain, "criterion": "mean shift in T_G >= +1.95"},
        "baseline_range": {"result": base, "criterion": "mean shift in T_B1 or T_B2 >= +1.95"},
        "model_inadequacy": {"result": model, "model": best, "shift": ext_shift,
                             "shift_other": {m: ext[m][1]["summary"]["d"]["median"] - ext[m][0]["summary"]["d"]["median"]
                                             for m in ext},
                             "criterion": "|shift| <= 1.95 for the lowest-BIC extension"},
    }
    (RES / "stage5_lmt.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["verdicts"], ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
