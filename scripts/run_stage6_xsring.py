"""Stage 6, 11.5: split the xs-ring implementation difference (Delta_impl = +2.68 muas in stage 5).

S0 stage 5 X1 (emcee) -> S1 X1 nested (3 runs, convergence) -> S2 XE3 (3 elliptical Gaussians)
-> S3 XE3 with intra-site baselines -> S4 XE3L (+ large centred Gaussian), cumulative.
Output: results/stage6_xsring.json. Resumable (results/stage6_fits/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow import literature as L  # noqa: E402
from bhshadow.fitting import Problem  # noqa: E402
from bhshadow.s5 import xs_dhat  # noqa: E402
from bhshadow.s6 import SEED6, multi_run, single, sym  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    s5 = {r["dataset"]: r for r in json.loads((RES / "stage5_systematics.json").read_text())["xsring"]["per_dataset"]}
    keys = [(doy, b) for doy in DS.DAYS for b in ("lo", "hi")]
    der = {"d_hat": xs_dhat}
    S = SEED6 + 2000
    rows = []
    for i, (doy, b) in enumerate(keys):
        ds = DS.load(doy, b)
        ds_in = DS.load(doy, b, drop_intrasite=False)
        name = ds.name
        eht = L.TABLE3_XSRING[name][0]
        r = {"dataset": name, "EHT": eht, "n_data": ds.n_data, "n_data_intrasite": ds_in.n_data}
        r["S0"] = {"d_hat": s5[name]["X1"]["d_hat"]["median"], "chi2_nu": s5[name]["X1"]["chi2_nu"],
                   "converged_tau": s5[name]["X1"]["converged"]}
        m = multi_run(f"x6_S1_X1_{doy}_{b}", Problem("X1", [ds]), S + 10 * i, derived=der)
        r["S1"] = {"d_hat": m["derived"]["d_hat"]["median"], "d_hat_sigma": sym(m["derived"]["d_hat"]),
                   "chi2_nu": m["chi2_nu"], "lnZ": m["logz_mean"], "lnZ_range": m["logz_range"],
                   "rhat_max": max(m["rhat"].values()), "tau_max": m["emcee"]["tau_max"], "converged": m["converged"],
                   "tau_max_32k": m["emcee32k"]["tau_max"], "C3p": m["C3p_tau"], "converged_p": m["converged_p"],
                   "C1": m["C1_rhat"], "C2": m["C2_logz"], "C3": m["C3_tau"]}
        steps = [("S2", "XE3", ds), ("S3", "XE3", ds_in), ("S4", "XE3L", ds_in)]
        for j, (s, model, dd) in enumerate(steps):
            f = single(f"x6_{s}_{model}_{doy}_{b}", Problem(model, [dd]), S + 10 * i + 5 + j, derived=der)
            r[s] = {"d_hat": f["derived"]["d_hat"]["median"], "d_hat_sigma": sym(f["derived"]["d_hat"]),
                    "chi2_nu": f["chi2_nu"], "lnZ": f["logz"], "pa": f["summary"]["pa"]["median"]}
            if model == "XE3L":
                r[s]["fL"] = f["summary"]["fL"]["median"]; r[s]["aL"] = f["summary"]["aL"]["median"]
        rows.append(r)
        print(name, {s: round(r[s]["d_hat"], 2) for s in ("S0", "S1", "S2", "S3", "S4")}, flush=True)
    steps = ["S0", "S1", "S2", "S3", "S4"]
    dimpl = {s: float(np.mean([r[s]["d_hat"] - r["EHT"] for r in rows])) for s in steps}
    contrib = {s: dimpl[s] - dimpl[p] for p, s in zip(steps[:-1], steps[1:])}
    thr = abs(dimpl["S0"]) / 3
    out = {"note": "stage 6 additional analysis (PREREGISTRATION 11.5)", "per_dataset": rows,
           "mean_delta_impl": dimpl, "contribution": contrib, "threshold": thr,
           "major": {s: abs(c) >= thr for s, c in contrib.items()},
           "verdict": "구현 차이 설명됨" if abs(dimpl["S4"]) <= thr else "남은 차이 있음",
           "X1_converged_count": int(sum(r["S1"]["converged"] for r in rows)),
           "X1_converged_count_changed_rule": int(sum(r["S1"]["converged_p"] for r in rows))}
    (RES / "stage6_xsring.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("mean_delta_impl", "contribution", "major", "verdict",
                                          "X1_converged_count")}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
