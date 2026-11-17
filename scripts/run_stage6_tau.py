"""Stage 6, change 11.10: C3' (emcee 32000 steps from the nested posterior) for every main model,
then the central-model selection under the pre-registered rule (C3) and the changed rule (C3').

Run after scripts/run_stage6_models.py. Updates results/stage6_models.json in place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.nested import load_samples  # noqa: E402
from bhshadow.s6 import NRUNS, SEED6, c3_long, joint6, sym  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage6_models import ORDER, select  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    path = RES / "stage6_models.json"
    M6 = json.loads(path.read_text())
    if "selection_preregistered" in M6:
        print("already updated"); return
    ds = DS.load_all()
    multi = {}
    for t in M6["table"]:
        m = t["model"]
        i = ORDER.index(m) if m in ORDER else 6
        pooled = np.vstack([load_samples(f"m6_{m}_r{r}") for r in range(NRUNS)])
        c = c3_long(f"m6_{m}", joint6(m, ds), pooled, SEED6 + 100 * i)
        t["C3p"] = c["C3p_tau"]; t["tau_max_32k"] = c["emcee32k"]["tau_max"]
        t["emcee32k_d"] = c["emcee32k"]["d"]["median"]
        t["converged_p"] = bool(t["C1"] and t["C2"] and t["C3p"])
        mj = json.loads((RES / "stage6_fits" / f"m6_{m}_multi.json").read_text())
        mj.update(c); mj["converged_p"] = t["converged_p"]
        (RES / "stage6_fits" / f"m6_{m}_multi.json").write_text(json.dumps(mj, indent=1))
        multi[m] = {"converged": t["converged_p"], "logz_mean": t["lnZ"], "k": t["k"], "bic": t["bic"]}
        print(f"{m}: tau8k={t['tau_max']:.0f} tau32k={t['tau_max_32k']:.0f} conv={t['converged']} conv_p={t['converged_p']}", flush=True)
    M6["selection_preregistered"] = M6.pop("selection")
    M6["central_preregistered"] = M6.pop("central")
    sel = select(multi)
    M6["selection_changed"] = sel
    C = sel["central"]
    mj = json.loads((RES / "stage6_fits" / f"m6_{C}_multi.json").read_text())
    d_c, s_raw = mj["summary"]["d"]["median"], sym(mj["summary"]["d"])
    M6["central"] = {"model": C, "d": d_c, "sigma_raw": s_raw, "chi2_nu": mj["chi2_nu"],
                     "stat": s_raw * max(1.0, np.sqrt(mj["chi2_nu"])), "summary": mj["summary"],
                     "rule": "changed rule C3' (PREREGISTRATION 11.10)"}
    for t in M6["table"]:
        t["dlnZ_vs_C"] = t["lnZ"] - mj["logz_mean"]
        t["dlnZ_over_chi2nuC"] = t["dlnZ_vs_C"] / mj["chi2_nu"]
    path.write_text(json.dumps(M6, indent=1))
    print(json.dumps({"preregistered": M6["selection_preregistered"], "changed": sel,
                      "central": {k: M6["central"][k] for k in ("model", "d", "stat", "chi2_nu")}}, indent=1))


if __name__ == "__main__":
    main()
