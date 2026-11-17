"""Stage 5, extra diagnostic (not pre-registered, no verdict): compare the log closure
amplitudes and their sigmas that eht-imaging uses with ours, 2017-04-11 hi, after the
same preprocessing as scripts/run_stage5_rml_diag.py. Output: results/stage5_rml_lca_check.json
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402

import ehtim as eh  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    d = DS.load("101", "hi")
    with contextlib.redirect_stdout(io.StringIO()):
        obs = eh.obsdata.load_uvfits(str(DS.uvfits_path("101", "hi")))
        obs.add_scans()
        obs = obs.avg_coherent(0.0, scan_avg=True)
        obs = obs.flag_uvdist(uv_min=0.1e9)
        obs = obs.add_fractional_noise(0.01)
        lca = obs.c_amplitudes(vtype="vis", mode="all", count="min", ctype="logcamp", debias=True)
        cp = obs.c_phases(vtype="vis", mode="all", count="min", ang_unit="deg")
    e_sig, e_val = np.asarray(lca["sigmaca"]), np.asarray(lca["camp"])
    o_sig, o_val = d.lca.sigma, d.lca.obs
    out = {
        "note": "extra diagnostic, not pre-registered",
        "ehtim_logcamp": {"n": int(len(e_val)), "median_sigma": float(np.median(e_sig)),
                          "median_abs_value": float(np.median(np.abs(e_val)))},
        "ours_logcamp": {"n": int(len(o_val)), "median_sigma": float(np.median(o_sig)),
                         "median_abs_value": float(np.median(np.abs(o_val)))},
        "ehtim_cphase": {"n": int(len(cp)), "median_sigma_deg": float(np.median(cp["sigmacp"]))},
        "ours_cphase": {"n": int(len(d.cp)), "median_sigma_deg": float(np.rad2deg(np.median(d.cp.sigma)))},
        "ehtim_snr_of_scan_vis_min": float(np.min(np.abs(obs.data["vis"]) / obs.data["sigma"])),
        "ehtim_n_scan_vis": int(len(obs.data)), "ours_n_scan_vis_after_cuts": int(len(d.tab)),
    }
    # quadrangles that contain both co-located ALMA and APEX (or JCMT and SMA): their log closure
    # amplitude is ~0 whatever the source structure, so they carry almost no size information
    def colocated(sts):
        return {"AA", "AP"} <= sts or {"JC", "SM"} <= sts
    e_co = np.array([colocated({r["t1"], r["t2"], r["t3"], r["t4"]}) for r in lca])
    o_co = np.array([colocated(set(lab.replace("/", "*").replace("*", "-").split("-"))) for lab in d.lca.label])
    out["colocated_quadrangles"] = {
        "ehtim_min_set": {"n": int(e_co.sum()), "of": int(len(e_co)),
                          "median_abs_value_colocated": float(np.median(np.abs(e_val[e_co]))),
                          "median_abs_value_other": float(np.median(np.abs(e_val[~e_co])))},
        "ours": {"n": int(o_co.sum()), "of": int(len(o_co)),
                 "median_abs_value_colocated": float(np.median(np.abs(o_val[o_co]))),
                 "median_abs_value_other": float(np.median(np.abs(o_val[~o_co])))}}
    (RES / "stage5_rml_lca_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
