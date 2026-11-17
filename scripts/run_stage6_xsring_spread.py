"""Stage 6 diagnostic (not pre-registered): run-to-run spread of the xs-ring d_hat in step S1
(three independent nested runs of X1 per data set), to judge whether the S2-S4 step contributions
(single runs) exceed the sampler noise. Reads saved samples only. Output: results/stage6_xsring_spread.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.nested import FITS6, load_samples  # noqa: E402
from bhshadow.s5 import xs_dhat  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    rows = []
    for doy in DS.DAYS:
        for b in ("lo", "hi"):
            tag = f"x6_S1_X1_{doy}_{b}"
            names = json.loads((FITS6 / f"{tag}_multi.json").read_text())["names"]
            v = [float(np.median(xs_dhat(load_samples(f"{tag}_r{r}"), names))) for r in range(3)]
            rows.append({"dataset": f"{doy} {b}", "d_hat_runs": v, "range": float(np.ptp(v)),
                         "sd": float(np.std(v, ddof=1))})
    sd = np.array([r["sd"] for r in rows])
    out = {"note": "diagnostic, not pre-registered", "per_dataset": rows, "mean_range": float(np.mean([r["range"] for r in rows])),
           "rms_sd_single_run": float(np.sqrt(np.mean(sd**2))),
           "noise_on_mean_of_8_single_runs": float(np.sqrt(np.mean(sd**2)) / np.sqrt(len(rows))),
           "noise_on_step_contribution": float(np.sqrt(2) * np.sqrt(np.mean(sd**2)) / np.sqrt(len(rows)))}
    (RES / "stage6_xsring_spread.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_dataset"}, indent=1))


if __name__ == "__main__":
    main()
