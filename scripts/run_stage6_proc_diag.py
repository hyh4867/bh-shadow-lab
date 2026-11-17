"""Stage 6 diagnostic (not pre-registered): why sigma_proc is large for E1 (1.95 muas) while it was
0.006 for M3 in stage 5. Fits M3 to the 10-s closure variant with the stage 6 sampler (one nested run)
and tabulates d and w for M3 and E1, full data vs 10-s variant. Output: results/stage6_proc_diag.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.nested import FITS6  # noqa: E402
from bhshadow.robust import incoherent_dataset  # noqa: E402
from bhshadow.s6 import SEED6, joint6, single  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    inc = [incoherent_dataset(d) for d in DS.load_all()]
    fM = single("p6_M3_tenseq", joint6("M3", inc), SEED6 + 1041)
    rows = {}
    for m, full_tag, inc_fit in (("M3", "m6_M3_multi", fM),
                                 ("E1", "m6_E1_multi", json.loads((FITS6 / "p6_E1_tenseq.json").read_text()))):
        full = json.loads((FITS6 / f"{full_tag}.json").read_text())
        rows[m] = {k: {"full": full["summary"][k]["median"], "tenseq": inc_fit["summary"][k]["median"]} for k in ("d", "w")}
        rows[m]["chi2_nu"] = {"full": full["chi2_nu"], "tenseq": inc_fit["chi2_nu"]}
        if m == "E1":
            for k in ("fg", "xg", "yg", "fwhmg"):
                rows[m][k] = {"full": full["summary"][k]["median"], "tenseq": inc_fit["summary"][k]["median"]}
    out = {"note": "diagnostic, not pre-registered", "models": rows,
           "shift_d": {m: r["d"]["tenseq"] - r["d"]["full"] for m, r in rows.items()},
           "shift_w": {m: r["w"]["tenseq"] - r["w"]["full"] for m, r in rows.items()}}
    (RES / "stage6_proc_diag.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
