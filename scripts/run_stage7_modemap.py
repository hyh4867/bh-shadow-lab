"""Stage 7, 12.1: mode map by differential evolution (12 seeds per model, 4 processes).

Each optimisation is saved to results/stage7_fits/de_<model>_s<i>.json (resumable).
Output: results/stage7_modemap.json (per model: all optima, distinct modes, best chi2, stage 6 nested
run chi2 for comparison).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.s6 import joint6  # noqa: E402
from bhshadow.s7 import FITS7, SEED7, de_fit, distinct_modes  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
MODELS = ["M3", "E1", "E2", "G1", "G2", "G3", "R2G3"]
NDE = 12


def task(args):
    m, i = args
    path = FITS7 / f"de_{m}_s{i}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = de_fit(joint6(m, DS.load_all()), SEED7 + 1000 + 100 * MODELS.index(m) + i)
    r["model"] = m
    path.write_text(json.dumps(r, indent=1))
    print(f"[de] {m} s{i}: chi2={r['chi2']:.2f} d={r['d']:.2f} nit={r['nit']} {r['runtime_s']:.0f}s", flush=True)
    return r


def main():
    FITS7.mkdir(parents=True, exist_ok=True)
    jobs = [(m, i) for m in MODELS for i in range(NDE)]
    with mp.get_context("spawn").Pool(4) as pool:
        res = pool.map(task, jobs, chunksize=1)
    out = {"note": "stage 7 additional analysis (PREREGISTRATION 12.1)", "n_seeds": NDE, "models": {}}
    for m in MODELS:
        rs = [r for r in res if r["model"] == m]
        s6 = json.loads((RES / "stage6_fits" / f"m6_{m}_multi.json").read_text())
        out["models"][m] = {"de": [{k: r[k] for k in ("seed", "chi2", "d", "nit", "nfev", "success", "runtime_s")}
                                   for r in rs],
                            "de_best_chi2": min(r["chi2"] for r in rs),
                            "de_best_x": min(rs, key=lambda r: r["chi2"])["x"],
                            "modes": distinct_modes(rs),
                            "stage6_run_chi2": [r["chi2_min"] for r in s6["runs"]],
                            "stage6_run_d": [r["d"]["median"] for r in s6["runs"]]}
        mm = out["models"][m]
        print(f"{m:5s} DE best {mm['de_best_chi2']:.1f}  modes {[(round(x['chi2'], 1), round(x['d'], 2), x['count']) for x in mm['modes']]}"
              f"  stage6 runs {[round(c, 1) for c in mm['stage6_run_chi2']]}", flush=True)
    (RES / "stage7_modemap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
