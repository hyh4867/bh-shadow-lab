"""Stage 4 (optional): small MLP closure quantities -> diameter, compared with MCMC.

Pre-registration 7.5. For each coverage (2017-04-06 hi, 2017-04-10 hi):
- fixed closure definitions from the real scan-averaged table (no SNR cut, intra-site removed),
- training set: random M3 parameters, same synthetic corruption as stage 3,
- test set: the stage 3 ring and crescent synthetic data (regenerated from the same seeds),
- compare RMSE of ML and of the stage 3 MCMC medians on the same test set,
- predict the real data.
Outputs: results/stage4_ml.json, results/fig_stage4_ml.png
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import plotstyle, synth  # noqa: E402
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.closure import closure_phases, log_closure_amplitudes  # noqa: E402
from bhshadow.models import mring1  # noqa: E402
from bhshadow.uvdata import assign_scans, prepare, read_uvfits  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
COVER = {"2017-04-06 hi": ("096", "hi"), "2017-04-10 hi": ("100", "hi")}
N_TRAIN = 20000
TEST_TRUTHS = ["ring38", "ring42", "ring46", "cres38", "cres42", "cres46"]
EXTRA_TRUTHS = ["xscres43"]


class Featuriser:
    """Fast scan averaging with a fixed (scan, baseline) grouping, then fixed closure definitions."""

    def __init__(self, raw):
        self.raw = raw
        scan = assign_scans(raw.time)
        key = np.array([f"{s}|{a}|{b}" for s, a, b in zip(scan, raw.t1, raw.t2)])
        self.uk, self.inv = np.unique(key, return_inverse=True)
        self.w = 1.0 / raw.sigma**2
        self.sw = np.bincount(self.inv, self.w)
        self.sig_avg = 1.0 / np.sqrt(self.sw)
        ref = prepare(raw, snr_min=0.0)  # rows: non-intra-site scan averages, same keys
        keys_ref = np.array([f"{s}|{a}|{b}" for s, a, b in zip(ref.scan, ref.t1, ref.t2)])
        pos = {k: i for i, k in enumerate(self.uk)}
        self.rows = np.array([pos[k] for k in keys_ref])
        self.cp = closure_phases(ref)
        self.lca = log_closure_amplitudes(ref)

    def __call__(self, vis10):
        re = np.bincount(self.inv, self.w * vis10.real) / self.sw
        im = np.bincount(self.inv, self.w * vis10.imag) / self.sw
        V = (re + 1j * im)[self.rows]
        sig = np.sqrt(self.sig_avg[self.rows] ** 2 + (0.01 * np.abs(V)) ** 2)
        amp = np.sqrt(np.clip(np.abs(V) ** 2 - sig**2, 0, None))
        amp = np.maximum(amp, 0.1 * sig)  # keep logs finite
        ph = np.angle(V)
        cp = (ph[self.cp.idx] * self.cp.sgn).sum(1)
        lca = np.clip((np.log(amp)[self.lca.idx] * self.lca.sgn).sum(1), -10, 10)
        return np.concatenate([np.sin(cp), np.cos(cp), lca])


def main():
    with (RES / "stage3_injection.csv").open() as f:
        s3 = list(csv.DictReader(f))
    out = {"pre_registration": "7.5", "n_train": N_TRAIN, "per_coverage": {}}
    plotstyle.apply()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (cov, (doy, band)) in zip(axs, COVER.items()):
        t0 = time.time()
        raw = read_uvfits(DS.uvfits_path(doy, band))
        feat = Featuriser(raw)
        rng = np.random.default_rng(synth.seed_for("ml-train", cov))
        P = np.column_stack([rng.uniform(30, 55, N_TRAIN), rng.uniform(5, 25, N_TRAIN),
                             rng.uniform(0, 0.5, N_TRAIN), rng.uniform(0, 360, N_TRAIN)])
        X = np.empty((N_TRAIN, len(feat.cp) * 2 + len(feat.lca)))
        for i, p in enumerate(P):
            vm = synth.FLUX * mring1(raw.u, raw.v, *p)
            X[i] = feat(synth.corrupt(raw, vm, synth.seed_for("ml-train", cov, i)).vis)
        sc = StandardScaler().fit(X)
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=500, random_state=0)
        mlp.fit(sc.transform(X), P[:, 0])
        res = {"n_features": X.shape[1], "train_time_s": time.time() - t0, "val_score_r2": float(mlp.best_validation_score_)}
        for group, truths in (("test", TEST_TRUTHS), ("extra_offset_crescent", EXTRA_TRUTHS)):
            yt, yml, ymc = [], [], []
            for truth in truths:
                for rep in range(10):
                    row = [r for r in s3 if r["truth"] == truth and r["coverage"] == cov and int(r["rep"]) == rep]
                    if not row:
                        continue
                    syn = synth.make(raw, truth, rep, cov)
                    yml.append(float(mlp.predict(sc.transform(feat(syn.vis)[None]))[0]))
                    ymc.append(float(row[0]["d_M3"])); yt.append(synth.TRUTHS[truth][1])
            yt, yml, ymc = map(np.array, (yt, yml, ymc))
            rm_ml = float(np.sqrt(np.mean((yml - yt) ** 2))); rm_mc = float(np.sqrt(np.mean((ymc - yt) ** 2)))
            res[group] = {"n": int(len(yt)), "rmse_ml": rm_ml, "rmse_mcmc": rm_mc,
                          "bias_ml": float(np.mean(yml - yt)), "bias_mcmc": float(np.mean(ymc - yt))}
            if group == "test":
                ratio = rm_ml / rm_mc
                res[group]["rmse_ratio_ml_over_mcmc"] = ratio
                res[group]["verdict"] = ("ML이 나음" if ratio <= 0.9 else ("MCMC가 나음" if ratio >= 1.1 else "비슷함"))
                ax.plot(yt + np.random.default_rng(0).uniform(-0.3, 0.3, len(yt)) - 0.4, yml - yt, "o", ms=4,
                        color=plotstyle.SERIES[1], label=f"MLP (RMSE {rm_ml:.2f} μas)")
                ax.plot(yt + np.random.default_rng(1).uniform(-0.3, 0.3, len(yt)) + 0.4, ymc - yt, "s", ms=4,
                        color=plotstyle.SERIES[0], label=f"MCMC M3 (RMSE {rm_mc:.2f} μas)")
        real = DS.load(doy, band)
        xr = feat(real.raw.vis)
        res["real_data_ml_d"] = float(mlp.predict(sc.transform(xr[None]))[0])
        j = json.loads((RES / "stage2_fits.json").read_text())
        res["real_data_mcmc_M3_d"] = j[cov]["M3"]["summary"]["d"]["median"]
        out["per_coverage"][cov] = res
        ax.axhline(0, color=plotstyle.TEXT2, lw=1)
        ax.set_title(f"{cov}: real data ML {res['real_data_ml_d']:.1f} vs MCMC {res['real_data_mcmc_M3_d']:.1f} μas")
        ax.set_xlabel("true diameter (μas)")
        ax.legend(fontsize=8)
        print(cov, json.dumps(res), flush=True)
    axs[0].set_ylabel("predicted − true d (μas)")
    fig.tight_layout(); fig.savefig(RES / "fig_stage4_ml.png"); plt.close(fig)
    (RES / "stage4_ml.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
