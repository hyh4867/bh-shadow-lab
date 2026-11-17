"""Stage 5 (optional, no verdict): why did the RML image not show a ring? PREREGISTRATION 10.9.

D1  chi^2 of the same images under eht-imaging `chisq` and under our closure code:
    the stage-4 RML image, a pixel rendering of the 2017-04-11 hi M3 best fit, a 40-muas Gaussian
D2  RML started from the M3 image (initial image and prior)
D3  RML with data weights x10 and more iterations
Output: results/stage5_rml_diag.json, results/fig_stage5_rml_diag.png
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhshadow import plotstyle  # noqa: E402
from bhshadow import dataset as DS  # noqa: E402
from bhshadow.closure import model_closure_phase, model_log_camp  # noqa: E402
from bhshadow.models import image_dft  # noqa: E402

import ehtim as eh  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
NPIX, FOV = 48, 160.0


def our_chi2(arr, psize_muas, d):
    V = image_dft(arr, psize_muas, d.tab.u, d.tab.v)[None, :]
    r = d.cp.obs - model_closure_phase(V, d.cp)[0]
    c_cp = float((2 * (1 - np.cos(r)) / d.cp.sigma**2).sum())
    r = d.lca.obs - model_log_camp(V, d.lca)[0]
    c_la = float((r**2 / d.lca.sigma**2).sum())
    return {"chi2_cp_per_closure": c_cp / len(d.cp), "chi2_lca_per_closure": c_la / len(d.lca),
            "n_cp": len(d.cp), "n_lca": len(d.lca)}


def eht_chi2(obs, img):
    return {"chisq_cphase": float(obs.chisq(img, dtype="cphase", ttype="direct")),
            "chisq_logcamp": float(obs.chisq(img, dtype="logcamp", ttype="direct"))}


def ring_diameter(img):
    arr = img.imarr(); n = arr.shape[0]; pd = img.psize / eh.RADPERUAS
    yy, xx = np.mgrid[0:n, 0:n]
    cy, cx = (arr * yy).sum() / arr.sum(), (arr * xx).sum() / arr.sum()
    radii = []
    for ang in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        rs = np.linspace(0, 40, 200)
        prof = [arr[int(round(cy - r / pd * np.cos(ang))) % n, int(round(cx - r / pd * np.sin(ang))) % n] for r in rs]
        radii.append(rs[int(np.argmax(prof))])
    return 2 * float(np.mean(radii))


def m3_image(template, d_, w_, beta, pa):
    """Pixel image of the M3 model, ehtim orientation (row 0 = north, column 0 = east)."""
    n = template.xdim; ps = template.psize / eh.RADPERUAS
    c = (n - 1) / 2
    col, row = np.meshgrid(np.arange(n), np.arange(n))
    X, Y = (c - col) * ps, (c - row) * ps
    R, PHI = np.hypot(X, Y), np.arctan2(X, Y)
    sd = w_ / (2 * np.sqrt(2 * np.log(2)))
    arr = np.exp(-0.5 * ((R - d_ / 2) / sd) ** 2) * (1 + 2 * beta * np.cos(PHI - np.deg2rad(pa)))
    arr = np.clip(arr, 0, None)
    arr *= 0.6 / arr.sum()
    im = template.copy()
    im.imvec = arr.ravel()
    return im


def rml(obs, init, prior, weights=20, maxit=300, rounds=3):
    img = init.copy()
    for i in range(rounds):
        imgr = eh.imager.Imager(obs, img, prior_im=prior, flux=0.6,
                                data_term={"cphase": weights, "logcamp": weights},
                                reg_term={"simple": 1, "tv2": 1, "flux": 10}, norm_reg=True, maxit=maxit, ttype="direct")
        imgr.make_image_I(show_updates=False)
        img = imgr.out_last().blur_circ(10 * eh.RADPERUAS) if i < rounds - 1 else imgr.out_last()
    return img


def main():
    np.random.seed(11)
    d = DS.load("101", "hi")
    fit = json.loads((RES / "stage2_fits.json").read_text())["2017-04-11 hi"]["M3"]["summary"]
    pars = [fit[k]["median"] for k in ("d", "w", "beta", "beta_pa")]
    buf = io.StringIO()
    out = {"note": "optional diagnosis, no verdict (PREREGISTRATION 10.9)", "M3_params_2017_04_11_hi": pars}
    with contextlib.redirect_stdout(buf):
        obs = eh.obsdata.load_uvfits(str(DS.uvfits_path("101", "hi")))
        obs.add_scans()
        obs = obs.avg_coherent(0.0, scan_avg=True)
        obs = obs.flag_uvdist(uv_min=0.1e9)
        obs = obs.add_fractional_noise(0.01)
        prior = eh.image.make_square(obs, NPIX, FOV * eh.RADPERUAS)
        prior = prior.add_gauss(0.6, (60 * eh.RADPERUAS, 60 * eh.RADPERUAS, 0, 0, 0))
        img_rml = rml(obs, prior, prior)  # same procedure as scripts/run_rml_image.py
        img_m3 = m3_image(prior, *pars)
        img_g = eh.image.make_square(obs, NPIX, FOV * eh.RADPERUAS).add_gauss(
            0.6, (40 * eh.RADPERUAS, 40 * eh.RADPERUAS, 0, 0, 0))
        img_d2 = rml(obs, img_m3, img_m3)
        img_d3 = rml(obs, prior, prior, weights=200, maxit=1000)
        n_cp_eht = len(obs.c_phases(count="min", vtype="vis"))
        n_lca_eht = len(obs.c_amplitudes(count="min", vtype="vis", ctype="logcamp"))
    ps = FOV / NPIX
    images = {"RML (stage 4 settings)": img_rml, "M3 best fit (pixels)": img_m3, "Gaussian 40 muas": img_g,
              "D2: RML from M3 image": img_d2, "D3: RML, weights x10": img_d3}
    out["D1_D2_D3"] = {}
    for k, im in images.items():
        arr = im.imarr()
        out["D1_D2_D3"][k] = {**eht_chi2(obs, im), **our_chi2(arr, ps, d), "ring_diameter_from_image": ring_diameter(im)}
    out["closure_counts"] = {"ehtim_min_cphase": n_cp_eht, "ehtim_min_logcamp": n_lca_eht,
                             "ours_cphase": len(d.cp), "ours_logcamp": len(d.lca)}
    # orientation check: our chi2 for the M3 image vs the analytic M3 chi2 per closure
    out["M3_analytic_chi2_nu_stage2"] = json.loads((RES / "stage2_fits.json").read_text())["2017-04-11 hi"]["M3"]["chi2_nu"]
    (RES / "stage5_rml_diag.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    plotstyle.apply()
    fig, axs = plt.subplots(1, len(images), figsize=(3.2 * len(images), 3.4))
    ext = np.array([1, -1, -1, 1]) * FOV / 2
    for ax, (k, im) in zip(axs, images.items()):
        ax.imshow(im.imarr(), origin="upper", extent=ext, cmap="afmhot"); ax.grid(False)
        r = out["D1_D2_D3"][k]
        ax.set_title(f"{k}\nours χ²/N: cp {r['chi2_cp_per_closure']:.1f}, lca {r['chi2_lca_per_closure']:.1f}\n"
                     f"ehtim: cp {r['chisq_cphase']:.2f}, lca {r['chisq_logcamp']:.2f}", fontsize=7.5)
        ax.set_xlabel("RA (μas) ← east")
    axs[0].set_ylabel("Dec (μas)")
    fig.tight_layout(); fig.savefig(RES / "fig_stage5_rml_diag.png"); plt.close(fig)


if __name__ == "__main__":
    main()
