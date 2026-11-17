"""Optional: one regularised maximum-likelihood (RML) image with eht-imaging, for reference only.

Not used for any verdict (PREREGISTRATION 8). Closure phases + log closure
amplitudes only, 2017-04-11 hi, Gaussian prior image, simple settings that were not
tuned against the result. Output: results/fig_rml_image.png, results/rml_image.json
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

import ehtim as eh  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    np.random.seed(11)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        obs = eh.obsdata.load_uvfits(str(DS.uvfits_path("101", "hi")))
        obs.add_scans()
        obs = obs.avg_coherent(0.0, scan_avg=True)
        obs = obs.flag_uvdist(uv_min=0.1e9)  # remove intra-site baselines
        obs = obs.add_fractional_noise(0.01)
        fov = 160 * eh.RADPERUAS
        npix = 48
        prior = eh.image.make_square(obs, npix, fov)
        prior = prior.add_gauss(0.6, (60 * eh.RADPERUAS, 60 * eh.RADPERUAS, 0, 0, 0))
        img = prior.copy()
        for i in range(3):
            imgr = eh.imager.Imager(obs, img, prior_im=prior, flux=0.6,
                                    data_term={"cphase": 20, "logcamp": 20},
                                    reg_term={"simple": 1, "tv2": 1, "flux": 10},
                                    norm_reg=True, maxit=300, ttype="direct")
            imgr.make_image_I(show_updates=False)
            img = imgr.out_last().blur_circ(10 * eh.RADPERUAS) if i < 2 else imgr.out_last()
        chi_cp = obs.chisq(img, dtype="cphase", ttype="direct")
        chi_lca = obs.chisq(img, dtype="logcamp", ttype="direct")
    arr = img.imarr()
    # ring diameter from the image: peak radius averaged over angle around the brightness centroid
    n = arr.shape[0]
    pd = img.psize / eh.RADPERUAS
    yy, xx = np.mgrid[0:n, 0:n]
    cy, cx = (arr * yy).sum() / arr.sum(), (arr * xx).sum() / arr.sum()
    radii = []
    for ang in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        rs = np.linspace(0, 40, 200)
        prof = [arr[int(round(cy - r / pd * np.cos(ang))) % n, int(round(cx - r / pd * np.sin(ang))) % n] for r in rs]
        radii.append(rs[int(np.argmax(prof))])
    d_img = 2 * float(np.mean(radii))
    out = {"dataset": "2017-04-11 hi", "chi2_cphase": float(chi_cp), "chi2_logcamp": float(chi_lca),
           "npix": npix, "fov_muas": 160, "ring_diameter_from_image_muas": d_img,
           "note": "reference only; settings fixed before looking at the image; not used for any verdict"}
    (RES / "rml_image.json").write_text(json.dumps(out, indent=1))
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ext = np.array([1, -1, -1, 1]) * n * pd / 2
    im = ax.imshow(arr, origin="upper", extent=ext, cmap="afmhot")
    ax.set_xlabel("relative RA (μas)  ← east"); ax.set_ylabel("relative Dec (μas)")
    ax.grid(False)
    ax.set_title(f"RML image (ehtim), 2017-04-11 hi\nreference only; ridge d ≈ {d_img:.0f} μas", fontsize=9)
    fig.colorbar(im, ax=ax, label="Jy / pixel")
    fig.tight_layout(); fig.savefig(RES / "fig_rml_image.png"); plt.close(fig)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
