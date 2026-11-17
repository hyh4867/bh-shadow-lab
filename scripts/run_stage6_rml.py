"""Stage 6, 11.7 (reference only, no verdict): RML retry with other closure sets.

R1  ehtim maxset=True (count="max" for closure phases and log closure amplitudes)
R2  ehtim minimal sets, co-located quadrangles removed from the log closure amplitudes
R3  maximal sets, co-located quadrangles removed from the log closure amplitudes
Same settings as scripts/run_rml_image.py otherwise. Output: results/stage6_rml.json, results/fig_stage6_rml.png
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage5_rml_diag import FOV, NPIX, eht_chi2, our_chi2, ring_diameter  # noqa: E402

import ehtim as eh  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
COLOC = [{"AA", "AP"}, {"JC", "SM"}]


def drop_coloc(tab):
    keep = np.array([not any(p <= {r["t1"], r["t2"], r["t3"], r["t4"]} for p in COLOC) for r in tab])
    return tab[keep]


def ring_check(img):
    """Ring if the brightness at the centroid is < 0.5 x the mean ridge brightness (11.7)."""
    arr = img.imarr(); n = arr.shape[0]; pd = img.psize / eh.RADPERUAS
    yy, xx = np.mgrid[0:n, 0:n]
    cy, cx = (arr * yy).sum() / arr.sum(), (arr * xx).sum() / arr.sum()
    peaks = []
    for ang in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        rs = np.linspace(0, 40, 200)
        prof = [arr[int(round(cy - r / pd * np.cos(ang))) % n, int(round(cx - r / pd * np.sin(ang))) % n] for r in rs]
        peaks.append(max(prof))
    centre = float(arr[int(round(cy)), int(round(cx))])
    ridge = float(np.mean(peaks))
    return {"centre": centre, "ridge_mean": ridge, "ratio": centre / ridge if ridge > 0 else np.nan,
            "ring": bool(centre < 0.5 * ridge)}


def rml(obs, prior, maxset):
    img = prior.copy()
    for i in range(3):
        imgr = eh.imager.Imager(obs, img, prior_im=prior, flux=0.6, data_term={"cphase": 20, "logcamp": 20},
                                reg_term={"simple": 1, "tv2": 1, "flux": 10}, norm_reg=True, maxit=300,
                                ttype="direct", maxset=maxset)
        imgr.make_image_I(show_updates=False)
        img = imgr.out_last().blur_circ(10 * eh.RADPERUAS) if i < 2 else imgr.out_last()
    return img


def main():
    np.random.seed(11)
    d = DS.load("101", "hi")
    buf = io.StringIO()
    out = {"note": "reference only, no verdict (PREREGISTRATION 11.7)"}
    with contextlib.redirect_stdout(buf):
        obs0 = eh.obsdata.load_uvfits(str(DS.uvfits_path("101", "hi")))
        obs0.add_scans()
        obs0 = obs0.avg_coherent(0.0, scan_avg=True)
        obs0 = obs0.flag_uvdist(uv_min=0.1e9)
        obs0 = obs0.add_fractional_noise(0.01)
        prior = eh.image.make_square(obs0, NPIX, FOV * eh.RADPERUAS)
        prior = prior.add_gauss(0.6, (60 * eh.RADPERUAS, 60 * eh.RADPERUAS, 0, 0, 0))
        cp_min = obs0.c_phases(count="min", vtype="vis")
        cp_max = obs0.c_phases(count="max", vtype="vis")
        lca_min = obs0.c_amplitudes(count="min", vtype="vis", ctype="logcamp")
        lca_max = obs0.c_amplitudes(count="max", vtype="vis", ctype="logcamp")
        sets = {"R0 (stage 4: min sets)": (cp_min, lca_min), "R1 (max sets)": (cp_max, lca_max),
                "R2 (min sets, co-located lca removed)": (cp_min, drop_coloc(lca_min)),
                "R3 (max sets, co-located lca removed)": (cp_max, drop_coloc(lca_max))}
        imgs = {}
        for k, (cp, la) in sets.items():
            o = obs0.copy()
            o.cphase, o.logcamp = cp, la
            imgs[k] = rml(o, prior, maxset=True)  # maxset=True: use the tables as given
    counts = {k: {"n_cp": len(cp), "n_lca": len(la),
                  "n_lca_colocated": int(sum(any(p <= {r["t1"], r["t2"], r["t3"], r["t4"]} for p in COLOC) for r in la))}
              for k, (cp, la) in sets.items()}
    ps = FOV / NPIX
    out["results"] = {}
    for k, im in imgs.items():
        o = obs0.copy(); o.cphase, o.logcamp = sets[k]
        with contextlib.redirect_stdout(buf):
            ec = eht_chi2(obs0, im)  # ehtim default closure sets of obs0, same for every image
        rc = ring_check(im)
        oc = our_chi2(im.imarr(), ps, d)  # our closure set (2017-04-11 hi): same yardstick for every image
        oc["n_cp_ours"], oc["n_lca_ours"] = oc.pop("n_cp"), oc.pop("n_lca")
        out["results"][k] = {**ec, **oc, **counts[k], **rc,
                             "ring_diameter_from_image": ring_diameter(im) if rc["ring"] else None}
    r0 = out["results"]["R0 (stage 4: min sets)"]["ring"]
    r23 = any(out["results"][k]["ring"] for k in out["results"] if k.startswith(("R2", "R3")))
    out["candidate1"] = "부합" if (r23 and not r0) else "부합하지 않음"
    (RES / "stage6_rml.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    plotstyle.apply()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (k, im) in zip(axes, imgs.items()):
        arr = im.imarr(); n = arr.shape[0]
        ext = np.array([1, -1, -1, 1]) * n * ps / 2
        ax.imshow(arr, origin="upper", extent=ext, cmap="afmhot")
        r = out["results"][k]
        ax.set_title(f"{k}\nring={r['ring']}  cp {r['chisq_cphase']:.2f} / lca {r['chisq_logcamp']:.2f}", fontsize=8)
        ax.set_xlabel("RA (μas)"); ax.grid(False)
    axes[0].set_ylabel("Dec (μas)")
    fig.suptitle("RML retry, 2017-04-11 hi (reference only; not used for any result)", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "fig_stage6_rml.png"); plt.close(fig)
    print(json.dumps(out, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
