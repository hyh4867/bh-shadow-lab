"""Regression tests on the real EHT data (skipped when data/raw/uvfits is absent).

Reference numbers come from the committed results/ files (stage 1-2 runs).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from bhshadow import DAYS
from bhshadow import dataset as DS
from bhshadow.fitting import Problem, optimise
from bhshadow.nullfit import null_fit
from bhshadow.uvdata import prepare, read_uvfits

pytestmark = [pytest.mark.data, pytest.mark.skipif(not DS.data_available(), reason="EHT uvfits not downloaded")]
RES = Path(__file__).resolve().parents[1] / "results"
CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "csv"


@pytest.mark.parametrize("doy", list(DAYS))
def test_reader_matches_release_csv(doy):
    raw = read_uvfits(DS.uvfits_path(doy, "hi"), conjugate=False)
    p = CSV / f"SR1_M87_2017_{doy}_hi_hops_netcal_StokesI.csv"
    if not p.exists():
        pytest.skip("csv not downloaded")
    c = np.genfromtxt(p, delimiter=",", comments="#", dtype=None, encoding=None)
    amp = np.array([r[5] for r in c], float)
    sig = np.array([r[7] for r in c], float)
    assert len(amp) == len(raw)
    assert np.max(np.abs(np.abs(raw.vis) - amp) / amp) < 1e-4
    assert np.max(np.abs(raw.sigma - sig) / sig) < 1e-4
    assert raw.meta["rr_eq_ll"]


def test_counts_match_stage1_summary():
    ref = json.loads((RES / "stage1_data_summary.json").read_text())["datasets"]
    for r in ref:
        doy = [k for k, v in DAYS.items() if r["dataset"].startswith(v)][0]
        d = DS.load(doy, r["dataset"].split()[-1])
        assert (len(d.raw), len(d.cp), len(d.lca), d.tab.meta["n_scans"]) == (
            r["n_10s"], r["n_cphase"], r["n_logcamp"], r["n_scans"])


def test_combined_null_regression():
    ref = json.loads((RES / "stage1_null.json").read_text())["combined"]
    rho, amp, sig, sid = [], [], [], []
    for doy in DAYS:
        for band in ("lo", "hi"):
            t = prepare(read_uvfits(DS.uvfits_path(doy, band)), snr_min=0.0)
            rho.append(t.rho); amp.append(np.abs(t.vis)); sig.append(t.sigma)
            sid.append(np.array([f"{doy}{band}{s}" for s in t.scan]))
    r = null_fit(*(np.concatenate(x) for x in (rho, amp, sig, sid)), n_boot=50)
    assert abs(r["b1_glambda"] - ref["b1_glambda"]) < 1e-6


@pytest.mark.parametrize("doy,band", [("096", "hi"), ("101", "lo")])
def test_m3_optimum_close_to_posterior_median(doy, band):
    ref = json.loads((RES / "stage2_fits.json").read_text())
    d = DS.load(doy, band)
    x, _ = optimise(Problem("M3", [d]), seed=1)
    assert abs(x[0] - ref[d.name]["M3"]["summary"]["d"]["median"]) < 0.3


def test_m3_beats_symmetric_and_disk_models_by_bic():
    ref = json.loads((RES / "stage2_fits.json").read_text())
    for name, v in ref.items():
        assert v["_bic_rank"][0] == "M3", name
