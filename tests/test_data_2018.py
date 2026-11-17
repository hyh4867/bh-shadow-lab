"""Data tests for the 2018 release (skipped when data/raw/uvfits2018 is absent)."""
from pathlib import Path

import numpy as np
import pytest

from bhshadow import dataset as DS
from bhshadow.uvdata import read_uvfits

pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not DS.data2018_available(), reason="EHT 2018 uvfits not downloaded")]
CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "csv2018"


def test_reader_matches_2018_csv():
    p = DS.uvfits_path_2018("111", "b3")
    c_path = CSV / (p.stem + ".csv")
    if not c_path.exists():
        pytest.skip("csv not downloaded")
    raw = read_uvfits(p, conjugate=False)
    c = np.genfromtxt(c_path, delimiter=",", comments="#", dtype=None, encoding=None)
    amp = np.array([r[5] for r in c], float)
    ph = np.array([r[6] for r in c], float)
    sig = np.array([r[7] for r in c], float)
    assert len(amp) == len(raw)
    assert np.max(np.abs(np.abs(raw.vis) - amp) / amp) < 1e-4
    assert np.max(np.abs(raw.sigma - sig) / sig) < 1e-4
    assert np.max(np.abs((np.rad2deg(np.angle(raw.vis)) - ph + 180) % 360 - 180)) < 0.01
    assert raw.meta["rr_eq_ll"]


def test_2018_intrasite_removed_and_stations():
    d = DS.load_2018("111", "b3")
    pairs = {frozenset((a, b)) for a, b in zip(d.tab.t1, d.tab.t2)}
    assert frozenset(("AA", "AX")) not in pairs and frozenset(("MM", "SW")) not in pairs
    assert {"GL", "LM", "PV"} <= set(d.tab.t1) | set(d.tab.t2)
    assert len(d.cp) > 20 and len(d.lca) > 20
