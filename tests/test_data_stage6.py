"""Data regression tests for stage 6 numbers (skipped without the 2017 data)."""
import json
from pathlib import Path

import numpy as np
import pytest

from bhshadow import dataset as DS
from bhshadow.s6 import joint6

pytestmark = [pytest.mark.data, pytest.mark.skipif(not DS.data_available(), reason="EHT uvfits not downloaded")]
RES = Path(__file__).resolve().parents[1] / "results"
FITS6 = RES / "stage6_fits"


@pytest.fixture(scope="module")
def dsets():
    return DS.load_all()


@pytest.mark.parametrize("model", ["M3", "E1", "G2"])
def test_chi2_of_committed_nested_best_fits(dsets, model):
    """chi2 of the committed best-fit parameters (best of 3 nested runs) reproduces the stored value."""
    fit = json.loads((FITS6 / f"m6_{model}_multi.json").read_text())
    prob = joint6(model, dsets)
    assert prob.names == fit["names"]
    assert abs(prob.chi2(np.array(fit["best"]))[0] - fit["chi2_min"]) < 1e-6 * fit["chi2_min"]


def test_colocated_cut_keeps_closure_phases_and_drops_quadrangles(dsets):
    cut = DS.load_all(drop_colocated=True)
    for a, b in zip(dsets, cut):
        assert len(a.cp) == len(b.cp)
        assert len(b.lca) <= len(a.lca)
    s = json.loads((RES / "stage6_systematics.json").read_text())["colocated_cut"]["counts"]
    assert [c["n_lca_cut"] for c in s] == [len(b.lca) for b in cut]


def test_stage6_selection_recorded_under_both_rules():
    m = json.loads((RES / "stage6_models.json").read_text())
    assert m["central_preregistered"]["model"] == "M3"
    assert m["central"]["model"] == "E1"
    t = {r["model"]: r for r in m["table"]}
    assert t["E1"]["C1"] and t["E1"]["C2"] and not t["E1"]["C3"] and t["E1"]["C3p"]
