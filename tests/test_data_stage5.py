"""Data regression tests for stage 5 numbers (skipped without the 2017 data)."""
import json
from pathlib import Path

import numpy as np
import pytest

from bhshadow import dataset as DS
from bhshadow.s5 import joint_problem

pytestmark = [pytest.mark.data, pytest.mark.skipif(not DS.data_available(), reason="EHT uvfits not downloaded")]
RES = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def dsets():
    return DS.load_all()


def test_joint_chi2_of_committed_best_fits(dsets):
    """chi2 of the committed best-fit parameters reproduces the stored values (M3 and E1)."""
    st2 = json.loads((RES / "stage2_joint.json").read_text())
    e1 = json.loads((RES / "stage5_fits" / "ext_E1_all.json").read_text())
    for model, fit in (("M3", st2["fit"]), ("E1", e1)):
        prob = joint_problem(model, dsets)
        assert prob.names == fit["names"]
        assert abs(prob.chi2(np.array(fit["best"]))[0] - fit["chi2_min"]) < 1e-6 * fit["chi2_min"]


def test_lmt_log_closure_amplitude_misfit(dsets):
    """Stage 5 finding: the M3 log-closure-amplitude misfit sits in LMT quadrangles."""
    s = json.loads((RES / "stage5_systematics.json").read_text())["lmt_reference"]["closure_chi2_M3_joint_best"]
    assert s["lca_LMT"]["chi2_per_closure"] > 5 * s["lca_other"]["chi2_per_closure"]


def test_e1_preferred_over_m3_by_bic():
    ext = json.loads((RES / "stage5_systematics.json").read_text())["model_extensions"]
    assert ext["central_model"] == "E1"
    assert ext["M3"]["bic"] - ext["E1"]["bic"] > 10
