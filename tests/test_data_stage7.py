"""Data tests for stage 7 (skipped without the 2017 data): the grouped model evaluation gives the same
chi2 as the stage 2-6 per-data-set path on the real closure sets."""
import json
from pathlib import Path

import numpy as np
import pytest

from bhshadow import dataset as DS
from bhshadow.s6 import joint6

pytestmark = [pytest.mark.data, pytest.mark.skipif(not DS.data_available(), reason="EHT uvfits not downloaded")]
FITS6 = Path(__file__).resolve().parents[1] / "results" / "stage6_fits"


@pytest.mark.parametrize("model", ["M3", "E2", "G3", "R2G3"])
def test_grouped_chi2_equals_per_dataset_on_real_data(model):
    prob = joint6(model, DS.load_all())
    best = np.array(json.loads((FITS6 / f"m6_{model}_multi.json").read_text())["best"])
    rng = np.random.default_rng(5)
    th = np.vstack([best, prob.lo + rng.random((20, prob.ndim)) * (prob.hi - prob.lo)])
    a, b = prob.chi2_parts(th), prob.chi2_parts_per_dataset(th)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
