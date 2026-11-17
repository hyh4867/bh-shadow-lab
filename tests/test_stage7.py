"""Unit tests for stage 7 (PREREGISTRATION 12): grouped model evaluation, vectorised prior transform,
emcee thinning, differential-evolution helper, mode grouping and the C4 check."""
import emcee
import numpy as np

from bhshadow import models as M
from bhshadow.closure import closure_phases, log_closure_amplitudes
from bhshadow.dataset import Dataset
from bhshadow.fitting import Problem
from bhshadow.nested import PriorTransform
from bhshadow.s7 import c4, de_fit, distinct_modes
from bhshadow.uvdata import VisTable
from test_stage6 import _FakeDS


def _synthetic_dataset(doy, band, seed, fn, p):
    """Five stations, two scans, model visibilities with 1 % noise."""
    st = ["AA", "AZ", "LM", "PV", "SM"]
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for i, a in enumerate(st) for b in st[i + 1:]]
    rows = [(s, a, b) for s in range(2) for a, b in pairs]
    n = len(rows)
    u = rng.uniform(-8e9, 8e9, n); v = rng.uniform(-8e9, 8e9, n)
    vis = fn(u, v, *p) * (1 + 0.01 * rng.standard_normal(n)) + 0j
    tab = VisTable(np.array([r[0] for r in rows], float), np.array([r[1] for r in rows]),
                   np.array([r[2] for r in rows]), u, v, vis, np.full(n, 0.01), meta={})
    tab.scan = np.array([r[0] for r in rows])
    return Dataset(f"{doy} {band}", doy, band, tab, tab, closure_phases(tab), log_closure_amplitudes(tab))


def _problem(model="G1"):
    fn = M.MODELS[model][0]
    p = [44.0, 8.0, 0.2, 150.0, 0.5, -20.0, 10.0, 30.0, 0.5, 40.0]
    ds = [_synthetic_dataset(doy, b, i, fn, p) for i, (doy, b) in enumerate([("095", "lo"), ("095", "hi"),
                                                                            ("096", "lo"), ("096", "hi")])]
    shared = ("d", "w") + tuple(M._gauss_names(1))
    return Problem(model, ds, groups=[0, 0, 1, 1], shared=shared), p


def test_grouped_evaluation_equals_per_dataset():
    prob, _ = _problem()
    rng = np.random.default_rng(0)
    th = prob.lo + rng.random((40, prob.ndim)) * (prob.hi - prob.lo)
    a = prob.chi2_parts(th); b = prob.chi2_parts_per_dataset(th)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_prior_transform_batch_equals_pointwise():
    ds = [_FakeDS("095"), _FakeDS("096")]
    shared = ("d", "w") + tuple(M._gauss_names(3))
    prob = Problem("R2G3", ds, groups=[0, 1], shared=shared)
    pt = PriorTransform(prob)
    U = np.random.default_rng(2).random((500, prob.ndim))
    assert np.allclose(pt.batch(U), np.array([pt(u) for u in U]))


def test_emcee_tau_with_thinning_is_in_steps():
    lp = lambda x: -0.5 * np.sum(x**2, axis=1)  # noqa: E731
    taus = []
    for tb in (1, 8):
        s = emcee.EnsembleSampler(24, 6, lp, vectorize=True)
        s.random_state = np.random.RandomState(3).get_state()
        st = s.run_mcmc(np.random.default_rng(0).normal(size=(24, 6)), 500); s.reset()
        s.run_mcmc(st, 24000 // tb, thin_by=tb)
        taus.append(s.get_autocorr_time(quiet=True) * tb)
    assert np.allclose(taus[0], taus[1], rtol=0.1)


def test_de_fit_recovers_synthetic_truth():
    prob, p = _problem()
    # fit only one group's worth of truth: shared d, w and the Gaussian, per-group beta
    r = de_fit(prob, seed=1, maxiter=300)
    assert abs(r["d"] - p[0]) < 1.0
    assert r["chi2"] < 3 * prob.n_data


def test_distinct_modes_and_c4():
    fits = [{"chi2": 100.0, "d": 45.0, "seed": 1}, {"chi2": 104.0, "d": 45.2, "seed": 2},
            {"chi2": 400.0, "d": 43.0, "seed": 3}, {"chi2": 101.0, "d": 47.0, "seed": 4}]
    modes = distinct_modes(fits)
    assert [m["count"] for m in modes] == [2, 1, 1]
    assert c4([105.0, 119.9], 100.0) and not c4([105.0, 121.0], 100.0)
