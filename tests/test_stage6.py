"""Unit tests for stage 6 (PREREGISTRATION 11): elliptical Gaussian, ring + Gaussians models,
prior transform (ordering, triangle fold), R-hat, co-located quadrangle cut."""
import numpy as np
import pytest

from bhshadow import models as M
from bhshadow.closure import log_closure_amplitudes
from bhshadow.fitting import Problem
from bhshadow.nested import PriorTransform, rhat
from bhshadow.uvdata import VisTable
from test_models import DX, U, V, X, Y, thin_ring_image


@pytest.mark.parametrize("x0,y0,a,q,th", [(10.0, -5.0, 14.0, 0.5, 30.0), (-8.0, 12.0, 18.0, 0.3, 120.0), (0.0, 0.0, 10.0, 1.0, 0.0)])
def test_egauss_matches_dft(x0, y0, a, q, th):
    sa = a / M.FWHM_PER_SD
    sb = sa * q
    t = np.deg2rad(th)
    xp = (X - x0) * np.sin(t) + (Y - y0) * np.cos(t)   # along major axis
    xq = (X - x0) * np.cos(t) - (Y - y0) * np.sin(t)
    img = np.exp(-0.5 * (xp**2 / sa**2 + xq**2 / sb**2))
    assert np.max(np.abs(M.image_dft(img, DX, U, V) - M.egauss(U, V, x0, y0, a, q, th))) < 2e-3


def test_egauss_circular_limit_equals_gauss_offset():
    assert np.allclose(M.egauss(U, V, 7, -3, 25, 1.0, 40), M.gauss_offset(U, V, 25, 7, -3))


def test_ring_plus_gaussians_matches_dft_and_flux():
    fn = M.MODELS["G2"][0]
    p = [44.0, 1.41, 0.0, 0.0, 0.5, -20, 10, 12, 0.5, 30, 0.3, 15, -15, 10, 0.8, 100]
    ring = thin_ring_image(44.0)
    ring = ring / ring.sum()

    def eg(x0, y0, a, q, th):
        sa = a / M.FWHM_PER_SD; sb = sa * q; t = np.deg2rad(th)
        xp = (X - x0) * np.sin(t) + (Y - y0) * np.cos(t); xq = (X - x0) * np.cos(t) - (Y - y0) * np.sin(t)
        g = np.exp(-0.5 * (xp**2 / sa**2 + xq**2 / sb**2))
        return g / g.sum()

    img = ring + 0.5 * eg(-20, 10, 12, 0.5, 30) + 0.3 * eg(15, -15, 10, 0.8, 100)
    va = fn(U, V, *p)
    assert np.max(np.abs(M.image_dft(img, DX, U, V) - va)) < 3e-3
    assert np.isclose(fn(np.array([0.0]), np.array([0.0]), *p)[0], 1.0)


def test_model_registry_sizes():
    assert len(M.MODELS["G3"][1]) == 4 + 18
    assert len(M.MODELS["R2G2"][1]) == 6 + 12
    assert len(M.MODELS["XE3L"][1]) == 7 + 18 + 2


class _FakeDS:
    def __init__(self, doy):
        self.doy = doy
        self.n_data = 10


def test_prior_transform_orders_x_and_folds_beta():
    ds = [_FakeDS("095"), _FakeDS("096")]
    names = M.MODELS["R2G2"][1]
    shared = ("d", "w") + tuple(M._gauss_names(2))
    prob = Problem("R2G2", ds, groups=[0, 1], shared=shared)
    pt = PriorTransform(prob)
    rng = np.random.default_rng(1)
    th = np.array([pt(rng.uniform(size=prob.ndim)) for _ in range(4000)])
    assert np.all(th[:, prob.names.index("x1")] <= th[:, prob.names.index("x2")])
    for g in (0, 1):
        b1, b2 = th[:, prob.slot[("beta", g)]], th[:, prob.slot[("beta2", g)]]
        assert np.all(b1 + b2 <= 0.5 + 1e-12)
        # uniform on the triangle: mean of beta = 0.5/3
        assert abs(b1.mean() - 0.5 / 3) < 0.01
    assert np.all(prob.in_prior(th))


def test_rhat_detects_offset_runs():
    rng = np.random.default_rng(0)
    same = [rng.normal(0, 1, (2000, 2)) for _ in range(3)]
    assert np.all(rhat(same) < 1.01)
    off = [rng.normal(m, 1, (2000, 2)) for m in (0, 0, 1.5)]
    assert np.all(rhat(off) > 1.1)


def test_rhat_periodic_wraps():
    rng = np.random.default_rng(0)
    runs = [((rng.normal(0, 3, (2000, 1))) % 360) for _ in range(3)]
    assert rhat(runs, periodic=np.array([True]))[0] < 1.01


def test_colocated_quadrangles_removed():
    st = ["AA", "AP", "LM", "PV", "SM"]
    rows = [(a, b) for i, a in enumerate(st) for b in st[i + 1:] if {a, b} != {"AA", "AP"}]
    t1 = np.array([r[0] for r in rows]); t2 = np.array([r[1] for r in rows])
    rng = np.random.default_rng(3)
    n = len(rows)
    tab = VisTable(np.zeros(n), t1, t2, rng.normal(0, 1e9, n), rng.normal(0, 1e9, n),
                   rng.uniform(0.5, 1, n) * np.exp(1j * rng.uniform(0, 6, n)), np.full(n, 0.01), meta={})
    tab.scan = np.zeros(n, int)
    full = log_closure_amplitudes(tab)
    cut = log_closure_amplitudes(tab, drop_colocated=True)
    assert any("AA" in l and "AP" in l for l in full.label)
    assert not any("AA" in l and "AP" in l for l in cut.label)
    assert 0 < len(cut) <= len(full)
