"""Unit tests for the stage 5 models (PREREGISTRATION 10.4-10.5): m=2 ring, ring + Gaussian,
xs-ring, and the extra prior constraint."""
import numpy as np
import pytest

from bhshadow import models as M
from test_models import DX, PHI, R, U, V, X, Y, thin_ring_image


@pytest.mark.parametrize("b1,pa1,b2,arg2", [(0.0, 0.0, 0.2, 0.0), (0.2, 30.0, 0.15, 100.0), (0.25, 200.0, 0.1, 300.0)])
def test_mring2_matches_dft(b1, pa1, b2, arg2):
    img = thin_ring_image(44.0) * (1 + 2 * b1 * np.cos(PHI - np.deg2rad(pa1)) + 2 * b2 * np.cos(2 * PHI - np.deg2rad(arg2)))
    vd = M.image_dft(img, DX, U, V)
    va = M.mring2(U, V, 44.0, 1.41, b1, pa1, b2, arg2)
    assert np.max(np.abs(vd - va)) < 3e-3


def test_mring2_reduces_to_mring1():
    assert np.allclose(M.mring2(U, V, 42, 10, 0.2, 77, 0.0, 12), M.mring1(U, V, 42, 10, 0.2, 77))


def test_gauss_offset_matches_dft():
    fwhm, x0, y0 = 12.0, 15.0, -8.0
    s = fwhm / (2 * np.sqrt(2 * np.log(2)))
    img = np.exp(-0.5 * ((X - x0) ** 2 + (Y - y0) ** 2) / s**2)
    assert np.max(np.abs(M.image_dft(img, DX, U, V) - M.gauss_offset(U, V, fwhm, x0, y0))) < 1e-3
    assert np.allclose(M.gauss_sd(U, V, s, x0, y0), M.gauss_offset(U, V, fwhm, x0, y0))


def xsring_image(r_out, psi, tau, pa, lnbeta, gamma):
    ri = r_out * (1 - psi)
    r0 = (1 - tau) * psi * r_out
    n = np.array([np.sin(np.deg2rad(pa)), np.cos(np.deg2rad(pa))])
    cx, cy = -r0 * n
    t = X * n[0] + Y * n[1]
    beta = np.exp(lnbeta)
    s = 0.5 * (beta + 1) + (1 - beta) / (2 * r_out) * t
    outer = R <= r_out
    inner = np.hypot(X - cx, Y - cy) <= ri
    return s * (outer & ~inner) + gamma * beta * inner


@pytest.mark.parametrize("pars", [(22.0, 0.3, 0.4, 200.0, np.log(0.4), 0.3),
                                  (21.0, 0.5, 0.1, 35.0, np.log(0.2), 0.0),
                                  (24.0, 0.2, 0.9, 300.0, np.log(1.5), 0.6)])
def test_xsring_matches_dft(pars):
    img = xsring_image(*pars)
    vd = M.image_dft(img, DX, U, V)
    va = M.xsring(U, V, *pars, 0.0)
    assert np.max(np.abs(vd - va)) < 8e-3


def test_xsring_flux_and_wide_side():
    pars = (22.0, 0.3, 0.4, 200.0, np.log(0.4), 0.3)
    img = xsring_image(*pars)
    # flux formula vs pixel sum, and Paper VI's V0 expression (h = 1)
    r_out, psi, tau, _, lnb, gam = pars
    b = np.exp(lnb)
    v0 = 0.5 * np.pi * r_out**2 * ((1 + b) - (1 - psi) ** 2 * ((1 + b) - psi * (1 - tau) * (1 - b) - 2 * gam * b))
    assert abs(M.xsring_flux_terms(r_out, psi, tau, lnb, gam) - v0) < 1e-9 * v0
    assert abs(img.sum() * DX**2 - v0) / v0 < 0.01
    assert np.allclose(M.xsring(np.zeros(1), np.zeros(1) + 1.0, *pars, 5.0), 1.0, atol=1e-6)
    cx, cy = (img * X).sum() / img.sum(), (img * Y).sum() / img.sum()
    assert abs(((np.rad2deg(np.arctan2(cx, cy)) - 200.0) + 180) % 360 - 180) < 1.0


def test_xsring_thin_symmetric_limit_is_thin_ring():
    # psi -> 0, tau = 1, beta = 1: ring of diameter R_out + R_in
    va = M.xsring(U / 4, V / 4, 21.0, 1e-3, 1.0, 0.0, 0.0, 0.0, 3.0)
    vr = M.blurred_ring(U / 4, V / 4, 2 * 21.0 * (1 - 0.5e-3), 3.0 * 2 * np.sqrt(2 * np.log(2)))
    assert np.max(np.abs(va - vr)) < 2e-3


def test_xsring_nuis2_vectorised_and_flux():
    p = np.array([[22, 0.3, 0.4, 200, -0.5, 0.2, 4, -20, 30, 10, -5, 8, 12, 0.3, 0.5],
                  [20, 0.2, 0.8, 100, 0.2, 0.1, 6, 0, 10, 0, 5, 10, 3, 0.1, 0.0]], float)
    out = M.xsring_nuis2(U, V, *p.T)
    assert out.shape == (2, len(U))
    assert np.allclose(out[1], M.xsring_nuis2(U, V, *p[1]))
    z = np.zeros(1)
    assert np.allclose(M.xsring_nuis2(z + 1.0, z, *p[0]), 1.0, atol=1e-6)
    e1 = M.mring1_gauss(z, z, 40, 10, 0.2, 30, 0.7, 10, 10, 50)
    assert np.allclose(e1, 1.0)


def test_e2_constraint_in_prior():
    from bhshadow.fitting import Problem

    class _D:  # minimal stand-in with no data
        n_data = 0
    prob = Problem("E2", [_D()])
    th = np.array([[42, 10, 0.3, 100, 0.1, 50], [42, 10, 0.3, 100, 0.25, 50]])
    assert list(prob.in_prior(th)) == [True, False]
