"""Unit tests: analytic visibilities vs direct Fourier sums, sign convention, null."""
import numpy as np
import pytest
from scipy.special import jn_zeros

from bhshadow import MUAS
from bhshadow import models as M

N, DX = 256, 0.5
C = (N - 1) / 2
COL, ROW = np.meshgrid(np.arange(N), np.arange(N))
X = (C - COL) * DX  # east positive, column 0 = east (left)
Y = (C - ROW) * DX  # north positive
R = np.hypot(X, Y)
PHI = np.arctan2(X, Y)  # position angle east of north
RNG = np.random.default_rng(0)
U = RNG.uniform(-8e9, 8e9, 40)
V = RNG.uniform(-8e9, 8e9, 40)


def thin_ring_image(d, beta=0.0, pa=0.0, width=0.6):
    img = np.exp(-0.5 * ((R - d / 2) / width) ** 2)
    return img * (1 + 2 * beta * np.cos(PHI - np.deg2rad(pa)))


@pytest.mark.parametrize("beta,pa", [(0.0, 0.0), (0.3, 0.0), (0.3, 90.0), (0.3, 120.0), (0.45, 250.0)])
def test_mring_matches_dft(beta, pa):
    # a radial Gaussian of sigma 0.6 muas ~ thin ring blurred by FWHM 1.41 muas
    vd = M.image_dft(thin_ring_image(44.0, beta, pa), DX, U, V)
    va = M.mring1(U, V, 44.0, 1.41, beta, pa)
    assert np.max(np.abs(vd - va)) < 3e-3


def test_mring_position_angle_not_mirrored():
    vd = M.image_dft(thin_ring_image(44.0, 0.3, 120.0), DX, U, V)
    # east-west mirror (240), north-south mirror (60), conjugation / 180 deg rotation (300)
    for wrong in (240.0, 60.0, 300.0):
        assert np.max(np.abs(vd - M.mring1(U, V, 44.0, 1.41, 0.3, wrong))) > 0.05


def test_disk_and_offset_crescent_match_dft():
    assert np.max(np.abs(M.image_dft((R <= 20).astype(float), DX, U, V) - M.disk(U, V, 40, 0))) < 2e-3
    ro, ri, s, pa = 23.5, 19.5, 2.5, 200.0
    x0, y0 = -s * np.sin(np.deg2rad(pa)), -s * np.cos(np.deg2rad(pa))
    img = ((R <= ro) & (np.hypot(X - x0, Y - y0) > ri)).astype(float)
    assert np.max(np.abs(M.image_dft(img, DX, U, V) - M.offset_crescent(U, V, ro, ri, s, pa, 0.0))) < 6e-3


def test_offset_crescent_bright_side():
    # brightness-weighted centroid lies toward pa
    ro, ri, s, pa = 23.5, 19.5, 2.5, 200.0
    x0, y0 = -s * np.sin(np.deg2rad(pa)), -s * np.cos(np.deg2rad(pa))
    img = ((R <= ro) & (np.hypot(X - x0, Y - y0) > ri)).astype(float)
    cx, cy = (img * X).sum() / img.sum(), (img * Y).sum() / img.sum()
    assert abs(((np.rad2deg(np.arctan2(cx, cy)) - pa) + 180) % 360 - 180) < 1.0


def test_unit_flux_at_origin():
    z = np.zeros(1)
    for fn, args in [(M.thin_ring, (40,)), (M.blurred_ring, (40, 10)), (M.mring1, (40, 10, 0.3, 50)),
                     (M.disk, (40, 10)), (M.gaussian, (30,)), (M.offset_crescent, (23, 19, 2, 10, 5))]:
        assert np.allclose(fn(z, z, *args), 1.0)


def test_thin_ring_first_null_position():
    d = 42.0
    b1 = jn_zeros(0, 1)[0] / (np.pi * d * MUAS)
    assert abs(M.thin_ring(np.array([b1]), np.array([0.0]), d)[0]) < 1e-12
    # Paper VI eq. 6: d0 ~ 45 (b1 / 3.5 Glambda)^-1 muas
    d_from_35 = jn_zeros(0, 1)[0] / (np.pi * 3.5e9) / MUAS
    assert abs(d_from_35 - 45.0) < 1.0


def test_blur_does_not_move_zeros():
    d = 42.0
    b1 = jn_zeros(0, 1)[0] / (np.pi * d * MUAS)
    assert abs(M.blurred_ring(np.array([b1]), np.array([0.0]), d, 15.0)[0]) < 1e-12


def test_vectorised_shapes():
    out = M.mring1(U, V, np.array([40.0, 44.0]), np.array([5.0, 10.0]), np.array([0.1, 0.2]), np.array([0.0, 90.0]))
    assert out.shape == (2, len(U))
    assert np.allclose(out[1], M.mring1(U, V, 44.0, 10.0, 0.2, 90.0))
