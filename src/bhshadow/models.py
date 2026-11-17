"""Geometric source models in the Fourier domain.

Convention (see docs/THEORY.md): V(u, v) = sum I(x, y) exp[-2 pi i (u x + v y)],
x = offset toward east (RA), y = offset toward north. Position angles are measured
east of north. Sizes are in micro-arcseconds, u and v in wavelengths.
All models are normalised to unit total flux at the origin.

Every function accepts parameter arrays of shape (n,) or scalars and uv arrays
of shape (m,), and returns complex visibilities of shape (n, m) (or (m,)).
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0, j1

from . import MUAS

FOUR_LN2 = 4.0 * np.log(2.0)


def _col(p):
    p = np.asarray(p, float)
    return p[..., None] if p.ndim else p


def blur(u, v, fwhm):
    rho2 = u**2 + v**2
    w = _col(fwhm) * MUAS
    return np.exp(-(np.pi * w) ** 2 * rho2 / FOUR_LN2)


def thin_ring(u, v, d):
    z = np.pi * _col(d) * MUAS * np.hypot(u, v)
    return j0(z) + 0j


def blurred_ring(u, v, d, w):
    return thin_ring(u, v, d) * blur(u, v, w)


def mring1(u, v, d, w, beta_amp, beta_pa_deg):
    """Blurred m=1 ring. Brightness along the ring: 1 + 2|b| cos(phi - arg b).

    V = [J0(z) + J1(z) (b e^{i chi} - b* e^{-i chi})] * blur,  chi = phi_u - pi/2,
    where phi_u is the position angle of (u, v) east of north (atan2(u, v)).
    """
    rho = np.hypot(u, v)
    z = np.pi * _col(d) * MUAS * rho
    phi_u = np.arctan2(u, v)
    # coefficient of e^{+i phi} in the brightness is |b| e^{-i pa} (max at phi = pa)
    b = _col(beta_amp) * np.exp(-1j * np.deg2rad(_col(beta_pa_deg)))
    e = np.exp(1j * (phi_u - np.pi / 2))
    V = j0(z) + j1(z) * (b * e - np.conj(b) * np.conj(e))
    return V * blur(u, v, w)


def _airy(z):
    """2 J1(z)/z with the z -> 0 limit."""
    z = np.asarray(z, float)
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-8
    out[nz] = 2.0 * j1(z[nz]) / z[nz]
    return out


def disk(u, v, D, w):
    z = np.pi * _col(D) * MUAS * np.hypot(u, v)
    return _airy(z) * blur(u, v, w) + 0j


def gaussian(u, v, fwhm):
    return blur(u, v, fwhm) + 0j


def offset_crescent(u, v, r_out, r_in, shift, pa_deg, fwhm):
    """Uniform disk (r_out) minus a uniform disk (r_in) shifted by `shift`.

    The inner disk is shifted away from position angle pa_deg, so the thick
    (bright) side of the crescent lies at pa_deg. Normalised to unit flux.
    """
    rho = np.hypot(u, v)
    ro, ri, s = (_col(x) * MUAS for x in (r_out, r_in, shift))
    pa = np.deg2rad(_col(pa_deg))
    x0, y0 = -s * np.sin(pa), -s * np.cos(pa)  # inner-disk centre (east, north)
    a_out = np.pi * ro**2 * _airy(2 * np.pi * ro * rho)
    a_in = np.pi * ri**2 * _airy(2 * np.pi * ri * rho) * np.exp(-2j * np.pi * (u * x0 + v * y0))
    return (a_out - a_in) / (np.pi * (ro**2 - ri**2)) * blur(u, v, fwhm)


def image_dft(img, dx_muas, u, v):
    """Direct Fourier sum of a pixel image (row 0 = north, column 0 = east).

    Used only in tests to check the analytic models and the sign convention.
    """
    ny, nx = img.shape
    x = (np.arange(nx)[::-1] - (nx - 1) / 2) * dx_muas * MUAS  # east to the left
    y = (np.arange(ny)[::-1] - (ny - 1) / 2) * dx_muas * MUAS  # north up
    X, Y = np.meshgrid(x, y)
    f = img.ravel() / img.sum()
    ph = np.exp(-2j * np.pi * (np.outer(u, X.ravel()) + np.outer(v, Y.ravel())))
    return ph @ f


# ---------------------------------------------------------------------------
# Stage 5 models (PREREGISTRATION section 10)
# ---------------------------------------------------------------------------

def gauss_offset(u, v, fwhm, x0, y0):
    """Unit-flux circular Gaussian (FWHM) centred at (x0 east, y0 north) in muas."""
    shift = np.exp(-2j * np.pi * (u * _col(x0) + v * _col(y0)) * MUAS)
    return blur(u, v, fwhm) * shift


def gauss_sd(u, v, sd, x0, y0):
    """Unit-flux circular Gaussian with standard deviation sd (Paper VI convention)."""
    rho2 = u**2 + v**2
    s = _col(sd) * MUAS
    return np.exp(-2 * np.pi**2 * s**2 * rho2) * np.exp(-2j * np.pi * (u * _col(x0) + v * _col(y0)) * MUAS)


def mring1_gauss(u, v, d, w, beta_amp, beta_pa_deg, fg, xg, yg, fwhmg):
    """E1: blurred m=1 ring plus one offset circular Gaussian with flux ratio fg."""
    fg_ = _col(fg)
    return (mring1(u, v, d, w, beta_amp, beta_pa_deg) + fg_ * gauss_offset(u, v, fwhmg, xg, yg)) / (1 + fg_)


def mring2(u, v, d, w, beta_amp, beta_pa_deg, beta2_amp, beta2_arg_deg):
    """E2: blurred ring with brightness 1 + 2|b1| cos(phi - eta1) + 2|b2| cos(2 phi - arg b2).

    General m-ring term: beta_k J_k(z) e^{i k (phi_u - pi/2)} (THEORY section 3), with the
    coefficient of e^{+i k phi} in the brightness equal to |b_k| e^{-i arg_k}.
    """
    rho = np.hypot(u, v)
    z = np.pi * _col(d) * MUAS * rho
    phi_u = np.arctan2(u, v)
    b1 = _col(beta_amp) * np.exp(-1j * np.deg2rad(_col(beta_pa_deg)))
    b2 = _col(beta2_amp) * np.exp(-1j * np.deg2rad(_col(beta2_arg_deg)))
    e = np.exp(1j * (phi_u - np.pi / 2))
    J0, J1 = j0(z), j1(z)
    zs = np.where(z > 1e-6, z, 1.0)
    J2 = np.where(z > 1e-6, 2 * J1 / zs - J0, 0.0)  # recurrence, faster than jv(2, z)
    V = J0 + J1 * (b1 * e - np.conj(b1) * np.conj(e)) + J2 * (b2 * e**2 + np.conj(b2) * np.conj(e) ** 2)
    return V * blur(u, v, w)


def _disk_amp(R, rho):
    """Fourier amplitude of a unit-brightness disk of radius R (radians): R J1(2 pi R rho) / rho."""
    k = 2 * np.pi * R
    return R * j1(k * rho) / rho


def _disk_amp_deriv(R, rho):
    """d/d rho of _disk_amp."""
    k = 2 * np.pi * R
    x = k * rho
    return R * (x * j0(x) - 2 * j1(x)) / rho**2


def xsring(u, v, r_out, psi, tau, pa_deg, lnbeta, gamma, sigma):
    """Paper VI xs-ring (Appendix B.1.1), Fourier transform derived independently (THEORY 7).

    Image: s(t) [D(R_out, 0) - D(R_in, c)] + K D(R_in, c), blurred by a Gaussian of standard
    deviation sigma, normalised to unit flux. t is the coordinate along n = (sin pa, cos pa),
    the direction of the widest part of the crescent; the inner disk centre is c = -r0 n.
    R_in = R_out (1 - psi), r0 = (1 - tau) psi R_out.
    Slash: s = h1 at t = -R_out (thin side), h2 = 1 at t = +R_out (wide side); beta = h1/h2.
    Floor amplitude K = gamma h1. Parameter sizes in muas.
    """
    rho = np.maximum(np.hypot(u, v), 1e-3)
    ro = _col(r_out) * MUAS
    ps = _col(psi)
    ri = ro * (1 - ps)
    r0 = (1 - _col(tau)) * ps * ro
    pa = np.deg2rad(_col(pa_deg))
    beta = np.exp(_col(lnbeta))
    K = _col(gamma) * beta
    s0 = 0.5 * (beta + 1.0)
    g = (1.0 - beta) / (2 * ro)
    nu = u * np.sin(pa) + v * np.cos(pa)  # n . u
    cos_nu = nu / rho
    shift = np.exp(2j * np.pi * nu * r0)  # exp(-2 pi i u.c), c = -r0 n
    A_out, dA_out = _disk_amp(ro, rho), _disk_amp_deriv(ro, rho)
    A_in, dA_in = _disk_amp(ri, rho), _disk_amp_deriv(ri, rho)
    S_out = s0 * A_out + g * (1j / (2 * np.pi)) * dA_out * cos_nu
    S_in = (s0 * A_in + g * ((1j / (2 * np.pi)) * dA_in * cos_nu - r0 * A_in)) * shift
    F = K * A_in * shift
    flux = np.pi * ro**2 * s0 - np.pi * ri**2 * (s0 - g * r0) + K * np.pi * ri**2
    sd = _col(sigma) * MUAS
    return (S_out - S_in + F) / flux * np.exp(-2 * np.pi**2 * sd**2 * rho**2)


def xsring_flux_terms(r_out, psi, tau, lnbeta, gamma):
    """Image-plane flux of the xs-ring before normalisation (h2 = 1), sizes in muas."""
    ro, ps = r_out, psi
    ri = ro * (1 - ps)
    r0 = (1 - tau) * ps * ro
    beta = np.exp(lnbeta)
    s0, g = 0.5 * (beta + 1), (1 - beta) / (2 * ro)
    return np.pi * ro**2 * s0 - np.pi * ri**2 * (s0 - g * r0) + gamma * beta * np.pi * ri**2


def xsring_nuis2(u, v, r_out, psi, tau, pa_deg, lnbeta, gamma, sigma,
                 x0, dx, y0, dy, sg1, sg2, f1, f2):
    """X1: xs-ring plus two circular nuisance Gaussians (positions ordered in x as in Paper VI B.2)."""
    f1_, f2_ = _col(f1), _col(f2)
    V = xsring(u, v, r_out, psi, tau, pa_deg, lnbeta, gamma, sigma)
    G1 = gauss_sd(u, v, sg1, x0, y0)
    G2 = gauss_sd(u, v, sg2, np.asarray(x0, float) + np.asarray(dx, float),
                  np.asarray(y0, float) + np.asarray(dy, float))
    return (V + f1_ * G1 + f2_ * G2) / (1 + f1_ + f2_)


# Registry used by the fitting code: name -> (function, parameter names, prior bounds)
MODELS = {
    "M1": (thin_ring, ["d"], [(20, 80)]),
    "M2": (blurred_ring, ["d", "w"], [(20, 80), (0, 40)]),
    "M3": (mring1, ["d", "w", "beta", "beta_pa"], [(20, 80), (0, 40), (0, 0.5), (0, 360)]),
    "D": (disk, ["D", "w"], [(20, 100), (0, 40)]),
    "G": (gaussian, ["fwhm"], [(5, 100)]),
    # stage 5 (PREREGISTRATION 10.4, 10.5)
    "E1": (mring1_gauss, ["d", "w", "beta", "beta_pa", "fg", "xg", "yg", "fwhmg"],
           [(20, 80), (0, 40), (0, 0.5), (0, 360), (0, 2), (-100, 100), (-100, 100), (5, 200)]),
    "E2": (mring2, ["d", "w", "beta", "beta_pa", "beta2", "beta2_arg"],
           [(20, 80), (0, 40), (0, 0.5), (0, 360), (0, 0.5), (0, 360)]),
    "X0": (xsring, ["r_out", "psi", "tau", "pa", "lnbeta", "gamma", "sigma"],
           [(5, 100), (0, 1), (0, 1), (0, 360), (-5, 5), (0, 1), (0, 100)]),
    "X1": (xsring_nuis2, ["r_out", "psi", "tau", "pa", "lnbeta", "gamma", "sigma",
                          "x0", "dx", "y0", "dy", "sg1", "sg2", "f1", "f2"],
           [(5, 100), (0, 1), (0, 1), (0, 360), (-5, 5), (0, 1), (0, 100),
            (-100, 100), (0, 100), (-100, 100), (-100, 100), (0, 100), (0, 100), (0, 2), (0, 2)]),
}

# Extra prior constraints: model -> function(dict of parameter arrays) -> bool array
CONSTRAINTS = {
    "E2": lambda p: p["beta"] + p["beta2"] <= 0.5,  # non-negative brightness
}


# ---------------------------------------------------------------------------
# Stage 6 models (PREREGISTRATION section 11)
# ---------------------------------------------------------------------------

FWHM_PER_SD = 2.0 * np.sqrt(2.0 * np.log(2.0))


def egauss(u, v, x0, y0, fwhm_maj, q, th_deg):
    """Unit-flux elliptical Gaussian centred at (x0 east, y0 north) muas.

    fwhm_maj: FWHM along the major axis (muas); q = minor/major axis ratio;
    th_deg: position angle of the major axis, east of north.
    """
    th = np.deg2rad(_col(th_deg))
    sa = _col(fwhm_maj) / FWHM_PER_SD * MUAS
    sb = sa * _col(q)
    up = u * np.sin(th) + v * np.cos(th)  # along the major axis
    uq = u * np.cos(th) - v * np.sin(th)
    amp = np.exp(-2 * np.pi**2 * (sa**2 * up**2 + sb**2 * uq**2))
    return amp * np.exp(-2j * np.pi * (u * _col(x0) + v * _col(y0)) * MUAS)


def _with_egauss(V, u, v, gpars):
    """(V_ring + sum f_i G_i) / (1 + sum f_i); gpars = [(f, x, y, a, q, th), ...]."""
    tot = 1.0
    for f, x, y, a, q, th in gpars:
        V = V + _col(f) * egauss(u, v, x, y, a, q, th)
        tot = tot + _col(f)
    return V / tot


class _RingGauss:
    """Ring model plus n elliptical Gaussians (a picklable class, for multiprocessing)."""

    def __init__(self, ring_fn, n_ring_args, n_gauss):
        self.ring_fn, self.n_ring, self.n_gauss = ring_fn, n_ring_args, n_gauss
        self.__name__ = f"{ring_fn.__name__}_eg{n_gauss}"

    def __call__(self, u, v, *p):
        k = self.n_ring
        ring = self.ring_fn(u, v, *p[:k])
        g = [p[k + 6 * i: k + 6 * (i + 1)] for i in range(self.n_gauss)]
        return _with_egauss(ring, u, v, g)


def _make_ring_gauss(ring_fn, n_ring_args, n_gauss):
    return _RingGauss(ring_fn, n_ring_args, n_gauss)


def _gauss_names(n):
    return [f"{k}{i}" for i in range(1, n + 1) for k in ("f", "x", "y", "a", "q", "th")]


# Paper VI style nuisance: flux ratio U(0, 2), centre U(-100, 100) muas, major FWHM U(5, 200) muas,
# axis ratio U(0.05, 1), major-axis PA U(0, 180) deg. Positions are ordered in x (PREREGISTRATION 11.2).
_GAUSS_BOUNDS = [(0, 2), (-100, 100), (-100, 100), (5, 200), (0.05, 1), (0, 180)]
_M3_NAMES, _M3_BOUNDS = ["d", "w", "beta", "beta_pa"], [(20, 80), (0, 40), (0, 0.5), (0, 360)]
_M2_NAMES = _M3_NAMES + ["beta2", "beta2_arg"]
_M2_BOUNDS = _M3_BOUNDS + [(0, 0.5), (0, 360)]
_XS_NAMES = ["r_out", "psi", "tau", "pa", "lnbeta", "gamma", "sigma"]
_XS_BOUNDS = [(5, 100), (0, 1), (0, 1), (0, 360), (-5, 5), (0, 1), (0, 100)]


def xsring_largegauss(u, v, r_out, psi, tau, pa_deg, lnbeta, gamma, sigma, *rest):
    """xs-ring + n elliptical Gaussians + one centred circular large Gaussian (last two parameters:
    flux ratio fL and FWHM aL in muas). Used only with intra-site baselines (11.5)."""
    g = rest[:-2]
    fL, aL = rest[-2], rest[-1]
    V = xsring(u, v, r_out, psi, tau, pa_deg, lnbeta, gamma, sigma)
    n = len(g) // 6
    tot = 1.0
    for i in range(n):
        f, x, y, a, q, th = g[6 * i: 6 * (i + 1)]
        V = V + _col(f) * egauss(u, v, x, y, a, q, th)
        tot = tot + _col(f)
    V = V + _col(fL) * blur(u, v, aL)
    return V / (tot + _col(fL))


for _n in (1, 2, 3):
    MODELS[f"G{_n}"] = (_make_ring_gauss(mring1, 4, _n), _M3_NAMES + _gauss_names(_n),
                        _M3_BOUNDS + _GAUSS_BOUNDS * _n)
    MODELS[f"R2G{_n}"] = (_make_ring_gauss(mring2, 6, _n), _M2_NAMES + _gauss_names(_n),
                          _M2_BOUNDS + _GAUSS_BOUNDS * _n)
    MODELS[f"XE{_n}"] = (_make_ring_gauss(xsring, 7, _n), _XS_NAMES + _gauss_names(_n),
                         _XS_BOUNDS + _GAUSS_BOUNDS * _n)
MODELS["XE3L"] = (xsring_largegauss, _XS_NAMES + _gauss_names(3) + ["fL", "aL"],
                  _XS_BOUNDS + _GAUSS_BOUNDS * 3 + [(0, 3), (500, 5000)])


def _x_ordered(p):
    xs = sorted(k for k in p if k[0] == "x" and k[1:].isdigit())
    ok = np.ones(np.shape(p[xs[0]]), bool) if xs else True
    for a, b in zip(xs[:-1], xs[1:]):
        ok = ok & (p[a] <= p[b])
    return ok


for _n in (2, 3):
    CONSTRAINTS[f"G{_n}"] = _x_ordered
    CONSTRAINTS[f"XE{_n}"] = _x_ordered
CONSTRAINTS["XE3L"] = _x_ordered
CONSTRAINTS["R2G1"] = CONSTRAINTS["E2"]
for _n in (2, 3):
    CONSTRAINTS[f"R2G{_n}"] = (lambda p: CONSTRAINTS["E2"](p) & _x_ordered(p))
