"""Mass from ring diameter: theta_g = d / alpha, M = theta_g D c^2 / G (PREREGISTRATION 7.3)."""
from __future__ import annotations

import numpy as np

from . import MUAS

G = 6.67430e-11
C = 299792458.0
MSUN = 1.98840987e30
MPC = 3.0856775814913673e22


def mass_from_theta_g(theta_g_muas, dist_mpc):
    """Mass in units of 1e9 Msun."""
    return np.asarray(theta_g_muas) * MUAS * np.asarray(dist_mpc) * MPC * C**2 / G / MSUN / 1e9


def mass_mc(d, d_err, alpha, alpha_err, dist=16.8, dist_err=0.8, n=1_000_000, seed=2019):
    rng = np.random.default_rng(seed)
    ds = rng.normal(d, d_err, n)
    al = rng.normal(alpha, alpha_err, n)
    D = rng.normal(dist, dist_err, n)
    ok = (al > 0) & (D > 0)
    th = ds[ok] / al[ok]
    M = mass_from_theta_g(th, D[ok])
    q = lambda x: np.percentile(x, [16, 50, 84])  # noqa: E731
    t16, t50, t84 = q(th)
    m16, m50, m84 = q(M)
    return {"theta_g": (float(t50), float(t50 - t16), float(t84 - t50)),
            "mass_1e9": (float(m50), float(m50 - m16), float(m84 - m50)),
            "inputs": {"d": d, "d_err": d_err, "alpha": alpha, "alpha_err": alpha_err,
                       "dist": dist, "dist_err": dist_err, "n": n, "seed": seed}}


def compare(m, lit):
    """m, lit: (value, minus, plus). Uses the error on the side facing the other value."""
    v, mm, mp = m
    lv, lm, lp = lit
    if v >= lv:
        s = np.hypot(mm, lp)
    else:
        s = np.hypot(mp, lm)
    z = (v - lv) / s
    verdict = "일치" if abs(z) <= 1 else ("약한 차이" if abs(z) <= 2 else "불일치")
    return {"diff": float(v - lv), "sigma": float(s), "z": float(z), "verdict": verdict}
