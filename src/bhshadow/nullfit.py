"""First-null location of the visibility amplitude (PREREGISTRATION 7.1).

Fit A(rho) = sqrt(k^2 (rho - b1)^2 + m^2) to scan-averaged debiased amplitudes
of non-intra-site baselines with 2.5 <= rho <= 4.5 Glambda by weighted least
squares. Uncertainty from a scan-level bootstrap.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

J0_ZERO = 2.404825557695773


def _fit(rho_g, amp, sig, b0=3.4):
    def res(p):
        k, b1, m = p
        return (np.sqrt((k * (rho_g - b1)) ** 2 + m**2) - amp) / sig

    k0 = max(np.ptp(amp) / max(np.ptp(rho_g), 1e-3), 1e-3)
    sol = least_squares(res, [k0, b0, max(amp.min(), 1e-3)],
                        bounds=([0, 2.5, 0], [np.inf, 4.5, np.inf]))
    return sol.x, float((sol.fun**2).sum())


def null_fit(rho, amp, sig, scan_id, lo=2.5e9, hi=4.5e9, n_boot=1000, seed=1):
    """rho in wavelengths. Returns dict with b1 (Glambda), bootstrap sigma, d_null (muas)."""
    m = (rho >= lo) & (rho <= hi)
    r, a, s, sc = rho[m] / 1e9, amp[m], sig[m], np.asarray(scan_id)[m]
    p, chi2 = _fit(r, a, s)
    rng = np.random.default_rng(seed)
    scans = np.unique(sc)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(scans, len(scans), replace=True)
        idx = np.concatenate([np.flatnonzero(sc == q) for q in pick])
        if len(idx) < 4:
            continue
        boots.append(_fit(r[idx], a[idx], s[idx], b0=p[1])[0][1])
    boots = np.array(boots)
    b1 = float(p[1])
    d = lambda b: J0_ZERO / (np.pi * b * 1e9) / 4.84813681109536e-12  # noqa: E731
    lo16, hi84 = np.percentile(boots, [16, 84])
    return {
        "b1_glambda": b1, "b1_boot_std": float(boots.std()),
        "b1_boot_16": float(lo16), "b1_boot_84": float(hi84),
        "k": float(p[0]), "m": float(p[2]), "chi2": chi2, "n_points": int(m.sum()),
        "n_scans": int(len(scans)),
        "d_null_muas": float(d(b1)), "d_null_boot_std": float(np.std(d(boots))),
    }
