"""Reading EHT uvfits files, scan averaging, systematic error and debiasing.

Processing order follows docs/PREREGISTRATION.md section 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import INTRA_SITE


@dataclass
class VisTable:
    """Flat table of Stokes I visibilities.

    time: hours UTC (from the observation date), u, v: wavelengths,
    vis: complex Jy, sigma: Jy, t1/t2: station codes, scan: scan index.
    """

    time: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    u: np.ndarray
    v: np.ndarray
    vis: np.ndarray
    sigma: np.ndarray
    scan: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.time)

    def select(self, mask: np.ndarray) -> "VisTable":
        return VisTable(
            self.time[mask], self.t1[mask], self.t2[mask], self.u[mask], self.v[mask],
            self.vis[mask], self.sigma[mask],
            None if self.scan is None else self.scan[mask], dict(self.meta),
        )

    @property
    def rho(self) -> np.ndarray:
        return np.hypot(self.u, self.v)

    @property
    def snr(self) -> np.ndarray:
        return np.abs(self.vis) / self.sigma

    def is_intrasite(self) -> np.ndarray:
        return np.array([frozenset((a, b)) in INTRA_SITE for a, b in zip(self.t1, self.t2)])


def read_uvfits(path: str | Path, conjugate: bool = True) -> VisTable:
    """Read a Stokes-I EHT 2019-D01-01 uvfits file.

    RR and LL both hold the Stokes I estimate; their errors are sqrt(2) times the
    Stokes I error (release README). So I = RR and sigma_I = sigma_RR / sqrt(2).

    The files were written by eht-imaging, whose Fourier convention is
    V = sum I exp(+2 pi i (u x + v y)) (ehtim.observing.obs_helpers.ftmatrix),
    and ehtim reads them back without conjugation. Our models use the
    exp(-2 pi i ...) convention (models.py), so by default the visibilities are
    conjugated here. This flips closure-phase signs and rotates position angles
    by 180 deg; it does not change amplitudes or ring diameters.
    """
    with fits.open(path) as h:
        g = h[0]
        freq = float(g.header["CRVAL4"])
        names = {int(r["NOSTA"]): r["ANNAME"].strip() for r in h["AIPS AN"].data}
        d = g.data
        u = np.asarray(d.par("UU---SIN"), float) * freq
        v = np.asarray(d.par("VV---SIN"), float) * freq
        bl = np.asarray(d.par("BASELINE"), float)
        jd = np.asarray(d.par("DATE"), float)  # astropy sums the two DATE params
        raw = np.asarray(d.data, float)[:, 0, 0, 0, 0, :, :]  # (N, stokes, 3)
        date_obs = g.header["DATE-OBS"]
    rr = raw[:, 0, 0] + 1j * raw[:, 0, 1]
    ll = raw[:, 1, 0] + 1j * raw[:, 1, 1]
    w_rr, w_ll = raw[:, 0, 2], raw[:, 1, 2]
    a1 = (bl // 256).astype(int)
    a2 = np.rint(bl - 256 * a1).astype(int)
    t1 = np.array([names[a] for a in a1])
    t2 = np.array([names[a] for a in a2])
    jd0 = np.floor(jd.min() - 0.5) + 0.5
    time = (jd - jd0) * 24.0
    ok = np.isfinite(rr) & np.isfinite(w_rr) & (w_rr > 0)
    sig = np.full(len(rr), np.nan)
    sig[ok] = 1.0 / np.sqrt(w_rr[ok]) / np.sqrt(2.0)
    meta = {
        "path": str(path), "freq_hz": freq, "date_obs": date_obs,
        "rr_eq_ll": bool(np.allclose(rr[ok], ll[ok]) and np.allclose(w_rr[ok], w_ll[ok])),
        "n_raw": int(len(rr)), "n_dropped": int((~ok).sum()),
    }
    vis = np.conj(rr[ok]) if conjugate else rr[ok]
    meta["conjugated"] = bool(conjugate)
    tab = VisTable(time[ok], t1[ok], t2[ok], u[ok], v[ok], vis, sig[ok], meta=meta)
    return tab


def assign_scans(time: np.ndarray, gap_sec: float = 60.0) -> np.ndarray:
    """Scan index from gaps >= gap_sec in the sorted unique timestamps."""
    ut = np.unique(time)
    breaks = np.diff(ut) * 3600.0 >= gap_sec
    sid = np.concatenate([[0], np.cumsum(breaks)])
    return sid[np.searchsorted(ut, time)]


def scan_average(tab: VisTable, gap_sec: float = 60.0) -> VisTable:
    """Coherent 1/sigma^2-weighted vector average per (scan, baseline)."""
    scan = assign_scans(tab.time, gap_sec)
    key = np.array([f"{s}|{a}|{b}" for s, a, b in zip(scan, tab.t1, tab.t2)])
    uk, inv = np.unique(key, return_inverse=True)
    w = 1.0 / tab.sigma**2
    sw = np.bincount(inv, w)
    avg = lambda x: np.bincount(inv, w * x) / sw  # noqa: E731
    vis = avg(tab.vis.real) + 1j * avg(tab.vis.imag)
    first = np.zeros(len(uk), int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    out = VisTable(
        avg(tab.time), tab.t1[first], tab.t2[first], avg(tab.u), avg(tab.v), vis,
        1.0 / np.sqrt(sw), scan[first], dict(tab.meta),
    )
    out.meta["n_scans"] = int(scan.max() + 1)
    order = np.lexsort((out.t2, out.t1, out.time))
    return out.select(order)


def add_systematic(tab: VisTable, frac: float = 0.01) -> VisTable:
    out = tab.select(np.ones(len(tab), bool))
    out.sigma = np.sqrt(tab.sigma**2 + (frac * np.abs(tab.vis)) ** 2)
    return out


def debiased_amp(tab: VisTable) -> np.ndarray:
    """sqrt(|V|^2 - sigma^2), set to 0 where |V| <= sigma."""
    a2 = np.abs(tab.vis) ** 2 - tab.sigma**2
    return np.sqrt(np.clip(a2, 0.0, None))


def prepare(tab: VisTable, sys_frac: float = 0.01, snr_min: float = 2.0,
            drop_intrasite: bool = True, drop_stations: tuple[str, ...] = ()) -> VisTable:
    """Pre-registered steps 2-5: scan average, 1% systematic, SNR cut, intra-site cut.

    The returned table has `vis` replaced by debiased amplitude times the original phase.
    """
    s = add_systematic(scan_average(tab), sys_frac)
    keep = s.snr >= snr_min  # SNR from the measured |V| (before debiasing)
    amp = debiased_amp(s)
    s.vis = amp * np.exp(1j * np.angle(s.vis))
    if drop_intrasite:
        keep &= ~s.is_intrasite()
    for st in drop_stations:
        keep &= (s.t1 != st) & (s.t2 != st)
    return s.select(keep)
