"""Loading the eight EHT data sets and building closure quantities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import DAYS
from .closure import ClosureSet, closure_phases, log_closure_amplitudes
from .uvdata import VisTable, prepare, read_uvfits

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "uvfits"
RAW2018 = ROOT / "data" / "raw" / "uvfits2018"
DAYS2018 = {"111": "2018-04-21", "112": "2018-04-22", "115": "2018-04-25"}
BANDS2018 = ("b1", "b2", "b3", "b4")


def uvfits_path(doy: str, band: str) -> Path:
    return RAW / f"SR1_M87_2017_{doy}_{band}_hops_netcal_StokesI.uvfits"


@dataclass
class Dataset:
    name: str
    doy: str
    band: str
    raw: VisTable
    tab: VisTable
    cp: ClosureSet
    lca: ClosureSet

    @property
    def date(self) -> str:
        return DAYS.get(self.doy) or DAYS2018[self.doy]

    @property
    def n_data(self) -> int:
        return len(self.cp) + len(self.lca)


def build(raw: VisTable, name: str = "", doy: str = "", band: str = "", drop_colocated: bool = False,
          **prep_kw) -> Dataset:
    tab = prepare(raw, **prep_kw)
    return Dataset(name, doy, band, raw, tab, closure_phases(tab),
                   log_closure_amplitudes(tab, drop_colocated=drop_colocated))


def load(doy: str, band: str, **prep_kw) -> Dataset:
    raw = _read_cached(uvfits_path(doy, band))
    return build(raw, f"{DAYS[doy]} {band}", doy, band, **prep_kw)


_RAW_CACHE: dict = {}


def _read_cached(path: Path) -> VisTable:
    """Read each uvfits file once per process (stage 6 builds many subsets from the same files)."""
    if path not in _RAW_CACHE:
        _RAW_CACHE[path] = read_uvfits(path)
    return _RAW_CACHE[path]


def load_all(**prep_kw) -> list[Dataset]:
    return [load(doy, band, **prep_kw) for doy in DAYS for band in ("lo", "hi")]


def uvfits_path_2018(doy: str, band: str) -> Path:
    return RAW2018 / f"L2V1_M87_2018_{doy}_{band}_hops_netcal_10s_StokesI.uvfits"


def load_2018(doy: str, band: str, **prep_kw) -> Dataset:
    raw = _read_cached(uvfits_path_2018(doy, band))
    return build(raw, f"{DAYS2018[doy]} {band}", doy, band, **prep_kw)


def data2018_available(days=("111", "115")) -> bool:
    return all(uvfits_path_2018(d, b).exists() for d in days for b in BANDS2018)


def data_available() -> bool:
    return all(uvfits_path(d, b).exists() for d in DAYS for b in ("lo", "hi"))


def stack_info(ds: list[Dataset]) -> list[dict]:
    out = []
    for d in ds:
        out.append({
            "dataset": d.name, "n_10s": len(d.raw), "n_scans": d.tab.meta.get("n_scans"),
            "n_scanavg_used": len(d.tab), "n_cphase": len(d.cp), "n_logcamp": len(d.lca),
            "stations": sorted(set(d.tab.t1) | set(d.tab.t2)),
            "rho_max_glambda": float(np.max(d.tab.rho) / 1e9),
        })
    return out
