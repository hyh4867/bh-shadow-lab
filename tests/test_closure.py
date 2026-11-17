"""Unit tests: closure quantities are gain-free, minimal sets have the right size."""
import numpy as np

from bhshadow.closure import closure_phases, log_closure_amplitudes, model_closure_phase, model_log_camp
from bhshadow.models import mring1
from bhshadow.uvdata import VisTable, assign_scans, scan_average

ST = ["AA", "AZ", "JC", "LM", "PV"]
POS = {"AA": (2.2, -5.4), "AZ": (-1.8, -5.0), "JC": (-5.5, -2.5), "LM": (-0.8, -6.0), "PV": (5.1, -0.3)}


def toy_table(n_times=3, gains=None, sigma=1e-3):
    rows = []
    for k in range(n_times):
        for i, a in enumerate(ST):
            for b in ST[i + 1:]:
                u = (POS[a][0] - POS[b][0]) * 0.6e9 * (1 + 0.05 * k)
                v = (POS[a][1] - POS[b][1]) * 0.6e9 * (1 - 0.03 * k)
                rows.append((k * 10.0 / 3600, a, b, u, v))
    t = np.array([r[0] for r in rows]); t1 = np.array([r[1] for r in rows]); t2 = np.array([r[2] for r in rows])
    u = np.array([r[3] for r in rows]); v = np.array([r[4] for r in rows])
    vis = mring1(u, v, 42.0, 12.0, 0.25, 200.0)
    sig = np.full(len(t), sigma)
    if gains is not None:
        g = np.array([gains[a] * np.conj(gains[b]) for a, b in zip(t1, t2)])
        vis, sig = vis * g, sig * np.abs(g)  # noise scales with the gains, so the SNR order is unchanged
    return VisTable(t, t1, t2, u, v, vis, sig, scan=np.arange(len(t)) // 10)


def test_minimal_set_sizes():
    tab = toy_table(n_times=1)
    cp, lca = closure_phases(tab), log_closure_amplitudes(tab)
    n = len(ST)
    assert len(cp) == (n - 1) * (n - 2) // 2  # 6 independent closure phases for 5 stations
    assert len(lca) == n * (n - 3) // 2  # 5 independent log closure amplitudes


def test_closure_quantities_are_gain_invariant():
    rng = np.random.default_rng(3)
    gains = {s: rng.lognormal(0, 0.3) * np.exp(1j * rng.uniform(0, 2 * np.pi)) for s in ST}
    a, b = toy_table(), toy_table(gains=gains)
    cpa, cpb = closure_phases(a), closure_phases(b)
    la, lb = log_closure_amplitudes(a), log_closure_amplitudes(b)
    assert np.allclose(np.angle(np.exp(1j * (cpa.obs - cpb.obs))), 0, atol=1e-10)
    assert np.allclose(la.obs, lb.obs, atol=1e-10)


def test_model_closure_matches_observed_for_noiseless_truth():
    tab = toy_table()
    cp, lca = closure_phases(tab), log_closure_amplitudes(tab)
    V = mring1(tab.u, tab.v, 42.0, 12.0, 0.25, 200.0)
    assert np.allclose(np.angle(np.exp(1j * (model_closure_phase(V, cp) - cp.obs))), 0, atol=1e-10)
    assert np.allclose(model_log_camp(V, lca), lca.obs, atol=1e-10)


def test_symmetric_source_closure_phase_is_0_or_pi():
    tab = toy_table(n_times=1)
    tab.vis = mring1(tab.u, tab.v, 42.0, 12.0, 0.0, 0.0)
    cp = closure_phases(tab)
    r = np.abs(np.angle(np.exp(1j * cp.obs)))
    assert np.all((r < 1e-9) | (np.abs(r - np.pi) < 1e-9))


def test_assign_scans_and_average():
    t = np.array([0, 10, 20, 400, 410]) / 3600.0
    assert assign_scans(t).tolist() == [0, 0, 0, 1, 1]
    tab = VisTable(t, np.array(["AA"] * 5), np.array(["LM"] * 5), np.ones(5), np.ones(5),
                   np.array([1, 1, 1, 2, 2], complex), np.array([1, 1, 1, 0.5, 0.5]))
    avg = scan_average(tab)
    assert len(avg) == 2
    assert np.allclose(avg.vis, [1, 2])
    assert np.allclose(avg.sigma, [1 / np.sqrt(3), 0.5 / np.sqrt(2)])
