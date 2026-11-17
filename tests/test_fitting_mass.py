"""Unit tests: fitting recovers a noiseless truth, null fit, mass conversion."""
import numpy as np

from bhshadow.closure import closure_phases, log_closure_amplitudes
from bhshadow.dataset import Dataset
from bhshadow.fitting import Problem, circ_stats, optimise, run_mcmc
from bhshadow.mass import compare, mass_from_theta_g, mass_mc
from bhshadow.nullfit import null_fit

from test_closure import toy_table


def toy_dataset():
    tab = toy_table(n_times=6, sigma=2e-3)
    return Dataset("toy", "000", "hi", tab, tab, closure_phases(tab), log_closure_amplitudes(tab))


def test_optimiser_recovers_truth():
    x, chi2 = optimise(Problem("M3", [toy_dataset()]), seed=1)
    assert abs(x[0] - 42.0) < 0.5 and chi2 < 1e-3


def test_mcmc_runs_and_brackets_truth():
    r = run_mcmc(Problem("M3", [toy_dataset()]), seed=2, nburn=200, nprod=400, retry=False)
    s = r["summary"]["d"]
    assert s["median"] - 5 * max(s["minus"], 1e-3) < 42.0 < s["median"] + 5 * max(s["plus"], 1e-3)
    assert set(r) >= {"chi2_nu", "bic", "tau", "converged"}


def test_joint_problem_parameter_layout():
    d = toy_dataset()
    p = Problem("M3", [d, d, d, d], groups=[0, 0, 1, 1], shared=("d", "w"))
    assert p.names == ["d", "w", "beta_0", "beta_1", "beta_pa_0", "beta_pa_1"]
    assert p.periodic.tolist() == [False, False, False, False, True, True]


def test_circular_stats_wraps():
    m, lo, hi = circ_stats(np.array([355.0, 5.0, 0.0, 359.0, 1.0]))
    assert min(m, 360 - m) < 1.0


def test_null_fit_finds_minimum():
    rng = np.random.default_rng(0)
    rho = rng.uniform(2.5e9, 4.5e9, 300)
    amp = np.sqrt((0.1 * (rho / 1e9 - 3.5)) ** 2 + 0.01**2) + rng.normal(0, 0.002, 300)
    r = null_fit(rho, amp, np.full(300, 0.002), np.arange(300) // 5, n_boot=100)
    assert abs(r["b1_glambda"] - 3.5) < 0.02
    assert abs(r["d_null_muas"] - 2.404825557695773 / (np.pi * 3.5e9) / 4.84813681109536e-12) < 0.5


def test_mass_conversion_matches_paper_vi():
    # Paper VI Table 7: theta_g = 3.8 muas at 16.8 Mpc corresponds to ~6.5e9 Msun
    assert abs(mass_from_theta_g(3.8, 16.8) - 6.5) < 0.1
    r = mass_mc(42.0, 0.0, 11.56, 0.0, 16.8, 0.0, n=1000)
    assert abs(r["mass_1e9"][0] - mass_from_theta_g(42 / 11.56, 16.8)) < 1e-9


def test_compare_verdicts():
    assert compare((6.5, 0.7, 0.7), (6.5, 0.7, 0.7))["verdict"] == "일치"
    assert compare((6.5, 0.5, 0.5), (3.3, 0.66, 0.84))["verdict"] == "불일치"


def test_periodic_angle_stays_bounded_when_unconstrained():
    # symmetric truth: beta -> 0, the angle is unconstrained; walkers must not drift to overflow
    from bhshadow.models import mring1
    d = toy_dataset()
    d.tab.vis = mring1(d.tab.u, d.tab.v, 42.0, 12.0, 0.0, 0.0)
    d.cp, d.lca = closure_phases(d.tab), log_closure_amplitudes(d.tab)
    p = Problem("M3", [d])
    r = run_mcmc(p, seed=5, nburn=300, nprod=600, retry=False)
    pa = r["_chain"][:, p.names.index("beta_pa")]
    assert np.all(np.isfinite(pa)) and pa.min() >= -360 and pa.max() <= 720
    assert not p.in_prior(np.array([[42, 10, 0.1, 1e6]]))[0]
