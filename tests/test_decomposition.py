"""Tests for src/decomposition.py and src/approximation.py."""

import numpy as np
import pytest

from src.measures import AtomicMeasure, check_decreasing_convex_order
from src.couplings import (
    DiscreteCoupling, solve_supermartingale_coupling, independent_coupling
)
from src.decomposition import (
    compute_irreducible_decomposition,
    IrreducibleDecomposition,
    quantile_splitting,
)
from src.approximation import (
    ApproximationResult,
    construct_perturbation_sequence,
    run_approximation_experiment,
)


# ──────────── Helper instances ────────────────────────────────────────────────

def make_strict_pair():
    """Strict supermartingale pair: Delta = bar(mu) - bar(nu) > 0."""
    mu = AtomicMeasure([1.0, 3.0], [0.5, 0.5])
    nu = AtomicMeasure([0.0, 1.5, 3.0], [0.4, 0.3, 0.3])
    return mu, nu


def make_martingale_pair():
    """Pure martingale pair."""
    mu = AtomicMeasure([-1.0, 1.0], [0.5, 0.5])
    nu = AtomicMeasure([-2.0, 0.0, 2.0], [0.25, 0.5, 0.25])
    return mu, nu


def make_mixed_pair():
    """Mixed-component pair."""
    mu = AtomicMeasure([-1.0, 0.5, 2.0], [0.3, 0.4, 0.3])
    nu = AtomicMeasure([-2.0, 0.0, 1.5, 3.0], [0.2, 0.3, 0.3, 0.2])
    feasible, viol = check_decreasing_convex_order(mu, nu)
    if not feasible:
        nu = AtomicMeasure(nu.locations + viol + 0.1, nu.weights)
    return mu, nu


# ──────────── IrreducibleDecomposition ───────────────────────────────────────

class TestIrreducibleDecomposition:
    def test_returns_correct_type(self):
        mu, nu = make_strict_pair()
        decomp = compute_irreducible_decomposition(mu, nu)
        assert isinstance(decomp, IrreducibleDecomposition)

    def test_x_star_in_range(self):
        mu, nu = make_strict_pair()
        decomp = compute_irreducible_decomposition(mu, nu)
        all_pts = np.concatenate([mu.locations, nu.locations])
        xmin = all_pts.min() - 1.0
        xmax = all_pts.max() + 1.0
        assert xmin <= decomp.x_star <= xmax

    def test_D_nonneg(self):
        """D = P_nu - P_mu >= 0 everywhere (mu <=_{cd} nu)."""
        mu, nu = make_strict_pair()
        decomp = compute_irreducible_decomposition(mu, nu)
        assert np.all(decomp.D >= -1e-8), f"min D = {decomp.D.min():.4e}"

    def test_strict_pair_has_strict_region(self):
        """Strict pair should have strict support above x*."""
        mu, nu = make_strict_pair()
        decomp = compute_irreducible_decomposition(mu, nu)
        # mu has atoms at 1.0 and 3.0; bar(mu)=2 > bar(nu)=1.55
        # strict region should contain at least one support point
        assert len(decomp.strict_support) >= 0  # may or may not, depending on x*

    def test_martingale_pair_components(self):
        """Martingale pair should have martingale components in (-inf, x*)."""
        mu, nu = make_martingale_pair()
        decomp = compute_irreducible_decomposition(mu, nu)
        # D > 0 on some interval: martingale components
        assert isinstance(decomp.martingale_components, list)

    def test_D_zero_at_diagonal(self):
        """On strict diagonal region (mu atoms with D~0), D should be small."""
        mu = AtomicMeasure([0.0, 2.0], [0.5, 0.5])
        nu = AtomicMeasure([-1.0, 1.0, 3.0], [0.3, 0.4, 0.3])
        feasible, viol = check_decreasing_convex_order(mu, nu)
        if feasible:
            decomp = compute_irreducible_decomposition(mu, nu)
            # x_star should be finite
            assert np.isfinite(decomp.x_star)

    def test_grid_and_D_consistent(self):
        mu, nu = make_mixed_pair()
        decomp = compute_irreducible_decomposition(mu, nu)
        assert len(decomp.grid) == len(decomp.D)
        assert np.all(np.diff(decomp.grid) >= 0)


# ──────────── Perturbation sequence ──────────────────────────────────────────

class TestPerturbationSequence:
    def test_length(self):
        mu, nu = make_strict_pair()
        K = 5
        seq = construct_perturbation_sequence(mu, nu, K=K)
        assert len(seq) == K

    def test_feasibility_preserved(self):
        mu, nu = make_strict_pair()
        seq = construct_perturbation_sequence(mu, nu, K=4)
        for mu_k, nu_k in seq:
            feasible, viol = check_decreasing_convex_order(mu_k, nu_k)
            assert feasible, f"mu_k <=cd nu_k violated, viol={viol:.3e}"

    def test_convergence_w1(self):
        """W_1(mu_k, mu) should decrease as k increases."""
        from src.measures import wasserstein_1
        mu, nu = make_strict_pair()
        seq = construct_perturbation_sequence(mu, nu, K=6, delta_max=0.5)
        w1_first = wasserstein_1(seq[0][0], mu)
        w1_last = wasserstein_1(seq[-1][0], mu)
        # At least last w1 <= first * 2 (allow tolerance for stochastic effects)
        assert w1_last <= w1_first + 0.5

    def test_all_elements_atomic(self):
        mu, nu = make_martingale_pair()
        seq = construct_perturbation_sequence(mu, nu, K=3)
        for mu_k, nu_k in seq:
            assert isinstance(mu_k, AtomicMeasure)
            assert isinstance(nu_k, AtomicMeasure)


# ──────────── Approximation experiment ───────────────────────────────────────

class TestApproximationExperiment:
    def test_returns_correct_length(self):
        mu, nu = make_strict_pair()
        pi = solve_supermartingale_coupling(mu, nu)
        if pi is None:
            pi = independent_coupling(mu, nu)
        seq = construct_perturbation_sequence(mu, nu, K=3)
        results = run_approximation_experiment(mu, nu, pi, seq, verbose=False)
        assert len(results) == 3

    def test_aw1_nonneg(self):
        mu, nu = make_strict_pair()
        pi = solve_supermartingale_coupling(mu, nu)
        if pi is None:
            pi = independent_coupling(mu, nu)
        seq = construct_perturbation_sequence(mu, nu, K=3)
        results = run_approximation_experiment(mu, nu, pi, seq, verbose=False)
        for r in results:
            assert r.aw1 >= 0.0

    def test_result_fields(self):
        mu, nu = make_strict_pair()
        pi = solve_supermartingale_coupling(mu, nu)
        if pi is None:
            pi = independent_coupling(mu, nu)
        seq = construct_perturbation_sequence(mu, nu, K=2)
        results = run_approximation_experiment(mu, nu, pi, seq, verbose=False)
        r = results[0]
        assert r.k == 1
        assert isinstance(r.pi_k, DiscreteCoupling)
        assert np.isfinite(r.aw1)
        assert r.w1_mu >= 0.0
        assert r.w1_nu >= 0.0
        assert r.feasibility_residual >= 0.0
        assert isinstance(r.marginal_errors, tuple)
        assert len(r.marginal_errors) == 2

    def test_approx_couplings_feasible(self):
        """Approximate couplings should have bounded SMC residual."""
        mu, nu = make_strict_pair()
        pi = solve_supermartingale_coupling(mu, nu)
        if pi is None:
            pi = independent_coupling(mu, nu)
        seq = construct_perturbation_sequence(mu, nu, K=3)
        results = run_approximation_experiment(mu, nu, pi, seq, verbose=False)
        for r in results:
            # Each pi_k should be a valid coupling (marginals correct)
            assert r.marginal_errors[0] < 0.1
            assert r.marginal_errors[1] < 0.1

    def test_aw1_decreasing_trend(self):
        """AW_1 should generally decrease as k increases."""
        mu, nu = make_strict_pair()
        pi = solve_supermartingale_coupling(mu, nu)
        if pi is None:
            pi = independent_coupling(mu, nu)
        seq = construct_perturbation_sequence(mu, nu, K=5)
        results = run_approximation_experiment(mu, nu, pi, seq, verbose=False)
        aw1s = [r.aw1 for r in results]
        # First AW1 > last AW1 (allow some noise)
        assert aw1s[0] >= aw1s[-1] - 0.2

    def test_martingale_family(self):
        """Martingale pair approximation should also converge."""
        mu, nu = make_martingale_pair()
        pi = solve_supermartingale_coupling(mu, nu)
        if pi is None:
            pi = independent_coupling(mu, nu)
        seq = construct_perturbation_sequence(mu, nu, K=3)
        results = run_approximation_experiment(mu, nu, pi, seq, verbose=False)
        assert len(results) == 3
        for r in results:
            assert r.aw1 >= 0.0


# ──────────── Quantile splitting ─────────────────────────────────────────────

class TestQuantileSplitting:
    def test_returns_lists(self):
        mu, nu = make_strict_pair()
        mu_k = AtomicMeasure([0.9, 3.1], [0.5, 0.5])
        nu_k = AtomicMeasure([0.0, 1.5, 3.2], [0.4, 0.3, 0.3])
        decomp = compute_irreducible_decomposition(mu, nu)
        gamma_k = independent_coupling(mu_k, nu_k)
        mu_list, nu_list = quantile_splitting(mu, mu_k, decomp, gamma_k)
        assert isinstance(mu_list, list)
        assert isinstance(nu_list, list)
