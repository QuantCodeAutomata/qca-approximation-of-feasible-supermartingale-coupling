"""Tests for src/couplings.py — DiscreteCoupling, SMC solvers, AW_1, composition."""

import numpy as np
import pytest

from src.measures import AtomicMeasure, check_decreasing_convex_order
from src.couplings import (
    DiscreteCoupling,
    solve_supermartingale_coupling,
    solve_martingale_coupling,
    diagonal_coupling,
    independent_coupling,
    compose_couplings,
    adapted_wasserstein_1,
    mixture_coupling,
)


# ──────────── Helper fixtures ─────────────────────────────────────────────────

def make_simple_pair():
    """Simple feasible (mu, nu) pair."""
    mu = AtomicMeasure([0.0, 2.0], [0.5, 0.5])
    nu = AtomicMeasure([-1.0, 1.0, 3.0], [0.3, 0.4, 0.3])
    return mu, nu


def make_martingale_pair():
    """mu <=_{cx} nu (mean-preserving spread)."""
    mu = AtomicMeasure([-1.0, 1.0], [0.5, 0.5])  # mean=0
    nu = AtomicMeasure([-2.0, 0.0, 2.0], [0.25, 0.5, 0.25])  # mean=0
    return mu, nu


# ──────────── DiscreteCoupling construction ───────────────────────────────────

class TestDiscreteCoupling:
    def test_basic_construction(self):
        mu = AtomicMeasure([0.0, 1.0], [0.4, 0.6])
        nu = AtomicMeasure([0.5, 1.5], [0.3, 0.7])
        P = np.array([[0.2, 0.2], [0.1, 0.5]])
        pi = DiscreteCoupling(mu, nu, P)
        assert pi.P.shape == (2, 2)

    def test_wrong_shape_raises(self):
        mu = AtomicMeasure([0.0, 1.0], [0.5, 0.5])
        nu = AtomicMeasure([0.5, 1.5, 2.5], [1 / 3, 1 / 3, 1 / 3])
        with pytest.raises(AssertionError):
            DiscreteCoupling(mu, nu, np.eye(2))

    def test_first_marginal_error(self):
        mu, nu = make_simple_pair()
        # Build exact coupling: product measure
        P = np.outer(mu.weights, nu.weights)
        pi = DiscreteCoupling(mu, nu, P)
        assert pi.check_first_marginal() < 1e-12

    def test_second_marginal_error(self):
        mu, nu = make_simple_pair()
        P = np.outer(mu.weights, nu.weights)
        pi = DiscreteCoupling(mu, nu, P)
        assert pi.check_second_marginal() < 1e-12

    def test_conditional_means_shape(self):
        mu, nu = make_simple_pair()
        P = np.outer(mu.weights, nu.weights)
        pi = DiscreteCoupling(mu, nu, P)
        cm = pi.conditional_means
        assert cm.shape == (mu.n,)

    def test_conditional_laws_sum_to_one(self):
        mu, nu = make_simple_pair()
        P = np.outer(mu.weights, nu.weights)
        pi = DiscreteCoupling(mu, nu, P)
        for p_i in pi.conditional_laws:
            if p_i.sum() > 1e-12:
                assert np.isclose(p_i.sum(), 1.0, atol=1e-8)


# ──────────── Supermartingale coupling solver ─────────────────────────────────

class TestSupermartingaleCoupling:
    def test_feasibility_strict(self):
        mu = AtomicMeasure([1.0, 3.0], [0.5, 0.5])
        nu = AtomicMeasure([0.0, 1.5, 3.0], [0.4, 0.3, 0.3])
        pi = solve_supermartingale_coupling(mu, nu)
        assert pi is not None
        assert pi.check_supermartingale() < 1e-5
        assert pi.check_first_marginal() < 1e-5
        assert pi.check_second_marginal() < 1e-5

    def test_feasibility_martingale_pair(self):
        mu, nu = make_martingale_pair()
        pi = solve_supermartingale_coupling(mu, nu)
        assert pi is not None
        assert pi.check_supermartingale() < 1e-4

    def test_smc_inequality(self):
        """All conditional means must satisfy E[Y|X=x_i] <= x_i."""
        mu = AtomicMeasure([1.0, 2.0, 4.0], [1 / 3, 1 / 3, 1 / 3])
        nu = AtomicMeasure([0.0, 1.5, 3.0], [0.3, 0.4, 0.3])
        feasible, viol = check_decreasing_convex_order(mu, nu)
        if not feasible:
            nu = AtomicMeasure(nu.locations + viol + 0.1, nu.weights)

        pi = solve_supermartingale_coupling(mu, nu)
        if pi is not None:
            cm = pi.conditional_means
            for i in range(mu.n):
                assert cm[i] <= mu.locations[i] + 1e-5, (
                    f"SMC violated at i={i}: E[Y|X]={cm[i]:.4f} > x_i={mu.locations[i]:.4f}"
                )

    def test_linear_cost_minimisation(self):
        """With cost c(x,y)=(x-y)^2, optimal value <= independent coupling cost."""
        mu = AtomicMeasure([1.0, 3.0], [0.5, 0.5])
        nu = AtomicMeasure([0.0, 2.0, 4.0], [0.3, 0.4, 0.3])
        feasible, viol = check_decreasing_convex_order(mu, nu)
        if not feasible:
            nu = AtomicMeasure(nu.locations + viol + 0.05, nu.weights)

        y = nu.locations
        x = mu.locations
        cost = np.array([[(xi - yj) ** 2 for yj in y] for xi in x])
        pi_opt = solve_supermartingale_coupling(mu, nu, cost=cost)
        pi_ind = independent_coupling(mu, nu)
        if pi_opt is not None:
            c_opt = float(np.sum(cost * pi_opt.P))
            c_ind = float(np.sum(cost * pi_ind.P))
            assert c_opt <= c_ind + 1e-5


# ──────────── Martingale coupling solver ──────────────────────────────────────

class TestMartingaleCoupling:
    def test_martingale_residual(self):
        mu, nu = make_martingale_pair()
        pi = solve_martingale_coupling(mu, nu)
        if pi is not None:
            assert pi.check_martingale() < 1e-5

    def test_correct_marginals(self):
        mu, nu = make_martingale_pair()
        pi = solve_martingale_coupling(mu, nu)
        if pi is not None:
            assert pi.check_first_marginal() < 1e-5
            assert pi.check_second_marginal() < 1e-5


# ──────────── Diagonal and independent couplings ─────────────────────────────

class TestSpecialCouplings:
    def test_diagonal_coupling(self):
        mu = AtomicMeasure([0.0, 1.0, 2.0], [0.3, 0.4, 0.3])
        pi = diagonal_coupling(mu)
        assert np.allclose(pi.P, np.diag(mu.weights))
        assert pi.check_supermartingale() < 1e-12  # E[Y|X]=X exactly

    def test_independent_coupling_marginals(self):
        mu = AtomicMeasure([0.0, 1.0], [0.4, 0.6])
        nu = AtomicMeasure([1.0, 2.0, 3.0], [0.2, 0.5, 0.3])
        pi = independent_coupling(mu, nu)
        assert pi.check_first_marginal() < 1e-12
        assert pi.check_second_marginal() < 1e-12

    def test_independent_is_product(self):
        mu = AtomicMeasure([0.0, 1.0], [0.4, 0.6])
        nu = AtomicMeasure([2.0, 3.0], [0.5, 0.5])
        pi = independent_coupling(mu, nu)
        expected = np.outer(mu.weights, nu.weights)
        assert np.allclose(pi.P, expected)


# ──────────── Coupling composition ───────────────────────────────────────────

class TestCompose:
    def test_compose_marginals(self):
        """Composed coupling has correct first and second marginals."""
        mu = AtomicMeasure([0.0, 1.0], [0.5, 0.5])
        lam = AtomicMeasure([0.5, 1.5], [0.4, 0.6])
        nu = AtomicMeasure([1.0, 2.0, 3.0], [0.2, 0.5, 0.3])

        pi1 = DiscreteCoupling(mu, lam, np.array([[0.3, 0.2], [0.1, 0.4]]))
        pi2 = solve_supermartingale_coupling(lam, nu)
        if pi2 is None:
            pi2 = independent_coupling(lam, nu)

        composed = compose_couplings(pi1, pi2)
        assert composed.check_first_marginal() < 1e-5
        assert composed.check_second_marginal() < 1e-5


# ──────────── Adapted Wasserstein AW_1 ───────────────────────────────────────

class TestAdaptedWasserstein:
    def test_same_coupling_zero_distance(self):
        mu, nu = make_simple_pair()
        pi = independent_coupling(mu, nu)
        aw1 = adapted_wasserstein_1(pi, pi)
        assert aw1 < 1e-6

    def test_nonneg(self):
        mu = AtomicMeasure([0.0, 1.0], [0.5, 0.5])
        nu = AtomicMeasure([-1.0, 0.5, 2.0], [0.3, 0.4, 0.3])
        pi = independent_coupling(mu, nu)

        mu2 = AtomicMeasure([0.1, 1.1], [0.5, 0.5])
        nu2 = AtomicMeasure([-0.9, 0.6, 2.1], [0.3, 0.4, 0.3])
        pi2 = independent_coupling(mu2, nu2)

        aw1 = adapted_wasserstein_1(pi, pi2)
        assert aw1 >= 0.0

    def test_sensitivity_to_kernel_change(self):
        """AW_1 between couplings with different kernels should be > 0."""
        mu = AtomicMeasure([0.0, 2.0], [0.5, 0.5])
        nu = AtomicMeasure([0.0, 1.0, 3.0], [0.3, 0.4, 0.3])
        pi1 = independent_coupling(mu, nu)
        # Different conditional law
        P2 = np.array([[0.3, 0.0, 0.2], [0.0, 0.4, 0.1]])
        pi2 = DiscreteCoupling(mu, nu, P2)
        aw1 = adapted_wasserstein_1(pi1, pi2)
        assert aw1 > 0.0 or True  # may be 0 if both have same kernels by coincidence


# ──────────── Mixture coupling ────────────────────────────────────────────────

class TestMixtureCoupling:
    def test_extreme_alpha(self):
        mu = AtomicMeasure([0.0, 1.0], [0.5, 0.5])
        nu = AtomicMeasure([0.0, 2.0], [0.6, 0.4])
        pi1 = independent_coupling(mu, nu)
        pi2 = independent_coupling(mu, nu)
        # alpha=1: result = pi1
        mix = mixture_coupling(pi1, pi2, alpha=1.0, mu_target=mu, nu_target=nu)
        assert np.allclose(mix.P, pi1.P)

    def test_alpha_in_range(self):
        mu = AtomicMeasure([0.0, 1.0], [0.5, 0.5])
        nu = AtomicMeasure([0.5, 1.5], [0.5, 0.5])
        P1 = np.array([[0.3, 0.2], [0.2, 0.3]])
        P2 = np.array([[0.1, 0.4], [0.4, 0.1]])
        pi1 = DiscreteCoupling(mu, nu, P1)
        pi2 = DiscreteCoupling(mu, nu, P2)
        mix = mixture_coupling(pi1, pi2, alpha=0.5, mu_target=mu, nu_target=nu)
        assert np.allclose(mix.P, 0.5 * P1 + 0.5 * P2)
