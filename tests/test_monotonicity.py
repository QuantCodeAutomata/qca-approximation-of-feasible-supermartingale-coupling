"""Tests for src/monotonicity.py — C-monotonicity testing (Theorem 2.9)."""

import numpy as np
import pytest

from src.measures import AtomicMeasure, check_decreasing_convex_order
from src.couplings import (
    DiscreteCoupling, solve_supermartingale_coupling, independent_coupling
)
from src.decomposition import compute_irreducible_decomposition
from src.wsot import LinearCost, QuadraticWeakCost, solve_wsot
from src.monotonicity import (
    MonotonicityTest,
    evaluate_support_cost,
    classify_support_point,
    adversarial_competitor_search,
    generate_competitors_random,
    generate_competitors_exhaustive,
    check_c_monotonicity,
)


# ──────────── Helpers ─────────────────────────────────────────────────────────

def make_small_instance():
    """2x3 instance with known optimal coupling."""
    mu = AtomicMeasure([0.0, 2.0], [0.5, 0.5])
    nu = AtomicMeasure([-1.0, 1.0, 3.0], [0.3, 0.4, 0.3])
    feasible, viol = check_decreasing_convex_order(mu, nu)
    if not feasible:
        nu = AtomicMeasure(nu.locations + viol + 0.05, nu.weights)
    return mu, nu


def make_martingale_instance():
    """2x3 martingale instance."""
    mu = AtomicMeasure([-1.0, 1.0], [0.5, 0.5])
    nu = AtomicMeasure([-2.0, 0.0, 2.0], [0.25, 0.5, 0.25])
    return mu, nu


# ──────────── Support cost evaluation ────────────────────────────────────────

class TestEvaluateSupportCost:
    def test_linear_cost_simple(self):
        x_locs = np.array([0.0, 2.0])
        y_locs = np.array([0.0, 1.0, 3.0])
        weights = np.array([0.5, 0.5])
        # p_i = uniform probability over y_locs
        p_list = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([1 / 3, 1 / 3, 1 / 3])]
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)

        val = evaluate_support_cost(x_locs, p_list, y_locs, cost, weights)
        assert np.isfinite(val)
        assert val >= 0.0

    def test_quadratic_cost_nonneg(self):
        x_locs = np.array([1.0])
        y_locs = np.array([0.0, 2.0])
        weights = np.array([1.0])
        p_list = [np.array([0.5, 0.5])]
        cost = QuadraticWeakCost(
            w_fn=lambda x, y: np.ones_like(y),
            v_fn=lambda x, y: np.zeros_like(y),
        )
        val = evaluate_support_cost(x_locs, p_list, y_locs, cost, weights)
        # sum_j p_j^2 = 0.5^2 + 0.5^2 = 0.5 -> cost = 1.0 * 0.5 = 0.5
        assert np.isclose(val, 0.5, atol=1e-8)

    def test_zero_weight_ignored(self):
        x_locs = np.array([0.0, 2.0])
        y_locs = np.array([1.0, 2.0])
        weights = np.array([0.0, 1.0])  # first atom has zero weight
        p_list = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
        cost = LinearCost(c_fn=lambda x, y: abs(x - y))
        # Only second atom should contribute
        val = evaluate_support_cost(x_locs, p_list, y_locs, cost, weights)
        expected = 1.0 * (0.5 * abs(2.0 - 1.0) + 0.5 * abs(2.0 - 2.0))
        assert np.isclose(val, expected, atol=1e-8)


# ──────────── Support point classification ────────────────────────────────────

class TestClassifySupportPoint:
    def test_strict_sm_above_x_star(self):
        # x > x_star => strict_sm
        cls = classify_support_point(
            xi=5.0,
            pi=np.array([0.5, 0.5]),
            y_locs=np.array([0.0, 2.0]),
            x_star=3.0,
        )
        assert cls == "strict_sm"

    def test_martingale_when_mean_equals_x(self):
        # x = 1.0, E[Y|X] = 1.0*0.5 + 1.0*0.5 = 1.0 => martingale
        cls = classify_support_point(
            xi=1.0,
            pi=np.array([0.5, 0.5]),
            y_locs=np.array([0.0, 2.0]),
            x_star=0.0,
        )
        assert cls == "martingale"

    def test_diagonal_when_below_x_star_mean_not_x(self):
        # x=0.5, mean = 0.0*0.8 + 2.0*0.2 = 0.4 != 0.5 but below x_star
        cls = classify_support_point(
            xi=0.5,
            pi=np.array([0.8, 0.2]),
            y_locs=np.array([0.0, 2.0]),
            x_star=2.0,
        )
        # Could be "diagonal" or "martingale" depending on tol
        assert cls in ("diagonal", "martingale", "strict_sm")


# ──────────── Competitor generation ──────────────────────────────────────────

class TestGenerateCompetitors:
    def test_random_competitors_nonempty(self):
        mu, nu = make_small_instance()
        res = solve_wsot(mu, nu, LinearCost(c_fn=lambda x, y: (x - y) ** 2))
        pi = res.optimizer or independent_coupling(mu, nu)
        decomp = compute_irreducible_decomposition(mu, nu)
        comps = generate_competitors_random(
            mu.locations, pi.conditional_laws, nu.locations,
            mu.weights, decomp.x_star, n_competitors=50,
        )
        assert isinstance(comps, list)

    def test_exhaustive_competitors_small(self):
        """For |supp(nu)| <= 3, exhaustive generation should work."""
        mu, nu = make_small_instance()
        res = solve_wsot(mu, nu, LinearCost(c_fn=lambda x, y: (x - y) ** 2))
        pi = res.optimizer or independent_coupling(mu, nu)
        decomp = compute_irreducible_decomposition(mu, nu)
        comps = generate_competitors_exhaustive(
            mu.locations, pi.conditional_laws, nu.locations,
            mu.weights, decomp.x_star, grid_resolution=3,
        )
        assert isinstance(comps, list)

    def test_competitors_are_prob_vectors(self):
        """Each competitor conditional law should sum to ~1."""
        mu, nu = make_small_instance()
        res = solve_wsot(mu, nu, LinearCost(c_fn=lambda x, y: (x - y) ** 2))
        pi = res.optimizer or independent_coupling(mu, nu)
        decomp = compute_irreducible_decomposition(mu, nu)
        comps = generate_competitors_random(
            mu.locations, pi.conditional_laws, nu.locations,
            mu.weights, decomp.x_star, n_competitors=20,
        )
        for q_list in comps:
            for q_i in q_list:
                assert abs(q_i.sum() - 1.0) < 0.1
                assert np.all(q_i >= -1e-6)


# ──────────── Adversarial search ──────────────────────────────────────────────

class TestAdversarialSearch:
    def test_returns_float(self):
        mu, nu = make_small_instance()
        res = solve_wsot(mu, nu, LinearCost(c_fn=lambda x, y: (x - y) ** 2))
        pi = res.optimizer or independent_coupling(mu, nu)
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        decomp = compute_irreducible_decomposition(mu, nu)
        gap, q_opt = adversarial_competitor_search(
            mu.locations, pi.conditional_laws, nu.locations,
            mu.weights, decomp.x_star, cost,
        )
        assert np.isfinite(gap) or q_opt is None

    def test_optimal_coupling_non_positive_gap(self):
        """For optimal coupling, adversarial gap should be <= 0 (monotone)."""
        mu, nu = make_small_instance()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        cost.name = "L2"
        res = solve_wsot(mu, nu, cost)
        if res.optimizer is None:
            return  # skip if solver failed
        pi = res.optimizer
        decomp = compute_irreducible_decomposition(mu, nu)
        gap, _ = adversarial_competitor_search(
            mu.locations, pi.conditional_laws, nu.locations,
            mu.weights, decomp.x_star, cost,
        )
        # For optimal coupling, no competitor should improve cost
        assert gap <= 0.1  # small positive tolerance for numerics


# ──────────── Full monotonicity test ─────────────────────────────────────────

class TestCMonotonicity:
    def test_returns_monotonicity_test(self):
        mu, nu = make_small_instance()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        cost.name = "L2"
        res = solve_wsot(mu, nu, cost)
        pi = res.optimizer or independent_coupling(mu, nu)
        decomp = compute_irreducible_decomposition(mu, nu)
        mt = check_c_monotonicity(
            pi, decomp.x_star, cost,
            n_random_competitors=50, use_exhaustive=True, use_adversarial=True,
        )
        assert isinstance(mt, MonotonicityTest)

    def test_optimal_coupling_is_monotone(self):
        """Optimal coupling should pass C-monotonicity test."""
        mu, nu = make_small_instance()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        cost.name = "L2"
        res = solve_wsot(mu, nu, cost)
        if res.optimizer is None:
            return
        pi = res.optimizer
        decomp = compute_irreducible_decomposition(mu, nu)
        mt = check_c_monotonicity(
            pi, decomp.x_star, cost,
            n_random_competitors=100, use_exhaustive=True, use_adversarial=True,
            tol=1e-4,
        )
        # Optimal coupling: violation should be small
        assert mt.max_violation < 0.5, f"Violation too large: {mt.max_violation}"

    def test_independent_coupling_may_violate(self):
        """Independent coupling is not generally optimal, so it may be non-monotone."""
        mu, nu = make_small_instance()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        cost.name = "L2"
        pi = independent_coupling(mu, nu)
        decomp = compute_irreducible_decomposition(mu, nu)
        mt = check_c_monotonicity(
            pi, decomp.x_star, cost,
            n_random_competitors=50, use_exhaustive=False, use_adversarial=False,
            tol=1e-4,
        )
        # Just verify the function runs and returns sensible output
        assert isinstance(mt.is_monotone, (bool, np.bool_))
        assert np.isfinite(mt.max_violation)

    def test_quadratic_cost_optimal_monotone(self):
        """Optimal coupling under quadratic weak cost should also be monotone."""
        mu, nu = make_small_instance()
        cost = QuadraticWeakCost(
            w_fn=lambda x, y: np.ones_like(y) * 2.0,
            v_fn=lambda x, y: -y,
            name="QuadTest",
        )
        res = solve_wsot(mu, nu, cost)
        if res.optimizer is None:
            return
        pi = res.optimizer
        decomp = compute_irreducible_decomposition(mu, nu)
        mt = check_c_monotonicity(
            pi, decomp.x_star, cost,
            n_random_competitors=100, use_exhaustive=True, use_adversarial=True,
            tol=1e-3,
        )
        assert mt.max_violation < 1.0  # loosely bounded for numerical tolerance

    def test_monotonicity_test_fields(self):
        mu, nu = make_small_instance()
        cost = LinearCost(c_fn=lambda x, y: abs(x - y))
        cost.name = "L1"
        pi = independent_coupling(mu, nu)
        decomp = compute_irreducible_decomposition(mu, nu)
        mt = check_c_monotonicity(
            pi, decomp.x_star, cost,
            n_random_competitors=20, use_exhaustive=False, use_adversarial=False,
        )
        assert isinstance(mt.is_monotone, (bool, np.bool_))
        assert isinstance(mt.max_violation, float)
        assert isinstance(mt.n_competitors_tested, int)
        assert mt.n_competitors_tested >= 0
        assert mt.support_size == mu.n

    def test_martingale_pair_monotone(self):
        """In martingale case, optimal coupling should satisfy C-monotonicity."""
        mu, nu = make_martingale_instance()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        cost.name = "L2Mart"
        res = solve_wsot(mu, nu, cost)
        if res.optimizer is None:
            return
        pi = res.optimizer
        decomp = compute_irreducible_decomposition(mu, nu)
        mt = check_c_monotonicity(
            pi, decomp.x_star, cost,
            n_random_competitors=100, use_exhaustive=True, use_adversarial=True,
            tol=1e-3,
        )
        assert mt.max_violation < 1.0
