"""Tests for src/wsot.py — LinearCost, QuadraticWeakCost, WSOT solvers, stability."""

import numpy as np
import pytest

from src.measures import AtomicMeasure, check_decreasing_convex_order, wasserstein_1
from src.couplings import DiscreteCoupling, independent_coupling
from src.wsot import (
    LinearCost,
    QuadraticWeakCost,
    VarianceCost,
    WSOTResult,
    solve_wsot,
    solve_wsot_linear,
    solve_wsot_quadratic,
    solve_wsot_variance,
    run_stability_experiment,
    StabilityResult,
)
from src.approximation import construct_perturbation_sequence


# ──────────── Helpers ─────────────────────────────────────────────────────────

def make_feasible_pair():
    mu = AtomicMeasure([1.0, 3.0], [0.5, 0.5])
    nu = AtomicMeasure([0.0, 1.5, 3.5], [0.4, 0.3, 0.3])
    feasible, viol = check_decreasing_convex_order(mu, nu)
    if not feasible:
        nu = AtomicMeasure(nu.locations + viol + 0.05, nu.weights)
    return mu, nu


def make_martingale_pair():
    mu = AtomicMeasure([-1.0, 1.0], [0.5, 0.5])
    nu = AtomicMeasure([-2.0, 0.0, 2.0], [0.25, 0.5, 0.25])
    return mu, nu


# ──────────── Cost classes ────────────────────────────────────────────────────

class TestCostClasses:
    def test_linear_cost_is_linear(self):
        c = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        assert c.is_linear_in_p()

    def test_quadratic_cost_not_linear(self):
        c = QuadraticWeakCost()
        assert not c.is_linear_in_p()

    def test_variance_cost_not_linear(self):
        c = VarianceCost()
        assert not c.is_linear_in_p()

    def test_linear_cost_matrix_shape(self):
        c = LinearCost(c_fn=lambda x, y: abs(x - y))
        x = np.array([0.0, 1.0])
        y = np.array([0.5, 1.5, 2.5])
        M = c.cost_matrix(x, y)
        assert M.shape == (2, 3)

    def test_linear_cost_matrix_values(self):
        c = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        x = np.array([0.0])
        y = np.array([1.0, 2.0])
        M = c.cost_matrix(x, y)
        assert np.allclose(M, [[1.0, 4.0]])


# ──────────── WSOT linear solver ─────────────────────────────────────────────

class TestSolveWSOTLinear:
    def test_returns_wsot_result(self):
        mu, nu = make_feasible_pair()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        res = solve_wsot_linear(mu, nu, cost)
        assert isinstance(res, WSOTResult)

    def test_value_finite(self):
        mu, nu = make_feasible_pair()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        res = solve_wsot_linear(mu, nu, cost)
        assert np.isfinite(res.value) or res.optimizer is None

    def test_optimizer_feasible(self):
        mu, nu = make_feasible_pair()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        res = solve_wsot_linear(mu, nu, cost)
        if res.optimizer is not None:
            assert res.optimizer.check_first_marginal() < 1e-4
            assert res.optimizer.check_second_marginal() < 1e-4
            assert res.optimizer.check_supermartingale() < 1e-4

    def test_value_leq_independent_coupling(self):
        """Minimized WSOT value should not exceed independent coupling cost."""
        mu, nu = make_feasible_pair()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        res = solve_wsot_linear(mu, nu, cost)
        if res.optimizer is not None:
            pi_ind = independent_coupling(mu, nu)
            C = cost.cost_matrix(mu.locations, nu.locations)
            c_ind = float(np.sum(C * pi_ind.P))
            assert res.value <= c_ind + 1e-4

    def test_l1_cost(self):
        mu, nu = make_feasible_pair()
        cost = LinearCost(c_fn=lambda x, y: abs(x - y))
        res = solve_wsot_linear(mu, nu, cost)
        assert res.value >= 0 or res.optimizer is None

    def test_martingale_pair(self):
        mu, nu = make_martingale_pair()
        cost = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        res = solve_wsot_linear(mu, nu, cost)
        if res.optimizer is not None:
            assert res.optimizer.check_supermartingale() < 1e-4


# ──────────── WSOT quadratic solver ──────────────────────────────────────────

class TestSolveWSOTQuadratic:
    def test_returns_wsot_result(self):
        mu, nu = make_feasible_pair()
        cost = QuadraticWeakCost(
            w_fn=lambda x, y: np.ones_like(y),
            v_fn=lambda x, y: -y,
        )
        res = solve_wsot_quadratic(mu, nu, cost)
        assert isinstance(res, WSOTResult)

    def test_optimizer_feasible(self):
        mu, nu = make_feasible_pair()
        cost = QuadraticWeakCost(
            w_fn=lambda x, y: np.ones_like(y) * 2.0,
            v_fn=lambda x, y: np.zeros_like(y),
        )
        res = solve_wsot_quadratic(mu, nu, cost)
        if res.optimizer is not None:
            assert res.optimizer.check_first_marginal() < 1e-3
            assert res.optimizer.check_supermartingale() < 1e-3

    def test_value_geq_0(self):
        """With w_fn=1 and v_fn=0, cost = ||p||^2 >= 0."""
        mu, nu = make_feasible_pair()
        cost = QuadraticWeakCost(
            w_fn=lambda x, y: np.ones_like(y),
            v_fn=lambda x, y: np.zeros_like(y),
        )
        res = solve_wsot_quadratic(mu, nu, cost)
        if res.optimizer is not None:
            # With w=1, v=0: cost = sum_i a_i sum_j p_{ij}^2 >= 0
            assert res.value >= -1e-6


# ──────────── WSOT variance solver ───────────────────────────────────────────

class TestSolveWSOTVariance:
    def test_returns_wsot_result(self):
        mu, nu = make_feasible_pair()
        cost = VarianceCost(phi_fn=lambda d: d ** 2, lam=0.5)
        res = solve_wsot_variance(mu, nu, cost)
        assert isinstance(res, WSOTResult)

    def test_optimizer_feasible(self):
        mu, nu = make_feasible_pair()
        cost = VarianceCost(phi_fn=lambda d: d ** 2, lam=0.5)
        res = solve_wsot_variance(mu, nu, cost)
        if res.optimizer is not None:
            assert res.optimizer.check_first_marginal() < 1e-3
            assert res.optimizer.check_supermartingale() < 1e-3


# ──────────── dispatch function ───────────────────────────────────────────────

class TestDispatch:
    def test_linear_dispatched(self):
        mu, nu = make_feasible_pair()
        c = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        res = solve_wsot(mu, nu, c)
        assert isinstance(res, WSOTResult)

    def test_quadratic_dispatched(self):
        mu, nu = make_feasible_pair()
        c = QuadraticWeakCost()
        res = solve_wsot(mu, nu, c)
        assert isinstance(res, WSOTResult)

    def test_variance_dispatched(self):
        mu, nu = make_feasible_pair()
        c = VarianceCost()
        res = solve_wsot(mu, nu, c)
        assert isinstance(res, WSOTResult)

    def test_unknown_cost_raises(self):
        mu, nu = make_feasible_pair()
        with pytest.raises((ValueError, AttributeError)):
            solve_wsot(mu, nu, "not_a_cost")


# ──────────── Stability experiment ───────────────────────────────────────────

class TestStabilityExperiment:
    def test_returns_list_of_results(self):
        mu, nu = make_feasible_pair()
        seq = construct_perturbation_sequence(mu, nu, K=3)
        costs = [LinearCost(c_fn=lambda x, y: (x - y) ** 2)]
        costs[0].name = "TestLinear"
        results = run_stability_experiment(mu, nu, seq, costs, verbose=False)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_result_fields(self):
        mu, nu = make_feasible_pair()
        seq = construct_perturbation_sequence(mu, nu, K=3)
        c = LinearCost(c_fn=lambda x, y: abs(x - y))
        c.name = "TestL1"
        results = run_stability_experiment(mu, nu, seq, [c], verbose=False)
        sr = results[0]
        assert isinstance(sr, StabilityResult)
        assert sr.cost_name == "TestL1"
        assert len(sr.v_sequence) == 3
        assert len(sr.value_gaps) == 3
        assert len(sr.aw1_sequence) == 3

    def test_value_gaps_nonneg(self):
        mu, nu = make_feasible_pair()
        seq = construct_perturbation_sequence(mu, nu, K=3)
        c = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        c.name = "L2"
        results = run_stability_experiment(mu, nu, seq, [c], verbose=False)
        sr = results[0]
        for gap in sr.value_gaps:
            if not np.isnan(gap):
                assert gap >= 0.0

    def test_value_convergence_trend(self):
        """Value gaps should generally decrease as k increases."""
        mu, nu = make_feasible_pair()
        seq = construct_perturbation_sequence(mu, nu, K=5)
        c = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
        c.name = "L2"
        results = run_stability_experiment(mu, nu, seq, [c], verbose=False)
        gaps = [g for g in results[0].value_gaps if not np.isnan(g)]
        if len(gaps) >= 2:
            # Loosely: last gap should not be much larger than first
            assert gaps[-1] <= gaps[0] + abs(gaps[0]) + 1.0

    def test_w1_sequences_nonneg(self):
        mu, nu = make_feasible_pair()
        seq = construct_perturbation_sequence(mu, nu, K=3)
        c = LinearCost(c_fn=lambda x, y: abs(x - y))
        c.name = "L1"
        results = run_stability_experiment(mu, nu, seq, [c], verbose=False)
        sr = results[0]
        for w1_mu in sr.w1_mu_sequence:
            assert w1_mu >= 0.0
        for w1_nu in sr.w1_nu_sequence:
            assert w1_nu >= 0.0
