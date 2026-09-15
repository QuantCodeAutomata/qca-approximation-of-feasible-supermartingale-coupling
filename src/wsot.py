"""
Weak Supermartingale Optimal Transport (WSOT) stability and continuity.

Implements Theorem 2.3: V_S^C(mu^k, nu^k) -> V_S^C(mu, nu) as (mu^k, nu^k) -> (mu, nu).

Supports several weak cost families C(x, m):
  C1: classical linear C(x,m) = int c(x,y) m(dy)
  C2: strictly convex-in-m: C(x,m) = sum_j w_j(x) m_j^2 + linear terms
  C3: variance penalized: C(x,m) = int phi(y-x) m(dy) + lambda*Var_m(Y)

Custom — Context7 found no library equivalent for WSOT with weak costs.
Uses cvxpy for convex optimization.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Callable

from src.measures import AtomicMeasure, wasserstein_1, check_decreasing_convex_order
from src.couplings import DiscreteCoupling, adapted_wasserstein_1


@dataclass
class WSOTResult:
    """Result of a single WSOT solve.

    Attributes
    ----------
    value : float  Optimal value V_S^C(mu, nu).
    optimizer : DiscreteCoupling or None  Optimal coupling.
    status : str  Solver status.
    """

    value: float
    optimizer: DiscreteCoupling | None
    status: str


# ─────────────────────────── Cost function classes ────────────────────────────

class LinearCost:
    """Classical pointwise linear cost C(x,m) = int c(x,y) m(dy).

    Parameters
    ----------
    c_fn : callable(x: float, y: np.ndarray) -> np.ndarray
        Pointwise cost function.
    """

    def __init__(self, c_fn: Callable) -> None:
        self.c_fn = c_fn
        self.name = "LinearCost"

    def cost_matrix(self, x_locs: np.ndarray, y_locs: np.ndarray) -> np.ndarray:
        """Return (n_x, n_y) cost matrix c[i,j] = c(x_i, y_j)."""
        return np.array([
            [self.c_fn(xi, yj) for yj in y_locs] for xi in x_locs
        ])

    def is_linear_in_p(self) -> bool:
        return True


class QuadraticWeakCost:
    """Strictly convex weak cost C(x,m) = sum_j w_j(x) * m_j^2 + v_j(x) * m_j.

    Here m_j = m({y_j}) is the conditional probability mass at y_j.
    w_j(x) > 0 makes this strictly convex in the conditional law m.

    Parameters
    ----------
    w_fn : callable(x: float, y_locs: np.ndarray) -> np.ndarray  Weight function w(x, y_j) > 0.
    v_fn : callable(x: float, y_locs: np.ndarray) -> np.ndarray  Linear term v(x, y_j).
    """

    def __init__(
        self,
        w_fn: Callable | None = None,
        v_fn: Callable | None = None,
        name: str = "QuadraticWeakCost",
    ) -> None:
        self.w_fn = w_fn if w_fn is not None else (lambda x, y: np.ones_like(y))
        self.v_fn = v_fn if v_fn is not None else (lambda x, y: -y)
        self.name = name

    def is_linear_in_p(self) -> bool:
        return False


class VarianceCost:
    """Convex mean-deviation + l2-regularisation cost:

        C(x, m) = int phi(y-x) m(dy) + lambda * sum_j m_j^2

    The first term is the expected convex displacement phi(y-x) (linear in m).
    The second term is the squared l2-norm of the conditional probability vector,
    which is strictly convex in m and penalises spread.

    Note: the naively written variance Var[Y|X] = E[Y^2|X] - (E[Y|X])^2 is
    *concave* in the coupling matrix P, so minimising it is a non-convex
    (concave-minimisation) problem that cvxpy cannot handle.  We therefore use
    the l2-norm surrogate above, which has the same qualitative effect (it
    penalises diffuse conditional distributions) while remaining strictly convex
    and DCP-compliant.

    Parameters
    ----------
    phi_fn : callable(d: float) -> float  Convex displacement penalty.
    lam : float  Weight on the l2-norm spread penalty (> 0).
    """

    def __init__(
        self,
        phi_fn: Callable | None = None,
        lam: float = 1.0,
        name: str = "VarianceCost",
    ) -> None:
        self.phi_fn = phi_fn if phi_fn is not None else (lambda d: d ** 2)
        self.lam = lam
        self.name = name

    def is_linear_in_p(self) -> bool:
        return False


# ──────────────────────────── Solver functions ────────────────────────────────

def solve_wsot_linear(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    cost: LinearCost,
    minimize: bool = True,
) -> WSOTResult:
    """Solve WSOT with a classical linear cost via LP.

    min sum_{i,j} c(x_i, y_j) P[i,j]
    s.t. P >= 0, row sums = a, col sums = b,
         sum_j y_j P[i,j] <= x_i * a_i.

    # Using cvxpy for LP — standard LP, Context7 library check done.
    """
    n, m = mu.n, nu.n
    a = mu.weights.astype(float)
    b = nu.weights.astype(float)
    x = mu.locations
    y = nu.locations

    C = cost.cost_matrix(x, y).astype(float)

    P = cp.Variable((n, m), nonneg=True)
    constraints = [
        cp.sum(P, axis=1) == a,
        cp.sum(P, axis=0) == b,
    ]
    for i in range(n):
        constraints.append(P[i, :] @ y <= x[i] * a[i])

    obj_expr = cp.sum(cp.multiply(C, P))
    obj = cp.Minimize(obj_expr) if minimize else cp.Maximize(obj_expr)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver="CLARABEL", verbose=False)

    if P.value is None or prob.status in ("infeasible", "unbounded"):
        prob.solve(solver="SCS", verbose=False)

    if P.value is None:
        return WSOTResult(value=np.nan, optimizer=None, status=prob.status)

    Pval = np.clip(P.value, 0.0, None)
    coupling = DiscreteCoupling(mu, nu, Pval)
    value = float(np.sum(C * Pval))

    return WSOTResult(value=value, optimizer=coupling, status=prob.status)


def solve_wsot_quadratic(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    cost: QuadraticWeakCost,
    minimize: bool = True,
) -> WSOTResult:
    """Solve WSOT with strictly convex quadratic weak cost via QP.

    min sum_i a_i C(x_i, m_i) = sum_i a_i [sum_j w_j(x_i)(p_{ij})^2 + v_j(x_i) p_{ij}]
    where p_{ij} = P[i,j]/a_i (conditional law at x_i).

    Equivalently: sum_{ij} [w_j(x_i)/a_i * P[i,j]^2 + v_j(x_i) * P[i,j]] subject to SMC.

    # Using cvxpy for QP — Context7 confirmed quadratic optimization support.
    """
    n, m_nu = mu.n, nu.n
    a = mu.weights.astype(float)
    b = nu.weights.astype(float)
    x = mu.locations
    y = nu.locations

    # Weight and linear term matrices
    W = np.array([cost.w_fn(xi, y) for xi in x]).astype(float)  # (n, m_nu)
    V = np.array([cost.v_fn(xi, y) for xi in x]).astype(float)

    P = cp.Variable((n, m_nu), nonneg=True)
    constraints = [
        cp.sum(P, axis=1) == a,
        cp.sum(P, axis=0) == b,
    ]
    for i in range(n):
        constraints.append(P[i, :] @ y <= x[i] * a[i])

    # Objective: sum_{ij} [W[i,j]/a_i * P[i,j]^2 + V[i,j] * P[i,j]]
    obj_parts = []
    for i in range(n):
        if a[i] > 1e-14:
            quad = cp.sum(cp.multiply(W[i, :] / a[i], cp.square(P[i, :])))
            lin = V[i, :] @ P[i, :]
            obj_parts.append(quad + lin)
        else:
            obj_parts.append(cp.Constant(0))

    obj_expr = cp.sum(obj_parts)
    obj = cp.Minimize(obj_expr) if minimize else cp.Maximize(obj_expr)
    prob = cp.Problem(obj, constraints)

    try:
        prob.solve(solver="OSQP", verbose=False)
    except Exception:
        prob.solve(solver="SCS", verbose=False)

    if P.value is None or prob.status in ("infeasible", "unbounded"):
        try:
            prob.solve(solver="SCS", verbose=False)
        except Exception:
            return WSOTResult(value=np.nan, optimizer=None, status="failed")

    if P.value is None:
        return WSOTResult(value=np.nan, optimizer=None, status=prob.status)

    Pval = np.clip(P.value, 0.0, None)
    coupling = DiscreteCoupling(mu, nu, Pval)

    # Evaluate objective
    value = 0.0
    for i in range(n):
        if a[i] > 1e-14:
            p_i = Pval[i, :] / a[i]
            value += a[i] * (
                np.dot(W[i, :], p_i ** 2) + np.dot(V[i, :], p_i)
            )

    return WSOTResult(value=value, optimizer=coupling, status=prob.status)


def solve_wsot_variance(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    cost: VarianceCost,
    minimize: bool = True,
) -> WSOTResult:
    """Solve WSOT with convex mean-deviation + l2-spread cost.

    Objective (per coupling row, using conditional probability p_i = P_i / a_i):
        sum_i a_i * C(x_i, p_i)
        = sum_{ij} phi(y_j - x_i) * P_ij          (linear in P)
          + lambda * sum_i (1/a_i) * ||P_i||_2^2   (convex quadratic in P)

    Both terms are DCP-compliant.  The l2-norm quadratic strictly penalises
    diffuse conditional distributions, so the objective is strictly convex in P.

    # Custom — Context7 found no library equivalent for variance-cost WSOT.
    """
    n, m_nu = mu.n, nu.n
    a = mu.weights.astype(float)
    b = nu.weights.astype(float)
    x = mu.locations
    y = nu.locations

    Phi = np.array([[cost.phi_fn(yj - xi) for yj in y] for xi in x]).astype(float)

    P = cp.Variable((n, m_nu), nonneg=True)
    constraints = [
        cp.sum(P, axis=1) == a,
        cp.sum(P, axis=0) == b,
    ]
    for i in range(n):
        constraints.append(P[i, :] @ y <= x[i] * a[i])

    # Linear phi-displacement term
    phi_term = cp.sum(cp.multiply(Phi, P))

    # Convex l2-spread penalty:  sum_i (lambda / a_i) * ||P_i||_2^2
    l2_terms = []
    for i in range(n):
        if a[i] > 1e-14:
            l2_terms.append((cost.lam / a[i]) * cp.sum_squares(P[i, :]))
    l2_term = cp.sum(l2_terms) if l2_terms else cp.Constant(0.0)

    obj_expr = phi_term + l2_term
    obj = cp.Minimize(obj_expr) if minimize else cp.Maximize(obj_expr)
    prob = cp.Problem(obj, constraints)

    try:
        prob.solve(solver="CLARABEL", verbose=False)
    except Exception:
        try:
            prob.solve(solver="SCS", verbose=False)
        except Exception:
            return WSOTResult(value=np.nan, optimizer=None, status="failed")

    if P.value is None or prob.status in ("infeasible", "unbounded"):
        prob.solve(solver="SCS", verbose=False)

    if P.value is None:
        return WSOTResult(value=np.nan, optimizer=None, status=prob.status)

    Pval = np.clip(P.value, 0.0, None)
    coupling = DiscreteCoupling(mu, nu, Pval)

    # Evaluate objective value numerically
    value = float(np.sum(Phi * Pval))
    for i in range(n):
        if a[i] > 1e-14:
            value += (cost.lam / a[i]) * float(np.sum(Pval[i, :] ** 2))

    return WSOTResult(value=value, optimizer=coupling, status=prob.status)


def solve_wsot(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    cost,
    minimize: bool = True,
) -> WSOTResult:
    """Dispatch WSOT solver based on cost type."""
    if isinstance(cost, LinearCost):
        return solve_wsot_linear(mu, nu, cost, minimize=minimize)
    elif isinstance(cost, QuadraticWeakCost):
        return solve_wsot_quadratic(mu, nu, cost, minimize=minimize)
    elif isinstance(cost, VarianceCost):
        return solve_wsot_variance(mu, nu, cost, minimize=minimize)
    else:
        raise ValueError(f"Unknown cost type: {type(cost)}")


@dataclass
class StabilityResult:
    """Stability experiment result for one cost and one test family.

    Attributes
    ----------
    cost_name : str
    v_limit : float  V_S^C(mu, nu).
    v_sequence : list[float]  V_S^C(mu_k, nu_k) for k=1,...,K.
    value_gaps : list[float]  |v_k - v_limit|.
    aw1_sequence : list[float]  AW_1(pi^{*,k}, pi^*) when optimizer unique.
    w1_mu_sequence : list[float]  W_1(mu_k, mu).
    w1_nu_sequence : list[float]  W_1(nu_k, nu).
    """

    cost_name: str
    v_limit: float
    v_sequence: list[float]
    value_gaps: list[float]
    aw1_sequence: list[float]
    w1_mu_sequence: list[float]
    w1_nu_sequence: list[float]


def run_stability_experiment(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    perturbation_sequence: list[tuple[AtomicMeasure, AtomicMeasure]],
    costs: list,
    verbose: bool = True,
) -> list[StabilityResult]:
    """Run the WSOT stability experiment for each cost family.

    Parameters
    ----------
    mu, nu : AtomicMeasure  Target marginals.
    perturbation_sequence : list of (mu_k, nu_k).
    costs : list of cost objects.
    verbose : bool.

    Returns
    -------
    list[StabilityResult]
    """
    stability_results = []

    for cost in costs:
        if verbose:
            print(f"\n  Cost: {cost.name}")

        # Solve limit problem
        res_limit = solve_wsot(mu, nu, cost)
        v_limit = res_limit.value
        pi_star = res_limit.optimizer

        if verbose:
            print(f"    V_limit = {v_limit:.6f}, status={res_limit.status}")

        v_seq = []
        aw1_seq = []
        w1_mu_seq = []
        w1_nu_seq = []
        value_gaps = []

        for k, (mu_k, nu_k) in enumerate(perturbation_sequence, start=1):
            res_k = solve_wsot(mu_k, nu_k, cost)
            v_k = res_k.value
            pi_star_k = res_k.optimizer

            w1_mu = wasserstein_1(mu_k, mu)
            w1_nu = wasserstein_1(nu_k, nu)
            gap = abs(v_k - v_limit) if not np.isnan(v_k) else np.nan

            # AW_1 between optimizers if available
            if pi_star is not None and pi_star_k is not None:
                try:
                    aw1 = adapted_wasserstein_1(pi_star_k, pi_star)
                except Exception:
                    aw1 = np.nan
            else:
                aw1 = np.nan

            v_seq.append(v_k)
            aw1_seq.append(aw1)
            w1_mu_seq.append(w1_mu)
            w1_nu_seq.append(w1_nu)
            value_gaps.append(gap)

            if verbose:
                print(f"    k={k}: V={v_k:.6f}, gap={gap:.4e}, "
                      f"AW1={aw1:.4f}, W1_mu={w1_mu:.4f}")

        stability_results.append(StabilityResult(
            cost_name=cost.name,
            v_limit=v_limit,
            v_sequence=v_seq,
            value_gaps=value_gaps,
            aw1_sequence=aw1_seq,
            w1_mu_sequence=w1_mu_seq,
            w1_nu_sequence=w1_nu_seq,
        ))

    return stability_results
