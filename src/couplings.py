"""
Transport couplings between atomic measures, supermartingale constraints,
martingale constraints, and adapted Wasserstein distance.

Custom implementations — Context7 found no library equivalent for
supermartingale/adapted-Wasserstein constructions.

A coupling pi between mu = sum_i a_i delta_{x_i} and nu = sum_j b_j delta_{y_j}
is stored as a matrix P[i,j] >= 0 with:
  row sums: P[i,:].sum() = a_i   (first marginal = mu)
  col sums: P[:,j].sum() = b_j   (second marginal = nu)
  supermartingale: sum_j y_j P[i,j] <= x_i * a_i  for all i
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
import ot

from src.measures import AtomicMeasure, wasserstein_1


class DiscreteCoupling:
    """Discrete coupling between two atomic measures.

    Parameters
    ----------
    mu : AtomicMeasure  First marginal.
    nu : AtomicMeasure  Second marginal.
    matrix : np.ndarray  Transport matrix P[i,j], shape (n_mu, n_nu).
    tol : float          Tolerance for feasibility checks.
    """

    def __init__(
        self,
        mu: AtomicMeasure,
        nu: AtomicMeasure,
        matrix: np.ndarray,
        tol: float = 1e-7,
    ) -> None:
        self.mu = mu
        self.nu = nu
        self.P = np.asarray(matrix, dtype=float)
        self.tol = tol
        assert self.P.shape == (mu.n, nu.n), (
            f"Matrix shape {self.P.shape} inconsistent with "
            f"mu.n={mu.n}, nu.n={nu.n}"
        )

    @property
    def conditional_means(self) -> np.ndarray:
        """Conditional mean E[Y | X=x_i] = sum_j y_j P[i,j] / a_i for a_i>0."""
        m = np.zeros(self.mu.n)
        for i in range(self.mu.n):
            if self.mu.weights[i] > 0:
                m[i] = (self.P[i, :] @ self.nu.locations) / self.mu.weights[i]
        return m

    @property
    def conditional_laws(self) -> list[np.ndarray]:
        """List of conditional probability vectors p_i = P[i,:]/a_i."""
        laws = []
        for i in range(self.mu.n):
            ai = self.mu.weights[i]
            if ai > 0:
                laws.append(self.P[i, :] / ai)
            else:
                laws.append(np.zeros(self.nu.n))
        return laws

    def check_first_marginal(self) -> float:
        """Max absolute error in first marginal; should be < tol."""
        return float(np.max(np.abs(self.P.sum(axis=1) - self.mu.weights)))

    def check_second_marginal(self) -> float:
        """Max absolute error in second marginal; should be < tol."""
        return float(np.max(np.abs(self.P.sum(axis=0) - self.nu.weights)))

    def check_supermartingale(self) -> float:
        """Supermartingale feasibility residual int (E[Y|X]-X)^+ dmu.

        Should be <= tol for pi in Pi_S(mu,nu).
        """
        cm = self.conditional_means
        violation = np.maximum(cm - self.mu.locations, 0.0) * self.mu.weights
        return float(violation.sum())

    def check_martingale(self) -> float:
        """Martingale residual int |E[Y|X]-X| dmu. Should be <= tol for Pi_M."""
        cm = self.conditional_means
        return float((np.abs(cm - self.mu.locations) * self.mu.weights).sum())

    def is_feasible(self, mode: str = "supermartingale") -> bool:
        """Return True if the coupling satisfies marginal and (super)martingale constraints."""
        m1 = self.check_first_marginal() <= self.tol
        m2 = self.check_second_marginal() <= self.tol
        if mode == "supermartingale":
            smc = self.check_supermartingale() <= self.tol
        elif mode == "martingale":
            smc = self.check_martingale() <= self.tol
        else:
            smc = True
        return m1 and m2 and smc

    def __repr__(self) -> str:
        return (
            f"DiscreteCoupling(mu.n={self.mu.n}, nu.n={self.nu.n}, "
            f"smc_residual={self.check_supermartingale():.2e})"
        )


def solve_supermartingale_coupling(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    cost: np.ndarray | None = None,
    minimize: bool = True,
    solver: str = "CLARABEL",
) -> DiscreteCoupling | None:
    """Solve a discrete (weak) supermartingale OT problem via cvxpy.

    Minimize/maximize sum_{i,j} cost[i,j] * P[i,j] subject to:
      P[i,j] >= 0
      sum_j P[i,j] = a_i   (first marginal)
      sum_i P[i,j] = b_j   (second marginal)
      sum_j y_j P[i,j] <= x_i * a_i   (supermartingale)

    Parameters
    ----------
    mu, nu : AtomicMeasure
    cost : (n_mu, n_nu) array or None.  If None, uses zero cost (feasibility).
    minimize : bool  Minimize if True, maximize if False.
    solver : str  cvxpy solver name.

    Returns
    -------
    DiscreteCoupling or None if infeasible.
    """
    n, m = mu.n, nu.n
    a = mu.weights.astype(float)
    b = nu.weights.astype(float)
    x = mu.locations
    y = nu.locations

    P = cp.Variable((n, m), nonneg=True)

    constraints = [
        cp.sum(P, axis=1) == a,
        cp.sum(P, axis=0) == b,
    ]
    # Supermartingale: sum_j y_j P[i,j] <= x_i * a_i
    for i in range(n):
        constraints.append(P[i, :] @ y <= x[i] * a[i])

    if cost is None:
        obj_expr = cp.Constant(0)
    else:
        C = np.asarray(cost, dtype=float)
        obj_expr = cp.sum(cp.multiply(C, P))

    obj = cp.Minimize(obj_expr) if minimize else cp.Maximize(obj_expr)
    prob = cp.Problem(obj, constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        # Try fallback solver
        try:
            prob.solve(solver="SCS", verbose=False)
        except Exception:
            return None

    if P.value is None or prob.status in ("infeasible", "unbounded"):
        return None

    Pval = np.clip(P.value, 0.0, None)
    # Renormalize rows to exact marginals
    row_sums = Pval.sum(axis=1)
    for i in range(n):
        if row_sums[i] > 1e-14:
            Pval[i, :] *= a[i] / row_sums[i]
    # Renormalize cols
    col_sums = Pval.sum(axis=0)
    for j in range(m):
        if col_sums[j] > 1e-14:
            Pval[:, j] *= b[j] / col_sums[j]

    return DiscreteCoupling(mu, nu, Pval)


def solve_martingale_coupling(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    cost: np.ndarray | None = None,
    minimize: bool = True,
    solver: str = "CLARABEL",
) -> DiscreteCoupling | None:
    """Solve a martingale OT problem (equality version of the supermartingale).

    Constraints: same as supermartingale but sum_j y_j P[i,j] == x_i * a_i.
    """
    n, m = mu.n, nu.n
    a = mu.weights.astype(float)
    b = nu.weights.astype(float)
    x = mu.locations
    y = nu.locations

    P = cp.Variable((n, m), nonneg=True)

    constraints = [
        cp.sum(P, axis=1) == a,
        cp.sum(P, axis=0) == b,
    ]
    for i in range(n):
        constraints.append(P[i, :] @ y == x[i] * a[i])

    if cost is None:
        obj_expr = cp.Constant(0)
    else:
        C = np.asarray(cost, dtype=float)
        obj_expr = cp.sum(cp.multiply(C, P))

    obj = cp.Minimize(obj_expr) if minimize else cp.Maximize(obj_expr)
    prob = cp.Problem(obj, constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        try:
            prob.solve(solver="SCS", verbose=False)
        except Exception:
            return None

    if P.value is None or prob.status in ("infeasible", "unbounded"):
        return None

    Pval = np.clip(P.value, 0.0, None)
    row_sums = Pval.sum(axis=1)
    for i in range(n):
        if row_sums[i] > 1e-14:
            Pval[i, :] *= a[i] / row_sums[i]

    return DiscreteCoupling(mu, nu, Pval)


def diagonal_coupling(mu: AtomicMeasure) -> DiscreteCoupling:
    """Return the diagonal coupling id_mu of mu with itself (X = Y a.s.).

    pi = sum_i a_i delta_{(x_i, x_i)}.
    """
    P = np.diag(mu.weights)
    return DiscreteCoupling(mu, mu, P)


def adapted_wasserstein_1(
    pi: DiscreteCoupling,
    pi_prime: DiscreteCoupling,
    tol: float = 1e-10,
) -> float:
    """Compute AW_1(pi, pi') between two discrete couplings.

    AW_1(pi, pi') = min_{q in Π(mu, mu')} sum_{i,i'} q[i,i'] * d(i,i')
    where d(i,i') = |x_i - x'_{i'}| + W_1(pi_{x_i}, pi'_{x'_{i'}}).

    Steps:
      1. Disintegrate each coupling into conditional laws.
      2. Compute W_1 between all pairs of conditional laws.
      3. Solve outer OT over x-supports.

    Custom — Context7 found no library equivalent (paper Definition, adapted W).
    """
    # Kernels: list of (n_nu,) vectors
    mu = pi.mu
    mu_p = pi_prime.mu

    laws = pi.conditional_laws  # list of n_mu vectors over nu.locations
    laws_p = pi_prime.conditional_laws  # list of n_mu' vectors over nu'.locations

    # Inner Wasserstein costs
    inner_cost = np.zeros((mu.n, mu_p.n))
    for i in range(mu.n):
        for ip in range(mu_p.n):
            # W_1 between laws[i] (over pi.nu.locations) and laws_p[ip] (over pi'.nu.locations)
            a_law = np.asarray(laws[i], dtype=float)
            b_law = np.asarray(laws_p[ip], dtype=float)
            # Make them proper distributions
            if a_law.sum() < tol or b_law.sum() < tol:
                inner_cost[i, ip] = 0.0
                continue
            # Cost matrix between nu locations and nu' locations
            M_inner = np.abs(
                pi.nu.locations[:, None] - pi_prime.nu.locations[None, :]
            ).astype(float)
            a_law = a_law / a_law.sum()
            b_law = b_law / b_law.sum()
            w1_inner = ot.emd2(a_law, b_law, M_inner)
            inner_cost[i, ip] = w1_inner

    # Outer cost: |x_i - x'_{i'}| + W_1 inner
    outer_cost = (
        np.abs(mu.locations[:, None] - mu_p.locations[None, :]) + inner_cost
    ).astype(float)

    a_outer = mu.weights.astype(float)
    b_outer = mu_p.weights.astype(float)

    aw = ot.emd2(a_outer, b_outer, outer_cost)
    return float(aw)


def compose_couplings(
    pi1: DiscreteCoupling,
    pi2: DiscreteCoupling,
    tol: float = 1e-10,
) -> DiscreteCoupling:
    """Compose pi1 in Pi(mu, lambda) and pi2 in Pi(lambda, nu) on the second coordinate.

    Result: pi(dx, dy) = int pi1(dx, dz) pi2_z(dy) in Pi(mu, nu).

    Composition formula: P_composed[i,j] = sum_k P1[i,k] * P2[k,j] / lambda_k.

    Custom — Context7 found no library equivalent.
    """
    mu = pi1.mu
    lam = pi1.nu
    nu = pi2.nu

    assert lam.n == pi2.mu.n, "Intermediate marginals must match."
    assert np.allclose(lam.weights, pi2.mu.weights, atol=tol), (
        "Second marginal of pi1 must match first marginal of pi2."
    )

    P1 = pi1.P  # shape (n_mu, n_lam)
    P2 = pi2.P  # shape (n_lam, n_nu)
    lam_w = lam.weights

    # Disintegrate pi2: kernel pi2_z[k, j] = P2[k, j] / lam_k
    P2_kernel = np.zeros_like(P2)
    for k in range(lam.n):
        if lam_w[k] > tol:
            P2_kernel[k, :] = P2[k, :] / lam_w[k]

    # Compose: P[i,j] = sum_k P1[i,k] * P2_kernel[k,j]
    P_comp = P1 @ P2_kernel  # (n_mu, n_nu)

    return DiscreteCoupling(mu, nu, P_comp)


def independent_coupling(mu: AtomicMeasure, nu: AtomicMeasure) -> DiscreteCoupling:
    """Return the product coupling mu x nu (P[i,j] = a_i * b_j)."""
    P = np.outer(mu.weights, nu.weights)
    return DiscreteCoupling(mu, nu, P)


def mixture_coupling(
    pi1: DiscreteCoupling,
    pi2: DiscreteCoupling,
    alpha: float,
    mu_target: AtomicMeasure,
    nu_target: AtomicMeasure,
) -> DiscreteCoupling:
    """Form the mixture alpha * pi1 + (1-alpha) * pi2 on (mu_target, nu_target).

    Both pi1 and pi2 must have the same first marginal mu_target and second marginal nu_target.
    The coupling matrices are interpolated.
    """
    assert 0.0 <= alpha <= 1.0
    P = alpha * pi1.P + (1.0 - alpha) * pi2.P
    return DiscreteCoupling(mu_target, nu_target, P)


def weak_cost_objective(
    pi: DiscreteCoupling,
    cost_fn,
) -> float:
    """Evaluate a weak cost C(x, m) over a discrete coupling.

    Parameters
    ----------
    pi : DiscreteCoupling
    cost_fn : callable(x: float, p: np.ndarray, y_locs: np.ndarray) -> float
        Takes source atom x, conditional probability vector p over nu.locations,
        and nu.locations array; returns C(x, m).

    Returns
    -------
    float : sum_i a_i C(x_i, m_i).
    """
    total = 0.0
    laws = pi.conditional_laws
    for i in range(pi.mu.n):
        ai = pi.mu.weights[i]
        xi = pi.mu.locations[i]
        if ai > 0:
            total += ai * cost_fn(xi, laws[i], pi.nu.locations)
    return total
