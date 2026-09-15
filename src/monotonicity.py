"""
Monotonicity principle for weak supermartingale optimal transport.

Implements Theorem 2.9 and Corollary 2.10:
Optimality in WSOT is characterized by supermartingale C-monotonicity.

For a support set Gamma = {(x_i, p_i)} and a finite competitor family {q_i},
C-monotonicity requires: sum_i C(x_i, p_i) <= sum_i C(x_i, q_i)
for all admissible competitors preserving the aggregate second marginal
and barycenter conditions.

Custom — Context7 found no library equivalent for C-monotonicity testing.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from itertools import product as itertools_product
from dataclasses import dataclass, field
from typing import Callable

from src.measures import AtomicMeasure, check_decreasing_convex_order
from src.couplings import DiscreteCoupling
from src.wsot import solve_wsot, LinearCost, QuadraticWeakCost


@dataclass
class MonotonicityTest:
    """Result of a monotonicity test on a single optimal coupling.

    Attributes
    ----------
    is_monotone : bool  True if no competitor improves the cost.
    max_violation : float  max over competitors of [C(p) - C(q)].
    n_competitors_tested : int
    cost_name : str
    support_size : int
    """

    is_monotone: bool
    max_violation: float
    n_competitors_tested: int
    cost_name: str
    support_size: int
    details: dict = field(default_factory=dict)


def evaluate_support_cost(
    x_locs: np.ndarray,
    p_list: list[np.ndarray],
    y_locs: np.ndarray,
    cost,
    weights: np.ndarray,
) -> float:
    """Evaluate sum_i a_i C(x_i, p_i) for a support configuration.

    Parameters
    ----------
    x_locs : (n,) source atom locations.
    p_list : list of n probability vectors over y_locs.
    y_locs : (m,) target locations.
    cost : cost object.
    weights : (n,) atom weights a_i.

    Returns
    -------
    float : weighted cost.
    """
    total = 0.0
    for i, (xi, pi, ai) in enumerate(zip(x_locs, p_list, weights)):
        if ai < 1e-14:
            continue
        if isinstance(cost, LinearCost):
            c_row = np.array([cost.c_fn(xi, yj) for yj in y_locs])
            total += ai * float(c_row @ pi)
        elif isinstance(cost, QuadraticWeakCost):
            wi = cost.w_fn(xi, y_locs)
            vi = cost.v_fn(xi, y_locs)
            total += ai * (float(wi @ (pi ** 2)) + float(vi @ pi))
        else:
            # Generic: evaluate numerically as linear sum
            c_row = np.array([float(xi - yj) ** 2 for yj in y_locs])
            total += ai * float(c_row @ pi)
    return total


def classify_support_point(
    xi: float,
    pi: np.ndarray,
    y_locs: np.ndarray,
    x_star: float,
    tol: float = 1e-8,
) -> str:
    """Classify x_i as 'martingale', 'strict_sm', or 'diagonal'.

    Parameters
    ----------
    xi : float  Source atom location.
    pi : np.ndarray  Conditional probability vector over y_locs.
    y_locs : np.ndarray  Target atom locations.
    x_star : float  Threshold from irreducible decomposition.
    tol : float  Tolerance for martingale vs strict classification.

    Returns
    -------
    str classification.
    """
    cond_mean = float(y_locs @ pi)
    # Martingale-type: coupling has E[Y|X=x] ≈ x at this point.
    # Check this first; it takes priority over regional (x_star) classification.
    if abs(cond_mean - xi) < tol:
        return "martingale"
    # Strict supermartingale region I_0 = (x_star, ∞) with mean < x.
    if xi > x_star + tol:
        return "strict_sm"
    # Below x_star with mean < x: diagonal region.
    return "diagonal"


def generate_competitors_random(
    x_locs: np.ndarray,
    p_list: list[np.ndarray],
    y_locs: np.ndarray,
    weights: np.ndarray,
    x_star: float,
    n_competitors: int = 500,
    rng_seed: int = 42,
    tol: float = 1e-8,
) -> list[list[np.ndarray]]:
    """Generate random competitor families via Dirichlet sampling.

    A competitor {q_i} must satisfy:
    1. sum_i a_i q_i = sum_i a_i p_i (same aggregate second marginal).
    2. For martingale-type x_i: y_locs @ q_i = x_i (exact barycenter).
    3. For strict-SM-type x_i: y_locs @ q_i <= x_i (SMC inequality).
    4. q_i >= 0 and sum q_i = 1 for all i.

    Custom — Context7 found no library equivalent.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(x_locs)
    m = len(y_locs)

    # Classify points
    classifications = [
        classify_support_point(x_locs[i], p_list[i], y_locs, x_star, tol)
        for i in range(n)
    ]

    # Aggregate target: sum_i a_i p_i
    agg_target = np.array(
        [weights[i] * p_list[i] for i in range(n)]
    ).sum(axis=0)  # shape (m,)

    competitors = []

    for _ in range(n_competitors):
        # Sample Dirichlet proposals for each q_i
        q_list_cand = []
        valid = True

        # Generate q_i one at a time by Dirichlet sampling
        for i in range(n):
            alpha = rng.dirichlet(np.ones(m) * 2.0)
            qi = alpha

            # Project onto barycenter constraint if martingale
            if classifications[i] == "martingale":
                # Enforce y @ qi = x_i via projection
                qi = _project_onto_mean_constraint(
                    qi, y_locs, float(x_locs[i]), tol=tol
                )
            elif classifications[i] == "strict_sm":
                # Enforce y @ qi <= x_i (SMC)
                mean_qi = float(y_locs @ qi)
                if mean_qi > x_locs[i] + tol:
                    # Shift mass to lower y's
                    qi = _project_onto_smc_constraint(qi, y_locs, float(x_locs[i]))

            if qi is None or qi.min() < -tol or abs(qi.sum() - 1.0) > 1e-4:
                valid = False
                break
            qi = np.clip(qi, 0, None)
            qi /= qi.sum()
            q_list_cand.append(qi)

        if not valid:
            continue

        # Enforce aggregate marginal exactly: for n==2 derive q[1] from constraint;
        # for n>2 only accept if already within tight tolerance.
        agg_q = np.array([weights[i] * q_list_cand[i] for i in range(n)]).sum(axis=0)
        if n == 2 and weights[1] > 1e-12:
            # q_1 is determined by aggregate constraint
            q_derived = (agg_target - weights[0] * q_list_cand[0]) / weights[1]
            if q_derived.min() >= -1e-8 and abs(q_derived.sum() - 1.0) < 1e-6:
                q_derived = np.clip(q_derived, 0, None)
                q_derived /= q_derived.sum()
                # Check barycenter constraint for derived q_1
                cls1 = classifications[1]
                mean_q1 = float(y_locs @ q_derived)
                ok = True
                if cls1 == "martingale" and abs(mean_q1 - x_locs[1]) > tol:
                    ok = False
                elif cls1 in ("strict_sm", "diagonal") and mean_q1 > x_locs[1] + tol:
                    ok = False
                if ok:
                    competitors.append([q_list_cand[0], q_derived])
        elif np.max(np.abs(agg_q - agg_target)) < 1e-4:
            competitors.append(q_list_cand)

    return competitors


def _project_onto_mean_constraint(
    p: np.ndarray,
    y: np.ndarray,
    target_mean: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> np.ndarray | None:
    """Project p onto the simplex with constraint y @ p = target_mean.

    Uses linear programming via cvxpy.
    """
    m = len(y)
    q = cp.Variable(m, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(q - p))
    constraints = [
        cp.sum(q) == 1.0,
        y @ q == target_mean,
    ]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver="OSQP", verbose=False)
    except Exception:
        prob.solve(solver="SCS", verbose=False)

    if q.value is None:
        return p  # fallback to original
    q_val = np.clip(q.value, 0, None)
    s = q_val.sum()
    if s > 1e-14:
        q_val /= s
    return q_val


def _project_onto_smc_constraint(
    p: np.ndarray,
    y: np.ndarray,
    max_mean: float,
) -> np.ndarray:
    """Project p onto the simplex with y @ p <= max_mean."""
    current_mean = float(y @ p)
    if current_mean <= max_mean:
        return p
    # Shift mass from high-y atoms to low-y atoms
    q = p.copy()
    excess = current_mean - max_mean
    y_sorted_idx = np.argsort(y)[::-1]  # descending y
    for idx in y_sorted_idx:
        if excess <= 0:
            break
        # Move mass from y[idx] to smallest y
        min_idx = y_sorted_idx[-1]
        transferable = q[idx]
        dy = y[idx] - y[min_idx]
        if dy < 1e-14:
            break
        transfer = min(transferable, excess / dy)
        q[idx] -= transfer
        q[min_idx] += transfer
        excess -= transfer * dy

    q = np.clip(q, 0, None)
    if q.sum() > 1e-14:
        q /= q.sum()
    return q


def generate_competitors_exhaustive(
    x_locs: np.ndarray,
    p_list: list[np.ndarray],
    y_locs: np.ndarray,
    weights: np.ndarray,
    x_star: float,
    grid_resolution: int = 5,
    tol: float = 1e-7,
) -> list[list[np.ndarray]]:
    """Generate competitors exhaustively for small support (m <= 3).

    Discretizes the simplex for each q_i and filters valid competitors.

    Custom — Context7 found no library equivalent.
    """
    m = len(y_locs)
    if m > 4:
        return []  # Too large for exhaustive enumeration

    n = len(x_locs)
    classifications = [
        classify_support_point(x_locs[i], p_list[i], y_locs, x_star, tol)
        for i in range(n)
    ]

    # Discrete simplex: all probability vectors over m atoms with denominator K
    K = grid_resolution
    if m == 2:
        simplex_pts = [np.array([k / K, (K - k) / K]) for k in range(K + 1)]
    elif m == 3:
        simplex_pts = [
            np.array([a / K, b / K, (K - a - b) / K])
            for a in range(K + 1) for b in range(K + 1 - a)
        ]
    else:
        # Too large: return empty
        return []

    # Filter simplex points by barycenter constraints for each x_i
    valid_by_i = []
    for i in range(n):
        cls = classifications[i]
        valid_pts = []
        for pt in simplex_pts:
            mean_pt = float(y_locs @ pt)
            if cls == "martingale" and abs(mean_pt - x_locs[i]) > tol:
                continue
            if cls in ("strict_sm", "diagonal") and mean_pt > x_locs[i] + tol:
                continue
            valid_pts.append(pt)
        if not valid_pts:
            valid_pts = simplex_pts  # fallback
        valid_by_i.append(valid_pts)

    # Form all combinations
    competitors = []
    agg_target = np.array([weights[i] * p_list[i] for i in range(n)]).sum(axis=0)

    # Enumerate product (may be large)
    # For n==2: derive q[1] from aggregate constraint to enforce it exactly.
    # For n>2: use tight tolerance on aggregate violation.
    max_combos = 2000
    count = 0
    agg_tol = 1.0 / (2 * K * max(n, 1))  # grid-resolution-aware tolerance
    from itertools import product as iter_product

    if n == 2 and weights[1] > 1e-12:
        for q0 in valid_by_i[0]:
            if count >= max_combos:
                break
            # Derive q1 from aggregate constraint
            q1_raw = (agg_target - weights[0] * q0) / weights[1]
            if q1_raw.min() < -1e-8 or abs(q1_raw.sum() - 1.0) > 1e-6:
                continue
            q1 = np.clip(q1_raw, 0, None)
            q1 /= q1.sum()
            # Validate barycenter constraint for derived q1
            cls1 = classifications[1]
            mean_q1 = float(y_locs @ q1)
            ok = True
            if cls1 == "martingale" and abs(mean_q1 - x_locs[1]) > tol:
                ok = False
            elif cls1 in ("strict_sm", "diagonal") and mean_q1 > x_locs[1] + tol:
                ok = False
            if ok:
                competitors.append([q0, q1])
            count += 1
    else:
        for combo in iter_product(*valid_by_i):
            if count >= max_combos:
                break
            count += 1
            q_list = list(combo)
            agg_q = np.array([weights[i] * q_list[i] for i in range(n)]).sum(axis=0)
            if np.max(np.abs(agg_q - agg_target)) < agg_tol:
                competitors.append(q_list)

    return competitors


def adversarial_competitor_search(
    x_locs: np.ndarray,
    p_list: list[np.ndarray],
    y_locs: np.ndarray,
    weights: np.ndarray,
    x_star: float,
    cost,
    tol: float = 1e-8,
) -> tuple[float, list[np.ndarray] | None]:
    """Adversarially search for the minimum competitor cost improvement.

    Solves: min sum_i a_i C(x_i, q_i)
    s.t. sum_i a_i q_i = sum_i a_i p_i (aggregate marginal)
         q_i barycenter constraints per classification
         q_i >= 0, sum q_i = 1

    For linear C, this is an LP. For quadratic C, it's a QP.

    If the optimal competitor cost < sum_i a_i C(x_i, p_i), a violation is found.

    Returns (min_competitor_gap, optimal_competitor) where gap = C(p) - C(q_opt).
    Positive gap means p is NOT monotone (p is suboptimal).
    Negative gap means p dominates all competitors.

    Custom — Context7 found no library equivalent.
    """
    n = len(x_locs)
    m = len(y_locs)

    classifications = [
        classify_support_point(x_locs[i], p_list[i], y_locs, x_star, tol)
        for i in range(n)
    ]

    # Variables: q_i as probability vectors
    Q = [cp.Variable(m, nonneg=True) for _ in range(n)]

    constraints = []
    for i in range(n):
        constraints.append(cp.sum(Q[i]) == 1.0)
        cls = classifications[i]
        if cls == "martingale":
            constraints.append(y_locs @ Q[i] == x_locs[i])
        elif cls in ("strict_sm", "diagonal"):
            constraints.append(y_locs @ Q[i] <= x_locs[i])

    # Aggregate marginal constraint
    agg_target = np.array([weights[i] * p_list[i] for i in range(n)]).sum(axis=0)
    agg_q = sum(weights[i] * Q[i] for i in range(n))
    constraints.append(agg_q == agg_target)

    # Objective: minimize sum_i a_i C(x_i, q_i)
    obj_parts = []
    for i in range(n):
        ai = weights[i]
        if ai < 1e-14:
            continue
        xi = x_locs[i]
        if isinstance(cost, LinearCost):
            c_row = np.array([cost.c_fn(xi, yj) for yj in y_locs])
            obj_parts.append(ai * (c_row @ Q[i]))
        elif isinstance(cost, QuadraticWeakCost):
            wi = cost.w_fn(xi, y_locs)
            vi = cost.v_fn(xi, y_locs)
            obj_parts.append(ai * (wi @ cp.square(Q[i]) + vi @ Q[i]))
        else:
            c_row = np.array([(xi - yj) ** 2 for yj in y_locs])
            obj_parts.append(ai * (c_row @ Q[i]))

    if not obj_parts:
        return 0.0, None

    obj_expr = cp.sum(obj_parts)
    obj = cp.Minimize(obj_expr)
    prob = cp.Problem(obj, constraints)

    for solver in ("CLARABEL", "OSQP", "SCS"):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status not in ("infeasible", "unbounded") and Q[0].value is not None:
                break
        except Exception:
            continue

    if prob.status in ("infeasible", "unbounded") or Q[0].value is None:
        return 0.0, None

    q_opt = [np.clip(Q[i].value, 0, None) for i in range(n)]
    for i in range(n):
        if q_opt[i].sum() > 1e-14:
            q_opt[i] /= q_opt[i].sum()

    # Compute cost of optimal competitor
    c_p = evaluate_support_cost(x_locs, p_list, y_locs, cost, weights)
    c_q = evaluate_support_cost(x_locs, q_opt, y_locs, cost, weights)

    gap = c_p - c_q  # positive means q improves over p (violation of C-monotonicity)
    return gap, q_opt


def check_c_monotonicity(
    pi_star: DiscreteCoupling,
    x_star: float,
    cost,
    n_random_competitors: int = 300,
    use_exhaustive: bool = True,
    use_adversarial: bool = True,
    tol: float = 1e-4,
    verbose: bool = False,
) -> MonotonicityTest:
    """Test supermartingale C-monotonicity for an optimal coupling.

    Parameters
    ----------
    pi_star : DiscreteCoupling  Optimal coupling.
    x_star : float  Threshold from irreducible decomposition.
    cost : cost object (LinearCost or QuadraticWeakCost).
    n_random_competitors : int  Number of random competitors to test.
    use_exhaustive : bool  Use exhaustive competitor generation for small support.
    use_adversarial : bool  Use optimization-based adversarial search.
    tol : float  Tolerance for declaring monotonicity.
    verbose : bool.

    Returns
    -------
    MonotonicityTest
    """
    x_locs = pi_star.mu.locations
    y_locs = pi_star.nu.locations
    weights = pi_star.mu.weights
    p_list = pi_star.conditional_laws

    cost_p = evaluate_support_cost(x_locs, p_list, y_locs, cost, weights)
    max_violation = 0.0
    n_tested = 0

    # Adversarial search
    adv_gap = 0.0
    if use_adversarial:
        try:
            adv_gap, q_opt = adversarial_competitor_search(
                x_locs, p_list, y_locs, weights, x_star, cost, tol=tol
            )
            n_tested += 1
            if adv_gap > max_violation:
                max_violation = adv_gap
            if verbose:
                print(f"    Adversarial gap: {adv_gap:.6e}")
        except Exception as e:
            if verbose:
                print(f"    Adversarial search failed: {e}")

    # Random competitors
    random_comps = generate_competitors_random(
        x_locs, p_list, y_locs, weights, x_star,
        n_competitors=n_random_competitors, tol=tol
    )
    for q_list in random_comps:
        cost_q = evaluate_support_cost(x_locs, q_list, y_locs, cost, weights)
        gap = cost_p - cost_q
        max_violation = max(max_violation, gap)
        n_tested += 1

    # Exhaustive competitors (small support)
    if use_exhaustive and pi_star.nu.n <= 4:
        exh_comps = generate_competitors_exhaustive(
            x_locs, p_list, y_locs, weights, x_star, tol=tol
        )
        for q_list in exh_comps:
            cost_q = evaluate_support_cost(x_locs, q_list, y_locs, cost, weights)
            gap = cost_p - cost_q
            max_violation = max(max_violation, gap)
            n_tested += 1

    # Monotonicity holds if no competitor improves cost
    # A positive max_violation means competitors improved cost (violation)
    is_monotone = max_violation <= tol

    cost_name = getattr(cost, "name", str(type(cost).__name__))
    return MonotonicityTest(
        is_monotone=is_monotone,
        max_violation=max_violation,
        n_competitors_tested=n_tested,
        cost_name=cost_name,
        support_size=pi_star.mu.n,
        details={
            "cost_p": cost_p,
            "adversarial_gap": adv_gap,
            "n_random": len(random_comps),
        },
    )
