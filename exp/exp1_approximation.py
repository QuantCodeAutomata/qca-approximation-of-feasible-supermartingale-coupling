"""
Experiment 1: Approximation of Feasible Supermartingale Couplings (Theorem 2.1).

Constructs three test families:
  Family A: strict irreducible (I_0 dominant)
  Family B: mixed-component (diagonal + martingale + strict)
  Family C: pure martingale limit

For each family:
  - Defines target marginals mu, nu with mu <=_{cd} nu
  - Defines target coupling pi in Pi_S(mu, nu)
  - Constructs perturbation sequence (mu^k, nu^k)
  - Runs componentwise approximation and measures AW_1 convergence

Results saved to results/exp1_*.{png,csv}
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.measures import AtomicMeasure, wasserstein_1, check_decreasing_convex_order
from src.couplings import (
    DiscreteCoupling,
    solve_supermartingale_coupling,
    diagonal_coupling,
    adapted_wasserstein_1,
    independent_coupling,
)
from src.decomposition import compute_irreducible_decomposition
from src.approximation import (
    construct_perturbation_sequence,
    run_approximation_experiment,
    ApproximationResult,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ───────────────────────── Family A: Strict irreducible ──────────────────────

def family_a_strict() -> tuple[AtomicMeasure, AtomicMeasure, DiscreteCoupling]:
    """Family A: mu supported on (ell, inf) with defect Delta = bar(mu) - bar(nu) > 0.

    mu = 0.5 delta_1 + 0.5 delta_3   (mean = 2)
    nu = 0.4 delta_0 + 0.3 delta_1.5 + 0.3 delta_3  (mean = 1.55)
    Delta = 2 - 1.55 = 0.45 > 0  → strict supermartingale.
    """
    mu = AtomicMeasure([1.0, 3.0], [0.5, 0.5])
    nu = AtomicMeasure([0.0, 1.5, 3.0], [0.4, 0.3, 0.3])

    # Verify mu <=_{cd} nu
    feasible, violation = check_decreasing_convex_order(mu, nu)
    if not feasible:
        raise ValueError(f"Family A: mu <=cd nu violated, violation={violation:.3e}")

    # Build a target supermartingale coupling
    pi = solve_supermartingale_coupling(mu, nu, cost=None)
    if pi is None:
        # Fallback: manual coupling
        # P[0,:] = [0.3, 0.1, 0.1] -> E[Y|X=1] = (0*0.3+1.5*0.1+3*0.1)/0.5 = 0.9
        # P[1,:] = [0.1, 0.2, 0.2] -> E[Y|X=3] = (0*0.1+1.5*0.2+3*0.2)/0.5 = 1.8
        P_manual = np.array([[0.3, 0.1, 0.1], [0.1, 0.2, 0.2]])
        pi = DiscreteCoupling(mu, nu, P_manual)

    return mu, nu, pi


# ───────────────────────── Family B: Mixed components ────────────────────────

def family_b_mixed() -> tuple[AtomicMeasure, AtomicMeasure, DiscreteCoupling]:
    """Family B: provably feasible mixture of diagonal, martingale, and strict components.

    Construction via component mixing (α=0.3):
      mu_mart  = 0.5 δ_{-1} + 0.5 δ_{1}                    (mean 0, martingale component)
      nu_mart  = 0.25 δ_{-2} + 0.5 δ_{0} + 0.25 δ_{2}      (mean 0, convex dilation)
      mu_strict = δ_{5}                                       (mean 5, strict SM atom)
      nu_strict = 0.6 δ_{4} + 0.4 δ_{6}                     (mean 4.8 < 5)

    mu = 0.3 mu_mart + 0.7 mu_strict  → atoms at {-1,1,5}, mean 3.5
    nu = 0.3 nu_mart + 0.7 nu_strict  → atoms at {-2,0,2,4,6}, mean 3.36

    Analytical D(x) = P_nu(x) - P_mu(x):
      D = 0   on (-∞, 0], [2, 4]   → diagonal region I_{-1}
      D > 0   on (0, 2)             → martingale component I_1
      D > 0   on (4, ∞)             → strict SM region I_0
    x* = 4, so mu atom -1 is in I_{-1}, atom 1 in I_1, atom 5 in I_0.
    """
    α = 0.3
    mu = AtomicMeasure(
        [-1.0, 1.0, 5.0],
        [α * 0.5, α * 0.5, 1.0 - α],  # [0.15, 0.15, 0.70]
    )
    nu = AtomicMeasure(
        [-2.0, 0.0, 2.0, 4.0, 6.0],
        [α * 0.25, α * 0.5, α * 0.25, (1 - α) * 0.6, (1 - α) * 0.4],
        # [0.075, 0.150, 0.075, 0.420, 0.280]
    )

    feasible, violation = check_decreasing_convex_order(mu, nu)
    if not feasible:
        raise ValueError(
            f"Family B construction failed: violation={violation:.3e}. "
            "Check component weights."
        )

    pi = solve_supermartingale_coupling(mu, nu, cost=None)
    if pi is None:
        pi = independent_coupling(mu, nu)

    return mu, nu, pi


# ───────────────────────── Family C: Pure martingale ─────────────────────────

def family_c_martingale() -> tuple[AtomicMeasure, AtomicMeasure, DiscreteCoupling]:
    """Family C: martingale transport (Delta = 0, mu <=_{cx} nu in convex order).

    mu = 0.5 delta_{-1} + 0.5 delta_1   (mean = 0)
    nu = 0.25 delta_{-2} + 0.5 delta_0 + 0.25 delta_2  (mean = 0)
    This is a classic martingale transport instance.
    """
    mu = AtomicMeasure([-1.0, 1.0], [0.5, 0.5])
    nu = AtomicMeasure([-2.0, 0.0, 2.0], [0.25, 0.5, 0.25])

    feasible, violation = check_decreasing_convex_order(mu, nu)
    if not feasible:
        nu = AtomicMeasure(nu.locations + violation + 0.01, nu.weights)

    pi = solve_supermartingale_coupling(mu, nu, cost=None)
    if pi is None:
        pi = independent_coupling(mu, nu)

    return mu, nu, pi


# ─────────────────────────── Plotting helpers ────────────────────────────────

def plot_put_functions(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    title: str = "Put Functions",
    save_path: str | None = None,
) -> None:
    """Plot P_mu and P_nu to visualise <=_{cd} order."""
    all_pts = np.sort(np.concatenate([mu.locations, nu.locations]))
    xmin, xmax = all_pts.min() - 1.0, all_pts.max() + 1.0
    grid = np.linspace(xmin, xmax, 500)

    p_mu = mu.put_function(grid)
    p_nu = nu.put_function(grid)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, p_mu, label=r"$P_\mu(x)$", color="steelblue", linewidth=2)
    ax.plot(grid, p_nu, label=r"$P_\nu(x)$", color="darkorange", linewidth=2)
    ax.fill_between(grid, p_mu, p_nu, alpha=0.15, color="green",
                    label=r"$D(x) = P_\nu - P_\mu \geq 0$")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Put function value", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_aw1_convergence(
    results_list: list[list[ApproximationResult]],
    family_names: list[str],
    save_path: str | None = None,
) -> None:
    """Plot AW_1(pi^k, pi) vs k for all families."""
    fig, axes = plt.subplots(1, len(results_list), figsize=(5 * len(results_list), 4))
    if len(results_list) == 1:
        axes = [axes]

    for ax, results, name in zip(axes, results_list, family_names):
        ks = [r.k for r in results]
        aw1s = [r.aw1 for r in results]
        w1_mus = [r.w1_mu for r in results]
        w1_nus = [r.w1_nu for r in results]

        ax.plot(ks, aw1s, "o-", label=r"$AW_1(\pi^k, \pi)$", color="navy", linewidth=2)
        ax.plot(ks, w1_mus, "s--", label=r"$W_1(\mu^k, \mu)$", color="steelblue", linewidth=1.5)
        ax.plot(ks, w1_nus, "^--", label=r"$W_1(\nu^k, \nu)$", color="darkorange", linewidth=1.5)

        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel("Distance", fontsize=11)
        ax.set_title(f"Family {name}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.suptitle(
        r"AW$_1$ Convergence — Approximation of Feasible SM Couplings (Thm 2.1)",
        fontsize=13
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_decomposition(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    title: str = "Irreducible Decomposition",
    save_path: str | None = None,
) -> None:
    """Plot D(x) and mark irreducible regions."""
    from src.decomposition import compute_irreducible_decomposition

    decomp = compute_irreducible_decomposition(mu, nu)
    grid = decomp.grid
    D = decomp.D

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, D, color="navy", linewidth=2, label=r"$D(x) = P_\nu - P_\mu$")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(decomp.x_star, color="red", linestyle="--",
               label=f"$x^* = {decomp.x_star:.2f}$")

    ax.fill_between(grid, 0, D, where=D > 0, alpha=0.2, color="green",
                    label="Martingale region")
    ax.fill_between(grid, 0, D, where=grid > decomp.x_star, alpha=0.2, color="orange",
                    label=r"Strict SM region $I_0$")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel(r"$D(x)$", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────── Main experiment ─────────────────────────────────

def run_experiment_1() -> pd.DataFrame:
    """Run the full Experiment 1 and return results table."""
    print("=" * 60)
    print("Experiment 1: Approximation of Feasible SM Couplings")
    print("=" * 60)

    families = [
        ("A_strict", family_a_strict),
        ("B_mixed", family_b_mixed),
        ("C_martingale", family_c_martingale),
    ]

    K = 10  # perturbation steps
    all_results_list = []
    all_names = []
    rows = []

    for fname, factory in families:
        print(f"\nFamily {fname}:")
        mu, nu, pi = factory()

        # Verify feasibility
        feas, viol = check_decreasing_convex_order(mu, nu)
        print(f"  mu <=cd nu: {feas}, violation={viol:.2e}")
        print(f"  pi SMC residual: {pi.check_supermartingale():.2e}")
        print(f"  pi m1 error: {pi.check_first_marginal():.2e}")
        print(f"  pi m2 error: {pi.check_second_marginal():.2e}")

        # Decomposition
        decomp = compute_irreducible_decomposition(mu, nu)
        print(f"  x* = {decomp.x_star:.4f}")
        print(f"  martingale components: {len(decomp.martingale_components)}")

        # Plot put functions
        plot_put_functions(
            mu, nu,
            title=f"Put Functions — Family {fname}",
            save_path=os.path.join(RESULTS_DIR, f"exp1_put_functions_{fname}.png"),
        )

        # Plot decomposition
        plot_decomposition(
            mu, nu,
            title=f"Irreducible Decomposition — Family {fname}",
            save_path=os.path.join(RESULTS_DIR, f"exp1_decomposition_{fname}.png"),
        )

        # Perturbation sequence
        seq = construct_perturbation_sequence(mu, nu, K=K, delta_max=0.5)

        # Run approximation
        print(f"  Running {K}-step approximation...")
        results = run_approximation_experiment(mu, nu, pi, seq, verbose=True)
        all_results_list.append(results)
        all_names.append(fname)

        # Collect rows
        for r in results:
            rows.append({
                "family": fname,
                "k": r.k,
                "AW1": r.aw1,
                "W1_mu": r.w1_mu,
                "W1_nu": r.w1_nu,
                "feas_residual": r.feasibility_residual,
                "put_residual": r.put_residual,
                "m1_error": r.marginal_errors[0],
                "m2_error": r.marginal_errors[1],
            })

    # Plot AW1 convergence
    plot_aw1_convergence(
        all_results_list,
        all_names,
        save_path=os.path.join(RESULTS_DIR, "exp1_aw1_convergence.png"),
    )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp1_results.csv"), index=False)

    # Print summary
    print("\n--- Summary: AW_1 convergence ---")
    for fname in [f[0] for f in families]:
        sub = df[df.family == fname]
        print(f"  Family {fname}: AW1(k=1)={sub.AW1.iloc[0]:.4f} -> "
              f"AW1(k=K)={sub.AW1.iloc[-1]:.4f}")

    return df


if __name__ == "__main__":
    df = run_experiment_1()
    print("\nExperiment 1 complete. Results saved to results/")
