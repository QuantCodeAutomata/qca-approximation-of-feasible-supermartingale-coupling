"""
Experiment 3: Monotonicity Principle for WSOT (Theorem 2.9, Corollary 2.10).

For small finite-support instances:
  - Solves WSOT to find optimal couplings pi*
  - Constructs competitor families {q_i}
  - Verifies C-monotonicity: sum_i C(x_i, p_i) <= sum_i C(x_i, q_i)
  - Tests adversarial search for minimum competitor gap
  - Specialises to classical SOT (linear cost) as a sanity check
  - Deliberately perturbs optimal couplings and checks monotonicity violation

Results saved to results/exp3_*.{png,csv}
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.measures import AtomicMeasure, check_decreasing_convex_order
from src.couplings import DiscreteCoupling, solve_supermartingale_coupling, independent_coupling
from src.decomposition import compute_irreducible_decomposition
from src.wsot import LinearCost, QuadraticWeakCost, solve_wsot
from src.monotonicity import (
    check_c_monotonicity,
    evaluate_support_cost,
    adversarial_competitor_search,
    MonotonicityTest,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ───────────────── Test instances ────────────────────────────────────────────

def instance_small_2x3() -> tuple[AtomicMeasure, AtomicMeasure, str]:
    """Small 2x3 instance: |supp(mu)|=2, |supp(nu)|=3."""
    mu = AtomicMeasure([0.0, 2.0], [0.5, 0.5])
    nu = AtomicMeasure([-1.0, 1.0, 3.0], [0.3, 0.4, 0.3])
    feasible, viol = check_decreasing_convex_order(mu, nu)
    if not feasible:
        nu = AtomicMeasure(nu.locations + viol + 0.05, nu.weights)
    return mu, nu, "small_2x3"


def instance_medium_3x4() -> tuple[AtomicMeasure, AtomicMeasure, str]:
    """Medium 3x4 instance: |supp(mu)|=3, |supp(nu)|=4."""
    mu = AtomicMeasure([0.0, 1.5, 3.0], [0.3, 0.4, 0.3])
    nu = AtomicMeasure([-0.5, 0.5, 2.0, 4.0], [0.2, 0.3, 0.3, 0.2])
    feasible, viol = check_decreasing_convex_order(mu, nu)
    if not feasible:
        nu = AtomicMeasure(nu.locations + viol + 0.05, nu.weights)
    return mu, nu, "medium_3x4"


def instance_martingale_2x3() -> tuple[AtomicMeasure, AtomicMeasure, str]:
    """Pure martingale 2x3 (mean-preserving spread)."""
    mu = AtomicMeasure([-1.0, 1.0], [0.5, 0.5])
    nu = AtomicMeasure([-2.0, 0.0, 2.0], [0.25, 0.5, 0.25])
    feasible, viol = check_decreasing_convex_order(mu, nu)
    if not feasible:
        nu = AtomicMeasure(nu.locations + viol + 0.05, nu.weights)
    return mu, nu, "martingale_2x3"


def instance_mixed_3x5() -> tuple[AtomicMeasure, AtomicMeasure, str]:
    """Mixed-component 3x5 instance (mu <=_{cd} nu by construction).

    mu mean = 1.0, nu mean = 0.875  =>  mean(nu) < mean(mu) (strict SM).
    P_nu >= P_mu verified analytically at all breakpoints.
    """
    mu = AtomicMeasure([-1.0, 1.0, 3.0], [0.25, 0.50, 0.25])
    nu = AtomicMeasure([-2.0, -0.5, 0.5, 2.0, 4.0], [0.10, 0.20, 0.35, 0.20, 0.15])
    return mu, nu, "mixed_3x5"


# ───────────────── Costs ─────────────────────────────────────────────────────

def build_costs_exp3() -> list:
    """Build cost families for exp3."""
    c1 = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
    c1.name = "C1_Linear_L2"

    c1b = LinearCost(c_fn=lambda x, y: abs(x - y))
    c1b.name = "C1b_Linear_L1"

    c2 = QuadraticWeakCost(
        w_fn=lambda x, y: np.ones_like(y) * 2.0,
        v_fn=lambda x, y: -y,
        name="C2_QuadraticWeak",
    )

    c2b = QuadraticWeakCost(
        w_fn=lambda x, y: (1.0 + y ** 2),
        v_fn=lambda x, y: np.zeros_like(y),
        name="C2b_SqWeightedQuad",
    )

    return [c1, c1b, c2, c2b]


# ───────────────── Perturbed (non-optimal) coupling ──────────────────────────

def perturb_coupling(pi: DiscreteCoupling, noise_scale: float = 0.1) -> DiscreteCoupling:
    """Perturb an optimal coupling to create a non-optimal one.

    Adds noise to the transport matrix and renormalizes while keeping feasibility.
    """
    P_noisy = pi.P + noise_scale * np.abs(np.random.default_rng(123).standard_normal(pi.P.shape))
    P_noisy = np.clip(P_noisy, 0, None)
    # Normalize rows
    for i in range(P_noisy.shape[0]):
        rs = P_noisy[i, :].sum()
        if rs > 1e-14:
            P_noisy[i, :] *= pi.mu.weights[i] / rs
    # Try to fix column sums
    col_sums = P_noisy.sum(axis=0)
    for j in range(P_noisy.shape[1]):
        if col_sums[j] > 1e-14:
            P_noisy[:, j] *= pi.nu.weights[j] / col_sums[j]
    # Re-normalize rows
    for i in range(P_noisy.shape[0]):
        rs = P_noisy[i, :].sum()
        if rs > 1e-14:
            P_noisy[i, :] *= pi.mu.weights[i] / rs
    return DiscreteCoupling(pi.mu, pi.nu, P_noisy)


# ───────────────── Plotting helpers ──────────────────────────────────────────

def plot_coupling_heatmap(
    pi: DiscreteCoupling,
    title: str = "Transport Matrix",
    save_path: str | None = None,
) -> None:
    """Plot coupling matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pi.P.T, aspect="auto", cmap="Blues", origin="lower")
    ax.set_xlabel("mu atom index", fontsize=11)
    ax.set_ylabel("nu atom index", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(pi.mu.n))
    ax.set_xticklabels([f"{x:.2f}" for x in pi.mu.locations])
    ax.set_yticks(range(pi.nu.n))
    ax.set_yticklabels([f"{y:.2f}" for y in pi.nu.locations])
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_monotonicity_summary(
    results_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Bar chart of max monotonicity violations per (instance, cost) pair."""
    fig, ax = plt.subplots(figsize=(10, 5))

    instances = results_df["instance"].unique()
    costs = results_df["cost"].unique()
    x = np.arange(len(instances))
    width = 0.8 / len(costs)

    for i, cost_name in enumerate(costs):
        sub = results_df[results_df.cost == cost_name]
        violations = [
            sub[sub.instance == inst].max_violation.values[0]
            if len(sub[sub.instance == inst]) > 0 else 0.0
            for inst in instances
        ]
        offset = (i - len(costs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, violations, width, label=cost_name, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=20, fontsize=9)
    ax.set_ylabel("Max monotonicity violation", fontsize=11)
    ax.set_title("Supermartingale C-Monotonicity Test (Theorem 2.9)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_cost_comparison(
    pi_star: DiscreteCoupling,
    pi_perturbed: DiscreteCoupling,
    cost,
    instance_name: str,
    save_path: str | None = None,
) -> None:
    """Bar chart comparing optimal vs perturbed coupling costs."""
    x_locs = pi_star.mu.locations
    y_locs = pi_star.nu.locations
    weights = pi_star.mu.weights

    p_opt = pi_star.conditional_laws
    p_pert = pi_perturbed.conditional_laws

    c_opt = evaluate_support_cost(x_locs, p_opt, y_locs, cost, weights)
    c_pert = evaluate_support_cost(x_locs, p_pert, y_locs, cost, weights)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([r"Optimal $\pi^*$", r"Perturbed $\hat{\pi}$"],
                  [c_opt, c_pert],
                  color=["steelblue", "darkorange"],
                  edgecolor="black",
                  width=0.5)
    ax.set_ylabel("Cost C(x, m)", fontsize=11)
    ax.set_title(
        f"Optimal vs Perturbed Cost\n{instance_name} / {getattr(cost, 'name', 'cost')}",
        fontsize=11
    )
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, [c_opt, c_pert]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + abs(val) * 0.02,
                f"{val:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ───────────────── Main experiment ───────────────────────────────────────────

def run_experiment_3() -> pd.DataFrame:
    """Run the full Experiment 3 and return results table."""
    print("=" * 60)
    print("Experiment 3: Monotonicity Principle (Theorem 2.9)")
    print("=" * 60)

    instances = [
        instance_small_2x3,
        instance_medium_3x4,
        instance_martingale_2x3,
        instance_mixed_3x5,
    ]
    costs = build_costs_exp3()
    rows = []

    for inst_fac in instances:
        mu, nu, iname = inst_fac()
        print(f"\nInstance: {iname}")
        print(f"  mu: {mu}")
        print(f"  nu: {nu}")

        feasible, viol = check_decreasing_convex_order(mu, nu)
        print(f"  mu <=cd nu: {feasible}, violation={viol:.2e}")

        # Compute x* for region classification
        decomp = compute_irreducible_decomposition(mu, nu)
        x_star = decomp.x_star
        print(f"  x* = {x_star:.4f}")

        for cost in costs:
            print(f"  Cost: {cost.name}")

            # Solve WSOT
            res = solve_wsot(mu, nu, cost)
            pi_star = res.optimizer
            v_star = res.value
            print(f"    V* = {v_star:.6f}, status = {res.status}")

            if pi_star is None:
                print("    Solver failed, using feasible coupling.")
                pi_star = solve_supermartingale_coupling(mu, nu, cost=None)
                if pi_star is None:
                    pi_star = independent_coupling(mu, nu)
                v_star = np.nan

            # Plot coupling heatmap
            plot_coupling_heatmap(
                pi_star,
                title=f"Optimal Coupling — {iname} / {cost.name}",
                save_path=os.path.join(
                    RESULTS_DIR,
                    f"exp3_coupling_{iname}_{cost.name}.png"
                ),
            )

            # Test C-monotonicity for optimal coupling
            mt = check_c_monotonicity(
                pi_star, x_star, cost,
                n_random_competitors=300,
                use_exhaustive=(mu.n <= 3 and nu.n <= 4),
                use_adversarial=True,
                tol=1e-6,
                verbose=True,
            )
            print(f"    Optimal: is_monotone={mt.is_monotone}, "
                  f"max_viol={mt.max_violation:.4e}, tested={mt.n_competitors_tested}")

            # Perturb coupling and test monotonicity
            pi_perturbed = perturb_coupling(pi_star, noise_scale=0.05)
            mt_pert = check_c_monotonicity(
                pi_perturbed, x_star, cost,
                n_random_competitors=300,
                use_exhaustive=(mu.n <= 3 and nu.n <= 4),
                use_adversarial=True,
                tol=1e-6,
                verbose=False,
            )
            print(f"    Perturbed: is_monotone={mt_pert.is_monotone}, "
                  f"max_viol={mt_pert.max_violation:.4e}")

            # Cost comparison plot
            plot_cost_comparison(
                pi_star, pi_perturbed, cost, iname,
                save_path=os.path.join(
                    RESULTS_DIR,
                    f"exp3_cost_compare_{iname}_{cost.name}.png"
                ),
            )

            rows.append({
                "instance": iname,
                "cost": cost.name,
                "V_star": v_star,
                "is_monotone": mt.is_monotone,
                "max_violation": mt.max_violation,
                "n_competitors": mt.n_competitors_tested,
                "is_monotone_perturbed": mt_pert.is_monotone,
                "max_violation_perturbed": mt_pert.max_violation,
                "SMC_residual_opt": pi_star.check_supermartingale(),
                "SMC_residual_pert": pi_perturbed.check_supermartingale(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp3_results.csv"), index=False)

    # Monotonicity summary plot
    plot_monotonicity_summary(
        df,
        save_path=os.path.join(RESULTS_DIR, "exp3_monotonicity_summary.png"),
    )

    # Print summary
    print("\n--- Summary: Monotonicity Tests ---")
    print(df[["instance", "cost", "is_monotone", "max_violation",
              "is_monotone_perturbed", "max_violation_perturbed"]].to_string(index=False))

    return df


if __name__ == "__main__":
    df = run_experiment_3()
    print("\nExperiment 3 complete. Results saved to results/")
