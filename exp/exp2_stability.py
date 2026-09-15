"""
Experiment 2: Stability and Continuity of WSOT Values and Optimizers (Theorem 2.3).

For each (mu, nu) test family:
  - Defines perturbation sequence (mu^k, nu^k) -> (mu, nu)
  - Solves WSOT under three cost families:
      C1: classical linear cost c(x,y) = (x-y)^2
      C2: strictly convex quadratic weak cost
      C3: variance-penalized cost
  - Records V_S^C(mu^k, nu^k) and verifies convergence to V_S^C(mu, nu)
  - Extracts optimizers and measures AW_1 convergence when unique

Results saved to results/exp2_*.{png,csv}
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.measures import AtomicMeasure, wasserstein_1, check_decreasing_convex_order
from src.couplings import DiscreteCoupling, adapted_wasserstein_1, independent_coupling
from src.wsot import (
    LinearCost, QuadraticWeakCost, VarianceCost,
    solve_wsot, run_stability_experiment, StabilityResult,
)
from src.approximation import construct_perturbation_sequence

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ───────────────── Test families ─────────────────────────────────────────────

def family_strict_irreducible() -> tuple[AtomicMeasure, AtomicMeasure, str]:
    """Strict irreducible test family."""
    mu = AtomicMeasure([1.0, 2.5, 4.0], [1 / 3, 1 / 3, 1 / 3])
    nu = AtomicMeasure([0.0, 1.0, 2.0, 3.5], [0.25, 0.3, 0.25, 0.2])
    feasible, viol = check_decreasing_convex_order(mu, nu)
    if not feasible:
        nu = AtomicMeasure(nu.locations + viol + 0.05, nu.weights)
    return mu, nu, "strict_irreducible"


def family_mixed_components() -> tuple[AtomicMeasure, AtomicMeasure, str]:
    """Mixed component test family built by explicit component mixing.

    Martingale component: mu_mart = 0.5 δ_{-1} + 0.5 δ_1, mean = 0
                          nu_mart = 0.4 δ_{-2} + 0.2 δ_0 + 0.4 δ_2, mean = 0
    Strict component:     mu_str  = 0.5 δ_3 + 0.5 δ_5, mean = 4
                          nu_str  = 0.7 δ_2 + 0.3 δ_5, mean = 2.9 < 4  (strict)
    Mixture weight:       alpha = 0.4 (martingale), 0.6 (strict)

    By linearity of put functions:
        P_{mu_mix} = 0.4 P_{mu_mart} + 0.6 P_{mu_str}
                  <= 0.4 P_{nu_mart} + 0.6 P_{nu_str} = P_{nu_mix}
    so mu_mix <=_{cd} nu_mix is guaranteed by construction.
    """
    alpha = 0.4  # martingale weight

    # Martingale component (mu_mart <=_{cd} nu_mart, equal means = 0)
    locs_mu_m = np.array([-1.0, 1.0])
    w_mu_m = np.array([0.5, 0.5])
    locs_nu_m = np.array([-2.0, 0.0, 2.0])
    w_nu_m = np.array([0.4, 0.2, 0.4])

    # Strict component (mu_str <=_{cd} nu_str, mean(nu_str) < mean(mu_str))
    locs_mu_s = np.array([3.0, 5.0])
    w_mu_s = np.array([0.5, 0.5])
    locs_nu_s = np.array([2.0, 5.0])
    w_nu_s = np.array([0.7, 0.3])

    # Merge support locations and combine weights
    all_mu_locs = np.concatenate([locs_mu_m, locs_mu_s])
    all_mu_w = np.concatenate([alpha * w_mu_m, (1.0 - alpha) * w_mu_s])
    all_nu_locs = np.concatenate([locs_nu_m, locs_nu_s])
    all_nu_w = np.concatenate([alpha * w_nu_m, (1.0 - alpha) * w_nu_s])

    # Consolidate duplicate locations
    def consolidate(locs: np.ndarray, weights: np.ndarray):
        unique_locs = np.unique(locs)
        new_w = np.array([weights[locs == ll].sum() for ll in unique_locs])
        return unique_locs, new_w

    locs_mu, w_mu = consolidate(all_mu_locs, all_mu_w)
    locs_nu, w_nu = consolidate(all_nu_locs, all_nu_w)

    mu = AtomicMeasure(locs_mu.tolist(), w_mu.tolist())
    nu = AtomicMeasure(locs_nu.tolist(), w_nu.tolist())
    return mu, nu, "mixed_components"


def family_martingale_limit() -> tuple[AtomicMeasure, AtomicMeasure, str]:
    """Pure martingale limit family."""
    mu = AtomicMeasure([-1.0, 0.0, 1.0], [1 / 3, 1 / 3, 1 / 3])
    nu = AtomicMeasure([-2.0, -1.0, 0.0, 1.0, 2.0], [0.1, 0.2, 0.4, 0.2, 0.1])
    feasible, viol = check_decreasing_convex_order(mu, nu)
    if not feasible:
        nu = AtomicMeasure(nu.locations + viol + 0.05, nu.weights)
    return mu, nu, "martingale_limit"


# ───────────────── Cost families ─────────────────────────────────────────────

def build_costs() -> list:
    """Build the three cost families.

    C1: Classical linear c(x,y) = (x-y)^2
    C2: Strictly convex quadratic weak cost w_j(x)=1+|x|, v_j(x)=-y_j
    C3: Variance-penalized cost phi(d)=d^2, lambda=0.5
    """
    c1 = LinearCost(c_fn=lambda x, y: (x - y) ** 2)
    c1.name = "C1_Linear_L2"

    c2 = QuadraticWeakCost(
        w_fn=lambda x, y: (1.0 + np.abs(x)) * np.ones_like(y),
        v_fn=lambda x, y: -y,
        name="C2_QuadraticWeak",
    )

    c3 = VarianceCost(
        phi_fn=lambda d: d ** 2,
        lam=0.5,
        name="C3_VariancePenalized",
    )

    return [c1, c2, c3]


# ───────────────── Plotting helpers ──────────────────────────────────────────

def plot_value_convergence(
    stability_results: list[StabilityResult],
    family_name: str,
    save_path: str | None = None,
) -> None:
    """Plot |V(mu^k,nu^k) - V(mu,nu)| vs k for each cost."""
    n_costs = len(stability_results)
    fig, axes = plt.subplots(1, n_costs, figsize=(5 * n_costs, 4))
    if n_costs == 1:
        axes = [axes]

    for ax, sr in zip(axes, stability_results):
        ks = np.arange(1, len(sr.v_sequence) + 1)
        gaps = [abs(v - sr.v_limit) if not np.isnan(v) else np.nan
                for v in sr.v_sequence]

        ax.semilogy(ks, gaps, "o-", color="navy", linewidth=2, label="Value gap")
        ax.semilogy(ks, sr.w1_mu_sequence, "s--", color="steelblue",
                    linewidth=1.5, label=r"$W_1(\mu^k,\mu)$")
        ax.semilogy(ks, sr.w1_nu_sequence, "^--", color="darkorange",
                    linewidth=1.5, label=r"$W_1(\nu^k,\nu)$")
        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel("Distance / gap", fontsize=11)
        ax.set_title(f"{sr.cost_name}\n(limit V={sr.v_limit:.4f})", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"WSOT Value Convergence — Family: {family_name}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_aw1_convergence_exp2(
    stability_results: list[StabilityResult],
    family_name: str,
    save_path: str | None = None,
) -> None:
    """Plot AW_1(pi^{*,k}, pi^*) vs k for each cost."""
    n_costs = len(stability_results)
    fig, axes = plt.subplots(1, n_costs, figsize=(5 * n_costs, 4))
    if n_costs == 1:
        axes = [axes]

    for ax, sr in zip(axes, stability_results):
        ks = np.arange(1, len(sr.aw1_sequence) + 1)
        aw1s = sr.aw1_sequence

        valid = [(k, a) for k, a in zip(ks, aw1s) if not np.isnan(a)]
        if valid:
            ks_v, a_v = zip(*valid)
            ax.plot(ks_v, a_v, "o-", color="crimson", linewidth=2,
                    label=r"$AW_1(\pi^{*,k}, \pi^*)$")

        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel(r"$AW_1$", fontsize=11)
        ax.set_title(f"{sr.cost_name}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Optimizer Convergence AW_1 — Family: {family_name}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ───────────────── Main experiment ───────────────────────────────────────────

def run_experiment_2() -> pd.DataFrame:
    """Run the full Experiment 2 and return results table."""
    print("=" * 60)
    print("Experiment 2: WSOT Stability and Continuity (Theorem 2.3)")
    print("=" * 60)

    families = [
        family_strict_irreducible,
        family_mixed_components,
        family_martingale_limit,
    ]
    costs = build_costs()
    K = 10

    all_rows = []

    for fac in families:
        mu, nu, fname = fac()
        print(f"\nFamily: {fname}")
        print(f"  mu: {mu}")
        print(f"  nu: {nu}")

        feasible, viol = check_decreasing_convex_order(mu, nu)
        print(f"  mu <=cd nu: {feasible}, violation={viol:.2e}")

        seq = construct_perturbation_sequence(mu, nu, K=K, delta_max=0.4)

        stability_results = run_stability_experiment(
            mu, nu, seq, costs, verbose=True
        )

        # Plots
        plot_value_convergence(
            stability_results, fname,
            save_path=os.path.join(RESULTS_DIR, f"exp2_value_conv_{fname}.png"),
        )
        plot_aw1_convergence_exp2(
            stability_results, fname,
            save_path=os.path.join(RESULTS_DIR, f"exp2_aw1_conv_{fname}.png"),
        )

        # Collect rows
        for sr in stability_results:
            for k_idx, (v_k, gap, aw1, w1mu, w1nu) in enumerate(
                zip(sr.v_sequence, sr.value_gaps,
                    sr.aw1_sequence, sr.w1_mu_sequence, sr.w1_nu_sequence),
                start=1
            ):
                all_rows.append({
                    "family": fname,
                    "cost": sr.cost_name,
                    "k": k_idx,
                    "V_limit": sr.v_limit,
                    "V_k": v_k,
                    "value_gap": gap,
                    "AW1": aw1,
                    "W1_mu": w1mu,
                    "W1_nu": w1nu,
                })

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp2_results.csv"), index=False)

    # Summary
    print("\n--- Summary: Value gap convergence ---")
    for fname in [f()[2] for f in families]:
        sub = df[df.family == fname]
        for cost_name in df.cost.unique():
            sub2 = sub[sub.cost == cost_name]
            if len(sub2) > 0:
                gap0 = sub2.value_gap.iloc[0]
                gapK = sub2.value_gap.iloc[-1]
                print(f"  {fname}/{cost_name}: gap(k=1)={gap0:.4e} -> gap(k=K)={gapK:.4e}")

    return df


if __name__ == "__main__":
    df = run_experiment_2()
    print("\nExperiment 2 complete. Results saved to results/")
