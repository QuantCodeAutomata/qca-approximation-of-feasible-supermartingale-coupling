# Approximation of Feasible Supermartingale Couplings

Numerical reproduction of three theorems from the paper on **Supermartingale Optimal
Transport (WSOT)** under marginal perturbations.

## Experiments

| ID | Theorem | Title |
|----|---------|-------|
| exp_1 | Thm 2.1 | Approximation of feasible supermartingale couplings |
| exp_2 | Thm 2.3 | Stability and continuity of WSOT values and optimizers |
| exp_3 | Thm 2.9 / Cor 2.10 | Monotonicity principle for weak WSOT |

All inputs are **synthetic** probability measures on ℝ — no market data is used.

## Repository Structure

```
├── data/           # Data placeholder (synthetic inputs generated in code)
├── src/            # Core implementation
│   ├── measures.py         # AtomicMeasure, put functions, Wasserstein W1
│   ├── couplings.py        # DiscreteCoupling, AW1, SM constraint, OT solvers
│   ├── decomposition.py    # Irreducible decomposition, lattice ops, convex hull
│   ├── approximation.py    # Exp 1: Theorem 2.1 approximation construction
│   ├── wsot.py             # Exp 2: Weak SOT optimization, cost classes
│   └── monotonicity.py     # Exp 3: C-monotonicity testing
├── exp/            # Experiment runner scripts
│   ├── exp1_approximation.py
│   ├── exp2_stability.py
│   └── exp3_monotonicity.py
├── tests/          # pytest unit and integration tests (119 tests)
├── results/        # Output: plots, CSVs, RESULTS.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `scipy`, `cvxpy`, `POT` (`ot`), `matplotlib`.

## Running Experiments

```bash
# Run all three experiments (saves results to results/)
python exp/exp1_approximation.py
python exp/exp2_stability.py
python exp/exp3_monotonicity.py
```

## Running Tests

```bash
pytest tests/ -v
```

All 119 tests pass.

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for detailed findings.

### Quick summary

- **Exp 1 (Thm 2.1):** AW₁(πᵏ, π) decreases 8–10× over 10 perturbation steps
  for all three component families (strict, mixed, martingale-limit).
- **Exp 2 (Thm 2.3):** WSOT value gaps converge to 0 under W₁-convergence of
  marginals for linear, convex, and strictly-convex-in-m weak cost functions.
- **Exp 3 (Thm 2.9):** All 16 computed optimal couplings satisfy C-monotonicity
  (max violation ≤ 10⁻⁸); perturbed couplings violate it in 10/16 cases.

## Key Mathematical Objects

| Symbol | Description |
|--------|-------------|
| P_η(x) | Put function: ∫(x−t)⁺ η(dt) |
| μ ≤_cd ν | Decreasing-convex order: P_μ ≤ P_ν pointwise |
| Π_S(μ,ν) | Supermartingale couplings: E[Y\|X] ≤ X a.s. |
| AW_r | Adapted Wasserstein distance (nested OT) |
| D = P_ν − P_μ | Defect function; x\* = sup{x : D(x)=0} |
| I₋₁, I₀, Iₙ | Irreducible components (diagonal, strict SM, martingale) |
| C(x, m) | Weak cost depending on x and full conditional law m |

## License

MIT
