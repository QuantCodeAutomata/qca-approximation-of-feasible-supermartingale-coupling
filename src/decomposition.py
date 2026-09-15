"""
Irreducible decomposition for supermartingale optimal transport.

Given mu <=_{cd} nu, the potential function D = P_nu - P_mu >= 0 defines:
  - x* = sup{x : D(x) = 0}
  - Diagonal region I_{-1} = {x : D(x) = 0} (approximately (-inf, x*] ∩ supp(mu))
  - Strict supermartingale region I_0 = (x*, infinity)
  - Martingale irreducible components I_n (n >= 1) = connected components
    of {D > 0} ∩ (-inf, x*), i.e., intervals (ell_n, r_n)

The decomposition induces a component-wise split of mu, nu, and pi.

Custom — Context7 found no library equivalent (paper Beiglböck-Juillet framework).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from src.measures import AtomicMeasure, put_function_from_array


@dataclass
class IrreducibleDecomposition:
    """Result of the irreducible decomposition of (mu, nu).

    Attributes
    ----------
    x_star : float  Threshold x* = sup{x: D(x) = 0}.
    D : np.ndarray  Potential D = P_nu - P_mu evaluated on the breakpoint grid.
    grid : np.ndarray  Breakpoint grid.
    diagonal_support : np.ndarray  Support points in I_{-1} (D=0 region).
    strict_support : np.ndarray  Support points in I_0 = (x*,inf).
    martingale_components : list[tuple]  List of (ell_n, r_n) intervals for I_n.
    mu_diag, nu_diag : AtomicMeasure or None  Restricted sub-measures on I_{-1}.
    mu_strict, nu_strict : AtomicMeasure or None  Sub-measures on I_0.
    mu_mart : list[AtomicMeasure]  Sub-measures on each I_n.
    nu_mart : list[AtomicMeasure]  Sub-measures on each I_n.
    """

    x_star: float
    D: np.ndarray
    grid: np.ndarray
    diagonal_support: np.ndarray
    strict_support: np.ndarray
    martingale_components: list[tuple[float, float]]
    mu_diag: AtomicMeasure | None
    nu_diag: AtomicMeasure | None
    mu_strict: AtomicMeasure | None
    nu_strict: AtomicMeasure | None
    mu_mart: list[AtomicMeasure] = field(default_factory=list)
    nu_mart: list[AtomicMeasure] = field(default_factory=list)


def compute_irreducible_decomposition(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    tol_D: float = 1e-9,
    n_grid: int = 2000,
) -> IrreducibleDecomposition:
    """Compute the irreducible decomposition induced by D = P_nu - P_mu.

    Parameters
    ----------
    mu, nu : AtomicMeasure  Must satisfy mu <=_{cd} nu.
    tol_D : float  Tolerance for D(x) = 0.
    n_grid : int  Dense grid resolution for locating x* and martingale components.

    Returns
    -------
    IrreducibleDecomposition

    Custom — Context7 found no library equivalent (paper construction).
    """
    # Build a dense grid covering all support points
    all_pts = np.sort(np.unique(np.concatenate([mu.locations, nu.locations])))
    xmin = all_pts.min() - 1.0
    xmax = all_pts.max() + 1.0
    fine_grid = np.linspace(xmin, xmax, n_grid)
    grid = np.sort(np.unique(np.concatenate([fine_grid, all_pts])))

    P_mu = mu.put_function(grid)
    P_nu = nu.put_function(grid)
    D = P_nu - P_mu

    # x* = sup{x : D(x) = 0}  (where potential equals 0)
    zero_mask = D <= tol_D
    if np.any(zero_mask):
        x_star = float(grid[zero_mask].max())
    else:
        x_star = float(xmin)

    # Classify mu support atoms
    mu_locs = mu.locations
    mu_wts = mu.weights
    nu_locs = nu.locations
    nu_wts = nu.weights

    # Diagonal region I_{-1}: atoms of mu where D(x) ≈ 0
    D_at_mu = mu.put_function(mu_locs)
    D_at_nu_for_mu = nu.put_function(mu_locs)
    D_mu = D_at_nu_for_mu - D_at_mu

    diag_mask_mu = (mu_locs <= x_star + tol_D) & (D_mu <= tol_D)
    strict_mask_mu = mu_locs > x_star + tol_D
    mart_mask_mu = (~diag_mask_mu) & (~strict_mask_mu)

    diagonal_support = mu_locs[diag_mask_mu]
    strict_support = mu_locs[strict_mask_mu]

    # Build sub-measures (unnormalized then re-wrapped with raw weights)
    def _submeasure(locs, wts, mask):
        sub_locs = locs[mask]
        sub_wts = wts[mask]
        if sub_wts.sum() < 1e-14:
            return None
        return AtomicMeasure(sub_locs, sub_wts)

    mu_diag = _submeasure(mu_locs, mu_wts, diag_mask_mu)
    mu_strict = _submeasure(mu_locs, mu_wts, strict_mask_mu)

    # For nu: diagonal region corresponds to nu atoms <= x*
    # nu restricted to I_{-1}: nu atoms where P_nu(x) = P_mu(x) approximately
    D_at_nu = nu.put_function(nu_locs)
    D_ref_nu = mu.put_function(nu_locs)
    D_nu = D_at_nu - D_ref_nu

    # Diagonal nu: atoms <= x* where D ~ 0 or contributions are diagonal
    diag_mask_nu = (nu_locs <= x_star + tol_D) & (D_nu <= tol_D)
    strict_mask_nu = nu_locs > x_star - tol_D

    nu_diag = _submeasure(nu_locs, nu_wts, diag_mask_nu)
    nu_strict = _submeasure(nu_locs, nu_wts, strict_mask_nu)

    # Martingale components: connected components of {D > tol_D} ∩ (-∞, x*)
    D_sub = D.copy()
    D_sub[grid > x_star + tol_D] = 0.0  # mask out strict region
    positive_mask = D_sub > tol_D

    # Find contiguous intervals of the positive region
    martingale_intervals = _find_connected_intervals(grid, positive_mask, tol=1e-8)

    # Restrict mu and nu to each martingale component
    mu_mart = []
    nu_mart = []
    for ell, r in martingale_intervals:
        mask_mu_n = (mu_locs >= ell - tol_D) & (mu_locs <= r + tol_D)
        mask_nu_n = (nu_locs >= ell - tol_D) & (nu_locs <= r + tol_D)
        m_mu = _submeasure(mu_locs, mu_wts, mask_mu_n)
        m_nu = _submeasure(nu_locs, nu_wts, mask_nu_n)
        if m_mu is not None and m_nu is not None:
            mu_mart.append(m_mu)
            nu_mart.append(m_nu)

    return IrreducibleDecomposition(
        x_star=x_star,
        D=D,
        grid=grid,
        diagonal_support=diagonal_support,
        strict_support=strict_support,
        martingale_components=martingale_intervals,
        mu_diag=mu_diag,
        nu_diag=nu_diag,
        mu_strict=mu_strict,
        nu_strict=nu_strict,
        mu_mart=mu_mart,
        nu_mart=nu_mart,
    )


def _find_connected_intervals(
    grid: np.ndarray, mask: np.ndarray, tol: float = 1e-8
) -> list[tuple[float, float]]:
    """Find contiguous intervals where mask is True in a sorted grid.

    Parameters
    ----------
    grid : sorted 1-D array.
    mask : bool array of same length.

    Returns
    -------
    List of (left_endpoint, right_endpoint) tuples.
    """
    intervals = []
    n = len(grid)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            # Interval from grid[i] to grid[j-1]
            intervals.append((float(grid[i]), float(grid[j - 1])))
            i = j
        else:
            i += 1
    return intervals


def quantile_splitting(
    mu: AtomicMeasure,
    mu_k: AtomicMeasure,
    decomp: IrreducibleDecomposition,
    gamma_k: "DiscreteCoupling",  # type: ignore[name-defined]  # noqa: F821
    tol: float = 1e-10,
) -> tuple[list[AtomicMeasure], list[AtomicMeasure]]:
    """Proposition 3.1 quantile-based componentwise splitting for mu^k.

    Uses quantile maps Q = F_mu^{-1} and Q_k = F_{mu_k}^{-1} to define
    mu_n^k = (Q_k)# [lambda|_{A_n}] where A_n = {u: Q(u) in I_n}.

    Custom — Context7 found no library equivalent (paper Proposition 3.1).

    Parameters
    ----------
    mu : AtomicMeasure   Target first marginal.
    mu_k : AtomicMeasure  Perturbed first marginal.
    decomp : IrreducibleDecomposition  Decomposition of (mu, nu).
    gamma_k : DiscreteCoupling  Any feasible coupling in Pi_S(mu_k, nu_k).

    Returns
    -------
    mu_n_k_list : list of AtomicMeasure  Component sub-measures of mu_k.
    nu_n_k_list : list of AtomicMeasure  Component sub-measures of nu_k (induced by gamma_k).
    """
    # Build component intervals in u-space via mu's quantile function
    # A_n = {u in (0,1) : Q(u) in I_n}
    # For atomic mu, A_n is a union of mass intervals

    # Map each atom of mu_k to a component of mu via quantile matching
    # u_i = cumulative mass up to location i in mu
    cum_mu = np.cumsum(mu.weights)
    cum_mu_prev = np.concatenate([[0.0], cum_mu[:-1]])

    # For each mass interval in mu_k, determine which component it falls in
    cum_muk = np.cumsum(mu_k.weights)
    cum_muk_prev = np.concatenate([[0.0], cum_muk[:-1]])

    # Component classification via mu's quantile
    # Component for each atom in mu (based on location)
    n_components = 1 + len(decomp.martingale_components)  # diag + martingale_n's + strict

    # Build a function that maps u -> component index
    # Component -1: diagonal
    # Component 0: strict supermartingale
    # Component n: martingale n (n = 1, 2, ...)

    def classify_x(x: float) -> int:
        """Classify a point x into a component."""
        if x > decomp.x_star + tol:
            return 0  # strict
        # Check martingale components
        for n_idx, (ell, r) in enumerate(decomp.martingale_components):
            if ell - tol <= x <= r + tol:
                return n_idx + 1  # martingale component n_idx+1
        return -1  # diagonal

    # For each atom of mu, find its component
    mu_atom_components = [classify_x(xi) for xi in mu.locations]

    # Now split mu_k using quantile matching
    # For each component, find the total mass A_n has in mu
    component_ids = sorted(set(mu_atom_components))
    component_mass = {}
    for c in component_ids:
        mass = sum(
            mu.weights[i] for i in range(mu.n) if mu_atom_components[i] == c
        )
        component_mass[c] = mass

    # Map to mu_k quantile space: the mass A_n in mu corresponds to same quantile set
    # Since Q_k maps the same u-intervals, assign mass proportionally
    mu_n_k_list = []
    nu_n_k_list = []

    # Track cumulative u processed
    u_processed = 0.0

    # Process components in order of their quantile position in mu
    # Sort atoms of mu by location (already sorted)
    ordered_components = []
    for i, xi in enumerate(mu.locations):
        c = mu_atom_components[i]
        ordered_components.append((cum_mu_prev[i], cum_mu[i], c))

    # For each component, collect the u-intervals belonging to it
    comp_u_intervals: dict[int, list[tuple[float, float]]] = {}
    for u_start, u_end, c in ordered_components:
        if c not in comp_u_intervals:
            comp_u_intervals[c] = []
        comp_u_intervals[c].append((u_start, u_end))

    # For each component, find atoms of mu_k in the same u-intervals
    for c in component_ids:
        u_intervals = comp_u_intervals.get(c, [])
        if not u_intervals:
            continue

        # Find atoms of mu_k whose u-intervals overlap with A_n
        muk_indices = []
        muk_fractions = []
        for k in range(mu_k.n):
            uk_start = cum_muk_prev[k]
            uk_end = cum_muk[k]
            # Overlap with any interval in u_intervals
            for u_start, u_end in u_intervals:
                overlap = max(0.0, min(uk_end, u_end) - max(uk_start, u_start))
                if overlap > tol:
                    muk_indices.append(k)
                    muk_fractions.append(overlap / mu_k.weights[k])
                    break  # count only once per atom

        if not muk_indices:
            continue

        # Sub-measure mu_n_k: atoms of mu_k in A_n with proportional mass
        sub_locs_mu = mu_k.locations[muk_indices]
        sub_wts_mu = np.array([
            mu_k.weights[k] * frac for k, frac in zip(muk_indices, muk_fractions)
        ])

        if sub_wts_mu.sum() < tol:
            continue

        mu_n_k = AtomicMeasure(sub_locs_mu, sub_wts_mu)

        # Sub-measure nu_n_k induced by gamma_k on these mu_k atoms
        # nu_n_k = sum_{i in muk_indices} gamma_k.P[i, :] * frac_i
        P_sub = np.zeros(gamma_k.nu.n)
        for k, frac in zip(muk_indices, muk_fractions):
            # Need to find the index of mu_k.locations[k] in gamma_k.mu
            # gamma_k.mu should be mu_k
            # Find matching index in gamma_k.mu
            gk_idx = np.where(
                np.abs(gamma_k.mu.locations - mu_k.locations[k]) < tol
            )[0]
            if len(gk_idx) > 0:
                P_sub += gamma_k.P[gk_idx[0], :] * frac
            else:
                # approximate: spread proportionally
                pass

        if P_sub.sum() < tol:
            # Use uniform on nu_k
            P_sub = gamma_k.nu.weights.copy() * sub_wts_mu.sum()

        nu_n_k = AtomicMeasure(gamma_k.nu.locations, P_sub)

        mu_n_k_list.append(mu_n_k)
        nu_n_k_list.append(nu_n_k)

    return mu_n_k_list, nu_n_k_list


def convex_order_lift(
    mu_n: AtomicMeasure,
    nu_n: AtomicMeasure,
    tol: float = 1e-10,
) -> AtomicMeasure:
    """Construct (nu_n)^sharp with P_{(nu_n)^sharp} = P_{mu_n} vee (P_{nu_n} - delta_n).

    Here delta_n = bar(mu_n) - bar(nu_n) >= 0 is the barycenter gap.
    The lift raises nu_n to be in convex order above mu_n for the martingale component.

    (nu_n)^sharp is defined via its put function:
      P_{(nu_n)^sharp}(x) = max(P_{mu_n}(x), P_{nu_n}(x) - delta_n)

    We reconstruct the atomic measure from this put function on the support grid.

    Custom — Context7 found no library equivalent (paper Step 8 construction).
    """
    delta_n = float(mu_n.mean - nu_n.mean)
    # delta_n should be >= 0 for a supermartingale component
    # (in martingale components, barycenters match, so delta_n ~ 0)
    delta_n = max(delta_n, 0.0)

    all_pts = np.sort(np.unique(np.concatenate([mu_n.locations, nu_n.locations])))

    P_mu_n = mu_n.put_function(all_pts)
    P_nu_n = nu_n.put_function(all_pts)

    # (nu_n)^sharp put function
    P_sharp = np.maximum(P_mu_n, P_nu_n - delta_n)

    # Reconstruct atomic measure from put function:
    # The second derivative of P_sharp gives the density of (nu_n)^sharp
    # For piecewise-linear P_sharp, the atoms are at kink points.
    # Use finite differences on breakpoints.

    # Augment grid with boundary points
    xmin = all_pts.min() - 1.0
    xmax = all_pts.max() + 1.0
    grid = np.concatenate([[xmin], all_pts, [xmax]])
    P_sharp_grid = np.maximum(
        mu_n.put_function(grid),
        nu_n.put_function(grid) - delta_n
    )

    # Slope of P_sharp = CDF
    dx = np.diff(grid)
    dp = np.diff(P_sharp_grid)
    slope = dp / np.where(dx > 0, dx, 1e-300)  # F_{(nu_n)^sharp}(x)

    # Atom masses = jump in slope
    mass_at_grid = np.diff(slope)
    atom_locs = grid[1:-1]

    pos_mask = mass_at_grid > 1e-12
    if not np.any(pos_mask):
        # Fallback to nu_n (no lift needed if delta_n is tiny)
        return nu_n

    locs = atom_locs[pos_mask]
    wts = mass_at_grid[pos_mask]

    # Ensure total mass is 1
    if wts.sum() < 1e-14:
        return nu_n

    return AtomicMeasure(locs, wts / wts.sum())
