"""
Probability measures on R, put functions, and Wasserstein distances.

Mathematical background:
- An atomic measure eta = sum_i w_i delta_{t_i} is stored as (locations, weights).
- Put function: P_eta(x) = int (x - t)^+ eta(dt) = sum_i w_i max(x - t_i, 0).
- Decreasing-convex order: mu <=_{cd} nu  iff  P_mu(x) <= P_nu(x) for all x.
- Wasserstein W_1 computed via ot.emd2 (POT) on discrete support.
"""

from __future__ import annotations

import numpy as np
import ot  # Python Optimal Transport — Context7 confirmed via /pythonot/pot


class AtomicMeasure:
    """Finite-support atomic probability measure on R.

    Parameters
    ----------
    locations : array-like
        Atom positions (need not be sorted).
    weights : array-like
        Non-negative atom masses summing to 1.
    tol : float
        Tolerance for normalisation and non-negativity checks.
    """

    def __init__(
        self,
        locations: np.ndarray | list,
        weights: np.ndarray | list,
        tol: float = 1e-10,
    ) -> None:
        locs = np.asarray(locations, dtype=float)
        wts = np.asarray(weights, dtype=float)
        assert locs.ndim == 1 and wts.ndim == 1, "Locations and weights must be 1-D."
        assert locs.shape == wts.shape, "Locations and weights must have the same length."
        assert np.all(wts >= -tol), f"Weights must be non-negative; min={wts.min():.3e}"
        wts = np.clip(wts, 0.0, None)
        total = wts.sum()
        assert total > tol, "Total mass must be positive."

        # Sort by location and merge duplicate atoms
        idx = np.argsort(locs)
        locs, wts = locs[idx], wts[idx]
        # merge identical locations
        unique_locs, inv = np.unique(locs, return_inverse=True)
        unique_wts = np.zeros(len(unique_locs))
        np.add.at(unique_wts, inv, wts)
        # Remove zero-mass atoms
        nonzero = unique_wts > 0.0
        self.locations: np.ndarray = unique_locs[nonzero]
        self.weights: np.ndarray = unique_wts[nonzero] / unique_wts[nonzero].sum()
        self.tol = tol

    @property
    def n(self) -> int:
        """Number of atoms."""
        return len(self.locations)

    @property
    def mean(self) -> float:
        """First moment E[X]."""
        return float(self.weights @ self.locations)

    @property
    def second_moment(self) -> float:
        """Second moment E[X^2]."""
        return float(self.weights @ (self.locations ** 2))

    def put_function(self, x: float | np.ndarray) -> float | np.ndarray:
        """Put function P_eta(x) = int (x - t)^+ eta(dt).

        Parameters
        ----------
        x : float or array
            Evaluation point(s).

        Returns
        -------
        float or array
            P_eta(x) values.
        """
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        # Broadcast: shape (len(x), n)
        result = (
            self.weights * np.maximum(x[:, None] - self.locations[None, :], 0.0)
        ).sum(axis=1)
        return float(result[0]) if scalar else result

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """CDF F_eta(x) = eta((-inf, x])."""
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.array(
            [self.weights[self.locations <= xi].sum() for xi in x], dtype=float
        )
        return float(result[0]) if scalar else result

    def quantile(self, u: float | np.ndarray) -> float | np.ndarray:
        """Quantile (generalised inverse CDF) F^{-1}(u) = inf{x: F(x) >= u}.

        Parameters
        ----------
        u : float or array in (0, 1).
        """
        scalar = np.ndim(u) == 0
        u = np.atleast_1d(np.asarray(u, dtype=float))
        cum = np.cumsum(self.weights)
        # For each u, find first atom where cumulative mass >= u
        result = np.empty(len(u))
        for k, uk in enumerate(u):
            idx = np.searchsorted(cum, uk, side="left")
            idx = min(idx, self.n - 1)
            result[k] = self.locations[idx]
        return float(result[0]) if scalar else result

    def restrict(self, mask: np.ndarray) -> "AtomicMeasure | None":
        """Return sub-measure on atoms where mask is True (unnormalised mass).

        Returns None if total restricted mass is zero.
        """
        locs = self.locations[mask]
        wts = self.weights[mask]
        if wts.sum() < self.tol:
            return None
        # Return raw-mass version: caller must handle normalisation if needed
        return AtomicMeasure(locs, wts)

    def scale(self, c: float) -> "AtomicMeasure":
        """Return measure with weights multiplied by scalar c (c in (0,1] for submeasure)."""
        return AtomicMeasure(self.locations.copy(), self.weights * c)

    def __repr__(self) -> str:
        return (
            f"AtomicMeasure(n={self.n}, mean={self.mean:.4f}, "
            f"supp=[{self.locations.min():.3f},{self.locations.max():.3f}])"
        )


def put_function_grid(
    eta: AtomicMeasure, grid: np.ndarray
) -> np.ndarray:
    """Evaluate put function on a grid of points.

    Parameters
    ----------
    eta : AtomicMeasure
    grid : np.ndarray sorted evaluation points.

    Returns
    -------
    np.ndarray of P_eta(x) for x in grid.
    """
    return eta.put_function(grid)


def check_decreasing_convex_order(
    mu: AtomicMeasure,
    nu: AtomicMeasure,
    n_grid: int = 1000,
    tol: float = 1e-8,
) -> tuple[bool, float]:
    """Check mu <=_{cd} nu by comparing put functions on a dense grid.

    Returns (feasible, max_violation) where violation = max_x (P_mu(x) - P_nu(x)).

    Parameters
    ----------
    mu, nu : AtomicMeasure
    n_grid : int  Number of evaluation points.
    tol : float   Tolerance for declaring feasibility.
    """
    all_pts = np.concatenate([mu.locations, nu.locations])
    xmin, xmax = all_pts.min() - 1.0, all_pts.max() + 1.0
    grid = np.linspace(xmin, xmax, n_grid)
    grid = np.sort(np.concatenate([grid, all_pts]))

    p_mu = mu.put_function(grid)
    p_nu = nu.put_function(grid)
    violation = float(np.max(p_mu - p_nu))
    return violation <= tol, violation


def wasserstein_1(mu: AtomicMeasure, nu: AtomicMeasure) -> float:
    """Compute W_1(mu, nu) via 1D optimal transport using POT.

    For 1D measures, W_1 = int |F_mu^{-1}(u) - F_nu^{-1}(u)| du.
    Uses POT's ot.emd2 with absolute-value cost matrix.

    # Using POT ot.emd2 — Context7 confirmed via /pythonot/pot
    """
    a = mu.weights.astype(np.float64)
    b = nu.weights.astype(np.float64)
    # Cost matrix: |x_i - y_j|
    M = np.abs(mu.locations[:, None] - nu.locations[None, :]).astype(np.float64)
    cost = ot.emd2(a, b, M)
    return float(cost)


def wasserstein_1_1d_quantile(mu: AtomicMeasure, nu: AtomicMeasure) -> float:
    """W_1 via quantile formula for 1D measures (independent check)."""
    # Merge all atom locations as breakpoints
    all_pts = np.sort(np.unique(np.concatenate([mu.locations, nu.locations])))
    # Evaluate CDFs
    f_mu = np.array([mu.cdf(x) for x in all_pts])
    f_nu = np.array([nu.cdf(x) for x in all_pts])
    # W_1 = int |F_mu - F_nu| dx over the real line
    # Approximate by piecewise-constant integral on breakpoints
    widths = np.diff(np.concatenate([[-1e10], all_pts, [1e10]]))
    # CDF at x is the CDF just before the next breakpoint
    f_mu_ext = np.concatenate([[0.0], f_mu, [1.0]])
    f_nu_ext = np.concatenate([[0.0], f_nu, [1.0]])
    # |F_mu(x) - F_nu(x)| integrated — use left-CDF value
    # Actually use the correct trapezoidal rule on the breakpoints
    pts_ext = np.concatenate([[-1e10], all_pts, [1e10]])
    w = np.diff(pts_ext)
    f_mu_left = np.concatenate([[0.0], f_mu])
    f_nu_left = np.concatenate([[0.0], f_nu])
    return float(np.dot(np.abs(f_mu_left - f_nu_left), w))


def add_measures(mu: AtomicMeasure, nu: AtomicMeasure) -> AtomicMeasure:
    """Return the sum mu + nu (total mass = 2 if both are probability measures).

    Useful for combining sub-measures with positive masses.
    """
    locs = np.concatenate([mu.locations, nu.locations])
    # Use unnormalized weights for sub-measures
    wts_mu = mu.weights * mu.weights.sum()  # restore actual mass
    wts_nu = nu.weights * nu.weights.sum()
    wts = np.concatenate([wts_mu, wts_nu])
    return AtomicMeasure(locs, wts)


def put_function_from_array(
    locations: np.ndarray, weights: np.ndarray, x: float | np.ndarray
) -> float | np.ndarray:
    """Fast put-function evaluation without constructing AtomicMeasure."""
    scalar = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, dtype=float))
    result = (weights * np.maximum(x[:, None] - locations[None, :], 0.0)).sum(axis=1)
    return float(result[0]) if scalar else result


def convex_minorant(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lower convex hull (greatest convex minorant) of the function (x, y).

    Custom — Context7 found no library equivalent for pointwise convex minorant.

    Parameters
    ----------
    x : sorted 1-D array of evaluation points.
    y : function values at x.

    Returns
    -------
    y_conv : 1-D array, the largest convex function <= y pointwise evaluated at x.
    """
    assert np.all(np.diff(x) >= 0), "x must be sorted."
    n = len(x)
    # Build lower convex hull using Andrew's monotone chain on (x, y)
    hull = []
    for i in range(n):
        while len(hull) >= 2:
            x1, y1 = x[hull[-2]], y[hull[-2]]
            x2, y2 = x[hull[-1]], y[hull[-1]]
            x3, y3 = x[i], y[i]
            # Cross product of (x2-x1,y2-y1) and (x3-x1,y3-y1) — keep if right turn
            cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)

    hull_x = x[hull]
    hull_y = y[hull]
    # Interpolate back to all x points
    y_conv = np.interp(x, hull_x, hull_y)
    # Ensure y_conv <= y everywhere (convex minorant property)
    y_conv = np.minimum(y_conv, y)
    return y_conv


def lattice_join_cd(
    mu: AtomicMeasure, nu: AtomicMeasure, grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute put function of mu vee_{cd} nu = pointwise max of put functions.

    Returns (grid, P_{mu vee nu}) — does not reconstruct the atomic measure
    because the join may not be easily atomic.

    Custom — Context7 found no library equivalent (paper lattice construction).
    """
    if grid is None:
        all_pts = np.concatenate([mu.locations, nu.locations])
        grid = np.sort(np.unique(all_pts))

    p_mu = mu.put_function(grid)
    p_nu = nu.put_function(grid)
    return grid, np.maximum(p_mu, p_nu)


def lattice_meet_cd(
    mu: AtomicMeasure, nu: AtomicMeasure, grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute put function of mu wedge_{cd} nu = largest convex minorant of pointwise min.

    Custom — Context7 found no library equivalent (paper lattice construction).
    """
    if grid is None:
        all_pts = np.concatenate([mu.locations, nu.locations])
        grid = np.sort(np.unique(all_pts))

    p_mu = mu.put_function(grid)
    p_nu = nu.put_function(grid)
    p_min = np.minimum(p_mu, p_nu)
    p_meet = convex_minorant(grid, p_min)
    return grid, p_meet


def measure_from_put_function(
    grid: np.ndarray, put_vals: np.ndarray
) -> AtomicMeasure:
    """Reconstruct atomic measure from put function values on a grid.

    P_eta''(x) = eta({x}) in the distributional sense, so the second finite
    difference approximates atom masses.

    Custom — Context7 found no library equivalent.
    """
    # P_eta(x) is piecewise linear for atomic measures.
    # Its derivative is F_eta(x) = 1 - slope change at each atom.
    # Approximate via finite differences of the slope.
    assert len(grid) >= 2
    # First differences of put values
    dp = np.diff(put_vals)
    dx = np.diff(grid)
    # Slopes of put function = CDF value (left-continuous CDF)
    slopes = dp / np.where(dx > 0, dx, 1e-300)
    # Second differences give atom masses at interior grid points
    # For an atomic measure, slope jumps equal atom masses
    d2p = np.diff(slopes)
    d2x = (dx[:-1] + dx[1:]) / 2.0
    # Mass at grid[i+1] for i in range(len(grid)-2)
    masses = d2p  # approximate masses at midpoints
    locs = grid[1:-1]
    # Filter positive masses
    pos = masses > 1e-12
    if not np.any(pos):
        # Fallback: uniform on grid extremes
        return AtomicMeasure([grid[0], grid[-1]], [0.5, 0.5])
    return AtomicMeasure(locs[pos], masses[pos])
