"""Auditable single-pair adapter of Li/Weaver-Yoritomo scheme V.

Li et al. (2020) minimize a network-wide antisymmetry-to-signal ratio.
For a whitened two-station experiment there is no independent collection of
array autocorrelation amplitudes with which to construct their signal term.
This adapter therefore fixes the sum of non-negative temporal weights to one
and minimizes only the positive-minus-negative branch energy.  The result is
deliberately labelled an adapter rather than an exact network implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class TemporalAntisymmetryResult:
    weights: np.ndarray
    objective: float
    uniform_objective: float
    objective_scale: float
    scaled_objective: float
    effective_window_count: float
    iterations: int
    optimizer_message: str
    scope: str = "single_pair_whitened_scheme_v_adapter"


def fit_temporal_antisymmetry_weights(
    positive_windows: np.ndarray,
    negative_windows: np.ndarray,
    lag_time_s: np.ndarray,
    *,
    window_s: tuple[float, float],
    maximum_iterations: int = 1000,
    tolerance: float = 1e-10,
) -> TemporalAntisymmetryResult:
    """Minimize weighted branch antisymmetry with physical non-negative weights.

    Parameters
    ----------
    positive_windows, negative_windows
        Matrices shaped ``(time_window, non_negative_lag_sample)``.  The two
        branches must already share receiver orientation.
    lag_time_s
        Uniform non-negative lag axis.
    window_s
        Inclusive lag range over which branch antisymmetry is minimized.
    """

    positive = np.asarray(positive_windows, dtype=np.float64)
    negative = np.asarray(negative_windows, dtype=np.float64)
    lag = np.asarray(lag_time_s, dtype=np.float64).reshape(-1)
    if (
        positive.ndim != 2
        or positive.shape != negative.shape
        or positive.shape[0] < 2
        or positive.shape[1] != lag.size
        or np.any(~np.isfinite(positive))
        or np.any(~np.isfinite(negative))
        or np.any(~np.isfinite(lag))
    ):
        raise ValueError("branch windows must be finite, aligned 2-D matrices")
    if lag.size < 2 or not np.allclose(np.diff(lag), np.diff(lag)[0]):
        raise ValueError("lag_time_s must be a uniform grid")
    start_s, stop_s = map(float, window_s)
    if not 0.0 <= start_s < stop_s <= lag[-1]:
        raise ValueError("window_s must lie inside the non-negative lag axis")
    selected = (lag >= start_s) & (lag <= stop_s)
    if np.count_nonzero(selected) < 2:
        raise ValueError("antisymmetry window must contain at least two samples")
    if maximum_iterations < 1 or tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("optimizer controls are invalid")

    antisymmetry = positive[:, selected] - negative[:, selected]
    raw_gram = antisymmetry @ antisymmetry.T / np.count_nonzero(selected)
    raw_gram = 0.5 * (raw_gram + raw_gram.T)
    count = positive.shape[0]
    initial = np.full(count, 1.0 / count)
    uniform_objective = float(initial @ raw_gram @ initial)
    if uniform_objective <= np.finfo(float).tiny:
        return TemporalAntisymmetryResult(
            weights=initial,
            objective=uniform_objective,
            uniform_objective=uniform_objective,
            objective_scale=1.0,
            scaled_objective=0.0,
            effective_window_count=float(count),
            iterations=0,
            optimizer_message="all selected branch differences are zero",
        )

    # SLSQP uses an absolute stopping tolerance.  Whitened CCF amplitudes can
    # make the raw quadratic objective ~1e-13, causing a false one-iteration
    # convergence at the uniform start.  Dividing the positive-semidefinite
    # quadratic form by its uniform-weight value preserves its minimizer and
    # makes the fit invariant to a global rescaling of all CCF amplitudes.
    gram = raw_gram / uniform_objective

    def objective(weights: np.ndarray) -> float:
        return float(weights @ gram @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * gram @ weights

    fit = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"maxiter": maximum_iterations, "ftol": tolerance, "disp": False},
    )
    if not fit.success:
        raise RuntimeError(f"temporal antisymmetry optimization failed: {fit.message}")
    weights = np.asarray(fit.x, dtype=np.float64)
    weights[np.abs(weights) < 10.0 * np.finfo(float).eps] = 0.0
    weights /= np.sum(weights)
    scaled_objective = objective(weights)
    if scaled_objective > 1.0 + max(tolerance, 1e-12):
        raise RuntimeError("temporal optimizer increased antisymmetry objective")
    fitted_objective = float(weights @ raw_gram @ weights)
    effective = 1.0 / float(np.sum(np.square(weights)))
    return TemporalAntisymmetryResult(
        weights=weights,
        objective=fitted_objective,
        uniform_objective=uniform_objective,
        objective_scale=uniform_objective,
        scaled_objective=scaled_objective,
        effective_window_count=effective,
        iterations=int(fit.nit),
        optimizer_message=str(fit.message),
    )
