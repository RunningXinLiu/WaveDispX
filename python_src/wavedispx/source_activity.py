"""Load reproducible time-varying ambient-source activity schedules."""

from __future__ import annotations

import numpy as np

from .config import AmbientSourceConfig


def load_source_activity_matrix(
    config: AmbientSourceConfig,
    window_count: int,
) -> np.ndarray:
    """Return window-by-source amplitude multipliers.

    Without a schedule every source has multiplier one.  A schedule is a long
    CSV with ``window_index,source_index,amplitude_multiplier``.  Unlisted
    entries are zero, making the active source set explicit and auditable.
    """
    if window_count < 1:
        raise ValueError("window_count must be positive")
    if config.activity_file is None:
        return np.ones((window_count, config.count), dtype=np.float64)
    table = np.genfromtxt(
        config.activity_file,
        delimiter=",",
        names=True,
        dtype=np.float64,
    )
    names = table.dtype.names
    required = {"window_index", "source_index", "amplitude_multiplier"}
    if names is None or not required.issubset(names):
        raise ValueError(
            "source activity CSV must contain window_index,source_index,"
            "amplitude_multiplier headers"
        )
    window_index = np.atleast_1d(table["window_index"])
    source_index = np.atleast_1d(table["source_index"])
    amplitude = np.atleast_1d(table["amplitude_multiplier"])
    if (
        np.any(~np.isfinite(window_index))
        or np.any(~np.isfinite(source_index))
        or np.any(~np.isfinite(amplitude))
        or np.any(window_index != np.floor(window_index))
        or np.any(source_index != np.floor(source_index))
        or np.any(window_index < 0)
        or np.any(window_index >= window_count)
        or np.any(source_index < 0)
        or np.any(source_index >= config.count)
        or np.any(amplitude < 0)
    ):
        raise ValueError("source activity rows contain invalid indexes or amplitudes")
    window_index = window_index.astype(np.int64)
    source_index = source_index.astype(np.int64)
    flat_index = window_index * config.count + source_index
    if np.unique(flat_index).size != flat_index.size:
        raise ValueError("source activity CSV contains duplicate window/source rows")
    activity = np.zeros((window_count, config.count), dtype=np.float64)
    activity[window_index, source_index] = amplitude
    if np.any(np.sum(activity**2, axis=1) == 0):
        raise ValueError("every source-activity window must contain positive power")
    return activity
