"""Cross-fitted, data-only tau-p weights for pair/branch NCF stacking.

The selector estimates the surface-wave slowness supported by the training
windows of a virtual-source gather.  Held-out windows are scored by the
fraction of their beam power that remains phase-compatible with that training
slowness for each receiver distance.  It never reads source coordinates or a
theoretical/active EGF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .processing import NCFPairBranches
from .spz_selection import effective_window_count, enforce_effective_count_floor


@dataclass(frozen=True)
class PairTauPSelection:
    band_center_hz: np.ndarray
    slowness_s_per_m: np.ndarray
    reference_slowness_s_per_m: np.ndarray
    positive_band_weights: np.ndarray
    negative_band_weights: np.ndarray
    positive_spz_score: np.ndarray
    negative_spz_score: np.ndarray
    directional_imbalance: np.ndarray
    diffuse_soft_band_weights: np.ndarray
    diffuse_hard_band_weights: np.ndarray
    diffuse_prearrival_hard_band_weights: np.ndarray | None
    diffuse_soft_effective_window_count: np.ndarray
    diffuse_hard_effective_window_count: np.ndarray
    positive_effective_window_count: np.ndarray
    negative_effective_window_count: np.ndarray
    fold_index: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    nnws_v2_directional_guard_band_weights: np.ndarray | None = None
    nnws_v2_directional_guard_control_band_weights: np.ndarray | None = None
    nnws_v2_positive_branch_guard_band_weights: np.ndarray | None = None
    nnws_v2_negative_branch_guard_band_weights: np.ndarray | None = None
    nnws_v2_positive_branch_guard_control_band_weights: np.ndarray | None = None
    nnws_v2_negative_branch_guard_control_band_weights: np.ndarray | None = None
    nnws_v2_positive_diffuse_prearrival_band_weights: np.ndarray | None = None
    nnws_v2_negative_diffuse_prearrival_band_weights: np.ndarray | None = None
    nnws_v2_positive_diffuse_prearrival_control_band_weights: np.ndarray | None = None
    nnws_v2_negative_diffuse_prearrival_control_band_weights: np.ndarray | None = None
    nnws_v2_directional_fraction: float | None = None
    nnws_v2_diffuse_prearrival_frequency_fraction: float | None = None
    diffuse_prearrival_directional_fraction: float | None = None
    scope: str = "pair_branch_frequency_contiguous_crossfit_tau_p_data_only"


def _fractional_rank(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size < 2 or np.any(~np.isfinite(data)):
        raise ValueError("rank input must contain at least two finite values")
    order = np.argsort(data, kind="stable")
    rank = np.empty(data.size, dtype=np.float64)
    sorted_data = data[order]
    start = 0
    while start < data.size:
        stop = start + 1
        while stop < data.size and sorted_data[stop] == sorted_data[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) / (data.size - 1)
        rank[order[start:stop]] = average_rank
        start = stop
    return rank


def joint_diffuse_prearrival_hard_weights(
    directional_imbalance: np.ndarray,
    positive_prearrival_fraction: np.ndarray,
    negative_prearrival_fraction: np.ndarray,
    *,
    retention_fraction: float = 0.5,
    directional_fraction: float = 0.75,
) -> np.ndarray:
    """Keep windows jointly low in azimuth imbalance and pre-arrival energy."""

    imbalance = np.asarray(directional_imbalance, dtype=np.float64)
    positive = np.asarray(positive_prearrival_fraction, dtype=np.float64)
    negative = np.asarray(negative_prearrival_fraction, dtype=np.float64)
    if (
        imbalance.ndim != 2
        or positive.ndim != 2
        or negative.shape != positive.shape
        or positive.shape[0] != imbalance.shape[0]
        or np.any(~np.isfinite(imbalance))
        or np.any(~np.isfinite(positive))
        or np.any(~np.isfinite(negative))
    ):
        raise ValueError("directional and pre-arrival scores must be finite and aligned")
    if not 0.0 < retention_fraction <= 1.0:
        raise ValueError("retention_fraction must lie in (0, 1]")
    if not 0.0 <= directional_fraction <= 1.0:
        raise ValueError("directional_fraction must lie in [0, 1]")
    retained_count = max(1, int(np.ceil(retention_fraction * imbalance.shape[0])))
    prearrival = np.median(0.5 * (positive + negative), axis=1)
    prearrival_rank = _fractional_rank(prearrival)
    weights = np.zeros_like(imbalance)
    for band_index in range(imbalance.shape[1]):
        combined_rank = (
            directional_fraction * _fractional_rank(imbalance[:, band_index])
            + (1.0 - directional_fraction) * prearrival_rank
        )
        weights[
            np.argsort(combined_rank, kind="stable")[:retained_count], band_index
        ] = 1.0
    return weights


def _score_to_weights(
    scores: np.ndarray, minimum_effective_fraction: float
) -> np.ndarray:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    if mad <= np.finfo(float).eps:
        return np.ones(scores.size, dtype=np.float64)
    standardized = (scores - median) / (1.4826 * mad)
    raw = np.exp(np.clip(standardized, -3.0, 3.0))
    return enforce_effective_count_floor(raw, minimum_effective_fraction)


def _beam_power(
    cube: np.ndarray,
    *,
    frequency_hz: np.ndarray,
    selected_frequency: np.ndarray,
    distance_m: np.ndarray,
    slowness_s_per_m: np.ndarray,
) -> np.ndarray:
    spectrum = np.fft.rfft(cube, axis=1)[:, selected_frequency, :]
    amplitude = np.abs(spectrum)
    phase_only = np.divide(
        spectrum,
        amplitude,
        out=np.zeros_like(spectrum),
        where=amplitude > np.finfo(float).eps,
    )
    selected_hz = frequency_hz[selected_frequency]
    steering = np.exp(
        2j
        * np.pi
        * selected_hz[:, None, None]
        * distance_m[None, :, None]
        * slowness_s_per_m[None, None, :]
    )
    beam = np.einsum("wfp,fps->wfs", phase_only, steering, optimize=True)
    beam /= max(distance_m.size, 1)
    return np.mean(np.square(np.abs(beam)), axis=1)


def fit_pair_crossfit_tau_p_weights(
    windows: Sequence[NCFPairBranches],
    *,
    dt_s: float,
    frequency_bands_hz: Sequence[tuple[float, float]],
    minimum_velocity_mps: float,
    maximum_velocity_mps: float,
    phase_tolerance_rad: float,
    folds: int,
    minimum_effective_fraction: float,
    slowness_count: int = 96,
) -> PairTauPSelection:
    """Fit pair/branch/frequency weights without source or EGF information."""
    if len(windows) < 2 or not 2 <= folds <= len(windows):
        raise ValueError("tau-p selection requires 2 <= folds <= windows")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not 0 < minimum_velocity_mps < maximum_velocity_mps:
        raise ValueError("velocity bounds are invalid")
    if not np.isfinite(phase_tolerance_rad) or phase_tolerance_rad <= 0:
        raise ValueError("phase_tolerance_rad must be finite and positive")
    if not 0 < minimum_effective_fraction <= 1:
        raise ValueError("minimum_effective_fraction must lie in (0, 1]")
    if slowness_count < 16:
        raise ValueError("slowness_count must be at least 16")
    bands = np.asarray(frequency_bands_hz, dtype=np.float64)
    if (
        bands.ndim != 2
        or bands.shape[1] != 2
        or bands.shape[0] < 1
        or np.any(~np.isfinite(bands))
        or np.any(bands[:, 0] <= 0)
        or np.any(bands[:, 1] <= bands[:, 0])
    ):
        raise ValueError("frequency_bands_hz must contain positive ordered bands")

    template = windows[0].positive
    sample_count, pair_count = template.data.shape
    lag = np.asarray(template.lag_time_s, dtype=np.float64)
    distance = np.asarray(template.distance_m, dtype=np.float64)
    if (
        lag.size != sample_count
        or not np.allclose(np.diff(lag), dt_s)
        or pair_count < 2
        or np.any(distance <= 0)
    ):
        raise ValueError("tau-p selection requires an aligned multi-distance gather")
    for window in windows:
        for branch_name in ("positive", "negative"):
            branch = getattr(window, branch_name)
            if (
                branch.data.shape != template.data.shape
                or not np.array_equal(branch.lag_time_s, template.lag_time_s)
                or not np.array_equal(branch.pair_i, template.pair_i)
                or not np.array_equal(branch.pair_j, template.pair_j)
                or not np.array_equal(branch.distance_m, template.distance_m)
            ):
                raise ValueError("tau-p windows are not aligned")

    frequency = np.fft.rfftfreq(sample_count, dt_s)
    slowness = np.linspace(
        1.0 / maximum_velocity_mps,
        1.0 / minimum_velocity_mps,
        slowness_count,
    )
    fold_index = np.minimum(
        folds - 1,
        np.floor(np.arange(len(windows)) * folds / len(windows)).astype(np.int64),
    )
    positive_cube = np.stack(
        [window.positive.data for window in windows], axis=0
    ).astype(np.float64)
    negative_cube = np.stack(
        [window.negative.data for window in windows], axis=0
    ).astype(np.float64)
    band_center = np.mean(bands, axis=1)
    reference_slowness = np.zeros((folds, bands.shape[0]), dtype=np.float64)
    positive_score = np.zeros((len(windows), bands.shape[0], pair_count))
    negative_score = np.zeros_like(positive_score)

    for band_index, (lower, upper) in enumerate(bands):
        selected = (frequency >= lower) & (frequency <= upper)
        if np.count_nonzero(selected) < 2:
            raise ValueError("a tau-p frequency band contains fewer than two bins")
        positive_power = _beam_power(
            positive_cube,
            frequency_hz=frequency,
            selected_frequency=selected,
            distance_m=distance,
            slowness_s_per_m=slowness,
        )
        negative_power = _beam_power(
            negative_cube,
            frequency_hz=frequency,
            selected_frequency=selected,
            distance_m=distance,
            slowness_s_per_m=slowness,
        )
        for fold in range(folds):
            held_out = fold_index == fold
            training = ~held_out
            training_power = np.mean(
                positive_power[training] + negative_power[training], axis=0
            )
            reference_index = int(np.argmax(training_power))
            reference = float(slowness[reference_index])
            reference_slowness[fold, band_index] = reference
            mismatch = (
                2.0
                * np.pi
                * band_center[band_index]
                * distance[:, None]
                * (slowness[None, :] - reference)
            )
            kernel = np.exp(
                -0.5 * np.square(mismatch / phase_tolerance_rad)
            )
            for window_index in np.flatnonzero(held_out):
                for output, power in (
                    (positive_score, positive_power[window_index]),
                    (negative_score, negative_power[window_index]),
                ):
                    denominator = max(float(np.sum(power)), np.finfo(float).eps)
                    output[window_index, band_index] = (
                        kernel @ power / denominator
                    )

    def weights_and_effective(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weights = np.empty_like(score)
        effective = np.empty((bands.shape[0], pair_count), dtype=np.float64)
        for band_index in range(bands.shape[0]):
            for pair_index in range(pair_count):
                vector = _score_to_weights(
                    score[:, band_index, pair_index],
                    minimum_effective_fraction,
                )
                weights[:, band_index, pair_index] = vector
                effective[band_index, pair_index] = effective_window_count(vector)
        return weights, effective

    positive_weights, positive_effective = weights_and_effective(positive_score)
    negative_weights, negative_effective = weights_and_effective(negative_score)
    directional_imbalance = np.mean(
        np.abs(positive_score - negative_score)
        / np.maximum(positive_score + negative_score, np.finfo(float).eps),
        axis=2,
    )
    diffuse_soft_weights = np.column_stack(
        [
            _score_to_weights(
                -directional_imbalance[:, band_index],
                minimum_effective_fraction,
            )
            for band_index in range(bands.shape[0])
        ]
    )
    retained_count = max(
        1, int(np.ceil(minimum_effective_fraction * len(windows)))
    )
    diffuse_hard_weights = np.zeros(
        (len(windows), bands.shape[0]), dtype=np.float64
    )
    diffuse_rank_score = np.mean(directional_imbalance, axis=1)
    diffuse_hard_weights[
        np.argsort(diffuse_rank_score, kind="stable")[:retained_count]
    ] = 1.0
    return PairTauPSelection(
        band_center_hz=band_center,
        slowness_s_per_m=slowness,
        reference_slowness_s_per_m=reference_slowness,
        positive_band_weights=positive_weights,
        negative_band_weights=negative_weights,
        positive_spz_score=positive_score,
        negative_spz_score=negative_score,
        directional_imbalance=directional_imbalance,
        diffuse_soft_band_weights=diffuse_soft_weights,
        diffuse_hard_band_weights=diffuse_hard_weights,
        diffuse_prearrival_hard_band_weights=None,
        diffuse_soft_effective_window_count=np.asarray(
            [
                effective_window_count(diffuse_soft_weights[:, band_index])
                for band_index in range(bands.shape[0])
            ]
        ),
        diffuse_hard_effective_window_count=np.asarray(
            [
                effective_window_count(diffuse_hard_weights[:, band_index])
                for band_index in range(bands.shape[0])
            ]
        ),
        positive_effective_window_count=positive_effective,
        negative_effective_window_count=negative_effective,
        fold_index=fold_index,
        pair_i=template.pair_i.copy(),
        pair_j=template.pair_j.copy(),
    )
