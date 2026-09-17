"""Deployable pair/branch cross-fit coherence weights for NCF stacking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .processing import NCFPairBranches
from .spz_selection import effective_window_count, enforce_effective_count_floor


@dataclass(frozen=True)
class PairCrossfitCoherenceWeights:
    positive_weights: np.ndarray
    negative_weights: np.ndarray
    positive_score: np.ndarray
    negative_score: np.ndarray
    positive_arrival_correlation: np.ndarray
    negative_arrival_correlation: np.ndarray
    positive_phase_coherence: np.ndarray
    negative_phase_coherence: np.ndarray
    positive_prearrival_fraction: np.ndarray
    negative_prearrival_fraction: np.ndarray
    positive_prearrival_weights: np.ndarray
    negative_prearrival_weights: np.ndarray
    fold_index: np.ndarray
    positive_effective_window_count: np.ndarray
    negative_effective_window_count: np.ndarray
    positive_prearrival_effective_window_count: np.ndarray
    negative_prearrival_effective_window_count: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    scope: str = "pair_branch_contiguous_crossfit_data_only"


def _robust_weights(scores: np.ndarray, minimum_effective_fraction: float) -> np.ndarray:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    if mad <= np.finfo(float).eps:
        return np.ones(scores.size, dtype=np.float64)
    standardized = (scores - median) / (1.4826 * mad)
    raw = np.exp(np.clip(standardized, -3.0, 3.0))
    return enforce_effective_count_floor(raw, minimum_effective_fraction)


def fit_pair_crossfit_coherence_weights(
    windows: Sequence[NCFPairBranches],
    *,
    dt_s: float,
    frequency_range_hz: tuple[float, float],
    minimum_velocity_mps: float,
    maximum_velocity_mps: float,
    folds: int,
    minimum_effective_fraction: float,
) -> PairCrossfitCoherenceWeights:
    """Score held-out windows against a training-only pair/branch reference."""
    if len(windows) < 2 or not 2 <= folds <= len(windows):
        raise ValueError("cross-fit selection requires 2 <= folds <= windows")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not 0 < frequency_range_hz[0] < frequency_range_hz[1]:
        raise ValueError("frequency_range_hz must be positive and increasing")
    if not 0 < minimum_velocity_mps < maximum_velocity_mps:
        raise ValueError("velocity bounds are invalid")
    if not 0 < minimum_effective_fraction <= 1:
        raise ValueError("minimum_effective_fraction must lie in (0, 1]")
    template = windows[0].positive
    sample_count, pair_count = template.data.shape
    lag = np.asarray(template.lag_time_s, dtype=np.float64)
    if lag.size != sample_count or not np.allclose(np.diff(lag), dt_s):
        raise ValueError("pair NCF lag sampling is inconsistent with dt_s")
    for window in windows:
        for branch_name in ("positive", "negative"):
            branch = getattr(window, branch_name)
            if (
                branch.data.shape != template.data.shape
                or not np.array_equal(branch.lag_time_s, template.lag_time_s)
                or not np.array_equal(branch.pair_i, template.pair_i)
                or not np.array_equal(branch.pair_j, template.pair_j)
            ):
                raise ValueError("pair cross-fit windows are not aligned")
    frequency = np.fft.rfftfreq(sample_count, dt_s)
    selected_frequency = (frequency >= frequency_range_hz[0]) & (
        frequency <= frequency_range_hz[1]
    )
    if np.count_nonzero(selected_frequency) < 2:
        raise ValueError("cross-fit band contains fewer than two frequencies")
    time = lag[:, None]
    distance = template.distance_m[None, :]
    arrival_mask = (time >= distance / maximum_velocity_mps) & (
        time <= distance / minimum_velocity_mps
    )
    prearrival_mask = time < distance / maximum_velocity_mps
    fold_index = np.minimum(
        folds - 1,
        np.floor(np.arange(len(windows)) * folds / len(windows)).astype(np.int64),
    )
    epsilon = np.finfo(float).eps

    def score_branch(branch_name: str) -> tuple[np.ndarray, ...]:
        cube = np.stack(
            [getattr(window, branch_name).data for window in windows], axis=0
        ).astype(np.float64)
        spectrum = np.fft.rfft(cube, axis=1)
        band_spectrum = np.zeros_like(spectrum)
        band_spectrum[:, selected_frequency, :] = spectrum[
            :, selected_frequency, :
        ]
        band_cube = np.fft.irfft(band_spectrum, n=sample_count, axis=1)
        arrival_correlation = np.zeros((len(windows), pair_count))
        phase_coherence = np.zeros_like(arrival_correlation)
        prearrival_fraction = np.zeros_like(arrival_correlation)
        for fold in range(folds):
            held_out = fold_index == fold
            training = ~held_out
            reference = np.mean(band_cube[training], axis=0)
            reference_spectrum = np.mean(
                spectrum[training][:, selected_frequency, :], axis=0
            )
            for pair_index in range(pair_count):
                signal = arrival_mask[:, pair_index]
                prearrival = prearrival_mask[:, pair_index]
                reference_signal = reference[signal, pair_index]
                reference_norm = max(
                    float(np.linalg.norm(reference_signal)), epsilon
                )
                for window_index in np.flatnonzero(held_out):
                    candidate_signal = band_cube[
                        window_index, signal, pair_index
                    ]
                    arrival_correlation[window_index, pair_index] = float(
                        np.dot(candidate_signal, reference_signal)
                        / max(
                            float(np.linalg.norm(candidate_signal))
                            * reference_norm,
                            epsilon,
                        )
                    )
                    candidate_spectrum = spectrum[
                        window_index, selected_frequency, pair_index
                    ]
                    cross_phase = candidate_spectrum * np.conj(
                        reference_spectrum[:, pair_index]
                    )
                    valid = np.abs(cross_phase) > epsilon
                    phase_coherence[window_index, pair_index] = (
                        float(np.mean(np.cos(np.angle(cross_phase[valid]))))
                        if np.any(valid)
                        else -1.0
                    )
                    candidate = band_cube[window_index, :, pair_index]
                    prearrival_fraction[window_index, pair_index] = float(
                        np.sum(np.square(candidate[prearrival]))
                        / max(float(np.sum(np.square(candidate))), epsilon)
                    )
        score = 0.5 * (arrival_correlation + phase_coherence) - 0.5 * (
            prearrival_fraction
        )
        weights = np.column_stack(
            [
                _robust_weights(score[:, pair_index], minimum_effective_fraction)
                for pair_index in range(pair_count)
            ]
        )
        effective = np.asarray(
            [effective_window_count(weights[:, index]) for index in range(pair_count)]
        )
        prearrival_weights = np.column_stack(
            [
                _robust_weights(
                    -prearrival_fraction[:, pair_index],
                    minimum_effective_fraction,
                )
                for pair_index in range(pair_count)
            ]
        )
        prearrival_effective = np.asarray(
            [
                effective_window_count(prearrival_weights[:, index])
                for index in range(pair_count)
            ]
        )
        return (
            weights,
            score,
            arrival_correlation,
            phase_coherence,
            prearrival_fraction,
            effective,
            prearrival_weights,
            prearrival_effective,
        )

    positive = score_branch("positive")
    negative = score_branch("negative")
    return PairCrossfitCoherenceWeights(
        positive_weights=positive[0],
        negative_weights=negative[0],
        positive_score=positive[1],
        negative_score=negative[1],
        positive_arrival_correlation=positive[2],
        negative_arrival_correlation=negative[2],
        positive_phase_coherence=positive[3],
        negative_phase_coherence=negative[3],
        positive_prearrival_fraction=positive[4],
        negative_prearrival_fraction=negative[4],
        positive_prearrival_weights=positive[6],
        negative_prearrival_weights=negative[6],
        fold_index=fold_index,
        positive_effective_window_count=positive[5],
        negative_effective_window_count=negative[5],
        positive_prearrival_effective_window_count=positive[7],
        negative_prearrival_effective_window_count=negative[7],
        pair_i=template.pair_i.copy(),
        pair_j=template.pair_j.copy(),
    )
