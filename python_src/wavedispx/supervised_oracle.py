"""Synthetic-only EGF-supervised upper bound for reachability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import nnls

from .egf_reference import PassiveEgfReference
from .processing import NCFBranches, NCFPairBranches
from .spz_selection import effective_window_count, enforce_effective_count_floor


@dataclass(frozen=True)
class SupervisedEgfOracleWeights:
    positive: np.ndarray
    negative: np.ndarray
    positive_effective_window_count: float
    negative_effective_window_count: float
    scope: str = "synthetic_theoretical_egf_supervised_upper_bound"


@dataclass(frozen=True)
class SupervisedPairEgfOracleWeights:
    positive: np.ndarray
    negative: np.ndarray
    positive_effective_window_count: np.ndarray
    negative_effective_window_count: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    positive_fit_success: np.ndarray
    negative_fit_success: np.ndarray
    scope: str = "synthetic_pairwise_theoretical_egf_supervised_upper_bound"


def _bandpass(data: np.ndarray, dt_s: float, band_hz: tuple[float, float]) -> np.ndarray:
    frequency = np.fft.rfftfreq(data.shape[0], dt_s)
    selected = (frequency >= band_hz[0]) & (frequency <= band_hz[1])
    spectrum = np.fft.rfft(data, axis=0)
    spectrum[~selected] = 0.0
    return np.fft.irfft(spectrum, n=data.shape[0], axis=0)


def _reference_on_distance_axis(
    reference: PassiveEgfReference,
    distance_m: np.ndarray,
    sample_count: int,
    component_name: str,
) -> np.ndarray:
    if component_name not in reference.component_names:
        raise ValueError(f"reference has no {component_name!r} component")
    component_index = reference.component_names.index(component_name)
    component = reference.components[:sample_count, :, component_index]
    pair_distance = reference.distance_m[reference.pair_distance_index]
    pair_bin = np.abs(pair_distance[:, None] - distance_m[None, :]).argmin(axis=1)
    output = np.empty((sample_count, distance_m.size), dtype=np.float64)
    pair_component = component[:, reference.pair_distance_index]
    for distance_index in range(distance_m.size):
        selected = pair_bin == distance_index
        if not np.any(selected):
            raise ValueError("active reference does not cover every NCF distance")
        output[:, distance_index] = np.mean(pair_component[:, selected], axis=1)
    return output


def fit_supervised_egf_oracle_weights(
    windows: Sequence[NCFBranches],
    reference: PassiveEgfReference,
    *,
    component_name: str,
    frequency_range_hz: tuple[float, float],
    minimum_velocity_mps: float,
    maximum_velocity_mps: float,
    minimum_effective_fraction: float,
    prearrival_penalty: float = 0.25,
    phase_equalization_weight: float = 1.0,
) -> SupervisedEgfOracleWeights:
    """Fit branch weights to the known EGF, solely to test what is achievable.

    This deliberately leaks the theoretical target and must never be presented
    as a field-data selector. Non-negative least squares is followed by the
    same hard effective-window floor used by the deployable experiments.
    """
    if len(windows) < 2:
        raise ValueError("at least two NCF windows are required")
    if not 0 < minimum_effective_fraction <= 1:
        raise ValueError("minimum_effective_fraction must lie in (0, 1]")
    if not np.isfinite(prearrival_penalty) or prearrival_penalty < 0:
        raise ValueError("prearrival_penalty must be finite and non-negative")
    if (
        not np.isfinite(phase_equalization_weight)
        or phase_equalization_weight < 0
    ):
        raise ValueError("phase_equalization_weight must be finite and non-negative")
    template = windows[0].symmetric
    lag = np.asarray(template.lag_time_s, dtype=np.float64)
    distance = np.asarray(template.distance_m, dtype=np.float64)
    if lag.size < 2 or not np.allclose(np.diff(lag), np.diff(lag)[0]):
        raise ValueError("NCF lag axis must be uniformly sampled")
    dt_s = float(lag[1] - lag[0])
    sample_count = min(lag.size, reference.time_s.size)
    reference_data = _reference_on_distance_axis(
        reference, distance, sample_count, component_name
    )
    reference_band = _bandpass(
        reference_data, dt_s, frequency_range_hz
    )
    time = lag[:sample_count, None]
    distance_grid = distance[None, :]
    signal = (time >= distance_grid / maximum_velocity_mps) & (
        time <= distance_grid / minimum_velocity_mps
    )
    prearrival = time < distance_grid / maximum_velocity_mps
    target_parts = [reference_band[signal]]
    if prearrival_penalty > 0:
        target_parts.append(np.zeros(np.count_nonzero(prearrival)))
    time_target = np.concatenate(target_parts)
    frequency = np.fft.rfftfreq(sample_count, dt_s)
    selected_frequency = (frequency >= frequency_range_hz[0]) & (
        frequency <= frequency_range_hz[1]
    )
    reference_spectrum = np.fft.rfft(reference_data, axis=0)[selected_frequency]
    reference_phase = reference_spectrum / np.maximum(
        np.abs(reference_spectrum), np.finfo(float).eps
    )
    phase_target = np.concatenate(
        (reference_phase.real.reshape(-1), reference_phase.imag.reshape(-1))
    )

    def branch_weights(branch_name: str) -> np.ndarray:
        time_columns = []
        phase_columns = []
        for window in windows:
            gather = getattr(window, branch_name)
            if (
                not np.array_equal(gather.lag_time_s, template.lag_time_s)
                or not np.array_equal(gather.distance_m, template.distance_m)
            ):
                raise ValueError("supervised Oracle windows are not aligned")
            candidate = np.asarray(gather.data[:sample_count], dtype=np.float64)
            if reference.calibration_operator == "negative_time_derivative_of_ncf":
                candidate = -np.gradient(candidate, dt_s, axis=0)
            elif reference.calibration_operator not in {
                "identity",
                "none_raw_green_function",
            }:
                raise ValueError("unsupported EGF calibration operator")
            candidate_band = _bandpass(candidate, dt_s, frequency_range_hz)
            parts = [candidate_band[signal]]
            if prearrival_penalty > 0:
                parts.append(
                    np.sqrt(prearrival_penalty) * candidate_band[prearrival]
                )
            time_column = np.concatenate(parts)
            time_columns.append(
                time_column
                / max(float(np.linalg.norm(time_column)), np.finfo(float).eps)
            )
            candidate_spectrum = np.fft.rfft(candidate, axis=0)[
                selected_frequency
            ]
            candidate_phase = candidate_spectrum / np.maximum(
                np.abs(candidate_spectrum), np.finfo(float).eps
            )
            phase_columns.append(
                np.concatenate(
                    (
                        candidate_phase.real.reshape(-1),
                        candidate_phase.imag.reshape(-1),
                    )
                )
            )
        time_design = np.column_stack(time_columns)
        time_scale = max(float(np.linalg.norm(time_target)), np.finfo(float).eps)
        design_blocks = [time_design]
        target_blocks = [time_target / time_scale]
        if phase_equalization_weight > 0:
            phase_design = np.column_stack(phase_columns)
            phase_scale = max(
                float(np.linalg.norm(phase_target)), np.finfo(float).eps
            )
            block_scale = np.sqrt(phase_equalization_weight)
            design_blocks.append(block_scale * phase_design / phase_scale)
            target_blocks.append(block_scale * phase_target / phase_scale)
        design = np.vstack(design_blocks)
        target = np.concatenate(target_blocks)
        scale = max(float(np.linalg.norm(design)), np.finfo(float).eps)
        weights, _ = nnls(
            design / scale,
            target / scale,
            maxiter=max(30, 10 * len(windows)),
        )
        if float(np.sum(weights)) <= np.finfo(float).eps:
            raise ValueError("supervised EGF Oracle found no positive solution")
        return enforce_effective_count_floor(weights, minimum_effective_fraction)

    positive = branch_weights("positive")
    negative = branch_weights("negative")
    return SupervisedEgfOracleWeights(
        positive=positive,
        negative=negative,
        positive_effective_window_count=effective_window_count(positive),
        negative_effective_window_count=effective_window_count(negative),
    )


def fit_supervised_pair_egf_oracle_weights(
    windows: Sequence[NCFPairBranches],
    reference: PassiveEgfReference,
    *,
    component_name: str,
    frequency_range_hz: tuple[float, float],
    minimum_velocity_mps: float,
    maximum_velocity_mps: float,
    minimum_effective_fraction: float,
    prearrival_penalty: float = 0.25,
    phase_equalization_weight: float = 1.0,
) -> SupervisedPairEgfOracleWeights:
    """Fit synthetic-only branch weights independently for every receiver pair."""
    if len(windows) < 2:
        raise ValueError("at least two pair NCF windows are required")
    if component_name not in reference.component_names:
        raise ValueError(f"reference has no {component_name!r} component")
    if not 0 < minimum_effective_fraction <= 1:
        raise ValueError("minimum_effective_fraction must lie in (0, 1]")
    if prearrival_penalty < 0 or phase_equalization_weight < 0:
        raise ValueError("Oracle penalties must be non-negative")
    template = windows[0].positive
    lag = np.asarray(template.lag_time_s, dtype=np.float64)
    if lag.size < 2 or not np.allclose(np.diff(lag), np.diff(lag)[0]):
        raise ValueError("pair NCF lag axis must be uniformly sampled")
    dt_s = float(lag[1] - lag[0])
    sample_count = min(lag.size, reference.time_s.size)
    component_index = reference.component_names.index(component_name)
    reference_pair_lookup = {
        (int(pair_i), int(pair_j)): pair_index
        for pair_index, (pair_i, pair_j) in enumerate(
            zip(reference.pair_i, reference.pair_j)
        )
    }
    reference_pair_index = np.asarray(
        [
            reference_pair_lookup[(int(pair_i), int(pair_j))]
            for pair_i, pair_j in zip(template.pair_i, template.pair_j)
        ],
        dtype=np.int64,
    )
    reference_distance_index = reference.pair_distance_index[
        reference_pair_index
    ]
    reference_data = reference.components[
        :sample_count, reference_distance_index, component_index
    ]
    frequency = np.fft.rfftfreq(sample_count, dt_s)
    selected_frequency = (frequency >= frequency_range_hz[0]) & (
        frequency <= frequency_range_hz[1]
    )
    epsilon = np.finfo(float).eps

    def solve_pair(branch_name: str, pair_index: int) -> tuple[np.ndarray, bool]:
        target_trace = reference_data[:, pair_index]
        target_band = _bandpass(
            target_trace[:, None], dt_s, frequency_range_hz
        )[:, 0]
        signal = (lag[:sample_count] >= template.distance_m[pair_index] / maximum_velocity_mps) & (
            lag[:sample_count] <= template.distance_m[pair_index] / minimum_velocity_mps
        )
        prearrival = lag[:sample_count] < (
            template.distance_m[pair_index] / maximum_velocity_mps
        )
        time_target_parts = [target_band[signal]]
        if prearrival_penalty > 0:
            time_target_parts.append(np.zeros(np.count_nonzero(prearrival)))
        time_target = np.concatenate(time_target_parts)
        time_target /= max(float(np.linalg.norm(time_target)), epsilon)
        target_spectrum = np.fft.rfft(target_trace)[selected_frequency]
        target_phase = target_spectrum / np.maximum(np.abs(target_spectrum), epsilon)
        phase_target = np.concatenate((target_phase.real, target_phase.imag))
        phase_target /= max(float(np.linalg.norm(phase_target)), epsilon)
        time_columns = []
        phase_columns = []
        for window in windows:
            branch = getattr(window, branch_name)
            if (
                not np.array_equal(branch.lag_time_s, template.lag_time_s)
                or not np.array_equal(branch.pair_i, template.pair_i)
                or not np.array_equal(branch.pair_j, template.pair_j)
            ):
                raise ValueError("pair Oracle windows are not aligned")
            candidate = np.asarray(
                branch.data[:sample_count, pair_index], dtype=np.float64
            )
            if reference.calibration_operator == "negative_time_derivative_of_ncf":
                candidate = -np.gradient(candidate, dt_s)
            elif reference.calibration_operator not in {
                "identity",
                "none_raw_green_function",
            }:
                raise ValueError("unsupported EGF calibration operator")
            candidate_band = _bandpass(
                candidate[:, None], dt_s, frequency_range_hz
            )[:, 0]
            time_parts = [candidate_band[signal]]
            if prearrival_penalty > 0:
                time_parts.append(
                    np.sqrt(prearrival_penalty) * candidate_band[prearrival]
                )
            time_column = np.concatenate(time_parts)
            time_columns.append(
                time_column / max(float(np.linalg.norm(time_column)), epsilon)
            )
            candidate_spectrum = np.fft.rfft(candidate)[selected_frequency]
            candidate_phase = candidate_spectrum / np.maximum(
                np.abs(candidate_spectrum), epsilon
            )
            phase_column = np.concatenate(
                (candidate_phase.real, candidate_phase.imag)
            )
            phase_columns.append(
                phase_column
                / max(float(np.linalg.norm(phase_column)), epsilon)
            )
        design_blocks = [np.column_stack(time_columns)]
        target_blocks = [time_target]
        if phase_equalization_weight > 0:
            scale = np.sqrt(phase_equalization_weight)
            design_blocks.append(scale * np.column_stack(phase_columns))
            target_blocks.append(scale * phase_target)
        design = np.vstack(design_blocks)
        target = np.concatenate(target_blocks)
        weights, _ = nnls(
            design,
            target,
            maxiter=max(30, 10 * len(windows)),
        )
        if float(np.sum(weights)) <= epsilon:
            return np.ones(len(windows), dtype=np.float64), False
        return (
            enforce_effective_count_floor(weights, minimum_effective_fraction),
            True,
        )

    positive_results = [
        solve_pair("positive", pair_index)
        for pair_index in range(template.pair_i.size)
    ]
    negative_results = [
        solve_pair("negative", pair_index)
        for pair_index in range(template.pair_i.size)
    ]
    positive = np.column_stack([result[0] for result in positive_results])
    negative = np.column_stack([result[0] for result in negative_results])
    return SupervisedPairEgfOracleWeights(
        positive=positive,
        negative=negative,
        positive_effective_window_count=np.asarray(
            [effective_window_count(positive[:, index]) for index in range(positive.shape[1])]
        ),
        negative_effective_window_count=np.asarray(
            [effective_window_count(negative[:, index]) for index in range(negative.shape[1])]
        ),
        pair_i=template.pair_i.copy(),
        pair_j=template.pair_j.copy(),
        positive_fit_success=np.asarray(
            [result[1] for result in positive_results], dtype=bool
        ),
        negative_fit_success=np.asarray(
            [result[1] for result in negative_results], dtype=bool
        ),
    )
