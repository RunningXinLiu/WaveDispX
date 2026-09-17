"""Quantitative dispersion-image and pick validation against theory."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .egf_reference import PassiveEgfReference
from .processing import DispersionImage, DispersionPicks, NCFGather


def egf_equivalence_metrics(
    ncf: NCFGather,
    reference: PassiveEgfReference,
    *,
    component_name: str,
    frequency_range_hz: tuple[float, float],
    minimum_velocity_mps: float,
    maximum_velocity_mps: float,
    correlation_normalization: str,
) -> dict[str, float | int | bool | str]:
    """Compare an NCF branch with the raw active reference without time shifting.

    Phase, arrival, unit-norm waveform, single-scalar spectral-shape, and early
    ghost metrics are always reported.  Absolute amplitude is reported only
    when both the reference calibration and receiver preprocessing preserve a
    physical amplitude convention.
    """
    if component_name not in reference.component_names:
        raise ValueError(f"reference has no {component_name!r} component")
    if minimum_velocity_mps <= 0 or maximum_velocity_mps <= minimum_velocity_mps:
        raise ValueError("velocity bounds must satisfy 0 < minimum < maximum")
    lag = np.asarray(ncf.lag_time_s, dtype=np.float64).reshape(-1)
    if lag.size < 2 or not np.allclose(np.diff(lag), np.diff(lag)[0]):
        raise ValueError("NCF lag_time_s must be uniformly sampled")
    dt = float(lag[1] - lag[0])
    if not np.isclose(dt, reference.time_s[1] - reference.time_s[0]):
        raise ValueError("NCF and active reference sampling intervals differ")
    sample_count = min(lag.size, reference.time_s.size)
    candidate = np.asarray(ncf.data[:sample_count], dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape[1] != ncf.distance_m.size:
        raise ValueError("NCF data must be lag-by-distance")
    if reference.calibration_operator == "negative_time_derivative_of_ncf":
        candidate = -np.gradient(candidate, dt, axis=0)
    elif reference.calibration_operator not in {
        "identity",
        "none_raw_green_function",
    }:
        raise ValueError(
            f"unsupported EGF calibration operator: {reference.calibration_operator}"
        )
    component_index = reference.component_names.index(component_name)
    reference_component = reference.components[
        :sample_count, :, component_index
    ]
    target_reference = np.empty_like(candidate)
    pair_distance = reference.distance_m[reference.pair_distance_index]
    pair_bin = np.abs(
        pair_distance[:, None] - np.asarray(ncf.distance_m)[None, :]
    ).argmin(axis=1)
    aggregated_pair_count = np.bincount(
        pair_bin, minlength=ncf.distance_m.size
    )
    if np.any(aggregated_pair_count == 0):
        raise ValueError("active reference has no receiver pairs for an NCF distance bin")
    pair_reference = reference_component[:, reference.pair_distance_index]
    for distance_index in range(ncf.distance_m.size):
        target_reference[:, distance_index] = np.mean(
            pair_reference[:, pair_bin == distance_index], axis=1
        )

    frequency = np.fft.rfftfreq(sample_count, dt)
    selected_frequency = (
        (frequency >= frequency_range_hz[0])
        & (frequency <= frequency_range_hz[1])
    )
    if np.count_nonzero(selected_frequency) < 2:
        raise ValueError("EGF comparison band contains fewer than two frequency samples")
    candidate_spectrum = np.fft.rfft(candidate, axis=0)
    reference_spectrum = np.fft.rfft(target_reference, axis=0)
    band_candidate_spectrum = np.zeros_like(candidate_spectrum)
    band_reference_spectrum = np.zeros_like(reference_spectrum)
    band_candidate_spectrum[selected_frequency] = candidate_spectrum[selected_frequency]
    band_reference_spectrum[selected_frequency] = reference_spectrum[selected_frequency]
    band_candidate = np.fft.irfft(
        band_candidate_spectrum, n=sample_count, axis=0
    )
    band_reference = np.fft.irfft(
        band_reference_spectrum, n=sample_count, axis=0
    )

    time = lag[:sample_count, None]
    distance = np.asarray(ncf.distance_m, dtype=np.float64)[None, :]
    signal_window = (
        (time >= distance / maximum_velocity_mps)
        & (time <= distance / minimum_velocity_mps)
    )
    candidate_signal = band_candidate[signal_window]
    reference_signal = band_reference[signal_window]
    candidate_norm = float(np.linalg.norm(candidate_signal))
    reference_norm = float(np.linalg.norm(reference_signal))
    epsilon = np.finfo(float).eps
    waveform_correlation = float(
        np.dot(candidate_signal, reference_signal)
        / max(candidate_norm * reference_norm, epsilon)
    )
    normalized_l2 = float(
        np.linalg.norm(
            candidate_signal / max(candidate_norm, epsilon)
            - reference_signal / max(reference_norm, epsilon)
        )
    )
    best_scale = float(
        np.dot(candidate_signal, reference_signal)
        / max(np.dot(candidate_signal, candidate_signal), epsilon)
    )
    scaled_l2 = float(
        np.linalg.norm(best_scale * candidate_signal - reference_signal)
        / max(reference_norm, epsilon)
    )

    selected_candidate = candidate_spectrum[selected_frequency]
    selected_reference = reference_spectrum[selected_frequency]
    reference_amplitude = np.abs(selected_reference)
    valid_phase = reference_amplitude > 1e-8 * reference_amplitude.max(initial=0.0)
    valid_phase &= np.abs(selected_candidate) > 1e-12 * np.abs(selected_candidate).max(
        initial=0.0
    )
    phase_difference = np.angle(
        selected_candidate[valid_phase] * np.conj(selected_reference[valid_phase])
    )
    phase_weight = reference_amplitude[valid_phase]
    phase_rms = float(
        np.sqrt(
            np.sum(phase_weight * phase_difference**2)
            / max(np.sum(phase_weight), epsilon)
        )
    )

    amplitude_candidate = np.abs(selected_candidate[valid_phase])
    amplitude_reference = reference_amplitude[valid_phase]
    log_amplitude_difference = np.log(
        np.maximum(amplitude_candidate, epsilon)
        / np.maximum(amplitude_reference, epsilon)
    )
    log_shape_difference = log_amplitude_difference - np.median(
        log_amplitude_difference
    )
    spectral_shape_log_rms = float(np.sqrt(np.mean(log_shape_difference**2)))
    # The interferometric identity is defined only up to a source/medium scale
    # factor here.  Whitening/one-bit further remove physical amplitude, so no
    # current reference operator establishes absolute amplitude equivalence.
    amplitude_available = False
    absolute_amplitude_log_rms = (
        float(np.sqrt(np.mean(log_amplitude_difference**2)))
        if amplitude_available
        else float("nan")
    )

    candidate_envelope = np.abs(hilbert(band_candidate, axis=0))
    reference_envelope = np.abs(hilbert(band_reference, axis=0))
    peak_bias = []
    for distance_index in range(ncf.distance_m.size):
        selected_time = np.flatnonzero(signal_window[:, distance_index])
        if selected_time.size == 0:
            continue
        candidate_peak = selected_time[
            int(np.argmax(candidate_envelope[selected_time, distance_index]))
        ]
        reference_peak = selected_time[
            int(np.argmax(reference_envelope[selected_time, distance_index]))
        ]
        peak_bias.append((candidate_peak - reference_peak) * dt)
    median_peak_bias = float(np.round(np.median(peak_bias), 12))

    prearrival = time < distance / maximum_velocity_mps
    total_energy = float(np.sum(band_candidate**2))
    prearrival_energy_fraction = float(
        np.sum(band_candidate[prearrival] ** 2) / max(total_energy, epsilon)
    )
    return {
        "valid_distance_count": int(ncf.distance_m.size),
        "waveform_correlation_no_shift": waveform_correlation,
        "normalized_l2_unit_norm": normalized_l2,
        "best_global_amplitude_scale": best_scale,
        "scaled_waveform_l2": scaled_l2,
        "phase_rms_rad": phase_rms,
        "median_envelope_peak_time_bias_s": median_peak_bias,
        "prearrival_ghost_energy_fraction": prearrival_energy_fraction,
        "spectral_shape_log_rms_after_global_scale": spectral_shape_log_rms,
        "absolute_amplitude_available": amplitude_available,
        "absolute_amplitude_log_rms": absolute_amplitude_log_rms,
        "time_shift_optimized": False,
        "reference_calibration_operator": reference.calibration_operator,
        "correlation_normalization": correlation_normalization,
    }


def _reference_velocity(
    frequency_hz: np.ndarray,
    theory_frequency_hz: np.ndarray,
    theory_velocity_mps: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(theory_frequency_hz) & np.isfinite(theory_velocity_mps)
    if np.count_nonzero(finite) < 2:
        return np.full(frequency_hz.shape, np.nan)
    theory_frequency = theory_frequency_hz[finite]
    theory_velocity = theory_velocity_mps[finite]
    reference = np.interp(frequency_hz, theory_frequency, theory_velocity)
    reference[(frequency_hz < theory_frequency[0]) | (frequency_hz > theory_frequency[-1])] = np.nan
    return reference


def _reference_modes(
    frequency_hz: np.ndarray,
    theory_frequency_hz: np.ndarray,
    theory_velocity_mps: np.ndarray,
) -> np.ndarray:
    theory_velocity = np.asarray(theory_velocity_mps, dtype=np.float64)
    if theory_velocity.ndim == 1:
        theory_velocity = theory_velocity[:, None]
    if theory_velocity.ndim != 2:
        raise ValueError("theory_velocity_mps must be frequency-by-mode")
    return np.column_stack(
        [
            _reference_velocity(
                frequency_hz,
                theory_frequency_hz,
                theory_velocity[:, mode_index],
            )
            for mode_index in range(theory_velocity.shape[1])
        ]
    )


def _contiguous_half_power_width(
    column: np.ndarray,
    peak_index: int,
    velocity_step_mps: float,
) -> float:
    threshold = 0.5 * column[peak_index]
    left = peak_index
    right = peak_index
    while left > 0 and column[left - 1] >= threshold:
        left -= 1
    while right + 1 < column.size and column[right + 1] >= threshold:
        right += 1
    return float((right - left + 1) * velocity_step_mps)


def dispersion_quality_metrics(
    image: DispersionImage,
    theory_frequency_hz: np.ndarray,
    theory_velocity_mps: np.ndarray,
    *,
    frequency_range_hz: tuple[float, float],
    near_tolerance_mps: float = 12.0,
    far_guard_mps: float = 30.0,
) -> dict[str, float | bool]:
    reference = _reference_modes(
        image.frequency_hz,
        np.asarray(theory_frequency_hz),
        np.asarray(theory_velocity_mps),
    )
    selected = (
        (image.frequency_hz >= frequency_range_hz[0])
        & (image.frequency_hz <= frequency_range_hz[1])
        & np.any(np.isfinite(reference), axis=1)
    )
    energy = image.energy[:, selected]
    reference = reference[selected]
    if energy.shape[1] == 0:
        raise ValueError("theory and image have no overlapping valid frequencies")
    energy = energy / np.maximum(energy.max(axis=0), np.finfo(float).eps)
    dv = float(np.median(np.diff(image.velocity_mps)))
    concentration = []
    width = []
    dominance = []
    dominant_near = []
    valid_mode_counts = []
    for column_index, reference_velocity in enumerate(reference):
        column = energy[:, column_index]
        reference_velocity = reference_velocity[np.isfinite(reference_velocity)]
        valid_mode_counts.append(reference_velocity.size)
        mode_distance = np.min(
            np.abs(image.velocity_mps[:, None] - reference_velocity[None, :]),
            axis=1,
        )
        near = mode_distance <= near_tolerance_mps
        far = mode_distance >= far_guard_mps
        concentration.append(float(column[near].sum() / max(column.sum(), np.finfo(float).eps)))
        near_indices = np.flatnonzero(near)
        strongest_near_index = near_indices[int(np.argmax(column[near]))]
        width.append(_contiguous_half_power_width(column, strongest_near_index, dv))
        dominance.append(float(column[near].max(initial=0.0) - column[far].max(initial=0.0)))
        peak_velocity = image.velocity_mps[int(np.argmax(column))]
        dominant_near.append(np.min(np.abs(peak_velocity - reference_velocity)) <= near_tolerance_mps)
    concentration_array = np.asarray(concentration)
    width_array = np.asarray(width)
    dominance_array = np.asarray(dominance)
    metrics: dict[str, float | bool] = {
        "valid_frequency_count": int(energy.shape[1]),
        "median_valid_theory_modes": float(np.median(valid_mode_counts)),
        "dominant_ridge_fraction": float(np.mean(dominant_near)),
        "median_ridge_concentration": float(np.median(concentration_array)),
        "median_half_power_width_mps": float(np.median(width_array)),
        "p90_half_power_width_mps": float(np.percentile(width_array, 90)),
        "p10_dominance_margin": float(np.percentile(dominance_array, 10)),
        "ambiguous_frequency_fraction": float(np.mean(dominance_array < 0.25)),
    }
    metrics["clean_ridge_pass"] = bool(
        metrics["dominant_ridge_fraction"] >= 0.90
        and metrics["median_ridge_concentration"] >= 0.25
        and metrics["median_half_power_width_mps"] <= 30.0
        and metrics["p10_dominance_margin"] >= 0.0
    )
    return metrics


def theoretical_mode_support_metrics(
    image: DispersionImage,
    theory_frequency_hz: np.ndarray,
    theory_velocity_mps: np.ndarray,
    *,
    energy_threshold: float = 0.35,
) -> dict[str, float | int]:
    """Sample normalized image energy along every theoretical mode branch."""
    if not 0 <= energy_threshold <= 1:
        raise ValueError("energy_threshold must be between zero and one")
    reference = _reference_modes(
        image.frequency_hz,
        np.asarray(theory_frequency_hz),
        np.asarray(theory_velocity_mps),
    )
    metrics: dict[str, float | int] = {}
    for mode_index in range(reference.shape[1]):
        valid = np.isfinite(reference[:, mode_index])
        prefix = f"mode_{mode_index}"
        metrics[f"{prefix}_valid_frequency_count"] = int(np.count_nonzero(valid))
        if not np.any(valid):
            metrics[f"{prefix}_median_energy"] = float("nan")
            metrics[f"{prefix}_fraction_ge_0p35"] = float("nan")
            continue
        velocity_index = np.abs(
            image.velocity_mps[:, None] - reference[None, :, mode_index]
        ).argmin(axis=0)
        sampled = image.energy[velocity_index, np.arange(image.frequency_hz.size)][valid]
        metrics[f"{prefix}_median_energy"] = float(np.median(sampled))
        metrics[f"{prefix}_fraction_ge_0p35"] = float(
            np.mean(sampled >= energy_threshold)
        )
    return metrics


def pick_error_metrics(
    picks: DispersionPicks,
    theory_frequency_hz: np.ndarray,
    theory_velocity_mps: np.ndarray,
) -> dict[str, float]:
    reference = _reference_velocity(
        picks.frequency_hz,
        np.asarray(theory_frequency_hz),
        np.asarray(theory_velocity_mps),
    )
    valid = np.isfinite(reference)
    error = picks.phase_velocity_mps[valid] - reference[valid]
    if error.size == 0:
        return {"pick_count": 0, "mae_mps": float("nan"), "rmse_mps": float("nan")}
    return {
        "pick_count": int(error.size),
        "mae_mps": float(np.mean(np.abs(error))),
        "rmse_mps": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error_mps": float(np.max(np.abs(error))),
    }
