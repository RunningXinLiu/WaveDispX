"""Quantitative dispersion-image and pick validation against theory."""

from __future__ import annotations

import numpy as np

from .processing import DispersionImage, DispersionPicks


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
