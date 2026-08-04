"""Smooth dynamic-programming dispersion ridge picker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DispersionPicks:
    frequency_hz: np.ndarray
    phase_velocity_mps: np.ndarray
    normalized_energy: np.ndarray
    half_power_uncertainty_mps: np.ndarray

    def as_array(self) -> np.ndarray:
        return np.column_stack(
            (
                self.frequency_hz,
                self.phase_velocity_mps,
                self.normalized_energy,
                self.half_power_uncertainty_mps,
            )
        )


def pick_dispersion_ridge(
    energy: np.ndarray,
    frequency_hz: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    frequency_range_hz: tuple[float, float] | None = None,
    velocity_range_mps: tuple[float, float] | None = None,
    smoothness: float = 1.5,
    maximum_jump_mps: float = np.inf,
    minimum_energy: float = 0.1,
) -> DispersionPicks:
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    velocity = np.asarray(velocity_mps, dtype=np.float64).reshape(-1)
    image = np.asarray(energy, dtype=np.float64)
    if image.shape != (velocity.size, frequency.size):
        raise ValueError("energy must be velocity-by-frequency")
    if frequency_range_hz is None:
        frequency_range_hz = (float(frequency.min()), float(frequency.max()))
    if velocity_range_mps is None:
        velocity_range_mps = (float(velocity.min()), float(velocity.max()))
    frequency_keep = (frequency >= frequency_range_hz[0]) & (frequency <= frequency_range_hz[1])
    velocity_keep = (velocity >= velocity_range_mps[0]) & (velocity <= velocity_range_mps[1])
    selected_frequency = frequency[frequency_keep]
    selected_velocity = velocity[velocity_keep]
    selected_energy = image[np.ix_(velocity_keep, frequency_keep)]
    if selected_frequency.size == 0 or selected_velocity.size == 0:
        raise ValueError("pick search ranges select no samples")
    dv = np.median(np.diff(selected_velocity)) if selected_velocity.size > 1 else 1.0
    difference = selected_velocity[:, None] - selected_velocity[None, :]
    transition = smoothness * (difference / dv) ** 2
    transition[np.abs(difference) > maximum_jump_mps] = np.inf
    score = np.full(selected_energy.shape, -np.inf)
    back_pointer = np.zeros(selected_energy.shape, dtype=np.int32)
    score[:, 0] = np.log(np.maximum(selected_energy[:, 0], np.finfo(float).eps))
    for frequency_index in range(1, selected_frequency.size):
        candidate = score[:, frequency_index - 1][None, :] - transition
        previous = np.argmax(candidate, axis=1)
        score[:, frequency_index] = (
            np.log(np.maximum(selected_energy[:, frequency_index], np.finfo(float).eps))
            + candidate[np.arange(selected_velocity.size), previous]
        )
        back_pointer[:, frequency_index] = previous
    ridge_index = np.zeros(selected_frequency.size, dtype=np.int32)
    ridge_index[-1] = int(np.argmax(score[:, -1]))
    for frequency_index in range(selected_frequency.size - 1, 0, -1):
        ridge_index[frequency_index - 1] = back_pointer[
            ridge_index[frequency_index], frequency_index
        ]
    column_index = np.arange(selected_frequency.size)
    ridge_energy = selected_energy[ridge_index, column_index]
    ridge_velocity = selected_velocity[ridge_index]
    uncertainty = np.zeros(selected_frequency.size)
    for frequency_index, center in enumerate(ridge_index):
        above = selected_energy[:, frequency_index] >= 0.5 * ridge_energy[frequency_index]
        left = int(center)
        right = int(center)
        while left > 0 and above[left - 1]:
            left -= 1
        while right + 1 < selected_velocity.size and above[right + 1]:
            right += 1
        uncertainty[frequency_index] = 0.5 * (selected_velocity[right] - selected_velocity[left])
    valid = ridge_energy >= minimum_energy
    return DispersionPicks(
        selected_frequency[valid], ridge_velocity[valid], ridge_energy[valid], uncertainty[valid]
    )

