"""Phase-shift, frequency-Bessel, and frequency-Hankel imaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import detrend
from scipy.signal.windows import tukey
from scipy.special import hankel1, jv, yv

from ..core import radial_transform_lookup


Method = Literal["phase_shift", "bessel", "hankel"]


@dataclass(frozen=True)
class DispersionImage:
    energy: np.ndarray
    raw_amplitude: np.ndarray
    frequency_hz: np.ndarray
    velocity_mps: np.ndarray
    method: Method


def combine_dispersion_images(
    images: list[DispersionImage] | tuple[DispersionImage, ...],
) -> DispersionImage:
    """Combine independently normalized component images by RMS energy.

    Each input must share the same frequency/velocity grid and imaging method.
    RMS combination preserves ridges that couple preferentially to one component
    without allowing a single high-amplitude component to dominate all columns.
    """
    if not images:
        raise ValueError("at least one dispersion image is required")
    reference = images[0]
    for image in images[1:]:
        if image.method != reference.method:
            raise ValueError("all dispersion images must use the same method")
        if image.energy.shape != reference.energy.shape:
            raise ValueError("all dispersion images must share one grid")
        if not np.array_equal(image.frequency_hz, reference.frequency_hz) or not np.array_equal(
            image.velocity_mps, reference.velocity_mps
        ):
            raise ValueError("all dispersion images must share one grid")
    energy = np.sqrt(np.mean([image.energy**2 for image in images], axis=0))
    column_max = energy.max(axis=0, initial=0.0)
    column_max[column_max == 0] = 1.0
    energy /= column_max
    raw = np.sqrt(np.mean([image.raw_amplitude**2 for image in images], axis=0))
    return DispersionImage(
        energy,
        raw,
        reference.frequency_hz,
        reference.velocity_mps,
        reference.method,
    )


def _power_of_two_at_least(value: int) -> int:
    return 1 << max(0, (int(value) - 1).bit_length())


def surface_wave_dispersion(
    data: np.ndarray,
    distance_m: np.ndarray,
    dt_s: float,
    velocity_mps: np.ndarray,
    *,
    frequency_min_hz: float = 0.0,
    frequency_max_hz: float,
    frequency_step_hz: float | None = None,
    method: Method = "phase_shift",
    component: Literal["vertical", "radial", "transverse"] = "vertical",
    zero_padding: float = 4.0,
    time_taper_fraction: float = 0.1,
    distance_taper_fraction: float = 0.0,
    spectrum_normalization: Literal["none", "phase"] = "none",
    radial_weighting: Literal["area", "uniform"] = "area",
    frequency_block_size: int = 16,
    radial_kernel_lookup_step: float = 0.0,
    workers: int = 0,
) -> DispersionImage:
    gather = np.asarray(data, dtype=np.float64)
    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    velocity = np.asarray(velocity_mps, dtype=np.float64).reshape(-1)
    if gather.ndim != 2 or gather.shape[1] != distance.size:
        raise ValueError("data must be time-by-distance and match distance_m")
    if velocity.size == 0 or np.any(velocity <= 0):
        raise ValueError("velocity_mps must be positive")
    keep = distance > 0
    gather = gather[:, keep]
    distance = distance[keep]
    order = np.argsort(distance)
    gather = gather[:, order]
    distance = distance[order]
    unique_distance, inverse_distance, distance_count = np.unique(
        distance, return_inverse=True, return_counts=True
    )
    if unique_distance.size < 2:
        raise ValueError("at least two unique positive distances are required")
    if unique_distance.size != distance.size:
        radial_gather = np.zeros(
            (gather.shape[0], unique_distance.size), dtype=np.float64
        )
        for original_index, radial_index in enumerate(inverse_distance):
            radial_gather[:, radial_index] += gather[:, original_index]
        gather = radial_gather / distance_count
        distance = unique_distance
    if method not in {"phase_shift", "bessel", "hankel"}:
        raise ValueError("unknown dispersion method")
    if not 0 <= time_taper_fraction <= 1 or not 0 <= distance_taper_fraction <= 1:
        raise ValueError("taper fractions must be between zero and one")
    if zero_padding < 1:
        raise ValueError("zero_padding must be >= 1")
    if frequency_step_hz is not None and (
        not np.isfinite(frequency_step_hz) or frequency_step_hz <= 0.0
    ):
        raise ValueError("frequency_step_hz must be positive and finite")
    if frequency_block_size < 1:
        raise ValueError("frequency_block_size must be >= 1")
    if radial_kernel_lookup_step < 0:
        raise ValueError("radial_kernel_lookup_step must be >= 0")

    gather = detrend(gather, axis=0, type="linear")
    gather *= tukey(gather.shape[0], time_taper_fraction).reshape(-1, 1)
    distance_taper = tukey(distance.size, distance_taper_fraction)
    if distance_taper_fraction == 0:
        distance_taper[:] = 1.0
    nfft = _power_of_two_at_least(int(np.ceil(gather.shape[0] * zero_padding)))
    if frequency_step_hz is not None:
        # An explicit display grid may require a denser FFT grid before complex
        # interpolation.  This only samples the finite record more densely; its
        # physical resolution remains 1 / (N * dt).
        requested_nfft = int(np.ceil(1.0 / (dt_s * frequency_step_hz)))
        nfft = max(nfft, _power_of_two_at_least(requested_nfft))
    full_spectrum = np.fft.fft(gather, nfft, axis=0)
    full_frequency = np.arange(nfft // 2 + 1, dtype=np.float64) / (nfft * dt_s)
    upper_frequency = min(frequency_max_hz, 0.5 / dt_s)
    if frequency_step_hz is None:
        mask = (
            (full_frequency > 0)
            & (full_frequency >= frequency_min_hz)
            & (full_frequency <= upper_frequency)
        )
        frequency = full_frequency[mask]
        spectrum = full_spectrum[np.flatnonzero(mask)]
    else:
        start_frequency = max(float(frequency_min_hz), float(frequency_step_hz))
        frequency_count = int(
            np.floor((upper_frequency - start_frequency) / frequency_step_hz + 1.0e-10)
        ) + 1
        if frequency_count <= 0:
            raise ValueError("the requested frequency grid is empty")
        frequency = start_frequency + frequency_step_hz * np.arange(
            frequency_count, dtype=np.float64
        )
        positive_spectrum = full_spectrum[: nfft // 2 + 1]
        spectrum = np.empty((frequency.size, gather.shape[1]), dtype=np.complex128)
        for receiver_index in range(gather.shape[1]):
            trace_spectrum = positive_spectrum[:, receiver_index]
            spectrum[:, receiver_index] = np.interp(
                frequency, full_frequency, trace_spectrum.real
            ) + 1j * np.interp(frequency, full_frequency, trace_spectrum.imag)
    if frequency.size == 0:
        raise ValueError("the requested frequency grid is empty")
    if spectrum_normalization == "phase":
        spectrum /= np.maximum(np.abs(spectrum), np.finfo(float).eps)
    elif spectrum_normalization != "none":
        raise ValueError("spectrum_normalization must be none or phase")

    raw = np.zeros((velocity.size, frequency.size), dtype=np.float64)
    inverse_velocity = 1.0 / velocity
    angular_frequency = 2 * np.pi * frequency
    if method == "phase_shift":
        phase_only = spectrum / np.maximum(np.abs(spectrum), np.finfo(float).eps)
        phase_only *= distance_taper
        for block_start in range(0, frequency.size, frequency_block_size):
            block_stop = min(frequency.size, block_start + frequency_block_size)
            omega = angular_frequency[block_start:block_stop]
            steering = np.exp(
                1j
                * omega[:, None, None]
                * inverse_velocity[None, :, None]
                * distance[None, None, :]
            )
            raw[:, block_start:block_stop] = np.abs(
                np.einsum(
                    "fvd,fd->vf",
                    steering,
                    phase_only[block_start:block_stop],
                    optimize=True,
                )
            )
    else:
        midpoint = 0.5 * (distance[:-1] + distance[1:])
        left_edge = max(0.0, distance[0] - 0.5 * (distance[1] - distance[0]))
        right_edge = distance[-1] + 0.5 * (distance[-1] - distance[-2])
        edges = np.concatenate(([left_edge], midpoint, [right_edge]))
        radial_weight = (
            0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)
            if radial_weighting == "area"
            else np.ones(distance.size)
        )
        order_bessel = 1 if component == "radial" else 0
        weighted = spectrum * (radial_weight * distance_taper)[None, :]
        lookup_argument = None
        lookup_kernel = None
        if radial_kernel_lookup_step > 0:
            minimum_argument = float(
                angular_frequency.min() * inverse_velocity.min() * distance.min()
            )
            maximum_argument = float(
                angular_frequency.max() * inverse_velocity.max() * distance.max()
            )
            lookup_count = int(
                np.ceil(
                    (maximum_argument - minimum_argument)
                    / radial_kernel_lookup_step
                )
            )
            lookup_argument = np.linspace(
                minimum_argument, maximum_argument, lookup_count + 1
            )
            lookup_kernel = jv(order_bessel, lookup_argument)
            if method == "hankel":
                lookup_kernel = lookup_kernel + 1j * yv(order_bessel, lookup_argument)
            raw = radial_transform_lookup(
                weighted,
                frequency,
                distance,
                velocity,
                lookup_kernel,
                lookup_minimum=float(lookup_argument[0]),
                lookup_step=float(lookup_argument[1] - lookup_argument[0]),
                workers=workers,
            )
        else:
            for block_start in range(0, frequency.size, frequency_block_size):
                block_stop = min(frequency.size, block_start + frequency_block_size)
                omega = angular_frequency[block_start:block_stop]
                kr = (
                    omega[:, None, None]
                    * inverse_velocity[None, :, None]
                    * distance[None, None, :]
                )
                kernel = (
                    jv(order_bessel, kr)
                    if method == "bessel"
                    else hankel1(order_bessel, kr)
                )
                raw[:, block_start:block_stop] = np.abs(
                    np.einsum(
                        "fvd,fd->vf",
                        kernel,
                        weighted[block_start:block_stop],
                        optimize=True,
                    )
                )
    column_max = raw.max(axis=0, initial=0.0)
    column_max[column_max == 0] = 1.0
    energy = raw / column_max
    return DispersionImage(energy, raw, frequency, velocity, method)
