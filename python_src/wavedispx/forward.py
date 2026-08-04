"""WaveDispX layered-medium forward modelling built on the C++ f-k core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import jv

from .config import ModelConfig
from .core import full_fk_response, vertical_fk_response


@dataclass(frozen=True)
class VerticalFkGreen:
    frequency_hz: np.ndarray
    wavenumber_radpm: np.ndarray
    radial_displacement: np.ndarray
    vertical_displacement: np.ndarray


@dataclass(frozen=True)
class VerticalReceiverGreen:
    time_s: np.ndarray
    distance_m: np.ndarray
    radial: np.ndarray | None
    vertical: np.ndarray


@dataclass(frozen=True)
class FullFkGreen:
    frequency_hz: np.ndarray
    wavenumber_radpm: np.ndarray
    horizontal_radial: np.ndarray
    horizontal_tangential: np.ndarray
    horizontal_vertical: np.ndarray
    vertical_radial: np.ndarray
    vertical_vertical: np.ndarray


@dataclass(frozen=True)
class FullReceiverGreen:
    time_s: np.ndarray
    distance_m: np.ndarray
    horizontal_radial: np.ndarray
    horizontal_tangential: np.ndarray
    horizontal_vertical: np.ndarray
    vertical_radial: np.ndarray
    vertical_vertical: np.ndarray


def _model_arrays(model: ModelConfig) -> tuple[np.ndarray, ...]:
    return (
        np.array([layer.vp_mps for layer in model.layers]),
        np.array([layer.vs_mps for layer in model.layers]),
        np.array([layer.damp_p for layer in model.layers]),
        np.array([layer.damp_s for layer in model.layers]),
        np.array([layer.density_kgm3 for layer in model.layers]),
        np.array([layer.thickness_m for layer in model.layers]),
    )


def compute_vertical_fk_green(
    model: ModelConfig,
    *,
    sample_count: int,
    dt_s: float,
    maximum_distance_m: float,
    delay_s: float,
    maximum_frequency_hz: float,
    workers: int = 0,
) -> VerticalFkGreen:
    result = vertical_fk_response(
        *_model_arrays(model),
        sample_count=sample_count,
        dt_s=dt_s,
        maximum_distance_m=maximum_distance_m,
        delay_s=delay_s,
        maximum_frequency_hz=maximum_frequency_hz,
        workers=workers,
    )
    return VerticalFkGreen(
        result["frequency_hz"],
        result["wavenumber_radpm"],
        result["vr"],
        result["vv"],
    )


def compute_full_fk_green(
    model: ModelConfig,
    *,
    sample_count: int,
    dt_s: float,
    maximum_distance_m: float,
    delay_s: float,
    maximum_frequency_hz: float,
    workers: int = 0,
) -> FullFkGreen:
    result = full_fk_response(
        *_model_arrays(model),
        sample_count=sample_count,
        dt_s=dt_s,
        maximum_distance_m=maximum_distance_m,
        delay_s=delay_s,
        maximum_frequency_hz=maximum_frequency_hz,
        workers=workers,
    )
    return FullFkGreen(
        result["frequency_hz"],
        result["wavenumber_radpm"],
        result["hr"],
        result["ht"],
        result["hv"],
        result["vr"],
        result["vv"],
    )


def _spectra_to_time(
    spectra: list[np.ndarray], sample_count: int, dt_s: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    positive_count = sample_count // 2 + 1
    frequency_taper = 0.5 * (
        np.cos(np.pi * np.arange(positive_count) / (positive_count - 1)) + 1.0
    )
    delta_omega = 2.0 * np.pi / (sample_count * dt_s)
    time = np.arange(sample_count, dtype=np.float64) * dt_s
    undo_exponential_window = np.exp(delta_omega * time)
    time_fields = []
    for spectrum in spectra:
        spectrum *= frequency_taper
        field = np.fft.irfft(spectrum, n=sample_count, axis=1)
        field *= undo_exponential_window
        time_fields.append(field.T)
    return time, time_fields


def synthesize_vertical_receivers(
    fk_green: VerticalFkGreen,
    distance_m: np.ndarray,
    *,
    sample_count: int,
    dt_s: float,
    distance_block_size: int = 128,
    include_radial: bool = True,
) -> VerticalReceiverGreen:
    """Inverse-Hankel synthesize vertical-load Green functions in blocks."""
    if sample_count % 2:
        sample_count -= 1
    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    if distance.size == 0 or np.any(distance <= 0) or np.any(~np.isfinite(distance)):
        raise ValueError("distance_m must contain finite positive offsets")
    wavenumber = np.asarray(fk_green.wavenumber_radpm, dtype=np.float64)
    if wavenumber.size < 2:
        raise ValueError("f-k Green function needs at least two wavenumbers")
    dky = wavenumber[1] - wavenumber[0]
    positive_count = sample_count // 2 + 1
    frequency_count = fk_green.frequency_hz.size
    if frequency_count > positive_count:
        raise ValueError("f-k frequency grid exceeds the requested temporal grid")
    radial_spectrum = (
        np.zeros((distance.size, positive_count), dtype=np.complex128)
        if include_radial
        else None
    )
    vertical_spectrum = np.zeros(
        (distance.size, positive_count), dtype=np.complex128
    )
    for first in range(0, distance.size, distance_block_size):
        block = distance[first : first + distance_block_size]
        radius = np.minimum(0.02 * block, 1.0)
        argument = np.outer(wavenumber, block)
        bell = np.exp(-np.outer(wavenumber**2, radius**2) / 4.0)
        vertical_kernel = wavenumber[:, None] * bell * jv(0, argument)
        if radial_spectrum is not None:
            radial_kernel = -wavenumber[:, None] * bell * jv(1, argument)
            radial_spectrum[first : first + block.size, :frequency_count] = (
                dky * (fk_green.radial_displacement @ radial_kernel).T
            )
        vertical_spectrum[first : first + block.size, :frequency_count] = (
            dky * (fk_green.vertical_displacement @ vertical_kernel).T
        )
    frequency_taper = 0.5 * (
        np.cos(np.pi * np.arange(positive_count) / (positive_count - 1)) + 1.0
    )
    if radial_spectrum is not None:
        radial_spectrum *= frequency_taper
    vertical_spectrum *= frequency_taper
    radial_time = (
        np.fft.irfft(radial_spectrum, n=sample_count, axis=1)
        if radial_spectrum is not None
        else None
    )
    vertical_time = np.fft.irfft(vertical_spectrum, n=sample_count, axis=1)
    delta_omega = 2.0 * np.pi / (sample_count * dt_s)
    time = np.arange(sample_count, dtype=np.float64) * dt_s
    undo_exponential_window = np.exp(delta_omega * time)
    if radial_time is not None:
        radial_time *= undo_exponential_window
    vertical_time *= undo_exponential_window
    return VerticalReceiverGreen(
        time,
        distance,
        radial_time.T if radial_time is not None else None,
        vertical_time.T,
    )


def synthesize_full_receivers(
    fk_green: FullFkGreen,
    distance_m: np.ndarray,
    *,
    sample_count: int,
    dt_s: float,
    distance_block_size: int = 128,
) -> FullReceiverGreen:
    """Inverse-Hankel synthesize all five discrete-wavenumber Green components."""
    if sample_count % 2:
        sample_count -= 1
    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    if distance.size == 0 or np.any(distance <= 0) or np.any(~np.isfinite(distance)):
        raise ValueError("distance_m must contain finite positive offsets")
    wavenumber = np.asarray(fk_green.wavenumber_radpm, dtype=np.float64)
    dky = wavenumber[1] - wavenumber[0]
    positive_count = sample_count // 2 + 1
    frequency_count = fk_green.frequency_hz.size
    spectra = [
        np.zeros((distance.size, positive_count), dtype=np.complex128)
        for _ in range(5)
    ]
    for first in range(0, distance.size, distance_block_size):
        block = distance[first : first + distance_block_size]
        radius = np.minimum(0.02 * block, 1.0)
        argument = np.outer(wavenumber, block)
        bell = np.exp(-np.outer(wavenumber**2, radius**2) / 4.0)
        bessel0 = jv(0, argument)
        bessel1 = jv(1, argument)
        bessel1_over_distance = bessel1 / block
        primary_kernel = bell * (
            wavenumber[:, None] * bessel0 - bessel1_over_distance
        )
        coupled_kernel = bell * bessel1_over_distance
        horizontal_vertical_kernel = bell * wavenumber[:, None] * bessel1
        vertical_radial_kernel = -horizontal_vertical_kernel
        vertical_vertical_kernel = bell * wavenumber[:, None] * bessel0
        block_slice = slice(first, first + block.size)
        frequency_slice = slice(0, frequency_count)
        spectra[0][block_slice, frequency_slice] = dky * (
            fk_green.horizontal_radial @ primary_kernel
            + fk_green.horizontal_tangential @ coupled_kernel
        ).T
        spectra[1][block_slice, frequency_slice] = dky * (
            fk_green.horizontal_tangential @ primary_kernel
            + fk_green.horizontal_radial @ coupled_kernel
        ).T
        spectra[2][block_slice, frequency_slice] = dky * (
            fk_green.horizontal_vertical @ horizontal_vertical_kernel
        ).T
        spectra[3][block_slice, frequency_slice] = dky * (
            fk_green.vertical_radial @ vertical_radial_kernel
        ).T
        spectra[4][block_slice, frequency_slice] = dky * (
            fk_green.vertical_vertical @ vertical_vertical_kernel
        ).T
    time, fields = _spectra_to_time(spectra, sample_count, dt_s)
    return FullReceiverGreen(time, distance, *fields)
