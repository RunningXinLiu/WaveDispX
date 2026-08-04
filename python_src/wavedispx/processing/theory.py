"""Elastic Rayleigh and Love phase velocities using disba/Dunkin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from disba import PhaseDispersion

from ..config import ModelConfig


@dataclass(frozen=True)
class TheoreticalDispersion:
    frequency_hz: np.ndarray
    rayleigh_mps: np.ndarray
    love_mps: np.ndarray
    solver: str = "disba/dunkin"


def _curves(
    solver: PhaseDispersion,
    frequency: np.ndarray,
    wave: str,
    mode_count: int,
) -> np.ndarray:
    curves = np.full((frequency.size, mode_count), np.nan, dtype=np.float64)
    periods = (1.0 / frequency)[::-1]
    for mode in range(mode_count):
        try:
            curve = solver(periods, mode=mode, wave=wave)
        except Exception:
            continue
        curve_frequency = 1.0 / np.asarray(curve.period, dtype=np.float64)
        curve_velocity = 1000.0 * np.asarray(curve.velocity, dtype=np.float64)
        if curve_frequency.size == 0:
            continue
        order = np.argsort(curve_frequency)
        curve_frequency = curve_frequency[order]
        curve_velocity = curve_velocity[order]
        inside = (frequency >= curve_frequency[0]) & (frequency <= curve_frequency[-1])
        curves[inside, mode] = np.interp(
            frequency[inside], curve_frequency, curve_velocity
        )
    return curves


def theoretical_dispersion(
    model: ModelConfig,
    frequency_hz: np.ndarray,
    *,
    rayleigh_modes: int = 4,
    love_modes: int = 4,
) -> TheoreticalDispersion:
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    if frequency.size == 0 or np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency_hz must be finite, positive, and strictly increasing")
    thickness_km = np.array(
        [0.0 if np.isinf(layer.thickness_m) else layer.thickness_m / 1000.0 for layer in model.layers]
    )
    vp_kmps = np.array([layer.vp_mps / 1000.0 for layer in model.layers])
    vs_kmps = np.array([layer.vs_mps / 1000.0 for layer in model.layers])
    density_gcm3 = np.array([layer.density_kgm3 / 1000.0 for layer in model.layers])
    solver = PhaseDispersion(
        thickness_km, vp_kmps, vs_kmps, density_gcm3, algorithm="dunkin", dc=0.0001
    )
    rayleigh = _curves(solver, frequency, "rayleigh", rayleigh_modes)
    love = _curves(solver, frequency, "love", love_modes)
    return TheoreticalDispersion(frequency, rayleigh, love)

