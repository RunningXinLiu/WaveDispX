"""Elastic Rayleigh and Love phase-velocity theory backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from disba import GroupDispersion, PhaseDispersion

from ..config import ModelConfig


@dataclass(frozen=True)
class TheoreticalDispersion:
    frequency_hz: np.ndarray
    rayleigh_mps: np.ndarray
    love_mps: np.ndarray
    solver: str = "disba/dunkin"
    rayleigh_group_mps: np.ndarray | None = None
    love_group_mps: np.ndarray | None = None


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
    backend: Literal["disba", "qedisp"] = "disba",
    qedisp_forward: str | Path | None = None,
    compute_group: bool = False,
    qedisp_group_library: str | Path | None = None,
) -> TheoreticalDispersion:
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    if frequency.size == 0 or np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency_hz must be finite, positive, and strictly increasing")
    if backend == "qedisp":
        from .qedisp import run_qedisp_forward

        result = run_qedisp_forward(
            model,
            frequency,
            rayleigh_modes=rayleigh_modes,
            love_modes=love_modes,
            executable=qedisp_forward,
            compute_group=compute_group,
            group_library=qedisp_group_library,
        )
        return TheoreticalDispersion(
            frequency_hz=result.frequency_hz,
            rayleigh_mps=result.rayleigh_mps,
            love_mps=result.love_mps,
            solver="QEDispInv/quadratic-extrema",
            rayleigh_group_mps=result.rayleigh_group_mps,
            love_group_mps=result.love_group_mps,
        )
    if backend != "disba":
        raise ValueError("backend must be 'disba' or 'qedisp'")

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
    rayleigh_group = None
    love_group = None
    if compute_group:
        group_solver = GroupDispersion(
            thickness_km,
            vp_kmps,
            vs_kmps,
            density_gcm3,
            algorithm="dunkin",
            dc=0.0001,
            dt=0.025,
        )
        rayleigh_group = _curves(
            group_solver, frequency, "rayleigh", rayleigh_modes
        )
        love_group = _curves(group_solver, frequency, "love", love_modes)
    return TheoreticalDispersion(
        frequency,
        rayleigh,
        love,
        rayleigh_group_mps=rayleigh_group,
        love_group_mps=love_group,
    )
