#!/usr/bin/env python3
"""Compute elastic Rayleigh/Love phase velocities for a MATLAB model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from disba import PhaseDispersion


def _vector(data: dict, name: str) -> np.ndarray:
    return np.asarray(data[name], dtype=float).reshape(-1)


def _curves(
    solver: PhaseDispersion,
    frequency: np.ndarray,
    wave: str,
    mode_count: int,
) -> np.ndarray:
    curves = np.full((frequency.size, mode_count), np.nan, dtype=float)
    # disba requires periods in ascending order.
    periods = (1.0 / frequency)[::-1]
    for mode in range(mode_count):
        try:
            curve = solver(periods, mode=mode, wave=wave)
        except Exception as exc:  # A missing high mode is a physical cutoff.
            print(f"{wave} mode {mode}: {exc}")
            continue
        curve_frequency = 1.0 / np.asarray(curve.period, dtype=float)
        curve_velocity = 1000.0 * np.asarray(curve.velocity, dtype=float)
        if curve_frequency.size == 0:
            print(f"{wave} mode {mode}: no root in requested frequency range")
            continue
        order = np.argsort(curve_frequency)
        curve_frequency = curve_frequency[order]
        curve_velocity = curve_velocity[order]
        inside = (frequency >= curve_frequency[0]) & (
            frequency <= curve_frequency[-1]
        )
        curves[inside, mode] = np.interp(
            frequency[inside], curve_frequency, curve_velocity
        )
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mat", type=Path)
    parser.add_argument("output_mat", type=Path)
    parser.add_argument("--rayleigh-modes", type=int, default=4)
    parser.add_argument("--love-modes", type=int, default=4)
    args = parser.parse_args()

    data = loadmat(args.input_mat)
    vp = _vector(data, "vp") / 1000.0
    vs = _vector(data, "vs") / 1000.0
    thickness = _vector(data, "thickness") / 1000.0
    density = _vector(data, "density")
    frequency = _vector(data, "frequency")

    if np.nanmedian(density) > 20:
        density = density / 1000.0
    thickness[~np.isfinite(thickness)] = 0.0
    if thickness[-1] != 0:
        raise ValueError("The last layer must be a half-space (Inf in MATLAB).")

    solver = PhaseDispersion(
        thickness, vp, vs, density, algorithm="dunkin", dc=0.0001
    )
    rayleigh = _curves(solver, frequency, "rayleigh", args.rayleigh_modes)
    love = _curves(solver, frequency, "love", args.love_modes)
    savemat(
        args.output_mat,
        {
            "frequency": frequency.reshape(1, -1),
            "rayleigh": rayleigh,
            "love": love,
        },
    )


if __name__ == "__main__":
    main()
