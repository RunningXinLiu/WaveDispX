#!/usr/bin/env python3
"""Generate a dense annular background plus a time-varying source sector."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _area_uniform_radius(
    rng: np.random.Generator,
    count: int,
    inner_radius_m: float,
    outer_radius_m: float,
) -> np.ndarray:
    return np.sqrt(
        rng.uniform(inner_radius_m**2, outer_radius_m**2, size=count)
    )


def generate_case(
    *,
    coordinates_path: Path,
    activity_path: Path,
    background_count: int = 144,
    sector_count: int = 72,
    window_count: int = 80,
    inner_radius_m: float = 1500.0,
    outer_radius_m: float = 1800.0,
    sector_center_deg: float = 45.0,
    sector_width_deg: float = 20.0,
    seed: int = 20260814,
    constant_total_power: bool = False,
) -> None:
    """Write deterministic source coordinates and the predeclared schedule."""
    if background_count < 16 or sector_count < 1 or window_count < 4:
        raise ValueError("source and window counts are too small")
    if not 0 < inner_radius_m < outer_radius_m:
        raise ValueError("source radii are invalid")
    if not 0 < sector_width_deg < 180:
        raise ValueError("sector_width_deg must lie in (0, 180)")

    rng = np.random.default_rng(seed)
    background_step = 2.0 * np.pi / background_count
    background_angle = background_step * (
        np.arange(background_count) + rng.uniform(0.15, 0.85, background_count)
    )
    sector_angle = np.deg2rad(
        sector_center_deg
        + rng.uniform(-0.5 * sector_width_deg, 0.5 * sector_width_deg, sector_count)
    )
    angle = np.concatenate((background_angle, sector_angle))
    radius = np.concatenate(
        (
            _area_uniform_radius(
                rng, background_count, inner_radius_m, outer_radius_m
            ),
            _area_uniform_radius(rng, sector_count, inner_radius_m, outer_radius_m),
        )
    )
    group = np.concatenate(
        (np.zeros(background_count, dtype=np.int64), np.ones(sector_count, dtype=np.int64))
    )
    coordinates_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        coordinates_path,
        np.column_stack(
            (
                radius * np.cos(angle),
                radius * np.sin(angle),
                group,
                np.mod(np.rad2deg(angle), 360.0),
                radius,
            )
        ),
        delimiter=",",
        header="x_m,y_m,source_group,azimuth_deg,radius_m",
        comments="",
        fmt="%.12g",
    )

    sector_amplitude = np.asarray((0.0, 0.75, 1.5, 2.5), dtype=np.float64)
    rows: list[tuple[int, int, float]] = []
    for window_index in range(window_count):
        amplitude = float(sector_amplitude[window_index % sector_amplitude.size])
        scale = (
            np.sqrt(
                (background_count + sector_count)
                / (background_count + sector_count * amplitude**2)
            )
            if constant_total_power
            else 1.0
        )
        rows.extend(
            (window_index, source_index, scale)
            for source_index in range(background_count)
        )
        if amplitude > 0:
            rows.extend(
                (window_index, source_index, amplitude * scale)
                for source_index in range(
                    background_count, background_count + sector_count
                )
            )
    activity_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        activity_path,
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header="window_index,source_index,amplitude_multiplier",
        comments="",
        fmt=("%d", "%d", "%.12g"),
    )


def generate_ideal_uniform_case(
    *,
    coordinates_path: Path,
    activity_path: Path,
    source_count: int = 216,
    window_count: int = 80,
    inner_radius_m: float = 1500.0,
    outer_radius_m: float = 1800.0,
    seed: int = 20260814,
) -> None:
    """Write a power-matched, dense uniform-annulus control source field."""

    if source_count < 16 or window_count < 1:
        raise ValueError("source and window counts are too small")
    rng = np.random.default_rng(seed)
    step = 2.0 * np.pi / source_count
    angle = step * (np.arange(source_count) + rng.uniform(0.15, 0.85, source_count))
    radius = _area_uniform_radius(rng, source_count, inner_radius_m, outer_radius_m)
    coordinates_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        coordinates_path,
        np.column_stack(
            (
                radius * np.cos(angle),
                radius * np.sin(angle),
                np.zeros(source_count, dtype=np.int64),
                np.mod(np.rad2deg(angle), 360.0),
                radius,
            )
        ),
        delimiter=",",
        header="x_m,y_m,source_group,azimuth_deg,radius_m",
        comments="",
        fmt="%.12g",
    )
    window_index = np.repeat(np.arange(window_count), source_count)
    source_index = np.tile(np.arange(source_count), window_count)
    activity_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        activity_path,
        np.column_stack((window_index, source_index, np.ones(window_index.size))),
        delimiter=",",
        header="window_index,source_index,amplitude_multiplier",
        comments="",
        fmt=("%d", "%d", "%.12g"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--background-count", type=int, default=144)
    parser.add_argument("--sector-count", type=int, default=72)
    parser.add_argument("--window-count", type=int, default=80)
    parser.add_argument("--inner-radius-m", type=float, default=1500.0)
    parser.add_argument("--outer-radius-m", type=float, default=1800.0)
    parser.add_argument("--sector-center-deg", type=float, default=45.0)
    parser.add_argument("--sector-width-deg", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--constant-total-power", action="store_true")
    parser.add_argument("--ideal-coordinates", type=Path)
    parser.add_argument("--ideal-activity", type=Path)
    args = parser.parse_args()
    generate_case(
        coordinates_path=args.coordinates,
        activity_path=args.activity,
        background_count=args.background_count,
        sector_count=args.sector_count,
        window_count=args.window_count,
        inner_radius_m=args.inner_radius_m,
        outer_radius_m=args.outer_radius_m,
        sector_center_deg=args.sector_center_deg,
        sector_width_deg=args.sector_width_deg,
        seed=args.seed,
        constant_total_power=args.constant_total_power,
    )
    if (args.ideal_coordinates is None) != (args.ideal_activity is None):
        parser.error("ideal coordinates and activity must be supplied together")
    if args.ideal_coordinates is not None:
        generate_ideal_uniform_case(
            coordinates_path=args.ideal_coordinates,
            activity_path=args.ideal_activity,
            source_count=args.background_count + args.sector_count,
            window_count=args.window_count,
            inner_radius_m=args.inner_radius_m,
            outer_radius_m=args.outer_radius_m,
            seed=args.seed + 1,
        )
    print(
        f"wrote {args.coordinates} and {args.activity} "
        f"({args.background_count}+{args.sector_count} sources, "
        f"{args.window_count} windows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
