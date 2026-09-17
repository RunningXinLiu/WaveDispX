#!/usr/bin/env python3
"""Diagnose pair-FTAN group-energy quality without automatic ridge picking.

The script scans receiver separation, NCF branch, and Gaussian-filter alpha.
For every candidate it measures whether the independent theoretical group-
velocity curve passes through the dominant FTAN energy.  The diagnostic is
intended to qualify an FTAN setup before window-selection methods are compared.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from wavedispx.config import load_config
from wavedispx.processing import pair_ftan, phase_to_group_velocity, theoretical_dispersion


def _component_index(group: h5py.Group, component: str) -> int:
    names = tuple(
        value.decode() if isinstance(value, bytes) else str(value)
        for value in group.attrs["component_names"]
    )
    return names.index(component)


def _nearest_distance_index(distance_m: np.ndarray, target_m: float) -> int:
    index = int(np.argmin(np.abs(distance_m - target_m)))
    if not np.isclose(distance_m[index], target_m, atol=0.25):
        raise ValueError(f"requested distance {target_m:g} m is absent")
    return index


def _load_traces(
    biased_results: Path,
    ideal_results: Path,
    distance_m: float,
    branch: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    with h5py.File(biased_results) as handle:
        distance = handle["ncf/distance_m"][:]
        index = _nearest_distance_index(distance, distance_m)
        lag = handle["ncf/lag_time_s"][:]
        biased = handle[
            f"ncf/strategies/baseline_linear/components/zz/{branch}"
        ][:, index]
        reference = handle["egf_reference"]
        reference_index = _nearest_distance_index(
            reference["distance_m"][:], distance_m
        )
        active = reference["components"][
            :, reference_index, _component_index(reference, "zz")
        ]
        active_time = reference["time_s"][:]
    with h5py.File(ideal_results) as handle:
        distance = handle["ncf/distance_m"][:]
        index = _nearest_distance_index(distance, distance_m)
        ideal = handle[
            f"ncf/strategies/baseline_linear/components/zz/{branch}"
        ][:, index]
        ideal_lag = handle["ncf/lag_time_s"][:]
    if lag.shape != ideal_lag.shape or not np.allclose(lag, ideal_lag):
        raise ValueError("biased and ideal lag axes differ")
    return active, active_time, ideal, lag, float(distance_m)


def _theory(config, frequency_min_hz: float, frequency_max_hz: float):
    frequency = np.linspace(frequency_min_hz, frequency_max_hz, 1001)
    result = theoretical_dispersion(
        config.model,
        frequency,
        rayleigh_modes=1,
        love_modes=0,
        backend="qedisp",
    )
    finite = np.isfinite(result.rayleigh_mps[:, 0])
    period = 1.0 / frequency[finite]
    phase = result.rayleigh_mps[finite, 0]
    order = np.argsort(period)
    period = period[order]
    phase = phase[order]
    return period, phase_to_group_velocity(period, phase)


def _path_metrics(
    amplitude: np.ndarray,
    velocity_mps: np.ndarray,
    theory_velocity_mps: np.ndarray,
    *,
    half_width_mps: float,
    trim_fraction: float = 0.08,
) -> dict[str, float]:
    image = np.asarray(amplitude, dtype=np.float64)
    mask = np.abs(velocity_mps[:, None] - theory_velocity_mps[None, :]) <= half_width_mps
    energy = np.square(image)
    inside_peak = np.max(np.where(mask, image, 0.0), axis=0)
    outside_peak = np.max(np.where(~mask, image, 0.0), axis=0)
    denominator = np.sum(energy, axis=0)
    tube_fraction = np.divide(
        np.sum(np.where(mask, energy, 0.0), axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.finfo(float).eps,
    )
    count = image.shape[1]
    edge = max(1, int(np.floor(trim_fraction * count)))
    keep = slice(edge, count - edge)
    eps = np.finfo(float).eps
    contrast_db = 20.0 * np.log10((inside_peak[keep] + eps) / (outside_peak[keep] + eps))
    return {
        "median_theory_tube_energy_fraction": float(np.median(tube_fraction[keep])),
        "median_theory_path_peak": float(np.median(inside_peak[keep])),
        "median_theory_path_contrast_db": float(np.median(contrast_db)),
        "period_fraction_theory_dominant": float(np.mean(contrast_db > 0.0)),
    }


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated number list")
    return values


def _parse_str_list(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    allowed = {"positive", "negative", "symmetric"}
    if not values or any(value not in allowed for value in values):
        raise argparse.ArgumentTypeError("branches must be positive,negative,symmetric")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biased-run-directory", type=Path, required=True)
    parser.add_argument("--ideal-run-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--distances-m", type=_parse_float_list, default=(200, 400, 600, 800, 1000))
    parser.add_argument("--branches", type=_parse_str_list, default=("positive", "negative", "symmetric"))
    parser.add_argument("--gaussian-alphas", type=_parse_float_list, default=(2.5, 5, 8, 12, 20))
    parser.add_argument("--period-count", type=int, default=96)
    parser.add_argument("--velocity-step-mps", type=float, default=2.0)
    parser.add_argument("--theory-tube-half-width-mps", type=float, default=15.0)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    frequency_min = config.processing.frequency_min_hz
    frequency_max = config.processing.frequency_max_hz
    theory_period, theory_group = _theory(config, frequency_min, frequency_max)
    common = {
        "frequency_range_hz": (frequency_min, frequency_max),
        "group_velocity_range_mps": (
            config.processing.velocity_min_mps,
            config.processing.velocity_max_mps,
        ),
        "phase_velocity_range_mps": (
            config.processing.velocity_min_mps,
            config.processing.velocity_max_mps,
        ),
        "period_count": args.period_count,
        "velocity_step_mps": args.velocity_step_mps,
        "phase_method": "frequency_domain",
        "minimum_snr": 0.0,
    }
    rows: list[dict[str, object]] = []
    for distance_m in args.distances_m:
        for branch in args.branches:
            active, active_time, ideal, lag, selected_distance = _load_traces(
                args.biased_run_directory / "results.h5",
                args.ideal_run_directory / "results.h5",
                distance_m,
                branch,
            )
            for alpha in args.gaussian_alphas:
                active_result = pair_ftan(
                    active,
                    dt_s=float(np.median(np.diff(active_time))),
                    distance_m=selected_distance,
                    input_kind="egf",
                    gaussian_alpha=alpha,
                    **common,
                )
                ideal_result = pair_ftan(
                    ideal,
                    dt_s=float(np.median(np.diff(lag))),
                    distance_m=selected_distance,
                    input_kind="correlation",
                    gaussian_alpha=alpha,
                    **common,
                )
                period = active_result.group_image.period_s
                theory_on_period = np.interp(period, theory_period, theory_group)
                active_metrics = _path_metrics(
                    active_result.group_image.amplitude,
                    active_result.group_image.velocity_mps,
                    theory_on_period,
                    half_width_mps=args.theory_tube_half_width_mps,
                )
                ideal_metrics = _path_metrics(
                    ideal_result.group_image.amplitude,
                    ideal_result.group_image.velocity_mps,
                    theory_on_period,
                    half_width_mps=args.theory_tube_half_width_mps,
                )
                row: dict[str, object] = {
                    "distance_m": selected_distance,
                    "branch": branch,
                    "gaussian_alpha": alpha,
                }
                row.update({f"active_{key}": value for key, value in active_metrics.items()})
                row.update({f"ideal_{key}": value for key, value in ideal_metrics.items()})
                row["qualification_score"] = min(
                    active_metrics["period_fraction_theory_dominant"],
                    ideal_metrics["period_fraction_theory_dominant"],
                )
                rows.append(row)

    rows.sort(key=lambda row: float(row["qualification_score"]), reverse=True)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "pair_group_ftan_diagnostic_scan.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config": str(args.config.resolve()),
        "biased_run_directory": str(args.biased_run_directory.resolve()),
        "ideal_run_directory": str(args.ideal_run_directory.resolve()),
        "frequency_range_hz": [frequency_min, frequency_max],
        "theory_tube_half_width_mps": args.theory_tube_half_width_mps,
        "automatic_ridge_picks_used": False,
        "best_candidate": rows[0],
        "current_1000m_symmetric_alpha5": next(
            (
                row
                for row in rows
                if np.isclose(float(row["distance_m"]), 1000.0)
                and row["branch"] == "symmetric"
                and np.isclose(float(row["gaussian_alpha"]), 5.0)
            ),
            None,
        ),
    }
    json_path = output / "pair_group_ftan_diagnostic_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
