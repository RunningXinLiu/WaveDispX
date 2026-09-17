#!/usr/bin/env python3
"""Compare WaveDispX pair FTAN with the supplied MATLAB demonstration picks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from wavedispx.processing import pair_ftan


def _distance_m(first_lon_lat: np.ndarray, second_lon_lat: np.ndarray) -> float:
    lon1, lat1 = np.radians(first_lon_lat[:2])
    lon2, lat2 = np.radians(second_lon_lat[:2])
    haversine = (
        np.sin(0.5 * (lat2 - lat1)) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon2 - lon1)) ** 2
    )
    return float(2.0 * 6_371_000.0 * np.arcsin(np.sqrt(haversine)))


def _metrics(error_mps: np.ndarray) -> dict[str, float]:
    absolute = np.abs(error_mps)
    return {
        "median_absolute_difference_mps": float(np.median(absolute)),
        "p90_absolute_difference_mps": float(np.percentile(absolute, 90.0)),
        "maximum_absolute_difference_mps": float(np.max(absolute)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "demo_directory",
        type=Path,
        help="Directory containing the provided EGFAnalysisTimeFreq demonstration",
    )
    parser.add_argument("--phase-method", choices=("time_domain", "frequency_domain"), default="time_domain")
    parser.add_argument("--phase-anchor-period-s", type=float, default=25.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_python/matlab_pair_ftan_reference"),
    )
    args = parser.parse_args()

    egf_path = args.demo_directory / "EGFs/GFcn.KMI-MC01_10-50s_10Mon.dat"
    group_path = args.demo_directory / "GDisp.GFcn.KMI-MC01_10-50s_10Mon.dat"
    phase_path = args.demo_directory / "CDisp.T.GFcn.KMI-MC01_10-50s_10Mon.dat"
    first_lines = egf_path.read_text().splitlines()[:2]
    first_lon_lat = np.fromstring(first_lines[0], sep=" ")
    second_lon_lat = np.fromstring(first_lines[1], sep=" ")
    distance_m = _distance_m(first_lon_lat, second_lon_lat)
    waveform = np.loadtxt(egf_path, skiprows=2)
    group_reference = np.loadtxt(group_path, skiprows=2)
    phase_reference = np.loadtxt(phase_path, skiprows=2)
    period_s = group_reference[:, 0]
    anchor_velocity_mps = float(
        1_000.0
        * np.interp(
            args.phase_anchor_period_s,
            phase_reference[:, 0],
            phase_reference[:, 1],
        )
    )

    result = pair_ftan(
        waveform[:, 1] + waveform[:, 2],
        dt_s=float(np.median(np.diff(waveform[:, 0]))),
        distance_m=distance_m,
        frequency_range_hz=(1.0 / period_s[-1], 1.0 / period_s[0]),
        group_velocity_range_mps=(2_000.0, 5_000.0),
        phase_velocity_range_mps=(2_000.0, 5_000.0),
        periods_s=period_s,
        velocity_step_mps=2.0,
        phase_method=args.phase_method,
        phase_velocity_anchor=(args.phase_anchor_period_s, anchor_velocity_mps),
        phase_minus_group_range_mps=(-100.0, 1_500.0),
        minimum_snr=0.0,
        maximum_jump_mps=120.0,
    )
    group_reference_mps = 1_000.0 * group_reference[:, 1]
    phase_reference_mps = 1_000.0 * phase_reference[:, 1]
    group_difference_mps = result.group_curve.velocity_mps - group_reference_mps
    phase_difference_mps = result.phase_curve.velocity_mps - phase_reference_mps
    interior = (period_s >= 10.0) & (period_s <= 45.0)
    summary = {
        "source": "provided MATLAB EGFAnalysisTimeFreq demonstration",
        "distance_m": distance_m,
        "period_range_s": [float(period_s[0]), float(period_s[-1])],
        "period_count": int(period_s.size),
        "phase_method": args.phase_method,
        "gaussian_alpha": result.gaussian_alpha,
        "phase_anchor_period_s": args.phase_anchor_period_s,
        "all_periods": {
            "group": _metrics(group_difference_mps),
            "phase": _metrics(phase_difference_mps),
        },
        "interior_10_45_s": {
            "selection_rule": "10 <= period_s <= 45; reported in addition to, not instead of, all periods",
            "group": _metrics(group_difference_mps[interior]),
            "phase": _metrics(phase_difference_mps[interior]),
        },
        "interpretation_guardrail": (
            "This checks numerical agreement with saved demonstration picks. "
            "Those picks are not independent ground truth, and the Python automatic "
            "tracker is not expected to duplicate GUI edge-period decisions exactly."
        ),
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "matlab_pair_ftan_reference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output / "matlab_pair_ftan_reference_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "period_s",
                "group_python_mps",
                "group_matlab_saved_mps",
                "group_difference_mps",
                "phase_python_mps",
                "phase_matlab_saved_mps",
                "phase_difference_mps",
                "snr",
            ]
        )
        writer.writerows(
            zip(
                period_s,
                result.group_curve.velocity_mps,
                group_reference_mps,
                group_difference_mps,
                result.phase_curve.velocity_mps,
                phase_reference_mps,
                phase_difference_mps,
                result.group_curve.snr,
            )
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
