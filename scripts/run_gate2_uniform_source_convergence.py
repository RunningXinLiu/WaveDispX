#!/usr/bin/env python3
"""Pre-registered Gate 2 uniform-source convergence experiment.

This script decides the background source count before any directional-source
selection method is evaluated.  It uses the WaveDispX DWM passive forward
operator, the independently calculated inter-receiver active Green function,
and the fixed Gate 0 comparison band.  The decision rule is recorded in
``docs/NCF_EGF_EQUIVALENCE_PLAN_CN.md`` v4.7.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.config import config_from_mapping
from wavedispx.egf_reference import build_passive_egf_reference
from wavedispx.geometry import build_geometry
from wavedispx.passive import accumulate_passive_correlation, build_passive_transfer
from wavedispx.quality import egf_equivalence_metrics


SOURCE_COUNTS = (72, 144, 288, 576, 1152)
NOISE_SEEDS = (4201, 4202, 4203)
WINDOW_COUNT = 120
WINDOW_S = 25.0
DT_S = 0.01
PAIR_DISTANCE_M = 600.0
EVALUATION_BAND_HZ = (4.0, 12.0)
VELOCITY_RANGE_MPS = (150.0, 500.0)
CC_TOLERANCE = 0.02
PHASE_TOLERANCE_RAD = 0.05
GEOMETRY_SEED = 20260815


def _uniform_annulus(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a reproducible stratified annulus without a preferred azimuth."""

    rng = np.random.default_rng(GEOMETRY_SEED + count)
    angular_step = 2.0 * np.pi / count
    angle = angular_step * (np.arange(count) + rng.uniform(0.1, 0.9, count))
    angle += rng.uniform(0.0, 2.0 * np.pi)
    radius = np.sqrt(rng.uniform(1500.0**2, 1800.0**2, count))
    return radius * np.cos(angle), radius * np.sin(angle)


def _write_sources(path: Path, count: int) -> None:
    x_m, y_m = _uniform_annulus(count)
    azimuth_deg = np.mod(np.rad2deg(np.arctan2(y_m, x_m)), 360.0)
    radius_m = np.hypot(x_m, y_m)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack((x_m, y_m, azimuth_deg, radius_m)),
        delimiter=",",
        header="x_m,y_m,azimuth_deg,radius_m",
        comments="",
        fmt="%.12g",
    )


def _mapping(source_path: Path, source_count: int, output_dir: Path) -> dict:
    return {
        "run": {
            "name": f"gate2_uniform_convergence_n{source_count}",
            "mode": "passive",
            "output_directory": str(output_dir),
            "workers": 4,
            "frequency_block_size": 8,
            "source_block_size": 64,
            "radial_interpolation_step_m": 0.5,
            "store_source_responses": False,
        },
        "model": {
            "layers": [
                {
                    "thickness_m": 20.0,
                    "vp_mps": 380.0,
                    "vs_mps": 220.0,
                    "density_kgm3": 1900.0,
                    "qp": float("inf"),
                    "qs": float("inf"),
                },
                {
                    "thickness_m": float("inf"),
                    "vp_mps": 760.0,
                    "vs_mps": 440.0,
                    "density_kgm3": 2150.0,
                    "qp": float("inf"),
                    "qs": float("inf"),
                },
            ]
        },
        "geometry": {
            "receivers": {
                "layout": "line",
                "start_x_m": -300.0,
                "stop_x_m": 300.0,
                "spacing_m": 600.0,
                "y_m": 0.0,
            }
        },
        "ambient_sources": {
            "component": "vertical",
            "layout": "file",
            "count": source_count,
            "seed": GEOMETRY_SEED,
            "coordinates_file": str(source_path),
        },
        "noise": {
            "temporal_model": "stationary_gaussian",
            "seed": NOISE_SEEDS[0],
            "duration_s": WINDOW_COUNT * WINDOW_S,
            "dt_s": DT_S,
            "window_s": WINDOW_S,
            "overlap": 0.0,
            "psd_shape": "flat_band",
            "dominant_frequency_hz": 8.0,
            "minimum_frequency_hz": 3.5,
            "maximum_frequency_hz": 12.5,
            "source_amplitude": 1.0,
            "independent_sources": True,
            "event_rate_hz": 0.4,
        },
        "source_selection": {"enabled": False},
        "nnws": {"enabled": False},
        "nnws_v2": {"enabled": False},
        "pair_spz": {"enabled": False},
        "processing": {
            "correlation_normalization": "spectral_whitening",
            "dispersion_method": "phase_shift",
            "theory_solver": "qedisp",
            "frequency_min_hz": EVALUATION_BAND_HZ[0],
            "frequency_max_hz": EVALUATION_BAND_HZ[1],
            "velocity_min_mps": VELOCITY_RANGE_MPS[0],
            "velocity_max_mps": VELOCITY_RANGE_MPS[1],
            "velocity_step_mps": 2.0,
            "distance_taper_fraction": 0.15,
            "zero_padding": 8.0,
            "radial_kernel_lookup_step": 0.002,
            "picker_smoothness": 0.35,
            "picker_maximum_jump_mps": 30.0,
            "picker_minimum_energy": 0.05,
            "rayleigh_modes": 1,
            # The shared config schema requires at least one Love mode even
            # though this vertical-source ZZ experiment never evaluates it.
            "love_modes": 1,
        },
    }


def _metrics(branch, reference) -> dict[str, object]:
    return egf_equivalence_metrics(
        branch,
        reference,
        component_name="zz",
        frequency_range_hz=EVALUATION_BAND_HZ,
        minimum_velocity_mps=VELOCITY_RANGE_MPS[0],
        maximum_velocity_mps=VELOCITY_RANGE_MPS[1],
        correlation_normalization="spectral_whitening",
    )


def _decision(rows: list[dict[str, object]]) -> dict[str, object]:
    medians: dict[int, dict[str, float]] = {}
    for count in SOURCE_COUNTS:
        selected = [
            row
            for row in rows
            if row["source_count"] == count and row["branch"] == "symmetric"
        ]
        medians[count] = {
            "waveform_cc": float(
                np.median([row["waveform_correlation_no_shift"] for row in selected])
            ),
            "phase_rms_rad": float(
                np.median([row["phase_rms_rad"] for row in selected])
            ),
        }
    reference = medians[SOURCE_COUNTS[-1]]
    pass_by_count: dict[int, bool] = {}
    for count in SOURCE_COUNTS:
        values = medians[count]
        pass_by_count[count] = bool(
            reference["waveform_cc"] - values["waveform_cc"] <= CC_TOLERANCE
            and values["phase_rms_rad"] - reference["phase_rms_rad"]
            <= PHASE_TOLERANCE_RAD
        )
    selected_count = SOURCE_COUNTS[-1]
    plateau_observed = False
    for index, count in enumerate(SOURCE_COUNTS[:-1]):
        next_count = SOURCE_COUNTS[index + 1]
        if pass_by_count[count] and pass_by_count[next_count]:
            selected_count = count
            plateau_observed = True
            break
    return {
        "selected_source_count": selected_count,
        "plateau_observed": plateau_observed,
        "reference_source_count": SOURCE_COUNTS[-1],
        "cc_tolerance": CC_TOLERANCE,
        "phase_tolerance_rad": PHASE_TOLERANCE_RAD,
        "median_by_source_count": {str(k): v for k, v in medians.items()},
        "within_reference_tolerance": {
            str(k): value for k, value in pass_by_count.items()
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    rows: list[dict[str, object]],
    decision: dict[str, object],
    traces: dict[tuple[int, int], np.ndarray],
    lag_time_s: np.ndarray,
    reference_trace: np.ndarray,
    output: Path,
) -> None:
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    metric_specs = (
        ("waveform_correlation_no_shift", "Waveform correlation", axes[0, 0]),
        ("phase_rms_rad", "Wrapped phase RMS (rad)", axes[0, 1]),
        ("normalized_l2_unit_norm", "Unit-norm waveform L2", axes[1, 0]),
    )
    for metric, ylabel, axis in metric_specs:
        for seed_index, seed in enumerate(NOISE_SEEDS):
            values = [
                next(
                    float(row[metric])
                    for row in rows
                    if row["source_count"] == count
                    and row["noise_seed"] == seed
                    and row["branch"] == "symmetric"
                )
                for count in SOURCE_COUNTS
            ]
            axis.plot(
                SOURCE_COUNTS,
                values,
                marker="o",
                linewidth=1.0,
                alpha=0.75,
                color=colors[seed_index],
                label=f"seed {seed}" if metric == metric_specs[0][0] else None,
            )
        medians = [
            np.median(
                [
                    float(row[metric])
                    for row in rows
                    if row["source_count"] == count
                    and row["branch"] == "symmetric"
                ]
            )
            for count in SOURCE_COUNTS
        ]
        axis.plot(
            SOURCE_COUNTS,
            medians,
            marker="s",
            linewidth=2.3,
            color="black",
            label="median" if metric == metric_specs[0][0] else None,
        )
        axis.axvline(
            int(decision["selected_source_count"]),
            color="#CC79A7",
            linestyle="--",
            linewidth=1.5,
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(SOURCE_COUNTS, labels=[str(value) for value in SOURCE_COUNTS])
        axis.set_xlabel("Uniform annulus source count")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2)

    axis = axes[1, 1]
    selected_count = int(decision["selected_source_count"])
    sample_count = min(lag_time_s.size, reference_trace.size)
    selected_time = lag_time_s[:sample_count]
    time_mask = selected_time <= 5.0
    reference = reference_trace[:sample_count].copy()
    reference /= max(np.max(np.abs(reference[time_mask])), np.finfo(float).eps)
    axis.plot(
        selected_time[time_mask],
        reference[time_mask],
        color="black",
        linewidth=2.0,
        label="active EGF",
    )
    for offset, seed in enumerate(NOISE_SEEDS):
        # Match the independently recorded interferometric calibration used by
        # egf_equivalence_metrics; plotting the raw correlation here would be
        # a physically inconsistent waveform comparison.
        trace = -np.gradient(
            traces[(selected_count, seed)][:sample_count], DT_S
        )
        trace /= max(np.max(np.abs(trace[time_mask])), np.finfo(float).eps)
        axis.plot(
            selected_time[time_mask],
            trace[time_mask] - 1.4 * (offset + 1),
            color=colors[offset],
            linewidth=1.0,
            label=f"calibrated NCF seed {seed}",
        )
    axis.axvspan(
        PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[1],
        PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[0],
        color="0.85",
        zorder=-1,
        label="evaluation arrival window",
    )
    axis.set_xlabel("Lag time (s)")
    axis.set_ylabel("Normalized waveform + offset")
    axis.set_title(f"Selected {selected_count}-source control")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    figure.suptitle(
        "Gate 2 uniform-source convergence before method comparison\n"
        f"decision: {selected_count} sources; plateau={decision['plateau_observed']}",
        fontsize=14,
    )
    figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    traces: dict[tuple[int, int], np.ndarray] = {}
    lag_time_s: np.ndarray | None = None
    reference_trace: np.ndarray | None = None
    started = time.perf_counter()

    for count in SOURCE_COUNTS:
        source_path = output_dir / "source_coordinates" / f"uniform_{count}.csv"
        _write_sources(source_path, count)
        base_config = config_from_mapping(
            _mapping(source_path, count, output_dir / f"n{count}"),
            base_directory=output_dir,
        )
        geometry = build_geometry(base_config.receivers, base_config.ambient_sources)
        transfer_start = time.perf_counter()
        transfer = build_passive_transfer(base_config, geometry)
        transfer_seconds = time.perf_counter() - transfer_start
        reference = build_passive_egf_reference(base_config, geometry)
        if reference_trace is None:
            component_index = reference.component_names.index("zz")
            reference_trace = reference.components[:, 0, component_index].copy()

        for seed in NOISE_SEEDS:
            print(f"source_count={count}, noise_seed={seed}", flush=True)
            config = replace(base_config, noise=replace(base_config.noise, seed=seed))
            run_start = time.perf_counter()
            correlation = accumulate_passive_correlation(
                config,
                geometry,
                transfer,
                egf_reference=reference,
            )
            run_seconds = time.perf_counter() - run_start
            branches = correlation.strategy_stacks[
                "baseline_linear"
            ].component_branches["zz"]
            if not np.isclose(branches.symmetric.distance_m[0], PAIR_DISTANCE_M):
                raise AssertionError("Gate 2 convergence did not produce the 600 m pair")
            lag_time_s = branches.symmetric.lag_time_s.copy()
            traces[(count, seed)] = branches.symmetric.data[:, 0].copy()
            for branch_name in ("positive", "negative", "symmetric"):
                metrics = _metrics(getattr(branches, branch_name), reference)
                rows.append(
                    {
                        "source_count": count,
                        "noise_seed": seed,
                        "branch": branch_name,
                        "window_count": WINDOW_COUNT,
                        "transfer_seconds": transfer_seconds,
                        "correlation_seconds": run_seconds,
                        **metrics,
                    }
                )

    decision = _decision(rows)
    _write_csv(output_dir / "gate2_uniform_source_convergence_metrics.csv", rows)
    np.savez_compressed(
        output_dir / "gate2_uniform_source_convergence_waveforms.npz",
        lag_time_s=lag_time_s,
        active_reference=reference_trace,
        **{
            f"n{count}_seed{seed}": trace
            for (count, seed), trace in traces.items()
        },
    )
    _plot(
        rows,
        decision,
        traces,
        lag_time_s,
        reference_trace,
        output_dir / "gate2_uniform_source_convergence",
    )
    summary = {
        "experiment": "Gate2_uniform_source_convergence",
        "status": "complete",
        "plan_version": "v4.7",
        "source_counts": SOURCE_COUNTS,
        "noise_seeds": NOISE_SEEDS,
        "window_count": WINDOW_COUNT,
        "window_s": WINDOW_S,
        "dt_s": DT_S,
        "pair_distance_m": PAIR_DISTANCE_M,
        "source_annulus_m": [1500.0, 1800.0],
        "source_psd_band_hz": [3.5, 12.5],
        "evaluation_band_hz": EVALUATION_BAND_HZ,
        "correlation_normalization": "spectral_whitening",
        "active_reference": "WaveDispX DWM interreceiver unit-impulse GF",
        "theory_solver": "qedisp",
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_boundary": (
            "This convergence decision applies only to the whitened ZZ phase/"
            "dispersion track and does not establish absolute-amplitude recovery."
        ),
    }
    (output_dir / "gate2_uniform_source_convergence_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output_python/gate2_uniform_source_convergence"),
    )
    args = parser.parse_args()
    run(args.output_directory.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
