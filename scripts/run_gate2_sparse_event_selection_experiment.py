#!/usr/bin/env python3
"""Run the v5.0 literature-matched sparse-event pair-selection experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from wavedispx.config import config_from_mapping
from wavedispx.egf_reference import build_passive_egf_reference
from wavedispx.geometry import build_geometry
from wavedispx.noise import synthesize_mixed_noise_window_band
from wavedispx.passive import build_passive_transfer
from wavedispx.processing import receiver_pair_ncf_branches
from wavedispx.spz_selection import enforce_effective_count_floor

from run_gate2_pair_selection_experiment import (
    DEFAULT_STACKMASTER_ROOT,
    EVALUATION_BAND_HZ,
    MAXIMUM_LAG_S,
    METHOD_NAMES,
    PAIR_DISTANCE_M,
    PERMUTATION_COUNT,
    VELOCITY_RANGE_MPS,
    _effective_count,
    _evaluate_stack,
    _fit_methods,
    _load_exact_robust,
    _weighted_stacks,
    _write_csv,
)
from run_gate2_uniform_source_convergence import (
    DT_S,
    GEOMETRY_SEED,
    WINDOW_S,
    _mapping as convergence_mapping,
    _uniform_annulus,
)


WINDOW_COUNT = 600
BACKGROUND_COUNT = 576
SECTOR_COUNT = 58
SECTOR_WIDTH_DEG = 12.0
EXPECTED_BACKGROUND_EVENTS = 10_000.0
EVENT_RATE_HZ = EXPECTED_BACKGROUND_EVENTS / (
    BACKGROUND_COUNT * WINDOW_COUNT * WINDOW_S
)
SCENARIOS = (
    ("uniform", None),
    ("far75", 75.0),
    ("near12", 12.0),
    ("mirror168", 168.0),
)
ACTIVE_ORACLE_MINIMUM_EFFECTIVE_FRACTION = 0.10
BASELINE_MINIMUM_CC = 0.65
BASELINE_MAXIMUM_PHASE_RMS_RAD = 1.2


def _sector_sources(center_deg: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(GEOMETRY_SEED + 50_000 + int(round(center_deg * 10)))
    angle = np.deg2rad(
        center_deg
        + rng.uniform(-0.5 * SECTOR_WIDTH_DEG, 0.5 * SECTOR_WIDTH_DEG, SECTOR_COUNT)
    )
    radius = np.sqrt(rng.uniform(1500.0**2, 1800.0**2, SECTOR_COUNT))
    return radius * np.cos(angle), radius * np.sin(angle)


def _coordinates(center_deg: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    background_x, background_y = _uniform_annulus(BACKGROUND_COUNT)
    if center_deg is None:
        return (
            background_x,
            background_y,
            np.zeros(BACKGROUND_COUNT, dtype=np.int64),
        )
    sector_x, sector_y = _sector_sources(center_deg)
    return (
        np.concatenate((background_x, sector_x)),
        np.concatenate((background_y, sector_y)),
        np.concatenate(
            (
                np.zeros(BACKGROUND_COUNT, dtype=np.int64),
                np.ones(SECTOR_COUNT, dtype=np.int64),
            )
        ),
    )


def _write_coordinates(
    path: Path,
    x_m: np.ndarray,
    y_m: np.ndarray,
    source_group: np.ndarray,
) -> None:
    np.savetxt(
        path,
        np.column_stack(
            (
                x_m,
                y_m,
                source_group,
                np.mod(np.rad2deg(np.arctan2(y_m, x_m)), 360.0),
                np.hypot(x_m, y_m),
            )
        ),
        delimiter=",",
        header="x_m,y_m,source_group,azimuth_deg,radius_m",
        comments="",
        fmt="%.12g",
    )


def _case_config(source_path: Path, source_count: int, output_dir: Path, seed: int):
    mapping = convergence_mapping(source_path, source_count, output_dir)
    mapping["noise"].update(
        {
            "temporal_model": "poisson_ricker",
            "seed": seed,
            "duration_s": WINDOW_COUNT * WINDOW_S,
            "psd_shape": "ricker",
            "dominant_frequency_hz": 8.0,
            "minimum_frequency_hz": 3.5,
            "maximum_frequency_hz": 12.5,
            "event_rate_hz": EVENT_RATE_HZ,
        }
    )
    return config_from_mapping(mapping, base_directory=output_dir)


def _simulate_event_ccfs(
    config,
    geometry,
    transfer,
    source_group: np.ndarray,
    activity: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]], np.ndarray]:
    rng = np.random.default_rng(config.noise.seed)
    matrices = {name: [] for name in ("positive", "negative", "symmetric")}
    event_rows: list[dict[str, Any]] = []
    sector_event_count = np.zeros(WINDOW_COUNT, dtype=np.int64)
    lag_time_s = None
    for window_index in range(WINDOW_COUNT):
        realization = synthesize_mixed_noise_window_band(
            transfer.spectrum,
            transfer.frequency_hz,
            config.noise,
            rng,
            return_source_events=True,
            source_block_size=config.source_block_size,
            source_activity_weights=activity,
        )
        if realization.source_events is None:
            raise AssertionError("poisson_ricker must return a source-event catalog")
        one_bit_record = np.sign(realization.record)
        branches = receiver_pair_ncf_branches(
            one_bit_record,
            geometry.receiver_x_m,
            geometry.receiver_y_m,
            DT_S,
            window_s=WINDOW_S,
            overlap=0.0,
            maximum_lag_s=MAXIMUM_LAG_S,
            normalization="spectral_whitening",
        )
        if branches.distance_m.size != 1 or not np.isclose(
            branches.distance_m[0], PAIR_DISTANCE_M
        ):
            raise AssertionError("v5.0 must produce one 600 m receiver pair")
        lag_time_s = branches.lag_time_s
        for name in matrices:
            matrices[name].append(getattr(branches, name).data[:, 0])
        for event_index, event in enumerate(realization.source_events):
            source_index = int(event["source_index"])
            multiplier = float(activity[source_index])
            if multiplier <= 0.0:
                continue
            group = int(source_group[source_index])
            sector_event_count[window_index] += group
            event_rows.append(
                {
                    "window_index": window_index,
                    "window_start_s": window_index * WINDOW_S,
                    "window_end_s": (window_index + 1) * WINDOW_S,
                    "event_index_in_window": event_index,
                    "source_index": source_index,
                    "source_group": group,
                    "time_in_window_s": float(event["time_in_window_s"]),
                    "event_time_s": window_index * WINDOW_S
                    + float(event["time_in_window_s"]),
                    "event_weight": float(event["weight"]) * multiplier,
                    "x_m": float(geometry.source_x_m[source_index]),
                    "y_m": float(geometry.source_y_m[source_index]),
                    "azimuth_deg": float(
                        np.mod(
                            np.rad2deg(
                                np.arctan2(
                                    geometry.source_y_m[source_index],
                                    geometry.source_x_m[source_index],
                                )
                            ),
                            360.0,
                        )
                    ),
                }
            )
    return (
        {name: np.asarray(rows, dtype=np.float64) for name, rows in matrices.items()},
        np.asarray(lag_time_s, dtype=np.float64),
        event_rows,
        sector_event_count,
    )


def _fft_bandpass(
    values: np.ndarray,
    *,
    dt_s: float,
    band_hz: tuple[float, float],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    frequency = np.fft.rfftfreq(array.shape[-1], dt_s)
    spectrum = np.fft.rfft(array, axis=-1)
    spectrum[..., (frequency < band_hz[0]) | (frequency > band_hz[1])] = 0.0
    return np.fft.irfft(spectrum, n=array.shape[-1], axis=-1)


def _active_common_nnls_weights(
    matrices: dict[str, np.ndarray],
    lag_time_s: np.ndarray,
    reference,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_count = min(lag_time_s.size, reference.time_s.size)
    target = np.asarray(reference.components[:sample_count, 0, 0], dtype=np.float64)
    target = _fft_bandpass(target, dt_s=DT_S, band_hz=EVALUATION_BAND_HZ)
    design_blocks = []
    target_blocks = []
    for branch in ("positive", "negative"):
        candidate = -np.gradient(
            np.asarray(matrices[branch][:, :sample_count], dtype=np.float64),
            DT_S,
            axis=1,
        )
        candidate = _fft_bandpass(
            candidate,
            dt_s=DT_S,
            band_hz=EVALUATION_BAND_HZ,
        )
        design_blocks.append(candidate.T)
        target_blocks.append(target)
    design = np.vstack(design_blocks)
    target_vector = np.concatenate(target_blocks)
    design_scale = max(float(np.linalg.norm(design)), np.finfo(float).eps)
    target_scale = max(float(np.linalg.norm(target_vector)), np.finfo(float).eps)
    raw_weights, residual = nnls(
        design / design_scale,
        target_vector / target_scale,
        maxiter=max(30, 10 * matrices["symmetric"].shape[0]),
    )
    if float(np.sum(raw_weights)) <= np.finfo(float).eps:
        raise RuntimeError("active-EGF common NNLS found no positive solution")
    weights = enforce_effective_count_floor(
        raw_weights,
        ACTIVE_ORACLE_MINIMUM_EFFECTIVE_FRACTION,
    )
    return weights, {
        "scope": "synthetic_active_egf_truth_leaking_reachability_only",
        "status": "complete",
        "nnls_residual": float(residual),
        "minimum_effective_fraction": ACTIVE_ORACLE_MINIMUM_EFFECTIVE_FRACTION,
    }


def _fit_all_methods(
    matrices: dict[str, np.ndarray],
    lag_time_s: np.ndarray,
    sector_event_count: np.ndarray,
    reference,
    robust,
    nnws_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    weights, diagnostics = _fit_methods(
        matrices,
        lag_time_s,
        sector_event_count,
        robust,
        nnws_seed=nnws_seed,
    )
    diagnostics["oracle_zero_sector"]["scope"] = (
        "truth_informed_windows_with_zero_sector_events"
    )
    try:
        active_weights, active_diagnostics = _active_common_nnls_weights(
            matrices,
            lag_time_s,
            reference,
        )
        weights["oracle_active_common_nnls"] = active_weights
        diagnostics["oracle_active_common_nnls"] = active_diagnostics
    except Exception as exc:
        diagnostics["oracle_active_common_nnls"] = {
            "scope": "synthetic_active_egf_truth_leaking_reachability_only",
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
        }
    return weights, diagnostics


def _run_case(
    *,
    scenario: str,
    center_deg: float | None,
    wavefield_seed: int,
    nnws_seed: int,
    output_dir: Path,
    robust,
    reference=None,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    case_dir = output_dir / f"seed_{wavefield_seed}" / scenario
    case_dir.mkdir(parents=True, exist_ok=True)
    x_m, y_m, source_group = _coordinates(center_deg)
    source_path = case_dir / "sources.csv"
    _write_coordinates(source_path, x_m, y_m, source_group)
    config = _case_config(source_path, x_m.size, case_dir, wavefield_seed)
    geometry = build_geometry(config.receivers, config.ambient_sources)
    transfer = build_passive_transfer(config, geometry)
    if reference is None:
        reference = build_passive_egf_reference(config, geometry)

    activity = np.ones(x_m.size, dtype=np.float64)
    matrices, lag_time_s, event_rows, sector_event_count = _simulate_event_ccfs(
        config,
        geometry,
        transfer,
        source_group,
        activity,
    )
    paired_clean_matrices = None
    if center_deg is not None:
        clean_activity = activity.copy()
        clean_activity[BACKGROUND_COUNT:] = 0.0
        (
            paired_clean_matrices,
            paired_clean_lag,
            clean_event_rows,
            clean_sector_event_count,
        ) = _simulate_event_ccfs(
            config,
            geometry,
            transfer,
            source_group,
            clean_activity,
        )
        if not np.array_equal(paired_clean_lag, lag_time_s):
            raise AssertionError("paired-clean lag axis changed")
        if np.any(clean_sector_event_count):
            raise AssertionError("paired-clean catalog retained sector events")
        main_background = [
            (row["window_index"], row["source_index"], row["time_in_window_s"], row["event_weight"])
            for row in event_rows
            if row["source_group"] == 0
        ]
        clean_background = [
            (row["window_index"], row["source_index"], row["time_in_window_s"], row["event_weight"])
            for row in clean_event_rows
        ]
        if main_background != clean_background:
            raise AssertionError("paired-clean background event catalog is not identical")

    weights, diagnostics = _fit_all_methods(
        matrices,
        lag_time_s,
        sector_event_count,
        reference,
        robust,
        nnws_seed,
    )
    method_order = METHOD_NAMES + ("oracle_active_common_nnls",)
    metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    effective_event_rows: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {
        "lag_time_s": lag_time_s,
        "positive_windows": matrices["positive"],
        "negative_windows": matrices["negative"],
        "symmetric_windows": matrices["symmetric"],
        "sector_event_count": sector_event_count,
        "source_x_m": x_m,
        "source_y_m": y_m,
        "source_group": source_group,
        "active_reference_time_s": np.asarray(reference.time_s, dtype=np.float64),
        "active_reference_zz": np.asarray(reference.components[:, 0, 0], dtype=np.float64),
    }

    if paired_clean_matrices is not None:
        clean_stacks = _weighted_stacks(
            paired_clean_matrices,
            np.ones(WINDOW_COUNT),
        )
        clean_metrics = _evaluate_stack(
            clean_stacks,
            lag_time_s,
            reference,
            valid_window_count=WINDOW_COUNT,
        )
        for branch, trace in clean_stacks.items():
            saved[f"stack__paired_clean_counterfactual__{branch}"] = trace
            metric_rows.append(
                {
                    "scenario": scenario,
                    "wavefield_seed": wavefield_seed,
                    "nnws_seed": nnws_seed,
                    "method": "paired_clean_counterfactual",
                    "control_replica": -1,
                    "branch": branch,
                    "status": "complete",
                    "failure_reason": "",
                    "effective_window_count": float(WINDOW_COUNT),
                    "nonzero_window_count": WINDOW_COUNT,
                    **clean_metrics[branch],
                }
            )

    for method in method_order:
        method_diagnostics = diagnostics.get(method, {})
        if method_diagnostics.get("status") != "failed":
            continue
        for branch in matrices:
            metric_rows.append(
                {
                    "scenario": scenario,
                    "wavefield_seed": wavefield_seed,
                    "nnws_seed": nnws_seed,
                    "method": method,
                    "control_replica": -1,
                    "branch": branch,
                    "status": "failed",
                    "failure_reason": method_diagnostics.get("failure_reason", ""),
                }
            )

    rng = np.random.default_rng(950_000 + wavefield_seed)
    for method, method_weights in weights.items():
        stacks = _weighted_stacks(matrices, method_weights)
        saved[f"weights__{method}"] = method_weights
        for branch, trace in stacks.items():
            saved[f"stack__{method}__{branch}"] = trace
        metrics = _evaluate_stack(
            stacks,
            lag_time_s,
            reference,
            valid_window_count=WINDOW_COUNT,
        )
        neff = _effective_count(method_weights)
        for branch, values in metrics.items():
            metric_rows.append(
                {
                    "scenario": scenario,
                    "wavefield_seed": wavefield_seed,
                    "nnws_seed": nnws_seed,
                    "method": method,
                    "control_replica": -1,
                    "branch": branch,
                    "status": "complete",
                    "failure_reason": "",
                    "effective_window_count": neff,
                    "nonzero_window_count": int(np.count_nonzero(method_weights)),
                    **values,
                }
            )
        for window_index, value in enumerate(method_weights):
            weight_rows.append(
                {
                    "scenario": scenario,
                    "wavefield_seed": wavefield_seed,
                    "window_index": window_index,
                    "window_start_s": window_index * WINDOW_S,
                    "window_end_s": (window_index + 1) * WINDOW_S,
                    "sector_event_count": int(sector_event_count[window_index]),
                    "method": method,
                    "weight": float(value),
                }
            )
        for event in event_rows:
            effective_event_rows.append(
                {
                    "scenario": scenario,
                    "wavefield_seed": wavefield_seed,
                    "method": method,
                    **event,
                    "window_weight": float(method_weights[int(event["window_index"])]),
                }
            )

        if method == "linear":
            continue
        permutation_metrics: list[tuple[int, float, dict[str, np.ndarray]]] = []
        for replica in range(PERMUTATION_COUNT):
            permuted_weights = method_weights[rng.permutation(WINDOW_COUNT)]
            control_stacks = _weighted_stacks(matrices, permuted_weights)
            control_metrics = _evaluate_stack(
                control_stacks,
                lag_time_s,
                reference,
                valid_window_count=WINDOW_COUNT,
            )
            permutation_metrics.append(
                (
                    replica,
                    float(control_metrics["symmetric"]["waveform_correlation_no_shift"]),
                    control_stacks,
                )
            )
            for branch, values in control_metrics.items():
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "wavefield_seed": wavefield_seed,
                        "nnws_seed": nnws_seed,
                        "method": f"{method}_permuted",
                        "control_replica": replica,
                        "branch": branch,
                        "status": "complete",
                        "failure_reason": "",
                        "effective_window_count": neff,
                        "nonzero_window_count": int(np.count_nonzero(method_weights)),
                        **values,
                    }
                )
        control_cc = np.asarray([item[1] for item in permutation_metrics])
        median_cc = float(np.median(control_cc))
        representative = min(
            permutation_metrics,
            key=lambda item: abs(item[1] - median_cc),
        )
        saved[f"representative_permutation__{method}"] = np.asarray(
            (representative[0], median_cc),
            dtype=np.float64,
        )
        for branch, trace in representative[2].items():
            saved[f"stack__{method}_permuted_median__{branch}"] = trace

    _write_csv(case_dir / "source_events.csv", event_rows)
    np.savez_compressed(case_dir / "gate2_sparse_event_selection.npz", **saved)
    (case_dir / "method_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=True) + "\n"
    )
    failed = [
        name
        for name, values in diagnostics.items()
        if values.get("status") == "failed"
    ]
    print(
        f"completed {scenario}, seed={wavefield_seed}, events={len(event_rows)}, "
        f"sector_events={int(np.sum(sector_event_count))}: "
        + ", ".join(
            f"{name} Neff={_effective_count(value):.1f}"
            for name, value in weights.items()
        )
        + (f"; failed={','.join(failed)}" if failed else ""),
        flush=True,
    )
    return reference, metric_rows, weight_rows, effective_event_rows


def run(
    *,
    output_dir: Path,
    wavefield_seeds: tuple[int, ...],
    nnws_seeds: tuple[int, ...],
    stackmaster_root: Path,
) -> dict[str, Any]:
    if len(nnws_seeds) not in {1, len(wavefield_seeds)}:
        raise ValueError("provide one NNWS seed or one per wavefield seed")
    output_dir.mkdir(parents=True, exist_ok=True)
    robust = _load_exact_robust(stackmaster_root)
    all_metrics: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for seed_index, wavefield_seed in enumerate(wavefield_seeds):
        nnws_seed = nnws_seeds[0] if len(nnws_seeds) == 1 else nnws_seeds[seed_index]
        reference = None
        for scenario, center_deg in SCENARIOS:
            reference, metrics, weights, events = _run_case(
                scenario=scenario,
                center_deg=center_deg,
                wavefield_seed=wavefield_seed,
                nnws_seed=nnws_seed,
                output_dir=output_dir,
                robust=robust,
                reference=reference,
            )
            all_metrics.extend(metrics)
            all_weights.extend(weights)
            all_events.extend(events)
    _write_csv(output_dir / "gate2_sparse_event_metrics.csv", all_metrics)
    _write_csv(output_dir / "gate2_sparse_event_window_weights.csv", all_weights)
    _write_csv(output_dir / "gate2_sparse_event_effective_events.csv", all_events)

    primary = [
        row
        for row in all_metrics
        if row.get("control_replica") == -1
        and row.get("branch") == "symmetric"
        and row.get("status") == "complete"
    ]
    baseline_rows = [
        row
        for row in primary
        if (row["scenario"] == "uniform" and row["method"] == "linear")
        or row["method"] == "paired_clean_counterfactual"
    ]
    baseline_gate_passed = bool(baseline_rows) and all(
        float(row["waveform_correlation_no_shift"]) >= BASELINE_MINIMUM_CC
        and float(row["phase_rms_rad"]) <= BASELINE_MAXIMUM_PHASE_RMS_RAD
        for row in baseline_rows
    )
    summary = {
        "experiment": "Gate2_v5_sparse_event_pair_selection",
        "status": "complete",
        "plan_version": "v5.0",
        "wavefield_seeds": wavefield_seeds,
        "nnws_seeds": nnws_seeds,
        "window_count": WINDOW_COUNT,
        "window_s": WINDOW_S,
        "dt_s": DT_S,
        "background_source_site_count": BACKGROUND_COUNT,
        "sector_source_site_count": SECTOR_COUNT,
        "expected_background_event_count": EXPECTED_BACKGROUND_EVENTS,
        "expected_sector_event_count": EXPECTED_BACKGROUND_EVENTS
        * SECTOR_COUNT
        / BACKGROUND_COUNT,
        "event_rate_hz_per_site": EVENT_RATE_HZ,
        "preprocessing": "one_bit_then_spectral_whitening",
        "evaluation_band_hz": EVALUATION_BAND_HZ,
        "baseline_minimum_cc": BASELINE_MINIMUM_CC,
        "baseline_maximum_phase_rms_rad": BASELINE_MAXIMUM_PHASE_RMS_RAD,
        "baseline_gate_passed": baseline_gate_passed,
        "failed_method_cases": sorted(
            {
                f"{row['scenario']}:{row['method']}"
                for row in all_metrics
                if row.get("status") == "failed"
                and row.get("control_replica") == -1
            }
        ),
        "permutation_count_per_method": PERMUTATION_COUNT,
        "active_oracle_scope": "truth_leaking_reachability_only",
        "source_truth_usage": "event audit and oracle diagnostics only",
    }
    (output_dir / "gate2_sparse_event_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output_python/gate2_sparse_event_development"),
    )
    parser.add_argument("--wavefield-seeds", type=_parse_seeds, default=(5301,))
    parser.add_argument("--nnws-seeds", type=_parse_seeds, default=(1000,))
    parser.add_argument(
        "--stackmaster-root",
        type=Path,
        default=DEFAULT_STACKMASTER_ROOT,
    )
    args = parser.parse_args()
    run(
        output_dir=args.output_directory.resolve(),
        wavefield_seeds=args.wavefield_seeds,
        nnws_seeds=args.nnws_seeds,
        stackmaster_root=args.stackmaster_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
