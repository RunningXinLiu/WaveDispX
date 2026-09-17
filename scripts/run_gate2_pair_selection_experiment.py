#!/usr/bin/env python3
"""Run the pre-registered Gate 2 pair-local source-selection experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import numpy as np

from nnws import rmsr_selective_stacking
from wavedispx.config import config_from_mapping
from wavedispx.egf_reference import build_passive_egf_reference
from wavedispx.geometry import build_geometry
from wavedispx.nnws import NNWSConfig, train_nnws_ccfs
from wavedispx.noise import synthesize_mixed_noise_window_band
from wavedispx.passive import build_passive_transfer
from wavedispx.processing import NCFGather, receiver_pair_ncf_branches
from wavedispx.quality import egf_equivalence_metrics
from wavedispx.temporal_reweighting import fit_temporal_antisymmetry_weights

from run_gate2_uniform_source_convergence import (
    DT_S,
    EVALUATION_BAND_HZ,
    GEOMETRY_SEED,
    PAIR_DISTANCE_M,
    VELOCITY_RANGE_MPS,
    WINDOW_COUNT,
    WINDOW_S,
    _mapping as convergence_mapping,
    _uniform_annulus,
)


BACKGROUND_COUNT = 576
SECTOR_COUNT = 144
SECTOR_WIDTH_DEG = 12.0
SECTOR_AMPLITUDES = (0.0, 0.75, 1.5, 2.5)
SCHEDULE_SEED = 20260816
STACKMASTER_COMMIT = "66909971f69df60cb25d9e8b732a102d939cd2aa"
DEFAULT_STACKMASTER_ROOT = Path(".wavedispx-tools/StackMaster-upstream")
RMSR_THRESHOLD = 5.0
MAXIMUM_LAG_S = 5.25
SIGNAL_WINDOW_S = (
    PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[1],
    PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[0],
)
MAXIMUM_PERIOD_S = 1.0 / EVALUATION_BAND_HZ[0]
PERMUTATION_COUNT = 20
METHOD_NAMES = (
    "linear",
    "rmsr_g5",
    "robust_stackmaster",
    "temporal_antisymmetry_adapter",
    "nnws_upstream",
    "nnws_population_eps_diagnostic",
    "oracle_zero_sector",
)
SCENARIOS = (
    ("uniform", None),
    ("far75", 75.0),
    ("near12", 12.0),
    ("mirror168", 168.0),
)


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_exact_robust(stackmaster_root: Path):
    root = stackmaster_root.resolve()
    if _git_head(root) != STACKMASTER_COMMIT:
        raise RuntimeError(f"StackMaster must be pinned to {STACKMASTER_COMMIT}")
    if "tslearn" not in sys.modules:
        tslearn = types.ModuleType("tslearn")
        tslearn_utils = types.ModuleType("tslearn.utils")
        tslearn_clustering = types.ModuleType("tslearn.clustering")
        tslearn_utils.to_time_series_dataset = None
        tslearn_clustering.TimeSeriesKMeans = None
        sys.modules["tslearn"] = tslearn
        sys.modules["tslearn.utils"] = tslearn_utils
        sys.modules["tslearn.clustering"] = tslearn_clustering
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from stackmaster.core import robust

    return robust


def _schedule() -> np.ndarray:
    if WINDOW_COUNT % len(SECTOR_AMPLITUDES):
        raise AssertionError("window count must balance all sector states")
    values = np.repeat(
        np.asarray(SECTOR_AMPLITUDES, dtype=np.float64),
        WINDOW_COUNT // len(SECTOR_AMPLITUDES),
    )
    return values[np.random.default_rng(SCHEDULE_SEED).permutation(WINDOW_COUNT)]


def _sector_sources(center_deg: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(GEOMETRY_SEED + int(round(center_deg * 10.0)))
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


def _activity(center_deg: float | None, power_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if center_deg is None:
        return np.ones((WINDOW_COUNT, BACKGROUND_COUNT)), np.zeros(WINDOW_COUNT)
    state = _schedule()
    activity = np.ones((WINDOW_COUNT, BACKGROUND_COUNT + SECTOR_COUNT))
    activity[:, BACKGROUND_COUNT:] = state[:, None]
    if power_mode == "constant_total_power":
        scale = np.sqrt(
            BACKGROUND_COUNT
            / (BACKGROUND_COUNT + SECTOR_COUNT * np.square(state))
        )
        activity *= scale[:, None]
    elif power_mode != "additional_power":
        raise ValueError("power_mode must be additional_power or constant_total_power")
    return activity, state


def _write_coordinates(
    path: Path, x_m: np.ndarray, y_m: np.ndarray, source_group: np.ndarray
) -> None:
    azimuth = np.mod(np.rad2deg(np.arctan2(y_m, x_m)), 360.0)
    np.savetxt(
        path,
        np.column_stack((x_m, y_m, source_group, azimuth, np.hypot(x_m, y_m))),
        delimiter=",",
        header="x_m,y_m,source_group,azimuth_deg,radius_m",
        comments="",
        fmt="%.12g",
    )


def _write_activity(path: Path, activity: np.ndarray) -> None:
    window_index, source_index = np.indices(activity.shape)
    np.savetxt(
        path,
        np.column_stack(
            (window_index.reshape(-1), source_index.reshape(-1), activity.reshape(-1))
        ),
        delimiter=",",
        header="window_index,source_index,amplitude_multiplier",
        comments="",
        fmt=("%d", "%d", "%.12g"),
    )


def _case_config(source_path: Path, source_count: int, output_dir: Path, seed: int):
    mapping = convergence_mapping(source_path, source_count, output_dir)
    mapping["noise"]["seed"] = seed
    return config_from_mapping(mapping, base_directory=output_dir)


def _simulate_short_ccfs(config, geometry, transfer, activity: np.ndarray):
    rng = np.random.default_rng(config.noise.seed)
    matrices = {name: [] for name in ("positive", "negative", "symmetric")}
    lag_time_s = None
    for window_index in range(WINDOW_COUNT):
        realization = synthesize_mixed_noise_window_band(
            transfer.spectrum,
            transfer.frequency_hz,
            config.noise,
            rng,
            source_block_size=config.source_block_size,
            source_activity_weights=activity[window_index],
        )
        branches = receiver_pair_ncf_branches(
            realization.record,
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
            raise AssertionError("Gate 2 must produce one 600 m receiver pair")
        lag_time_s = branches.lag_time_s
        for name in matrices:
            matrices[name].append(getattr(branches, name).data[:, 0])
    return (
        {name: np.asarray(values, dtype=np.float64) for name, values in matrices.items()},
        np.asarray(lag_time_s, dtype=np.float64),
    )


def _full_lag_matrix(matrices: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (matrices["negative"][:, 1:][:, ::-1], matrices["positive"]), axis=1
    )


def _effective_count(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    return float(np.square(np.sum(values)) / np.sum(np.square(values)))


def _weighted_stacks(
    matrices: dict[str, np.ndarray], weights: np.ndarray
) -> dict[str, np.ndarray]:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    window_count = next(iter(matrices.values())).shape[0]
    if values.size != window_count or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("method weights must be one finite non-negative value per window")
    denominator = float(np.sum(values))
    if denominator <= np.finfo(float).eps:
        raise RuntimeError("method produced a zero-weight stack")
    return {
        name: np.sum(matrix * values[:, None], axis=0) / denominator
        for name, matrix in matrices.items()
    }


def _fit_methods(
    matrices: dict[str, np.ndarray],
    lag_time_s: np.ndarray,
    sector_state: np.ndarray,
    robust,
    *,
    nnws_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    full = _full_lag_matrix(matrices)
    window_count = full.shape[0]
    weights: dict[str, np.ndarray] = {"linear": np.ones(window_count)}
    diagnostics: dict[str, dict[str, Any]] = {
        "linear": {"scope": "all_windows", "status": "complete"}
    }

    def register(name: str, values: np.ndarray) -> np.ndarray:
        candidate = np.asarray(values, dtype=np.float64).reshape(-1)
        if candidate.size != window_count:
            raise RuntimeError(
                f"{name} returned {candidate.size} weights for {window_count} windows"
            )
        if np.any(~np.isfinite(candidate)) or np.any(candidate < 0.0):
            raise RuntimeError(f"{name} returned non-finite or negative weights")
        if float(np.sum(candidate)) <= np.finfo(float).eps:
            raise RuntimeError(f"{name} produced an unusable zero-weight stack")
        weights[name] = candidate
        return candidate

    try:
        _, _, rmsr_mask = rmsr_selective_stacking(
            full,
            tau=0.5 * MAXIMUM_PERIOD_S,
            tmin=SIGNAL_WINDOW_S[0],
            tmax=SIGNAL_WINDOW_S[1],
            delta=DT_S,
            threshold=RMSR_THRESHOLD,
        )
        register("rmsr_g5", rmsr_mask.astype(np.float64))
        diagnostics["rmsr_g5"] = {
            "scope": "Xie_Fan_G_1_plus_5_over_M",
            "status": "complete",
            "selected_window_count": int(np.count_nonzero(rmsr_mask)),
        }
    except Exception as exc:
        diagnostics["rmsr_g5"] = {
            "scope": "Xie_Fan_G_1_plus_5_over_M",
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
        }

    try:
        _, robust_weights, robust_steps = robust(full, stat=True)
        register("robust_stackmaster", robust_weights)
        diagnostics["robust_stackmaster"] = {
            "scope": "exact_pinned_StackMaster_robust",
            "status": "complete",
            "iterations": int(robust_steps),
        }
    except Exception as exc:
        diagnostics["robust_stackmaster"] = {
            "scope": "exact_pinned_StackMaster_robust",
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
        }

    try:
        temporal = fit_temporal_antisymmetry_weights(
            matrices["positive"],
            matrices["negative"],
            lag_time_s,
            window_s=(0.0, 5.0),
        )
        register("temporal_antisymmetry_adapter", temporal.weights)
        diagnostics["temporal_antisymmetry_adapter"] = {
            "scope": temporal.scope,
            "status": "complete",
            "objective": temporal.objective,
            "uniform_objective": temporal.uniform_objective,
            "objective_scale": temporal.objective_scale,
            "scaled_objective": temporal.scaled_objective,
            "iterations": temporal.iterations,
        }
    except Exception as exc:
        diagnostics["temporal_antisymmetry_adapter"] = {
            "scope": "single_pair_whitened_scheme_v_adapter",
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
        }

    try:
        nnws = train_nnws_ccfs(
            full,
            dt_s=DT_S,
            signal_window_s=SIGNAL_WINDOW_S,
            maximum_period_s=MAXIMUM_PERIOD_S,
            config=NNWSConfig(
                epochs=200,
                learning_rate=1e-3,
                snr_weight=0.8,
                symmetry_weight=0.2,
                std_weight=0.01,
                std_mode="upstream_sample",
                seed=nnws_seed,
                device="cpu",
            ),
        )
        register("nnws_upstream", nnws.weights)
        diagnostics["nnws_upstream"] = {
            "scope": "exact_upstream_v1_pairwise",
            "status": "complete",
            "training_snr": nnws.training_snr,
            "reported_snr": nnws.reported_snr,
            "symmetry_ratio": nnws.symmetry_ratio,
            "final_loss": nnws.history["total_loss"][-1],
        }
    except Exception as exc:
        diagnostics["nnws_upstream"] = {
            "scope": "exact_upstream_v1_pairwise",
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
        }

    try:
        nnws_stable = train_nnws_ccfs(
            full,
            dt_s=DT_S,
            signal_window_s=SIGNAL_WINDOW_S,
            maximum_period_s=MAXIMUM_PERIOD_S,
            config=NNWSConfig(
                epochs=200,
                learning_rate=1e-3,
                snr_weight=0.8,
                symmetry_weight=0.2,
                std_weight=0.01,
                std_mode="population_eps",
                seed=nnws_seed,
                device="cpu",
            ),
        )
        register("nnws_population_eps_diagnostic", nnws_stable.weights)
        diagnostics["nnws_population_eps_diagnostic"] = {
            "scope": "numerical_stability_diagnostic_not_exact_upstream",
            "status": "complete",
            "training_snr": nnws_stable.training_snr,
            "reported_snr": nnws_stable.reported_snr,
            "symmetry_ratio": nnws_stable.symmetry_ratio,
            "final_loss": nnws_stable.history["total_loss"][-1],
        }
    except Exception as exc:
        diagnostics["nnws_population_eps_diagnostic"] = {
            "scope": "numerical_stability_diagnostic_not_exact_upstream",
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
        }

    oracle = np.isclose(sector_state, 0.0).astype(np.float64)
    if not np.any(oracle):
        oracle = np.ones(WINDOW_COUNT)
    weights["oracle_zero_sector"] = oracle
    diagnostics["oracle_zero_sector"] = {
        "scope": "synthetic_truth_upper_bound_only",
        "status": "complete",
        "selected_window_count": int(np.count_nonzero(oracle)),
    }
    return weights, diagnostics


def _gather(
    trace: np.ndarray,
    lag_time_s: np.ndarray,
    *,
    valid_window_count: int = WINDOW_COUNT,
) -> NCFGather:
    return NCFGather(
        data=np.asarray(trace, dtype=np.float64)[:, None],
        lag_time_s=lag_time_s,
        distance_m=np.asarray((PAIR_DISTANCE_M,)),
        pair_count=np.asarray((1,), dtype=np.int64),
        valid_windows=valid_window_count,
    )


def _evaluate_stack(
    stacks: dict[str, np.ndarray],
    lag_time_s: np.ndarray,
    reference,
    *,
    valid_window_count: int = WINDOW_COUNT,
) -> dict[str, dict[str, object]]:
    return {
        branch: egf_equivalence_metrics(
            _gather(trace, lag_time_s, valid_window_count=valid_window_count),
            reference,
            component_name="zz",
            frequency_range_hz=EVALUATION_BAND_HZ,
            minimum_velocity_mps=VELOCITY_RANGE_MPS[0],
            maximum_velocity_mps=VELOCITY_RANGE_MPS[1],
            correlation_normalization="spectral_whitening",
        )
        for branch, trace in stacks.items()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_case(
    *,
    scenario: str,
    center_deg: float | None,
    power_mode: str,
    wavefield_seed: int,
    output_dir: Path,
    robust,
    nnws_seed: int,
    reference=None,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    case_name = scenario if center_deg is None else f"{scenario}_{power_mode}"
    case_dir = output_dir / f"seed_{wavefield_seed}" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    x_m, y_m, source_group = _coordinates(center_deg)
    source_path = case_dir / "sources.csv"
    _write_coordinates(source_path, x_m, y_m, source_group)
    activity, sector_state = _activity(center_deg, power_mode)
    _write_activity(case_dir / "source_activity.csv", activity)
    config = _case_config(source_path, x_m.size, case_dir, wavefield_seed)
    geometry = build_geometry(config.receivers, config.ambient_sources)
    transfer = build_passive_transfer(config, geometry)
    if reference is None:
        reference = build_passive_egf_reference(config, geometry)
    matrices, lag_time_s = _simulate_short_ccfs(config, geometry, transfer, activity)
    paired_clean_matrices = None
    if center_deg is not None:
        paired_clean_activity = np.ones_like(activity)
        paired_clean_activity[:, BACKGROUND_COUNT:] = 0.0
        paired_clean_matrices, paired_clean_lag = _simulate_short_ccfs(
            config,
            geometry,
            transfer,
            paired_clean_activity,
        )
        if not np.array_equal(paired_clean_lag, lag_time_s):
            raise AssertionError("paired-clean lag axis changed")
    weights, diagnostics = _fit_methods(
        matrices,
        lag_time_s,
        sector_state,
        robust,
        nnws_seed=nnws_seed,
    )

    metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {
        "lag_time_s": lag_time_s,
        "positive_windows": matrices["positive"],
        "negative_windows": matrices["negative"],
        "symmetric_windows": matrices["symmetric"],
        "sector_state": sector_state,
        "activity": activity,
        "source_x_m": x_m,
        "source_y_m": y_m,
        "source_group": source_group,
        "active_reference_time_s": np.asarray(reference.time_s, dtype=np.float64),
        "active_reference_zz": np.asarray(
            reference.components[:, 0, reference.component_names.index("zz")],
            dtype=np.float64,
        ),
    }
    if paired_clean_matrices is not None:
        paired_clean_stacks = _weighted_stacks(
            paired_clean_matrices,
            np.ones(WINDOW_COUNT),
        )
        paired_clean_metrics = _evaluate_stack(
            paired_clean_stacks,
            lag_time_s,
            reference,
        )
        for branch, trace in paired_clean_stacks.items():
            saved[f"stack__paired_clean_counterfactual__{branch}"] = trace
            metric_rows.append(
                {
                    "scenario": scenario,
                    "power_mode": power_mode,
                    "case_name": case_name,
                    "wavefield_seed": wavefield_seed,
                    "nnws_seed": nnws_seed,
                    "method": "paired_clean_counterfactual",
                    "control_replica": -1,
                    "branch": branch,
                    "status": "complete",
                    "failure_reason": "",
                    "effective_window_count": float(WINDOW_COUNT),
                    "nonzero_window_count": WINDOW_COUNT,
                    **paired_clean_metrics[branch],
                }
            )
    for method in METHOD_NAMES:
        method_diagnostics = diagnostics.get(method, {})
        if method_diagnostics.get("status") != "failed":
            continue
        for branch in matrices:
            metric_rows.append(
                {
                    "scenario": scenario,
                    "power_mode": power_mode,
                    "case_name": case_name,
                    "wavefield_seed": wavefield_seed,
                    "nnws_seed": nnws_seed,
                    "method": method,
                    "control_replica": -1,
                    "branch": branch,
                    "status": "failed",
                    "failure_reason": method_diagnostics.get("failure_reason", ""),
                }
            )
    rng = np.random.default_rng(900000 + wavefield_seed)
    for method, method_weights in weights.items():
        stacks = _weighted_stacks(matrices, method_weights)
        saved[f"weights__{method}"] = method_weights
        for branch, trace in stacks.items():
            saved[f"stack__{method}__{branch}"] = trace
        metrics = _evaluate_stack(stacks, lag_time_s, reference)
        neff = _effective_count(method_weights)
        for branch, values in metrics.items():
            metric_rows.append(
                {
                    "scenario": scenario,
                    "power_mode": power_mode,
                    "case_name": case_name,
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
                    "power_mode": power_mode,
                    "case_name": case_name,
                    "wavefield_seed": wavefield_seed,
                    "window_index": window_index,
                    "sector_amplitude": sector_state[window_index],
                    "method": method,
                    "weight": float(value),
                }
            )

        raw_power = np.mean(np.square(activity), axis=0)
        effective_power = np.average(
            np.square(activity), axis=0, weights=method_weights
        )
        for source_index in range(x_m.size):
            source_rows.append(
                {
                    "scenario": scenario,
                    "power_mode": power_mode,
                    "case_name": case_name,
                    "wavefield_seed": wavefield_seed,
                    "method": method,
                    "source_index": source_index,
                    "source_group": int(source_group[source_index]),
                    "x_m": float(x_m[source_index]),
                    "y_m": float(y_m[source_index]),
                    "azimuth_deg": float(
                        np.mod(np.rad2deg(np.arctan2(y_m[source_index], x_m[source_index])), 360.0)
                    ),
                    "raw_mean_power": float(raw_power[source_index]),
                    "effective_mean_power": float(effective_power[source_index]),
                }
            )

        if method == "linear":
            continue
        permutation_metrics: list[tuple[int, float, dict[str, np.ndarray]]] = []
        for replica in range(PERMUTATION_COUNT):
            permuted_weights = method_weights[rng.permutation(WINDOW_COUNT)]
            control_stacks = _weighted_stacks(matrices, permuted_weights)
            control_metrics = _evaluate_stack(control_stacks, lag_time_s, reference)
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
                        "power_mode": power_mode,
                        "case_name": case_name,
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
        representative = min(permutation_metrics, key=lambda item: abs(item[1] - median_cc))
        saved[f"representative_permutation__{method}"] = np.asarray(
            (representative[0], median_cc), dtype=np.float64
        )
        for branch, trace in representative[2].items():
            saved[f"stack__{method}_permuted_median__{branch}"] = trace

    np.savez_compressed(case_dir / "gate2_pair_selection.npz", **saved)
    (case_dir / "method_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=True) + "\n"
    )
    print(
        f"completed {case_name}, seed={wavefield_seed}: "
        + ", ".join(
            f"{name} Neff={_effective_count(value):.1f}"
            for name, value in weights.items()
        )
        + (
            "; failed="
            + ",".join(
                name
                for name, values in diagnostics.items()
                if values.get("status") == "failed"
            )
            if any(values.get("status") == "failed" for values in diagnostics.values())
            else ""
        ),
        flush=True,
    )
    return reference, metric_rows, weight_rows, source_rows


def run(
    *,
    output_dir: Path,
    wavefield_seeds: tuple[int, ...],
    nnws_seeds: tuple[int, ...],
    stackmaster_root: Path,
) -> dict[str, Any]:
    if len(nnws_seeds) not in {1, len(wavefield_seeds)}:
        raise ValueError("provide one NNWS seed for all runs or one per wavefield seed")
    output_dir.mkdir(parents=True, exist_ok=True)
    robust = _load_exact_robust(stackmaster_root)
    all_metrics: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    all_sources: list[dict[str, Any]] = []
    reference = None
    for seed_index, wavefield_seed in enumerate(wavefield_seeds):
        nnws_seed = nnws_seeds[0] if len(nnws_seeds) == 1 else nnws_seeds[seed_index]
        for scenario, center_deg in SCENARIOS:
            power_modes = ("uniform",) if center_deg is None else (
                "additional_power",
                "constant_total_power",
            )
            for power_mode in power_modes:
                reference, metrics, weights, sources = _run_case(
                    scenario=scenario,
                    center_deg=center_deg,
                    power_mode=power_mode,
                    wavefield_seed=wavefield_seed,
                    output_dir=output_dir,
                    robust=robust,
                    nnws_seed=nnws_seed,
                    reference=reference,
                )
                all_metrics.extend(metrics)
                all_weights.extend(weights)
                all_sources.extend(sources)
    _write_csv(output_dir / "gate2_pair_metrics.csv", all_metrics)
    _write_csv(output_dir / "gate2_window_weights.csv", all_weights)
    _write_csv(output_dir / "gate2_source_effective_power.csv", all_sources)
    summary = {
        "experiment": "Gate2_pair_local_source_selection",
        "status": "complete",
        "plan_version": "v4.9",
        "wavefield_seeds": wavefield_seeds,
        "nnws_seeds": nnws_seeds,
        "background_source_count": BACKGROUND_COUNT,
        "sector_source_count": SECTOR_COUNT,
        "sector_width_deg": SECTOR_WIDTH_DEG,
        "scenario_centers_deg": {name: center for name, center in SCENARIOS},
        "window_count": WINDOW_COUNT,
        "window_s": WINDOW_S,
        "dt_s": DT_S,
        "evaluation_band_hz": EVALUATION_BAND_HZ,
        "pair_distance_m": PAIR_DISTANCE_M,
        "methods": METHOD_NAMES,
        "failed_primary_rows": sum(
            row.get("status") == "failed" and row.get("control_replica") == -1
            for row in all_metrics
        ),
        "failed_method_cases": sorted(
            {
                f"{row['case_name']}:{row['method']}"
                for row in all_metrics
                if row.get("status") == "failed" and row.get("control_replica") == -1
            }
        ),
        "permutation_count_per_method": PERMUTATION_COUNT,
        "stackmaster_commit": STACKMASTER_COMMIT,
        "temporal_method_scope": "single_pair_whitened_scheme_v_adapter",
        "nnws_population_eps_scope": "diagnostic_only_not_exact_upstream",
        "paired_clean_scope": "same_geometry_same_random_stream_sector_activity_zero",
        "source_truth_usage": "oracle and post-stacking mechanism audit only",
    }
    (output_dir / "gate2_pair_selection_summary.json").write_text(
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
        default=Path("output_python/gate2_pair_selection_development"),
    )
    parser.add_argument("--wavefield-seeds", type=_parse_seeds, default=(4301,))
    parser.add_argument("--nnws-seeds", type=_parse_seeds, default=(0,))
    parser.add_argument(
        "--stackmaster-root", type=Path, default=DEFAULT_STACKMASTER_ROOT
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
