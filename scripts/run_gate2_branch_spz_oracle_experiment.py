#!/usr/bin/env python3
"""Run the pre-registered v5.2 truth-only branch-SPZ oracle development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wavedispx.beamforming_selection import angular_target_weights, circular_distance_deg
from wavedispx.egf_reference import PassiveEgfReference

from run_gate2_pair_selection_experiment import (
    PERMUTATION_COUNT,
    _effective_count,
    _evaluate_stack,
    _write_csv,
)
from run_gate2_sparse_event_selection_experiment import SCENARIOS, WINDOW_COUNT


PLAN_VERSION = "v5.2"
POSITIVE_SOURCE_CENTER_DEG = 0.0
NEGATIVE_SOURCE_CENTER_DEG = 180.0
SPZ_HALF_WIDTH_DEG = 20.0
TARGET_REGULARIZATION = 0.05
MINIMUM_EFFECTIVE_FRACTION = 0.50


def _target(azimuth_deg: np.ndarray, center_deg: float) -> np.ndarray:
    target = (
        circular_distance_deg(azimuth_deg, center_deg) <= SPZ_HALF_WIDTH_DEG
    ).astype(np.float64)
    if not np.any(target):
        raise RuntimeError("SPZ target contains no azimuth samples")
    return target / np.sum(target)


def _weighted(trace_by_window: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    traces = np.asarray(trace_by_window, dtype=np.float64)
    if traces.ndim != 2 or traces.shape[0] != values.size:
        raise ValueError("branch traces and weights are incompatible")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("branch weights must be finite and non-negative")
    denominator = float(np.sum(values))
    if denominator <= np.finfo(float).eps:
        raise RuntimeError("branch weights have zero sum")
    return np.sum(traces * values[:, None], axis=0) / denominator


def _branch_stacks(
    positive_windows: np.ndarray,
    negative_windows: np.ndarray,
    positive_weights: np.ndarray,
    negative_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    positive = _weighted(positive_windows, positive_weights)
    negative = _weighted(negative_windows, negative_weights)
    return {
        "positive": positive,
        "negative": negative,
        "symmetric": 0.5 * (positive + negative),
    }


def _reference(npz: np.lib.npyio.NpzFile) -> PassiveEgfReference:
    active = np.asarray(npz["active_reference_zz"], dtype=np.float64)
    time = np.asarray(npz["active_reference_time_s"], dtype=np.float64)
    return PassiveEgfReference(
        time_s=time,
        distance_m=np.array([600.0]),
        components=active[:, None, None],
        component_names=("zz",),
        pair_i=np.array([0], dtype=np.int64),
        pair_j=np.array([1], dtype=np.int64),
        pair_distance_index=np.array([0], dtype=np.int64),
        source_time_function="unit_impulse",
        calibration_operator="negative_time_derivative_of_ncf",
        delay_removed_s=0.0,
        maximum_frequency_hz=12.5,
        build_seconds=0.0,
    )


def _append_metrics(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    seed: int,
    method: str,
    replica: int,
    effective_window_count: float,
    metrics: dict[str, dict[str, object]],
) -> None:
    for branch, values in metrics.items():
        rows.append(
            {
                "scenario": scenario,
                "wavefield_seed": seed,
                "method": method,
                "control_replica": replica,
                "branch": branch,
                "status": "complete",
                "effective_window_count": effective_window_count,
                **values,
            }
        )


def _case(
    *,
    input_root: Path,
    output_root: Path,
    scenario: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    input_case = input_root / f"seed_{seed}" / scenario
    output_case = output_root / f"seed_{seed}" / scenario
    output_case.mkdir(parents=True, exist_ok=True)
    with np.load(input_case / "gate2_source_aware_beamforming.npz") as source:
        lag = np.asarray(source["lag_time_s"], dtype=np.float64)
        positive_windows = np.asarray(source["positive_windows"], dtype=np.float64)
        negative_windows = np.asarray(source["negative_windows"], dtype=np.float64)
        truth = np.asarray(source["truth_angular_power"], dtype=np.float64)
        azimuth = np.asarray(source["beam_azimuth_deg"], dtype=np.float64)
        reference = _reference(source)
        saved: dict[str, np.ndarray] = {
            "lag_time_s": lag,
            "active_reference_time_s": np.asarray(
                source["active_reference_time_s"], dtype=np.float64
            ),
            "active_reference_zz": np.asarray(
                source["active_reference_zz"], dtype=np.float64
            ),
            "beam_azimuth_deg": azimuth,
            "truth_angular_power": truth,
            "target_positive": _target(azimuth, POSITIVE_SOURCE_CENTER_DEG),
            "target_negative": _target(azimuth, NEGATIVE_SOURCE_CENTER_DEG),
            "stack__linear__positive": np.asarray(
                source["stack__linear__positive"], dtype=np.float64
            ),
            "stack__linear__negative": np.asarray(
                source["stack__linear__negative"], dtype=np.float64
            ),
            "stack__linear__symmetric": np.asarray(
                source["stack__linear__symmetric"], dtype=np.float64
            ),
        }
        for branch in ("positive", "negative", "symmetric"):
            clean_key = f"stack__paired_clean_counterfactual__{branch}"
            if clean_key in source.files:
                saved[clean_key] = np.asarray(source[clean_key], dtype=np.float64)

    positive_fit = angular_target_weights(
        truth,
        saved["target_positive"],
        regularization=TARGET_REGULARIZATION,
        minimum_effective_fraction=MINIMUM_EFFECTIVE_FRACTION,
    )
    negative_fit = angular_target_weights(
        truth,
        saved["target_negative"],
        regularization=TARGET_REGULARIZATION,
        minimum_effective_fraction=MINIMUM_EFFECTIVE_FRACTION,
    )
    method = "oracle_branch_spz"
    stacks = _branch_stacks(
        positive_windows,
        negative_windows,
        positive_fit.weights,
        negative_fit.weights,
    )
    saved["weights__oracle_branch_spz__positive"] = positive_fit.weights
    saved["weights__oracle_branch_spz__negative"] = negative_fit.weights
    for branch, trace in stacks.items():
        saved[f"stack__{method}__{branch}"] = trace

    metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    linear_stacks = {
        branch: saved[f"stack__linear__{branch}"]
        for branch in ("positive", "negative", "symmetric")
    }
    _append_metrics(
        metric_rows,
        scenario=scenario,
        seed=seed,
        method="linear",
        replica=-1,
        effective_window_count=float(WINDOW_COUNT),
        metrics=_evaluate_stack(linear_stacks, lag, reference),
    )
    if "stack__paired_clean_counterfactual__symmetric" in saved:
        clean_stacks = {
            branch: saved[f"stack__paired_clean_counterfactual__{branch}"]
            for branch in ("positive", "negative", "symmetric")
        }
        _append_metrics(
            metric_rows,
            scenario=scenario,
            seed=seed,
            method="paired_clean_counterfactual",
            replica=-1,
            effective_window_count=float(WINDOW_COUNT),
            metrics=_evaluate_stack(clean_stacks, lag, reference),
        )
    method_neff = min(
        _effective_count(positive_fit.weights),
        _effective_count(negative_fit.weights),
    )
    _append_metrics(
        metric_rows,
        scenario=scenario,
        seed=seed,
        method=method,
        replica=-1,
        effective_window_count=method_neff,
        metrics=_evaluate_stack(stacks, lag, reference),
    )
    for window_index in range(WINDOW_COUNT):
        for branch, weights in (
            ("positive", positive_fit.weights),
            ("negative", negative_fit.weights),
        ):
            weight_rows.append(
                {
                    "scenario": scenario,
                    "wavefield_seed": seed,
                    "window_index": window_index,
                    "method": method,
                    "branch": branch,
                    "weight": float(weights[window_index]),
                }
            )

    rng = np.random.default_rng(1_052_000 + seed)
    permutation_cc: list[float] = []
    for replica in range(PERMUTATION_COUNT):
        permuted_stacks = _branch_stacks(
            positive_windows,
            negative_windows,
            positive_fit.weights[rng.permutation(WINDOW_COUNT)],
            negative_fit.weights[rng.permutation(WINDOW_COUNT)],
        )
        permuted_metrics = _evaluate_stack(permuted_stacks, lag, reference)
        permutation_cc.append(
            float(permuted_metrics["symmetric"]["waveform_correlation_no_shift"])
        )
        _append_metrics(
            metric_rows,
            scenario=scenario,
            seed=seed,
            method=f"{method}_permuted",
            replica=replica,
            effective_window_count=method_neff,
            metrics=permuted_metrics,
        )

    np.savez_compressed(
        output_case / "gate2_branch_spz_oracle.npz",
        **saved,
    )
    diagnostics = {
        "scenario": scenario,
        "wavefield_seed": seed,
        "method": method,
        "truth_usage": "source_event_azimuth_truth_leaking_diagnostic_only",
        "positive_source_center_deg": POSITIVE_SOURCE_CENTER_DEG,
        "negative_source_center_deg": NEGATIVE_SOURCE_CENTER_DEG,
        "spz_half_width_deg": SPZ_HALF_WIDTH_DEG,
        "positive_neff": positive_fit.effective_window_count,
        "negative_neff": negative_fit.effective_window_count,
        "positive_unweighted_misfit": positive_fit.unweighted_misfit,
        "positive_weighted_misfit": positive_fit.weighted_misfit,
        "negative_unweighted_misfit": negative_fit.unweighted_misfit,
        "negative_weighted_misfit": negative_fit.weighted_misfit,
        "permuted_symmetric_cc_median": float(np.median(permutation_cc)),
    }
    (output_case / "method_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(
        f"completed {scenario}, seed={seed}, positive Neff={positive_fit.effective_window_count:.1f}, "
        f"negative Neff={negative_fit.effective_window_count:.1f}",
        flush=True,
    )
    return metric_rows, weight_rows, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("output_python/gate2_source_aware_development"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output_python/gate2_branch_spz_development"),
    )
    parser.add_argument("--wavefield-seed", type=int, default=5501)
    args = parser.parse_args()
    input_root = args.input_directory.resolve()
    output_root = args.output_directory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for scenario, _ in SCENARIOS:
        metrics, weights, case_diagnostics = _case(
            input_root=input_root,
            output_root=output_root,
            scenario=scenario,
            seed=args.wavefield_seed,
        )
        all_metrics.extend(metrics)
        all_weights.extend(weights)
        diagnostics.append(case_diagnostics)
    _write_csv(output_root / "gate2_branch_spz_metrics.csv", all_metrics)
    _write_csv(output_root / "gate2_branch_spz_window_weights.csv", all_weights)
    summary = {
        "experiment": "Gate2_v5.2_truth_only_branch_stationary_phase_oracle",
        "plan_version": PLAN_VERSION,
        "status": "complete",
        "wavefield_seed": args.wavefield_seed,
        "input_directory": str(input_root),
        "positive_source_center_deg": POSITIVE_SOURCE_CENTER_DEG,
        "negative_source_center_deg": NEGATIVE_SOURCE_CENTER_DEG,
        "spz_half_width_deg": SPZ_HALF_WIDTH_DEG,
        "truth_usage": "oracle only; not deployable",
        "active_truth_usage": "evaluation only",
        "ftan_gate_status": "not_yet_evaluated",
        "beam_method_authorized": False,
        "blind_authorized": False,
        "diagnostics": diagnostics,
    }
    (output_root / "gate2_branch_spz_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
