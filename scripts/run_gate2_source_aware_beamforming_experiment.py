#!/usr/bin/env python3
"""Run the pre-registered v5.1 auxiliary-array source-azimuth experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr

from wavedispx.beamforming_selection import (
    PhaseOnlyBeamformer,
    angular_uniformity_weights,
    circular_distance_deg,
)
from wavedispx.config import config_from_mapping
from wavedispx.egf_reference import build_passive_egf_reference
from wavedispx.geometry import build_geometry
from wavedispx.noise import synthesize_mixed_noise_window_band
from wavedispx.passive import build_passive_transfer
from wavedispx.processing import receiver_pair_ncf_branches

from run_gate2_pair_selection_experiment import (
    MAXIMUM_LAG_S,
    PAIR_DISTANCE_M,
    PERMUTATION_COUNT,
    _effective_count,
    _evaluate_stack,
    _weighted_stacks,
    _write_csv,
)
from run_gate2_sparse_event_selection_experiment import (
    BACKGROUND_COUNT,
    BASELINE_MAXIMUM_PHASE_RMS_RAD,
    BASELINE_MINIMUM_CC,
    EVENT_RATE_HZ,
    SCENARIOS,
    SECTOR_COUNT,
    WINDOW_COUNT,
    _coordinates,
    _write_coordinates,
)
from run_gate2_uniform_source_convergence import DT_S, WINDOW_S, _mapping


PLAN_VERSION = "v5.1"
AUXILIARY_RECEIVER_COUNT = 13
AUXILIARY_RADIUS_M = 45.0
BEAM_FREQUENCIES_HZ = np.arange(4.5, 12.0, 1.0)
BEAM_AZIMUTH_DEG = np.arange(0.0, 360.0, 5.0)
BEAM_VELOCITY_MPS = np.arange(160.0, 441.0, 40.0)
BEAM_SNAPSHOT_S = 4.0
BEAM_OVERLAP = 0.5
UNIFORMITY_REGULARIZATION = 0.05
MINIMUM_EFFECTIVE_FRACTION = 0.50
DIRECTION_MINIMUM_SPEARMAN = 0.25
DIRECTION_MINIMUM_AUC = 0.65
DIRECTION_MAXIMUM_PEAK_ERROR_DEG = 10.0
SECTOR_AUDIT_HALF_WIDTH_DEG = 10.0


def _auxiliary_coordinates() -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(AUXILIARY_RECEIVER_COUNT, dtype=np.float64)
    radius = AUXILIARY_RADIUS_M * np.sqrt(index / (AUXILIARY_RECEIVER_COUNT - 1))
    angle = np.deg2rad(index * 137.507764)
    return radius * np.cos(angle), radius * np.sin(angle)


def _receiver_coordinates() -> tuple[np.ndarray, np.ndarray]:
    auxiliary_x, auxiliary_y = _auxiliary_coordinates()
    return (
        np.concatenate(((-300.0, 300.0), auxiliary_x)),
        np.concatenate(((0.0, 0.0), auxiliary_y)),
    )


def _write_receivers(path: Path) -> None:
    x_m, y_m = _receiver_coordinates()
    np.savetxt(
        path,
        np.column_stack((x_m, y_m)),
        delimiter=",",
        header="x_m,y_m",
        comments="",
        fmt="%.12g",
    )


def _pair_config(source_path: Path, source_count: int, output_dir: Path, seed: int):
    mapping = _mapping(source_path, source_count, output_dir)
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


def _array_config(
    source_path: Path,
    receiver_path: Path,
    source_count: int,
    output_dir: Path,
    seed: int,
):
    mapping = _mapping(source_path, source_count, output_dir)
    mapping["run"]["name"] = f"gate2_source_aware_beam_seed_{seed}"
    mapping["geometry"]["receivers"] = {
        "layout": "file",
        "coordinates_file": str(receiver_path),
        # Required by the shared array-processing schema.  This experiment
        # never distance-bins the auxiliary receivers; only indices 0/1 form
        # the target CCF.
        "distance_bin_m": 10.0,
    }
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


def _event_angular_power(event_rows: list[dict[str, Any]]) -> np.ndarray:
    power = np.zeros((WINDOW_COUNT, BEAM_AZIMUTH_DEG.size), dtype=np.float64)
    for event in event_rows:
        distance = circular_distance_deg(BEAM_AZIMUTH_DEG, float(event["azimuth_deg"]))
        bin_index = int(np.argmin(distance))
        power[int(event["window_index"]), bin_index] += float(event["event_weight"]) ** 2
    empty = np.sum(power, axis=1) <= np.finfo(float).eps
    power[empty] = 1.0
    return power / np.sum(power, axis=1, keepdims=True)


def _simulate(
    config,
    geometry,
    transfer,
    source_group: np.ndarray,
    activity: np.ndarray,
    *,
    beamformer: PhaseOnlyBeamformer | None,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray | None,
]:
    rng = np.random.default_rng(config.noise.seed)
    matrices = {name: [] for name in ("positive", "negative", "symmetric")}
    event_rows: list[dict[str, Any]] = []
    sector_event_count = np.zeros(WINDOW_COUNT, dtype=np.int64)
    beam_power: list[np.ndarray] = []
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
            raise AssertionError("poisson_ricker must return source events")
        if beamformer is not None:
            beam_power.append(
                beamformer.estimate(realization.record[:, 2:]).angular_power
            )
        target_record = np.sign(realization.record[:, :2])
        branches = receiver_pair_ncf_branches(
            target_record,
            geometry.receiver_x_m[:2],
            geometry.receiver_y_m[:2],
            DT_S,
            window_s=WINDOW_S,
            overlap=0.0,
            maximum_lag_s=MAXIMUM_LAG_S,
            normalization="spectral_whitening",
        )
        if branches.distance_m.size != 1 or not np.isclose(
            branches.distance_m[0], PAIR_DISTANCE_M
        ):
            raise AssertionError("target receivers must remain the 600 m pair")
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
            azimuth_deg = float(
                np.mod(
                    np.rad2deg(
                        np.arctan2(
                            geometry.source_y_m[source_index],
                            geometry.source_x_m[source_index],
                        )
                    ),
                    360.0,
                )
            )
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
                    "azimuth_deg": azimuth_deg,
                }
            )
    return (
        {name: np.asarray(values, dtype=np.float64) for name, values in matrices.items()},
        np.asarray(lag_time_s, dtype=np.float64),
        event_rows,
        sector_event_count,
        np.asarray(beam_power, dtype=np.float64) if beamformer is not None else None,
    )


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    label = np.asarray(labels, dtype=bool).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(np.count_nonzero(label))
    negative_count = label.size - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float(
        (np.sum(ranks[label]) - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _sector_beam_power(power: np.ndarray, center_deg: float) -> np.ndarray:
    mask = circular_distance_deg(BEAM_AZIMUTH_DEG, center_deg) <= SECTOR_AUDIT_HALF_WIDTH_DEG
    return np.sum(power[:, mask], axis=1)


def _case(
    *,
    scenario: str,
    center_deg: float | None,
    wavefield_seed: int,
    output_dir: Path,
    reference=None,
) -> tuple[
    Any,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    np.ndarray,
]:
    case_dir = output_dir / f"seed_{wavefield_seed}" / scenario
    case_dir.mkdir(parents=True, exist_ok=True)
    source_x, source_y, source_group = _coordinates(center_deg)
    source_path = case_dir / "sources.csv"
    receiver_path = case_dir / "receivers.csv"
    _write_coordinates(source_path, source_x, source_y, source_group)
    _write_receivers(receiver_path)
    pair_config = _pair_config(source_path, source_x.size, case_dir, wavefield_seed)
    if reference is None:
        pair_geometry = build_geometry(pair_config.receivers, pair_config.ambient_sources)
        reference = build_passive_egf_reference(pair_config, pair_geometry)
    config = _array_config(
        source_path,
        receiver_path,
        source_x.size,
        case_dir,
        wavefield_seed,
    )
    geometry = build_geometry(config.receivers, config.ambient_sources)
    transfer = build_passive_transfer(config, geometry)
    auxiliary_x, auxiliary_y = _auxiliary_coordinates()
    beamformer = PhaseOnlyBeamformer(
        auxiliary_x,
        auxiliary_y,
        dt_s=DT_S,
        window_samples=int(round(WINDOW_S / DT_S)),
        snapshot_s=BEAM_SNAPSHOT_S,
        overlap=BEAM_OVERLAP,
        frequencies_hz=BEAM_FREQUENCIES_HZ,
        azimuth_deg=BEAM_AZIMUTH_DEG,
        velocity_mps=BEAM_VELOCITY_MPS,
    )
    activity = np.ones(source_x.size, dtype=np.float64)
    matrices, lag_time_s, event_rows, sector_event_count, beam_power = _simulate(
        config,
        geometry,
        transfer,
        source_group,
        activity,
        beamformer=beamformer,
    )
    if beam_power is None or beam_power.shape != (WINDOW_COUNT, BEAM_AZIMUTH_DEG.size):
        raise AssertionError("beam power matrix is incomplete")
    truth_power = _event_angular_power(event_rows)

    paired_clean_matrices = None
    if center_deg is not None:
        clean_activity = activity.copy()
        clean_activity[BACKGROUND_COUNT:] = 0.0
        (
            paired_clean_matrices,
            clean_lag,
            clean_events,
            clean_sector_count,
            _,
        ) = _simulate(
            config,
            geometry,
            transfer,
            source_group,
            clean_activity,
            beamformer=None,
        )
        if not np.array_equal(clean_lag, lag_time_s) or np.any(clean_sector_count):
            raise AssertionError("paired-clean target changed its lag or retained sector events")
        main_background = [
            (row["window_index"], row["source_index"], row["time_in_window_s"], row["event_weight"])
            for row in event_rows
            if row["source_group"] == 0
        ]
        clean_background = [
            (row["window_index"], row["source_index"], row["time_in_window_s"], row["event_weight"])
            for row in clean_events
        ]
        if main_background != clean_background:
            raise AssertionError("paired-clean background event catalog is not identical")

    beam_fit = angular_uniformity_weights(
        beam_power,
        regularization=UNIFORMITY_REGULARIZATION,
        minimum_effective_fraction=MINIMUM_EFFECTIVE_FRACTION,
    )
    oracle_fit = angular_uniformity_weights(
        truth_power,
        regularization=UNIFORMITY_REGULARIZATION,
        minimum_effective_fraction=MINIMUM_EFFECTIVE_FRACTION,
    )
    methods = {
        "linear": np.ones(WINDOW_COUNT, dtype=np.float64),
        "beam_uniformity": beam_fit.weights,
        "oracle_event_uniformity": oracle_fit.weights,
    }
    method_diagnostics = {
        "beam_uniformity": {
            "success": beam_fit.success,
            "message": beam_fit.message,
            "iterations": beam_fit.iterations,
            "unweighted_misfit": beam_fit.unweighted_misfit,
            "weighted_misfit": beam_fit.weighted_misfit,
            "effective_window_count": beam_fit.effective_window_count,
            "truth_usage": "none",
        },
        "oracle_event_uniformity": {
            "success": oracle_fit.success,
            "message": oracle_fit.message,
            "iterations": oracle_fit.iterations,
            "unweighted_misfit": oracle_fit.unweighted_misfit,
            "weighted_misfit": oracle_fit.weighted_misfit,
            "effective_window_count": oracle_fit.effective_window_count,
            "truth_usage": "source_event_azimuth_truth_leaking_diagnostic_only",
        },
    }
    metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {
        "lag_time_s": lag_time_s,
        "positive_windows": matrices["positive"],
        "negative_windows": matrices["negative"],
        "symmetric_windows": matrices["symmetric"],
        "beam_azimuth_deg": BEAM_AZIMUTH_DEG,
        "beam_velocity_mps": BEAM_VELOCITY_MPS,
        "beam_angular_power": beam_power,
        "truth_angular_power": truth_power,
        "sector_event_count": sector_event_count,
        "active_reference_time_s": np.asarray(reference.time_s, dtype=np.float64),
        "active_reference_zz": np.asarray(reference.components[:, 0, 0], dtype=np.float64),
    }
    if paired_clean_matrices is not None:
        clean_stacks = _weighted_stacks(paired_clean_matrices, np.ones(WINDOW_COUNT))
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
                    "method": "paired_clean_counterfactual",
                    "control_replica": -1,
                    "branch": branch,
                    "status": "complete",
                    "effective_window_count": float(WINDOW_COUNT),
                    **clean_metrics[branch],
                }
            )
    rng = np.random.default_rng(1_051_000 + wavefield_seed)
    for method, weights in methods.items():
        stacks = _weighted_stacks(matrices, weights)
        saved[f"weights__{method}"] = weights
        for branch, trace in stacks.items():
            saved[f"stack__{method}__{branch}"] = trace
        metrics = _evaluate_stack(
            stacks,
            lag_time_s,
            reference,
            valid_window_count=WINDOW_COUNT,
        )
        neff = _effective_count(weights)
        for branch, values in metrics.items():
            metric_rows.append(
                {
                    "scenario": scenario,
                    "wavefield_seed": wavefield_seed,
                    "method": method,
                    "control_replica": -1,
                    "branch": branch,
                    "status": "complete",
                    "effective_window_count": neff,
                    **values,
                }
            )
        for window_index, weight in enumerate(weights):
            weight_rows.append(
                {
                    "scenario": scenario,
                    "wavefield_seed": wavefield_seed,
                    "window_index": window_index,
                    "sector_event_count": int(sector_event_count[window_index]),
                    "method": method,
                    "weight": float(weight),
                }
            )
        if method == "linear":
            continue
        permutation_cc = []
        for replica in range(PERMUTATION_COUNT):
            permuted = weights[rng.permutation(WINDOW_COUNT)]
            control_stacks = _weighted_stacks(matrices, permuted)
            control_metrics = _evaluate_stack(
                control_stacks,
                lag_time_s,
                reference,
                valid_window_count=WINDOW_COUNT,
            )
            permutation_cc.append(
                float(control_metrics["symmetric"]["waveform_correlation_no_shift"])
            )
            for branch, values in control_metrics.items():
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "wavefield_seed": wavefield_seed,
                        "method": f"{method}_permuted",
                        "control_replica": replica,
                        "branch": branch,
                        "status": "complete",
                        "effective_window_count": neff,
                        **values,
                    }
                )
        method_diagnostics[method]["permuted_symmetric_cc_median"] = float(
            np.median(permutation_cc)
        )

    direction = {
        "scenario": scenario,
        "wavefield_seed": wavefield_seed,
        "sector_center_deg": center_deg,
        "status": "not_applicable" if center_deg is None else "complete",
    }
    if center_deg is not None:
        sector_power = _sector_beam_power(beam_power, center_deg)
        correlation = spearmanr(sector_power, sector_event_count).statistic
        auc = _roc_auc(sector_event_count > 0, sector_power)
        direction.update(
            {
                "sector_power_spearman": float(correlation),
                "sector_presence_auc": auc,
            }
        )

    _write_csv(case_dir / "source_events.csv", event_rows)
    np.savez_compressed(case_dir / "gate2_source_aware_beamforming.npz", **saved)
    (case_dir / "method_diagnostics.json").write_text(
        json.dumps(method_diagnostics, indent=2) + "\n"
    )
    print(
        f"completed {scenario}, seed={wavefield_seed}, events={len(event_rows)}, "
        f"sector_events={int(np.sum(sector_event_count))}, "
        f"beam Neff={beam_fit.effective_window_count:.1f}, "
        f"oracle Neff={oracle_fit.effective_window_count:.1f}",
        flush=True,
    )
    return reference, metric_rows, weight_rows, direction, np.mean(beam_power, axis=0)


def _waveform_gate(metrics: list[dict[str, Any]], diagnostics_by_case: dict[str, Any]) -> dict[str, Any]:
    primary = [
        row
        for row in metrics
        if row["control_replica"] == -1 and row["method"] in {"linear", "beam_uniformity"}
    ]
    lookup = {(row["scenario"], row["method"], row["branch"]): row for row in primary}
    checks = []
    for scenario, _ in SCENARIOS:
        linear = lookup[(scenario, "linear", "symmetric")]
        beam = lookup[(scenario, "beam_uniformity", "symmetric")]
        positive_linear = lookup[(scenario, "linear", "positive")]
        positive_beam = lookup[(scenario, "beam_uniformity", "positive")]
        negative_linear = lookup[(scenario, "linear", "negative")]
        negative_beam = lookup[(scenario, "beam_uniformity", "negative")]
        permutation_median = diagnostics_by_case[scenario]["beam_uniformity"][
            "permuted_symmetric_cc_median"
        ]
        checks.append(
            {
                "scenario": scenario,
                "cc_improved": beam["waveform_correlation_no_shift"]
                > linear["waveform_correlation_no_shift"],
                "l2_improved": beam["normalized_l2_unit_norm"]
                < linear["normalized_l2_unit_norm"],
                "phase_improved": beam["phase_rms_rad"] < linear["phase_rms_rad"],
                "prearrival_not_worse": beam["prearrival_ghost_energy_fraction"]
                <= linear["prearrival_ghost_energy_fraction"],
                "positive_branch_cc_improved": positive_beam[
                    "waveform_correlation_no_shift"
                ]
                > positive_linear["waveform_correlation_no_shift"],
                "negative_branch_cc_improved": negative_beam[
                    "waveform_correlation_no_shift"
                ]
                > negative_linear["waveform_correlation_no_shift"],
                "beat_permutation": beam["waveform_correlation_no_shift"]
                > permutation_median,
            }
        )
    uniform = next(item for item in checks if item["scenario"] == "uniform")
    biased = [item for item in checks if item["scenario"] != "uniform"]
    uniform_non_degraded = all(
        uniform[key]
        for key in (
            "l2_improved",
            "phase_improved",
            "prearrival_not_worse",
        )
    )
    biased_passed = all(all(value for key, value in item.items() if key != "scenario") for item in biased)
    return {
        "scope": "waveform_phase_prearrival_only_ftan_still_required",
        "uniform_non_degraded": uniform_non_degraded,
        "biased_all_passed": biased_passed,
        "passed": bool(uniform_non_degraded and biased_passed),
        "checks": checks,
    }


def run(*, output_dir: Path, wavefield_seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    mean_beam_by_scenario: dict[str, np.ndarray] = {}
    diagnostics_by_case: dict[str, Any] = {}
    reference = None
    for scenario, center_deg in SCENARIOS:
        reference, metrics, weights, direction, mean_beam = _case(
            scenario=scenario,
            center_deg=center_deg,
            wavefield_seed=wavefield_seed,
            output_dir=output_dir,
            reference=reference,
        )
        all_metrics.extend(metrics)
        all_weights.extend(weights)
        direction_rows.append(direction)
        mean_beam_by_scenario[scenario] = mean_beam
        diagnostics_by_case[scenario] = json.loads(
            (
                output_dir
                / f"seed_{wavefield_seed}"
                / scenario
                / "method_diagnostics.json"
            ).read_text()
        )

    uniform_mean = mean_beam_by_scenario["uniform"]
    for row in direction_rows:
        if row["status"] != "complete":
            continue
        excess = mean_beam_by_scenario[str(row["scenario"])] - uniform_mean
        peak_deg = float(BEAM_AZIMUTH_DEG[int(np.argmax(excess))])
        peak_error = float(circular_distance_deg(peak_deg, float(row["sector_center_deg"])))
        row["mean_excess_peak_azimuth_deg"] = peak_deg
        row["mean_excess_peak_error_deg"] = peak_error
        row["spearman_passed"] = row["sector_power_spearman"] >= DIRECTION_MINIMUM_SPEARMAN
        row["auc_passed"] = row["sector_presence_auc"] >= DIRECTION_MINIMUM_AUC
        row["peak_passed"] = peak_error <= DIRECTION_MAXIMUM_PEAK_ERROR_DEG
        row["direction_gate_passed"] = bool(
            row["spearman_passed"] and row["auc_passed"] and row["peak_passed"]
        )

    _write_csv(output_dir / "gate2_source_aware_metrics.csv", all_metrics)
    _write_csv(output_dir / "gate2_source_aware_window_weights.csv", all_weights)
    _write_csv(output_dir / "gate2_source_aware_direction_metrics.csv", direction_rows)
    baseline = [
        row
        for row in all_metrics
        if row["control_replica"] == -1
        and row["branch"] == "symmetric"
        and (
            (row["scenario"] == "uniform" and row["method"] == "linear")
            or row["method"] == "paired_clean_counterfactual"
        )
    ]
    baseline_passed = bool(baseline) and all(
        row["waveform_correlation_no_shift"] >= BASELINE_MINIMUM_CC
        and row["phase_rms_rad"] <= BASELINE_MAXIMUM_PHASE_RMS_RAD
        for row in baseline
    )
    direction_passed = all(
        bool(row.get("direction_gate_passed"))
        for row in direction_rows
        if row["status"] == "complete"
    )
    waveform_gate = _waveform_gate(all_metrics, diagnostics_by_case)
    summary = {
        "experiment": "Gate2_v5.1_auxiliary_array_source_aware_pair_weighting",
        "plan_version": PLAN_VERSION,
        "status": "complete",
        "wavefield_seed": wavefield_seed,
        "target_pair_m": [[-300.0, 0.0], [300.0, 0.0]],
        "auxiliary_receiver_count": AUXILIARY_RECEIVER_COUNT,
        "beam_frequencies_hz": BEAM_FREQUENCIES_HZ.tolist(),
        "beam_azimuth_deg": BEAM_AZIMUTH_DEG.tolist(),
        "beam_velocity_mps": BEAM_VELOCITY_MPS.tolist(),
        "beam_snapshot_s": BEAM_SNAPSHOT_S,
        "beam_overlap": BEAM_OVERLAP,
        "baseline_gate_passed": baseline_passed,
        "direction_gate_passed": direction_passed,
        "waveform_gate": waveform_gate,
        "ftan_gate_status": "not_yet_evaluated",
        "blind_authorized": False,
        "blind_authorization_rule": (
            "direction, waveform, and independent FTAN gates must all pass before seeds 5601-5605"
        ),
        "source_truth_usage": "direction audit and oracle_event_uniformity only",
        "active_truth_usage": "evaluation only",
    }
    (output_dir / "gate2_source_aware_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output_python/gate2_source_aware_development"),
    )
    parser.add_argument("--wavefield-seed", type=int, default=5501)
    args = parser.parse_args()
    run(output_dir=args.output_directory.resolve(), wavefield_seed=args.wavefield_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
