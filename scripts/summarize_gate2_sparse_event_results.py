#!/usr/bin/env python3
"""Summarize blind v5.0 waveform, permutation, source, and FTAN evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_gate2_sparse_event_qc import _ftan_results, _save, _style, _theory


DEPLOYABLE_METHODS = (
    "rmsr_g5",
    "robust_stackmaster",
    "temporal_antisymmetry_adapter",
    "nnws_upstream",
)
METHOD_LABELS = {
    "rmsr_g5": "RMSR-SS",
    "robust_stackmaster": "Robust",
    "temporal_antisymmetry_adapter": "Temporal adapter",
    "nnws_upstream": "NNWS upstream",
    "nnws_population_eps_diagnostic": "NNWS population-eps diagnostic",
    "oracle_zero_sector": "Zero-sector-event oracle",
    "oracle_active_common_nnls": "Active-EGF NNLS oracle",
}
COLORS = {
    "rmsr_g5": "#0072B2",
    "robust_stackmaster": "#CC79A7",
    "temporal_antisymmetry_adapter": "#009E73",
    "nnws_upstream": "#D55E00",
}


def _image_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return float("nan")
    x = x[keep] - np.mean(x[keep])
    y = y[keep] - np.mean(y[keep])
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else float("nan")


def _line_energy(
    amplitude: np.ndarray,
    velocity_mps: np.ndarray,
    target_velocity_mps: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            np.interp(target_velocity_mps[index], velocity_mps, amplitude[:, index])
            if np.isfinite(target_velocity_mps[index])
            else np.nan
            for index in range(target_velocity_mps.size)
        ],
        dtype=np.float64,
    )


def _phase_adjacent_velocity(
    phase_mps: np.ndarray,
    period_s: np.ndarray,
    cycle: int,
    distance_m: float = 600.0,
) -> np.ndarray:
    frequency = 1.0 / period_s
    inverse = 1.0 / phase_mps + cycle / (distance_m * frequency)
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = 1.0 / inverse
    velocity[~np.isfinite(velocity) | (velocity <= 0.0)] = np.nan
    return velocity


def _ftan_rows(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed_dir in sorted(run_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        for case_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            scenario = case_dir.name
            archive = case_dir / "gate2_sparse_event_selection.npz"
            if not archive.exists():
                continue
            with np.load(archive) as npz:
                results = _ftan_results(npz)
                for key in npz.files:
                    prefix = "stack__"
                    suffix = "__symmetric"
                    if not key.startswith(prefix) or not key.endswith(suffix):
                        continue
                    method = key[len(prefix) : -len(suffix)]
                    if method in results:
                        continue
                    from wavedispx.processing import pair_ftan

                    results[method] = pair_ftan(
                        npz[key],
                        dt_s=float(np.median(np.diff(npz["lag_time_s"]))),
                        distance_m=600.0,
                        frequency_range_hz=(4.0, 12.0),
                        group_velocity_range_mps=(150.0, 500.0),
                        phase_velocity_range_mps=(150.0, 500.0),
                        period_count=121,
                        velocity_step_mps=2.0,
                        gaussian_alpha=120.0,
                        phase_method="frequency_domain",
                        minimum_snr=0.0,
                        input_kind="correlation",
                    )
            theory_period, theory_group, theory_phase = _theory(case_dir, seed)
            active = results["active_reference"]
            for method, result in results.items():
                if result is None or method.endswith("_permuted_median"):
                    continue
                periods = result.group_image.period_s
                group = np.interp(periods, theory_period, theory_group)
                phase = np.interp(periods, theory_period, theory_phase)
                group_energy = _line_energy(
                    result.group_image.amplitude,
                    result.group_image.velocity_mps,
                    group,
                )
                correct_phase = _line_energy(
                    result.phase_image.amplitude,
                    result.phase_image.velocity_mps,
                    phase,
                )
                adjacent = []
                for cycle in (-1, 1):
                    adjacent_velocity = _phase_adjacent_velocity(phase, periods, cycle)
                    adjacent.append(
                        _line_energy(
                            result.phase_image.amplitude,
                            result.phase_image.velocity_mps,
                            adjacent_velocity,
                        )
                    )
                adjacent_energy = np.nanmax(np.vstack(adjacent), axis=0)
                rows.append(
                    {
                        "wavefield_seed": seed,
                        "scenario": scenario,
                        "method": method,
                        "group_image_correlation_to_active": _image_correlation(
                            result.group_image.amplitude,
                            active.group_image.amplitude,
                        ),
                        "phase_image_correlation_to_active": _image_correlation(
                            result.phase_image.amplitude,
                            active.phase_image.amplitude,
                        ),
                        "mean_group_theory_line_energy": float(np.nanmean(group_energy)),
                        "median_phase_correct_branch_advantage": float(
                            np.nanmedian(correct_phase - adjacent_energy)
                        ),
                    }
                )
    return rows


def _paired_rows(run_root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_root / "gate2_sparse_event_metrics.csv")
    main = metrics[
        (metrics.control_replica == -1)
        & (metrics.branch == "symmetric")
        & (metrics.status == "complete")
    ].copy()
    events = pd.read_csv(run_root / "gate2_sparse_event_effective_events.csv")
    output = []
    for (seed, scenario), case in main.groupby(["wavefield_seed", "scenario"]):
        linear = case[case.method == "linear"].iloc[0]
        for _, row in case.iterrows():
            method = row.method
            if method in {"linear", "paired_clean_counterfactual"}:
                continue
            controls = metrics[
                (metrics.wavefield_seed == seed)
                & (metrics.scenario == scenario)
                & (metrics.branch == "symmetric")
                & (metrics.method == f"{method}_permuted")
            ]
            selected_events = events[
                (events.wavefield_seed == seed)
                & (events.scenario == scenario)
                & (events.method == method)
            ]
            source_means = selected_events.groupby("source_group").window_weight.mean()
            sector_ratio = (
                float(source_means.get(1, np.nan) / source_means.get(0, np.nan))
                if source_means.get(0, 0.0) > 0.0
                else float("nan")
            )
            output.append(
                {
                    "wavefield_seed": seed,
                    "scenario": scenario,
                    "method": method,
                    "effective_window_count": row.effective_window_count,
                    "waveform_cc": row.waveform_correlation_no_shift,
                    "delta_cc_vs_linear": row.waveform_correlation_no_shift
                    - linear.waveform_correlation_no_shift,
                    "delta_l2_vs_linear": row.normalized_l2_unit_norm
                    - linear.normalized_l2_unit_norm,
                    "delta_phase_rms_vs_linear": row.phase_rms_rad
                    - linear.phase_rms_rad,
                    "delta_prearrival_vs_linear": row.prearrival_ghost_energy_fraction
                    - linear.prearrival_ghost_energy_fraction,
                    "permuted_median_cc": controls.waveform_correlation_no_shift.median(),
                    "delta_cc_vs_permuted_median": row.waveform_correlation_no_shift
                    - controls.waveform_correlation_no_shift.median(),
                    "sector_to_background_event_weight_ratio": sector_ratio,
                }
            )
    return pd.DataFrame(output)


def _paired_figure(paired: pd.DataFrame, output: Path) -> None:
    specs = (
        ("delta_cc_vs_linear", "Δ waveform CC vs linear", 1.0),
        ("delta_l2_vs_linear", "Δ unit-norm L2 vs linear", -1.0),
        ("delta_phase_rms_vs_linear", "Δ phase RMS vs linear (rad)", -1.0),
        ("delta_cc_vs_permuted_median", "Δ CC vs matched permutation", 1.0),
    )
    scenarios = ("uniform", "far75", "near12", "mirror168")
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), constrained_layout=True)
    rng = np.random.default_rng(1234)
    for axis, (metric, ylabel, favorable_sign) in zip(axes.reshape(-1), specs):
        for method_index, method in enumerate(DEPLOYABLE_METHODS):
            selected = paired[paired.method == method]
            for scenario_index, scenario in enumerate(scenarios):
                values = selected[selected.scenario == scenario][metric].to_numpy()
                x = scenario_index + (method_index - 1.5) * 0.10
                jitter = rng.uniform(-0.018, 0.018, values.size)
                axis.scatter(
                    np.full(values.size, x) + jitter,
                    values,
                    s=18,
                    color=COLORS[method],
                    alpha=0.75,
                    label=METHOD_LABELS[method]
                    if scenario_index == 0
                    else None,
                )
                if values.size:
                    axis.plot(
                        [x - 0.035, x + 0.035],
                        [np.median(values), np.median(values)],
                        color=COLORS[method],
                        linewidth=2.0,
                    )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        if favorable_sign > 0:
            axis.axhspan(0.0, axis.get_ylim()[1], color="#009E73", alpha=0.05)
        else:
            axis.axhspan(axis.get_ylim()[0], 0.0, color="#009E73", alpha=0.05)
        axis.set_xticks(range(len(scenarios)), scenarios)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=2)
    figure.suptitle(
        "Gate 2 v5 blind paired evidence | points are all frozen seeds; bars are medians",
        fontsize=10.0,
    )
    _save(figure, output / "gate2_sparse_event_blind_paired_metrics")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("output_python/gate2_sparse_event_blind"),
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_directory.resolve()
    output = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else run_root / "summary"
    )
    output.mkdir(parents=True, exist_ok=True)

    paired = _paired_rows(run_root)
    paired.to_csv(output / "gate2_sparse_event_paired_metrics.csv", index=False)
    ftan = pd.DataFrame(_ftan_rows(run_root))
    ftan.to_csv(output / "gate2_sparse_event_ftan_metrics.csv", index=False)
    merged = paired.merge(ftan, on=["wavefield_seed", "scenario", "method"], how="left")
    linear_ftan = ftan[ftan.method == "linear"].drop(columns="method").rename(
        columns={
            "group_image_correlation_to_active": "linear_group_image_correlation_to_active",
            "phase_image_correlation_to_active": "linear_phase_image_correlation_to_active",
            "mean_group_theory_line_energy": "linear_mean_group_theory_line_energy",
            "median_phase_correct_branch_advantage": "linear_median_phase_correct_branch_advantage",
        }
    )
    merged = merged.merge(
        linear_ftan,
        on=["wavefield_seed", "scenario"],
        how="left",
    )
    merged["delta_group_image_correlation_vs_linear"] = (
        merged.group_image_correlation_to_active
        - merged.linear_group_image_correlation_to_active
    )
    merged["delta_phase_image_correlation_vs_linear"] = (
        merged.phase_image_correlation_to_active
        - merged.linear_phase_image_correlation_to_active
    )
    merged["delta_group_theory_line_energy_vs_linear"] = (
        merged.mean_group_theory_line_energy
        - merged.linear_mean_group_theory_line_energy
    )
    merged["delta_phase_correct_branch_advantage_vs_linear"] = (
        merged.median_phase_correct_branch_advantage
        - merged.linear_median_phase_correct_branch_advantage
    )
    merged.to_csv(output / "gate2_sparse_event_complete_evidence.csv", index=False)

    aggregate_rows = []
    for (scenario, method), group in merged.groupby(["scenario", "method"]):
        aggregate_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "seed_count": int(group.wavefield_seed.nunique()),
                "median_delta_cc_vs_linear": float(group.delta_cc_vs_linear.median()),
                "improved_cc_seed_count": int(np.count_nonzero(group.delta_cc_vs_linear > 0)),
                "median_delta_l2_vs_linear": float(group.delta_l2_vs_linear.median()),
                "improved_l2_seed_count": int(np.count_nonzero(group.delta_l2_vs_linear < 0)),
                "median_delta_phase_vs_linear": float(group.delta_phase_rms_vs_linear.median()),
                "improved_phase_seed_count": int(
                    np.count_nonzero(group.delta_phase_rms_vs_linear < 0)
                ),
                "median_delta_prearrival_vs_linear": float(
                    group.delta_prearrival_vs_linear.median()
                ),
                "improved_prearrival_seed_count": int(
                    np.count_nonzero(group.delta_prearrival_vs_linear < 0)
                ),
                "median_delta_cc_vs_permuted": float(
                    group.delta_cc_vs_permuted_median.median()
                ),
                "beat_permuted_seed_count": int(
                    np.count_nonzero(group.delta_cc_vs_permuted_median > 0)
                ),
                "median_sector_to_background_event_weight_ratio": float(
                    group.sector_to_background_event_weight_ratio.median()
                ),
                "median_group_image_correlation_to_active": float(
                    group.group_image_correlation_to_active.median()
                ),
                "median_phase_image_correlation_to_active": float(
                    group.phase_image_correlation_to_active.median()
                ),
                "median_phase_correct_branch_advantage": float(
                    group.median_phase_correct_branch_advantage.median()
                ),
                "median_delta_group_image_correlation_vs_linear": float(
                    group.delta_group_image_correlation_vs_linear.median()
                ),
                "improved_group_image_seed_count": int(
                    np.count_nonzero(group.delta_group_image_correlation_vs_linear > 0)
                ),
                "median_delta_phase_image_correlation_vs_linear": float(
                    group.delta_phase_image_correlation_vs_linear.median()
                ),
                "improved_phase_image_seed_count": int(
                    np.count_nonzero(group.delta_phase_image_correlation_vs_linear > 0)
                ),
                "median_delta_group_theory_line_energy_vs_linear": float(
                    group.delta_group_theory_line_energy_vs_linear.median()
                ),
                "median_delta_phase_correct_branch_advantage_vs_linear": float(
                    group.delta_phase_correct_branch_advantage_vs_linear.median()
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / "gate2_sparse_event_aggregate.csv", index=False)
    _style()
    _paired_figure(paired[paired.method.isin(DEPLOYABLE_METHODS)], output)
    summary = {
        "plan_version": "v5.0",
        "run_directory": str(run_root),
        "wavefield_seeds": sorted(int(value) for value in paired.wavefield_seed.unique()),
        "primary_success_rule": (
            "A method must improve CC and L2, reduce phase RMS and prearrival, beat its "
            "matched permutation, avoid uniform degradation, and show directionally "
            "consistent blind-seed behavior. No single metric is sufficient."
        ),
        "aggregate_csv": str(output / "gate2_sparse_event_aggregate.csv"),
        "complete_evidence_csv": str(output / "gate2_sparse_event_complete_evidence.csv"),
    }
    (output / "gate2_sparse_event_evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
