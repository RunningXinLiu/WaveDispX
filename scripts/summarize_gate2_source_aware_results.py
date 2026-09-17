#!/usr/bin/env python3
"""Quantify v5.1 waveform and FTAN evidence without automatic ridge picking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_gate2_source_aware_qc import _ftan
from plot_gate2_sparse_event_qc import _theory
from summarize_gate2_sparse_event_results import (
    _image_correlation,
    _line_energy,
    _phase_adjacent_velocity,
)


PRIMARY_METHODS = ("beam_uniformity", "oracle_event_uniformity")
SCENARIOS = ("uniform", "far75", "near12", "mirror168")
BIAS_SCENARIOS = ("far75", "near12", "mirror168")


def _ftan_rows(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed_dir in sorted(run_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        for scenario in SCENARIOS:
            case_dir = seed_dir / scenario
            archive = case_dir / "gate2_source_aware_beamforming.npz"
            if not archive.exists():
                continue
            with np.load(archive) as npz:
                results = _ftan(npz)
            theory_period, theory_group, theory_phase = _theory(case_dir, seed)
            active = results["active_reference"]
            for method, result in results.items():
                if result is None:
                    continue
                periods = result.group_image.period_s
                group_target = np.interp(periods, theory_period, theory_group)
                phase_target = np.interp(periods, theory_period, theory_phase)
                group_energy = _line_energy(
                    result.group_image.amplitude,
                    result.group_image.velocity_mps,
                    group_target,
                )
                correct_phase = _line_energy(
                    result.phase_image.amplitude,
                    result.phase_image.velocity_mps,
                    phase_target,
                )
                adjacent_phase = []
                for cycle in (-1, 1):
                    adjacent_velocity = _phase_adjacent_velocity(
                        phase_target,
                        periods,
                        cycle,
                    )
                    adjacent_phase.append(
                        _line_energy(
                            result.phase_image.amplitude,
                            result.phase_image.velocity_mps,
                            adjacent_velocity,
                        )
                    )
                adjacent_energy = np.nanmax(np.vstack(adjacent_phase), axis=0)
                advantage = correct_phase - adjacent_energy
                valid = np.isfinite(advantage)
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
                        "mean_group_theory_line_energy": float(
                            np.nanmean(group_energy)
                        ),
                        "median_phase_correct_branch_advantage": float(
                            np.nanmedian(advantage)
                        ),
                        "phase_correct_branch_win_fraction": float(
                            np.count_nonzero(advantage[valid] > 0.0)
                            / max(np.count_nonzero(valid), 1)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _waveform_rows(run_root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_root / "gate2_source_aware_metrics.csv")
    primary = metrics[
        (metrics.control_replica == -1)
        & (metrics.branch == "symmetric")
        & (metrics.status == "complete")
    ].copy()
    output: list[dict[str, object]] = []
    for (seed, scenario), case in primary.groupby(["wavefield_seed", "scenario"]):
        linear = case[case.method == "linear"].iloc[0]
        for _, row in case.iterrows():
            method = str(row.method)
            controls = metrics[
                (metrics.wavefield_seed == seed)
                & (metrics.scenario == scenario)
                & (metrics.branch == "symmetric")
                & (metrics.method == f"{method}_permuted")
            ]
            output.append(
                {
                    "wavefield_seed": int(seed),
                    "scenario": scenario,
                    "method": method,
                    "effective_window_count": row.effective_window_count,
                    "waveform_cc": row.waveform_correlation_no_shift,
                    "normalized_l2": row.normalized_l2_unit_norm,
                    "phase_rms_rad": row.phase_rms_rad,
                    "prearrival_fraction": row.prearrival_ghost_energy_fraction,
                    "delta_cc_vs_linear": row.waveform_correlation_no_shift
                    - linear.waveform_correlation_no_shift,
                    "delta_l2_vs_linear": row.normalized_l2_unit_norm
                    - linear.normalized_l2_unit_norm,
                    "delta_phase_vs_linear": row.phase_rms_rad
                    - linear.phase_rms_rad,
                    "delta_prearrival_vs_linear": row.prearrival_ghost_energy_fraction
                    - linear.prearrival_ghost_energy_fraction,
                    "permuted_median_cc": controls.waveform_correlation_no_shift.median(),
                    "permuted_median_l2": controls.normalized_l2_unit_norm.median(),
                    "permuted_median_phase": controls.phase_rms_rad.median(),
                    "permuted_median_prearrival": controls.prearrival_ghost_energy_fraction.median(),
                    "delta_cc_vs_permuted": row.waveform_correlation_no_shift
                    - controls.waveform_correlation_no_shift.median(),
                    "delta_l2_vs_permuted": row.normalized_l2_unit_norm
                    - controls.normalized_l2_unit_norm.median(),
                    "delta_phase_vs_permuted": row.phase_rms_rad
                    - controls.phase_rms_rad.median(),
                    "delta_prearrival_vs_permuted": row.prearrival_ghost_energy_fraction
                    - controls.prearrival_ghost_energy_fraction.median(),
                }
            )
    return pd.DataFrame(output)


def _add_ftan_deltas(complete: pd.DataFrame) -> pd.DataFrame:
    linear = complete[complete.method == "linear"][
        [
            "wavefield_seed",
            "scenario",
            "group_image_correlation_to_active",
            "phase_image_correlation_to_active",
            "mean_group_theory_line_energy",
            "median_phase_correct_branch_advantage",
            "phase_correct_branch_win_fraction",
        ]
    ].rename(
        columns={
            name: f"linear_{name}"
            for name in (
                "group_image_correlation_to_active",
                "phase_image_correlation_to_active",
                "mean_group_theory_line_energy",
                "median_phase_correct_branch_advantage",
                "phase_correct_branch_win_fraction",
            )
        }
    )
    complete = complete.merge(
        linear,
        on=["wavefield_seed", "scenario"],
        how="left",
    )
    for name in (
        "group_image_correlation_to_active",
        "phase_image_correlation_to_active",
        "mean_group_theory_line_energy",
        "median_phase_correct_branch_advantage",
        "phase_correct_branch_win_fraction",
    ):
        complete[f"delta_{name}_vs_linear"] = (
            complete[name] - complete[f"linear_{name}"]
        )
    return complete


def _gate(complete: pd.DataFrame, method: str) -> dict[str, object]:
    selected = complete[complete.method == method]
    checks: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        row = selected[selected.scenario == scenario].iloc[0]
        waveform_pass = bool(
            row.delta_cc_vs_linear > 0.0
            and row.delta_l2_vs_linear < 0.0
            and row.delta_phase_vs_linear < 0.0
            and row.delta_prearrival_vs_linear <= 0.0
        )
        permutation_pass = bool(
            row.delta_cc_vs_permuted > 0.0
            and row.delta_l2_vs_permuted < 0.0
            and row.delta_phase_vs_permuted < 0.0
            and row.delta_prearrival_vs_permuted <= 0.0
        )
        ftan_pass = bool(
            row.delta_group_image_correlation_to_active_vs_linear > 0.0
            and row.delta_phase_image_correlation_to_active_vs_linear > 0.0
            and row.delta_mean_group_theory_line_energy_vs_linear >= 0.0
            and row.delta_median_phase_correct_branch_advantage_vs_linear >= 0.0
        )
        checks.append(
            {
                "scenario": scenario,
                "waveform_joint_pass": waveform_pass,
                "matched_permutation_joint_pass": permutation_pass,
                "ftan_joint_pass": ftan_pass,
                "delta_cc_vs_linear": float(row.delta_cc_vs_linear),
                "delta_l2_vs_linear": float(row.delta_l2_vs_linear),
                "delta_phase_vs_linear": float(row.delta_phase_vs_linear),
                "delta_prearrival_vs_linear": float(
                    row.delta_prearrival_vs_linear
                ),
                "delta_group_image_correlation_vs_linear": float(
                    row.delta_group_image_correlation_to_active_vs_linear
                ),
                "delta_phase_image_correlation_vs_linear": float(
                    row.delta_phase_image_correlation_to_active_vs_linear
                ),
                "delta_group_theory_line_energy_vs_linear": float(
                    row.delta_mean_group_theory_line_energy_vs_linear
                ),
                "delta_phase_correct_branch_advantage_vs_linear": float(
                    row.delta_median_phase_correct_branch_advantage_vs_linear
                ),
            }
        )
    biased = [item for item in checks if item["scenario"] in BIAS_SCENARIOS]
    uniform = next(item for item in checks if item["scenario"] == "uniform")
    passed = bool(
        all(
            item["waveform_joint_pass"]
            and item["matched_permutation_joint_pass"]
            and item["ftan_joint_pass"]
            for item in biased
        )
        and uniform["waveform_joint_pass"]
        and uniform["ftan_joint_pass"]
    )
    return {
        "method": method,
        "passed": passed,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("output_python/gate2_source_aware_development"),
    )
    args = parser.parse_args()
    run_root = args.run_directory.resolve()
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    waveform = _waveform_rows(run_root)
    ftan = _ftan_rows(run_root)
    complete = _add_ftan_deltas(
        waveform.merge(
            ftan,
            on=["wavefield_seed", "scenario", "method"],
            how="left",
        )
    )
    waveform.to_csv(summary_dir / "gate2_source_aware_waveform_metrics.csv", index=False)
    ftan.to_csv(summary_dir / "gate2_source_aware_ftan_metrics.csv", index=False)
    complete.to_csv(
        summary_dir / "gate2_source_aware_complete_evidence.csv",
        index=False,
    )

    beam_gate = _gate(complete, "beam_uniformity")
    oracle_gate = _gate(complete, "oracle_event_uniformity")
    if oracle_gate["passed"]:
        failure_class = "beam_feature_or_array_is_insufficient"
    else:
        failure_class = "common_window_angular_uniformity_target_is_insufficient"
    summary = {
        "experiment": "Gate2_v5.1_source_aware_independent_ftan_audit",
        "plan_version": "v5.1",
        "wavefield_seeds": sorted(
            int(value) for value in complete.wavefield_seed.unique()
        ),
        "beam_uniformity_gate": beam_gate,
        "oracle_event_uniformity_gate": oracle_gate,
        "failure_classification": failure_class,
        "blind_authorized": False,
        "blind_reason": (
            "Frozen blind seeds 5601-5605 require direction, waveform, matched-"
            "permutation, and independent FTAN gates to pass. The joint EGF gate failed."
        ),
        "automatic_ftan_picks_used": False,
        "complete_evidence_csv": str(
            summary_dir / "gate2_source_aware_complete_evidence.csv"
        ),
    }
    summary_path = summary_dir / "gate2_source_aware_evidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    runner_summary_path = run_root / "gate2_source_aware_summary.json"
    runner_summary = json.loads(runner_summary_path.read_text())
    runner_summary["ftan_gate_status"] = (
        "passed" if beam_gate["passed"] else "failed"
    )
    runner_summary["ftan_evidence_summary"] = str(summary_path)
    runner_summary["failure_classification"] = failure_class
    runner_summary["blind_authorized"] = False
    runner_summary_path.write_text(json.dumps(runner_summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
