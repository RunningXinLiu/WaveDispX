#!/usr/bin/env python3
"""Quantify the pre-registered v5.2 truth-only branch-SPZ oracle gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wavedispx.processing import pair_ftan

from plot_gate2_sparse_event_qc import _theory
from run_gate2_pair_selection_experiment import (
    EVALUATION_BAND_HZ,
    PAIR_DISTANCE_M,
    VELOCITY_RANGE_MPS,
)
from summarize_gate2_sparse_event_results import (
    _image_correlation,
    _line_energy,
    _phase_adjacent_velocity,
)


METHOD = "oracle_branch_spz"
SCENARIOS = ("uniform", "far75", "near12", "mirror168")
BIAS_SCENARIOS = ("far75", "near12", "mirror168")


def _one_ftan(
    trace: np.ndarray,
    dt_s: float,
    *,
    input_kind: str,
):
    return pair_ftan(
        trace,
        dt_s=dt_s,
        distance_m=PAIR_DISTANCE_M,
        frequency_range_hz=EVALUATION_BAND_HZ,
        group_velocity_range_mps=VELOCITY_RANGE_MPS,
        phase_velocity_range_mps=VELOCITY_RANGE_MPS,
        period_count=121,
        velocity_step_mps=2.0,
        gaussian_alpha=120.0,
        phase_method="frequency_domain",
        minimum_snr=0.0,
        input_kind=input_kind,
    )


def _ftan_rows(run_root: Path, input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed_dir in sorted(run_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        for scenario in SCENARIOS:
            case_dir = seed_dir / scenario
            archive = case_dir / "gate2_branch_spz_oracle.npz"
            if not archive.exists():
                continue
            with np.load(archive) as npz:
                lag = np.asarray(npz["lag_time_s"], dtype=np.float64)
                dt = float(np.median(np.diff(lag)))
                active_time = np.asarray(npz["active_reference_time_s"], dtype=np.float64)
                results = {
                    "active_reference": _one_ftan(
                        npz["active_reference_zz"],
                        float(np.median(np.diff(active_time))),
                        input_kind="egf",
                    )
                }
                for method in ("linear", "paired_clean_counterfactual", METHOD):
                    key = f"stack__{method}__symmetric"
                    if key in npz.files:
                        results[method] = _one_ftan(
                            npz[key],
                            dt,
                            input_kind="correlation",
                        )
            theory_case = input_root / f"seed_{seed}" / scenario
            theory_period, theory_group, theory_phase = _theory(theory_case, seed)
            active = results["active_reference"]
            for method, result in results.items():
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
                adjacent = []
                for cycle in (-1, 1):
                    adjacent.append(
                        _line_energy(
                            result.phase_image.amplitude,
                            result.phase_image.velocity_mps,
                            _phase_adjacent_velocity(
                                phase_target,
                                periods,
                                cycle,
                            ),
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
                        "mean_group_theory_line_energy": float(
                            np.nanmean(group_energy)
                        ),
                        "median_phase_correct_branch_advantage": float(
                            np.nanmedian(correct_phase - adjacent_energy)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _waveform_rows(run_root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_root / "gate2_branch_spz_metrics.csv")
    primary = metrics[
        (metrics.control_replica == -1) & (metrics.status == "complete")
    ]
    rows: list[dict[str, object]] = []
    for (seed, scenario), case in primary.groupby(["wavefield_seed", "scenario"]):
        linear = case[case.method == "linear"].set_index("branch")
        controls = metrics[
            (metrics.wavefield_seed == seed)
            & (metrics.scenario == scenario)
            & (metrics.method == f"{METHOD}_permuted")
        ]
        for method in case.method.unique():
            selected = case[case.method == method].set_index("branch")
            symmetric = selected.loc["symmetric"]
            linear_symmetric = linear.loc["symmetric"]
            rows.append(
                {
                    "wavefield_seed": int(seed),
                    "scenario": scenario,
                    "method": method,
                    "waveform_cc": symmetric.waveform_correlation_no_shift,
                    "normalized_l2": symmetric.normalized_l2_unit_norm,
                    "phase_rms_rad": symmetric.phase_rms_rad,
                    "prearrival_fraction": symmetric.prearrival_ghost_energy_fraction,
                    "positive_branch_cc": selected.loc[
                        "positive", "waveform_correlation_no_shift"
                    ],
                    "negative_branch_cc": selected.loc[
                        "negative", "waveform_correlation_no_shift"
                    ],
                    "delta_cc_vs_linear": symmetric.waveform_correlation_no_shift
                    - linear_symmetric.waveform_correlation_no_shift,
                    "delta_l2_vs_linear": symmetric.normalized_l2_unit_norm
                    - linear_symmetric.normalized_l2_unit_norm,
                    "delta_phase_vs_linear": symmetric.phase_rms_rad
                    - linear_symmetric.phase_rms_rad,
                    "delta_prearrival_vs_linear": symmetric.prearrival_ghost_energy_fraction
                    - linear_symmetric.prearrival_ghost_energy_fraction,
                    "delta_positive_cc_vs_linear": selected.loc[
                        "positive", "waveform_correlation_no_shift"
                    ]
                    - linear.loc["positive", "waveform_correlation_no_shift"],
                    "delta_negative_cc_vs_linear": selected.loc[
                        "negative", "waveform_correlation_no_shift"
                    ]
                    - linear.loc["negative", "waveform_correlation_no_shift"],
                    "delta_cc_vs_permuted": symmetric.waveform_correlation_no_shift
                    - controls[
                        controls.branch == "symmetric"
                    ].waveform_correlation_no_shift.median(),
                    "delta_l2_vs_permuted": symmetric.normalized_l2_unit_norm
                    - controls[
                        controls.branch == "symmetric"
                    ].normalized_l2_unit_norm.median(),
                    "delta_phase_vs_permuted": symmetric.phase_rms_rad
                    - controls[controls.branch == "symmetric"].phase_rms_rad.median(),
                    "delta_prearrival_vs_permuted": symmetric.prearrival_ghost_energy_fraction
                    - controls[
                        controls.branch == "symmetric"
                    ].prearrival_ghost_energy_fraction.median(),
                }
            )
    return pd.DataFrame(rows)


def _complete(waveform: pd.DataFrame, ftan: pd.DataFrame) -> pd.DataFrame:
    complete = waveform.merge(
        ftan,
        on=["wavefield_seed", "scenario", "method"],
        how="left",
    )
    linear = complete[complete.method == "linear"][
        [
            "wavefield_seed",
            "scenario",
            "group_image_correlation_to_active",
            "phase_image_correlation_to_active",
            "mean_group_theory_line_energy",
            "median_phase_correct_branch_advantage",
        ]
    ].rename(
        columns={
            name: f"linear_{name}"
            for name in (
                "group_image_correlation_to_active",
                "phase_image_correlation_to_active",
                "mean_group_theory_line_energy",
                "median_phase_correct_branch_advantage",
            )
        }
    )
    complete = complete.merge(linear, on=["wavefield_seed", "scenario"], how="left")
    for name in (
        "group_image_correlation_to_active",
        "phase_image_correlation_to_active",
        "mean_group_theory_line_energy",
        "median_phase_correct_branch_advantage",
    ):
        complete[f"delta_{name}_vs_linear"] = (
            complete[name] - complete[f"linear_{name}"]
        )
    return complete


def _gate(complete: pd.DataFrame) -> dict[str, object]:
    selected = complete[complete.method == METHOD]
    checks = []
    for scenario in SCENARIOS:
        row = selected[selected.scenario == scenario].iloc[0]
        waveform_pass = bool(
            row.delta_cc_vs_linear > 0.0
            and row.delta_l2_vs_linear < 0.0
            and row.delta_phase_vs_linear < 0.0
            and row.delta_prearrival_vs_linear <= 0.0
            and row.delta_positive_cc_vs_linear > 0.0
            and row.delta_negative_cc_vs_linear > 0.0
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
                "waveform_and_both_branches_passed": waveform_pass,
                "matched_permutation_passed": permutation_pass,
                "ftan_passed": ftan_pass,
                **{
                    name: float(row[name])
                    for name in (
                        "delta_cc_vs_linear",
                        "delta_l2_vs_linear",
                        "delta_phase_vs_linear",
                        "delta_prearrival_vs_linear",
                        "delta_positive_cc_vs_linear",
                        "delta_negative_cc_vs_linear",
                        "delta_group_image_correlation_to_active_vs_linear",
                        "delta_phase_image_correlation_to_active_vs_linear",
                        "delta_mean_group_theory_line_energy_vs_linear",
                        "delta_median_phase_correct_branch_advantage_vs_linear",
                    )
                },
            }
        )
    biased = [item for item in checks if item["scenario"] in BIAS_SCENARIOS]
    uniform = next(item for item in checks if item["scenario"] == "uniform")
    passed = bool(
        all(
            item["waveform_and_both_branches_passed"]
            and item["matched_permutation_passed"]
            and item["ftan_passed"]
            for item in biased
        )
        and uniform["waveform_and_both_branches_passed"]
        and uniform["ftan_passed"]
    )
    return {"method": METHOD, "passed": passed, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("output_python/gate2_branch_spz_development"),
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("output_python/gate2_source_aware_development"),
    )
    args = parser.parse_args()
    run_root = args.run_directory.resolve()
    input_root = args.input_directory.resolve()
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    waveform = _waveform_rows(run_root)
    ftan = _ftan_rows(run_root, input_root)
    complete = _complete(waveform, ftan)
    waveform.to_csv(summary_dir / "gate2_branch_spz_waveform_metrics.csv", index=False)
    ftan.to_csv(summary_dir / "gate2_branch_spz_ftan_metrics.csv", index=False)
    complete.to_csv(summary_dir / "gate2_branch_spz_complete_evidence.csv", index=False)
    gate = _gate(complete)
    summary = {
        "experiment": "Gate2_v5.2_truth_only_branch_spz_complete_gate",
        "plan_version": "v5.2",
        "wavefield_seeds": sorted(int(x) for x in complete.wavefield_seed.unique()),
        "oracle_gate": gate,
        "failure_classification": (
            None if gate["passed"] else "whole_window_branch_spz_selection_is_insufficient"
        ),
        "beam_method_authorized": bool(gate["passed"]),
        "blind_authorized": False,
        "automatic_ftan_picks_used": False,
        "complete_evidence_csv": str(
            summary_dir / "gate2_branch_spz_complete_evidence.csv"
        ),
    }
    summary_path = summary_dir / "gate2_branch_spz_evidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    runner_summary_path = run_root / "gate2_branch_spz_summary.json"
    runner_summary = json.loads(runner_summary_path.read_text())
    runner_summary["ftan_gate_status"] = "passed" if gate["passed"] else "failed"
    runner_summary["oracle_gate_passed"] = bool(gate["passed"])
    runner_summary["beam_method_authorized"] = bool(gate["passed"])
    runner_summary["failure_classification"] = summary["failure_classification"]
    runner_summary["evidence_summary"] = str(summary_path)
    runner_summary_path.write_text(json.dumps(runner_summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
