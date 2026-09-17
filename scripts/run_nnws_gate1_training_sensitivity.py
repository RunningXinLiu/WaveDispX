#!/usr/bin/env python3
"""Retrain upstream NNWS for Gate 1 and audit initialization/prior sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_nnws_gate1_upstream_reproduction import (
    DELTA_S,
    RMSR_THRESHOLD,
    SCENARIOS,
    TAU_S,
    TMAX_S,
    TMIN_S,
    UPSTREAM_COMMIT,
    _git_head,
    _load_upstream_api,
    _method_metrics,
    _save_figure,
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)) or any(seed < 0 for seed in result):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _train_once(
    dataset: np.ndarray,
    *,
    seed: int,
    velocity_scale: float,
    api: dict[str, Any],
    truth: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    from nnws.training import TrainingConfig, apply_stacking_network, train

    tmin = TMIN_S / velocity_scale
    tmax = TMAX_S / velocity_scale
    started = time.perf_counter()
    result = train(
        dataset,
        tau=TAU_S,
        tmin=tmin,
        tmax=tmax,
        delta=DELTA_S,
        config=TrainingConfig(epochs=200, seed=seed, device="cpu"),
    )
    stack, weights = apply_stacking_network(result.model, dataset)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(stack)):
        raise RuntimeError("upstream training produced non-finite stack/weights")
    metric, normalized, _, _ = _method_metrics(
        "nnws_retrained",
        stack,
        truth,
        api,
        weight=weights,
    )
    metric.update(
        {
            "seed": seed,
            "velocity_scale": velocity_scale,
            "training_tmin_s": tmin,
            "training_tmax_s": tmax,
            "training_reference_reported_snr": float(
                api["snr"](normalized, TAU_S, tmin, tmax, DELTA_S)
            ),
            "training_final_total_loss": result.history["total_loss"][-1],
            "training_final_snr_loss": result.history["snr_loss"][-1],
            "training_final_symmetry_loss": result.history["sym_loss"][-1],
            "training_final_std_loss": result.history["std_loss"][-1],
            "elapsed_seconds": time.perf_counter() - started,
            "weight_min": float(np.min(weights)),
            "weight_median": float(np.median(weights)),
            "weight_max": float(np.max(weights)),
        }
    )
    return metric, weights


def _rmsr_prior_sensitivity(
    dataset: np.ndarray,
    *,
    velocity_scale: float,
    api: dict[str, Any],
    truth: np.ndarray,
) -> dict[str, Any]:
    tmin = TMIN_S / velocity_scale
    tmax = TMAX_S / velocity_scale
    _, stack, selected = api["rmsr"](
        dataset,
        TAU_S,
        tmin,
        tmax,
        DELTA_S,
        threshold=RMSR_THRESHOLD,
    )
    metric, normalized, _, _ = _method_metrics(
        "rmsr_g5",
        stack,
        truth,
        api,
        selected=selected,
    )
    metric.update(
        {
            "seed": None,
            "velocity_scale": velocity_scale,
            "training_tmin_s": tmin,
            "training_tmax_s": tmax,
            "training_reference_reported_snr": float(
                api["snr"](normalized, TAU_S, tmin, tmax, DELTA_S)
            ),
            "training_final_total_loss": None,
            "training_final_snr_loss": None,
            "training_final_symmetry_loss": None,
            "training_final_std_loss": None,
            "elapsed_seconds": None,
            "weight_min": None,
            "weight_median": None,
            "weight_max": None,
        }
    )
    return metric


def _plot_initialization(rows: list[dict[str, Any]], output_dir: Path) -> None:
    nominal = [
        row
        for row in rows
        if row["method"] == "nnws_retrained" and row["velocity_scale"] == 1.0
    ]
    metrics = (
        ("waveform_cc_full", "Full-window waveform CC", True),
        ("reported_snr", "Reported SNR", True),
        ("symmetry_rms_ratio", "Antisymmetry / symmetry RMS", False),
        ("group_velocity_mae_percent", "Group-velocity MAE (%)", False),
    )
    figure, axes = plt.subplots(1, 4, figsize=(14.2, 3.8), constrained_layout=True)
    scenario_names = [scenario.name for scenario in SCENARIOS]
    labels = [scenario.label for scenario in SCENARIOS]
    for axis, (key, ylabel, higher_is_better) in zip(axes, metrics):
        values = [
            [float(row[key]) for row in nominal if row["scenario"] == name]
            for name in scenario_names
        ]
        axis.boxplot(values, tick_labels=labels, showmeans=True)
        for x_position, scenario_values in enumerate(values, start=1):
            axis.scatter(
                np.full(len(scenario_values), x_position),
                scenario_values,
                s=15,
                color="#e31a1c",
                alpha=0.75,
                zorder=3,
            )
        axis.set_ylabel(ylabel + (" ↑" if higher_is_better else " ↓"))
        axis.tick_params(axis="x", rotation=35, labelsize=7)
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle("Gate 1 exact-upstream NNWS retraining: five initialization seeds")
    _save_figure(figure, output_dir, "gate1_nnws_initialization_sensitivity")


def _plot_prior(rows: list[dict[str, Any]], output_dir: Path) -> None:
    selected = [row for row in rows if row["scenario"] == "test_azi40"]
    metrics = (
        ("waveform_cc_full", "Waveform CC", True),
        ("reported_snr", "Reported SNR (fixed evaluation window)", True),
        ("symmetry_rms_ratio", "Antisymmetry / symmetry RMS", False),
        ("group_velocity_mae_percent", "Group-velocity MAE (%)", False),
    )
    figure, axes = plt.subplots(1, 4, figsize=(14.2, 3.8), constrained_layout=True)
    scales = (0.9, 1.0, 1.1)
    for axis, (key, ylabel, higher_is_better) in zip(axes, metrics):
        for method, color, marker in (
            ("rmsr_g5", "#1f78b4", "s"),
            ("nnws_retrained", "#e31a1c", "o"),
        ):
            center = []
            low = []
            high = []
            for scale in scales:
                values = np.asarray(
                    [
                        float(row[key])
                        for row in selected
                        if row["method"] == method and row["velocity_scale"] == scale
                    ]
                )
                center.append(float(np.median(values)))
                low.append(float(np.percentile(values, 25)))
                high.append(float(np.percentile(values, 75)))
            center_array = np.asarray(center)
            axis.plot(scales, center_array, color=color, marker=marker, label=method)
            axis.fill_between(scales, low, high, color=color, alpha=0.18)
        axis.set_xlabel("Reference velocity / true velocity")
        axis.set_ylabel(ylabel + (" ↑" if higher_is_better else " ↓"))
        axis.set_xticks(scales)
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Gate 1 test_azi40: ±10% reference-velocity sensitivity")
    _save_figure(figure, output_dir, "gate1_reference_velocity_sensitivity")


def run(upstream_root: Path, output_dir: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    if _git_head(upstream_root) != UPSTREAM_COMMIT:
        raise RuntimeError(f"NNWS upstream must be pinned to {UPSTREAM_COMMIT}")
    output_dir.mkdir(parents=True, exist_ok=True)
    api = _load_upstream_api(upstream_root)
    rows: list[dict[str, Any]] = []
    saved_weights: dict[str, np.ndarray] = {}
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    total = len(SCENARIOS) * len(seeds) + 2 * len(seeds)
    completed = 0
    for scenario in SCENARIOS:
        dataset = np.load(
            upstream_root / "syn_datasets" / f"dataset_{scenario.name}.npy"
        )
        truth = np.asarray(api["normalize"](api["truth"](wavelet=scenario.truth_wavelet)))
        for seed in seeds:
            completed += 1
            print(
                f"[{completed}/{total}] {scenario.name}, seed={seed}, velocity=1.0",
                flush=True,
            )
            try:
                metric, weights = _train_once(
                    dataset,
                    seed=seed,
                    velocity_scale=1.0,
                    api=api,
                    truth=truth,
                )
                metric["scenario"] = scenario.name
                rows.append(metric)
                saved_weights[f"{scenario.name}_v1.0_seed{seed}"] = weights
            except Exception as exc:  # keep failed seeds visible in the audit
                failures.append(
                    {
                        "scenario": scenario.name,
                        "seed": seed,
                        "velocity_scale": 1.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    scenario = next(item for item in SCENARIOS if item.name == "test_azi40")
    dataset = np.load(upstream_root / "syn_datasets" / "dataset_test_azi40.npy")
    truth = np.asarray(api["normalize"](api["truth"](wavelet=scenario.truth_wavelet)))
    for velocity_scale in (0.9, 1.0, 1.1):
        rmsr_metric = _rmsr_prior_sensitivity(
            dataset,
            velocity_scale=velocity_scale,
            api=api,
            truth=truth,
        )
        rmsr_metric["scenario"] = scenario.name
        rows.append(rmsr_metric)
        if velocity_scale == 1.0:
            continue
        for seed in seeds:
            completed += 1
            print(
                f"[{completed}/{total}] test_azi40, seed={seed}, "
                f"velocity={velocity_scale:.1f}",
                flush=True,
            )
            try:
                metric, weights = _train_once(
                    dataset,
                    seed=seed,
                    velocity_scale=velocity_scale,
                    api=api,
                    truth=truth,
                )
                metric["scenario"] = scenario.name
                rows.append(metric)
                saved_weights[f"test_azi40_v{velocity_scale:.1f}_seed{seed}"] = weights
            except Exception as exc:
                failures.append(
                    {
                        "scenario": scenario.name,
                        "seed": seed,
                        "velocity_scale": velocity_scale,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    _write_csv(output_dir / "gate1_training_sensitivity_metrics.csv", rows)
    if failures:
        _write_csv(output_dir / "gate1_training_failures.csv", failures)
    np.savez_compressed(output_dir / "gate1_retrained_weights.npz", **saved_weights)
    _plot_initialization(rows, output_dir)
    _plot_prior(rows, output_dir)

    nominal = [
        row
        for row in rows
        if row["method"] == "nnws_retrained" and row["velocity_scale"] == 1.0
    ]
    scenario_summary: dict[str, Any] = {}
    for scenario in SCENARIOS:
        subset = [row for row in nominal if row["scenario"] == scenario.name]
        scenario_summary[scenario.name] = {
            "successful_seeds": len(subset),
            "waveform_cc_median": float(
                np.median([row["waveform_cc_full"] for row in subset])
            ) if subset else None,
            "waveform_cc_range": [
                float(np.min([row["waveform_cc_full"] for row in subset])),
                float(np.max([row["waveform_cc_full"] for row in subset])),
            ] if subset else None,
            "group_velocity_mae_percent_median": float(
                np.median([row["group_velocity_mae_percent"] for row in subset])
            ) if subset else None,
            "n_eff_median": float(np.median([row["n_eff"] for row in subset]))
            if subset
            else None,
        }
    summary = {
        "experiment": "Gate1_exact_upstream_retraining_and_reference_prior_sensitivity",
        "upstream_commit": UPSTREAM_COMMIT,
        "seeds": list(seeds),
        "epochs": 200,
        "device": "cpu",
        "loss_weights": [0.8, 0.2, 0.01],
        "reference_velocity_scales": [0.9, 1.0, 1.1],
        "scenario_summary": scenario_summary,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Five seeds are a reproducibility audit, not the paper SI's 100-run distribution.",
            "The ±10% test rescales tmin/tmax inversely with reference velocity.",
            "Overlap sensitivity cannot be reproduced from the released precomputed CCF matrices.",
        ],
    }
    (output_dir / "gate1_training_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(".wavedispx-tools/NNWS-upstream"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_python/nnws_gate1_training_sensitivity"),
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("0,1,2,3,4"))
    args = parser.parse_args(argv)
    summary = run(
        args.upstream_root.resolve(), args.output_dir.resolve(), args.seeds
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
