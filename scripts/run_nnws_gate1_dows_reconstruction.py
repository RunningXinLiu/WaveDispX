#!/usr/bin/env python3
"""Transparent reconstruction of the Fan et al. SI DOWS ablation.

The SI specifies the optimizer, loss, sigmoid-constrained direct weights, and
epochs, but does not publish the implementation or raw-weight initialization.
This script therefore labels every result as a reconstruction and scans the
otherwise unidentified initialization scale.  It must not be reported as an
exact reproduction of Figure S8.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_nnws_gate1_upstream_reproduction import (
    DELTA_S,
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
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _train_dows(
    dataset: np.ndarray,
    *,
    seed: int,
    initialization_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    import torch

    from nnws.losses import SNRLoss, SymmetryLoss, WeightStdLoss

    torch.manual_seed(seed)
    values = torch.as_tensor(dataset, dtype=torch.float32)
    raw_weight = torch.nn.Parameter(
        initialization_scale * torch.randn(dataset.shape[0], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam((raw_weight,), lr=1.0e-3)
    snr_loss = SNRLoss(TAU_S, TMIN_S, TMAX_S, DELTA_S)
    symmetry_loss = SymmetryLoss()
    std_loss = WeightStdLoss()
    final_components: tuple[float, float, float, float] | None = None
    for _ in range(200):
        weight = torch.sigmoid(raw_weight)
        stack = (weight[:, None] * values).sum(dim=0) / weight.sum().clamp_min(1.0e-8)
        components = (snr_loss(stack), symmetry_loss(stack), std_loss(weight))
        total = 0.8 * components[0] + 0.2 * components[1] + 0.01 * components[2]
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        final_components = (
            float(total.detach()),
            float(components[0].detach()),
            float(components[1].detach()),
            float(components[2].detach()),
        )
    assert final_components is not None
    with torch.no_grad():
        weight = torch.sigmoid(raw_weight)
        stack = (weight[:, None] * values).sum(dim=0) / weight.sum().clamp_min(1.0e-8)
    stack_array = stack.cpu().numpy()
    weight_array = weight.cpu().numpy()
    if not np.all(np.isfinite(stack_array)) or not np.all(np.isfinite(weight_array)):
        raise RuntimeError("DOWS reconstruction produced non-finite values")
    return stack_array, weight_array, {
        "final_total_loss": final_components[0],
        "final_snr_loss": final_components[1],
        "final_symmetry_loss": final_components[2],
        "final_std_loss": final_components[3],
    }


def _plot_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    nominal = [row for row in rows if row["initialization_scale"] == 1.0]
    metrics = (
        ("waveform_cc_full", "Full-window waveform CC", True),
        ("reported_snr", "Reported SNR", True),
        ("symmetry_rms_ratio", "Antisymmetry / symmetry RMS", False),
        ("group_velocity_mae_percent", "Group-velocity MAE (%)", False),
    )
    figure, axes = plt.subplots(1, 4, figsize=(14.2, 3.8), constrained_layout=True)
    for axis, (key, ylabel, higher_is_better) in zip(axes, metrics):
        data = [
            [float(row[key]) for row in nominal if row["scenario"] == scenario.name]
            for scenario in SCENARIOS
        ]
        axis.boxplot(data, tick_labels=[scenario.label for scenario in SCENARIOS])
        for position, values in enumerate(data, start=1):
            axis.scatter(
                np.full(len(values), position),
                values,
                color="#6a3d9a",
                s=14,
                alpha=0.75,
            )
        axis.set_ylabel(ylabel + (" ↑" if higher_is_better else " ↓"))
        axis.tick_params(axis="x", rotation=35, labelsize=7)
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle("Gate 1 DOWS reconstruction, N(0,1) raw-weight initialization")
    _save_figure(figure, output_dir, "gate1_dows_reconstruction_metrics")

    azi40 = [row for row in rows if row["scenario"] == "test_azi40"]
    figure, axes = plt.subplots(1, 4, figsize=(14.2, 3.8), constrained_layout=True)
    scales = (0.1, 1.0, 3.0)
    for axis, (key, ylabel, higher_is_better) in zip(axes, metrics):
        median = []
        low = []
        high = []
        for scale in scales:
            values = np.asarray(
                [float(row[key]) for row in azi40 if row["initialization_scale"] == scale]
            )
            median.append(float(np.median(values)))
            low.append(float(np.percentile(values, 25)))
            high.append(float(np.percentile(values, 75)))
        axis.plot(scales, median, marker="o", color="#6a3d9a")
        axis.fill_between(scales, low, high, color="#6a3d9a", alpha=0.2)
        axis.set_xlabel("Raw-weight normal initialization σ")
        axis.set_ylabel(ylabel + (" ↑" if higher_is_better else " ↓"))
        axis.set_xscale("log")
        axis.set_xticks(scales, labels=("0.1", "1", "3"))
        axis.grid(True, alpha=0.25)
    figure.suptitle("DOWS reconstruction ambiguity: unpublished initialization scale")
    _save_figure(figure, output_dir, "gate1_dows_initialization_scale_sensitivity")


def run(upstream_root: Path, output_dir: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    if _git_head(upstream_root) != UPSTREAM_COMMIT:
        raise RuntimeError(f"NNWS upstream must be pinned to {UPSTREAM_COMMIT}")
    output_dir.mkdir(parents=True, exist_ok=True)
    api = _load_upstream_api(upstream_root)
    rows: list[dict[str, Any]] = []
    weights: dict[str, np.ndarray] = {}
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    combinations = [
        (scenario, scale, seed)
        for scenario in SCENARIOS
        for scale in ((0.1, 1.0, 3.0) if scenario.name == "test_azi40" else (1.0,))
        for seed in seeds
    ]
    for index, (scenario, scale, seed) in enumerate(combinations, start=1):
        print(
            f"[{index}/{len(combinations)}] {scenario.name}, seed={seed}, init_sigma={scale}",
            flush=True,
        )
        dataset = np.load(
            upstream_root / "syn_datasets" / f"dataset_{scenario.name}.npy"
        )
        truth = np.asarray(api["normalize"](api["truth"](wavelet=scenario.truth_wavelet)))
        try:
            stack, weight, loss = _train_dows(
                dataset, seed=seed, initialization_scale=scale
            )
            metric, _, _, _ = _method_metrics(
                "dows_reconstructed",
                stack,
                truth,
                api,
                weight=weight,
            )
            metric.update(
                {
                    "scenario": scenario.name,
                    "seed": seed,
                    "initialization_distribution": "Normal(0, sigma)",
                    "initialization_scale": scale,
                    **loss,
                }
            )
            rows.append(metric)
            weights[f"{scenario.name}_sigma{scale}_seed{seed}"] = weight
        except Exception as exc:
            failures.append(
                {
                    "scenario": scenario.name,
                    "seed": seed,
                    "initialization_scale": scale,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    _write_csv(output_dir / "gate1_dows_reconstruction_metrics.csv", rows)
    if failures:
        _write_csv(output_dir / "gate1_dows_reconstruction_failures.csv", failures)
    np.savez_compressed(output_dir / "gate1_dows_reconstructed_weights.npz", **weights)
    _plot_results(rows, output_dir)

    nominal_summary: dict[str, Any] = {}
    for scenario in SCENARIOS:
        subset = [
            row
            for row in rows
            if row["scenario"] == scenario.name and row["initialization_scale"] == 1.0
        ]
        nominal_summary[scenario.name] = {
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
        "experiment": "Gate1_SI_DOWS_transparent_reconstruction",
        "status": "reconstruction_not_exact_Figure_S8_reproduction",
        "upstream_commit": UPSTREAM_COMMIT,
        "si_specified": {
            "parameterization": "one trainable raw weight per CCF followed by sigmoid",
            "loss_weights": [0.8, 0.2, 0.01],
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "epochs": 200,
        },
        "not_specified_by_si": {
            "raw_weight_initialization": "chosen here as Normal(0, sigma)",
            "implementation_source": "not released in the pinned NNWS repository",
        },
        "seeds": list(seeds),
        "initialization_scales": [0.1, 1.0, 3.0],
        "nominal_sigma1_summary": nominal_summary,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "gate1_dows_reconstruction_summary.json").write_text(
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
        default=Path("output_python/nnws_gate1_dows_reconstruction"),
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("0,1,2,3,4"))
    args = parser.parse_args(argv)
    summary = run(
        args.upstream_root.resolve(), args.output_dir.resolve(), args.seeds
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
