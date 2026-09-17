#!/usr/bin/env python3
"""Audit the Fan et al. (2026) Gate-1 synthetic data and saved weights.

This script deliberately works on the exact 200-km, 3-km/s upstream data.
It is a mechanism-reproduction experiment and must not be mixed with the
600-m, 4--12-Hz Gate-2 experiment in WaveDispX.

The source truth is used only after stacking to explain the weights/masks.  It
never enters RMSR-SS or NNWS.  RMSR-SS uses the SI-prescribed synthetic value
G=1+5/M.  The upstream test_uni figure's legacy G=1+1/M dispersion result is
reported only as a sensitivity row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


UPSTREAM_COMMIT = "04af38f042ec594c0a093178a16c1f72bd0e3380"
STACKMASTER_COMMIT = "66909971f69df60cb25d9e8b732a102d939cd2aa"
TAU_S = 8.0
TMIN_S = 64.0
TMAX_S = 69.0
DELTA_S = 1.0
DISTANCE_KM = 200.0
THEORY_SPEED_KMPS = 3.0
RMSR_THRESHOLD = 5.0


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    truth_wavelet: str
    added_sector_deg: tuple[float, float] | None


SCENARIOS = (
    Scenario("test_uni", "quasi-uniform", "pi", None),
    Scenario("test_azi40", "35–40° persistent sector", "sigma2", (35.0, 40.0)),
    Scenario("test_azi65", "60–65° persistent sector", "sigma2", (60.0, 65.0)),
    Scenario("test_azi90", "85–90° persistent sector", "sigma2", (85.0, 90.0)),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_upstream_api(upstream_root: Path):
    source_root = upstream_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from nnws.figures import _filtered_ground_truth, ftan_group_velocity
    from nnws.metrics import (
        normalize_max_abs,
        reported_snr,
        rmsr_selective_stacking,
    )
    from nnws.stacking import weighted_stack

    return {
        "truth": _filtered_ground_truth,
        "ftan": ftan_group_velocity,
        "normalize": normalize_max_abs,
        "snr": reported_snr,
        "rmsr": rmsr_selective_stacking,
        "weighted": weighted_stack,
    }


def _safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size != second.size or first.size < 2:
        return float("nan")
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _normalized_l2_no_shift(trace: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 error after one global scale, without a time shift."""
    trace = np.asarray(trace, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    denominator = float(np.dot(trace, trace))
    scale = float(np.dot(trace, truth) / denominator) if denominator > 0.0 else 0.0
    truth_norm = float(np.linalg.norm(truth))
    if truth_norm == 0.0:
        return float("nan")
    return float(np.linalg.norm(scale * trace - truth) / truth_norm)


def _symmetry_ratio(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=np.float64)
    midpoint = trace.size // 2
    symmetric = trace[: midpoint + 1][::-1] + trace[midpoint:]
    numerator = _rms(trace - trace[::-1])
    denominator = _rms(symmetric)
    return numerator / denominator if denominator > 0.0 else float("nan")


def _precursor_ratio(trace: np.ndarray) -> float:
    """Two-branch pre-arrival RMS divided by two-branch signal-window RMS."""
    trace = np.asarray(trace, dtype=np.float64)
    midpoint = trace.size // 2
    negative = trace[: midpoint + 1][::-1]
    positive = trace[midpoint:]
    start = int(TMIN_S / DELTA_S) - 2 * int(TAU_S / DELTA_S)
    stop = int(TMAX_S / DELTA_S) + 2 * int(TAU_S / DELTA_S)
    precursor = np.concatenate((negative[1:start], positive[1:start]))
    signal = np.concatenate((negative[start:stop], positive[start:stop]))
    denominator = _rms(signal)
    return _rms(precursor) / denominator if denominator > 0.0 else float("nan")


def _branch_metrics(trace: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    midpoint = trace.size // 2
    positive = trace[midpoint:]
    negative = trace[: midpoint + 1][::-1]
    truth_positive = truth[midpoint:]
    return (
        _safe_corr(positive, truth_positive),
        _safe_corr(negative, truth_positive),
        _safe_corr(positive, negative),
    )


def _source_segment_index(
    activation_s: np.ndarray,
    *,
    segment_span_s: float,
    segment_count: int,
) -> np.ndarray:
    """Map source activation time with the strict boundaries in upstream code."""
    activation = np.asarray(activation_s, dtype=np.float64)
    segment = np.floor(activation / segment_span_s).astype(np.int64)
    valid = (
        (segment >= 0)
        & (segment < segment_count)
        & (activation > segment_span_s * segment)
        & (activation < segment_span_s * (segment + 1))
    )
    return np.where(valid, segment, -1)


def _axis_distance_deg(theta_rad: np.ndarray) -> np.ndarray:
    degrees = np.mod(np.degrees(theta_rad), 360.0)
    to_zero = np.minimum(degrees, 360.0 - degrees)
    to_180 = np.abs(degrees - 180.0)
    return np.minimum(to_zero, to_180)


def _weighted_fraction(mask: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    return float(np.sum(weights[mask]) / denominator) if denominator > 0.0 else float("nan")


def _selective_stack_stackmaster(
    data: np.ndarray,
    *,
    correlation_minimum: float = 0.2,
    maximum_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Dependency-free reproduction of StackMaster selective(), commit above."""
    matrix = np.asarray(data, dtype=np.float64)
    midpoint = matrix.shape[1] // 2
    half_window = int(TMAX_S + 2 * TAU_S)
    start = midpoint - half_window
    stop = midpoint + half_window + 1
    current = np.mean(matrix, axis=0)
    selected = np.ones(matrix.shape[0], dtype=bool)
    residual = np.inf
    step = 0
    while residual > 1.0e-5 and step <= maximum_steps:
        coefficients = np.array(
            [_safe_corr(current[start:stop], row[start:stop]) for row in matrix]
        )
        selected = coefficients >= correlation_minimum
        step += 1
        if not np.any(selected):
            return np.full(matrix.shape[1], np.nan), selected, step
        previous = current
        current = np.mean(matrix[selected], axis=0)
        denominator = float(np.linalg.norm(current) * matrix.shape[1])
        residual = (
            float(np.linalg.norm(current - previous) / denominator)
            if denominator > 0.0
            else 0.0
        )
    return current, selected, step


def _method_metrics(
    name: str,
    stack: np.ndarray,
    truth: np.ndarray,
    api: dict[str, Any],
    *,
    weight: np.ndarray | None = None,
    selected: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    normalized = np.asarray(api["normalize"](stack), dtype=np.float64)
    periods, velocity = api["ftan"](
        np.linspace(-500.0, 500.0, 1001), normalized, distance=DISTANCE_KM
    )
    error_percent = (velocity - THEORY_SPEED_KMPS) / THEORY_SPEED_KMPS * 100.0
    positive_cc, negative_cc, branch_cc = _branch_metrics(normalized, truth)
    finite_error = np.abs(error_percent[np.isfinite(error_percent)])
    if weight is None:
        neff = float(np.count_nonzero(selected)) if selected is not None else 600.0
    else:
        weight = np.asarray(weight, dtype=np.float64).reshape(-1)
        denominator = float(np.sum(weight**2))
        neff = float(np.sum(weight) ** 2 / denominator) if denominator > 0.0 else 0.0
    metrics = {
        "method": name,
        "reported_snr": float(
            api["snr"](normalized, TAU_S, TMIN_S, TMAX_S, DELTA_S)
        ),
        "waveform_cc_full": _safe_corr(truth, normalized),
        "relative_l2_no_shift": _normalized_l2_no_shift(normalized, truth),
        "symmetry_rms_ratio": _symmetry_ratio(normalized),
        "precursor_rms_ratio": _precursor_ratio(normalized),
        "positive_branch_cc": positive_cc,
        "negative_branch_cc": negative_cc,
        "causal_anticausal_cc": branch_cc,
        "group_velocity_mae_percent": float(np.mean(finite_error)),
        "group_velocity_median_abs_error_percent": float(np.median(finite_error)),
        "group_velocity_max_abs_error_percent": float(np.max(finite_error)),
        "n_eff": neff,
        "selected_window_count": (
            int(np.count_nonzero(selected)) if selected is not None else None
        ),
    }
    return metrics, normalized, periods, error_percent


def _density_grid(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    *,
    limit: float = 620.0,
    bins: int = 150,
) -> np.ndarray:
    valid = np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        return np.zeros((bins, bins), dtype=np.float64)
    histogram, _, _ = np.histogram2d(
        x[valid],
        y[valid],
        bins=bins,
        range=((-limit, limit), (-limit, limit)),
        weights=weight[valid],
    )
    histogram = gaussian_filter(histogram, sigma=2.0)
    total = float(np.sum(histogram))
    return histogram / total if total > 0.0 else histogram


def _plot_stations(axis: plt.Axes) -> None:
    axis.axhline(0.0, color="white", linestyle="--", linewidth=0.7, alpha=0.7)
    axis.scatter(
        (-100.0, 100.0),
        (0.0, 0.0),
        marker="v",
        s=38,
        facecolor="#e31a1c",
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )
    axis.text(-100.0, 35.0, "A", ha="center", va="bottom", fontsize=7)
    axis.text(100.0, 35.0, "B", ha="center", va="bottom", fontsize=7)
    axis.set_xlim(-620.0, 620.0)
    axis.set_ylim(-620.0, 620.0)
    axis.set_aspect("equal")


def _save_figure(figure: plt.Figure, output_dir: Path, basename: str) -> None:
    figure.savefig(output_dir / f"{basename}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{basename}.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_source_audit(records: list[dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(4, 4, figsize=(13.2, 12.4), constrained_layout=True)
    column_titles = (
        "All activated sources",
        "RMSR retained sources",
        "RMSR rejected sources",
        "NNWS effective sources",
    )
    for row, record in enumerate(records):
        source = record["source"]
        x = source["x_km"]
        y = source["y_km"]
        valid = source["segment_code"] >= 0
        weights = (
            valid.astype(float),
            source["rmsr_retained"].astype(float),
            source["rmsr_rejected"].astype(float),
            source["nnws_source_weight"],
        )
        grids = [_density_grid(x, y, weight) for weight in weights]
        vmax = max(float(np.max(grid)) for grid in grids)
        for column, (axis, title, weight, grid) in enumerate(
            zip(axes[row], column_titles, weights, grids)
        ):
            image = axis.imshow(
                grid.T,
                origin="lower",
                extent=(-620.0, 620.0, -620.0, 620.0),
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            _plot_stations(axis)
            axis_distance = source["axis_distance_deg"]
            fraction = _weighted_fraction(axis_distance <= 10.0, weight)
            count = int(np.count_nonzero(weight > 0.0))
            axis.set_title(
                f"{title}\nN={count:,}; SPZ±10°={fraction:.3f}", fontsize=8
            )
            if row == 3:
                axis.set_xlabel("X (km)")
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(f"{record['scenario'].label}\nY (km)")
            else:
                axis.set_yticklabels([])
        figure.colorbar(image, ax=axes[row].tolist(), shrink=0.58, pad=0.01).set_label(
            "normalized smoothed source contribution", fontsize=7
        )
    figure.suptitle(
        "Gate 1 source audit (source truth used only after stacking)", fontsize=14
    )
    _save_figure(figure, output_dir, "gate1_source_selection_audit")


def _plot_waveform_dispersion(records: list[dict[str, Any]], output_dir: Path) -> None:
    colors = {
        "linear": "#4d4d4d",
        "rmsr_g5": "#1f78b4",
        "nnws_saved": "#e31a1c",
        "selective_cc02": "#33a02c",
    }
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 13.2), constrained_layout=True)
    lag = np.arange(-500.0, 501.0)
    for row, record in enumerate(records):
        waveform_axis, velocity_axis = axes[row]
        names = ("linear", "rmsr_g5", "nnws_saved", "selective_cc02")
        offset = 3.0
        for index, name in enumerate(names):
            trace = record["traces"][name]
            metric = record["metric_by_method"][name]
            baseline = (len(names) - index) * offset
            waveform_axis.plot(
                lag,
                trace + baseline,
                color=colors[name],
                linewidth=0.9,
                label=name if row == 0 else None,
            )
            waveform_axis.text(
                -176.0,
                baseline + 0.78,
                f"{name}: CC={metric['waveform_cc_full']:.2f}, "
                f"SNR={metric['reported_snr']:.1f}",
                fontsize=7,
                color=colors[name],
            )
        waveform_axis.plot(
            lag,
            record["truth"] - offset,
            color="black",
            linewidth=1.3,
            label="theoretical CCF" if row == 0 else None,
        )
        waveform_axis.axvline(-DISTANCE_KM / THEORY_SPEED_KMPS, color="0.6", ls="--", lw=0.7)
        waveform_axis.axvline(DISTANCE_KM / THEORY_SPEED_KMPS, color="0.6", ls="--", lw=0.7)
        waveform_axis.set_xlim(-180.0, 180.0)
        waveform_axis.set_yticks([])
        waveform_axis.set_ylabel(record["scenario"].label)
        if row == 3:
            waveform_axis.set_xlabel("Lag time (s)")
        for name in names:
            velocity_axis.plot(
                record["periods"][name],
                record["group_error"][name],
                color=colors[name],
                linewidth=1.1,
                marker="o",
                markersize=2.2,
                label=name if row == 0 else None,
            )
        velocity_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        velocity_axis.set_xlim(5.0, 16.0)
        velocity_axis.set_ylim(-4.0, 6.0)
        velocity_axis.grid(True, alpha=0.25)
        velocity_axis.set_ylabel("Group-velocity error (%)")
        if row == 3:
            velocity_axis.set_xlabel("Period (s)")
    axes[0, 0].set_title("Normalized NCF waveform and exact theoretical CCF")
    axes[0, 1].set_title("Paper-style FTAN group-velocity error")
    axes[0, 0].legend(ncol=3, fontsize=7, loc="upper center")
    axes[0, 1].legend(ncol=2, fontsize=7, loc="upper right")
    figure.suptitle("Gate 1 waveform and dispersion audit", fontsize=14)
    _save_figure(figure, output_dir, "gate1_waveform_group_velocity_audit")


def _plot_window_audit(records: list[dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(13.0, 11.0), constrained_layout=True)
    for row, record in enumerate(records):
        dataset = record["dataset"]
        weight = record["window_weight"]
        selected = record["rmsr_selected"]
        order = np.argsort(weight)[::-1]
        segment = dataset[order, 320:681]
        scale = float(np.percentile(np.abs(segment), 99.0))
        axes[row, 0].imshow(
            segment,
            aspect="auto",
            cmap="seismic",
            vmin=-scale,
            vmax=scale,
            extent=(-180.0, 180.0, dataset.shape[0], 0),
            interpolation="nearest",
        )
        axes[row, 0].set_ylabel(f"{record['scenario'].label}\nCCF rank")
        axes[row, 0].set_title("CCFs sorted by NNWS weight" if row == 0 else "")
        axes[row, 1].plot(weight, color="#e31a1c", linewidth=0.8, label="NNWS weight")
        axes[row, 1].scatter(
            np.flatnonzero(selected),
            np.full(np.count_nonzero(selected), -0.035),
            s=4,
            color="#1f78b4",
            label="RMSR retained" if row == 0 else None,
            clip_on=False,
        )
        axes[row, 1].set_ylim(-0.08, 1.05 * max(1.0, float(np.max(weight))))
        axes[row, 1].set_ylabel("Weight / mask")
        axes[row, 1].grid(True, alpha=0.2)
        axes[row, 1].set_title("Original window order" if row == 0 else "")
        if row == 3:
            axes[row, 0].set_xlabel("Lag time (s)")
            axes[row, 1].set_xlabel("Window index")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    figure.suptitle("Gate 1 short-CCF weights and RMSR decisions", fontsize=14)
    _save_figure(figure, output_dir, "gate1_window_weight_audit")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gate_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {record["scenario"].name: record for record in records}

    def metric(scenario: str, method: str, key: str) -> float:
        return float(by_name[scenario]["metric_by_method"][method][key])

    checks = {
        "uniform_nnws_cc_non_degradation": (
            metric("test_uni", "nnws_saved", "waveform_cc_full")
            >= metric("test_uni", "linear", "waveform_cc_full") - 0.02
        ),
        "uniform_nnws_group_non_degradation": (
            metric("test_uni", "nnws_saved", "group_velocity_mae_percent")
            <= metric("test_uni", "linear", "group_velocity_mae_percent") + 0.25
        ),
        "azi40_nnws_cc_better_than_rmsr": (
            metric("test_azi40", "nnws_saved", "waveform_cc_full")
            > metric("test_azi40", "rmsr_g5", "waveform_cc_full")
        ),
        "all_directional_nnws_cc_better_than_linear": all(
            metric(name, "nnws_saved", "waveform_cc_full")
            > metric(name, "linear", "waveform_cc_full")
            for name in ("test_azi40", "test_azi65", "test_azi90")
        ),
        "all_directional_nnws_symmetry_better_than_linear": all(
            metric(name, "nnws_saved", "symmetry_rms_ratio")
            < metric(name, "linear", "symmetry_rms_ratio")
            for name in ("test_azi40", "test_azi65", "test_azi90")
        ),
    }
    return {
        "status": "in_progress_not_a_final_gate_decision",
        "checks": checks,
        "all_main_mechanism_checks_pass": bool(all(checks.values())),
        "completed_in_companion_training_audit": [
            "five frozen initialization seeds for all four scenarios",
            "test_azi40 ±10% reference-velocity sensitivity",
            "1000-s paper versus 1001-s upstream source-window mapping sensitivity",
            "SI-defined DOWS transparent reconstruction with initialization-scale audit",
            "pinned StackMaster tf-PWS p=2 in the companion tf-PWS audit",
        ],
        "remaining_before_gate1_decision": [
            "overlap sensitivity requires continuous synthetics not present in the released CCF matrices",
        ],
    }


def run(upstream_root: Path, stackmaster_root: Path, output_dir: Path) -> dict[str, Any]:
    if _git_head(upstream_root) != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"NNWS upstream must be pinned to {UPSTREAM_COMMIT}; got {_git_head(upstream_root)}"
        )
    stackmaster_head = _git_head(stackmaster_root)
    if stackmaster_head != STACKMASTER_COMMIT:
        raise RuntimeError(
            f"StackMaster must be pinned to {STACKMASTER_COMMIT}; got {stackmaster_head}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    api = _load_upstream_api(upstream_root)
    records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    provenance_files: list[dict[str, str]] = []

    for scenario in SCENARIOS:
        dataset_path = upstream_root / "syn_datasets" / f"dataset_{scenario.name}.npy"
        info_path = upstream_root / "syn_datasets" / f"info_{scenario.name}.npy"
        weight_path = upstream_root / "syn_results" / f"{scenario.name}_nn_weight.npy"
        for role, path in (
            ("dataset", dataset_path),
            ("source_info", info_path),
            ("saved_nnws_weight", weight_path),
        ):
            provenance_files.append(
                {
                    "scenario": scenario.name,
                    "role": role,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
        dataset = np.load(dataset_path)
        info = np.load(info_path)
        window_weight = np.load(weight_path).reshape(-1).astype(np.float64)
        if dataset.shape != (600, 1001):
            raise ValueError(f"Unexpected {scenario.name} dataset shape: {dataset.shape}")
        if window_weight.size != dataset.shape[0]:
            raise ValueError(f"Unexpected {scenario.name} weight shape")
        expected_sources = 10_000 if scenario.name == "test_uni" else 11_000
        if info.shape != (expected_sources, 3):
            raise ValueError(f"Unexpected {scenario.name} source-info shape: {info.shape}")

        linear, rmsr, rmsr_selected = api["rmsr"](
            dataset, TAU_S, TMIN_S, TMAX_S, DELTA_S, threshold=RMSR_THRESHOLD
        )
        nnws = api["weighted"](dataset, window_weight)
        selective, selective_selected, selective_steps = _selective_stack_stackmaster(
            dataset
        )
        truth = np.asarray(api["truth"](wavelet=scenario.truth_wavelet), dtype=np.float64)
        truth = np.asarray(api["normalize"](truth), dtype=np.float64)

        method_inputs = {
            "linear": (linear, None, None),
            "rmsr_g5": (rmsr, None, rmsr_selected),
            "nnws_saved": (nnws, window_weight, None),
            "selective_cc02": (selective, None, selective_selected),
        }
        traces: dict[str, np.ndarray] = {}
        periods: dict[str, np.ndarray] = {}
        group_error: dict[str, np.ndarray] = {}
        metric_by_method: dict[str, dict[str, Any]] = {}
        for method, (stack, weight, selected) in method_inputs.items():
            metric, normalized, method_periods, error = _method_metrics(
                method,
                stack,
                truth,
                api,
                weight=weight,
                selected=selected,
            )
            metric.update(
                {
                    "scenario": scenario.name,
                    "scenario_label": scenario.label,
                    "truth_wavelet": scenario.truth_wavelet,
                    "rmsr_threshold_numerator": (
                        RMSR_THRESHOLD if method == "rmsr_g5" else None
                    ),
                    "selective_iterations": (
                        selective_steps if method == "selective_cc02" else None
                    ),
                }
            )
            metric_rows.append(metric)
            metric_by_method[method] = metric
            traces[method] = normalized
            periods[method] = method_periods
            group_error[method] = error

        if scenario.name == "test_uni":
            _, legacy, legacy_selected = api["rmsr"](
                dataset, TAU_S, TMIN_S, TMAX_S, DELTA_S, threshold=1.0
            )
            legacy_metric, _, _, _ = _method_metrics(
                "rmsr_g1_legacy_test_uni_dispersion",
                legacy,
                truth,
                api,
                selected=legacy_selected,
            )
            legacy_metric.update(
                {
                    "scenario": scenario.name,
                    "scenario_label": scenario.label,
                    "truth_wavelet": scenario.truth_wavelet,
                    "rmsr_threshold_numerator": 1.0,
                    "selective_iterations": None,
                }
            )
            metric_rows.append(legacy_metric)

        segment_code = _source_segment_index(
            info[:, 2], segment_span_s=1001.0, segment_count=dataset.shape[0]
        )
        segment_paper = _source_segment_index(
            info[:, 2], segment_span_s=1000.0, segment_count=dataset.shape[0]
        )
        valid = segment_code >= 0
        source_nnws_weight = np.zeros(info.shape[0], dtype=np.float64)
        source_rmsr_retained = np.zeros(info.shape[0], dtype=bool)
        source_nnws_weight[valid] = window_weight[segment_code[valid]]
        source_rmsr_retained[valid] = rmsr_selected[segment_code[valid]]
        valid_paper = segment_paper >= 0
        source_nnws_weight_paper = np.zeros(info.shape[0], dtype=np.float64)
        source_rmsr_retained_paper = np.zeros(info.shape[0], dtype=bool)
        source_nnws_weight_paper[valid_paper] = window_weight[segment_paper[valid_paper]]
        source_rmsr_retained_paper[valid_paper] = rmsr_selected[
            segment_paper[valid_paper]
        ]
        radius, theta, activation = info.T
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        axis_distance = _axis_distance_deg(theta)
        theta_degree = np.mod(np.degrees(theta), 360.0)
        added_sector = (
            np.zeros(info.shape[0], dtype=bool)
            if scenario.added_sector_deg is None
            else (
                (theta_degree >= scenario.added_sector_deg[0])
                & (theta_degree <= scenario.added_sector_deg[1])
                & (radius >= 450.0)
                & (radius <= 600.0)
            )
        )
        source = {
            "x_km": x,
            "y_km": y,
            "axis_distance_deg": axis_distance,
            "segment_code": segment_code,
            "segment_paper": segment_paper,
            "rmsr_retained": valid & source_rmsr_retained,
            "rmsr_rejected": valid & ~source_rmsr_retained,
            "nnws_source_weight": source_nnws_weight,
        }
        for mapping_name, mapping_valid, rmsr_source_weight, nnws_source_weight in (
            (
                "upstream_strict_1001",
                valid,
                (valid & source_rmsr_retained).astype(float),
                source_nnws_weight,
            ),
            (
                "paper_strict_1000",
                valid_paper,
                (valid_paper & source_rmsr_retained_paper).astype(float),
                source_nnws_weight_paper,
            ),
        ):
            for method, contribution in (
                ("linear", mapping_valid.astype(float)),
                ("rmsr_g5", rmsr_source_weight),
                ("nnws_saved", nnws_source_weight),
            ):
                mapping_rows.append(
                    {
                        "scenario": scenario.name,
                        "mapping": mapping_name,
                        "method": method,
                        "assigned_source_count": int(np.count_nonzero(mapping_valid)),
                        "positive_contribution_source_count": int(
                            np.count_nonzero(contribution > 0.0)
                        ),
                        "spz_axis_within_10deg_fraction": _weighted_fraction(
                            axis_distance <= 10.0, contribution
                        ),
                        "added_sector_fraction": _weighted_fraction(
                            added_sector, contribution
                        )
                        if scenario.added_sector_deg is not None
                        else None,
                    }
                )
        for index in range(info.shape[0]):
            source_rows.append(
                {
                    "scenario": scenario.name,
                    "source_index": index,
                    "radius_km": float(radius[index]),
                    "theta_deg": float(np.mod(np.degrees(theta[index]), 360.0)),
                    "x_km": float(x[index]),
                    "y_km": float(y[index]),
                    "activation_s": float(activation[index]),
                    "segment_code_1001": int(segment_code[index]),
                    "segment_paper_1000": int(segment_paper[index]),
                    "segment_mapping_differs": bool(segment_code[index] != segment_paper[index]),
                    "axis_distance_deg": float(axis_distance[index]),
                    "rmsr_retained": bool(source["rmsr_retained"][index]),
                    "rmsr_rejected": bool(source["rmsr_rejected"][index]),
                    "nnws_source_weight": float(source_nnws_weight[index]),
                    "rmsr_retained_paper_1000": bool(
                        valid_paper[index] & source_rmsr_retained_paper[index]
                    ),
                    "nnws_source_weight_paper_1000": float(
                        source_nnws_weight_paper[index]
                    ),
                }
            )
        for window_index in range(dataset.shape[0]):
            window_rows.append(
                {
                    "scenario": scenario.name,
                    "window_index": window_index,
                    "upstream_start_s": float(1001 * window_index),
                    "upstream_end_s": float(1001 * (window_index + 1)),
                    "paper_start_s": float(1000 * window_index),
                    "paper_end_s": float(1000 * (window_index + 1)),
                    "rmsr_g5_selected": bool(rmsr_selected[window_index]),
                    "nnws_saved_weight": float(window_weight[window_index]),
                    "selective_cc02_selected": bool(selective_selected[window_index]),
                }
            )

        records.append(
            {
                "scenario": scenario,
                "dataset": dataset,
                "truth": truth,
                "traces": traces,
                "periods": periods,
                "group_error": group_error,
                "metric_by_method": metric_by_method,
                "window_weight": window_weight,
                "rmsr_selected": rmsr_selected,
                "source": source,
                "mapping_difference_count": int(np.count_nonzero(segment_code != segment_paper)),
            }
        )

    _write_csv(output_dir / "gate1_metrics.csv", metric_rows)
    _write_csv(output_dir / "gate1_window_weights.csv", window_rows)
    _write_csv(output_dir / "gate1_source_audit.csv", source_rows)
    _write_csv(output_dir / "gate1_source_mapping_sensitivity.csv", mapping_rows)
    _write_csv(output_dir / "gate1_input_provenance.csv", provenance_files)
    _plot_source_audit(records, output_dir)
    _plot_waveform_dispersion(records, output_dir)
    _plot_window_audit(records, output_dir)

    summary = {
        "experiment": "Fan_2026_Gate1_exact_upstream_saved_weight_audit",
        "nnws_upstream_commit": UPSTREAM_COMMIT,
        "stackmaster_commit": STACKMASTER_COMMIT,
        "parameters": {
            "medium_velocity_kmps": THEORY_SPEED_KMPS,
            "station_distance_km": DISTANCE_KM,
            "ccf_shape": [600, 1001],
            "delta_s": DELTA_S,
            "period_band_s": [5.0, 16.0],
            "ftan_alpha": 2.5,
            "rmsr_G": "1 + 5/M",
            "selective_cc_threshold": 0.2,
            "selective_window_s": [-85.0, 85.0],
        },
        "scope_warning": (
            "Gate 1 uses the paper's 200-km constant-speed scale; Gate 2 separately "
            "uses the WaveDispX 600-m, 4-12-Hz dispersive model."
        ),
        "paper_code_differences": {
            "source_segment_mapping": (
                "paper says 1000 s; upstream mechanism code uses strict 1001*segment boundaries"
            ),
            "uniform_rmsr_dispersion": (
                "SI prescribes G=1+5/M for all synthetics; upstream test_uni figure "
                "uses G=1+1/M only in its dispersion panel"
            ),
            "theoretical_wavelet": (
                "upstream test_uni figure uses wavelet='pi'; directional figure uses 'sigma2'"
            ),
        },
        "source_mapping_difference_count": {
            record["scenario"].name: record["mapping_difference_count"]
            for record in records
        },
        "source_mapping_sensitivity": mapping_rows,
        "metrics": metric_rows,
        "gate1": _gate_checks(records),
    }
    (output_dir / "gate1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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
        "--stackmaster-root",
        type=Path,
        default=Path(".wavedispx-tools/StackMaster-upstream"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_python/nnws_gate1_upstream_reproduction"),
    )
    args = parser.parse_args(argv)
    summary = run(
        args.upstream_root.resolve(),
        args.stackmaster_root.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(summary["gate1"], indent=2, ensure_ascii=False))
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
