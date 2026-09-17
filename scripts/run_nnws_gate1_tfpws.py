#!/usr/bin/env python3
"""Run the exact pinned StackMaster tf-PWS function on Fan Gate-1 data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_nnws_gate1_upstream_reproduction import (
    DELTA_S,
    DISTANCE_KM,
    RMSR_THRESHOLD,
    SCENARIOS,
    STACKMASTER_COMMIT,
    TAU_S,
    THEORY_SPEED_KMPS,
    TMAX_S,
    TMIN_S,
    UPSTREAM_COMMIT,
    _git_head,
    _load_upstream_api,
    _method_metrics,
    _save_figure,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exact_tfpws(stackmaster_root: Path):
    """Import pinned core.tfpws without installing its unused tslearn extra."""
    if _git_head(stackmaster_root) != STACKMASTER_COMMIT:
        raise RuntimeError(f"StackMaster must be pinned to {STACKMASTER_COMMIT}")
    try:
        import stockwell  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "stockwell==1.2 is required; install the project's stackmaster extra"
        ) from exc

    # StackMaster imports tslearn at module import time for clusterstack only.
    # tfpws does not call either symbol.  Stubbing these two unused imports lets
    # us execute the unmodified, pinned tfpws function instead of copying it.
    if "tslearn" not in sys.modules:
        tslearn = types.ModuleType("tslearn")
        tslearn_utils = types.ModuleType("tslearn.utils")
        tslearn_clustering = types.ModuleType("tslearn.clustering")
        tslearn_utils.to_time_series_dataset = None
        tslearn_clustering.TimeSeriesKMeans = None
        sys.modules["tslearn"] = tslearn
        sys.modules["tslearn.utils"] = tslearn_utils
        sys.modules["tslearn.clustering"] = tslearn_clustering
    if str(stackmaster_root) not in sys.path:
        sys.path.insert(0, str(stackmaster_root))
    from stackmaster.core import tfpws

    return tfpws


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    records: list[dict[str, Any]], output_dir: Path
) -> None:
    colors = {"linear": "#555555", "tfpws_p2": "#6a3d9a", "nnws_saved": "#e31a1c"}
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 12.8), constrained_layout=True)
    lag = np.arange(-500.0, 501.0)
    for row, record in enumerate(records):
        waveform_axis, group_axis = axes[row]
        for index, method in enumerate(("linear", "tfpws_p2", "nnws_saved")):
            baseline = (2 - index) * 3.0
            metric = record["metrics"][method]
            waveform_axis.plot(
                lag,
                record["traces"][method] + baseline,
                color=colors[method],
                linewidth=1.0,
                label=method if row == 0 else None,
            )
            waveform_axis.text(
                -176.0,
                baseline + 0.75,
                f"{method}: CC={metric['waveform_cc_full']:.3f}, "
                f"SNR={metric['reported_snr']:.2f}",
                color=colors[method],
                fontsize=7,
            )
            group_axis.plot(
                record["periods"][method],
                record["group_error"][method],
                color=colors[method],
                marker="o",
                markersize=2.2,
                linewidth=1.1,
                label=method if row == 0 else None,
            )
        waveform_axis.plot(
            lag,
            record["truth"] - 3.0,
            color="black",
            linewidth=1.3,
            label="theoretical CCF" if row == 0 else None,
        )
        for sign in (-1.0, 1.0):
            waveform_axis.axvline(
                sign * DISTANCE_KM / THEORY_SPEED_KMPS,
                color="0.65",
                linestyle="--",
                linewidth=0.7,
            )
        waveform_axis.set_xlim(-180.0, 180.0)
        waveform_axis.set_yticks([])
        waveform_axis.set_ylabel(record["scenario"].label)
        group_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        group_axis.set_xlim(5.0, 16.0)
        group_axis.set_ylim(-4.0, 6.0)
        group_axis.grid(True, alpha=0.25)
        group_axis.set_ylabel("Group-velocity error (%)")
        if row == 3:
            waveform_axis.set_xlabel("Lag time (s)")
            group_axis.set_xlabel("Period (s)")
    axes[0, 0].set_title("Normalized NCF waveform")
    axes[0, 1].set_title("Paper-style FTAN group-velocity error")
    axes[0, 0].legend(ncol=2, fontsize=7, loc="upper center")
    axes[0, 1].legend(fontsize=7, loc="upper right")
    figure.suptitle("Gate 1 pinned StackMaster tf-PWS (p=2) audit")
    _save_figure(figure, output_dir, "gate1_stackmaster_tfpws_audit")


def run(upstream_root: Path, stackmaster_root: Path, output_dir: Path) -> dict[str, Any]:
    if _git_head(upstream_root) != UPSTREAM_COMMIT:
        raise RuntimeError(f"NNWS upstream must be pinned to {UPSTREAM_COMMIT}")
    tfpws = _load_exact_tfpws(stackmaster_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    api = _load_upstream_api(upstream_root)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, scenario in enumerate(SCENARIOS, start=1):
        print(f"[{index}/{len(SCENARIOS)}] StackMaster tfpws p=2: {scenario.name}", flush=True)
        dataset = np.load(
            upstream_root / "syn_datasets" / f"dataset_{scenario.name}.npy"
        )
        saved_weight = np.load(
            upstream_root / "syn_results" / f"{scenario.name}_nn_weight.npy"
        ).reshape(-1)
        truth = np.asarray(api["normalize"](api["truth"](wavelet=scenario.truth_wavelet)))
        linear, _, _ = api["rmsr"](
            dataset,
            TAU_S,
            TMIN_S,
            TMAX_S,
            DELTA_S,
            threshold=RMSR_THRESHOLD,
        )
        nnws = api["weighted"](dataset, saved_weight)
        try:
            tfpws_stack = np.asarray(tfpws(dataset, p=2, axis=0), dtype=np.float64)
            if tfpws_stack.shape != (1001,) or not np.all(np.isfinite(tfpws_stack)):
                raise RuntimeError(
                    f"tf-PWS returned shape={tfpws_stack.shape}, "
                    f"finite={bool(np.all(np.isfinite(tfpws_stack)))}"
                )
        except Exception as exc:
            failures.append(
                {"scenario": scenario.name, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        traces: dict[str, np.ndarray] = {}
        periods: dict[str, np.ndarray] = {}
        group_error: dict[str, np.ndarray] = {}
        metrics: dict[str, dict[str, Any]] = {}
        for method, stack, weight in (
            ("linear", linear, None),
            ("tfpws_p2", tfpws_stack, None),
            ("nnws_saved", nnws, saved_weight),
        ):
            metric, normalized, method_periods, error = _method_metrics(
                method, stack, truth, api, weight=weight
            )
            metric.update({"scenario": scenario.name, "tfpws_exponent": 2 if method == "tfpws_p2" else None})
            rows.append(metric)
            metrics[method] = metric
            traces[method] = normalized
            periods[method] = method_periods
            group_error[method] = error
            arrays[f"{scenario.name}_{method}"] = normalized
        records.append(
            {
                "scenario": scenario,
                "truth": truth,
                "traces": traces,
                "periods": periods,
                "group_error": group_error,
                "metrics": metrics,
            }
        )
    if rows:
        _write_csv(output_dir / "gate1_tfpws_metrics.csv", rows)
        np.savez_compressed(output_dir / "gate1_tfpws_waveforms.npz", **arrays)
    if failures:
        _write_csv(output_dir / "gate1_tfpws_failures.csv", failures)
    if records:
        _plot(records, output_dir)

    tfpws_rows = [row for row in rows if row["method"] == "tfpws_p2"]
    summary = {
        "experiment": "Gate1_exact_pinned_StackMaster_tfpws",
        "status": "complete" if len(tfpws_rows) == len(SCENARIOS) else "failed_or_partial",
        "nnws_upstream_commit": UPSTREAM_COMMIT,
        "stackmaster_commit": STACKMASTER_COMMIT,
        "stackmaster_core_sha256": _sha256(stackmaster_root / "stackmaster" / "core.py"),
        "stockwell_version": "1.2",
        "tfpws_exponent": 2,
        "input_preprocessing": "released Fan synthetic CCF matrix without additional normalization",
        "metrics": tfpws_rows,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_boundary": (
            "tf-PWS is a nonlinear phase-coherence stack without per-window retained/rejected "
            "weights; it is a display/SNR baseline, not a source-selection mechanism."
        ),
    }
    (output_dir / "gate1_tfpws_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root", type=Path, default=Path(".wavedispx-tools/NNWS-upstream")
    )
    parser.add_argument(
        "--stackmaster-root",
        type=Path,
        default=Path(".wavedispx-tools/StackMaster-upstream"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_python/nnws_gate1_tfpws"),
    )
    args = parser.parse_args(argv)
    summary = run(
        args.upstream_root.resolve(),
        args.stackmaster_root.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
