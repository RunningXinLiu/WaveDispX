#!/usr/bin/env python3
"""Evaluate passive NCF strategies with two-station FTAN and QEDisp truth.

Figure contract
---------------
Core conclusion: a selector is credible only when its pair FTAN ridge moves
toward the independently calculated 1-D dispersion curve and outperforms its
time-permuted control.  Archetype: quantitative multi-panel comparison.
Backend: Python/matplotlib.  Exports: editable SVG/PDF and raster PNG/TIFF.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.config import load_config
from wavedispx.processing import pair_ftan, phase_to_group_velocity, theoretical_dispersion


DEFAULT_STRATEGIES = (
    "baseline_linear",
    "nnws_v2_shared_oof",
    "nnws_v2_tau_p_branch_guard_frequency",
    "nnws_v2_tau_p_branch_guard_frequency_permuted_time_control",
    "pair_crossfit_tau_p_spz_frequency",
)
LABELS = {
    "active_reference": "Active EGF reference",
    "baseline_linear": "Linear NCF",
    "nnws_v2_shared_oof": "NNWS-v2 OOF",
    "nnws_v2_tau_p_branch_guard_frequency": "V2 + pair/branch tau-p",
    "nnws_v2_tau_p_branch_guard_frequency_permuted_time_control": (
        "V2 + guard, time-permuted"
    ),
    "pair_crossfit_tau_p_spz_frequency": "OOF pair/branch tau-p",
}
COLORS = {
    "active_reference": "#111111",
    "baseline_linear": "#7A7A7A",
    "nnws_v2_shared_oof": "#009E73",
    "nnws_v2_tau_p_branch_guard_frequency": "#CC79A7",
    "nnws_v2_tau_p_branch_guard_frequency_permuted_time_control": "#D55E00",
    "pair_crossfit_tau_p_spz_frequency": "#0072B2",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "legend.fontsize": 6.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.11,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def _save(figure: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        stem.with_suffix(".svg"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".png"),
        stem.with_suffix(".tiff"),
    ]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    figure.savefig(outputs[2], dpi=300, bbox_inches="tight")
    figure.savefig(outputs[3], dpi=600, bbox_inches="tight")
    plt.close(figure)
    return outputs


def _parse_strategies(value: str) -> tuple[str, ...]:
    strategies = tuple(item.strip() for item in value.split(",") if item.strip())
    if not strategies or len(strategies) != len(set(strategies)):
        raise argparse.ArgumentTypeError("strategies must be a unique comma list")
    return strategies


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    if finite.size == 0:
        return {
            "median_absolute_error_mps": float("nan"),
            "p90_absolute_error_mps": float("nan"),
            "maximum_absolute_error_mps": float("nan"),
        }
    return {
        "median_absolute_error_mps": float(np.median(finite)),
        "p90_absolute_error_mps": float(np.percentile(finite, 90.0)),
        "maximum_absolute_error_mps": float(np.max(finite)),
    }


def _image_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return float("nan")
    x = x[keep] - np.mean(x[keep])
    y = y[keep] - np.mean(y[keep])
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--branch", choices=("positive", "negative", "symmetric"), default="symmetric")
    parser.add_argument(
        "--strategies",
        type=_parse_strategies,
        default=DEFAULT_STRATEGIES,
    )
    parser.add_argument("--frequency-min-hz", type=float, default=None)
    parser.add_argument("--frequency-max-hz", type=float, default=None)
    parser.add_argument("--period-count", type=int, default=96)
    parser.add_argument("--velocity-step-mps", type=float, default=2.0)
    parser.add_argument("--phase-maximum-jump-mps", type=float, default=10.0)
    parser.add_argument("--gaussian-alpha", type=float, default=None)
    parser.add_argument("--phase-anchor-period-s", type=float, default=0.20)
    parser.add_argument(
        "--phase-method",
        choices=("time_domain", "frequency_domain"),
        default="frequency_domain",
    )
    parser.add_argument("--theory-solver", choices=("disba", "qedisp"), default="qedisp")
    parser.add_argument("--output-directory", type=Path, default=None)
    args = parser.parse_args()

    run = args.run_directory.resolve()
    results_path = run / "results.h5"
    config = load_config(args.config)
    frequency_min = args.frequency_min_hz or config.processing.frequency_min_hz
    frequency_max = args.frequency_max_hz or config.processing.frequency_max_hz
    if not 1.0 / frequency_max <= args.phase_anchor_period_s <= 1.0 / frequency_min:
        parser.error("phase anchor period must lie inside the FTAN period range")

    with h5py.File(results_path) as handle:
        lag_time_s = handle["ncf/lag_time_s"][:]
        ncf_distance_m = handle["ncf/distance_m"][:]
        target_distance = (
            float(np.max(ncf_distance_m))
            if args.distance_m is None
            else args.distance_m
        )
        ncf_index = int(np.argmin(np.abs(ncf_distance_m - target_distance)))
        selected_distance_m = float(ncf_distance_m[ncf_index])
        available = set(handle["ncf/strategies"])
        missing = [strategy for strategy in args.strategies if strategy not in available]
        if missing:
            parser.error(f"strategies missing from results.h5: {', '.join(missing)}")
        traces = {
            strategy: handle[
                f"ncf/strategies/{strategy}/components/zz/{args.branch}"
            ][:, ncf_index]
            for strategy in args.strategies
        }
        reference_distance_m = handle["egf_reference/distance_m"][:]
        reference_index = int(
            np.argmin(np.abs(reference_distance_m - selected_distance_m))
        )
        component_names = tuple(
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle["egf_reference"].attrs["component_names"]
        )
        component_index = component_names.index("zz")
        reference_time_s = handle["egf_reference/time_s"][:]
        reference_trace = handle["egf_reference/components"][
            :, reference_index, component_index
        ]

    theory_frequency_hz = np.linspace(frequency_min, frequency_max, 1001)
    theory = theoretical_dispersion(
        config.model,
        theory_frequency_hz,
        rayleigh_modes=max(1, config.processing.rayleigh_modes),
        love_modes=0,
        backend=args.theory_solver,
    )
    phase_values = theory.rayleigh_mps[:, 0]
    keep = np.isfinite(phase_values)
    theory_period_s = 1.0 / theory_frequency_hz[keep]
    theory_phase_mps = phase_values[keep]
    order = np.argsort(theory_period_s)
    theory_period_s = theory_period_s[order]
    theory_phase_mps = theory_phase_mps[order]
    theory_group_mps = phase_to_group_velocity(theory_period_s, theory_phase_mps)
    anchor_velocity_mps = float(
        np.interp(args.phase_anchor_period_s, theory_period_s, theory_phase_mps)
    )

    common = {
        "distance_m": selected_distance_m,
        "frequency_range_hz": (frequency_min, frequency_max),
        "group_velocity_range_mps": (
            config.processing.velocity_min_mps,
            config.processing.velocity_max_mps,
        ),
        "phase_velocity_range_mps": (
            config.processing.velocity_min_mps,
            config.processing.velocity_max_mps,
        ),
        "period_count": args.period_count,
        "velocity_step_mps": args.velocity_step_mps,
        "phase_method": args.phase_method,
        "phase_maximum_jump_mps": args.phase_maximum_jump_mps,
        "gaussian_alpha": args.gaussian_alpha,
        "phase_minus_group_range_mps": (-30.0, 220.0),
        "phase_velocity_anchor": (
            args.phase_anchor_period_s,
            anchor_velocity_mps,
        ),
        "minimum_snr": 2.0,
    }
    results = {
        "active_reference": pair_ftan(
            reference_trace,
            dt_s=float(np.median(np.diff(reference_time_s))),
            input_kind="egf",
            **common,
        )
    }
    ncf_dt_s = float(np.median(np.diff(lag_time_s)))
    for strategy, trace in traces.items():
        results[strategy] = pair_ftan(
            trace,
            dt_s=ncf_dt_s,
            input_kind="correlation",
            **common,
        )

    period_s = results["active_reference"].group_curve.period_s
    frequency_hz = 1.0 / period_s
    expected_group_mps = np.interp(period_s, theory_period_s, theory_group_mps)
    expected_phase_mps = np.interp(period_s, theory_period_s, theory_phase_mps)
    records: dict[str, dict[str, object]] = {}
    for name, result in results.items():
        group_valid = result.group_curve.valid_mask & np.isfinite(expected_group_mps)
        phase_valid = result.phase_curve.valid_mask & np.isfinite(expected_phase_mps)
        group_error = result.group_curve.velocity_mps - expected_group_mps
        phase_error = result.phase_curve.velocity_mps - expected_phase_mps
        cycle_difference = selected_distance_m * frequency_hz * (
            1.0 / result.phase_curve.velocity_mps - 1.0 / expected_phase_mps
        )
        cycle_offset = np.rint(cycle_difference)
        fractional_cycle_residual = np.abs(cycle_difference - cycle_offset)
        records[name] = {
            "group_valid_count": int(np.count_nonzero(group_valid)),
            "phase_valid_count": int(np.count_nonzero(phase_valid)),
            "group": _finite_summary(group_error[group_valid]),
            "phase": _finite_summary(phase_error[phase_valid]),
            "phase_cycle": {
                "median_absolute_cycle_difference": float(
                    np.median(np.abs(cycle_difference[phase_valid]))
                ),
                "median_fractional_cycle_residual": float(
                    np.median(fractional_cycle_residual[phase_valid])
                ),
                "maximum_absolute_integer_cycle_offset": int(
                    np.max(np.abs(cycle_offset[phase_valid]), initial=0)
                ),
            },
            "group_image_correlation_to_active_reference": _image_correlation(
                result.group_image.amplitude,
                results["active_reference"].group_image.amplitude,
            ),
            "phase_image_correlation_to_active_reference": _image_correlation(
                result.phase_image.amplitude,
                results["active_reference"].phase_image.amplitude,
            ),
        }

    output = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else run / "pair_ftan_qc"
    )
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_directory": str(run),
        "results_h5": str(results_path),
        "config": str(args.config.resolve()),
        "distance_m": selected_distance_m,
        "branch": args.branch,
        "frequency_range_hz": [frequency_min, frequency_max],
        "theory_solver": theory.solver,
        "phase_method": args.phase_method,
        "phase_anchor": {
            "period_s": args.phase_anchor_period_s,
            "velocity_mps": anchor_velocity_mps,
            "provenance": "QEDisp fundamental Rayleigh mode; validation only",
        },
        "strategies": records,
        "interpretation_guardrail": (
            "The phase curve uses a theory-assisted cycle anchor. The active EGF "
            "and time-permuted strategy are independent controls; this is not a "
            "blind field-data phase-velocity test."
        ),
    }
    (output / "passive_pair_ftan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output / "passive_pair_ftan_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "strategy",
                "period_s",
                "frequency_hz",
                "group_pick_mps",
                "group_theory_mps",
                "group_valid",
                "phase_pick_mps",
                "phase_theory_mps",
                "phase_valid",
                "phase_cycle_difference",
                "fractional_cycle_residual",
                "integer_cycle_offset",
            ]
        )
        for name, result in results.items():
            group_valid = result.group_curve.valid_mask & np.isfinite(expected_group_mps)
            phase_valid = result.phase_curve.valid_mask & np.isfinite(expected_phase_mps)
            cycle_difference = selected_distance_m * frequency_hz * (
                1.0 / result.phase_curve.velocity_mps - 1.0 / expected_phase_mps
            )
            cycle_offset = np.rint(cycle_difference)
            residual = np.abs(cycle_difference - cycle_offset)
            writer.writerows(
                zip(
                    np.repeat(name, period_s.size),
                    period_s,
                    frequency_hz,
                    result.group_curve.velocity_mps,
                    expected_group_mps,
                    group_valid.astype(int),
                    result.phase_curve.velocity_mps,
                    expected_phase_mps,
                    phase_valid.astype(int),
                    cycle_difference,
                    residual,
                    cycle_offset.astype(int),
                )
            )

    _style()
    panel_names = ("active_reference",) + args.strategies
    column_count = 3
    row_count = int(np.ceil(len(panel_names) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(10.8, 3.25 * row_count),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for panel_index, (axis, name) in enumerate(zip(axes.flat, panel_names)):
        result = results[name]
        axis.pcolormesh(
            period_s,
            result.phase_image.velocity_mps,
            result.phase_image.amplitude,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        axis.plot(theory_period_s, theory_phase_mps, color="white", linewidth=1.5)
        valid = result.phase_curve.valid_mask
        axis.plot(
            period_s[valid],
            result.phase_curve.velocity_mps[valid],
            color="#00D7FF",
            linewidth=1.0,
        )
        metric = records[name]["phase"]["median_absolute_error_mps"]
        axis.set_title(f"{LABELS.get(name, name)}\nmedian |c − QEDisp| = {metric:.2f} m/s")
        axis.set_xlabel("Period (s)")
        axis.set_ylabel("Phase velocity (m/s)")
        _panel(axis, chr(ord("a") + panel_index))
    for axis in axes.flat[len(panel_names) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Pair FTAN phase energy | {selected_distance_m:.0f} m | {args.branch} branch",
        fontsize=10.0,
    )
    _save(figure, output / "passive_pair_ftan_phase_energy")

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), constrained_layout=True)
    for name, result in results.items():
        scale = np.max(np.abs(result.processed_trace))
        axes[0, 0].plot(
            result.time_s,
            result.processed_trace / max(scale, np.finfo(float).eps),
            color=COLORS.get(name),
            linewidth=0.75,
            label=LABELS.get(name, name),
        )
    axes[0, 0].set_xlim(
        max(0.0, selected_distance_m / config.processing.velocity_max_mps - 0.2),
        selected_distance_m / config.processing.velocity_min_mps + 0.2,
    )
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Individually normalized amplitude")
    axes[0, 0].set_title("Stacked pair waveform after FTAN input conversion")
    axes[0, 0].legend(loc="upper right", ncols=2)
    _panel(axes[0, 0], "a")

    for axis, kind, panel in (
        (axes[0, 1], "group", "b"),
        (axes[1, 0], "phase", "c"),
    ):
        theory_curve = theory_group_mps if kind == "group" else theory_phase_mps
        axis.plot(
            theory_period_s,
            theory_curve,
            color="#111111",
            linewidth=1.6,
            label="QEDisp fundamental",
        )
        for name, result in results.items():
            curve = result.group_curve if kind == "group" else result.phase_curve
            axis.plot(
                curve.period_s[curve.valid_mask],
                curve.velocity_mps[curve.valid_mask],
                color=COLORS.get(name),
                linewidth=0.9,
                label=LABELS.get(name, name),
            )
        axis.set_xlim(period_s[0], period_s[-1])
        axis.set_ylim(
            config.processing.velocity_min_mps,
            config.processing.velocity_max_mps,
        )
        axis.set_xlabel("Period (s)")
        axis.set_ylabel(f"{kind.title()} velocity (m/s)")
        axis.set_title(f"{kind.title()}-velocity FTAN ridge")
        _panel(axis, panel)

    names = list(results)
    y = np.arange(len(names))
    group_mae = [records[name]["group"]["median_absolute_error_mps"] for name in names]
    phase_mae = [records[name]["phase"]["median_absolute_error_mps"] for name in names]
    axes[1, 1].barh(y - 0.18, group_mae, height=0.34, color="#0072B2", label="Group")
    axes[1, 1].barh(y + 0.18, phase_mae, height=0.34, color="#D55E00", label="Phase")
    axes[1, 1].set_yticks(y, [LABELS.get(name, name) for name in names])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Median absolute error to QEDisp (m/s)")
    axes[1, 1].set_title("Known-truth FTAN error; lower is better")
    axes[1, 1].legend(loc="lower right")
    _panel(axes[1, 1], "d")
    _save(figure, output / "passive_pair_ftan_curve_qc")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
