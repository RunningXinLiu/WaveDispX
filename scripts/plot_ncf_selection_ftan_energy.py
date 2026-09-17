#!/usr/bin/env python3
"""Plot pick-free group/phase FTAN energy for the NCF selection experiment.

Figure contract
---------------
Core conclusion: retained windows are credible only when both group- and
phase-velocity FTAN energy move toward the independent theoretical curves and
away from the rejected/time-permuted controls.  No automatic ridge pick is
drawn or used by the primary metrics.  Backend: Python/matplotlib.  Exports:
editable SVG/PDF and raster PNG/TIFF, plus CSV/JSON source data.
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


METHODS = (
    "active_reference",
    "ideal_uniform_ncf",
    "biased_all_windows",
    "data_only_retained",
    "data_only_rejected",
    "time_permuted_control",
)
LABELS = {
    "active_reference": "Active EGF",
    "ideal_uniform_ncf": "Ideal uniform-source NCF",
    "biased_all_windows": "Biased-source, all windows",
    "data_only_retained": "Data-only retained",
    "data_only_rejected": "Data-only rejected",
    "time_permuted_control": "Time-permuted control",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel(axis: plt.Axes, index: int) -> None:
    axis.text(
        -0.11,
        1.04,
        chr(ord("a") + index),
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


def _component_index(group: h5py.Group, component: str) -> int:
    names = tuple(
        value.decode() if isinstance(value, bytes) else str(value)
        for value in group.attrs["component_names"]
    )
    return names.index(component)


def _load_biased_traces(
    results_path: Path,
    *,
    target_distance_m: float | None,
    branch: str,
    random_control_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with h5py.File(results_path) as handle:
        if "pair_window_ncf" not in handle:
            raise ValueError(
                "strict two-station FTAN requires pair_window_ncf; rerun with "
                "nnws_v2.pair_ftan_receiver_index"
            )
        pair_group = handle["pair_window_ncf"]
        selected_distance = float(pair_group.attrs["distance_m"])
        pair_i = int(pair_group.attrs["pair_i"])
        pair_j = int(pair_group.attrs["pair_j"])
        if target_distance_m is not None and not np.isclose(
            selected_distance, target_distance_m, atol=0.25
        ):
            raise ValueError("saved strict pair does not match --distance-m")
        lag = pair_group["lag_time_s"][:]
        windows = pair_group[branch][:].astype(np.float64)
        hard_band_weights = handle[
            "pair_crossfit_tau_p_spz/diffuse_hard_band_weights"
        ][:]
        hard_window = np.mean(hard_band_weights, axis=1) > 0.5
        if hard_window.size != windows.shape[0] or not np.any(hard_window) or np.all(
            hard_window
        ):
            raise ValueError("strict-pair hard mask is empty, full, or misaligned")
        permutation = np.random.default_rng(random_control_seed + 47).permutation(
            windows.shape[0]
        )
        permuted_window = hard_window[permutation]
        baseline = np.mean(windows, axis=0)
        retained = np.mean(windows[hard_window], axis=0)
        rejected = np.mean(windows[~hard_window], axis=0)
        permuted = np.mean(windows[permuted_window], axis=0)
        all_count = float(windows.shape[0])
        retained_count = float(np.count_nonzero(hard_window))
        reconstructed = (
            retained_count * retained
            + (all_count - retained_count) * rejected
        ) / all_count
        if not np.allclose(reconstructed, baseline, rtol=1e-12, atol=1e-15):
            raise AssertionError("strict-pair retained/rejected stacks do not reconstruct baseline")
        reference_group = handle["egf_reference"]
        reference_distance = reference_group["distance_m"][:]
        reference_index = int(
            np.argmin(np.abs(reference_distance - selected_distance))
        )
        reference_trace = reference_group["components"][
            :, reference_index, _component_index(reference_group, "zz")
        ]
        reference_time = reference_group["time_s"][:]
        calibration_operator = str(reference_group.attrs["calibration_operator"])
    return (
        {
            "active_reference": reference_trace,
            "biased_all_windows": baseline,
            "data_only_retained": retained,
            "data_only_rejected": rejected,
            "time_permuted_control": permuted,
        },
        {
            "distance_m": selected_distance,
            "lag_time_s": lag,
            "reference_time_s": reference_time,
            "calibration_operator": calibration_operator,
            "all_window_count": all_count,
            "retained_effective_window_count": retained_count,
            "pair_i": pair_i,
            "pair_j": pair_j,
            "strict_two_station": True,
        },
    )


def _load_ideal_trace(
    results_path: Path,
    *,
    distance_m: float,
    branch: str,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(results_path) as handle:
        if "pair_window_ncf" not in handle:
            raise ValueError("ideal run lacks strict pair_window_ncf")
        pair_group = handle["pair_window_ncf"]
        if not np.isclose(float(pair_group.attrs["distance_m"]), distance_m):
            raise ValueError("ideal and biased runs do not share the strict pair")
        lag = pair_group["lag_time_s"][:]
        trace = np.mean(pair_group[branch][:].astype(np.float64), axis=0)
    return trace, lag


def _bandpass_fft(
    trace: np.ndarray,
    dt_s: float,
    frequency_range_hz: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    frequency = np.fft.rfftfreq(values.size, dt_s)
    spectrum = np.fft.rfft(values)
    keep = (frequency >= frequency_range_hz[0]) & (
        frequency <= frequency_range_hz[1]
    )
    spectrum[~keep] = 0.0
    return np.fft.irfft(spectrum, n=values.size)


def _image_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else float("nan")


def _theory_tube_fraction(
    amplitude: np.ndarray,
    velocity_mps: np.ndarray,
    theory_velocity_mps: np.ndarray,
    half_width_mps: float,
) -> tuple[float, np.ndarray]:
    energy = np.square(np.asarray(amplitude, dtype=np.float64))
    mask = np.abs(
        velocity_mps[:, None] - theory_velocity_mps[None, :]
    ) <= half_width_mps
    denominator = np.sum(energy, axis=0)
    fraction = np.divide(
        np.sum(np.where(mask, energy, 0.0), axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.finfo(float).eps,
    )
    return float(np.median(fraction)), fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biased-run-directory", type=Path, required=True)
    parser.add_argument("--ideal-run-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--distance-m", type=float, default=600.0)
    parser.add_argument(
        "--branch",
        choices=("positive", "negative", "symmetric"),
        default="symmetric",
    )
    parser.add_argument("--frequency-min-hz", type=float, default=None)
    parser.add_argument("--frequency-max-hz", type=float, default=None)
    parser.add_argument("--theory-tube-half-width-mps", type=float, default=15.0)
    parser.add_argument("--period-count", type=int, default=96)
    parser.add_argument("--velocity-step-mps", type=float, default=2.0)
    parser.add_argument("--gaussian-alpha", type=float, default=20.0)
    parser.add_argument("--theory-solver", choices=("disba", "qedisp"), default="qedisp")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    biased_run = args.biased_run_directory.resolve()
    ideal_run = args.ideal_run_directory.resolve()
    config = load_config(args.config)
    frequency_min = args.frequency_min_hz or config.processing.frequency_min_hz
    frequency_max = args.frequency_max_hz or config.processing.frequency_max_hz
    traces, metadata = _load_biased_traces(
        biased_run / "results.h5",
        target_distance_m=args.distance_m,
        branch=args.branch,
        random_control_seed=config.nnws_v2.random_control_seed,
    )
    ideal_trace, ideal_lag = _load_ideal_trace(
        ideal_run / "results.h5",
        distance_m=float(metadata["distance_m"]),
        branch=args.branch,
    )
    traces["ideal_uniform_ncf"] = ideal_trace

    calibration_operator = str(metadata["calibration_operator"])
    if calibration_operator == "negative_time_derivative_of_ncf":
        for method in METHODS:
            if method == "active_reference":
                continue
            time = ideal_lag if method == "ideal_uniform_ncf" else np.asarray(
                metadata["lag_time_s"]
            )
            dt_s = float(np.median(np.diff(time)))
            traces[method] = -np.gradient(traces[method], dt_s)
    elif calibration_operator not in {"identity", "none"}:
        raise ValueError(f"unsupported NCF calibration operator: {calibration_operator}")

    theory_frequency_hz = np.linspace(frequency_min, frequency_max, 1001)
    theory = theoretical_dispersion(
        config.model,
        theory_frequency_hz,
        rayleigh_modes=max(1, config.processing.rayleigh_modes),
        love_modes=0,
        backend=args.theory_solver,
    )
    finite = np.isfinite(theory.rayleigh_mps[:, 0])
    theory_period = 1.0 / theory_frequency_hz[finite]
    theory_phase = theory.rayleigh_mps[finite, 0]
    order = np.argsort(theory_period)
    theory_period = theory_period[order]
    theory_phase = theory_phase[order]
    theory_group = phase_to_group_velocity(theory_period, theory_phase)
    anchor_period = float(0.95 / frequency_min)
    anchor_velocity = float(np.interp(anchor_period, theory_period, theory_phase))
    common = {
        "distance_m": float(metadata["distance_m"]),
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
        "phase_method": "frequency_domain",
        "phase_velocity_anchor": (anchor_period, anchor_velocity),
        "phase_maximum_jump_mps": 10.0,
        "minimum_snr": 0.0,
        "gaussian_alpha": args.gaussian_alpha,
    }
    results = {}
    for method in METHODS:
        if method == "active_reference":
            time = np.asarray(metadata["reference_time_s"])
            input_kind = "egf"
        elif method == "ideal_uniform_ncf":
            time = ideal_lag
            input_kind = "egf"
        else:
            time = np.asarray(metadata["lag_time_s"])
            input_kind = "egf"
        results[method] = pair_ftan(
            traces[method],
            dt_s=float(np.median(np.diff(time))),
            input_kind=input_kind,
            **common,
        )

    period = results["active_reference"].group_image.period_s
    theory_group_on_period = np.interp(period, theory_period, theory_group)
    theory_phase_on_period = np.interp(period, theory_period, theory_phase)
    reference_group_image = results["active_reference"].group_image.amplitude
    reference_phase_image = results["active_reference"].phase_image.amplitude
    metrics: dict[str, dict[str, float]] = {}
    tube_fraction_by_method: dict[str, dict[str, np.ndarray]] = {}
    for method, result in results.items():
        group_tube, group_fraction = _theory_tube_fraction(
            result.group_image.amplitude,
            result.group_image.velocity_mps,
            theory_group_on_period,
            args.theory_tube_half_width_mps,
        )
        phase_tube, phase_fraction = _theory_tube_fraction(
            result.phase_image.amplitude,
            result.phase_image.velocity_mps,
            theory_phase_on_period,
            args.theory_tube_half_width_mps,
        )
        tube_fraction_by_method[method] = {
            "group": group_fraction,
            "phase": phase_fraction,
        }
        metrics[method] = {
            "median_group_theory_tube_energy_fraction": group_tube,
            "median_phase_theory_tube_energy_fraction": phase_tube,
            "group_image_correlation_to_active_egf": _image_correlation(
                result.group_image.amplitude, reference_group_image
            ),
            "phase_image_correlation_to_active_egf": _image_correlation(
                result.phase_image.amplitude, reference_phase_image
            ),
        }

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "biased_run_directory": str(biased_run),
        "ideal_run_directory": str(ideal_run),
        "config": str(args.config.resolve()),
        "distance_m": metadata["distance_m"],
        "branch": args.branch,
        "frequency_range_hz": [frequency_min, frequency_max],
        "theory_solver": theory.solver,
        "gaussian_alpha": args.gaussian_alpha,
        "ncf_calibration_operator": calibration_operator,
        "theory_tube_half_width_mps": args.theory_tube_half_width_mps,
        "primary_metrics_use_automatic_picks": False,
        "strict_two_station": metadata["strict_two_station"],
        "receiver_pair": [metadata["pair_i"], metadata["pair_j"]],
        "all_window_count": metadata["all_window_count"],
        "retained_effective_window_count": metadata[
            "retained_effective_window_count"
        ],
        "methods": metrics,
    }
    (output / "ncf_selection_ftan_energy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    common_count = min(
        len(np.asarray(metadata["reference_time_s"])),
        len(np.asarray(metadata["lag_time_s"])),
    )
    common_time = np.asarray(metadata["lag_time_s"])[:common_count]
    dt_common = float(np.median(np.diff(common_time)))
    filtered_waveforms = {
        method: _bandpass_fft(
            traces[method][:common_count],
            dt_common,
            (frequency_min, frequency_max),
        )
        for method in METHODS
    }
    arrival = (
        common_time >= float(metadata["distance_m"]) / config.processing.velocity_max_mps
    ) & (
        common_time <= float(metadata["distance_m"]) / config.processing.velocity_min_mps
    )
    reference_waveform = filtered_waveforms["active_reference"]
    waveform_metrics = {}
    for method, waveform in filtered_waveforms.items():
        reference_part = reference_waveform[arrival]
        waveform_part = waveform[arrival]
        reference_scale = max(np.linalg.norm(reference_part), np.finfo(float).eps)
        waveform_scale = max(np.linalg.norm(waveform_part), np.finfo(float).eps)
        normalized_reference = reference_part / reference_scale
        normalized_waveform = waveform_part / waveform_scale
        waveform_metrics[method] = {
            "waveform_correlation_to_active_egf": _image_correlation(
                waveform_part, reference_part
            ),
            "normalized_waveform_rmse_to_active_egf": float(
                np.sqrt(np.mean((normalized_waveform - normalized_reference) ** 2))
            ),
        }
        metrics[method].update(waveform_metrics[method])

    summary["methods"] = metrics
    (output / "ncf_selection_ftan_energy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output / "ncf_selection_pair_waveform_source_data.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "time_s", "calibrated_bandpassed_amplitude"])
        for method in METHODS:
            writer.writerows(
                zip(
                    np.repeat(method, common_time.size),
                    common_time,
                    filtered_waveforms[method],
                )
            )
    with (output / "ncf_selection_ftan_energy_source_data.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "kind",
                "period_s",
                "theory_velocity_mps",
                "theory_tube_energy_fraction",
            ]
        )
        for method in METHODS:
            for kind, theory_values in (
                ("group", theory_group_on_period),
                ("phase", theory_phase_on_period),
            ):
                writer.writerows(
                    zip(
                        np.repeat(method, period.size),
                        np.repeat(kind, period.size),
                        period,
                        theory_values,
                        tube_fraction_by_method[method][kind],
                    )
                )

    _style()
    figure, axes = plt.subplots(
        2,
        len(METHODS),
        figsize=(18.0, 6.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row, (kind, theory_values) in enumerate(
        (("group", theory_group), ("phase", theory_phase))
    ):
        for column, method in enumerate(METHODS):
            axis = axes[row, column]
            result = results[method]
            ftan_image = result.group_image if kind == "group" else result.phase_image
            image = axis.pcolormesh(
                period,
                ftan_image.velocity_mps,
                ftan_image.amplitude,
                shading="auto",
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                rasterized=True,
            )
            axis.plot(
                theory_period,
                theory_values,
                color="black",
                linewidth=3.2,
                alpha=0.75,
                solid_capstyle="round",
            )
            axis.plot(
                theory_period,
                theory_values,
                color="white",
                linestyle=(0, (4, 2)),
                linewidth=1.8,
                label=f"Theoretical {kind} velocity",
            )
            axis.set_title(LABELS[method])
            axis.set_xlabel("Period (s)")
            if column == 0:
                axis.set_ylabel(f"{kind.title()} velocity (m/s)")
            _panel(axis, row * len(METHODS) + column)
    colorbar = figure.colorbar(image, ax=axes, pad=0.01, shrink=0.9)
    colorbar.set_label("Column-normalized FTAN amplitude")
    figure.suptitle(
        f"Pick-free strict pair FTAN | pair {metadata['pair_i']}–{metadata['pair_j']} | "
        f"{metadata['distance_m']:.0f} m | Gaussian alpha={args.gaussian_alpha:g} | "
        "white dashed = independent 1-D theory",
        fontsize=10.5,
    )
    _save(figure, output / "ncf_selection_group_phase_ftan_energy")

    method_colors = {
        "active_reference": "#111111",
        "ideal_uniform_ncf": "#009E73",
        "biased_all_windows": "#7A7A7A",
        "data_only_retained": "#0072B2",
        "data_only_rejected": "#E69F00",
        "time_permuted_control": "#CC79A7",
    }
    figure, axes = plt.subplots(
        2, 3, figsize=(11.8, 6.0), sharex=True, sharey=True, constrained_layout=True
    )
    display = (
        common_time
        >= 0.85 * float(metadata["distance_m"]) / config.processing.velocity_max_mps
    ) & (
        common_time
        <= 1.08 * float(metadata["distance_m"]) / config.processing.velocity_min_mps
    )
    reference_display = reference_waveform[display]
    reference_display /= max(
        float(np.max(np.abs(reference_display))), np.finfo(float).eps
    )
    for index, (axis, method) in enumerate(zip(axes.flat, METHODS)):
        waveform = filtered_waveforms[method][display]
        waveform /= max(float(np.max(np.abs(waveform))), np.finfo(float).eps)
        if method != "active_reference":
            axis.plot(
                common_time[display],
                reference_display,
                color="#111111",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="Active EGF",
            )
        axis.plot(
            common_time[display],
            waveform,
            color=method_colors[method],
            linewidth=1.15,
            label=LABELS[method],
        )
        axis.axvspan(
            float(metadata["distance_m"]) / config.processing.velocity_max_mps,
            float(metadata["distance_m"]) / config.processing.velocity_min_mps,
            color="#D9D9D9",
            alpha=0.22,
            zorder=-2,
        )
        axis.set_title(
            f"{LABELS[method]}\n"
            f"CC={waveform_metrics[method]['waveform_correlation_to_active_egf']:.2f}"
        )
        axis.set_xlabel("Lag time (s)")
        if index % 3 == 0:
            axis.set_ylabel("Per-trace normalized amplitude")
        axis.legend(frameon=False, loc="lower right")
        _panel(axis, index)
    figure.suptitle(
        f"Strict two-station waveform QC | pair {metadata['pair_i']}–"
        f"{metadata['pair_j']} | {metadata['distance_m']:.0f} m | "
        f"{frequency_min:g}–{frequency_max:g} Hz",
        fontsize=10.5,
    )
    _save(figure, output / "ncf_selection_strict_pair_waveforms")

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    x = np.arange(len(METHODS))
    width = 0.36
    for axis, metric_prefix, title in (
        (axes[0], "theory_tube_energy_fraction", "Energy inside theory ± tube"),
        (axes[1], "image_correlation_to_active_egf", "FTAN image correlation to active EGF"),
    ):
        group_values = [metrics[name][f"median_group_{metric_prefix}"] if metric_prefix.startswith("theory") else metrics[name][f"group_{metric_prefix}"] for name in METHODS]
        phase_values = [metrics[name][f"median_phase_{metric_prefix}"] if metric_prefix.startswith("theory") else metrics[name][f"phase_{metric_prefix}"] for name in METHODS]
        axis.bar(x - width / 2, group_values, width=width, color="#0072B2", label="Group")
        axis.bar(x + width / 2, phase_values, width=width, color="#D55E00", label="Phase")
        axis.set_xticks(x, [LABELS[name] for name in METHODS], rotation=25, ha="right")
        axis.set_ylabel("Fraction" if metric_prefix.startswith("theory") else "Correlation")
        axis.set_title(title)
        axis.legend(loc="best")
    _panel(axes[0], 0)
    _panel(axes[1], 1)
    _save(figure, output / "ncf_selection_ftan_energy_metrics")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
