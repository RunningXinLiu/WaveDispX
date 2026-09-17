#!/usr/bin/env python3
"""Plot v5.0 source-event, waveform, group-FTAN, and phase-FTAN QC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wavedispx.processing import pair_ftan, theoretical_dispersion

from run_gate2_sparse_event_selection_experiment import (
    EVALUATION_BAND_HZ,
    PAIR_DISTANCE_M,
    VELOCITY_RANGE_MPS,
    _case_config,
)


METHODS = (
    ("active_reference", "Active EGF"),
    ("paired_clean_counterfactual", "Paired clean"),
    ("linear", "Linear"),
    ("rmsr_g5", "RMSR-SS"),
    ("robust_stackmaster", "Robust"),
    ("temporal_antisymmetry_adapter", "Temporal adapter"),
    ("nnws_upstream", "NNWS upstream"),
)
SOURCE_METHODS = (
    ("linear", "Raw event count", "effective"),
    ("rmsr_g5", "RMSR retained", "effective"),
    ("rmsr_g5", "RMSR rejected", "rejected"),
    ("robust_stackmaster", "Robust effective", "effective"),
    ("temporal_antisymmetry_adapter", "Temporal effective", "effective"),
    ("nnws_upstream", "NNWS effective", "effective"),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.7,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=450, bbox_inches="tight")
    plt.close(figure)


def _panel(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.09,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
    )


def _theory(case_dir: Path, seed: int):
    sources = np.genfromtxt(case_dir / "sources.csv", delimiter=",", names=True)
    config = _case_config(case_dir / "sources.csv", sources.size, case_dir, seed)
    frequency = np.linspace(EVALUATION_BAND_HZ[0], EVALUATION_BAND_HZ[1], 801)
    theory = theoretical_dispersion(
        config.model,
        frequency,
        rayleigh_modes=1,
        love_modes=0,
        backend="qedisp",
        compute_group=True,
    )
    if theory.rayleigh_group_mps is None:
        raise RuntimeError("QEDisp native group velocity is unavailable")
    period = 1.0 / frequency
    order = np.argsort(period)
    return (
        period[order],
        theory.rayleigh_group_mps[:, 0][order],
        theory.rayleigh_mps[:, 0][order],
    )


def _ftan_results(npz: np.lib.npyio.NpzFile):
    lag = npz["lag_time_s"]
    active_time = npz["active_reference_time_s"]
    active = npz["active_reference_zz"]
    common = {
        "distance_m": PAIR_DISTANCE_M,
        "frequency_range_hz": EVALUATION_BAND_HZ,
        "group_velocity_range_mps": VELOCITY_RANGE_MPS,
        "phase_velocity_range_mps": VELOCITY_RANGE_MPS,
        "period_count": 121,
        "velocity_step_mps": 2.0,
        "gaussian_alpha": 120.0,
        "phase_method": "frequency_domain",
        "minimum_snr": 0.0,
    }
    output = {
        "active_reference": pair_ftan(
            active,
            dt_s=float(np.median(np.diff(active_time))),
            input_kind="egf",
            **common,
        )
    }
    for method, _ in METHODS[1:]:
        key = f"stack__{method}__symmetric"
        if key not in npz.files:
            output[method] = None
            continue
        output[method] = pair_ftan(
            npz[key],
            dt_s=float(np.median(np.diff(lag))),
            input_kind="correlation",
            **common,
        )
    return output


def _bandpass_trace(trace: np.ndarray, dt_s: float) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    frequency = np.fft.rfftfreq(values.size, dt_s)
    spectrum = np.fft.rfft(values)
    spectrum[
        (frequency < EVALUATION_BAND_HZ[0])
        | (frequency > EVALUATION_BAND_HZ[1])
    ] = 0.0
    return np.fft.irfft(spectrum, n=values.size)


def _calibrated_waveforms(npz: np.lib.npyio.NpzFile):
    lag = npz["lag_time_s"]
    lag_dt = float(np.median(np.diff(lag)))
    active_time = npz["active_reference_time_s"]
    active_dt = float(np.median(np.diff(active_time)))
    output = {
        "active_reference": (
            active_time,
            _bandpass_trace(npz["active_reference_zz"], active_dt),
        )
    }
    for method, _ in METHODS[1:]:
        key = f"stack__{method}__symmetric"
        if key not in npz.files:
            output[method] = None
            continue
        # This is the explicit interferometric calibration used by the
        # active-EGF waveform metrics.  FTAN below retains its independently
        # validated correlation-input convention.
        calibrated = -np.gradient(npz[key], lag_dt)
        output[method] = (lag, _bandpass_trace(calibrated, lag_dt))
    return output


def _waveform_ftan_figure(
    run_root: Path,
    case_dir: Path,
    scenario: str,
    seed: int,
    output: Path,
) -> None:
    with np.load(case_dir / "gate2_sparse_event_selection.npz") as npz:
        results = _ftan_results(npz)
        waveforms = _calibrated_waveforms(npz)
    theory_period, theory_group, theory_phase = _theory(case_dir, seed)
    metrics = pd.read_csv(run_root / "gate2_sparse_event_metrics.csv")
    metrics = metrics[
        (metrics.wavefield_seed == seed)
        & (metrics.scenario == scenario)
        & (metrics.control_replica == -1)
        & (metrics.branch == "symmetric")
    ]

    figure, axes = plt.subplots(
        3,
        len(METHODS),
        figsize=(15.0, 6.5),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )
    active_time, active_trace = waveforms["active_reference"]
    active_wave = active_trace / max(
        np.max(np.abs(active_trace)), np.finfo(float).eps
    )
    image = None
    for column, (method, label) in enumerate(METHODS):
        result = results.get(method)
        if result is None:
            message = (
                "not applicable\n(uniform control)"
                if scenario == "uniform" and method == "paired_clean_counterfactual"
                else "method failed\n(no substituted trace)"
            )
            color = "#555555" if message.startswith("not applicable") else "#B2182B"
            for row in range(3):
                axes[row, column].text(
                    0.5,
                    0.5,
                    message,
                    ha="center",
                    va="center",
                    transform=axes[row, column].transAxes,
                    color=color,
                )
            axes[0, column].set_title(label)
            continue
        wave_time, wave_trace = waveforms[method]
        wave = wave_trace / max(
            np.max(np.abs(wave_trace)), np.finfo(float).eps
        )
        axes[0, column].plot(wave_time, wave, color="#0072B2", linewidth=0.75)
        common_count = min(wave_time.size, active_wave.size)
        axes[0, column].plot(
            active_time[:common_count],
            active_wave[:common_count],
            color="#222222",
            linewidth=0.55,
            alpha=0.65,
            linestyle="--",
        )
        axes[0, column].axvspan(
            PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[1],
            PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[0],
            color="#F0E442",
            alpha=0.12,
        )
        axes[0, column].set_xlim(0.0, 5.25)
        axes[0, column].set_ylim(-1.05, 1.05)
        title = label
        if method != "active_reference":
            row = metrics[metrics.method == method]
            if not row.empty and row.iloc[0].status == "complete":
                title += f"\nCC={row.iloc[0].waveform_correlation_no_shift:.3f}"
        axes[0, column].set_title(title)

        image = axes[1, column].pcolormesh(
            result.group_image.period_s,
            result.group_image.velocity_mps,
            result.group_image.amplitude,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        axes[1, column].plot(theory_period, theory_group, color="cyan", linewidth=1.1)
        axes[2, column].pcolormesh(
            result.phase_image.period_s,
            result.phase_image.velocity_mps,
            result.phase_image.amplitude,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        axes[2, column].plot(theory_period, theory_phase, color="cyan", linewidth=1.1)
        axes[2, column].set_xlabel("Period (s)")
    axes[0, 0].set_ylabel("Normalized waveform")
    axes[1, 0].set_ylabel("Group velocity (m/s)")
    axes[2, 0].set_ylabel("Phase velocity (m/s)")
    axes[1, 0].set_ylim(VELOCITY_RANGE_MPS)
    axes[2, 0].set_ylim(VELOCITY_RANGE_MPS)
    for row, label in enumerate(("a", "b", "c")):
        _panel(axes[row, 0], label)
    figure.suptitle(
        f"Gate 2 v5 sparse-event QC | {scenario} | seed {seed}\n"
        "Waveform: 4–12 Hz and −d/dt NCF calibration; FTAN: per-period normalized; cyan = QEDisp fundamental Rayleigh",
        fontsize=10.0,
    )
    if image is not None:
        figure.colorbar(
            image,
            ax=axes[1:, :],
            label="Per-period normalized energy",
            shrink=0.7,
        )
    _save(figure, output / f"{scenario}_seed{seed}_waveform_ftan_normalized")


def _ftan_common_scale_figure(
    case_dir: Path,
    scenario: str,
    seed: int,
    output: Path,
) -> None:
    with np.load(case_dir / "gate2_sparse_event_selection.npz") as npz:
        results = _ftan_results(npz)
    theory_period, theory_group, theory_phase = _theory(case_dir, seed)
    active = results["active_reference"]
    calibrated: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for method, _ in METHODS:
        result = results.get(method)
        if result is None:
            calibrated[method] = None
            continue
        common_count = min(result.processed_trace.size, active.processed_trace.size)
        candidate = result.processed_trace[:common_count]
        target = active.processed_trace[:common_count]
        denominator = float(np.dot(candidate, candidate))
        scale = (
            float(np.dot(candidate, target) / denominator)
            if denominator > np.finfo(float).eps
            else 0.0
        )
        group_raw = np.abs(scale) * np.asarray(result.group_image.raw_amplitude)
        phase_raw = np.maximum(
            scale * np.asarray(result.phase_image.raw_amplitude),
            0.0,
        )
        calibrated[method] = (group_raw, phase_raw)
    group_vmax = max(
        float(np.max(values[0]))
        for values in calibrated.values()
        if values is not None
    )
    phase_vmax = max(
        float(np.max(values[1]))
        for values in calibrated.values()
        if values is not None
    )

    figure, axes = plt.subplots(
        2,
        len(METHODS),
        figsize=(15.0, 4.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    group_image = None
    phase_image = None
    for column, (method, label) in enumerate(METHODS):
        result = results.get(method)
        values = calibrated.get(method)
        axes[0, column].set_title(label)
        if result is None or values is None:
            message = (
                "not applicable\n(uniform control)"
                if scenario == "uniform" and method == "paired_clean_counterfactual"
                else "method failed"
            )
            color = "#555555" if message.startswith("not applicable") else "#B2182B"
            for row in range(2):
                axes[row, column].text(
                    0.5,
                    0.5,
                    message,
                    transform=axes[row, column].transAxes,
                    ha="center",
                    va="center",
                    color=color,
                )
            continue
        group_image = axes[0, column].pcolormesh(
            result.group_image.period_s,
            result.group_image.velocity_mps,
            values[0],
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=group_vmax,
            rasterized=True,
        )
        phase_image = axes[1, column].pcolormesh(
            result.phase_image.period_s,
            result.phase_image.velocity_mps,
            values[1],
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=phase_vmax,
            rasterized=True,
        )
        axes[0, column].plot(theory_period, theory_group, color="cyan", linewidth=1.1)
        axes[1, column].plot(theory_period, theory_phase, color="cyan", linewidth=1.1)
        axes[1, column].set_xlabel("Period (s)")
    axes[0, 0].set_ylabel("Group velocity (m/s)")
    axes[1, 0].set_ylabel("Phase velocity (m/s)")
    axes[0, 0].set_ylim(VELOCITY_RANGE_MPS)
    _panel(axes[0, 0], "a")
    _panel(axes[1, 0], "b")
    if group_image is not None:
        figure.colorbar(group_image, ax=axes[0, :], label="Common-scale group amplitude", shrink=0.72)
    if phase_image is not None:
        figure.colorbar(phase_image, ax=axes[1, :], label="Common-scale positive phase amplitude", shrink=0.72)
    figure.suptitle(
        f"Gate 2 v5 common-scale FTAN | {scenario} | seed {seed}\n"
        "One global least-squares trace scale per method and one color scale per row; one-bit data do not preserve physical amplitude",
        fontsize=10.0,
    )
    _save(figure, output / f"{scenario}_seed{seed}_ftan_common_scale")


def _aggregate_site_values(events: pd.DataFrame, mode: str) -> pd.DataFrame:
    values = events.copy()
    if mode == "effective":
        values["display_weight"] = values.window_weight
    elif mode == "rejected":
        values["display_weight"] = 1.0 - values.window_weight
    else:
        raise ValueError(mode)
    return (
        values.groupby(["source_index", "source_group", "x_m", "y_m", "azimuth_deg"])
        .display_weight.sum()
        .reset_index()
    )


def _source_figure(
    run_root: Path,
    scenario: str,
    seed: int,
    output: Path,
) -> None:
    events = pd.read_csv(run_root / "gate2_sparse_event_effective_events.csv")
    events = events[(events.wavefield_seed == seed) & (events.scenario == scenario)]
    weights = pd.read_csv(run_root / "gate2_sparse_event_window_weights.csv")
    weights = weights[(weights.wavefield_seed == seed) & (weights.scenario == scenario)]
    figure, axes = plt.subplots(2, 4, figsize=(12.2, 6.3), constrained_layout=True)
    flat_axes = axes.reshape(-1)
    scatter = None
    for panel_index, (method, title, mode) in enumerate(SOURCE_METHODS):
        axis = flat_axes[panel_index]
        selected = events[events.method == method]
        if selected.empty:
            axis.text(0.5, 0.5, "method failed", transform=axis.transAxes, ha="center")
            axis.set_title(title)
            continue
        sites = _aggregate_site_values(selected, mode)
        normalized = sites.display_weight / max(sites.display_weight.max(), 1e-12)
        scatter = axis.scatter(
            sites.x_m,
            sites.y_m,
            c=normalized,
            s=4.0 + 24.0 * normalized,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.8,
            linewidths=0.0,
            rasterized=True,
        )
        # CSV round-tripping stores the registered groups as 0=background,
        # 1=sector.  Accept the explicit text label as well for hand-built
        # audit tables.
        sector_label = sites.source_group.astype(str)
        sector_sites = sites[sector_label.isin(("1", "sector"))]
        if not sector_sites.empty:
            axis.scatter(
                sector_sites.x_m,
                sector_sites.y_m,
                facecolors="none",
                edgecolors="#D62728",
                s=34,
                linewidths=0.45,
                label="sector sites" if panel_index == 0 else None,
                rasterized=True,
            )
        axis.scatter((-300.0, 300.0), (0.0, 0.0), marker="v", s=28, c="#D55E00")
        axis.set_aspect("equal")
        axis.set_xlim(-1900, 1900)
        axis.set_ylim(-1900, 1900)
        axis.set_title(title)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        if panel_index == 0 and not sector_sites.empty:
            axis.legend(frameon=False, loc="lower left")
        _panel(axis, chr(ord("a") + panel_index))

    azimuth_axis = flat_axes[6]
    bins = np.linspace(0.0, 360.0, 37)
    for method, label, color in (
        ("linear", "raw", "#555555"),
        ("rmsr_g5", "RMSR", "#0072B2"),
        ("nnws_upstream", "NNWS", "#D55E00"),
    ):
        selected = events[events.method == method]
        if selected.empty:
            continue
        hist, edges = np.histogram(
            selected.azimuth_deg,
            bins=bins,
            weights=selected.window_weight,
        )
        hist = hist / max(hist.sum(), 1e-12)
        azimuth_axis.step(edges[:-1], hist, where="post", label=label, color=color)
    azimuth_axis.set_xlabel("Source azimuth (deg)")
    azimuth_axis.set_ylabel("Effective event fraction")
    azimuth_axis.legend(frameon=False)
    _panel(azimuth_axis, "g")

    relation_axis = flat_axes[7]
    for method, label, color in (
        ("rmsr_g5", "RMSR", "#0072B2"),
        ("temporal_antisymmetry_adapter", "temporal", "#009E73"),
        ("nnws_upstream", "NNWS", "#D55E00"),
    ):
        selected = weights[weights.method == method]
        if selected.empty:
            continue
        grouped = selected.groupby("sector_event_count").weight.agg(["mean", "count"])
        overall_mean = max(float(selected.weight.mean()), np.finfo(float).eps)
        relation_axis.plot(
            grouped.index,
            grouped["mean"] / overall_mean,
            marker="o",
            ms=3,
            label=label,
            color=color,
        )
    relation_axis.set_xlabel("Sector events in window")
    relation_axis.set_ylabel("Mean weight / method mean")
    relation_axis.legend(frameon=False)
    _panel(relation_axis, "h")
    if scatter is not None:
        figure.colorbar(scatter, ax=flat_axes[:6].tolist(), label="Normalized site contribution", shrink=0.75)
    figure.suptitle(
        f"Gate 2 v5 source-event audit | {scenario} | seed {seed}\n"
        "Continuous weights are shown as effective contribution; only RMSR has literal retained/rejected events",
        fontsize=10.0,
    )
    _save(figure, output / f"{scenario}_seed{seed}_source_event_audit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("output_python/gate2_sparse_event_development"),
    )
    parser.add_argument("--seed", type=int, default=5301)
    parser.add_argument(
        "--scenarios",
        default="uniform,far75,near12,mirror168",
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_directory.resolve()
    output = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else run_root / "figures"
    )
    scenarios = tuple(item.strip() for item in args.scenarios.split(",") if item.strip())
    _style()
    for scenario in scenarios:
        case_dir = run_root / f"seed_{args.seed}" / scenario
        _waveform_ftan_figure(run_root, case_dir, scenario, args.seed, output)
        _ftan_common_scale_figure(case_dir, scenario, args.seed, output)
        _source_figure(run_root, scenario, args.seed, output)
        print(f"completed QC figures for {scenario}, seed={args.seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
