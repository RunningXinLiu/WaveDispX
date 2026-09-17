#!/usr/bin/env python3
"""QC the dense-azimuth ambient-source experiment against a theoretical EGF.

The data-only selector is judged by source-bias detection, calibrated waveform
agreement, NCF moveout, dispersion-energy agreement, and theory-referenced
phase-velocity picks.  Known-source masks are shown only as synthetic Oracles;
the time-permuted mask is the negative control.
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
from matplotlib.patches import Wedge
import numpy as np

from wavedispx.nnws_qc import (
    _dispersion_similarity,
    calibrated_waveform_comparison,
    reference_dispersion_for_strategy,
)
from wavedispx.processing import surface_wave_dispersion


STRATEGIES = (
    "baseline_linear",
    "nnws_v2_tau_p_directional_guard_frequency",
    "nnws_v2_tau_p_directional_guard_frequency_permuted_time_control",
    "pair_crossfit_tau_p_diffuse_soft_frequency",
    "pair_crossfit_tau_p_diffuse_hard_frequency",
    "pair_crossfit_tau_p_diffuse_hard_frequency_permuted_time_control",
    "oracle_hard_mask",
)
LABELS = {
    "reference": "Theoretical EGF",
    "baseline_linear": "Linear (all windows)",
    "nnws_v2_tau_p_directional_guard_frequency": "V2 + diffuse tau-p guard",
    "nnws_v2_tau_p_directional_guard_frequency_permuted_time_control": (
        "V2 + diffuse guard, time-permuted"
    ),
    "pair_crossfit_tau_p_diffuse_soft_frequency": "Data-only soft",
    "pair_crossfit_tau_p_diffuse_hard_frequency": "Data-only hard",
    "pair_crossfit_tau_p_diffuse_hard_frequency_permuted_time_control": (
        "Hard mask, time-permuted"
    ),
    "oracle_hard_mask": "Known-source Oracle mask",
}
SHORT_LABELS = {
    "baseline_linear": "Linear",
    "nnws_v2_tau_p_directional_guard_frequency": "V2 diffuse",
    "nnws_v2_tau_p_directional_guard_frequency_permuted_time_control": (
        "V2 perm."
    ),
    "pair_crossfit_tau_p_diffuse_soft_frequency": "Soft",
    "pair_crossfit_tau_p_diffuse_hard_frequency": "Hard",
    "pair_crossfit_tau_p_diffuse_hard_frequency_permuted_time_control": (
        "Permuted"
    ),
    "oracle_hard_mask": "Oracle",
}
COLORS = {
    "reference": "#111111",
    "baseline_linear": "#7A7A7A",
    "nnws_v2_tau_p_directional_guard_frequency": "#CC79A7",
    "nnws_v2_tau_p_directional_guard_frequency_permuted_time_control": (
        "#D55E00"
    ),
    "pair_crossfit_tau_p_diffuse_soft_frequency": "#56B4E9",
    "pair_crossfit_tau_p_diffuse_hard_frequency": "#0072B2",
    "pair_crossfit_tau_p_diffuse_hard_frequency_permuted_time_control": (
        "#E69F00"
    ),
    "oracle_hard_mask": "#009E73",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.3,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def _save(figure: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    tiff_path = stem.with_suffix(".tiff")
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(tiff_path, dpi=600, bbox_inches="tight")
    outputs = [svg_path, pdf_path, png_path, tiff_path]
    plt.close(figure)
    return outputs


def _activity(path: Path, window_count: int, source_count: int) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True)
    output = np.zeros((window_count, source_count), dtype=np.float64)
    for row in np.atleast_1d(table):
        output[int(row["window_index"]), int(row["source_index"])] = float(
            row["amplitude_multiplier"]
        )
    return output


def _load_selector(run: Path) -> dict[str, np.ndarray | float]:
    resolved = json.loads((run / "resolved_config.json").read_text())
    with h5py.File(run / "results.h5") as handle:
        imbalance = handle["pair_crossfit_tau_p_spz/directional_imbalance"][:]
        soft = handle["pair_crossfit_tau_p_spz/diffuse_soft_band_weights"][:]
        hard = handle["pair_crossfit_tau_p_spz/diffuse_hard_band_weights"][:]
        band_center = handle["pair_crossfit_tau_p_spz/band_center_hz"][:]
        oracle = handle["source_selection/hard_selected"][:].astype(bool)
    source_count = np.loadtxt(run / "sources.csv", delimiter=",", skiprows=1).shape[0]
    activity = _activity(
        Path(resolved["ambient_sources"]["activity_file"]),
        hard.shape[0],
        source_count,
    )
    background_count = int(
        resolved["ambient_sources"].get("directional_background_source_count", 144)
    )
    if not 0 < background_count < source_count:
        background_count = 144
    background_amplitude = np.median(activity[:, :background_count], axis=1)
    sector_amplitude = np.divide(
        np.max(activity[:, background_count:], axis=1),
        background_amplitude,
        out=np.zeros(hard.shape[0], dtype=np.float64),
        where=background_amplitude > np.finfo(float).eps,
    )
    hard_window = np.mean(hard, axis=1) > 0.5
    return {
        "activity": activity,
        "imbalance": imbalance,
        "soft": soft,
        "hard": hard,
        "hard_window": hard_window,
        "band_center": band_center,
        "oracle": oracle,
        "sector_amplitude": sector_amplitude,
        "accuracy": float(np.mean(hard_window == oracle)),
        "background_count": background_count,
    }


def plot_source_selector_audit(run: Path, output: Path) -> tuple[list[Path], dict]:
    _style()
    resolved = json.loads((run / "resolved_config.json").read_text())
    sources = np.loadtxt(run / "sources.csv", delimiter=",", skiprows=1)
    receivers = np.loadtxt(run / "receivers.csv", delimiter=",", skiprows=1)
    selector = _load_selector(run)
    background_count = int(selector["background_count"])
    window = np.arange(len(selector["sector_amplitude"]))
    figure, axes = plt.subplots(
        2, 2, figsize=(10.8, 7.5), constrained_layout=True
    )

    maximum_radius = 1.06 * float(np.max(np.hypot(sources[:, 0], sources[:, 1])))
    axes[0, 0].add_patch(
        Wedge(
            (0, 0),
            maximum_radius,
            35,
            55,
            width=0.22 * maximum_radius,
            color="#E69F00",
            alpha=0.14,
            linewidth=0,
        )
    )
    axes[0, 0].scatter(
        sources[:background_count, 0],
        sources[:background_count, 1],
        s=10,
        color="#8D8D8D",
        alpha=0.72,
        linewidth=0,
        label=f"annular background (n={background_count})",
    )
    axes[0, 0].scatter(
        sources[background_count:, 0],
        sources[background_count:, 1],
        s=13,
        color="#D55E00",
        alpha=0.86,
        linewidth=0,
        label=f"extra +45 deg sector (n={len(sources) - background_count})",
    )
    axes[0, 0].plot(receivers[:, 0], receivers[:, 1], color="#111111", lw=1.2)
    axes[0, 0].scatter(
        receivers[[0, -1], 0],
        receivers[[0, -1], 1],
        marker="v",
        s=35,
        color="#111111",
        zorder=4,
        label="Receiver line (1 km)",
    )
    axes[0, 0].annotate(
        "temporally varying\ndense sector",
        xy=(0.69 * maximum_radius, 0.69 * maximum_radius),
        xytext=(0.10 * maximum_radius, 0.76 * maximum_radius),
        arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 0.9},
        color="#9C3F00",
    )
    axes[0, 0].set(
        aspect="equal",
        xlim=(-maximum_radius, maximum_radius),
        ylim=(-maximum_radius, maximum_radius),
        xlabel="x (m)",
        ylabel="y (m)",
        title="Many-source field with one over-represented azimuth",
    )
    axes[0, 0].legend(frameon=False, loc="lower left")

    state_colors = {0.0: "#BDBDBD", 0.75: "#56B4E9", 1.5: "#E69F00", 2.5: "#D55E00"}
    axes[0, 1].step(
        window,
        selector["sector_amplitude"],
        where="mid",
        color="#333333",
        lw=0.8,
    )
    for state, color in state_colors.items():
        chosen = np.isclose(selector["sector_amplitude"], state)
        axes[0, 1].scatter(
            window[chosen],
            selector["sector_amplitude"][chosen],
            s=15,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            label=f"sector amplitude = {state:g}",
            zorder=3,
        )
    axes[0, 1].set(
        xlabel="25-s window index",
        ylabel="extra-sector amplitude multiplier",
        title="Controlled temporal change in azimuthal source density",
    )
    axes[0, 1].legend(frameon=False, ncol=2)

    mean_imbalance = np.mean(selector["imbalance"], axis=1)
    for state, color in state_colors.items():
        state_window = np.isclose(selector["sector_amplitude"], state)
        retained_window = state_window & selector["hard_window"]
        rejected_window = state_window & ~selector["hard_window"]
        axes[1, 0].scatter(
            window[retained_window],
            mean_imbalance[retained_window],
            s=26,
            marker="o",
            facecolor=color,
            edgecolor=color,
            linewidth=0.75,
            label=f"state {state:g}",
        )
        axes[1, 0].scatter(
            window[rejected_window],
            mean_imbalance[rejected_window],
            s=16,
            marker="o",
            facecolor="none",
            edgecolor=color,
            linewidth=0.75,
        )
    selected_max = float(np.max(mean_imbalance[selector["hard_window"]]))
    axes[1, 0].axhline(
        selected_max,
        color="#0072B2",
        ls="--",
        lw=0.8,
        label="Hard-selection rank cutoff",
    )
    axes[1, 0].set(
        xlabel="25-s window index",
        ylabel="mean bilateral directional imbalance",
        title=(
            "Data observable tracks the imposed source asymmetry\n"
            "filled markers = retained by data-only hard selector"
        ),
    )
    axes[1, 0].legend(frameon=False, ncol=2)

    states = np.asarray(sorted(state_colors))
    retained = np.asarray(
        [
            np.mean(
                selector["hard_window"][
                    np.isclose(selector["sector_amplitude"], state)
                ]
            )
            for state in states
        ]
    )
    oracle_retained = np.asarray(
        [
            np.mean(
                selector["oracle"][
                    np.isclose(selector["sector_amplitude"], state)
                ]
            )
            for state in states
        ]
    )
    x = np.arange(states.size)
    axes[1, 1].bar(
        x - 0.18,
        retained,
        width=0.36,
        color="#0072B2",
        label="Data-only hard",
    )
    axes[1, 1].bar(
        x + 0.18,
        oracle_retained,
        width=0.36,
        facecolor="none",
        edgecolor="#009E73",
        linewidth=1.1,
        label="Known-source Oracle",
    )
    axes[1, 1].set(
        xticks=x,
        xticklabels=[f"{state:g}" for state in states],
        ylim=(0, 1.08),
        xlabel="extra-sector amplitude multiplier",
        ylabel="retained-window fraction",
        title=f"Mask agreement with source-known Oracle = {selector['accuracy']:.1%}",
    )
    axes[1, 1].legend(frameon=False)
    for axis, label in zip(axes.flat, "abcd"):
        axis.grid(alpha=0.13)
        _panel(axis, label)
    figure.suptitle(
        "Directional-density benchmark and selector audit",
        fontsize=11,
    )
    files = _save(figure, output / "directional_density_source_selector_audit")
    summary = {
        "window_count": int(len(window)),
        "source_count": int(len(sources)),
        "background_source_count": background_count,
        "dense_sector_source_count": int(len(sources) - background_count),
        "hard_mask_oracle_accuracy": float(selector["accuracy"]),
        "hard_retention_by_sector_amplitude": {
            f"{state:g}": float(value) for state, value in zip(states, retained)
        },
        "oracle_retention_by_sector_amplitude": {
            f"{state:g}": float(value)
            for state, value in zip(states, oracle_retained)
        },
    }
    return files, summary


def plot_retained_rejected_source_contributions(
    run: Path, output: Path
) -> tuple[list[Path], dict, Path]:
    """Trace hard-selected windows back to source-power contributions."""
    _style()
    sources = np.loadtxt(run / "sources.csv", delimiter=",", skiprows=1)
    receivers = np.loadtxt(run / "receivers.csv", delimiter=",", skiprows=1)
    selector = _load_selector(run)
    activity = np.asarray(selector["activity"], dtype=np.float64)
    retained_window = np.asarray(selector["hard_window"], dtype=bool)
    background_count = int(selector["background_count"])
    source_power = np.square(activity)
    total_power = np.sum(source_power, axis=0)
    retained_power = np.sum(source_power[retained_window], axis=0)
    rejected_power = np.sum(source_power[~retained_window], axis=0)
    rejected_fraction = np.divide(
        rejected_power,
        total_power,
        out=np.zeros_like(rejected_power),
        where=total_power > 0,
    )
    maximum_power = max(float(np.max(total_power)), np.finfo(float).eps)
    maximum_radius = 1.06 * float(np.max(np.hypot(sources[:, 0], sources[:, 1])))
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)

    def power_map(axis: plt.Axes, values: np.ndarray, title: str) -> None:
        axis.add_patch(
            Wedge(
                (0, 0),
                maximum_radius,
                35,
                55,
                width=0.22 * maximum_radius,
                color="#E69F00",
                alpha=0.12,
                linewidth=0,
            )
        )
        sizes = 7.0 + 70.0 * values / maximum_power
        axis.scatter(
            sources[:background_count, 0],
            sources[:background_count, 1],
            s=sizes[:background_count],
            color="#8D8D8D",
            alpha=0.68,
            linewidth=0,
            label="Annular background",
        )
        axis.scatter(
            sources[background_count:, 0],
            sources[background_count:, 1],
            s=sizes[background_count:],
            color="#D55E00",
            alpha=0.82,
            linewidth=0,
            label="Dense +45 deg sector",
        )
        axis.plot(receivers[:, 0], receivers[:, 1], color="#111111", lw=1.0)
        axis.scatter(
            receivers[[0, -1], 0],
            receivers[[0, -1], 1],
            marker="v",
            s=30,
            color="#111111",
            zorder=4,
        )
        axis.set(
            aspect="equal",
            xlim=(-maximum_radius, maximum_radius),
            ylim=(-maximum_radius, maximum_radius),
            xlabel="x (m)",
            ylabel="y (m)",
            title=title,
        )

    power_map(
        axes[0, 0],
        total_power,
        "All source contribution\nmarker area scales with cumulative source power",
    )
    power_map(
        axes[0, 1],
        retained_power,
        "Retained source contribution\nlow directional-imbalance windows",
    )
    power_map(
        axes[1, 0],
        rejected_power,
        "Rejected source contribution\nhigh directional-imbalance windows",
    )
    axes[0, 0].legend(frameon=False, loc="lower left")

    source_classes = ("Background", "Dense sector")
    class_indices = (
        np.arange(background_count),
        np.arange(background_count, len(sources)),
    )
    class_retained_fraction = []
    class_rejected_fraction = []
    for indices in class_indices:
        class_total = float(np.sum(total_power[indices]))
        class_retained_fraction.append(float(np.sum(retained_power[indices])) / class_total)
        class_rejected_fraction.append(float(np.sum(rejected_power[indices])) / class_total)
    x = np.arange(len(source_classes))
    axes[1, 1].bar(
        x,
        class_retained_fraction,
        color="#0072B2",
        width=0.58,
        label="Retained power",
    )
    axes[1, 1].bar(
        x,
        class_rejected_fraction,
        bottom=class_retained_fraction,
        color="#E69F00",
        width=0.58,
        label="Rejected power",
    )
    for index, fraction in enumerate(class_rejected_fraction):
        axes[1, 1].text(
            index,
            1.025,
            f"{fraction:.1%} rejected",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[1, 1].set(
        xticks=x,
        xticklabels=source_classes,
        ylim=(0, 1.11),
        ylabel="fraction of cumulative source power",
        title=(
            "Power-budget consequence of window rejection\n"
            "amplitude-squared weighting matches correlation power"
        ),
    )
    axes[1, 1].legend(frameon=False, loc="center left")
    for axis, label in zip(axes.flat, "abcd"):
        axis.grid(alpha=0.13)
        _panel(axis, label)
    figure.suptitle(
        "Retained and rejected source-power provenance",
        fontsize=11,
    )
    files = _save(figure, output / "directional_density_retained_rejected_sources")

    source_csv = output / "directional_density_source_contribution_audit.csv"
    with source_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "source_index",
                "source_class",
                "x_m",
                "y_m",
                "total_power_exposure",
                "retained_power_exposure",
                "rejected_power_exposure",
                "rejected_power_fraction",
            ),
        )
        writer.writeheader()
        for source_index in range(len(sources)):
            writer.writerow(
                {
                    "source_index": source_index,
                    "source_class": (
                        "background"
                        if source_index < background_count
                        else "dense_sector"
                    ),
                    "x_m": sources[source_index, 0],
                    "y_m": sources[source_index, 1],
                    "total_power_exposure": total_power[source_index],
                    "retained_power_exposure": retained_power[source_index],
                    "rejected_power_exposure": rejected_power[source_index],
                    "rejected_power_fraction": rejected_fraction[source_index],
                }
            )
    summary = {
        "power_definition": "sum_over_windows(amplitude_multiplier_squared)",
        "retained_window_count": int(np.count_nonzero(retained_window)),
        "rejected_window_count": int(np.count_nonzero(~retained_window)),
        "background_rejected_power_fraction": class_rejected_fraction[0],
        "dense_sector_rejected_power_fraction": class_rejected_fraction[1],
        "source_contribution_csv": str(source_csv),
    }
    return files, summary, source_csv


def _retained_rejected_ncf(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    total_count = int(
        manifest["ncf_strategies"]["baseline_linear"]["nonzero_window_count"]
    )
    retained_count = int(
        manifest["ncf_strategies"][
            "pair_crossfit_tau_p_diffuse_hard_frequency"
        ]["nonzero_window_count"]
    )
    rejected_count = total_count - retained_count
    if retained_count <= 0 or rejected_count <= 0:
        raise ValueError("retained/rejected NCF decomposition requires both groups")
    with h5py.File(run / "results.h5") as handle:
        baseline_group = handle["ncf/strategies/baseline_linear"]
        retained_group = handle[
            "ncf/strategies/pair_crossfit_tau_p_diffuse_hard_frequency"
        ]
        lag = baseline_group["lag_time_s"][:].astype(np.float64)
        distance = baseline_group["distance_m"][:].astype(np.float64)
        retained_distance = retained_group["distance_m"][:].astype(np.float64)
        if not np.array_equal(distance, retained_distance):
            raise ValueError("baseline and retained NCF distance supports differ")
        baseline = baseline_group["components/zz/symmetric"][:].astype(np.float64)
        retained = retained_group["components/zz/symmetric"][:].astype(np.float64)
    rejected = (total_count * baseline - retained_count * retained) / rejected_count
    reconstruction = (
        retained_count * retained + rejected_count * rejected
    ) / total_count
    if not np.allclose(reconstruction, baseline, rtol=1e-11, atol=1e-12):
        raise AssertionError("retained/rejected NCFs do not reconstruct the baseline")
    return lag, distance, baseline, retained, rejected, total_count, retained_count


def plot_retained_rejected_waveforms(run: Path, output: Path) -> list[Path]:
    _style()
    resolved = json.loads((run / "resolved_config.json").read_text())
    processing = resolved["processing"]
    band = (
        float(processing["frequency_min_hz"]),
        float(processing["frequency_max_hz"]),
    )
    lag, distance, baseline, reference = calibrated_waveform_comparison(
        run / "results.h5", "baseline_linear", "symmetric", frequency_range_hz=band
    )
    retained_lag, retained_distance, retained, _ = calibrated_waveform_comparison(
        run / "results.h5",
        "pair_crossfit_tau_p_diffuse_hard_frequency",
        "symmetric",
        frequency_range_hz=band,
    )
    if not np.array_equal(lag, retained_lag) or not np.array_equal(
        distance, retained_distance
    ):
        raise ValueError("calibrated retained and baseline NCF supports differ")
    _, _, _, _, _, total_count, retained_count = _retained_rejected_ncf(run)
    rejected_count = total_count - retained_count
    rejected = (total_count * baseline - retained_count * retained) / rejected_count
    targets = np.asarray((200.0, 400.0, 600.0))
    indices = np.abs(distance[:, None] - targets[None, :]).argmin(axis=0)
    vmin = float(processing["velocity_min_mps"])
    vmax = float(processing["velocity_max_mps"])
    curves = (
        ("Theoretical EGF", reference, "#111111", "--", 1.35),
        ("All windows", baseline, "#7A7A7A", "-", 0.9),
        ("Retained contribution", retained, "#0072B2", "-", 1.25),
        ("Rejected contribution", rejected, "#E69F00", "-", 1.0),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), sharey=True)
    for axis, distance_index, label in zip(axes, indices, "abc"):
        arrival_start = distance[distance_index] / vmax
        arrival_end = distance[distance_index] / vmin
        display = (lag >= 0.8 * arrival_start) & (lag <= 1.05 * arrival_end)
        axis.axvspan(arrival_start, arrival_end, color="#DDDDDD", alpha=0.5)
        for curve_label, values, color, linestyle, linewidth in curves:
            curve = values[:, distance_index]
            scale = max(float(np.max(np.abs(curve[display]))), np.finfo(float).eps)
            axis.plot(
                lag,
                curve / scale,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=curve_label,
            )
        axis.set(
            xlim=(max(0.0, 0.8 * arrival_start), min(float(lag[-1]), 1.05 * arrival_end)),
            ylim=(-1.15, 1.15),
            xlabel="lag time (s)",
            title=f"distance = {distance[distance_index]:.0f} m",
        )
        axis.grid(alpha=0.13)
        _panel(axis, label)
    axes[0].set_ylabel("per-trace normalized amplitude")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(
        "Retained-only and rejected-only symmetric NCF contributions",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0.0, 0.15, 1.0, 0.94))
    return _save(figure, output / "directional_density_retained_rejected_waveforms")


def plot_retained_rejected_dispersion(
    run: Path, output: Path
) -> tuple[list[Path], dict]:
    _style()
    resolved = json.loads((run / "resolved_config.json").read_text())
    processing = resolved["processing"]
    lag, distance, _, _, rejected, total_count, retained_count = (
        _retained_rejected_ncf(run)
    )
    reference, reference_quality = reference_dispersion_for_strategy(
        run, "baseline_linear"
    )
    with h5py.File(run / "results.h5") as handle:
        baseline_group = handle["dispersion/strategies/baseline_linear"]
        retained_group = handle[
            "dispersion/strategies/pair_crossfit_tau_p_diffuse_hard_frequency"
        ]
        velocity = baseline_group["velocity_mps"][:]
        frequency = baseline_group["frequency_hz"][:]
        baseline_energy = baseline_group["energy"][:]
        retained_energy = retained_group["energy"][:]
        theory_frequency = handle["theory/frequency_hz"][:]
        theory_velocity = handle["theory/rayleigh_mps"][:]
    rejected_image = surface_wave_dispersion(
        rejected,
        distance,
        float(lag[1] - lag[0]),
        velocity,
        frequency_min_hz=float(processing["frequency_min_hz"]),
        frequency_max_hz=float(processing["frequency_max_hz"]),
        method=str(processing["dispersion_method"]),
        zero_padding=float(processing["zero_padding"]),
        distance_taper_fraction=float(processing["distance_taper_fraction"]),
        frequency_block_size=8,
        radial_kernel_lookup_step=float(processing["radial_kernel_lookup_step"]),
        workers=0,
    )
    if not np.allclose(rejected_image.frequency_hz, frequency):
        raise ValueError("rejected and saved dispersion frequency grids differ")
    energies = {
        "Theoretical EGF": reference.energy,
        "All windows": baseline_energy,
        "Retained contribution": retained_energy,
        "Rejected contribution": rejected_image.energy,
    }
    similarities = {
        name: _dispersion_similarity(energy, reference.energy, velocity)
        for name, energy in energies.items()
        if name != "Theoretical EGF"
    }
    figure, axes = plt.subplots(
        1, 4, figsize=(14.8, 3.8), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for axis, (name, energy), label in zip(axes, energies.items(), "abcd"):
        image = axis.imshow(
            energy,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(frequency[0], frequency[-1], velocity[0], velocity[-1]),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        for mode_index in range(theory_velocity.shape[1]):
            axis.plot(theory_frequency, theory_velocity[:, mode_index], "w--", lw=0.75)
        if name == "Theoretical EGF":
            subtitle = f"clean reference = {bool(reference_quality['clean_ridge_pass'])}"
        else:
            similarity = similarities[name]
            subtitle = (
                f"image CC={similarity['image_correlation']:.3f}\n"
                f"ridge MAE={similarity['dominant_ridge_mae_to_egf_mps']:.1f} m/s"
            )
        axis.set(
            xlabel="frequency (Hz)",
            ylabel="phase velocity (m/s)",
            title=f"{name}\n{subtitle}",
        )
        _panel(axis, label)
    assert image is not None
    figure.colorbar(image, ax=axes, pad=0.012, shrink=0.86).set_label(
        "column-normalized energy"
    )
    figure.suptitle(
        "Dispersion provenance | dashed curves = theoretical Rayleigh modes",
        fontsize=10.5,
    )
    files = _save(figure, output / "directional_density_retained_rejected_dispersion")
    summary = {
        "decomposition": (
            f"baseline=({retained_count}*retained+"
            f"{total_count-retained_count}*rejected)/{total_count}"
        ),
        "reference_clean_ridge_pass": bool(reference_quality["clean_ridge_pass"]),
        "similarity_to_theoretical_egf_dispersion": similarities,
    }
    return files, summary


def _waveform_data(run: Path):
    resolved = json.loads((run / "resolved_config.json").read_text())
    processing = resolved["processing"]
    band = (
        float(processing["frequency_min_hz"]),
        float(processing["frequency_max_hz"]),
    )
    data = {}
    lag = distance = reference = None
    for strategy in STRATEGIES:
        lag, distance, candidate, reference = calibrated_waveform_comparison(
            run / "results.h5", strategy, "symmetric", frequency_range_hz=band
        )
        data[strategy] = candidate
    return lag, distance, reference, data, band, processing


def plot_waveforms(run: Path, output: Path) -> list[Path]:
    _style()
    lag, distance, reference, data, band, processing = _waveform_data(run)
    targets = np.asarray((200.0, 500.0, 1000.0))
    indices = np.abs(distance[:, None] - targets[None, :]).argmin(axis=0)
    vmin = float(processing["velocity_min_mps"])
    vmax = float(processing["velocity_max_mps"])
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), sharey=True)
    for axis, distance_index, label in zip(axes, indices, "abc"):
        arrival_start = distance[distance_index] / vmax
        arrival_end = distance[distance_index] / vmin
        display = (lag >= 0.8 * arrival_start) & (lag <= 1.05 * arrival_end)
        axis.axvspan(arrival_start, arrival_end, color="#DDDDDD", alpha=0.5)
        curves = (("reference", reference[:, distance_index]),) + tuple(
            (strategy, data[strategy][:, distance_index]) for strategy in STRATEGIES
        )
        for name, curve in curves:
            scale = max(float(np.max(np.abs(curve[display]))), np.finfo(float).eps)
            axis.plot(
                lag,
                curve / scale,
                color=COLORS[name],
                linestyle="--" if name == "reference" else "-",
                linewidth=(
                    1.35
                    if name
                    in {
                        "reference",
                        "pair_crossfit_tau_p_diffuse_hard_frequency",
                    }
                    else 0.85
                ),
                label=LABELS[name],
            )
        axis.set(
            xlim=(max(0.0, 0.8 * arrival_start), min(float(lag[-1]), 1.05 * arrival_end)),
            ylim=(-1.15, 1.15),
            xlabel="lag time (s)",
            title=f"distance = {distance[distance_index]:.0f} m",
        )
        axis.grid(alpha=0.13)
        _panel(axis, label)
    axes[0].set_ylabel("per-trace normalized amplitude")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(
        f"Calibrated symmetric NCF versus theoretical EGF | {band[0]:g}-{band[1]:g} Hz",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0.0, 0.18, 1.0, 0.94))
    return _save(figure, output / "directional_density_waveform_overlays")


def _column_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(
        np.max(np.abs(values), axis=0, keepdims=True), np.finfo(float).eps
    )


def plot_gathers(run: Path, output: Path) -> list[Path]:
    _style()
    lag, distance, reference, data, _, processing = _waveform_data(run)
    panel_names = (
        "reference",
        "baseline_linear",
        "pair_crossfit_tau_p_diffuse_soft_frequency",
        "pair_crossfit_tau_p_diffuse_hard_frequency",
        "pair_crossfit_tau_p_diffuse_hard_frequency_permuted_time_control",
        "oracle_hard_mask",
    )
    vmin = float(processing["velocity_min_mps"])
    vmax = float(processing["velocity_max_mps"])
    figure, axes = plt.subplots(
        2, 3, figsize=(11.8, 6.7), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for axis, name, label in zip(axes.flat, panel_names, "abcdef"):
        values = reference if name == "reference" else data[name]
        image = axis.imshow(
            _column_normalize(values),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(distance[0], distance[-1], lag[0], lag[-1]),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axis.plot(distance, distance / vmax, "k:", linewidth=0.7)
        axis.plot(distance, distance / vmin, "k:", linewidth=0.7)
        axis.set(xlabel="distance (m)", ylabel="lag time (s)", title=LABELS[name])
        _panel(axis, label)
    assert image is not None
    figure.colorbar(image, ax=axes, pad=0.012, shrink=0.86).set_label(
        "per-distance normalized amplitude"
    )
    figure.suptitle(
        "Symmetric NCF moveout | dotted lines bound expected surface-wave arrivals",
        fontsize=10.5,
    )
    return _save(figure, output / "directional_density_ncf_gathers")


def _metrics(run: Path) -> tuple[dict, dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    reference, reference_quality = reference_dispersion_for_strategy(
        run, "baseline_linear"
    )
    records = {}
    with h5py.File(run / "results.h5") as handle:
        velocity = handle["dispersion/strategies/baseline_linear/velocity_mps"][:]
        for strategy in STRATEGIES:
            egf = manifest["strategy_egf_quality"][strategy]["zz_symmetric"]
            dispersion = manifest["strategy_dispersion_quality"][strategy]
            similarity = _dispersion_similarity(
                handle[f"dispersion/strategies/{strategy}/energy"][:],
                reference.energy,
                velocity,
            )
            records[strategy] = {
                "waveform_correlation": egf["waveform_correlation_no_shift"],
                "phase_rms_rad": egf["phase_rms_rad"],
                "prearrival_fraction": egf["prearrival_ghost_energy_fraction"],
                "spectral_shape_log_rms": egf[
                    "spectral_shape_log_rms_after_global_scale"
                ],
                "phase_velocity_pick_mae_mps": dispersion["picks"]["mae_mps"],
                "dispersion_image_correlation": similarity["image_correlation"],
                "dispersion_ridge_mae_to_egf_mps": similarity[
                    "dominant_ridge_mae_to_egf_mps"
                ],
                "clean_ridge_pass": dispersion["image"]["clean_ridge_pass"],
            }
    return records, reference_quality


def plot_dispersion(run: Path, output: Path) -> list[Path]:
    _style()
    reference, reference_quality = reference_dispersion_for_strategy(
        run, "baseline_linear"
    )
    metrics, _ = _metrics(run)
    with h5py.File(run / "results.h5") as handle:
        frequency = handle["dispersion/strategies/baseline_linear/frequency_hz"][:]
        velocity = handle["dispersion/strategies/baseline_linear/velocity_mps"][:]
        theory_frequency = handle["theory/frequency_hz"][:]
        theory_velocity = handle["theory/rayleigh_mps"][:]
        energy = {
            strategy: handle[f"dispersion/strategies/{strategy}/energy"][:]
            for strategy in STRATEGIES
        }
    panel_names = ("reference",) + STRATEGIES
    figure, axes = plt.subplots(
        2, 3, figsize=(12.2, 7.1), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for axis, name, label in zip(axes.flat, panel_names, "abcdef"):
        values = reference.energy if name == "reference" else energy[name]
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(frequency[0], frequency[-1], velocity[0], velocity[-1]),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        for mode_index in range(theory_velocity.shape[1]):
            axis.plot(theory_frequency, theory_velocity[:, mode_index], "w--", lw=0.75)
        if name == "reference":
            subtitle = f"clean reference = {bool(reference_quality['clean_ridge_pass'])}"
        else:
            subtitle = (
                f"image CC={metrics[name]['dispersion_image_correlation']:.3f}; "
                f"ridge MAE={metrics[name]['dispersion_ridge_mae_to_egf_mps']:.1f} m/s\n"
                f"theory-pick MAE={metrics[name]['phase_velocity_pick_mae_mps']:.1f} m/s"
            )
        axis.set(
            xlabel="frequency (Hz)",
            ylabel="phase velocity (m/s)",
            title=f"{LABELS[name]}\n{subtitle}",
        )
        _panel(axis, label)
    assert image is not None
    figure.colorbar(image, ax=axes, pad=0.012, shrink=0.86).set_label(
        "column-normalized energy"
    )
    figure.suptitle(
        "Dispersion-energy validation | dashed curves = theoretical Rayleigh modes",
        fontsize=10.5,
    )
    return _save(figure, output / "directional_density_dispersion")


def plot_metric_scorecard(run: Path, output: Path) -> tuple[list[Path], dict]:
    _style()
    records, reference_quality = _metrics(run)
    metric_specs = (
        ("waveform_correlation", "waveform correlation", True),
        ("phase_rms_rad", "phase RMS (rad)", False),
        ("prearrival_fraction", "pre-arrival energy fraction", False),
        ("spectral_shape_log_rms", "spectral-shape log RMS", False),
        ("dispersion_image_correlation", "dispersion image CC", True),
        ("dispersion_ridge_mae_to_egf_mps", "ridge MAE to EGF (m/s)", False),
        ("phase_velocity_pick_mae_mps", "pick MAE to theory (m/s)", False),
    )
    x = np.arange(len(STRATEGIES))
    figure, axes = plt.subplots(2, 4, figsize=(12.8, 6.8), constrained_layout=True)
    for axis, (metric, title, higher), label in zip(axes.flat, metric_specs, "abcdefg"):
        values = [records[strategy][metric] for strategy in STRATEGIES]
        axis.plot(x, values, color="#B5B5B5", lw=0.8, zorder=1)
        axis.scatter(
            x,
            values,
            s=34,
            color=[COLORS[strategy] for strategy in STRATEGIES],
            edgecolor="black",
            linewidth=0.35,
            zorder=2,
        )
        for index, value in enumerate(values):
            axis.annotate(
                f"{value:.3g}",
                (index, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=6.2,
            )
        axis.set(
            xticks=x,
            title=f"{title}\n{'higher' if higher else 'lower'} is better",
        )
        axis.set_xticklabels(
            [SHORT_LABELS[strategy] for strategy in STRATEGIES],
            rotation=28,
            ha="right",
            rotation_mode="anchor",
        )
        axis.grid(alpha=0.13)
        _panel(axis, label)
    axes.flat[-1].axis("off")
    figure.suptitle(
        "EGF-equivalence scorecard | no metric is replaced by a visual SNR judgment",
        fontsize=10.7,
    )
    files = _save(figure, output / "directional_density_metric_scorecard")
    return files, {"reference_dispersion": reference_quality, "strategies": records}


def write_metrics_csv(records: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = list(next(iter(records.values())).keys())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["strategy", *metric_names])
        writer.writeheader()
        for strategy in STRATEGIES:
            writer.writerow({"strategy": strategy, **records[strategy]})
    return path


def _seed_records(seed_root: Path) -> tuple[list[dict], list[dict]]:
    metric_rows = []
    selector_rows = []
    for run in sorted(path for path in seed_root.glob("seed_*") if path.is_dir()):
        seed = int(run.name.split("_")[-1])
        records, _ = _metrics(run)
        for strategy, values in records.items():
            metric_rows.append({"noise_seed": seed, "strategy": strategy, **values})
        selector = _load_selector(run)
        selector_rows.append(
            {
                "noise_seed": seed,
                "hard_mask_oracle_accuracy": float(selector["accuracy"]),
            }
        )
    if not metric_rows:
        raise ValueError(f"no completed seed runs found under {seed_root}")
    return metric_rows, selector_rows


def _write_rows_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_multiseed(
    seed_root: Path, output: Path
) -> tuple[list[Path], dict, list[dict], list[dict]]:
    _style()
    rows, selector_rows = _seed_records(seed_root)
    seeds = sorted({int(row["noise_seed"]) for row in rows})
    metric_specs = (
        ("waveform_correlation", "waveform correlation", True),
        ("phase_rms_rad", "phase RMS (rad)", False),
        ("prearrival_fraction", "pre-arrival energy fraction", False),
        ("spectral_shape_log_rms", "spectral-shape log RMS", False),
        ("dispersion_image_correlation", "dispersion image CC", True),
        ("dispersion_ridge_mae_to_egf_mps", "ridge MAE to EGF (m/s)", False),
        ("phase_velocity_pick_mae_mps", "pick MAE to theory (m/s)", False),
    )
    x = np.arange(len(STRATEGIES))
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.9), constrained_layout=True)
    medians: dict[str, dict[str, float]] = {strategy: {} for strategy in STRATEGIES}
    wins_hard_vs_linear = {}
    wins_hard_vs_permuted = {}
    hard = "pair_crossfit_tau_p_diffuse_hard_frequency"
    permuted = "pair_crossfit_tau_p_diffuse_hard_frequency_permuted_time_control"
    for axis, (metric, title, higher), label in zip(axes.flat, metric_specs, "abcdefg"):
        for seed in seeds:
            values = [
                next(
                    float(row[metric])
                    for row in rows
                    if row["noise_seed"] == seed and row["strategy"] == strategy
                )
                for strategy in STRATEGIES
            ]
            axis.plot(x, values, color="#C4C4C4", lw=0.65, alpha=0.85, zorder=1)
        median_values = []
        for strategy in STRATEGIES:
            values = [
                float(row[metric]) for row in rows if row["strategy"] == strategy
            ]
            medians[strategy][metric] = float(np.median(values))
            median_values.append(medians[strategy][metric])
        axis.scatter(
            x,
            median_values,
            s=38,
            color=[COLORS[strategy] for strategy in STRATEGIES],
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        for index, value in enumerate(median_values):
            axis.annotate(
                f"{value:.3g}",
                (index, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=6.1,
            )
        axis.set(xticks=x, title=f"{title}\n{'higher' if higher else 'lower'} is better")
        axis.set_xticklabels(
            [SHORT_LABELS[strategy] for strategy in STRATEGIES],
            rotation=28,
            ha="right",
            rotation_mode="anchor",
        )
        axis.grid(alpha=0.13)
        _panel(axis, label)
        compare = (lambda left, right: left > right) if higher else (
            lambda left, right: left < right
        )
        wins_hard_vs_linear[metric] = sum(
            compare(
                next(
                    float(row[metric])
                    for row in rows
                    if row["noise_seed"] == seed and row["strategy"] == hard
                ),
                next(
                    float(row[metric])
                    for row in rows
                    if row["noise_seed"] == seed
                    and row["strategy"] == "baseline_linear"
                ),
            )
            for seed in seeds
        )
        wins_hard_vs_permuted[metric] = sum(
            compare(
                next(
                    float(row[metric])
                    for row in rows
                    if row["noise_seed"] == seed and row["strategy"] == hard
                ),
                next(
                    float(row[metric])
                    for row in rows
                    if row["noise_seed"] == seed and row["strategy"] == permuted
                ),
            )
            for seed in seeds
        )

    accuracy = np.asarray(
        [row["hard_mask_oracle_accuracy"] for row in selector_rows], dtype=float
    )
    axes.flat[-1].scatter(
        np.arange(len(seeds)),
        accuracy,
        s=28,
        color="#0072B2",
        edgecolor="black",
        linewidth=0.35,
    )
    axes.flat[-1].axhline(
        np.median(accuracy), color="#0072B2", ls="--", lw=0.85
    )
    axes.flat[-1].set(
        xticks=np.arange(len(seeds)),
        xticklabels=[str(seed) for seed in seeds],
        ylim=(0, 1.03),
        xlabel="noise seed",
        ylabel="mask agreement",
        title=f"hard-mask agreement with Oracle\nmedian = {np.median(accuracy):.1%}",
    )
    axes.flat[-1].grid(alpha=0.13)
    _panel(axes.flat[-1], "h")
    figure.suptitle(
        f"Fixed-parameter validation across {len(seeds)} unseen noise seeds | thin lines = paired seeds; dots = medians",
        fontsize=10.7,
    )
    files = _save(figure, output / "directional_density_multiseed_scorecard")
    summary = {
        "seed_count": len(seeds),
        "noise_seeds": seeds,
        "medians": medians,
        "hard_wins_vs_linear": wins_hard_vs_linear,
        "hard_wins_vs_permuted_control": wins_hard_vs_permuted,
        "hard_mask_oracle_accuracy": {
            "median": float(np.median(accuracy)),
            "minimum": float(np.min(accuracy)),
            "maximum": float(np.max(accuracy)),
        },
    }
    return files, summary, rows, selector_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, default=None)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    output = args.output_directory.resolve()
    files = []
    source_files, source_summary = plot_source_selector_audit(run, output)
    files.extend(source_files)
    contribution_files, contribution_summary, source_contribution_csv = (
        plot_retained_rejected_source_contributions(run, output)
    )
    files.extend(contribution_files)
    files.extend(plot_retained_rejected_waveforms(run, output))
    decomposition_files, decomposition_summary = plot_retained_rejected_dispersion(
        run, output
    )
    files.extend(decomposition_files)
    files.extend(plot_waveforms(run, output))
    files.extend(plot_gathers(run, output))
    files.extend(plot_dispersion(run, output))
    metric_files, metric_summary = plot_metric_scorecard(run, output)
    files.extend(metric_files)
    metrics_csv = write_metrics_csv(
        metric_summary["strategies"], output / "directional_density_metrics.csv"
    )
    multiseed_summary = None
    multiseed_metrics_csv = None
    multiseed_selector_csv = None
    if args.seed_root is not None:
        multiseed_files, multiseed_summary, seed_rows, selector_rows = plot_multiseed(
            args.seed_root.resolve(), output
        )
        files.extend(multiseed_files)
        multiseed_metrics_csv = _write_rows_csv(
            seed_rows, output / "directional_density_multiseed_metrics.csv"
        )
        multiseed_selector_csv = _write_rows_csv(
            selector_rows, output / "directional_density_multiseed_selector.csv"
        )
    summary = output / "directional_density_qc_summary.json"
    summary.write_text(
        json.dumps(
            {
                "run_directory": str(run),
                "source_selector_audit": source_summary,
                "retained_rejected_source_contributions": contribution_summary,
                "retained_rejected_dispersion": decomposition_summary,
                **metric_summary,
                "multiseed": multiseed_summary,
                "interpretation": {
                    "data_only": True,
                    "uses_theoretical_egf_for_training_or_selection": False,
                    "known_source_oracle_is_deployable": False,
                    "single_seed_exploratory": True,
                    "complete_egf_gate": (
                        "not passed because pre-arrival energy worsens despite "
                        "waveform, phase, and dispersion improvements"
                    ),
                },
            },
            indent=2,
        )
        + "\n"
    )
    manifest = output / "directional_density_qc_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "figures": [str(path) for path in files],
                "metrics_csv": str(metrics_csv),
                "source_contribution_csv": str(source_contribution_csv),
                "multiseed_metrics_csv": (
                    str(multiseed_metrics_csv)
                    if multiseed_metrics_csv is not None
                    else None
                ),
                "multiseed_selector_csv": (
                    str(multiseed_selector_csv)
                    if multiseed_selector_csv is not None
                    else None
                ),
                "summary": str(summary),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(files)} figure files, {metrics_csv}, {summary}, and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
