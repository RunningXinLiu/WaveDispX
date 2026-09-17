"""Reproducible QC figures and summaries for pair-resolved NNWS experiments.

The routines in this module consume only saved WaveDispX artifacts.  They do
not retrain NNWS or alter any stack, so every plotted curve can be traced back
to ``results.h5`` and ``manifest.json``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .processing import DispersionImage, surface_wave_dispersion
from .quality import dispersion_quality_metrics

METHOD_COLORS = {
    "reference": "#111111",
    "baseline_linear": "#5A5A5A",
    "nnws_common_linear": "#5A5A5A",
    "nnws_random_neff_control": "#5875A4",
    "nnws_pairwise": "#C44E52",
    "nnws_hybrid_random_neff_control": "#5875A4",
    "nnws_hybrid_full_aperture": "#C44E52",
    "nnws_common_oracle_hard_mask": "#CC8A00",
}
METHOD_LABELS = {
    "reference": "theoretical EGF",
    "baseline_linear": "full-aperture linear",
    "nnws_common_linear": "common linear",
    "nnws_random_neff_control": "random Neff control",
    "nnws_pairwise": "NNWS",
    "nnws_hybrid_random_neff_control": "hybrid random Neff control",
    "nnws_hybrid_full_aperture": "hybrid NNWS",
    "nnws_common_oracle_hard_mask": "source-oracle hard mask",
}
COMMON_STRATEGIES = (
    "nnws_common_linear",
    "nnws_random_neff_control",
    "nnws_pairwise",
)
FULL_APERTURE_STRATEGIES = (
    "baseline_linear",
    "nnws_hybrid_random_neff_control",
    "nnws_hybrid_full_aperture",
)
SUMMARY_STRATEGIES = (*COMMON_STRATEGIES, *FULL_APERTURE_STRATEGIES)


def configure_figure_style() -> None:
    """Apply compact, editable publication/QC defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(figure: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_stem.with_suffix(".svg")
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(svg_path)
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)
    return [svg_path, pdf_path, png_path]


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def _read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    return np.atleast_1d(table["x_m"]), np.atleast_1d(table["y_m"])


def _load_activity_from_resolved(
    resolved: Mapping[str, object], window_count: int
) -> np.ndarray:
    ambient = resolved["ambient_sources"]
    assert isinstance(ambient, dict)
    source_count = int(ambient["count"])
    raw_path = ambient.get("activity_file")
    if raw_path is None:
        return np.ones((window_count, source_count), dtype=np.float64)
    table = np.genfromtxt(raw_path, delimiter=",", names=True, dtype=np.float64)
    activity = np.zeros((window_count, source_count), dtype=np.float64)
    window_index = np.atleast_1d(table["window_index"]).astype(int)
    source_index = np.atleast_1d(table["source_index"]).astype(int)
    activity[window_index, source_index] = np.atleast_1d(
        table["amplitude_multiplier"]
    )
    return activity


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.atleast_1d(values)
    )


def _normalise_columns(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    result = np.asarray(data, dtype=np.float64).copy()
    if mask is None:
        scale = np.max(np.abs(result), axis=0, initial=0.0)
    else:
        scale = np.array(
            [
                np.max(np.abs(result[mask[:, column], column]), initial=0.0)
                for column in range(result.shape[1])
            ]
        )
    scale[scale == 0] = 1.0
    return result / scale


def _bandpass_fft(
    data: np.ndarray,
    dt_s: float,
    frequency_range_hz: tuple[float, float],
) -> np.ndarray:
    sample_count = data.shape[0]
    frequency = np.fft.rfftfreq(sample_count, dt_s)
    selected = (frequency >= frequency_range_hz[0]) & (
        frequency <= frequency_range_hz[1]
    )
    spectrum = np.fft.rfft(data, axis=0)
    spectrum[~selected] = 0
    return np.fft.irfft(spectrum, n=sample_count, axis=0)


def calibrated_waveform_comparison(
    results_h5: str | Path,
    strategy: str,
    branch: str = "symmetric",
    *,
    component: str = "zz",
    frequency_range_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return band-limited calibrated NCF and distance-aggregated EGF.

    This deliberately matches :func:`wavedispx.quality.egf_equivalence_metrics`:
    no time shift is estimated, the NCF calibration operator is honored, and
    the active reference is averaged over exactly the receiver pairs assigned
    to each saved strategy distance bin.
    """
    with h5py.File(results_h5, "r") as handle:
        strategy_group = handle[f"ncf/strategies/{strategy}"]
        lag = strategy_group["lag_time_s"][:].astype(np.float64)
        distance = strategy_group["distance_m"][:].astype(np.float64)
        candidate = strategy_group[f"components/{component}/{branch}"][:].astype(
            np.float64
        )
        reference_group = handle["egf_reference"]
        reference_time = reference_group["time_s"][:].astype(np.float64)
        sample_count = min(lag.size, reference_time.size)
        lag = lag[:sample_count]
        candidate = candidate[:sample_count]
        dt_s = float(lag[1] - lag[0])
        operator = str(reference_group.attrs["calibration_operator"])
        if operator == "negative_time_derivative_of_ncf":
            candidate = -np.gradient(candidate, dt_s, axis=0)
        elif operator not in {"identity", "none_raw_green_function"}:
            raise ValueError(f"unsupported EGF calibration operator: {operator}")

        component_names = _decode_strings(reference_group.attrs["component_names"])
        component_index = component_names.index(component)
        reference_component = reference_group["components"][
            :sample_count, :, component_index
        ].astype(np.float64)
        reference_distance = reference_group["distance_m"][:]
        pair_distance_index = reference_group["pair_distance_index"][:].astype(int)
        pair_distance = reference_distance[pair_distance_index]
        pair_bin = np.abs(pair_distance[:, None] - distance[None, :]).argmin(axis=1)
        reference = np.empty_like(candidate)
        pair_reference = reference_component[:, pair_distance_index]
        for distance_index in range(distance.size):
            selected_pair = pair_bin == distance_index
            if not np.any(selected_pair):
                raise ValueError("active reference has no pair in one NCF distance bin")
            reference[:, distance_index] = np.mean(
                pair_reference[:, selected_pair], axis=1
            )

    return (
        lag,
        distance,
        _bandpass_fft(candidate, dt_s, frequency_range_hz),
        _bandpass_fft(reference, dt_s, frequency_range_hz),
    )


def reference_dispersion_for_strategy(
    run_directory: str | Path,
    strategy: str,
) -> tuple[DispersionImage, dict[str, float | int | bool]]:
    """Image the saved active EGF on exactly one strategy's distance support."""
    run_directory = Path(run_directory)
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    processing = resolved["processing"]
    frequency_range = (
        float(processing["frequency_min_hz"]),
        float(processing["frequency_max_hz"]),
    )
    lag, distance, _, reference = calibrated_waveform_comparison(
        run_directory / "results.h5",
        strategy,
        "symmetric",
        frequency_range_hz=frequency_range,
    )
    with h5py.File(run_directory / "results.h5", "r") as handle:
        dispersion_path = f"dispersion/strategies/{strategy}/velocity_mps"
        velocity = handle[
            dispersion_path
            if dispersion_path in handle
            else "dispersion/strategies/baseline_linear/velocity_mps"
        ][:]
        theory_frequency = handle["theory/frequency_hz"][:]
        theory_velocity = handle["theory/rayleigh_mps"][:]
    image = surface_wave_dispersion(
        reference,
        distance,
        float(lag[1] - lag[0]),
        velocity,
        frequency_min_hz=frequency_range[0],
        frequency_max_hz=frequency_range[1],
        method=str(processing["dispersion_method"]),
        zero_padding=float(processing["zero_padding"]),
        distance_taper_fraction=float(processing["distance_taper_fraction"]),
        frequency_block_size=8,
        radial_kernel_lookup_step=float(processing["radial_kernel_lookup_step"]),
        workers=0,
    )
    quality = dispersion_quality_metrics(
        image,
        theory_frequency,
        theory_velocity,
        frequency_range_hz=frequency_range,
    )
    return image, quality


def _dispersion_similarity(
    candidate: np.ndarray,
    reference: np.ndarray,
    velocity_mps: np.ndarray,
) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    epsilon = np.finfo(float).eps
    correlation = float(
        np.vdot(candidate, reference)
        / max(np.linalg.norm(candidate) * np.linalg.norm(reference), epsilon)
    )
    normalized_l2 = float(
        np.linalg.norm(
            candidate / max(np.linalg.norm(candidate), epsilon)
            - reference / max(np.linalg.norm(reference), epsilon)
        )
    )
    candidate_ridge = velocity_mps[np.argmax(candidate, axis=0)]
    reference_ridge = velocity_mps[np.argmax(reference, axis=0)]
    return {
        "image_correlation": correlation,
        "unit_norm_l2": normalized_l2,
        "dominant_ridge_mae_to_egf_mps": float(
            np.mean(np.abs(candidate_ridge - reference_ridge))
        ),
    }


def extract_seed_summary_rows(
    run_directory: str | Path,
    seed: int,
    *,
    strategies: Iterable[str] = SUMMARY_STRATEGIES,
) -> list[dict[str, object]]:
    """Extract auditable scalar rows from one completed benchmark run."""
    run_directory = Path(run_directory)
    manifest = json.loads((run_directory / "manifest.json").read_text())
    rows: list[dict[str, object]] = []
    nnws_manifest = manifest.get("nnws_pairwise") or {}
    for strategy in strategies:
        strategy_quality = manifest["strategy_egf_quality"].get(strategy, {})
        statistics = manifest["ncf_strategies"].get(strategy, {})
        for comparison_name, metrics in strategy_quality.items():
            branch = comparison_name.removeprefix("zz_")
            rows.append(
                {
                    "noise_seed": seed,
                    "strategy": strategy,
                    "branch": branch,
                    "sufficient": bool(statistics.get("sufficient", False)),
                    "effective_window_count": statistics.get(
                        "effective_window_count", float("nan")
                    ),
                    "nonzero_window_count": statistics.get(
                        "nonzero_window_count", float("nan")
                    ),
                    "waveform_correlation_no_shift": metrics.get(
                        "waveform_correlation_no_shift", float("nan")
                    ),
                    "phase_rms_rad": metrics.get("phase_rms_rad", float("nan")),
                    "prearrival_ghost_energy_fraction": metrics.get(
                        "prearrival_ghost_energy_fraction", float("nan")
                    ),
                    "spectral_shape_log_rms_after_global_scale": metrics.get(
                        "spectral_shape_log_rms_after_global_scale", float("nan")
                    ),
                    "median_envelope_peak_time_bias_s": metrics.get(
                        "median_envelope_peak_time_bias_s", float("nan")
                    ),
                    "valid_distance_count": metrics.get(
                        "valid_distance_count", 0
                    ),
                    "pair_coverage": nnws_manifest.get(
                        "pair_coverage", float("nan")
                    ),
                    "eligible_pair_count": nnws_manifest.get(
                        "eligible_pair_count", 0
                    ),
                    "failed_pair_count": nnws_manifest.get("failed_pair_count", 0),
                    "run_directory": str(run_directory.resolve()),
                }
            )
    return rows


def write_seed_summary_csv(rows: list[dict[str, object]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("summary rows must not be empty")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_source_and_weight_qc(
    run_directory: str | Path,
    output_directory: str | Path,
) -> list[Path]:
    """Plot source geometry/activity, oracle metrics, and pairwise NNWS weights."""
    configure_figure_style()
    run_directory = Path(run_directory)
    output_directory = Path(output_directory)
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    window_s = float(resolved["noise"]["window_s"])
    source_x, source_y = _read_xy(run_directory / "sources.csv")
    receiver_x, receiver_y = _read_xy(run_directory / "receivers.csv")

    with h5py.File(run_directory / "results.h5", "r") as handle:
        selection = handle["source_selection"]
        window_index = selection["window_index"][:].astype(int)
        activity = _load_activity_from_resolved(resolved, window_index.size)
        activity_power = activity**2
        mean_activity = np.mean(activity_power, axis=0)
        center_x = float(np.mean(receiver_x))
        center_y = float(np.mean(receiver_y))
        azimuth = np.degrees(np.arctan2(source_y - center_y, source_x - center_x))
        source_order = np.argsort(azimuth)
        metrics = {
            name: np.mean(selection[name][:], axis=1)
            for name in (
                "endfire_fraction",
                "bidirectional_balance",
                "azimuth_entropy",
            )
        }
        oracle_soft = selection["soft_weight"][:]
        oracle_hard = selection["hard_selected"][:].astype(bool)

        nnws = handle["nnws_pairwise"]
        pair_distance = nnws["distance_m"][:]
        eligible = nnws["eligible"][:].astype(bool)
        sufficient = nnws["sufficient"][:].astype(bool)
        weights = nnws["weights"][:].astype(np.float64)
        random_weights = nnws["controls/random_neff/weights"][:].astype(np.float64)
        effective_windows = nnws["effective_window_count"][:]
        pair_coverage = float(nnws.attrs["pair_coverage"])
        eligible_count = int(nnws.attrs["eligible_pair_count"])
        successful_count = int(nnws.attrs["successful_pair_count"])

    usable_pair = eligible & sufficient
    sorted_pair = np.flatnonzero(usable_pair)[np.argsort(pair_distance[usable_pair])]
    if sorted_pair.size == 0:
        raise ValueError("no sufficient NNWS pairs are available for QC")
    usable_weights = weights[:, sorted_pair]
    random_usable = random_weights[:, sorted_pair]
    median_weight = np.median(usable_weights, axis=1)
    weight_p10, weight_p90 = np.percentile(usable_weights, [10, 90], axis=1)
    random_median = np.median(random_usable, axis=1)

    figure = plt.figure(figsize=(11.6, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 0.92, 0.92))
    axes = [figure.add_subplot(grid[row, column]) for row in range(3) for column in range(2)]

    scatter = axes[0].scatter(
        source_x,
        source_y,
        c=mean_activity,
        s=42,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.4,
        label="Sources",
    )
    axes[0].scatter(
        receiver_x,
        receiver_y,
        marker="v",
        s=22,
        c="#222222",
        label="Receivers",
        zorder=3,
    )
    axes[0].axhline(center_y, color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Source–receiver geometry")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(frameon=False, loc="best")
    colorbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("mean source power multiplier")

    image = axes[1].imshow(
        activity_power[:, source_order].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(window_index[0], window_index[-1] + 1, 0, source_order.size),
        cmap="magma",
    )
    source_tick_step = max(1, source_order.size // 7)
    source_ticks = np.arange(source_order.size)[::source_tick_step] + 0.5
    axes[1].set_yticks(source_ticks)
    axes[1].set_yticklabels(
        [f"{value:.0f}" for value in azimuth[source_order][::source_tick_step]]
    )
    axes[1].set(
        xlabel=f"{window_s:g}-s window index",
        ylabel="source azimuth (deg)",
        title="Prescribed source activity",
    )
    colorbar = figure.colorbar(image, ax=axes[1], pad=0.02)
    colorbar.set_label("power multiplier")

    for name, color, label in (
        ("endfire_fraction", "#0072B2", "endfire fraction"),
        ("bidirectional_balance", "#009E73", "bidirectional balance"),
        ("azimuth_entropy", "#CC79A7", "azimuth entropy"),
    ):
        axes[2].plot(window_index, metrics[name], color=color, label=label)
    axes[2].fill_between(
        window_index,
        0,
        1,
        where=oracle_hard,
        step="mid",
        color="#E69F00",
        alpha=0.12,
        label="Oracle-selected",
    )
    axes[2].set(
        xlabel=f"{window_s:g}-s window index",
        ylabel="score",
        ylim=(-0.03, 1.03),
        title="Source-field selection diagnostics",
    )
    axes[2].legend(frameon=False, ncol=2)

    axes[3].fill_between(
        window_index,
        weight_p10,
        weight_p90,
        color=METHOD_COLORS["nnws_pairwise"],
        alpha=0.18,
        label="NNWS pair P10–P90",
    )
    axes[3].plot(
        window_index,
        median_weight,
        color=METHOD_COLORS["nnws_pairwise"],
        label="NNWS pair median",
    )
    axes[3].plot(
        window_index,
        random_median,
        color=METHOD_COLORS["nnws_random_neff_control"],
        label="Random control median",
    )
    oracle_scaled = oracle_soft / max(float(np.max(oracle_soft)), np.finfo(float).eps)
    axes[3].plot(
        window_index,
        oracle_scaled,
        color=METHOD_COLORS["nnws_common_oracle_hard_mask"],
        linestyle="--",
        label="Oracle soft (unit max)",
    )
    axes[3].set(
        xlabel=f"{window_s:g}-s window index",
        ylabel="weight",
        title="Window weighting across eligible pairs",
    )
    axes[3].legend(frameon=False, ncol=2)

    heatmap = axes[4].imshow(
        usable_weights.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(window_index[0], window_index[-1] + 1, 0, sorted_pair.size),
        cmap="viridis",
        vmin=0,
        vmax=max(1.0, float(np.nanmax(usable_weights))),
    )
    y_position = np.arange(sorted_pair.size) + 0.5
    axes[4].set_yticks(y_position[:: max(1, sorted_pair.size // 7)])
    axes[4].set_yticklabels(
        [f"{value:g}" for value in pair_distance[sorted_pair][:: max(1, sorted_pair.size // 7)]]
    )
    axes[4].set(
        xlabel=f"{window_s:g}-s window index",
        ylabel="pair distance (m)",
        title="Pair-resolved NNWS weights",
    )
    colorbar = figure.colorbar(heatmap, ax=axes[4], pad=0.02)
    colorbar.set_label("NNWS weight")

    axes[5].scatter(
        pair_distance[eligible & ~sufficient],
        effective_windows[eligible & ~sufficient],
        marker="x",
        c="#999999",
        label="Failed eligible pair",
    )
    axes[5].scatter(
        pair_distance[usable_pair],
        effective_windows[usable_pair],
        s=25,
        c=METHOD_COLORS["nnws_pairwise"],
        label="Sufficient pair",
    )
    axes[5].axhline(
        float(resolved["nnws"]["minimum_effective_windows"]),
        color="#222222",
        linestyle="--",
        linewidth=0.8,
        label="Minimum effective-window count",
    )
    axes[5].set(
        xlabel="pair distance (m)",
        ylabel="effective window count",
        title=f"Pair sufficiency: {successful_count}/{eligible_count} ({pair_coverage:.0%})",
    )
    axes[5].legend(frameon=False)

    for axis, label in zip(axes, "abcdef"):
        _panel_label(axis, label)
        axis.grid(alpha=0.16)
    figure.suptitle(
        f"NNWS source-field and weight audit | saved {window_index.size}-window benchmark",
        fontsize=10.5,
    )
    return _save_figure(figure, output_directory / "nnws_qc_sources_weights")


def plot_waveform_overlays(
    run_directory: str | Path,
    output_directory: str | Path,
    *,
    branches: tuple[str, ...] = ("positive", "negative", "symmetric"),
) -> list[Path]:
    """Overlay full-aperture stacks and the theoretical EGF at three distances."""
    configure_figure_style()
    run_directory = Path(run_directory)
    output_directory = Path(output_directory)
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    frequency_range = (
        float(resolved["processing"]["frequency_min_hz"]),
        float(resolved["processing"]["frequency_max_hz"]),
    )
    minimum_velocity = float(resolved["processing"]["velocity_min_mps"])
    maximum_velocity = float(resolved["processing"]["velocity_max_mps"])
    strategy_data: dict[tuple[str, str], np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    lag = distance = None
    for branch in branches:
        for strategy in FULL_APERTURE_STRATEGIES:
            lag, distance, candidate, reference = calibrated_waveform_comparison(
                run_directory / "results.h5",
                strategy,
                branch,
                frequency_range_hz=frequency_range,
            )
            strategy_data[(branch, strategy)] = candidate
            references[branch] = reference
    assert lag is not None and distance is not None
    maximum_distance = float(np.max(distance))
    target_distances = (
        np.asarray((150.0, 350.0, maximum_distance))
        if maximum_distance >= 400.0
        else np.asarray((100.0, 150.0, maximum_distance))
    )
    selected_distance_index = np.unique(
        np.abs(distance[:, None] - target_distances[None, :]).argmin(axis=0)
    )
    figure, axes = plt.subplots(
        len(branches),
        selected_distance_index.size,
        figsize=(11.2, 7.3),
        sharex=True,
        squeeze=False,
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        bottom=0.07,
        top=0.83,
        wspace=0.14,
        hspace=0.30,
    )
    for row, branch in enumerate(branches):
        for column, distance_index in enumerate(selected_distance_index):
            axis = axes[row, column]
            signal_start = distance[distance_index] / maximum_velocity
            signal_end = distance[distance_index] / minimum_velocity
            signal_mask = (lag >= signal_start) & (lag <= signal_end)
            axis.axvspan(signal_start, signal_end, color="#DDDDDD", alpha=0.55)
            curves: list[tuple[str, np.ndarray]] = [
                ("reference", references[branch][:, distance_index])
            ] + [
                (strategy, strategy_data[(branch, strategy)][:, distance_index])
                for strategy in FULL_APERTURE_STRATEGIES
            ]
            for name, curve in curves:
                scale = np.max(np.abs(curve[signal_mask]), initial=0.0)
                if scale == 0:
                    scale = np.max(np.abs(curve), initial=1.0)
                axis.plot(
                    lag,
                    curve / max(scale, np.finfo(float).eps),
                    color=METHOD_COLORS[name],
                    linestyle="--" if name == "reference" else "-",
                    linewidth=(
                        1.35
                        if name in {"reference", "nnws_hybrid_full_aperture"}
                        else 1.0
                    ),
                    alpha=0.95,
                    label=METHOD_LABELS[name],
                )
            axis.axvline(signal_start, color="#777777", linewidth=0.6, linestyle=":")
            axis.set_xlim(
                0,
                min(
                    float(lag[-1]),
                    1.2 * float(distance[selected_distance_index[-1]])
                    / minimum_velocity,
                ),
            )
            axis.set_ylim(-1.18, 1.18)
            axis.grid(alpha=0.16)
            axis.set_title(f"{branch} | {distance[distance_index]:g} m")
            if column == 0:
                axis.set_ylabel("amplitude (per-curve normalized)")
            if row == len(branches) - 1:
                axis.set_xlabel("lag time (s)")
            _panel_label(axis, chr(ord("a") + row * selected_distance_index.size + column))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"Full-aperture NCF–EGF overlays | {frequency_range[0]:g}–{frequency_range[1]:g} Hz | gray = expected arrival",
        fontsize=10.5,
        y=0.985,
    )
    return _save_figure(
        figure, output_directory / "nnws_qc_waveform_overlays_full_aperture"
    )


def plot_ncf_gathers(
    run_directory: str | Path,
    output_directory: str | Path,
) -> list[Path]:
    """Plot symmetric full-aperture NCF gathers beside the reference and residual."""
    configure_figure_style()
    run_directory = Path(run_directory)
    output_directory = Path(output_directory)
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    frequency_range = (
        float(resolved["processing"]["frequency_min_hz"]),
        float(resolved["processing"]["frequency_max_hz"]),
    )
    minimum_velocity = float(resolved["processing"]["velocity_min_mps"])
    maximum_velocity = float(resolved["processing"]["velocity_max_mps"])
    data: dict[str, np.ndarray] = {}
    lag = distance = reference = None
    for strategy in FULL_APERTURE_STRATEGIES:
        lag, distance, candidate, reference = calibrated_waveform_comparison(
            run_directory / "results.h5",
            strategy,
            "symmetric",
            frequency_range_hz=frequency_range,
        )
        data[strategy] = _normalise_columns(candidate)
    assert lag is not None and distance is not None and reference is not None
    data["reference"] = _normalise_columns(reference)
    residual = data["nnws_hybrid_full_aperture"] - data["reference"]
    panels = (
        ("reference", data["reference"]),
        ("baseline_linear", data["baseline_linear"]),
        (
            "nnws_hybrid_random_neff_control",
            data["nnws_hybrid_random_neff_control"],
        ),
        ("nnws_hybrid_full_aperture", data["nnws_hybrid_full_aperture"]),
        ("residual", residual),
    )
    figure, axes = plt.subplots(1, 5, figsize=(12.4, 3.65), sharey=True, constrained_layout=True)
    for axis, (name, values), label in zip(axes, panels, "abcde"):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(distance[0], distance[-1], lag[0], lag[-1]),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axis.plot(distance, distance / maximum_velocity, color="#222222", linestyle=":", linewidth=0.8)
        axis.plot(distance, distance / minimum_velocity, color="#222222", linestyle=":", linewidth=0.8)
        title = "hybrid NNWS − EGF" if name == "residual" else METHOD_LABELS[name]
        axis.set(xlabel="distance (m)", title=title)
        _panel_label(axis, label)
    axes[0].set_ylabel("lag time (s)")
    colorbar = figure.colorbar(image, ax=axes, pad=0.012, shrink=0.9)
    colorbar.set_label("per-distance normalized amplitude")
    figure.suptitle(
        f"Full-aperture symmetric NCF gathers | {frequency_range[0]:g}–{frequency_range[1]:g} Hz | dotted = {maximum_velocity:g}–{minimum_velocity:g} m/s",
        fontsize=10.5,
    )
    return _save_figure(
        figure, output_directory / "nnws_qc_ncf_gathers_full_aperture"
    )


def plot_dispersion_qc(
    run_directory: str | Path,
    output_directory: str | Path,
) -> list[Path]:
    """Compare NCF dispersion with the resolution-limited active-EGF image."""
    configure_figure_style()
    run_directory = Path(run_directory)
    output_directory = Path(output_directory)
    full_strategies = (
        "baseline_linear",
        "nnws_hybrid_random_neff_control",
        "nnws_hybrid_full_aperture",
    )
    far_strategies = (
        "nnws_common_linear",
        "nnws_random_neff_control",
        "nnws_pairwise",
    )
    plotted_strategies = (*full_strategies, *far_strategies)
    with h5py.File(run_directory / "results.h5", "r") as handle:
        frequency = handle["dispersion/strategies/baseline_linear/frequency_hz"][:]
        velocity = handle["dispersion/strategies/baseline_linear/velocity_mps"][:]
        available = {
            strategy
            for strategy in plotted_strategies
            if f"dispersion/strategies/{strategy}/energy" in handle
        }
        energies = {
            strategy: handle[f"dispersion/strategies/{strategy}/energy"][:].astype(
                np.float64
            )
            for strategy in available
        }
        theory_frequency = handle["theory/frequency_hz"][:]
        theory_velocity = handle["theory/rayleigh_mps"][:]
        quality = {
            strategy: dict(
                handle[f"quality/strategies/{strategy}/dispersion/image"].attrs
            )
            for strategy in available
        }
    full_reference, full_reference_quality = reference_dispersion_for_strategy(
        run_directory, "baseline_linear"
    )
    far_reference, far_reference_quality = reference_dispersion_for_strategy(
        run_directory, "nnws_common_linear"
    )
    panels: tuple[tuple[str, np.ndarray | None, Mapping[str, object] | None, dict[str, float] | None], ...] = (
        ("full EGF reference", full_reference.energy, full_reference_quality, None),
        *(
            (
                METHOD_LABELS[strategy],
                energies.get(strategy),
                quality.get(strategy),
                (
                    _dispersion_similarity(
                        energies[strategy], full_reference.energy, velocity
                    )
                    if strategy in available
                    else None
                ),
            )
            for strategy in full_strategies
        ),
        ("far-only EGF reference", far_reference.energy, far_reference_quality, None),
        *(
            (
                METHOD_LABELS[strategy],
                energies.get(strategy),
                quality.get(strategy),
                (
                    _dispersion_similarity(
                        energies[strategy], far_reference.energy, velocity
                    )
                    if strategy in available
                    else None
                ),
            )
            for strategy in far_strategies
        ),
    )
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(13.4, 7.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, (name, energy, panel_quality, similarity), label in zip(
        axes.flat, panels, "abcdefgh"
    ):
        if energy is None or panel_quality is None:
            axis.set_facecolor("#F1F1F1")
            axis.text(
                0.5,
                0.54,
                "not produced",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#555555",
            )
            axis.text(
                0.5,
                0.43,
                "pair-coverage gate failed",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#777777",
            )
            axis.set(xlabel="frequency (Hz)", title=name)
            _panel_label(axis, label)
            continue
        image = axis.imshow(
            energy,
            origin="lower",
            aspect="auto",
            extent=(frequency[0], frequency[-1], velocity[0], velocity[-1]),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        ambiguous = float(panel_quality["ambiguous_frequency_fraction"])
        title = f"{name}\nambiguous={ambiguous:.0%}"
        if similarity is not None:
            title += f", EGF-image CC={similarity['image_correlation']:.3f}"
        for mode_index in range(theory_velocity.shape[1]):
            axis.plot(
                theory_frequency,
                theory_velocity[:, mode_index],
                color="white",
                linestyle="--",
                linewidth=0.8,
            )
        axis.set(xlabel="frequency (Hz)", title=title)
        _panel_label(axis, label)
    colorbar = figure.colorbar(image, ax=axes, pad=0.012, shrink=0.9)
    colorbar.set_label("column-normalized energy")
    axes[0, 0].set_ylabel("phase velocity (m/s)")
    axes[1, 0].set_ylabel("phase velocity (m/s)")
    figure.suptitle(
        "Dispersion QC against the EGF-imaging resolution ceiling | dashed = theoretical modes",
        fontsize=10.5,
    )
    return _save_figure(figure, output_directory / "nnws_qc_dispersion")


def write_effectiveness_gate(
    run_directory: str | Path,
    output_path: str | Path,
) -> Path:
    """Write strict pass/fail gates that prevent one favorable metric overclaim."""
    run_directory = Path(run_directory)
    manifest = json.loads((run_directory / "manifest.json").read_text())
    comparisons = (
        ("far_offset_common_support", "nnws_pairwise", "nnws_common_linear"),
        (
            "full_aperture_hybrid",
            "nnws_hybrid_full_aperture",
            "baseline_linear",
        ),
    )
    output: dict[str, object] = {
        "rule": (
            "overall pass requires clean dispersion plus non-worse waveform "
            "correlation, phase RMS, pre-arrival energy, and spectral-shape error"
        ),
        "absolute_amplitude_tested": False,
        "comparisons": {},
    }
    comparison_output = output["comparisons"]
    assert isinstance(comparison_output, dict)
    for label, candidate_name, control_name in comparisons:
        candidate_egf = manifest["strategy_egf_quality"].get(
            candidate_name, {}
        ).get("zz_symmetric")
        control_egf = manifest["strategy_egf_quality"].get(
            control_name, {}
        ).get("zz_symmetric")
        candidate_dispersion = manifest["strategy_dispersion_quality"].get(
            candidate_name, {}
        ).get("image")
        with h5py.File(run_directory / "results.h5", "r") as handle:
            candidate_path = f"dispersion/strategies/{candidate_name}/energy"
            control_path = f"dispersion/strategies/{control_name}/energy"
            has_dispersion = candidate_path in handle and control_path in handle
        if (
            not has_dispersion
            or candidate_dispersion is None
            or candidate_egf is None
            or control_egf is None
        ):
            comparison_output[label] = {
                "candidate": candidate_name,
                "control": control_name,
                "status": "not_evaluated_pair_coverage_gate_failed",
                "checks": {"dispersion_available": False},
                "overall_pass": False,
                "candidate_values": candidate_egf or {},
                "control_values": control_egf or {},
            }
            continue
        reference_image, reference_quality = reference_dispersion_for_strategy(
            run_directory, control_name
        )
        with h5py.File(run_directory / "results.h5", "r") as handle:
            velocity = handle[
                f"dispersion/strategies/{candidate_name}/velocity_mps"
            ][:]
            candidate_energy = handle[
                f"dispersion/strategies/{candidate_name}/energy"
            ][:]
            control_energy = handle[
                f"dispersion/strategies/{control_name}/energy"
            ][:]
        candidate_dispersion_similarity = _dispersion_similarity(
            candidate_energy, reference_image.energy, velocity
        )
        control_dispersion_similarity = _dispersion_similarity(
            control_energy, reference_image.energy, velocity
        )
        checks = {
            "reference_dispersion_resolved": bool(
                reference_quality["clean_ridge_pass"]
            ),
            "dispersion_image_correlation_non_worse": (
                candidate_dispersion_similarity["image_correlation"]
                >= control_dispersion_similarity["image_correlation"]
            ),
            "dispersion_ridge_mae_non_worse": (
                candidate_dispersion_similarity["dominant_ridge_mae_to_egf_mps"]
                <= control_dispersion_similarity["dominant_ridge_mae_to_egf_mps"]
            ),
            "waveform_correlation_non_worse": (
                candidate_egf["waveform_correlation_no_shift"]
                >= control_egf["waveform_correlation_no_shift"]
            ),
            "phase_rms_non_worse": (
                candidate_egf["phase_rms_rad"] <= control_egf["phase_rms_rad"]
            ),
            "prearrival_energy_non_worse": (
                candidate_egf["prearrival_ghost_energy_fraction"]
                <= control_egf["prearrival_ghost_energy_fraction"]
            ),
            "spectral_shape_non_worse": (
                candidate_egf["spectral_shape_log_rms_after_global_scale"]
                <= control_egf["spectral_shape_log_rms_after_global_scale"]
            ),
        }
        comparison_output[label] = {
            "candidate": candidate_name,
            "control": control_name,
            "checks": checks,
            "overall_pass": all(checks.values()),
            "candidate_ambiguous_frequency_fraction": candidate_dispersion[
                "ambiguous_frequency_fraction"
            ],
            "reference_ambiguous_frequency_fraction": reference_quality[
                "ambiguous_frequency_fraction"
            ],
            "candidate_dispersion_similarity": candidate_dispersion_similarity,
            "control_dispersion_similarity": control_dispersion_similarity,
            "candidate_values": {
                "waveform_correlation_no_shift": candidate_egf[
                    "waveform_correlation_no_shift"
                ],
                "phase_rms_rad": candidate_egf["phase_rms_rad"],
                "prearrival_ghost_energy_fraction": candidate_egf[
                    "prearrival_ghost_energy_fraction"
                ],
                "spectral_shape_log_rms_after_global_scale": candidate_egf[
                    "spectral_shape_log_rms_after_global_scale"
                ],
            },
            "control_values": {
                "waveform_correlation_no_shift": control_egf[
                    "waveform_correlation_no_shift"
                ],
                "phase_rms_rad": control_egf["phase_rms_rad"],
                "prearrival_ghost_energy_fraction": control_egf[
                    "prearrival_ghost_energy_fraction"
                ],
                "spectral_shape_log_rms_after_global_scale": control_egf[
                    "spectral_shape_log_rms_after_global_scale"
                ],
            },
        }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output_path


def _load_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_multiseed_summary(
    summary_csv: str | Path,
    output_directory: str | Path,
    *,
    branch: str = "symmetric",
) -> list[Path]:
    """Plot paired multi-seed EGF metrics and NNWS-minus-linear deltas."""
    configure_figure_style()
    summary_csv = Path(summary_csv)
    output_directory = Path(output_directory)
    rows = [
        row
        for row in _load_summary_csv(summary_csv)
        if row["branch"] == branch and row["strategy"] in COMMON_STRATEGIES
    ]
    seeds = sorted({int(row["noise_seed"]) for row in rows})
    if len(seeds) < 2:
        raise ValueError("multi-seed QC requires at least two completed seeds")
    row_index = {(int(row["noise_seed"]), row["strategy"]): row for row in rows}
    metrics = (
        ("waveform_correlation_no_shift", "waveform correlation", True),
        ("phase_rms_rad", "phase RMS (rad)", False),
        ("prearrival_ghost_energy_fraction", "pre-arrival energy fraction", False),
        (
            "spectral_shape_log_rms_after_global_scale",
            "spectral-shape log RMS",
            False,
        ),
    )
    figure, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), constrained_layout=True)
    x = np.arange(len(COMMON_STRATEGIES))
    for column, (metric, ylabel, higher_is_better) in enumerate(metrics):
        top = axes[0, column]
        values_by_strategy = {
            strategy: np.asarray(
                [float(row_index[(seed, strategy)][metric]) for seed in seeds]
            )
            for strategy in COMMON_STRATEGIES
        }
        for seed_index, _seed in enumerate(seeds):
            values = [values_by_strategy[strategy][seed_index] for strategy in COMMON_STRATEGIES]
            top.plot(x, values, color="#BBBBBB", linewidth=0.65, alpha=0.75, zorder=1)
        for strategy_index, strategy in enumerate(COMMON_STRATEGIES):
            values = values_by_strategy[strategy]
            top.scatter(
                np.full(values.size, strategy_index),
                values,
                s=18,
                color=METHOD_COLORS[strategy],
                alpha=0.85,
                zorder=2,
            )
            top.plot(
                strategy_index,
                np.median(values),
                marker="_",
                markersize=14,
                markeredgewidth=2.2,
                color="#111111",
                zorder=3,
            )
        top.set_xticks(x)
        top.set_xticklabels(
            [METHOD_LABELS[item] for item in COMMON_STRATEGIES],
            rotation=22,
            rotation_mode="anchor",
            ha="right",
        )
        top.set_ylabel(ylabel)
        top.set_title("higher is better" if higher_is_better else "lower is better")
        top.grid(axis="y", alpha=0.18)
        _panel_label(top, chr(ord("a") + column))

        bottom = axes[1, column]
        delta = values_by_strategy["nnws_pairwise"] - values_by_strategy[
            "nnws_common_linear"
        ]
        bottom.axhline(0, color="#333333", linewidth=0.8)
        bottom.scatter(
            np.arange(delta.size),
            delta,
            color=METHOD_COLORS["nnws_pairwise"],
            s=22,
        )
        bottom.plot(np.arange(delta.size), delta, color="#CFCFCF", linewidth=0.65, zorder=0)
        bottom.set(
            xlabel="noise realization",
            ylabel="NNWS − common linear",
            xticks=np.arange(delta.size),
            xticklabels=[str(index + 1) for index in range(delta.size)],
        )
        improvement = delta > 0 if higher_is_better else delta < 0
        bottom.set_title(
            f"improved {np.count_nonzero(improvement)}/{delta.size}; median Δ={np.median(delta):.3g}"
        )
        bottom.grid(axis="y", alpha=0.18)
        _panel_label(bottom, chr(ord("e") + column))
    figure.suptitle(
        f"Paired multi-seed EGF validation | {branch} branch | n={len(seeds)} noise realizations",
        fontsize=10.5,
    )
    return _save_figure(figure, output_directory / "nnws_qc_multiseed_metrics")


def write_qc_manifest(
    output_directory: str | Path,
    files: Iterable[Path],
    *,
    run_directory: str | Path,
    summary_csv: str | Path | None = None,
) -> Path:
    output_directory = Path(output_directory)
    output = output_directory / "qc_manifest.json"
    payload: Mapping[str, object] = {
        "run_directory": str(Path(run_directory).resolve()),
        "multi_seed_summary_csv": (
            str(Path(summary_csv).resolve()) if summary_csv is not None else None
        ),
        "figures": [str(Path(path).resolve()) for path in files],
        "waveform_calibration": (
            "saved EGF calibration operator; FFT rectangular bandpass; no time shift"
        ),
        "waveform_display_normalization": (
            "each distance trace normalized independently for display only"
        ),
        "dispersion_display_normalization": (
            "energy is column-normalized by the WaveDispX imaging operator"
        ),
        "scientific_limit": (
            "dispersion panels are diagnostic when ambiguous_frequency_fraction is 1"
        ),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return output
