"""Required model, geometry, gather, and dispersion summary figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .active import ActiveShotGather
from .config import RunConfig
from .das import normalized_gauge_response
from .geometry import Geometry
from .processing import DispersionImage, DispersionPicks, TheoreticalDispersion
from .noise import source_spectral_amplitude


def plot_ncf_branches(axis, branches, *, title: str):
    """Display actual negative/positive NCF, never mirror the symmetric stack.

    Stored branch axes are both nonnegative lag magnitudes. Reverse the
    negative branch, omit its duplicate zero, and sort distances for display.
    """
    positive, negative = branches.positive, branches.negative
    if (not np.array_equal(positive.lag_time_s, negative.lag_time_s)
            or not np.array_equal(positive.distance_m, negative.distance_m)):
        raise ValueError("positive/negative NCF axes must match")
    lag = np.concatenate((-negative.lag_time_s[:0:-1], positive.lag_time_s))
    data = np.concatenate((negative.data[:0:-1], positive.data), axis=0)
    order = np.argsort(positive.distance_m)
    distance = positive.distance_m[order]
    scale = max(float(np.percentile(np.abs(data), 99.0)), np.finfo(float).tiny)
    artist = axis.pcolormesh(lag, distance, data[:, order].T,
                            shading="nearest", cmap="seismic", vmin=-scale, vmax=scale)
    axis.set(xlabel="Lag time (s)", ylabel="Interstation distance (m)", title=title)
    axis.set_xlim(lag[0], lag[-1])
    axis.set_ylim(distance[0], distance[-1])
    axis.axvline(0, color="0.4", linewidth=0.6, linestyle=":")
    return artist


def _plot_das_geometry(axis, geometry) -> None:
    if geometry is None:
        return
    segments = np.stack(
        (geometry.endpoint_minus_xy_m, geometry.endpoint_plus_xy_m), axis=1
    )
    collection = LineCollection(
        segments,
        colors="tab:orange",
        linewidths=1.2,
        alpha=0.8,
        label=f"DAS gauges ({geometry.gauge_label})",
    )
    axis.add_collection(collection)
    stride = max(1, geometry.channel_count // 80)
    endpoint = geometry.endpoint_xy_m[::stride]
    axis.scatter(
        endpoint[:, 0],
        endpoint[:, 1],
        s=5,
        c="tab:orange",
        alpha=0.7,
    )


def _layer_steps(config: RunConfig, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite_depth = sum(
        layer.thickness_m for layer in config.model.layers if np.isfinite(layer.thickness_m)
    )
    maximum_depth = max(40.0, 4.0 * finite_depth)
    depth = [0.0]
    expanded = []
    current = 0.0
    for layer, value in zip(config.model.layers, values):
        bottom = maximum_depth if np.isinf(layer.thickness_m) else current + layer.thickness_m
        depth.extend((current, bottom))
        expanded.extend((value, value))
        current = bottom
    return np.asarray(expanded), np.asarray(depth[1:])


def _plot_dispersion(
    axis,
    image: DispersionImage,
    theory: np.ndarray,
    picks: DispersionPicks,
    title: str,
) -> None:
    axis.imshow(
        image.energy,
        origin="lower",
        aspect="auto",
        extent=(
            image.frequency_hz[0],
            image.frequency_hz[-1],
            image.velocity_mps[0],
            image.velocity_mps[-1],
        ),
        cmap="turbo",
        vmin=0,
        vmax=1,
    )
    for mode in range(theory.shape[1]):
        axis.plot(image.frequency_hz, theory[:, mode], "w--", linewidth=1.0)
    axis.plot(picks.frequency_hz, picks.phase_velocity_mps, "k-", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Phase velocity (m/s)")


def plot_active_summary(
    config: RunConfig,
    geometry: Geometry,
    shot: ActiveShotGather,
    rayleigh: DispersionImage,
    love: DispersionImage,
    rayleigh_picks: DispersionPicks,
    love_picks: DispersionPicks,
    theory: TheoreticalDispersion,
    output_path: str | Path,
    *,
    rayleigh_combined: DispersionImage | None = None,
    rayleigh_combined_picks: DispersionPicks | None = None,
) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    vp = np.array([layer.vp_mps for layer in config.model.layers])
    vs = np.array([layer.vs_mps for layer in config.model.layers])
    density = np.array([layer.density_kgm3 for layer in config.model.layers])
    vp_step, depth = _layer_steps(config, vp)
    vs_step, _ = _layer_steps(config, vs)
    density_step, density_depth = _layer_steps(config, density)
    axes[0, 0].plot(vp_step, depth, "r-", label="$V_P$")
    axes[0, 0].plot(vs_step, depth, "b-", label="$V_S$")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(xlabel="Velocity (m/s)", ylabel="Depth (m)", title="Velocity model")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(density_step, density_depth, "k-")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(xlabel="Density (kg/m³)", ylabel="Depth (m)", title="Density model")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 2].scatter(
        geometry.receiver_x_m, geometry.receiver_y_m, marker="v", s=22, label="Receivers"
    )
    axes[0, 2].scatter(
        geometry.source_x_m, geometry.source_y_m, marker="*", s=150, c="red", label="Source"
    )
    _plot_das_geometry(
        axes[0, 2], shot.das_record.geometry if shot.das_record is not None else None
    )
    axes[0, 2].set(xlabel="x (m)", ylabel="y (m)", title="Active source–receiver geometry")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.25)
    distance_order = np.argsort(shot.distance_m)
    gather = (
        shot.das_record.data[:, distance_order]
        if shot.das_record is not None
        else shot.vertical_vertical[:, distance_order]
    )
    sorted_distance = shot.distance_m[distance_order]
    if shot.vector_xyz is not None:
        gather = shot.vector_xyz[:, distance_order, 2]
    scale = np.percentile(np.abs(gather), 99.5)
    axes[1, 0].imshow(
        gather,
        origin="upper",
        aspect="auto",
        extent=(sorted_distance[0], sorted_distance[-1], shot.time_s[-1], shot.time_s[0]),
        cmap="seismic",
        vmin=-scale,
        vmax=scale,
    )
    axes[1, 0].set(
        xlabel="Offset (m)",
        ylabel="Time (s)",
        title=(
            f"DAS {shot.das_record.measurement} shot gather"
            if shot.das_record is not None
            else "Vector-force Z shot gather" if shot.vector_xyz is not None else "Vertical shot gather"
        ),
    )
    plotted_rayleigh = rayleigh_combined or rayleigh
    plotted_rayleigh_picks = rayleigh_combined_picks or rayleigh_picks
    rayleigh_title = (
        ("DAS Rayleigh dispersion" if shot.das_record is not None else "Combined Rayleigh dispersion (RR/ZR/RZ/ZZ)")
        if rayleigh_combined is not None
        else "Rayleigh dispersion (ZZ)"
    )
    _plot_dispersion(
        axes[1, 1],
        plotted_rayleigh,
        theory.rayleigh_mps,
        plotted_rayleigh_picks,
        rayleigh_title,
    )
    if shot.das_record is not None:
        for mode in range(theory.rayleigh_mps.shape[1]):
            velocity = theory.rayleigh_mps[:, mode]
            valid = np.isfinite(velocity) & (velocity > 0)
            response = np.full(velocity.shape, np.nan)
            gauge_lengths = (
                shot.das_record.geometry.gauge_length_by_channel_m
            )
            channel_response = np.abs(
                normalized_gauge_response(
                    (
                        2.0
                        * np.pi
                        * theory.frequency_hz[valid, None]
                        / velocity[valid, None]
                    ),
                    gauge_lengths[None, :],
                )
            )
            response[valid] = np.median(channel_response, axis=1)
            axes[1, 2].plot(
                theory.frequency_hz,
                response,
                linewidth=1.8 if mode == 0 else 1.0,
                label=f"Rayleigh mode {mode} (channel median)",
            )
        axes[1, 2].set(
            xlabel="Frequency (Hz)",
            ylabel="|sinc(kL/2)|",
            title="Finite-gauge sensitivity along fibre",
            ylim=(0, 1.05),
        )
        axes[1, 2].grid(alpha=0.25)
        axes[1, 2].legend()
    else:
        _plot_dispersion(
            axes[1, 2], love, theory.love_mps, love_picks, "Love dispersion"
        )
    figure.suptitle(f"WaveDispX active | {config.name}", fontsize=15)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_dispersion_comparison(
    images: dict[str, DispersionImage],
    theory: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    """Plot component or imaging-method panels on a common color scale."""
    if not images:
        raise ValueError("images must not be empty")
    column_count = min(3, len(images))
    row_count = int(np.ceil(len(images) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.8 * column_count, 4.5 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (name, image) in zip(axes.flat, images.items()):
        axis.imshow(
            image.energy,
            origin="lower",
            aspect="auto",
            extent=(
                image.frequency_hz[0],
                image.frequency_hz[-1],
                image.velocity_mps[0],
                image.velocity_mps[-1],
            ),
            cmap="turbo",
            vmin=0,
            vmax=1,
        )
        for mode in range(theory.shape[1]):
            axis.plot(image.frequency_hz, theory[:, mode], "w--", linewidth=1.0)
        axis.set(
            xlabel="Frequency (Hz)",
            ylabel="Phase velocity (m/s)",
            title=name,
        )
    for axis in axes.flat[len(images) :]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=15)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_passive_summary(
    config: RunConfig,
    simulation,
    output_path: str | Path,
) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    vp = np.array([layer.vp_mps for layer in config.model.layers])
    vs = np.array([layer.vs_mps for layer in config.model.layers])
    density = np.array([layer.density_kgm3 for layer in config.model.layers])
    vp_step, depth = _layer_steps(config, vp)
    vs_step, _ = _layer_steps(config, vs)
    density_step, density_depth = _layer_steps(config, density)
    axes[0, 0].plot(vp_step, depth, "r-", label="$V_P$")
    axes[0, 0].plot(vs_step, depth, "b-", label="$V_S$")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(xlabel="Velocity (m/s)", ylabel="Depth (m)", title="Velocity model")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(density_step, density_depth, "k-")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(xlabel="Density (kg/m³)", ylabel="Depth (m)", title="Density model")
    axes[0, 1].grid(alpha=0.25)
    geometry = simulation.geometry
    axes[0, 2].scatter(
        geometry.source_x_m, geometry.source_y_m, marker="*", s=38, c="red", label="Sources"
    )
    axes[0, 2].scatter(
        geometry.receiver_x_m, geometry.receiver_y_m, marker="v", s=18, label="Receivers"
    )
    _plot_das_geometry(axes[0, 2], simulation.transfer.das_geometry)
    axes[0, 2].set(
        xlabel="x (m)", ylabel="y (m)", title="Passive source–receiver geometry"
    )
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].set_aspect("equal", adjustable="datalim")
    component = "das" if config.das.enabled else "zz"
    plot_ncf_branches(
        axes[1, 0],
        simulation.strategy_stacks["baseline_linear"].component_branches[component],
        title=(
            "Mixed-record two-sided DAS-DAS NCF"
            if config.das.enabled
            else "Mixed-record two-sided ZZ NCF"
            if config.ambient_sources.component == "three_component"
            else "Mixed-record two-sided NCF"
        ),
    )
    _plot_dispersion(
        axes[1, 1],
        simulation.dispersion,
        simulation.theory.rayleigh_mps,
        simulation.picks,
        (
            "Passive DAS Rayleigh dispersion"
            if config.das.enabled
            else "Passive combined Rayleigh dispersion (ZZ/RR)"
            if config.ambient_sources.component == "three_component"
            else "Passive Rayleigh dispersion"
        ),
    )
    psd_amplitude = source_spectral_amplitude(
        simulation.transfer.frequency_hz, config.noise
    )
    axes[1, 2].plot(
        simulation.transfer.frequency_hz, psd_amplitude**2, "k-", linewidth=2
    )
    axes[1, 2].fill_between(
        simulation.transfer.frequency_hz, 0, psd_amplitude**2, color="tab:blue", alpha=0.25
    )
    axes[1, 2].set(
        xlabel="Frequency (Hz)", ylabel="Normalized power", title="Ambient-source PSD"
    )
    axes[1, 2].set_ylim(bottom=0)
    axes[1, 2].grid(alpha=0.25)
    figure.suptitle(
        f"WaveDispX passive | {config.name} | {config.noise.duration_s:g} s | "
        f"{simulation.independent_window_count} independent windows | "
        f"{config.ambient_sources.component}",
        fontsize=15,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
