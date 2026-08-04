"""Required model, geometry, gather, and dispersion summary figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .active import ActiveShotGather
from .config import RunConfig
from .geometry import Geometry
from .processing import DispersionImage, DispersionPicks, TheoreticalDispersion
from .noise import source_spectral_amplitude


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
    axes[0, 2].set(xlabel="x (m)", ylabel="y (m)", title="Active source–receiver geometry")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.25)
    distance_order = np.argsort(shot.distance_m)
    gather = shot.vertical_vertical[:, distance_order]
    sorted_distance = shot.distance_m[distance_order]
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
    axes[1, 0].set(xlabel="Offset (m)", ylabel="Time (s)", title="Vertical shot gather")
    plotted_rayleigh = rayleigh_combined or rayleigh
    plotted_rayleigh_picks = rayleigh_combined_picks or rayleigh_picks
    rayleigh_title = (
        "Combined Rayleigh dispersion (RR/ZR/RZ/ZZ)"
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
    _plot_dispersion(axes[1, 2], love, theory.love_mps, love_picks, "Love dispersion")
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
    axes[0, 2].set(
        xlabel="x (m)", ylabel="y (m)", title="Passive source–receiver geometry"
    )
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].set_aspect("equal", adjustable="datalim")
    ncf = simulation.ncf.data
    ncf_scale = np.percentile(np.abs(ncf), 99.0)
    axes[1, 0].imshow(
        ncf,
        origin="lower",
        aspect="auto",
        extent=(
            simulation.ncf.distance_m[0],
            simulation.ncf.distance_m[-1],
            simulation.ncf.lag_time_s[0],
            simulation.ncf.lag_time_s[-1],
        ),
        cmap="seismic",
        vmin=-ncf_scale,
        vmax=ncf_scale,
    )
    axes[1, 0].set(
        xlabel="Interstation distance (m)",
        ylabel="Lag time (s)",
        title=(
            "Mixed-record symmetric ZZ NCF"
            if config.ambient_sources.component == "three_component"
            else "Mixed-record symmetric NCF"
        ),
    )
    _plot_dispersion(
        axes[1, 1],
        simulation.dispersion,
        simulation.theory.rayleigh_mps,
        simulation.picks,
        (
            "Passive combined Rayleigh dispersion (ZZ/RR)"
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
