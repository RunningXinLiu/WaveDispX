#!/usr/bin/env python3
"""Audit and plot the coupled 2–12 Hz passive experiment design."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.config import ReceiverLineConfig, load_config
from wavedispx.geometry import build_geometry
from wavedispx.noise import source_spectral_amplitude
from wavedispx.preflight import build_preflight_review
from wavedispx.processing import theoretical_dispersion


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
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
    plt.close(figure)
    return [svg_path, pdf_path, png_path, tiff_path]


def _maximum_gap_deg(angle_rad: np.ndarray) -> float:
    angle = np.sort(np.mod(np.asarray(angle_rad), 2.0 * np.pi))
    gaps = np.diff(np.concatenate((angle, angle[:1] + 2.0 * np.pi)))
    return float(np.rad2deg(np.max(gaps)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/passive_nnws_v2_directional_density_2_12hz.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_python/broadband_2_12hz_system_design"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.mode != "passive" or config.noise is None or config.ambient_sources is None:
        raise ValueError("the design requires a passive configuration")
    if not isinstance(config.receivers, ReceiverLineConfig):
        raise ValueError("the current design audit requires a line array")
    geometry = build_geometry(config.receivers, config.ambient_sources)
    review = build_preflight_review(config)
    source_table = np.genfromtxt(
        config.ambient_sources.coordinates_file,
        delimiter=",",
        names=True,
    )
    group = np.asarray(source_table["source_group"], dtype=np.int64)
    background = group == 0
    sector = group == 1

    frequency = np.linspace(1.0, 14.0, 1301)
    assert np.all(np.diff(frequency) > 0.0)
    theory = theoretical_dispersion(
        config.model,
        frequency,
        rayleigh_modes=config.processing.rayleigh_modes,
        love_modes=1,
        backend=config.processing.theory_solver,
    )
    fundamental = theory.rayleigh_mps[:, 0]
    target = (
        (frequency >= config.processing.frequency_min_hz)
        & (frequency <= config.processing.frequency_max_hz)
    )
    wavelength = fundamental / frequency
    receiver_spacing = config.receivers.spacing_m
    receiver_aperture = config.receivers.stop_x_m - config.receivers.start_x_m
    samples_per_wavelength = wavelength / receiver_spacing
    aperture_wavelengths = receiver_aperture / wavelength
    source_power = source_spectral_amplitude(frequency, config.noise) ** 2
    if source_power.max(initial=0.0) > 0.0:
        source_power /= source_power.max()

    source_radius = np.hypot(geometry.source_x_m, geometry.source_y_m)
    source_angle = np.arctan2(geometry.source_y_m, geometry.source_x_m)
    mean_background_radius = float(np.mean(source_radius[background]))
    mean_background_arc_spacing = float(
        2.0 * np.pi * mean_background_radius / np.count_nonzero(background)
    )
    shortest_wavelength = float(np.min(wavelength[target]))
    longest_wavelength = float(np.max(wavelength[target]))
    source_receiver_distance = np.hypot(
        geometry.receiver_x_m[:, None] - geometry.source_x_m[None, :],
        geometry.receiver_y_m[:, None] - geometry.source_y_m[None, :],
    )
    maximum_distance = float(np.ceil(np.max(source_receiver_distance)))
    reference_frequency = max(
        config.noise.minimum_frequency_hz,
        0.25 * config.noise.dominant_frequency_hz,
    )
    required_green_window = float(
        2.0
        * maximum_distance
        / min(layer.vs_mps for layer in config.model.layers)
        / 0.862
        + 2.0 / reference_frequency
    )
    window_samples = int(round(config.noise.window_s / config.noise.dt_s))
    rfft_frequency = np.fft.rfftfreq(window_samples, config.noise.dt_s)
    transfer_frequency_count = int(
        np.count_nonzero(source_spectral_amplitude(rfft_frequency, config.noise) > 0.0)
    )
    transfer_bytes = (
        transfer_frequency_count
        * geometry.receiver_x_m.size
        * geometry.source_x_m.size
        * np.dtype(np.complex64).itemsize
    )
    transfer_gib = transfer_bytes / 1024.0**3
    window_count = int(round(config.noise.duration_s / config.noise.window_s))
    mixing_complex_products = (
        transfer_frequency_count
        * geometry.receiver_x_m.size
        * geometry.source_x_m.size
        * window_count
    )

    probe_frequency = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    probe_velocity = np.interp(probe_frequency, frequency, fundamental)
    summary = {
        "config": str(args.config.resolve()),
        "preflight_ready": review.ready,
        "preflight_checks": [asdict(check) for check in review.checks],
        "target_band_hz": [
            config.processing.frequency_min_hz,
            config.processing.frequency_max_hz,
        ],
        "receiver_count": int(geometry.receiver_x_m.size),
        "receiver_spacing_m": receiver_spacing,
        "receiver_aperture_m": receiver_aperture,
        "background_source_count": int(np.count_nonzero(background)),
        "sector_source_count": int(np.count_nonzero(sector)),
        "source_radius_range_m": [float(source_radius.min()), float(source_radius.max())],
        "maximum_source_azimuth_gap_deg": _maximum_gap_deg(source_angle),
        "mean_background_arc_spacing_m": mean_background_arc_spacing,
        "shortest_r0_wavelength_m": shortest_wavelength,
        "longest_r0_wavelength_m": longest_wavelength,
        "minimum_samples_per_r0_wavelength": float(np.min(samples_per_wavelength[target])),
        "minimum_aperture_wavelengths": float(np.min(aperture_wavelengths[target])),
        "minimum_source_receiver_distance_m": float(np.min(source_receiver_distance)),
        "maximum_source_receiver_distance_m": float(np.max(source_receiver_distance)),
        "green_response_window_required_s": required_green_window,
        "configured_window_s": config.noise.window_s,
        "frequency_resolution_hz": 1.0 / config.noise.window_s,
        "independent_window_count": window_count,
        "estimated_transfer_gib": transfer_gib,
        "estimated_mixing_complex_products": int(mixing_complex_products),
        "correlation_normalization": config.processing.correlation_normalization,
        "fundamental_velocity_mps": {
            f"{value:g}_hz": float(velocity)
            for value, velocity in zip(probe_frequency, probe_velocity)
        },
        "rayleigh_mode_cutoff_hz": [
            float(frequency[np.flatnonzero(np.isfinite(theory.rayleigh_mps[:, mode]))[0]])
            if np.any(np.isfinite(theory.rayleigh_mps[:, mode]))
            else None
            for mode in range(theory.rayleigh_mps.shape[1])
        ],
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "broadband_system_design_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    source_data_path = output / "broadband_system_frequency_data.csv"
    with source_data_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "rayleigh_r0_mps",
                "rayleigh_r1_mps",
                "rayleigh_r2_mps",
                "r0_wavelength_m",
                "receiver_samples_per_r0_wavelength",
                "aperture_r0_wavelengths",
                "relative_source_power",
            ]
        )
        for index, value in enumerate(frequency):
            writer.writerow(
                [
                    value,
                    theory.rayleigh_mps[index, 0],
                    theory.rayleigh_mps[index, 1],
                    theory.rayleigh_mps[index, 2],
                    wavelength[index],
                    samples_per_wavelength[index],
                    aperture_wavelengths[index],
                    source_power[index],
                ]
            )

    _style()
    blue = "#0072B2"
    pale_blue = "#82BFE3"
    orange = "#D55E00"
    gray = "#777777"
    light_gray = "#E8E8E8"
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)

    axes[0, 0].scatter(
        geometry.source_x_m[background] / 1000.0,
        geometry.source_y_m[background] / 1000.0,
        s=3.2,
        color="#8F8F8F",
        alpha=0.62,
        linewidth=0,
        rasterized=True,
        label=f"Uniform background (n={np.count_nonzero(background)})",
    )
    axes[0, 0].scatter(
        geometry.source_x_m[sector] / 1000.0,
        geometry.source_y_m[sector] / 1000.0,
        s=4.0,
        color=orange,
        alpha=0.72,
        linewidth=0,
        rasterized=True,
        label=f"Extra +45 deg sector (n={np.count_nonzero(sector)})",
    )
    axes[0, 0].plot(
        geometry.receiver_x_m / 1000.0,
        geometry.receiver_y_m / 1000.0,
        color="#111111",
        lw=1.4,
        label=f"Receiver line (n={geometry.receiver_x_m.size})",
    )
    axes[0, 0].scatter(
        geometry.receiver_x_m[[0, -1]] / 1000.0,
        geometry.receiver_y_m[[0, -1]] / 1000.0,
        marker="v",
        s=22,
        color="#111111",
        zorder=4,
    )
    axes[0, 0].set(
        aspect="equal",
        xlim=(-2.8, 2.8),
        ylim=(-2.8, 2.8),
        xlabel="X distance (km)",
        ylabel="Y distance (km)",
    )
    axes[0, 0].set_title("Source field and receiver geometry")
    axes[0, 0].text(
        0.04,
        0.14,
        "Uniform background\nn = 1024",
        transform=axes[0, 0].transAxes,
        va="bottom",
        color="#666666",
    )
    axes[0, 0].annotate(
        "Extra +45 deg sector\nn = 512",
        xy=(1.72, 1.72),
        xytext=(0.25, 2.42),
        arrowprops={"arrowstyle": "->", "color": orange, "lw": 0.9},
        color="#9C3F00",
    )
    axes[0, 0].text(
        0.50,
        0.43,
        "241 receivers\n5 m spacing; 1.2 km aperture",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="top",
        color="#222222",
    )
    axes[0, 0].text(
        0.97,
        0.04,
        "2.2–2.6 km source ring",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="bottom",
        color="#333333",
    )
    _panel(axes[0, 0], "a")

    axes[0, 1].axvspan(2.0, 12.0, color=light_gray, alpha=0.8, zorder=0)
    axes[0, 1].plot(frequency, theory.rayleigh_mps[:, 0], color=blue, lw=2.1, label="R0")
    axes[0, 1].plot(
        frequency,
        theory.rayleigh_mps[:, 1],
        color=pale_blue,
        lw=1.3,
        ls="--",
        label="R1",
    )
    axes[0, 1].plot(
        frequency,
        theory.rayleigh_mps[:, 2],
        color=pale_blue,
        lw=1.3,
        ls=":",
        label="R2",
    )
    axes[0, 1].set(
        xlim=(1, 14),
        ylim=(180, 480),
        xlabel="Frequency (Hz)",
        ylabel="Rayleigh phase velocity (m/s)",
    )
    axes[0, 1].set_title("20 m low-velocity layer")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].text(
        0.04,
        0.07,
        "R0: 362 m/s at 2 Hz\n203 m/s at 12 Hz",
        transform=axes[0, 1].transAxes,
        color="#174A68",
        va="bottom",
    )
    _panel(axes[0, 1], "b")

    axes[1, 0].plot(
        frequency[target],
        samples_per_wavelength[target],
        color=blue,
        lw=2.0,
        label="Receiver samples per wavelength",
    )
    axes[1, 0].plot(
        frequency[target],
        aperture_wavelengths[target],
        color=orange,
        lw=1.8,
        label="Aperture in wavelengths",
    )
    axes[1, 0].axhline(3.0, color=blue, lw=0.8, ls=":")
    axes[1, 0].axhline(5.0, color=orange, lw=0.8, ls=":")
    axes[1, 0].set(
        xlim=(2, 12),
        ylim=(0, 75),
        xlabel="Frequency (Hz)",
        ylabel="Dimensionless coverage",
    )
    axes[1, 0].set_title("Spatial sampling across the target band")
    axes[1, 0].legend(loc="upper left")
    axes[1, 0].text(
        0.98,
        0.08,
        f"minimum samples/wavelength = {np.min(samples_per_wavelength[target]):.2f}\n"
        f"minimum aperture coverage = {np.min(aperture_wavelengths[target]):.2f}",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="bottom",
        color="#333333",
    )
    _panel(axes[1, 0], "c")

    axes[1, 1].axvspan(2.0, 12.0, color=light_gray, alpha=0.8, zorder=0)
    axes[1, 1].plot(frequency, source_power, color=blue, lw=2.0)
    axes[1, 1].set(
        xlim=(0, 14),
        ylim=(-0.02, 1.12),
        xlabel="Frequency (Hz)",
        ylabel="Relative source power",
    )
    axes[1, 1].set_title("Noise spectrum and computational envelope")
    axes[1, 1].text(
        0.04,
        0.08,
        "flat power across 2–12 Hz\n"
        f"40 s/window; df = {1.0 / config.noise.window_s:.3f} Hz\n"
        f"120 windows; transfer about {transfer_gib:.2f} GiB",
        transform=axes[1, 1].transAxes,
        va="bottom",
        color="#174A68",
    )
    axes[1, 1].text(
        0.98,
        0.80,
        "raw NCF primary\n(no whitening)",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        color=gray,
    )
    _panel(axes[1, 1], "d")

    outputs = _save(figure, output / "broadband_2_12hz_system_design")
    print(
        json.dumps(
            {
                "figure_outputs": [str(path) for path in outputs],
                "source_data": str(source_data_path),
                "summary": str(summary_path),
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
