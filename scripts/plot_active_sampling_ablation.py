#!/usr/bin/env python3
"""Test spatial-alias sensitivity by decimating one broadband active gather."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.config import load_config
from wavedispx.processing import (
    DispersionImage,
    combine_dispersion_images,
    pick_dispersion_ridge,
    surface_wave_dispersion,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "examples/active_broadband_2_12hz_validation.toml"
RUN_DIRECTORY = ROOT / "output_python/active_broadband_2_12hz_validation"
OUTPUT_DIRECTORY = ROOT / "output_python/active_sampling_ablation"
DECIMATION_FACTORS = (1, 2, 4)
DISPLAY_VELOCITY_MIN_MPS = 150.0
DISPLAY_VELOCITY_MAX_MPS = 1500.0
DISPLAY_VELOCITY_STEP_MPS = 4.0


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _phase_shift_image(
    components: np.ndarray,
    distance_m: np.ndarray,
    dt_s: float,
    velocity_mps: np.ndarray,
    config,
) -> DispersionImage:
    common = dict(
        frequency_min_hz=config.processing.frequency_min_hz,
        frequency_max_hz=config.processing.frequency_max_hz,
        method="phase_shift",
        zero_padding=config.processing.zero_padding,
        distance_taper_fraction=config.processing.distance_taper_fraction,
        frequency_block_size=config.frequency_block_size,
        workers=config.workers,
    )
    rz = surface_wave_dispersion(
        components[:, :, 3],
        distance_m,
        dt_s,
        velocity_mps,
        component="radial",
        **common,
    )
    zz = surface_wave_dispersion(
        components[:, :, 4],
        distance_m,
        dt_s,
        velocity_mps,
        component="vertical",
        **common,
    )
    return combine_dispersion_images((rz, zz))


def _alias_safe_frequency(
    frequency_hz: np.ndarray,
    phase_velocity_mps: np.ndarray,
    spacing_m: float,
) -> float:
    valid = np.isfinite(phase_velocity_mps)
    safe = valid & (spacing_m <= phase_velocity_mps / (2.0 * frequency_hz))
    if not np.any(safe):
        return 0.0
    return float(np.max(frequency_hz[safe]))


def _predicted_alias_velocity(
    frequency_hz: np.ndarray,
    phase_velocity_mps: np.ndarray,
    spacing_m: float,
) -> np.ndarray:
    wavenumber = 2.0 * np.pi * frequency_hz / phase_velocity_mps
    aliased_wavenumber = np.abs(wavenumber - 2.0 * np.pi / spacing_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = 2.0 * np.pi * frequency_hz / aliased_wavenumber
    spatially_aliased = spacing_m > phase_velocity_mps / (2.0 * frequency_hz)
    velocity[~spatially_aliased] = np.nan
    return velocity


def _alias_energy_ratio(
    image: DispersionImage,
    theory_frequency_hz: np.ndarray,
    theory_velocity_mps: np.ndarray,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    theory_at_image = np.interp(
        image.frequency_hz, theory_frequency_hz, theory_velocity_mps
    )
    alias_velocity = _predicted_alias_velocity(
        image.frequency_hz, theory_at_image, spacing_m
    )
    ratios = np.full(image.frequency_hz.size, np.nan, dtype=np.float64)
    for frequency_index, (target, alias) in enumerate(
        zip(theory_at_image, alias_velocity)
    ):
        if not np.isfinite(alias) or not (
            image.velocity_mps[0] <= alias <= image.velocity_mps[-1]
        ):
            continue
        target_index = int(np.argmin(np.abs(image.velocity_mps - target)))
        alias_index = int(np.argmin(np.abs(image.velocity_mps - alias)))
        target_energy = image.energy[target_index, frequency_index]
        if target_energy > 0.0:
            ratios[frequency_index] = (
                image.energy[alias_index, frequency_index] / target_energy
            )
    return alias_velocity, ratios


def main() -> None:
    config = load_config(CONFIG_PATH)
    with h5py.File(RUN_DIRECTORY / "results.h5") as handle:
        components = handle["shot/components"][:]
        distance_m = handle["shot/distance_m"][:]
        theory_frequency = handle["theory/frequency_hz"][:]
        theory_rayleigh = handle["theory/rayleigh_mps"][:]
    assert np.all(np.diff(theory_frequency) > 0.0)

    velocity_count = int(
        round(
            (DISPLAY_VELOCITY_MAX_MPS - DISPLAY_VELOCITY_MIN_MPS)
            / DISPLAY_VELOCITY_STEP_MPS
        )
    )
    velocity_mps = (
        DISPLAY_VELOCITY_MIN_MPS
        + DISPLAY_VELOCITY_STEP_MPS * np.arange(velocity_count + 1)
    )
    base_spacing_m = config.receivers.spacing_m
    results = []
    for factor in DECIMATION_FACTORS:
        subset = slice(None, None, factor)
        image = _phase_shift_image(
            components[:, subset, :],
            distance_m[subset],
            config.active_source.dt_s,
            velocity_mps,
            config,
        )
        pick = pick_dispersion_ridge(
            image.energy,
            image.frequency_hz,
            image.velocity_mps,
            frequency_range_hz=(2.0, 12.0),
            velocity_range_mps=(150.0, 500.0),
            smoothness=config.processing.picker_smoothness,
            maximum_jump_mps=config.processing.picker_maximum_jump_mps,
            minimum_energy=config.processing.picker_minimum_energy,
        )
        theory_pick = np.interp(
            pick.frequency_hz, theory_frequency, theory_rayleigh[:, 0]
        )
        error = pick.phase_velocity_mps - theory_pick
        spacing_m = base_spacing_m * factor
        alias_safe_hz = _alias_safe_frequency(
            theory_frequency, theory_rayleigh[:, 0], spacing_m
        )
        if alias_safe_hz >= 11.99:
            alias_safe_hz = 12.0
        alias_velocity, alias_ratio = _alias_energy_ratio(
            image,
            theory_frequency,
            theory_rayleigh[:, 0],
            spacing_m,
        )
        high = pick.frequency_hz >= 8.0
        valid_alias = np.isfinite(alias_ratio)
        results.append(
            {
                "factor": factor,
                "spacing_m": spacing_m,
                "receiver_count": int(distance_m[subset].size),
                "image": image,
                "pick": pick,
                "error": error,
                "alias_safe_hz": alias_safe_hz,
                "alias_velocity_mps": alias_velocity,
                "alias_energy_ratio": alias_ratio,
                "median_alias_to_r0_energy_ratio": (
                    float(np.median(alias_ratio[valid_alias]))
                    if np.any(valid_alias)
                    else None
                ),
                "p95_alias_to_r0_energy_ratio": (
                    float(np.quantile(alias_ratio[valid_alias], 0.95))
                    if np.any(valid_alias)
                    else None
                ),
                "mae_mps": float(np.mean(np.abs(error))),
                "high_band_mae_mps": float(np.mean(np.abs(error[high]))),
                "high_band_max_abs_error_mps": float(np.max(np.abs(error[high]))),
            }
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIRECTORY / "active_sampling_ablation_source_data.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "spacing_m",
                "receiver_count",
                "alias_safe_frequency_hz",
                "frequency_hz",
                "picked_velocity_mps",
                "theory_r0_mps",
                "signed_error_mps",
                "normalized_pick_energy",
                "predicted_alias_velocity_mps",
                "alias_to_r0_energy_ratio",
            ]
        )
        for result in results:
            pick = result["pick"]
            theory_pick = pick.phase_velocity_mps - result["error"]
            writer.writerows(
                (
                    result["spacing_m"],
                    result["receiver_count"],
                    result["alias_safe_hz"],
                    frequency,
                    velocity,
                    theory_velocity,
                    error,
                    energy,
                    alias_velocity,
                    alias_ratio,
                )
                for frequency, velocity, theory_velocity, error, energy,
                alias_velocity, alias_ratio in zip(
                    pick.frequency_hz,
                    pick.phase_velocity_mps,
                    theory_pick,
                    result["error"],
                    pick.normalized_energy,
                    np.interp(
                        pick.frequency_hz,
                        result["image"].frequency_hz,
                        result["alias_velocity_mps"],
                    ),
                    np.interp(
                        pick.frequency_hz,
                        result["image"].frequency_hz,
                        result["alias_energy_ratio"],
                    ),
                )
            )

    summary = {
        "source_wavelet": config.active_source.wavelet,
        "target_band_hz": [2.0, 12.0],
        "cases": [
            {
                key: value
                for key, value in result.items()
                if key
                in {
                    "spacing_m",
                    "receiver_count",
                    "alias_safe_hz",
                    "mae_mps",
                    "high_band_mae_mps",
                    "high_band_max_abs_error_mps",
                    "median_alias_to_r0_energy_ratio",
                    "p95_alias_to_r0_energy_ratio",
                }
            }
            for result in results
        ],
    }
    summary_path = OUTPUT_DIRECTORY / "active_sampling_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    _style()
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    for panel_index, (axis, result) in enumerate(zip(axes, results)):
        image = result["image"]
        axis.pcolormesh(
            image.frequency_hz,
            image.velocity_mps,
            image.energy,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            rasterized=True,
        )
        axis.plot(
            theory_frequency,
            theory_rayleigh[:, 0],
            color="white",
            lw=1.2,
            label="Theory R0",
        )
        axis.plot(
            result["pick"].frequency_hz,
            result["pick"].phase_velocity_mps,
            color="#111111",
            lw=0.75,
            label="Automatic pick",
        )
        valid_alias_curve = np.isfinite(result["alias_velocity_mps"])
        axis.plot(
            image.frequency_hz[valid_alias_curve],
            result["alias_velocity_mps"][valid_alias_curve],
            color="#56B4E9",
            lw=1.0,
            ls=":",
            label="Predicted alias",
        )
        if result["alias_safe_hz"] < 11.99:
            axis.axvline(
                result["alias_safe_hz"],
                color="#56B4E9",
                lw=1.0,
                ls="--",
                label="Nyquist limit",
            )
        axis.set(
            xlim=(2.0, 12.0),
            ylim=(DISPLAY_VELOCITY_MIN_MPS, DISPLAY_VELOCITY_MAX_MPS),
            xlabel="Frequency (Hz)",
            title=(
                f"{chr(97 + panel_index)}  {result['spacing_m']:.0f} m spacing; "
                f"{result['receiver_count']} receivers"
            ),
        )
        if panel_index == 0:
            axis.set_ylabel("Phase velocity (m/s)")
        axis.text(
            0.04,
            0.05,
            f"MAE: {result['mae_mps']:.2f} m/s\n"
            f"8–12 Hz MAE: {result['high_band_mae_mps']:.2f} m/s\n"
            + (
                "No in-band alias predicted"
                if result["median_alias_to_r0_energy_ratio"] is None
                else (
                    "P95 alias/R0 energy: "
                    f"{result['p95_alias_to_r0_energy_ratio']:.2f}"
                )
            ),
            transform=axis.transAxes,
            color="white",
            va="bottom",
        )
        legend = axis.legend(loc="upper right")
        for text_item in legend.get_texts():
            text_item.set_color("white")

    stem = OUTPUT_DIRECTORY / "active_sampling_ablation"
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
    print(
        json.dumps(
            {
                "figure_outputs": [str(path) for path in outputs],
                "source_data": str(csv_path),
                "summary": str(summary_path),
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
