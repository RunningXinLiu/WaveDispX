#!/usr/bin/env python3
"""Plot and audit a proposed 1-D model before an expensive passive run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.config import ModelConfig
from wavedispx.processing import theoretical_dispersion


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.2,
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
        -0.16,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def _load(path: Path) -> tuple[dict, ModelConfig]:
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    return document, ModelConfig.from_root(document)


def _poisson_ratio(vp_mps: float, vs_mps: float) -> float:
    vp2 = vp_mps**2
    vs2 = vs_mps**2
    return (vp2 - 2.0 * vs2) / (2.0 * (vp2 - vs2))


def _finite_cutoff(frequency: np.ndarray, curve: np.ndarray) -> float | None:
    valid = np.isfinite(curve)
    return float(frequency[valid][0]) if np.any(valid) else None


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


def _step_profile(model: ModelConfig, maximum_depth_m: float) -> tuple[np.ndarray, ...]:
    depth = [0.0]
    vp = []
    vs = []
    density = []
    top = 0.0
    for layer in model.layers:
        bottom = (
            maximum_depth_m
            if math.isinf(layer.thickness_m)
            else min(maximum_depth_m, top + layer.thickness_m)
        )
        depth.extend([top, bottom])
        vp.extend([layer.vp_mps, layer.vp_mps])
        vs.extend([layer.vs_mps, layer.vs_mps])
        density.extend([layer.density_kgm3, layer.density_kgm3])
        top = bottom
        if top >= maximum_depth_m:
            break
    return (
        np.asarray(depth[1:], dtype=np.float64),
        np.asarray(vp, dtype=np.float64),
        np.asarray(vs, dtype=np.float64),
        np.asarray(density, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model_designs/rayleigh_h20_two_layer.toml"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("examples/passive_nnws_v2_directional_density_1000m.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_python/model_design_rayleigh_h20"),
    )
    args = parser.parse_args()

    proposed_document, proposed_model = _load(args.model)
    _, baseline_model = _load(args.baseline)
    design = proposed_document["design"]
    frequency = np.linspace(
        float(design["frequency_min_hz"]),
        float(design["frequency_max_hz"]),
        1401,
    )
    target_min = float(design["target_frequency_min_hz"])
    target_max = float(design["target_frequency_max_hz"])
    receiver_spacing = float(design["receiver_spacing_m"])

    proposed = theoretical_dispersion(
        proposed_model, frequency, rayleigh_modes=3, love_modes=1
    )
    baseline = theoretical_dispersion(
        baseline_model, frequency, rayleigh_modes=1, love_modes=1
    )
    fundamental = proposed.rayleigh_mps[:, 0]
    baseline_fundamental = baseline.rayleigh_mps[:, 0]
    target = (frequency >= target_min) & (frequency <= target_max)
    low = (frequency >= target_min) & (frequency <= 6.0)
    high = (frequency >= 7.0) & (frequency <= target_max)
    low_slope = float(np.polyfit(frequency[low], fundamental[low], 1)[0])
    high_slope = float(np.polyfit(frequency[high], fundamental[high], 1)[0])
    baseline_high_slope = float(
        np.polyfit(frequency[high], baseline_fundamental[high], 1)[0]
    )
    assert np.all(np.diff(frequency) > 0.0)
    target_velocity = np.interp(
        np.array([target_min, 6.0, 8.0, target_max]),
        frequency,
        fundamental,
    )
    shortest_wavelength = float(np.nanmin(fundamental[target] / frequency[target]))
    spacing_fraction = receiver_spacing / shortest_wavelength

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_data = output / "model_dispersion_source_data.csv"
    with source_data.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "proposed_rayleigh_r0_mps",
                "proposed_rayleigh_r1_mps",
                "proposed_rayleigh_r2_mps",
                "proposed_love_l0_mps",
                "baseline_rayleigh_r0_mps",
            ]
        )
        for index, value in enumerate(frequency):
            writer.writerow(
                [
                    value,
                    proposed.rayleigh_mps[index, 0],
                    proposed.rayleigh_mps[index, 1],
                    proposed.rayleigh_mps[index, 2],
                    proposed.love_mps[index, 0],
                    baseline_fundamental[index],
                ]
            )

    summary = {
        "model_name": str(design["name"]),
        "model_file": str(args.model.resolve()),
        "baseline_file": str(args.baseline.resolve()),
        "layers": [
            {
                "thickness_m": (
                    "inf" if math.isinf(layer.thickness_m) else layer.thickness_m
                ),
                "vp_mps": layer.vp_mps,
                "vs_mps": layer.vs_mps,
                "density_kgm3": layer.density_kgm3,
                "poisson_ratio": _poisson_ratio(layer.vp_mps, layer.vs_mps),
            }
            for layer in proposed_model.layers
        ],
        "target_band_hz": [target_min, target_max],
        "fundamental_velocity_mps": {
            f"{value:g}_hz": float(velocity)
            for value, velocity in zip(
                [target_min, 6.0, 8.0, target_max], target_velocity
            )
        },
        "fundamental_monotonic_in_target_band": bool(
            np.all(np.diff(fundamental[target]) <= 1.0e-8)
        ),
        "low_band_slope_mps_per_hz": low_slope,
        "high_band_slope_mps_per_hz": high_slope,
        "absolute_high_to_low_slope_ratio": abs(high_slope / low_slope),
        "baseline_high_band_slope_mps_per_hz": baseline_high_slope,
        "rayleigh_mode_cutoff_hz": [
            _finite_cutoff(frequency, proposed.rayleigh_mps[:, mode])
            for mode in range(proposed.rayleigh_mps.shape[1])
        ],
        "shortest_fundamental_wavelength_m_in_target_band": shortest_wavelength,
        "receiver_spacing_m": receiver_spacing,
        "receiver_spacing_over_shortest_wavelength": spacing_fraction,
        "sampling_note": (
            "passes the two-samples-per-wavelength limit but is marginal; "
            "5 m spacing is preferred for the final validation"
        ),
        "solver": proposed.solver,
    }
    summary_path = output / "model_dispersion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    _style()
    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#8A8A8A"
    pale_blue = "#8CC7E8"
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.75),
        gridspec_kw={"width_ratios": [0.82, 1.15, 1.25]},
        constrained_layout=True,
    )

    depth, vp, vs, _ = _step_profile(proposed_model, 60.0)
    axes[0].plot(vs, depth, color=blue, lw=2.0, label="Vs")
    axes[0].plot(vp, depth, color=orange, lw=1.7, label="Vp")
    axes[0].axhline(20.0, color="#B8B8B8", lw=0.8, ls=":")
    axes[0].set(xlim=(150, 850), ylim=(60, 0), xlabel="Velocity (m/s)", ylabel="Depth (m)")
    axes[0].set_title("Proposed 1-D structure")
    axes[0].legend(loc="lower right")
    axes[0].text(
        0.04,
        0.96,
        "0–20 m\nVs = 220 m/s\nrho = 1900 kg/m³",
        transform=axes[0].transAxes,
        va="top",
        color="#333333",
    )
    axes[0].text(
        0.04,
        0.33,
        "Half-space\nVs = 440 m/s\nrho = 2150 kg/m³",
        transform=axes[0].transAxes,
        va="top",
        color="#333333",
    )
    _panel(axes[0], "a")

    axes[1].axvspan(target_min, target_max, color="#E8E8E8", alpha=0.75, zorder=0)
    axes[1].plot(
        frequency,
        fundamental,
        color=blue,
        lw=2.1,
        label="Rayleigh R0",
    )
    for mode, linestyle in ((1, "--"), (2, ":")):
        axes[1].plot(
            frequency,
            proposed.rayleigh_mps[:, mode],
            color=pale_blue,
            lw=1.2,
            ls=linestyle,
            label=f"Rayleigh R{mode}",
        )
    axes[1].set(
        xlim=(frequency[0], frequency[-1]),
        ylim=(180, 480),
        xlabel="Frequency (Hz)",
        ylabel="Phase velocity (m/s)",
    )
    axes[1].set_title("Full theoretical modal structure")
    axes[1].legend(loc="upper right")
    axes[1].text(
        (target_min + target_max) / 2.0,
        186,
        "target band",
        ha="center",
        va="bottom",
        color="#555555",
    )
    _panel(axes[1], "b")

    axes[2].plot(
        frequency[target],
        baseline_fundamental[target],
        color=gray,
        lw=1.5,
        ls="--",
        label="Current 10 m layer",
    )
    axes[2].plot(
        frequency[target],
        fundamental[target],
        color=blue,
        lw=2.2,
        label="Proposed 20 m layer",
    )
    sample_frequency = np.array([target_min, 6.0, 8.0, target_max])
    axes[2].scatter(sample_frequency, target_velocity, s=15, color=blue, zorder=3)
    axes[2].set(
        xlim=(target_min, target_max),
        ylim=(165, 380),
        xlabel="Frequency (Hz)",
        ylabel="R0 phase velocity (m/s)",
    )
    axes[2].set_title("Fundamental mode in target band")
    axes[2].legend(loc="upper right")
    axes[2].text(
        0.04,
        0.07,
        "Proposed: 322 → 205 m/s\n"
        f"high-band slope = {high_slope:.1f} m/s/Hz\n"
        f"dx/lambda_min = {spacing_fraction:.2f}",
        transform=axes[2].transAxes,
        color="#174A68",
        va="bottom",
    )
    _panel(axes[2], "c")

    outputs = _save(figure, output / "proposed_1d_model_dispersion")
    print(json.dumps({"figure_outputs": [str(path) for path in outputs], **summary}, indent=2))


if __name__ == "__main__":
    main()
