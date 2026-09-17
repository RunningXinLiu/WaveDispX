#!/usr/bin/env python3
"""QC the 2–12 Hz active-source observation system without hiding weak bands."""

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

from wavedispx.active import active_source_wavelet
from wavedispx.config import load_config
from wavedispx.processing import pick_dispersion_ridge, theoretical_dispersion


METHODS = ("phase_shift", "bessel", "hankel")
METHOD_LABELS = {
    "phase_shift": "Phase shift",
    "bessel": "Bessel",
    "hankel": "Hankel",
}
METHOD_COLORS = {
    "phase_shift": "#0072B2",
    "bessel": "#D55E00",
    "hankel": "#009E73",
}


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


def _run_data(config_path: Path, run_directory: Path) -> dict:
    config = load_config(config_path)
    manifest = json.loads((run_directory / "manifest.json").read_text())
    with h5py.File(run_directory / "results.h5") as handle:
        time_s = handle["shot/time_s"][:]
        vertical = handle["shot/components"][:, :, 3:5]
        spectrum_frequency = np.fft.rfftfreq(
            time_s.size, config.active_source.dt_s
        )
        source_power = np.abs(
            np.fft.rfft(
                active_source_wavelet(
                    time_s,
                    config.active_source.wavelet,
                    config.active_source.dominant_frequency_hz,
                )
            )
        ) ** 2
        receiver_power = np.mean(
            np.abs(np.fft.rfft(vertical, axis=0)) ** 2,
            axis=(1, 2),
        )
        band = (
            (spectrum_frequency >= config.processing.frequency_min_hz)
            & (spectrum_frequency <= config.processing.frequency_max_hz)
        )
        source_peak = float(np.max(source_power[band]))
        receiver_peak = float(np.max(receiver_power[band]))
        source_power = source_power / (source_peak if source_peak > 0.0 else 1.0)
        receiver_power = receiver_power / (
            receiver_peak if receiver_peak > 0.0 else 1.0
        )
        images = {}
        picks = {}
        errors = {}
        for method in METHODS:
            group = handle[f"dispersion/method_comparison/{method}"]
            images[method] = {
                "energy": group["energy"][:],
                "frequency_hz": group["frequency_hz"][:],
                "velocity_mps": group["velocity_mps"][:],
            }
            image = images[method]
            pick = pick_dispersion_ridge(
                image["energy"],
                image["frequency_hz"],
                image["velocity_mps"],
                frequency_range_hz=(
                    config.processing.frequency_min_hz,
                    config.processing.frequency_max_hz,
                ),
                velocity_range_mps=(
                    config.processing.velocity_min_mps,
                    config.processing.velocity_max_mps,
                ),
                smoothness=config.processing.picker_smoothness,
                maximum_jump_mps=config.processing.picker_maximum_jump_mps,
                minimum_energy=config.processing.picker_minimum_energy,
            )
            theory_frequency = handle["theory/frequency_hz"][:]
            theory_r0 = handle["theory/rayleigh_mps"][:, 0]
            assert np.all(np.diff(theory_frequency) > 0.0)
            reference = np.interp(
                pick.frequency_hz, theory_frequency, theory_r0
            )
            picks[method] = pick
            errors[method] = pick.phase_velocity_mps - reference
        timings = {
            key.removeprefix("timing_").removesuffix("_s"): float(value)
            for key, value in handle.attrs.items()
            if key.startswith("timing_") and key.endswith("_s")
        }
    return {
        "config": config,
        "manifest": manifest,
        "spectrum_frequency_hz": spectrum_frequency,
        "source_power": source_power,
        "receiver_power": receiver_power,
        "images": images,
        "picks": picks,
        "errors": errors,
        "timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--broad-config",
        type=Path,
        default=Path("examples/active_broadband_2_12hz_validation.toml"),
    )
    parser.add_argument(
        "--broad-run",
        type=Path,
        default=Path("output_python/active_broadband_2_12hz_validation"),
    )
    parser.add_argument(
        "--low-config",
        type=Path,
        default=Path("examples/active_low_2_6hz_validation.toml"),
    )
    parser.add_argument(
        "--low-run",
        type=Path,
        default=Path("output_python/active_low_2_6hz_validation"),
    )
    parser.add_argument(
        "--high-config",
        type=Path,
        default=Path("examples/active_high_6_12hz_validation.toml"),
    )
    parser.add_argument(
        "--high-run",
        type=Path,
        default=Path("output_python/active_high_6_12hz_validation"),
    )
    parser.add_argument(
        "--passive-design-summary",
        type=Path,
        default=Path(
            "output_python/broadband_2_12hz_system_design/"
            "broadband_system_design_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_python/active_broadband_2_12hz_qc"),
    )
    args = parser.parse_args()

    broad = _run_data(args.broad_config, args.broad_run)
    low = _run_data(args.low_config, args.low_run)
    high = _run_data(args.high_config, args.high_run)
    passive_design = json.loads(args.passive_design_summary.read_text())
    reference_frequency = np.linspace(2.0, 12.0, 1001)
    theory = theoretical_dispersion(
        low["config"].model,
        reference_frequency,
        rayleigh_modes=3,
        love_modes=1,
        backend=low["config"].processing.theory_solver,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pick_rows = []
    method_mae = {"impulse": {}, "low": {}, "high": {}}
    for run_name, run in (("impulse", broad), ("low", low), ("high", high)):
        for method in METHODS:
            pick = run["picks"][method]
            error = run["errors"][method]
            method_mae[run_name][method] = float(np.mean(np.abs(error)))
            pick_rows.extend(
                (
                    run_name,
                    method,
                    frequency,
                    velocity,
                    difference,
                    energy,
                    uncertainty,
                )
                for frequency, velocity, difference, energy, uncertainty in zip(
                    pick.frequency_hz,
                    pick.phase_velocity_mps,
                    error,
                    pick.normalized_energy,
                    pick.half_power_uncertainty_mps,
                )
            )
    pick_path = output / "active_pick_error_source_data.csv"
    with pick_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shot_band",
                "method",
                "frequency_hz",
                "picked_velocity_mps",
                "signed_error_mps",
                "normalized_energy",
                "half_power_uncertainty_mps",
            ]
        )
        writer.writerows(pick_rows)

    spectrum_path = output / "active_spectral_coverage_source_data.csv"
    with spectrum_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shot",
                "frequency_hz",
                "band_normalized_source_power",
                "band_normalized_receiver_power",
            ]
        )
        for name, run in (("broad", broad), ("low", low), ("high", high)):
            writer.writerows(
                (
                    name,
                    frequency,
                    source_power,
                    receiver_power,
                )
                for frequency, source_power, receiver_power in zip(
                    run["spectrum_frequency_hz"],
                    run["source_power"],
                    run["receiver_power"],
                )
            )

    def band_minimum(run: dict) -> float:
        config = run["config"]
        keep = (
            (run["spectrum_frequency_hz"] >= config.processing.frequency_min_hz)
            & (run["spectrum_frequency_hz"] <= config.processing.frequency_max_hz)
        )
        return float(np.min(run["receiver_power"][keep]))

    def band_coverage(run: dict, minimum_normalized_power: float = 0.01) -> float:
        config = run["config"]
        keep = (
            (run["spectrum_frequency_hz"] >= config.processing.frequency_min_hz)
            & (run["spectrum_frequency_hz"] <= config.processing.frequency_max_hz)
        )
        return float(np.mean(run["receiver_power"][keep] >= minimum_normalized_power))

    summary = {
        "ideal_impulse_calibration": {
            "wavelet": broad["config"].active_source.wavelet,
            "band_hz": [2.0, 12.0],
            "band_normalized_minimum_receiver_power": band_minimum(broad),
            "receiver_band_fraction_above_minus_20_db": band_coverage(broad),
            "mae_mps": method_mae["impulse"],
            "primary_quality": broad["manifest"]["quality"]["rayleigh_combined_image"],
            "runtime_s": broad["timings"],
        },
        "two_shot_validation": {
            "low_dominant_frequency_hz": low["config"].active_source.dominant_frequency_hz,
            "high_dominant_frequency_hz": high["config"].active_source.dominant_frequency_hz,
            "low_band_normalized_minimum_receiver_power": band_minimum(low),
            "high_band_normalized_minimum_receiver_power": band_minimum(high),
            "mae_mps": method_mae,
            "low_runtime_s": low["timings"],
            "high_runtime_s": high["timings"],
            "primary_low_quality": low["manifest"]["quality"]["rayleigh_combined_image"],
            "primary_high_quality": high["manifest"]["quality"]["rayleigh_combined_image"],
        },
        "passive_design_estimate": {
            "transfer_gib": passive_design["estimated_transfer_gib"],
            "mixing_complex_products": passive_design[
                "estimated_mixing_complex_products"
            ],
        },
    }
    summary_path = output / "active_broadband_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    _style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)

    axes[0, 0].axvspan(2.0, 6.0, color="#D8ECF7", alpha=0.65, zorder=0)
    axes[0, 0].axvspan(6.0, 12.0, color="#F7E3D8", alpha=0.65, zorder=0)
    for run, color, label in (
        (broad, "#111111", "Ideal unit impulse"),
        (low, "#0072B2", "Low-band 3.8 Hz shot"),
        (high, "#D55E00", "High-band 8.8 Hz shot"),
    ):
        positive_epsilon = np.finfo(float).tiny
        power_db = 10.0 * np.log10(
            np.maximum(run["receiver_power"], positive_epsilon)
        )
        axes[0, 0].plot(
            run["spectrum_frequency_hz"],
            power_db,
            color=color,
            lw=1.7 if run is broad else 1.2,
            ls="-" if run is broad else "--",
            label=label,
        )
    axes[0, 0].set(
        xlim=(1.5, 12.5),
        ylim=(-60.0, 2.0),
        xlabel="Frequency (Hz)",
        ylabel="Band-normalized receiver power (dB)",
    )
    axes[0, 0].set_title("Raw recorded spectral coverage")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].text(
        0.04,
        0.05,
        "No per-frequency normalization\n"
        "Unit-impulse source spectrum is flat (0 dB)",
        transform=axes[0, 0].transAxes,
        va="bottom",
        color="#333333",
    )
    _panel(axes[0, 0], "a")

    image = broad["images"]["phase_shift"]
    axes[0, 1].pcolormesh(
        image["frequency_hz"],
        image["velocity_mps"],
        image["energy"],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        rasterized=True,
    )
    axes[0, 1].plot(
        reference_frequency,
        theory.rayleigh_mps[:, 0],
        color="white",
        lw=1.2,
        label="Theory R0",
    )
    axes[0, 1].plot(
        reference_frequency,
        theory.rayleigh_mps[:, 1],
        color="white",
        lw=0.8,
        ls="--",
        alpha=0.85,
        label="Theory R1",
    )
    axes[0, 1].plot(
        reference_frequency,
        theory.rayleigh_mps[:, 2],
        color="white",
        lw=0.8,
        ls=":",
        alpha=0.85,
        label="Theory R2",
    )
    pick = broad["picks"]["phase_shift"]
    axes[0, 1].plot(
        pick.frequency_hz,
        pick.phase_velocity_mps,
        color="#111111",
        lw=0.75,
    )
    axes[0, 1].set(
        xlim=(2.0, 12.0),
        ylim=(150, 500),
        xlabel="Frequency (Hz)",
        ylabel="Phase velocity (m/s)",
    )
    axes[0, 1].set_title("Unit-impulse phase-shift dispersion")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].text(
        0.04,
        0.05,
        "Black: automatic R0 pick\nColumn-normalized image; raw coverage is in a",
        transform=axes[0, 1].transAxes,
        color="white",
        va="bottom",
    )
    _panel(axes[0, 1], "b")

    for method in METHODS:
        axes[1, 0].plot(
            broad["picks"][method].frequency_hz,
            broad["errors"][method],
            color=METHOD_COLORS[method],
            lw=1.2,
            label=METHOD_LABELS[method],
        )
    axes[1, 0].axhline(0.0, color="#777777", lw=0.8, ls=":")
    axes[1, 0].axvline(6.0, color="#B8B8B8", lw=0.8, ls="--")
    axes[1, 0].set(
        xlim=(2.0, 12.0),
        ylim=(-8.0, 8.0),
        xlabel="Frequency (Hz)",
        ylabel="Picked R0 minus theory (m/s)",
    )
    axes[1, 0].set_title("Signed pick error")
    axes[1, 0].legend(loc="lower right")
    _panel(axes[1, 0], "c")

    x = np.arange(len(METHODS), dtype=np.float64)
    width = 0.24
    axes[1, 1].bar(
        x - width,
        [method_mae["impulse"][method] for method in METHODS],
        width,
        color="#222222",
        label="Unit impulse",
    )
    axes[1, 1].bar(
        x,
        [method_mae["low"][method] for method in METHODS],
        width,
        color="#56B4E9",
        label="2–6 Hz shot",
    )
    axes[1, 1].bar(
        x + width,
        [method_mae["high"][method] for method in METHODS],
        width,
        color="#E69F00",
        label="6–12 Hz shot",
    )
    axes[1, 1].set_xticks(x, [METHOD_LABELS[method] for method in METHODS])
    axes[1, 1].set(
        ylim=(0, 1.15),
        ylabel="R0 mean absolute error (m/s)",
    )
    axes[1, 1].set_title("Cross-method agreement and runtime")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].text(
        0.04,
        0.94,
        f"Unit impulse: {broad['timings']['total']:.2f} s total\n"
        f"Ricker controls: {low['timings']['total']:.2f} / "
        f"{high['timings']['total']:.2f} s\n"
        f"Impulse coverage > -20 dB: {100.0 * band_coverage(broad):.1f}%",
        transform=axes[1, 1].transAxes,
        va="top",
        color="#333333",
    )
    _panel(axes[1, 1], "d")

    outputs = _save(figure, output / "active_broadband_2_12hz_qc")
    print(
        json.dumps(
            {
                "figure_outputs": [str(path) for path in outputs],
                "pick_source_data": str(pick_path),
                "spectrum_source_data": str(spectrum_path),
                "summary": str(summary_path),
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
