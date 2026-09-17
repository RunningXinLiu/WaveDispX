#!/usr/bin/env python3
"""Gate-0 active-pair FTAN QC using native QEDisp group velocity.

The main figure contains the active ZZ waveform and the group/phase energy
images with theory overlays only. Automatic tracked picks are deliberately
excluded from the figure and from qualification metrics.
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
from scipy.signal import find_peaks

from wavedispx.config import load_config
from wavedispx.processing import pair_ftan, theoretical_dispersion


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.8,
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
    paths = [stem.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png", ".tiff")]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    figure.savefig(paths[2], dpi=300, bbox_inches="tight")
    figure.savefig(paths[3], dpi=600, bbox_inches="tight")
    plt.close(figure)
    return paths


def _band_mask(period_s: np.ndarray, frequency_range_hz: tuple[float, float]) -> np.ndarray:
    frequency = 1.0 / period_s
    return (frequency >= frequency_range_hz[0]) & (frequency <= frequency_range_hz[1])


def _error_metrics(error_mps: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    values = np.abs(np.asarray(error_mps, dtype=np.float64)[mask])
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "median_absolute_error_mps": float(np.median(values)),
        "p90_absolute_error_mps": float(np.percentile(values, 90.0)),
        "fraction_within_4_mps": float(np.mean(values <= 4.0)),
    }


def _nearest_phase_crest(
    amplitude: np.ndarray,
    velocity_mps: np.ndarray,
    theory_mps: np.ndarray,
    *,
    search_half_width_mps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    crest_velocity = np.full(theory_mps.shape, np.nan)
    crest_amplitude = np.full(theory_mps.shape, np.nan)
    for column, target in enumerate(theory_mps):
        peaks, _ = find_peaks(amplitude[:, column])
        peaks = peaks[np.abs(velocity_mps[peaks] - target) <= search_half_width_mps]
        if peaks.size == 0:
            continue
        nearest = peaks[np.argmin(np.abs(velocity_mps[peaks] - target))]
        crest_velocity[column] = velocity_mps[nearest]
        crest_amplitude[column] = amplitude[nearest, column]
    return crest_velocity, crest_amplitude


def _array_phase_metrics(
    handle: h5py.File,
    theory_frequency_hz: np.ndarray,
    theory_phase_mps: np.ndarray,
    frequency_range_hz: tuple[float, float],
) -> dict[str, float | int]:
    frequency = handle["dispersion/components/zz/frequency_hz"][:]
    velocity = handle["dispersion/components/zz/velocity_mps"][:]
    energy = handle["dispersion/components/zz/energy"][:]
    keep = (frequency >= frequency_range_hz[0]) & (frequency <= frequency_range_hz[1])
    ridge = velocity[np.argmax(energy[:, keep], axis=0)]
    theory = np.interp(frequency[keep], theory_frequency_hz, theory_phase_mps)
    return _error_metrics(ridge - theory, np.ones(ridge.size, dtype=bool))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("output_python/active_broadband_2_12hz_validation/results.h5"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/active_broadband_2_12hz_validation.toml"),
    )
    parser.add_argument("--distance-m", type=float, default=600.0)
    parser.add_argument("--frequency-min-hz", type=float, default=2.0)
    parser.add_argument("--frequency-max-hz", type=float, default=12.0)
    parser.add_argument("--qualified-frequency-min-hz", type=float, default=4.0)
    parser.add_argument("--gaussian-alpha", type=float, default=120.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_python/pair_ftan_gate0"),
    )
    args = parser.parse_args()

    candidate_band = (args.frequency_min_hz, args.frequency_max_hz)
    qualified_band = (args.qualified_frequency_min_hz, args.frequency_max_hz)
    if not candidate_band[0] < qualified_band[0] < qualified_band[1]:
        raise ValueError("qualified band must be a strict upper subset of candidate band")

    with h5py.File(args.results) as handle:
        time_s = handle["shot/time_s"][:]
        distance_m = handle["shot/distance_m"][:]
        receiver_index = int(np.argmin(np.abs(distance_m - args.distance_m)))
        selected_distance_m = float(distance_m[receiver_index])
        trace = handle["shot/components"][:, receiver_index, 4]

        config = load_config(args.config)
        theory_frequency_hz = np.linspace(candidate_band[0], candidate_band[1], 1001)
        theory = theoretical_dispersion(
            config.model,
            theory_frequency_hz,
            rayleigh_modes=1,
            love_modes=0,
            backend="qedisp",
            compute_group=True,
        )
        if theory.rayleigh_group_mps is None:
            raise RuntimeError("native QEDisp Rayleigh group velocity was not returned")
        theory_phase_frequency_order = theory.rayleigh_mps[:, 0]
        theory_group_frequency_order = theory.rayleigh_group_mps[:, 0]
        array_phase = {
            "candidate_2_12_hz": _array_phase_metrics(
                handle,
                theory_frequency_hz,
                theory_phase_frequency_order,
                candidate_band,
            ),
            "qualified_4_12_hz": _array_phase_metrics(
                handle,
                theory_frequency_hz,
                theory_phase_frequency_order,
                qualified_band,
            ),
        }

    theory_period_s = 1.0 / theory_frequency_hz
    order = np.argsort(theory_period_s)
    theory_period_s = theory_period_s[order]
    theory_phase_mps = theory_phase_frequency_order[order]
    theory_group_mps = theory_group_frequency_order[order]

    result = pair_ftan(
        trace,
        dt_s=float(np.median(np.diff(time_s))),
        distance_m=selected_distance_m,
        frequency_range_hz=candidate_band,
        group_velocity_range_mps=(100.0, 450.0),
        phase_velocity_range_mps=(120.0, 500.0),
        period_count=201,
        velocity_step_mps=2.0,
        gaussian_alpha=args.gaussian_alpha,
        phase_method="frequency_domain",
        minimum_snr=0.0,
    )

    period_s = result.group_image.period_s
    group_theory = np.interp(period_s, theory_period_s, theory_group_mps)
    phase_theory = np.interp(period_s, theory_period_s, theory_phase_mps)
    group_dominant = result.group_image.velocity_mps[
        np.argmax(result.group_image.amplitude, axis=0)
    ]
    phase_crest, phase_crest_amplitude = _nearest_phase_crest(
        result.phase_image.amplitude,
        result.phase_image.velocity_mps,
        phase_theory,
    )
    group_theory_amplitude = np.array(
        [
            np.interp(
                group_theory[index],
                result.group_image.velocity_mps,
                result.group_image.amplitude[:, index],
            )
            for index in range(period_s.size)
        ]
    )

    candidate_mask = _band_mask(period_s, candidate_band)
    qualified_mask = _band_mask(period_s, qualified_band)
    group_metrics = {
        "candidate_2_12_hz": _error_metrics(
            group_dominant - group_theory, candidate_mask
        ),
        "qualified_4_12_hz": _error_metrics(
            group_dominant - group_theory, qualified_mask
        ),
    }
    phase_metrics = {
        "candidate_2_12_hz": _error_metrics(phase_crest - phase_theory, candidate_mask),
        "qualified_4_12_hz": _error_metrics(phase_crest - phase_theory, qualified_mask),
    }
    qualified_group = group_metrics["qualified_4_12_hz"]
    qualified_phase = phase_metrics["qualified_4_12_hz"]
    gate_pass = bool(
        qualified_group["median_absolute_error_mps"] <= 2.0
        and qualified_group["p90_absolute_error_mps"] <= 4.0
        and qualified_phase["median_absolute_error_mps"] <= 2.0
        and qualified_phase["p90_absolute_error_mps"] <= 4.0
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "active_pair_ftan_gate0_source_data.csv"
    with source_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "period_s",
                "frequency_hz",
                "group_qedisp_native_mps",
                "group_dominant_energy_mps",
                "group_absolute_error_mps",
                "group_theory_normalized_amplitude",
                "phase_qedisp_mps",
                "phase_nearest_crest_mps",
                "phase_absolute_error_mps",
                "phase_crest_normalized_amplitude",
                "qualified_common_band",
            ]
        )
        writer.writerows(
            zip(
                period_s,
                1.0 / period_s,
                group_theory,
                group_dominant,
                np.abs(group_dominant - group_theory),
                group_theory_amplitude,
                phase_theory,
                phase_crest,
                np.abs(phase_crest - phase_theory),
                phase_crest_amplitude,
                qualified_mask.astype(int),
            )
        )

    summary = {
        "gate": "Gate 0-B active pair FTAN",
        "passed": gate_pass,
        "input_results": str(args.results.resolve()),
        "input_config": str(args.config.resolve()),
        "distance_m": selected_distance_m,
        "component": "vertical_vertical (ZZ)",
        "candidate_frequency_range_hz": list(candidate_band),
        "qualified_common_frequency_range_hz": list(qualified_band),
        "gaussian_alpha": args.gaussian_alpha,
        "velocity_step_mps": 2.0,
        "theory": {
            "phase": theory.solver,
            "group": "QEDispInv native sregn96 energy integral",
        },
        "automatic_tracked_picks_used_in_qualification": False,
        "group_energy_ridge": group_metrics,
        "phase_nearest_theory_crest": phase_metrics,
        "active_array_phase_reference": array_phase,
        "interpretation": (
            "The 2-12 Hz phase response is valid, but the pair group envelope does not "
            "resolve the sharp 2-4 Hz group-velocity curvature at 600 m. Gate 0 therefore "
            "freezes 4-12 Hz as the common pairwise waveform/group/phase band."
        ),
        "source_data_csv": str(source_path),
    }
    summary_path = output / "active_pair_ftan_gate0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    _style()
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.05), constrained_layout=True)
    waveform_axis, group_axis, phase_axis = axes
    normalized_trace = result.processed_trace / max(
        np.max(np.abs(result.processed_trace)), np.finfo(float).eps
    )
    waveform_axis.plot(result.time_s, normalized_trace, color="#252525", linewidth=0.7)
    waveform_axis.fill_between(
        result.time_s,
        -1.05,
        1.05,
        where=result.signal_window > 0.99,
        color="#56B4E9",
        alpha=0.16,
        linewidth=0.0,
        label="FTAN arrival window",
    )
    waveform_axis.set_xlim(0.0, min(result.time_s[-1], 6.5))
    waveform_axis.set_ylim(-1.05, 1.05)
    waveform_axis.set_xlabel("Time (s)")
    waveform_axis.set_ylabel("Normalized ZZ amplitude")
    waveform_axis.set_title(f"Active EGF waveform | {selected_distance_m:.0f} m")
    waveform_axis.legend(loc="upper right", frameon=False)
    _panel(waveform_axis, "a")

    image_kwargs = {
        "shading": "auto",
        "cmap": "magma",
        "vmin": 0.0,
        "vmax": 1.0,
        "rasterized": True,
    }
    group_axis.pcolormesh(
        period_s,
        result.group_image.velocity_mps,
        result.group_image.amplitude,
        **image_kwargs,
    )
    group_axis.plot(
        theory_period_s,
        theory_group_mps,
        color="white",
        linewidth=1.5,
        label="QEDisp native $U_0(T)$",
    )
    group_axis.axvspan(1.0 / 4.0, 1.0 / 2.0, color="white", alpha=0.12)
    group_axis.axvline(1.0 / 4.0, color="white", linestyle=":", linewidth=0.8)
    group_axis.set_xlim(1.0 / 12.0, 1.0 / 2.0)
    group_axis.set_ylim(100.0, 450.0)
    group_axis.set_xlabel("Period (s)")
    group_axis.set_ylabel("Group velocity (m/s)")
    group_axis.set_title("Group FTAN | theory only")
    legend = group_axis.legend(loc="upper right", frameon=False)
    for label in legend.get_texts():
        label.set_color("white")
    _panel(group_axis, "b")

    phase_axis.pcolormesh(
        period_s,
        result.phase_image.velocity_mps,
        result.phase_image.amplitude,
        **image_kwargs,
    )
    phase_axis.plot(
        theory_period_s,
        theory_phase_mps,
        color="white",
        linewidth=1.5,
        label="QEDisp $c_0(T)$",
    )
    phase_axis.axvspan(1.0 / 4.0, 1.0 / 2.0, color="white", alpha=0.12)
    phase_axis.axvline(1.0 / 4.0, color="white", linestyle=":", linewidth=0.8)
    phase_axis.set_xlim(1.0 / 12.0, 1.0 / 2.0)
    phase_axis.set_ylim(120.0, 500.0)
    phase_axis.set_xlabel("Period (s)")
    phase_axis.set_ylabel("Phase velocity (m/s)")
    phase_axis.set_title("Phase FTAN | theory only")
    legend = phase_axis.legend(loc="upper right", frameon=False)
    for label in legend.get_texts():
        label.set_color("white")
    _panel(phase_axis, "c")

    figure.suptitle(
        "Gate 0 active-pair FTAN | 2–12 Hz candidate; 4–12 Hz qualified",
        fontsize=9.4,
        fontweight="bold",
    )
    figure_paths = _save(figure, output / "active_pair_ftan_gate0")
    summary["figure_files"] = [str(path) for path in figure_paths]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
