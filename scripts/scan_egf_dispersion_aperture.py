#!/usr/bin/env python3
"""Scan the dispersion-resolution ceiling of the active EGF versus aperture."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

from wavedispx.config import ReceiverLineConfig, load_config
from wavedispx.egf_reference import build_passive_egf_reference
from wavedispx.geometry import Geometry, receiver_positions
from wavedispx.nnws_qc import configure_figure_style
from wavedispx.processing import surface_wave_dispersion, theoretical_dispersion
from wavedispx.quality import dispersion_quality_metrics


def _apertures(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("apertures must be positive comma-separated values")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/passive_nnws_anisotropic_benchmark.toml"),
    )
    parser.add_argument(
        "--apertures-m",
        type=_apertures,
        default=_apertures("100,200,300,400,600"),
    )
    parser.add_argument("--spacing-m", type=float, default=10.0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output_python/egf_dispersion_aperture_scan"),
    )
    args = parser.parse_args()
    if args.spacing_m <= 0:
        parser.error("--spacing-m must be positive")

    base = load_config(args.config)
    processing = base.processing
    velocity = np.arange(
        processing.velocity_min_mps,
        processing.velocity_max_mps + 0.5 * processing.velocity_step_mps,
        processing.velocity_step_mps,
    )
    rows: list[dict[str, float | int | bool]] = []
    images = []
    for aperture in args.apertures_m:
        interval_count = round(aperture / args.spacing_m)
        if not np.isclose(interval_count * args.spacing_m, aperture):
            parser.error("every aperture must be an integer multiple of spacing")
        receivers = ReceiverLineConfig(
            -0.5 * aperture,
            0.5 * aperture,
            args.spacing_m,
            0.0,
        )
        config = replace(base, receivers=receivers, config_path=None)
        receiver_x, receiver_y = receiver_positions(receivers)
        geometry = Geometry(
            receiver_x,
            receiver_y,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        reference = build_passive_egf_reference(config, geometry)
        maximum_lag_s = max(
            1.3
            * aperture
            / (0.90 * min(layer.vs_mps for layer in base.model.layers)),
            aperture / processing.velocity_min_mps
            + 4.0 / processing.frequency_min_hz,
        )
        lag_count = min(
            reference.time_s.size,
            int(round(maximum_lag_s / base.noise.dt_s)) + 1,
        )
        image = surface_wave_dispersion(
            reference.components[
                :lag_count, :, reference.component_names.index("zz")
            ],
            reference.distance_m,
            base.noise.dt_s,
            velocity,
            frequency_min_hz=processing.frequency_min_hz,
            frequency_max_hz=processing.frequency_max_hz,
            method=processing.dispersion_method,
            zero_padding=processing.zero_padding,
            distance_taper_fraction=processing.distance_taper_fraction,
            frequency_block_size=base.frequency_block_size,
            radial_kernel_lookup_step=processing.radial_kernel_lookup_step,
            workers=base.workers,
        )
        theory = theoretical_dispersion(
            base.model,
            image.frequency_hz,
            rayleigh_modes=processing.rayleigh_modes,
            love_modes=processing.love_modes,
            backend=processing.theory_solver,
        )
        quality = dispersion_quality_metrics(
            image,
            theory.frequency_hz,
            theory.rayleigh_mps,
            frequency_range_hz=(
                processing.frequency_min_hz,
                processing.frequency_max_hz,
            ),
        )
        rows.append(
            {
                "aperture_m": aperture,
                "spacing_m": args.spacing_m,
                "receiver_count": receiver_x.size,
                "distance_bin_count": reference.distance_m.size,
                "dominant_ridge_fraction": quality["dominant_ridge_fraction"],
                "ambiguous_frequency_fraction": quality[
                    "ambiguous_frequency_fraction"
                ],
                "median_half_power_width_mps": quality[
                    "median_half_power_width_mps"
                ],
                "median_ridge_concentration": quality[
                    "median_ridge_concentration"
                ],
                "p10_dominance_margin": quality["p10_dominance_margin"],
                "clean_ridge_pass": quality["clean_ridge_pass"],
            }
        )
        images.append((aperture, image, theory))
        print(
            f"aperture={aperture:g} m receivers={receiver_x.size} "
            f"ambiguous={quality['ambiguous_frequency_fraction']:.1%} "
            f"half_width={quality['median_half_power_width_mps']:.1f} m/s",
            flush=True,
        )

    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "egf_dispersion_aperture_scan.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    configure_figure_style()
    figure, axes = plt.subplots(
        2,
        len(images),
        figsize=(3.05 * len(images), 6.1),
        constrained_layout=True,
    )
    for column, (aperture, image, theory) in enumerate(images):
        axis = axes[0, column]
        shown = axis.imshow(
            image.energy,
            origin="lower",
            aspect="auto",
            extent=(
                image.frequency_hz[0],
                image.frequency_hz[-1],
                image.velocity_mps[0],
                image.velocity_mps[-1],
            ),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        for mode_index in range(theory.rayleigh_mps.shape[1]):
            axis.plot(
                theory.frequency_hz,
                theory.rayleigh_mps[:, mode_index],
                "w--",
                linewidth=0.8,
            )
        axis.set_title(
            f"{aperture:g} m | {rows[column]['receiver_count']} receivers\n"
            f"ambiguous={rows[column]['ambiguous_frequency_fraction']:.0%}"
        )
        axis.set_xlabel("frequency (Hz)")
        if column == 0:
            axis.set_ylabel("phase velocity (m/s)")
        else:
            axis.tick_params(labelleft=False)
    colorbar = figure.colorbar(shown, ax=axes[0, :], pad=0.01, shrink=0.88)
    colorbar.set_label("column-normalized EGF energy")

    aperture = np.asarray([float(row["aperture_m"]) for row in rows])
    ambiguous = np.asarray(
        [float(row["ambiguous_frequency_fraction"]) for row in rows]
    )
    half_width = np.asarray(
        [float(row["median_half_power_width_mps"]) for row in rows]
    )
    axes[1, 0].plot(aperture, ambiguous, "o-", color="#C44E52")
    axes[1, 0].set(
        xlabel="receiver aperture (m)",
        ylabel="ambiguous-frequency fraction",
        ylim=(-0.03, 1.03),
        title="Absolute ridge-resolution diagnostic",
    )
    axes[1, 1].plot(aperture, half_width, "o-", color="#5875A4")
    axes[1, 1].set(
        xlabel="receiver aperture (m)",
        ylabel="median half-power width (m/s)",
        title="Velocity resolution",
    )
    for axis in axes[1, 2:]:
        axis.set_visible(False)
    for axis in axes[1, :2]:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Active-EGF dispersion resolution ceiling versus receiver aperture",
        fontsize=10.5,
    )
    figure.savefig(output_directory / "egf_dispersion_aperture_scan.svg")
    figure.savefig(output_directory / "egf_dispersion_aperture_scan.pdf")
    figure.savefig(
        output_directory / "egf_dispersion_aperture_scan.png", dpi=300
    )
    plt.close(figure)
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
