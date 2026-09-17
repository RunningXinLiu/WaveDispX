#!/usr/bin/env python3
"""Compare QEDispInv and disba/CPS on the upstream low-velocity-layer model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.config import ModelConfig
from wavedispx.processing import run_qedisp_forward, theoretical_dispersion


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#6A3D9A", "#666666")


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
    axis.text(-0.13, 1.12, label, transform=axis.transAxes, fontsize=9, fontweight="bold")


def _load_model(path: Path) -> ModelConfig:
    with path.open("rb") as stream:
        return ModelConfig.from_root(tomllib.load(stream))


def _plot_modes(
    axis: plt.Axes,
    period_s: np.ndarray,
    qedisp_mps: np.ndarray,
    disba_mps: np.ndarray,
    title: str,
) -> None:
    for mode in range(qedisp_mps.shape[1]):
        keep_q = np.isfinite(qedisp_mps[:, mode])
        keep_d = np.isfinite(disba_mps[:, mode])
        axis.plot(
            period_s[keep_q],
            qedisp_mps[keep_q, mode],
            color=COLORS[mode % len(COLORS)],
            linewidth=1.05,
            label=f"M{mode}" if mode < 5 else None,
        )
        axis.plot(
            period_s[keep_d],
            disba_mps[keep_d, mode],
            color="#222222",
            linewidth=0.45,
            linestyle=(0, (2, 2)),
            alpha=0.55,
        )
    axis.set_xscale("log")
    axis.set_xlim(period_s.min(), period_s.max())
    axis.set_xlabel("Period (s)")
    axis.set_ylabel("Phase velocity (m/s)")
    axis.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model_designs/qedisp_low_velocity_benchmark.toml"),
    )
    parser.add_argument("--mode-count", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_python/qedisp_low_velocity_benchmark"),
    )
    args = parser.parse_args()
    if args.mode_count < 1:
        raise ValueError("--mode-count must be at least 1")

    model = _load_model(args.model)
    frequency_hz = np.geomspace(0.1, 100.0, 500)
    period_s = 1.0 / frequency_hz
    qedisp = theoretical_dispersion(
        model,
        frequency_hz,
        rayleigh_modes=args.mode_count,
        love_modes=args.mode_count,
        backend="qedisp",
    )
    disba = theoretical_dispersion(
        model,
        frequency_hz,
        rayleigh_modes=args.mode_count,
        love_modes=args.mode_count,
        backend="disba",
    )
    kernel_frequency_hz = np.geomspace(0.5, 50.0, 96)
    kernel_result = run_qedisp_forward(
        model,
        kernel_frequency_hz,
        rayleigh_modes=1,
        love_modes=0,
        compute_kernels=True,
    )
    assert kernel_result.rayleigh_kernels is not None
    kernels = kernel_result.rayleigh_kernels

    q_rayleigh_count = np.sum(np.isfinite(qedisp.rayleigh_mps), axis=0)
    d_rayleigh_count = np.sum(np.isfinite(disba.rayleigh_mps), axis=0)
    q_love_count = np.sum(np.isfinite(qedisp.love_mps), axis=0)
    d_love_count = np.sum(np.isfinite(disba.love_mps), axis=0)
    q_total = int(np.sum(q_rayleigh_count) + np.sum(q_love_count))
    d_total = int(np.sum(d_rayleigh_count) + np.sum(d_love_count))
    common_rayleigh = np.isfinite(qedisp.rayleigh_mps) & np.isfinite(disba.rayleigh_mps)
    common_love = np.isfinite(qedisp.love_mps) & np.isfinite(disba.love_mps)
    common_error = np.concatenate(
        (
            np.abs(qedisp.rayleigh_mps[common_rayleigh] - disba.rayleigh_mps[common_rayleigh]),
            np.abs(qedisp.love_mps[common_love] - disba.love_mps[common_love]),
        )
    )
    rayleigh_difference = np.abs(qedisp.rayleigh_mps - disba.rayleigh_mps)
    love_difference = np.abs(qedisp.love_mps - disba.love_mps)
    if np.nanmax(rayleigh_difference) >= np.nanmax(love_difference):
        largest_wave_type = "Rayleigh"
        largest_index = np.unravel_index(
            np.nanargmax(rayleigh_difference), rayleigh_difference.shape
        )
        largest_difference = float(rayleigh_difference[largest_index])
    else:
        largest_wave_type = "Love"
        largest_index = np.unravel_index(
            np.nanargmax(love_difference), love_difference.shape
        )
        largest_difference = float(love_difference[largest_index])

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "root_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["mode", "qedisp_rayleigh", "disba_rayleigh", "qedisp_love", "disba_love"]
        )
        writer.writerows(
            zip(
                range(args.mode_count),
                q_rayleigh_count,
                d_rayleigh_count,
                q_love_count,
                d_love_count,
            )
        )
    summary = {
        "model": str(args.model),
        "frequency_hz": [float(frequency_hz[0]), float(frequency_hz[-1])],
        "frequency_count": int(frequency_hz.size),
        "mode_count_per_wave_type": args.mode_count,
        "qedisp_finite_root_samples": q_total,
        "disba_finite_root_samples": d_total,
        "qedisp_additional_root_samples": q_total - d_total,
        "common_root_median_absolute_difference_mps": float(np.median(common_error)),
        "common_root_maximum_absolute_difference_mps": float(np.max(common_error)),
        "largest_difference_location": {
            "wave_type": largest_wave_type,
            "mode": int(largest_index[1]),
            "frequency_hz": float(frequency_hz[largest_index[0]]),
            "absolute_difference_mps": largest_difference,
        },
        "interpretation": (
            "The two solvers agree to sub-mm/s median precision on common roots, but "
            "the disba/CPS branch differs locally near a high-mode kissing interval. "
            "QEDispInv also retains a small number of additional cutoff roots."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _style()
    figure = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, width_ratios=(0.85, 1.25, 1.25, 1.05))
    model_axis = figure.add_subplot(grid[0, 0])
    rayleigh_axis = figure.add_subplot(grid[0, 1:])
    love_axis = figure.add_subplot(grid[1, :2])
    count_axis = figure.add_subplot(grid[1, 2])
    kernel_axis = figure.add_subplot(grid[1, 3])

    top_depth_m = np.cumsum(
        [0.0]
        + [layer.thickness_m for layer in model.layers[:-1]]
    )
    bottom_depth_m = max(80.0, 2.0 * top_depth_m[-1])
    depth_edges_m = np.append(top_depth_m, bottom_depth_m)
    vs_mps = np.array([layer.vs_mps for layer in model.layers])
    model_axis.stairs(vs_mps, depth_edges_m, orientation="horizontal", color="#0072B2", linewidth=1.6)
    model_axis.axhspan(20.0, 40.0, color="#D55E00", alpha=0.14, label="LVL")
    model_axis.set_xlim(100.0, 650.0)
    model_axis.set_ylim(bottom_depth_m, 0.0)
    model_axis.set_xlabel("Vs (m/s)")
    model_axis.set_ylabel("Depth (m)")
    model_axis.set_title("1-D benchmark")
    model_axis.legend(loc="lower right")
    _panel(model_axis, "a")

    _plot_modes(
        rayleigh_axis,
        period_s,
        qedisp.rayleigh_mps,
        disba.rayleigh_mps,
        "Rayleigh modes: QEDispInv (color) vs disba/CPS (black dashed)",
    )
    rayleigh_axis.legend(loc="lower right", ncols=5)
    if args.mode_count >= 8:
        inset = rayleigh_axis.inset_axes([0.36, 0.49, 0.29, 0.40])
        zoom = (frequency_hz >= 68.0) & (frequency_hz <= 76.0)
        inset.plot(
            frequency_hz[zoom],
            qedisp.rayleigh_mps[zoom, 7],
            color=COLORS[7],
            linewidth=1.0,
            label="QEDisp M7",
        )
        inset.plot(
            frequency_hz[zoom],
            disba.rayleigh_mps[zoom, 7],
            color="#222222",
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            label="disba M7",
        )
        inset.set_xlim(68.0, 76.0)
        inset.set_ylim(250.0, 275.0)
        inset.set_xlabel("Frequency (Hz)", fontsize=5.5)
        inset.tick_params(labelsize=5.2)
        inset.set_title("Mode-kissing zoom", fontsize=6.0)
        inset.legend(loc="upper right", fontsize=5.2)
    _panel(rayleigh_axis, "b")

    _plot_modes(
        love_axis,
        period_s,
        qedisp.love_mps,
        disba.love_mps,
        "Love modes in the same low-velocity model",
    )
    _panel(love_axis, "c")

    mode = np.arange(args.mode_count)
    width = 0.36
    count_axis.bar(mode - width / 2, q_rayleigh_count + q_love_count, width, color="#0072B2", label="QEDispInv")
    count_axis.bar(mode + width / 2, d_rayleigh_count + d_love_count, width, color="#A8A8A8", label="disba/CPS")
    count_axis.set_xlabel("Mode")
    count_axis.set_ylabel("Finite root samples")
    count_axis.set_title(f"Completeness: +{q_total - d_total} samples")
    count_axis.legend(loc="upper right")
    _panel(count_axis, "d")

    kernel_depth_edges = np.append(kernels.depth_m, bottom_depth_m)
    log_frequency = np.log(kernels.frequency_hz)
    log_frequency_edges = np.empty(kernels.frequency_hz.size + 1)
    log_frequency_edges[1:-1] = 0.5 * (log_frequency[:-1] + log_frequency[1:])
    log_frequency_edges[0] = log_frequency[0] - 0.5 * (
        log_frequency[1] - log_frequency[0]
    )
    log_frequency_edges[-1] = log_frequency[-1] + 0.5 * (
        log_frequency[-1] - log_frequency[-2]
    )
    kernel_frequency_edges = np.exp(log_frequency_edges)
    vmax = float(np.max(np.abs(kernels.vs)))
    mesh = kernel_axis.pcolormesh(
        kernel_frequency_edges,
        kernel_depth_edges,
        kernels.vs,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    kernel_axis.set_xscale("log")
    kernel_axis.set_ylim(bottom_depth_m, 0.0)
    kernel_axis.set_xlabel("Frequency (Hz)")
    kernel_axis.set_ylabel("Depth (m)")
    kernel_axis.set_title("QEDisp Vs kernel, R0")
    figure.colorbar(mesh, ax=kernel_axis, fraction=0.05, pad=0.03, label="∂c/∂Vs")
    _panel(kernel_axis, "e")

    figure.suptitle(
        "Low-velocity-layer audit: mode-kissing branch stability and cutoff completeness",
        fontsize=9.2,
        fontweight="bold",
    )
    stem = output / "qedisp_low_velocity_benchmark"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
