#!/usr/bin/env python3
"""Plot v5.1 auxiliary-array direction, waveform, and pair-FTAN QC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wavedispx.beamforming_selection import circular_distance_deg
from wavedispx.processing import pair_ftan

from plot_gate2_sparse_event_qc import _bandpass_trace, _theory
from run_gate2_pair_selection_experiment import EVALUATION_BAND_HZ, PAIR_DISTANCE_M, VELOCITY_RANGE_MPS
from run_gate2_source_aware_beamforming_experiment import SECTOR_AUDIT_HALF_WIDTH_DEG


METHODS = (
    ("active_reference", "Active EGF"),
    ("paired_clean_counterfactual", "Paired clean"),
    ("linear", "Linear"),
    ("beam_uniformity", "Beam uniformity"),
    ("oracle_event_uniformity", "Event-truth oracle"),
)


def _style() -> None:
    """Apply explicit publication-safe settings for standalone source QA."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.7,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(figure: plt.Figure, stem: Path) -> None:
    """Export an editable-vector plus high-resolution raster QC bundle."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def _ftan(npz: np.lib.npyio.NpzFile):
    common = {
        "distance_m": PAIR_DISTANCE_M,
        "frequency_range_hz": EVALUATION_BAND_HZ,
        "group_velocity_range_mps": VELOCITY_RANGE_MPS,
        "phase_velocity_range_mps": VELOCITY_RANGE_MPS,
        "period_count": 121,
        "velocity_step_mps": 2.0,
        "gaussian_alpha": 120.0,
        "phase_method": "frequency_domain",
        "minimum_snr": 0.0,
    }
    active_time = npz["active_reference_time_s"]
    output = {
        "active_reference": pair_ftan(
            npz["active_reference_zz"],
            dt_s=float(np.median(np.diff(active_time))),
            input_kind="egf",
            **common,
        )
    }
    lag = npz["lag_time_s"]
    for method, _ in METHODS[1:]:
        key = f"stack__{method}__symmetric"
        output[method] = (
            pair_ftan(
                npz[key],
                dt_s=float(np.median(np.diff(lag))),
                input_kind="correlation",
                **common,
            )
            if key in npz.files
            else None
        )
    return output


def _waveform_ftan(
    run_root: Path,
    case_dir: Path,
    scenario: str,
    seed: int,
    output: Path,
) -> None:
    with np.load(case_dir / "gate2_source_aware_beamforming.npz") as npz:
        lag = npz["lag_time_s"]
        lag_dt = float(np.median(np.diff(lag)))
        active_time = npz["active_reference_time_s"]
        active = _bandpass_trace(
            npz["active_reference_zz"], float(np.median(np.diff(active_time)))
        )
        results = _ftan(npz)
        waveforms: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
            "active_reference": (active_time, active)
        }
        for method, _ in METHODS[1:]:
            key = f"stack__{method}__symmetric"
            waveforms[method] = (
                (lag, _bandpass_trace(-np.gradient(npz[key], lag_dt), lag_dt))
                if key in npz.files
                else None
            )
    metrics = pd.read_csv(run_root / "gate2_source_aware_metrics.csv")
    metrics = metrics[
        (metrics.wavefield_seed == seed)
        & (metrics.scenario == scenario)
        & (metrics.control_replica == -1)
        & (metrics.branch == "symmetric")
    ]
    theory_period, theory_group, theory_phase = _theory(case_dir, seed)
    figure, axes = plt.subplots(
        3,
        len(METHODS),
        figsize=(12.0, 6.4),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )
    active_normalized = active / max(np.max(np.abs(active)), np.finfo(float).eps)
    image = None
    for column, (method, label) in enumerate(METHODS):
        result = results[method]
        waveform = waveforms[method]
        if result is None or waveform is None:
            message = "not applicable" if scenario == "uniform" else "missing"
            for row in range(3):
                axes[row, column].text(0.5, 0.5, message, ha="center", va="center", transform=axes[row, column].transAxes)
            axes[0, column].set_title(label)
            continue
        time, trace = waveform
        normalized = trace / max(np.max(np.abs(trace)), np.finfo(float).eps)
        axes[0, column].plot(time, normalized, color="#0072B2", linewidth=0.8)
        axes[0, column].plot(active_time, active_normalized, "--", color="#333333", linewidth=0.55, alpha=0.7)
        axes[0, column].axvspan(
            PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[1],
            PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[0],
            color="#F0E442",
            alpha=0.12,
        )
        axes[0, column].set_xlim(0.0, 5.25)
        axes[0, column].set_ylim(-1.05, 1.05)
        title = label
        if method != "active_reference":
            row = metrics[metrics.method == method]
            if not row.empty:
                title += f"\nCC={row.iloc[0].waveform_correlation_no_shift:.3f}"
        axes[0, column].set_title(title)
        image = axes[1, column].pcolormesh(
            result.group_image.period_s,
            result.group_image.velocity_mps,
            result.group_image.amplitude,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            rasterized=True,
        )
        axes[1, column].plot(theory_period, theory_group, color="cyan", linewidth=1.0)
        axes[2, column].pcolormesh(
            result.phase_image.period_s,
            result.phase_image.velocity_mps,
            result.phase_image.amplitude,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            rasterized=True,
        )
        axes[2, column].plot(theory_period, theory_phase, color="cyan", linewidth=1.0)
        axes[2, column].set_xlabel("Period (s)")
    axes[0, 0].set_ylabel("Normalized waveform")
    axes[1, 0].set_ylabel("Group velocity (m/s)")
    axes[2, 0].set_ylabel("Phase velocity (m/s)")
    axes[1, 0].set_ylim(VELOCITY_RANGE_MPS)
    axes[2, 0].set_ylim(VELOCITY_RANGE_MPS)
    if image is not None:
        figure.colorbar(image, ax=axes[1:, :], label="Per-period normalized energy", shrink=0.72)
    figure.suptitle(
        f"Gate 2-S v5.1 target-pair QC | {scenario} | seed {seed}\n"
        "Waveform: 4–12 Hz and −d/dt NCF; cyan: QEDisp fundamental Rayleigh; no automatic picks",
        fontsize=10,
    )
    _save(figure, output / f"{scenario}_seed{seed}_waveform_ftan")


def _site_contribution(
    events: pd.DataFrame,
    weights: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    selected = weights[weights.method == method][["window_index", "weight"]]
    merged = events.merge(selected, on="window_index", how="left")
    merged["contribution"] = np.square(merged.event_weight) * merged.weight
    return (
        merged.groupby(["source_index", "source_group", "x_m", "y_m"])
        .contribution.sum()
        .reset_index()
    )


def _direction_source_figure(
    run_root: Path,
    case_dir: Path,
    scenario: str,
    center_deg: float,
    seed: int,
    output: Path,
) -> None:
    with np.load(case_dir / "gate2_source_aware_beamforming.npz") as npz:
        azimuth = npz["beam_azimuth_deg"]
        beam = npz["beam_angular_power"]
        truth = npz["truth_angular_power"]
        sector_count = npz["sector_event_count"]
    with np.load(
        run_root / f"seed_{seed}" / "uniform" / "gate2_source_aware_beamforming.npz"
    ) as uniform_npz:
        uniform_beam = np.mean(uniform_npz["beam_angular_power"], axis=0)
        uniform_truth = np.mean(uniform_npz["truth_angular_power"], axis=0)
    events = pd.read_csv(case_dir / "source_events.csv")
    weights = pd.read_csv(run_root / "gate2_source_aware_window_weights.csv")
    weights = weights[(weights.wavefield_seed == seed) & (weights.scenario == scenario)]
    receivers = np.genfromtxt(case_dir / "receivers.csv", delimiter=",", names=True)
    figure, axes = plt.subplots(2, 4, figsize=(12.5, 6.4), constrained_layout=True)
    methods = (("linear", "Raw event energy"), ("beam_uniformity", "Beam-weighted"), ("oracle_event_uniformity", "Event-oracle weighted"))
    scatter = None
    for panel, (method, title) in enumerate(methods):
        axis = axes[0, panel]
        site = _site_contribution(events, weights, method)
        normalized = site.contribution / max(float(site.contribution.max()), np.finfo(float).eps)
        scatter = axis.scatter(site.x_m, site.y_m, c=normalized, s=4 + 22 * normalized, cmap="viridis", vmin=0, vmax=1, alpha=0.8, linewidths=0, rasterized=True)
        sector = site[site.source_group == 1]
        axis.scatter(sector.x_m, sector.y_m, facecolors="none", edgecolors="#D62728", s=32, linewidths=0.4)
        axis.scatter(receivers["x_m"][:2], receivers["y_m"][:2], marker="v", s=28, c="#D55E00", label="Target pair")
        axis.scatter(receivers["x_m"][2:], receivers["y_m"][2:], marker="+", s=10, c="#111111", linewidths=0.45, label="Beam array" if panel == 0 else None)
        axis.set_aspect("equal")
        axis.set_xlim(-1900, 1900)
        axis.set_ylim(-1900, 1900)
        axis.set_title(title)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        if panel == 0:
            axis.legend(frameon=False, loc="lower left")
    axes[0, 3].axis("off")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes[0, :3], label="Normalized source-site contribution", shrink=0.72)

    axes[1, 0].plot(azimuth, np.mean(truth, axis=0), label=scenario, color="#D55E00")
    axes[1, 0].plot(azimuth, uniform_truth, label="Uniform", color="#555555")
    axes[1, 0].axvspan(center_deg - 6, center_deg + 6, color="#D62728", alpha=0.12)
    axes[1, 0].set_title("Event-truth angular power")
    axes[1, 0].set_xlabel("Backazimuth (deg)")
    axes[1, 0].set_ylabel("Mean normalized power")
    axes[1, 0].legend(frameon=False)

    mean_beam = np.mean(beam, axis=0)
    axes[1, 1].plot(azimuth, mean_beam, label=scenario, color="#0072B2")
    axes[1, 1].plot(azimuth, uniform_beam, label="Uniform", color="#555555")
    axes[1, 1].plot(azimuth, mean_beam - uniform_beam, label="Excess", color="#009E73")
    axes[1, 1].axvspan(center_deg - 6, center_deg + 6, color="#D62728", alpha=0.12)
    axes[1, 1].set_title("Observed beam power")
    axes[1, 1].set_xlabel("Backazimuth (deg)")
    axes[1, 1].legend(frameon=False)

    sector_mask = circular_distance_deg(azimuth, center_deg) <= SECTOR_AUDIT_HALF_WIDTH_DEG
    sector_power = np.sum(beam[:, sector_mask], axis=1)
    axes[1, 2].scatter(sector_count, sector_power, s=8, alpha=0.45, color="#0072B2")
    axes[1, 2].set_title("Direction observable vs truth")
    axes[1, 2].set_xlabel("Sector events in window")
    axes[1, 2].set_ylabel("Beam sector power")

    for method, label, color in (("beam_uniformity", "beam", "#0072B2"), ("oracle_event_uniformity", "event oracle", "#D55E00")):
        selected = weights[weights.method == method]
        grouped = selected.groupby("sector_event_count").weight.mean()
        axes[1, 3].plot(grouped.index, grouped.values, marker="o", ms=3, label=label, color=color)
    axes[1, 3].set_title("Window-weight response")
    axes[1, 3].set_xlabel("Sector events in window")
    axes[1, 3].set_ylabel("Mean weight")
    axes[1, 3].legend(frameon=False)
    figure.suptitle(
        f"Gate 2-S v5.1 source-direction audit | {scenario} | seed {seed}\n"
        "Red outlines are registered sector sites; source truth is audit-only",
        fontsize=10,
    )
    _save(figure, output / f"{scenario}_seed{seed}_source_direction")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=Path("output_python/gate2_source_aware_development"))
    parser.add_argument("--seed", type=int, default=5501)
    parser.add_argument("--output-directory", type=Path, default=None)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    output = args.output_directory.resolve() if args.output_directory else root / "figures"
    _style()
    for scenario, center_deg in (("uniform", None), ("far75", 75.0), ("near12", 12.0), ("mirror168", 168.0)):
        case = root / f"seed_{args.seed}" / scenario
        _waveform_ftan(root, case, scenario, args.seed, output)
        if center_deg is not None:
            _direction_source_figure(root, case, scenario, center_deg, args.seed, output)
        print(f"completed source-aware QC for {scenario}, seed={args.seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
