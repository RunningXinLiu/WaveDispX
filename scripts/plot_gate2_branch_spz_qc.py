#!/usr/bin/env python3
"""Plot v5.2 branch waveforms, FTAN, and truth-only SPZ weighting audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wavedispx.processing import pair_ftan

from plot_gate2_sparse_event_qc import _bandpass_trace, _theory
from run_gate2_branch_spz_oracle_experiment import (
    NEGATIVE_SOURCE_CENTER_DEG,
    POSITIVE_SOURCE_CENTER_DEG,
    SPZ_HALF_WIDTH_DEG,
)
from run_gate2_pair_selection_experiment import (
    EVALUATION_BAND_HZ,
    PAIR_DISTANCE_M,
    VELOCITY_RANGE_MPS,
)


METHODS = (
    ("active_reference", "Active EGF"),
    ("paired_clean_counterfactual", "Paired clean"),
    ("linear", "Linear"),
    ("oracle_branch_spz", "Branch-SPZ oracle"),
)


def _style() -> None:
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
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def _one_ftan(trace: np.ndarray, dt_s: float, *, input_kind: str):
    return pair_ftan(
        trace,
        dt_s=dt_s,
        distance_m=PAIR_DISTANCE_M,
        frequency_range_hz=EVALUATION_BAND_HZ,
        group_velocity_range_mps=VELOCITY_RANGE_MPS,
        phase_velocity_range_mps=VELOCITY_RANGE_MPS,
        period_count=121,
        velocity_step_mps=2.0,
        gaussian_alpha=120.0,
        phase_method="frequency_domain",
        minimum_snr=0.0,
        input_kind=input_kind,
    )


def _waveform_ftan(
    run_root: Path,
    input_root: Path,
    scenario: str,
    seed: int,
    output: Path,
) -> None:
    case_dir = run_root / f"seed_{seed}" / scenario
    with np.load(case_dir / "gate2_branch_spz_oracle.npz") as npz:
        lag = np.asarray(npz["lag_time_s"], dtype=np.float64)
        dt = float(np.median(np.diff(lag)))
        active_time = np.asarray(npz["active_reference_time_s"], dtype=np.float64)
        active_dt = float(np.median(np.diff(active_time)))
        active = _bandpass_trace(npz["active_reference_zz"], active_dt)
        waveforms: dict[str, dict[str, np.ndarray] | None] = {
            "active_reference": {"positive": active, "negative": active}
        }
        ftan = {
            "active_reference": _one_ftan(
                npz["active_reference_zz"], active_dt, input_kind="egf"
            )
        }
        for method, _ in METHODS[1:]:
            symmetric_key = f"stack__{method}__symmetric"
            if symmetric_key not in npz.files:
                waveforms[method] = None
                ftan[method] = None
                continue
            waveforms[method] = {
                branch: _bandpass_trace(
                    -np.gradient(npz[f"stack__{method}__{branch}"], dt),
                    dt,
                )
                for branch in ("positive", "negative")
            }
            ftan[method] = _one_ftan(
                npz[symmetric_key], dt, input_kind="correlation"
            )
    metrics = pd.read_csv(run_root / "gate2_branch_spz_metrics.csv")
    metrics = metrics[
        (metrics.wavefield_seed == seed)
        & (metrics.scenario == scenario)
        & (metrics.control_replica == -1)
    ]
    theory_case = input_root / f"seed_{seed}" / scenario
    theory_period, theory_group, theory_phase = _theory(theory_case, seed)
    figure, axes = plt.subplots(
        4,
        len(METHODS),
        figsize=(10.6, 8.0),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )
    active_normalized = active / max(np.max(np.abs(active)), np.finfo(float).eps)
    image = None
    for column, (method, label) in enumerate(METHODS):
        method_waveforms = waveforms[method]
        result = ftan[method]
        if method_waveforms is None or result is None:
            message = "not applicable" if scenario == "uniform" else "missing"
            for row in range(4):
                axes[row, column].text(
                    0.5,
                    0.5,
                    message,
                    ha="center",
                    va="center",
                    transform=axes[row, column].transAxes,
                )
            axes[0, column].set_title(label)
            continue
        title = label
        if method != "active_reference":
            selected = metrics[
                (metrics.method == method) & (metrics.branch == "symmetric")
            ]
            if not selected.empty:
                title += f"\nsymmetric CC={selected.iloc[0].waveform_correlation_no_shift:.3f}"
        axes[0, column].set_title(title)
        for row, branch in enumerate(("positive", "negative")):
            trace = method_waveforms[branch]
            normalized = trace / max(np.max(np.abs(trace)), np.finfo(float).eps)
            time = active_time if method == "active_reference" else lag
            axes[row, column].plot(time, normalized, color="#0072B2", linewidth=0.8)
            axes[row, column].plot(
                active_time,
                active_normalized,
                "--",
                color="#333333",
                linewidth=0.55,
                alpha=0.7,
            )
            axes[row, column].axvspan(
                PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[1],
                PAIR_DISTANCE_M / VELOCITY_RANGE_MPS[0],
                color="#F0E442",
                alpha=0.12,
            )
            axes[row, column].set_xlim(0.0, 5.25)
            axes[row, column].set_ylim(-1.05, 1.05)
        image = axes[2, column].pcolormesh(
            result.group_image.period_s,
            result.group_image.velocity_mps,
            result.group_image.amplitude,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            rasterized=True,
        )
        axes[2, column].plot(theory_period, theory_group, color="cyan", linewidth=1.0)
        axes[3, column].pcolormesh(
            result.phase_image.period_s,
            result.phase_image.velocity_mps,
            result.phase_image.amplitude,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            rasterized=True,
        )
        axes[3, column].plot(theory_period, theory_phase, color="cyan", linewidth=1.0)
        axes[3, column].set_xlabel("Period (s)")
    axes[0, 0].set_ylabel("Positive branch\nnormalized waveform")
    axes[1, 0].set_ylabel("Negative branch\nnormalized waveform")
    axes[2, 0].set_ylabel("Group velocity (m/s)")
    axes[3, 0].set_ylabel("Phase velocity (m/s)")
    axes[2, 0].set_ylim(VELOCITY_RANGE_MPS)
    axes[3, 0].set_ylim(VELOCITY_RANGE_MPS)
    if image is not None:
        figure.colorbar(
            image,
            ax=axes[2:, :],
            label="Per-period normalized energy",
            shrink=0.75,
        )
    figure.suptitle(
        f"Gate 2-S v5.2 truth-only branch-SPZ QC | {scenario} | seed {seed}\n"
        "Waveforms: 4–12 Hz and −d/dt NCF; cyan: QEDisp fundamental Rayleigh; no automatic picks",
        fontsize=10,
    )
    _save(figure, output / f"{scenario}_seed{seed}_branch_waveform_ftan")


def _site_contribution(
    events: pd.DataFrame,
    weights: pd.DataFrame | None,
) -> pd.DataFrame:
    merged = events.copy()
    if weights is None:
        merged["weight"] = 1.0
    else:
        merged = merged.merge(
            weights[["window_index", "weight"]],
            on="window_index",
            how="left",
        )
    merged["contribution"] = np.square(merged.event_weight) * merged.weight
    return (
        merged.groupby(["source_index", "source_group", "x_m", "y_m"])
        .contribution.sum()
        .reset_index()
    )


def _source_audit(
    run_root: Path,
    input_root: Path,
    scenario: str,
    seed: int,
    output: Path,
) -> None:
    case_dir = run_root / f"seed_{seed}" / scenario
    input_case = input_root / f"seed_{seed}" / scenario
    events = pd.read_csv(input_case / "source_events.csv")
    weights = pd.read_csv(run_root / "gate2_branch_spz_window_weights.csv")
    weights = weights[
        (weights.wavefield_seed == seed) & (weights.scenario == scenario)
    ]
    with np.load(case_dir / "gate2_branch_spz_oracle.npz") as npz:
        azimuth = np.asarray(npz["beam_azimuth_deg"], dtype=np.float64)
        truth = np.asarray(npz["truth_angular_power"], dtype=np.float64)
        targets = {
            "positive": np.asarray(npz["target_positive"], dtype=np.float64),
            "negative": np.asarray(npz["target_negative"], dtype=np.float64),
        }
    receivers = np.genfromtxt(input_case / "receivers.csv", delimiter=",", names=True)
    figure, axes = plt.subplots(2, 3, figsize=(10.6, 6.4), constrained_layout=True)
    map_specs = (
        ("Raw event energy", None),
        ("Positive-branch weighted", weights[weights.branch == "positive"]),
        ("Negative-branch weighted", weights[weights.branch == "negative"]),
    )
    scatter = None
    for axis, (title, selected_weights) in zip(axes[0], map_specs):
        site = _site_contribution(events, selected_weights)
        normalized = site.contribution / max(
            float(site.contribution.max()), np.finfo(float).eps
        )
        scatter = axis.scatter(
            site.x_m,
            site.y_m,
            c=normalized,
            s=4 + 22 * normalized,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.8,
            linewidths=0.0,
            rasterized=True,
        )
        sector = site[site.source_group == 1]
        axis.scatter(
            sector.x_m,
            sector.y_m,
            facecolors="none",
            edgecolors="#D62728",
            s=32,
            linewidths=0.4,
        )
        axis.scatter(
            receivers["x_m"][:2],
            receivers["y_m"][:2],
            marker="v",
            s=28,
            c="#D55E00",
            label="Target pair",
        )
        axis.scatter(
            receivers["x_m"][2:],
            receivers["y_m"][2:],
            marker="+",
            s=10,
            c="#111111",
            linewidths=0.45,
            label="Beam array",
        )
        axis.set_aspect("equal")
        axis.set_xlim(-1900, 1900)
        axis.set_ylim(-1900, 1900)
        axis.set_title(title)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
    axes[0, 0].legend(frameon=False, loc="lower left")
    if scatter is not None:
        figure.colorbar(
            scatter,
            ax=axes[0, :],
            label="Normalized source-site contribution",
            shrink=0.75,
        )

    raw = np.mean(truth, axis=0)
    axes[1, 0].plot(azimuth, raw, color="#555555", label="Raw")
    for branch, color in (("positive", "#0072B2"), ("negative", "#D55E00")):
        selected = weights[weights.branch == branch].sort_values("window_index")
        effective = truth.T @ selected.weight.to_numpy()
        effective /= np.sum(effective)
        axes[1, 0].plot(azimuth, effective, color=color, label=branch.capitalize())
    axes[1, 0].set_title("Effective event-truth angular power")
    axes[1, 0].set_xlabel("Backazimuth (deg)")
    axes[1, 0].set_ylabel("Normalized power")
    axes[1, 0].legend(frameon=False)

    for branch, color in (("positive", "#0072B2"), ("negative", "#D55E00")):
        axes[1, 1].plot(azimuth, targets[branch], color=color, label=f"{branch.capitalize()} target")
    axes[1, 1].axvspan(
        POSITIVE_SOURCE_CENTER_DEG - SPZ_HALF_WIDTH_DEG,
        POSITIVE_SOURCE_CENTER_DEG + SPZ_HALF_WIDTH_DEG,
        color="#0072B2",
        alpha=0.08,
    )
    axes[1, 1].axvspan(
        NEGATIVE_SOURCE_CENTER_DEG - SPZ_HALF_WIDTH_DEG,
        NEGATIVE_SOURCE_CENTER_DEG + SPZ_HALF_WIDTH_DEG,
        color="#D55E00",
        alpha=0.08,
    )
    axes[1, 1].set_title("Pre-registered branch targets")
    axes[1, 1].set_xlabel("Backazimuth (deg)")
    axes[1, 1].set_ylabel("Target mass")
    axes[1, 1].legend(frameon=False)

    positive = weights[weights.branch == "positive"].sort_values("window_index")
    negative = weights[weights.branch == "negative"].sort_values("window_index")
    axes[1, 2].scatter(
        positive.weight,
        negative.weight,
        s=8,
        alpha=0.45,
        color="#009E73",
    )
    axes[1, 2].axline((1.0, 1.0), slope=1.0, color="#777777", linewidth=0.7)
    axes[1, 2].set_title("Independent branch weights")
    axes[1, 2].set_xlabel("Positive-branch weight")
    axes[1, 2].set_ylabel("Negative-branch weight")
    figure.suptitle(
        f"Gate 2-S v5.2 truth-only branch-SPZ source audit | {scenario} | seed {seed}\n"
        "Red outlines are registered biased-sector sites; source truth is oracle-only",
        fontsize=10,
    )
    _save(figure, output / f"{scenario}_seed{seed}_branch_source_audit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("output_python/gate2_branch_spz_development"),
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("output_python/gate2_source_aware_development"),
    )
    parser.add_argument("--seed", type=int, default=5501)
    args = parser.parse_args()
    run_root = args.run_directory.resolve()
    input_root = args.input_directory.resolve()
    output = run_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _style()
    for scenario in ("uniform", "far75", "near12", "mirror168"):
        _waveform_ftan(run_root, input_root, scenario, args.seed, output)
        _source_audit(run_root, input_root, scenario, args.seed, output)
        print(f"completed branch-SPZ QC for {scenario}, seed={args.seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
