#!/usr/bin/env python3
"""Plot auditable source, waveform, gather, dispersion, and seed QC.

The panels deliberately include failed deployable proxies and the synthetic-only
pairwise EGF Oracle.  The Oracle is an architectural reachability bound, not a
field-data method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.nnws_qc import (
    _dispersion_similarity,
    calibrated_waveform_comparison,
    reference_dispersion_for_strategy,
)


STRATEGIES = (
    "baseline_linear",
    "nnws_v2_shared_oof",
    "pair_crossfit_coherence",
    "pair_crossfit_tau_p_spz_frequency",
    "oracle_pair_branch_spz_frequency",
    "supervised_pair_egf_oracle_upper_bound",
)
LABELS = {
    "reference": "Theoretical EGF",
    "baseline_linear": "Linear",
    "nnws_v2_shared_oof": "Shared NNWS-v2",
    "pair_crossfit_coherence": "Data-only pair proxy",
    "pair_crossfit_tau_p_spz_frequency": "Cross-fit tau-p/SPZ",
    "oracle_pair_branch_spz_frequency": "Known-source pair SPZ",
    "supervised_pair_egf_oracle_upper_bound": "Pair EGF Oracle (upper bound)",
    "supervised_pair_egf_oracle_permuted_time_control": "Permuted Oracle control",
}
COLORS = {
    "reference": "#111111",
    "baseline_linear": "#777777",
    "nnws_v2_shared_oof": "#0072B2",
    "pair_crossfit_coherence": "#009E73",
    "pair_crossfit_tau_p_spz_frequency": "#56B4E9",
    "oracle_pair_branch_spz_frequency": "#D55E00",
    "supervised_pair_egf_oracle_upper_bound": "#CC79A7",
    "supervised_pair_egf_oracle_permuted_time_control": "#E69F00",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, options in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 300})):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", **options)
        outputs.append(path)
    plt.close(figure)
    return outputs


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


def _activity(path: Path, window_count: int, source_count: int) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True)
    # Match load_source_activity_matrix exactly: once a schedule is supplied,
    # omitted window/source combinations are inactive rather than unity gain.
    output = np.zeros((window_count, source_count), dtype=np.float64)
    for row in np.atleast_1d(table):
        output[int(row["window_index"]), int(row["source_index"])] = float(
            row["amplitude_multiplier"]
        )
    return output


def plot_sources_and_weights(run: Path, output: Path) -> list[Path]:
    _style()
    resolved = json.loads((run / "resolved_config.json").read_text())
    sources = np.loadtxt(run / "sources.csv", delimiter=",", skiprows=1)
    receivers = np.loadtxt(run / "receivers.csv", delimiter=",", skiprows=1)
    with h5py.File(run / "results.h5") as handle:
        pair_i = handle["supervised_pair_egf_oracle/pair_i"][:].astype(int)
        pair_j = handle["supervised_pair_egf_oracle/pair_j"][:].astype(int)
        oracle_positive = handle["supervised_pair_egf_oracle/positive"][:]
        oracle_negative = handle["supervised_pair_egf_oracle/negative"][:]
        spz_positive = handle["pair_spz/pair_positive_branch_band_weights"][:]
        spz_negative = handle["pair_spz/pair_negative_branch_band_weights"][:]
        proxy_positive = handle[
            "pair_crossfit_coherence/positive_weights"
        ][:]
        proxy_negative = handle[
            "pair_crossfit_coherence/negative_weights"
        ][:]
    window_count = oracle_positive.shape[0]
    activity = _activity(
        Path(resolved["ambient_sources"]["activity_file"]),
        window_count,
        sources.shape[0],
    )
    source_angle = np.mod(np.arctan2(sources[:, 1], sources[:, 0]), 2 * np.pi)
    source_order = np.argsort(source_angle)
    pair_distance = np.hypot(
        receivers[pair_j, 0] - receivers[pair_i, 0],
        receivers[pair_j, 1] - receivers[pair_i, 1],
    )
    pair_order = np.argsort(pair_distance)
    figure, axes = plt.subplots(
        2, 2, figsize=(11.2, 7.2), constrained_layout=True
    )
    geometry = axes[0, 0].scatter(
        sources[:, 0],
        sources[:, 1],
        c=np.mean(activity, axis=0),
        s=55,
        cmap="magma",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0, 0].plot(receivers[:, 0], receivers[:, 1], "k-", linewidth=1.0)
    axes[0, 0].scatter(
        receivers[[0, -1], 0],
        receivers[[0, -1], 1],
        marker="v",
        s=35,
        color="black",
    )
    axes[0, 0].set(
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title="Known anisotropic source field and 1-km receiver line",
    )
    figure.colorbar(geometry, ax=axes[0, 0], pad=0.02).set_label(
        "mean activity"
    )
    activity_image = axes[0, 1].imshow(
        activity[:, source_order].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    axes[0, 1].set(
        xlabel="window index",
        ylabel="source rank by azimuth",
        title="Time-varying source amplitudes (unlisted = inactive)",
    )
    figure.colorbar(activity_image, ax=axes[0, 1], pad=0.02).set_label(
        "activity multiplier"
    )
    weight_image = axes[1, 0].imshow(
        0.5
        * (oracle_positive[:, pair_order] + oracle_negative[:, pair_order]).T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[1, 0].set(
        xlabel="window index",
        ylabel="receiver-pair rank by distance",
        title="Supervised pair/branch Oracle weights (Neff ≥ 30)",
    )
    figure.colorbar(weight_image, ax=axes[1, 0], pad=0.02).set_label(
        "mean branch weight"
    )
    spz_mean = 0.5 * (
        np.mean(spz_positive, axis=1) + np.mean(spz_negative, axis=1)
    )
    oracle_mean = 0.5 * (oracle_positive + oracle_negative)
    proxy_mean = 0.5 * (proxy_positive + proxy_negative)
    axes[1, 1].scatter(
        oracle_mean.reshape(-1),
        spz_mean.reshape(-1),
        s=5,
        alpha=0.18,
        color=COLORS["oracle_pair_branch_spz_frequency"],
        label="Known-source SPZ",
    )
    axes[1, 1].scatter(
        oracle_mean.reshape(-1),
        proxy_mean.reshape(-1),
        s=5,
        alpha=0.18,
        color=COLORS["pair_crossfit_coherence"],
        label="Data-only proxy",
    )
    axes[1, 1].set(
        xlabel="supervised Oracle weight",
        ylabel="deployable/proxy weight",
        title="Available proxies weakly reproduce the reachable weights",
    )
    axes[1, 1].legend(frameon=False)
    for axis, label in zip(axes.flat, "abcd"):
        axis.grid(alpha=0.12)
        _panel(axis, label)
    figure.suptitle(
        "Source and weight audit | known-source and supervised methods are Oracle diagnostics",
        fontsize=10.5,
    )
    return _save(figure, output / "pairwise_source_weight_audit")


def _waveform_data(run: Path):
    resolved = json.loads((run / "resolved_config.json").read_text())
    processing = resolved["processing"]
    band = (
        float(processing["frequency_min_hz"]),
        float(processing["frequency_max_hz"]),
    )
    data = {}
    lag = distance = reference = None
    for strategy in STRATEGIES:
        lag, distance, candidate, reference = calibrated_waveform_comparison(
            run / "results.h5", strategy, "symmetric", frequency_range_hz=band
        )
        data[strategy] = candidate
    return lag, distance, reference, data, band, processing


def plot_waveforms(run: Path, output: Path) -> list[Path]:
    _style()
    lag, distance, reference, data, band, processing = _waveform_data(run)
    targets = np.asarray((200.0, 500.0, 1000.0))
    indices = np.abs(distance[:, None] - targets[None, :]).argmin(axis=0)
    vmin = float(processing["velocity_min_mps"])
    vmax = float(processing["velocity_max_mps"])
    figure, axes = plt.subplots(1, 3, figsize=(11.7, 3.8), sharey=True)
    for axis, distance_index, label in zip(axes, indices, "abc"):
        arrival_start = distance[distance_index] / vmax
        arrival_end = distance[distance_index] / vmin
        display = (lag >= 0.8 * distance[distance_index] / vmax) & (
            lag <= 1.05 * distance[distance_index] / vmin
        )
        axis.axvspan(
            arrival_start,
            arrival_end,
            color="#DDDDDD",
            alpha=0.5,
        )
        curves = (("reference", reference[:, distance_index]),) + tuple(
            (strategy, data[strategy][:, distance_index])
            for strategy in STRATEGIES
        )
        for name, curve in curves:
            scale = max(float(np.max(np.abs(curve[display]))), np.finfo(float).eps)
            axis.plot(
                lag,
                curve / scale,
                color=COLORS[name],
                linestyle="--" if name == "reference" else "-",
                linewidth=1.25 if name in {"reference", STRATEGIES[-1]} else 0.85,
                label=LABELS[name],
            )
        axis.set(
            xlim=(
                max(0.0, 0.8 * distance[distance_index] / vmax),
                min(float(lag[-1]), 1.05 * distance[distance_index] / vmin),
            ),
            ylim=(-1.15, 1.15),
            xlabel="lag time (s)",
            title=f"distance = {distance[distance_index]:.0f} m",
        )
        axis.grid(alpha=0.12)
        _panel(axis, label)
    axes[0].set_ylabel("per-trace normalized amplitude")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(
        f"Arrival-window NCF versus theoretical EGF | {band[0]:g}–{band[1]:g} Hz | gray = expected arrival",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0.0, 0.16, 1.0, 0.94))
    return _save(figure, output / "pairwise_waveform_overlays")


def _column_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(
        np.max(np.abs(values), axis=0, keepdims=True), np.finfo(float).eps
    )


def plot_gathers(run: Path, output: Path) -> list[Path]:
    _style()
    lag, distance, reference, data, _, processing = _waveform_data(run)
    panels = (("reference", reference),) + tuple(
        (strategy, data[strategy]) for strategy in STRATEGIES
    )
    vmin = float(processing["velocity_min_mps"])
    vmax = float(processing["velocity_max_mps"])
    figure, axes = plt.subplots(
        2, 4, figsize=(14.2, 6.4), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for axis, (name, values), label in zip(axes.flat, panels, "abcdefg"):
        image = axis.imshow(
            _column_normalize(values),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(distance[0], distance[-1], lag[0], lag[-1]),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axis.plot(distance, distance / vmax, "k:", linewidth=0.7)
        axis.plot(distance, distance / vmin, "k:", linewidth=0.7)
        axis.set(xlabel="distance (m)", ylabel="lag time (s)", title=LABELS[name])
        _panel(axis, label)
    axes.flat[-1].axis("off")
    assert image is not None
    figure.colorbar(image, ax=axes, pad=0.012, shrink=0.86).set_label(
        "per-distance normalized amplitude"
    )
    figure.suptitle(
        "Symmetric NCF gathers | dotted lines bound expected surface-wave arrivals",
        fontsize=10.5,
    )
    return _save(figure, output / "pairwise_ncf_gathers")


def plot_dispersion(run: Path, output: Path) -> list[Path]:
    _style()
    reference, reference_quality = reference_dispersion_for_strategy(
        run, "baseline_linear"
    )
    with h5py.File(run / "results.h5") as handle:
        frequency = handle["dispersion/strategies/baseline_linear/frequency_hz"][:]
        velocity = handle["dispersion/strategies/baseline_linear/velocity_mps"][:]
        theory_frequency = handle["theory/frequency_hz"][:]
        theory_velocity = handle["theory/rayleigh_mps"][:]
        energies = {
            strategy: handle[f"dispersion/strategies/{strategy}/energy"][:]
            for strategy in STRATEGIES
        }
    panels = (("reference", reference.energy, None),) + tuple(
        (
            strategy,
            energies[strategy],
            _dispersion_similarity(energies[strategy], reference.energy, velocity),
        )
        for strategy in STRATEGIES
    )
    figure, axes = plt.subplots(
        2, 4, figsize=(14.2, 6.5), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for axis, (name, energy, similarity), label in zip(
        axes.flat, panels, "abcdefg"
    ):
        image = axis.imshow(
            energy,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(frequency[0], frequency[-1], velocity[0], velocity[-1]),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        for mode_index in range(theory_velocity.shape[1]):
            axis.plot(
                theory_frequency,
                theory_velocity[:, mode_index],
                "w--",
                linewidth=0.7,
            )
        title = LABELS[name]
        if similarity is not None:
            title += (
                f"\nimage CC={similarity['image_correlation']:.3f}; "
                f"ridge MAE={similarity['dominant_ridge_mae_to_egf_mps']:.2f} m/s"
            )
        else:
            title += (
                f"\nclean reference={bool(reference_quality['clean_ridge_pass'])}"
            )
        axis.set(xlabel="frequency (Hz)", ylabel="phase velocity (m/s)", title=title)
        _panel(axis, label)
    axes.flat[-1].axis("off")
    assert image is not None
    figure.colorbar(image, ax=axes, pad=0.012, shrink=0.86).set_label(
        "column-normalized energy"
    )
    figure.suptitle(
        "Dispersion energy QC | dashed curves = theoretical Rayleigh modes",
        fontsize=10.5,
    )
    return _save(figure, output / "pairwise_dispersion")


def _seed_records(seed_root: Path):
    strategies = (
        "baseline_linear",
        "nnws_v2_shared_oof",
        "supervised_pair_egf_oracle_permuted_time_control",
        "supervised_pair_egf_oracle_upper_bound",
    )
    records = []
    for run in sorted(path for path in seed_root.glob("seed_*") if path.is_dir()):
        manifest = json.loads((run / "manifest.json").read_text())
        reference, _ = reference_dispersion_for_strategy(run, "baseline_linear")
        with h5py.File(run / "results.h5") as handle:
            velocity = handle[
                "dispersion/strategies/baseline_linear/velocity_mps"
            ][:]
            for strategy in strategies:
                egf = manifest["strategy_egf_quality"][strategy]["zz_symmetric"]
                dispersion = manifest["strategy_dispersion_quality"][strategy]
                similarity = _dispersion_similarity(
                    handle[f"dispersion/strategies/{strategy}/energy"][:],
                    reference.energy,
                    velocity,
                )
                records.append(
                    {
                        "seed": int(run.name.split("_")[-1]),
                        "strategy": strategy,
                        "cc": egf["waveform_correlation_no_shift"],
                        "phase": egf["phase_rms_rad"],
                        "pre": egf["prearrival_ghost_energy_fraction"],
                        "spec": egf[
                            "spectral_shape_log_rms_after_global_scale"
                        ],
                        "pick": dispersion["picks"]["mae_mps"],
                        "image_cc": similarity["image_correlation"],
                        "ridge": similarity["dominant_ridge_mae_to_egf_mps"],
                    }
                )
    return strategies, records


def plot_multiseed(seed_root: Path, output: Path) -> list[Path]:
    _style()
    strategies, records = _seed_records(seed_root)
    metrics = (
        ("cc", "waveform correlation", True),
        ("phase", "phase RMS (rad)", False),
        ("pre", "pre-arrival fraction", False),
        ("spec", "spectral-shape RMS", False),
        ("image_cc", "dispersion image CC", True),
        ("ridge", "ridge MAE to EGF (m/s)", False),
        ("pick", "pick MAE to theory (m/s)", False),
    )
    seeds = sorted({record["seed"] for record in records})
    figure, axes = plt.subplots(2, 4, figsize=(12.4, 6.3), constrained_layout=True)
    for axis, metric_spec, label in zip(axes.flat, metrics, "abcdefg"):
        metric, title, higher = metric_spec
        for seed in seeds:
            values = [
                next(
                    record[metric]
                    for record in records
                    if record["seed"] == seed and record["strategy"] == strategy
                )
                for strategy in strategies
            ]
            axis.plot(range(len(strategies)), values, color="#BBBBBB", linewidth=0.65)
        medians = [
            np.median(
                [
                    record[metric]
                    for record in records
                    if record["strategy"] == strategy
                ]
            )
            for strategy in strategies
        ]
        axis.scatter(
            range(len(strategies)),
            medians,
            s=32,
            color=[COLORS[strategy] for strategy in strategies],
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        direction = "higher is better" if higher else "lower is better"
        axis.set(title=f"{title}\n{direction}", xticks=range(len(strategies)))
        axis.set_xticklabels(
            [LABELS[strategy] for strategy in strategies],
            rotation=25,
            ha="right",
            rotation_mode="anchor",
        )
        axis.grid(alpha=0.12)
        _panel(axis, label)
    axes.flat[-1].axis("off")
    figure.suptitle(
        f"Five-seed architecture test | thin lines = paired seeds; dots = medians | n={len(seeds)}",
        fontsize=10.5,
    )
    return _save(figure, output / "pairwise_multiseed")


def write_summary(run: Path, seed_root: Path | None, path: Path) -> Path:
    manifest = json.loads((run / "manifest.json").read_text())
    reference, reference_quality = reference_dispersion_for_strategy(
        run, "baseline_linear"
    )
    with h5py.File(run / "results.h5") as handle:
        velocity = handle["dispersion/strategies/baseline_linear/velocity_mps"][:]
        single = {}
        for strategy in STRATEGIES:
            egf = manifest["strategy_egf_quality"][strategy]["zz_symmetric"]
            dispersion = manifest["strategy_dispersion_quality"][strategy]
            similarity = _dispersion_similarity(
                handle[f"dispersion/strategies/{strategy}/energy"][:],
                reference.energy,
                velocity,
            )
            single[strategy] = {
                "egf": egf,
                "dispersion": dispersion,
                "dispersion_similarity_to_egf": similarity,
            }
    output = {
        "reference_dispersion": reference_quality,
        "single_seed": single,
        "interpretation": {
            "deployable_pair_proxy_passes_complete_egf_gate": False,
            "deployable_pair_tau_p_passes_complete_egf_gate": False,
            "known_source_spz_passes_complete_egf_gate": False,
            "supervised_pair_oracle_is_deployable": False,
            "architecture_supported_by_oracle": "pair-specific, branch-specific weights",
            "field_data_next_step": "replace target leakage with pairwise beamforming/3C source-azimuth features and validate against an independent reference",
        },
    }
    if seed_root is not None:
        strategies, records = _seed_records(seed_root)
        candidate = "supervised_pair_egf_oracle_upper_bound"
        baseline = "baseline_linear"
        directions = {
            "cc": "higher",
            "phase": "lower",
            "pre": "lower",
            "spec": "lower",
            "pick": "lower",
            "image_cc": "higher",
            "ridge": "lower",
        }
        wins = {}
        seeds = sorted({record["seed"] for record in records})
        for metric, direction in directions.items():
            count = 0
            for seed in seeds:
                left = next(
                    record[metric]
                    for record in records
                    if record["seed"] == seed and record["strategy"] == candidate
                )
                right = next(
                    record[metric]
                    for record in records
                    if record["seed"] == seed and record["strategy"] == baseline
                )
                count += int(left > right if direction == "higher" else left < right)
            wins[metric] = count
        output["multiseed_pair_oracle_wins_vs_linear"] = {
            "seed_count": len(seeds),
            **wins,
        }
        output["multiseed_strategies"] = list(strategies)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, default=None)
    args = parser.parse_args()
    files = []
    files.extend(plot_sources_and_weights(args.run_directory, args.output_directory))
    files.extend(plot_waveforms(args.run_directory, args.output_directory))
    files.extend(plot_gathers(args.run_directory, args.output_directory))
    files.extend(plot_dispersion(args.run_directory, args.output_directory))
    if args.seed_root is not None:
        files.extend(plot_multiseed(args.seed_root, args.output_directory))
    summary = write_summary(
        args.run_directory,
        args.seed_root,
        args.output_directory / "pairwise_egf_qc_summary.json",
    )
    manifest = args.output_directory / "pairwise_egf_qc_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_directory": str(args.run_directory.resolve()),
                "seed_root": (
                    str(args.seed_root.resolve()) if args.seed_root is not None else None
                ),
                "figures": [str(path.resolve()) for path in files],
                "summary": str(summary.resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(files)} figures, {summary}, and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
