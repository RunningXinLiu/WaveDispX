#!/usr/bin/env python3
"""Create auditable NNWS-v2 source, waveform, gather, and dispersion QC.

Figure contract
---------------
Core conclusion: NNWS-v2 improves several EGF waveform attributes in the
known-truth synthetic test, but its dispersion and field-data gains are not
uniform enough to claim general reliability.
Archetype: quantitative grid.  Backend: Python/matplotlib.  Exports: editable
SVG/PDF plus 300-dpi PNG.  All methods use one fixed color across every panel.
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
    configure_figure_style,
    reference_dispersion_for_strategy,
)


COLORS = {
    "reference": "#111111",
    "baseline_linear": "#6B6B6B",
    "nnws_v2_random_neff_control": "#56B4E9",
    "nnws_v2_shared_oof": "#009E73",
}
LABELS = {
    "reference": "Theoretical EGF",
    "baseline_linear": "Linear",
    "nnws_v2_random_neff_control": "Permuted same-Neff",
    "nnws_v2_shared_oof": "NNWS-v2 OOF",
}
STRATEGIES = (
    "baseline_linear",
    "nnws_v2_random_neff_control",
    "nnws_v2_shared_oof",
)


def _style() -> None:
    configure_figure_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
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


def _read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(table["x_m"]), np.atleast_1d(table["y_m"])


def _activity_matrix(path: Path, window_count: int, source_count: int) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True)
    matrix = np.zeros((window_count, source_count), dtype=np.float64)
    matrix[
        np.atleast_1d(table["window_index"]).astype(int),
        np.atleast_1d(table["source_index"]).astype(int),
    ] = np.atleast_1d(table["amplitude_multiplier"])
    return matrix


def plot_source_weights(run_directory: Path, output_directory: Path) -> list[Path]:
    _style()
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    source_x, source_y = _read_xy(run_directory / "sources.csv")
    receiver_x, receiver_y = _read_xy(run_directory / "receivers.csv")
    with h5py.File(run_directory / "results.h5", "r") as handle:
        weights = handle["nnws_v2/weights"][:]
        random_weights = handle["nnws_v2/random_control_weights"][:]
        folds = handle["nnws_v2/fold_index"][:]
        selection = handle["source_selection"]
        endfire = np.mean(selection["endfire_fraction"][:], axis=1)
        balance = np.mean(selection["bidirectional_balance"][:], axis=1)
        oracle_soft = selection["soft_weight"][:]
    activity_path = Path(resolved["ambient_sources"]["activity_file"])
    activity = _activity_matrix(activity_path, weights.size, source_x.size)
    source_angle = np.mod(np.arctan2(source_y, source_x), 2 * np.pi)
    source_order = np.argsort(source_angle)
    mean_activity = np.mean(activity, axis=0)
    weight_activity_correlation = float(
        np.corrcoef(weights, np.sum(activity, axis=1))[0, 1]
    )
    oracle_correlation = float(np.corrcoef(weights, oracle_soft)[0, 1])

    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), constrained_layout=True)
    scatter = axes[0, 0].scatter(
        source_x,
        source_y,
        c=mean_activity,
        s=48,
        cmap="magma",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0, 0].plot(receiver_x, receiver_y, color="#333333", linewidth=1.0)
    axes[0, 0].scatter(
        receiver_x[[0, -1]],
        receiver_y[[0, -1]],
        marker="v",
        s=42,
        color="#0072B2",
        label="Receiver aperture",
    )
    axes[0, 0].set(
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title="Known synthetic source field",
    )
    axes[0, 0].legend(frameon=False)
    cbar = figure.colorbar(scatter, ax=axes[0, 0], pad=0.02)
    cbar.set_label("mean activity multiplier")

    image = axes[0, 1].imshow(
        activity[:, source_order].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        extent=(0, weights.size, 0, source_x.size),
    )
    axes[0, 1].set(
        xlabel="20-s window index",
        ylabel="source rank by azimuth",
        title="Time-varying source amplitudes",
    )
    cbar = figure.colorbar(image, ax=axes[0, 1], pad=0.02)
    cbar.set_label("activity multiplier")

    window = np.arange(weights.size)
    for fold in np.unique(folds):
        selected = folds == fold
        axes[1, 0].axvspan(
            window[selected][0],
            window[selected][-1] + 1,
            color=plt.cm.Greys(0.12 + 0.08 * (int(fold) % 2)),
            zorder=0,
        )
    axes[1, 0].plot(
        window,
        weights,
        color=COLORS["nnws_v2_shared_oof"],
        label="OOF shared weight",
    )
    axes[1, 0].plot(
        window,
        random_weights,
        color=COLORS["nnws_v2_random_neff_control"],
        alpha=0.75,
        label="Permuted same-Neff",
    )
    axes[1, 0].set(
        xlabel="20-s window index",
        ylabel="weight",
        title=f"Contiguous folds; Neff={weights.sum() ** 2 / np.square(weights).sum():.1f}/{weights.size}",
    )
    axes[1, 0].legend(frameon=False, ncol=2)

    axes[1, 1].plot(window, endfire, color="#E69F00", label="Endfire fraction")
    axes[1, 1].plot(window, balance, color="#CC79A7", label="Bidirectional balance")
    axes[1, 1].plot(
        window,
        oracle_soft / max(np.max(oracle_soft), np.finfo(float).eps),
        color="#0072B2",
        linestyle="--",
        label="Oracle score",
    )
    axes[1, 1].plot(
        window,
        weights / max(np.max(weights), np.finfo(float).eps),
        color=COLORS["nnws_v2_shared_oof"],
        label="V2 weight",
    )
    axes[1, 1].set(
        xlabel="20-s window index",
        ylabel="unit-scale diagnostic",
        ylim=(-0.03, 1.06),
        title=(
            f"Independent source diagnostics; corr(weight, oracle)={oracle_correlation:.2f}"
        ),
    )
    axes[1, 1].legend(frameon=False, ncol=2)
    axes[1, 1].text(
        0.01,
        0.03,
        f"corr(weight, total activity)={weight_activity_correlation:.2f}",
        transform=axes[1, 1].transAxes,
        fontsize=6.5,
    )
    for axis, label in zip(axes.flat, "abcd"):
        _panel(axis, label)
        axis.grid(alpha=0.14)
    figure.suptitle(
        "NNWS-v2 source-field and shared-weight audit | synthetic known truth",
        fontsize=10.5,
    )
    return _save(figure, output_directory / "nnws_v2_sources_weights")


def plot_waveforms(run_directory: Path, output_directory: Path) -> list[Path]:
    _style()
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    band = (
        float(resolved["processing"]["frequency_min_hz"]),
        float(resolved["processing"]["frequency_max_hz"]),
    )
    vmin = float(resolved["processing"]["velocity_min_mps"])
    vmax = float(resolved["processing"]["velocity_max_mps"])
    data = {}
    lag = distance = reference = None
    for strategy in STRATEGIES:
        lag, distance, candidate, reference = calibrated_waveform_comparison(
            run_directory / "results.h5",
            strategy,
            "symmetric",
            frequency_range_hz=band,
        )
        data[strategy] = candidate
    assert lag is not None and distance is not None and reference is not None
    targets = np.asarray((150.0, 350.0, float(np.max(distance))))
    indices = np.unique(np.abs(distance[:, None] - targets[None, :]).argmin(axis=0))
    figure, axes = plt.subplots(
        1, indices.size, figsize=(11.5, 3.5), constrained_layout=False
    )
    figure.subplots_adjust(
        left=0.06, right=0.99, bottom=0.16, top=0.74, wspace=0.16
    )
    axes = np.atleast_1d(axes)
    for axis, distance_index, label in zip(axes, indices, "abc"):
        signal = (lag >= distance[distance_index] / vmax) & (
            lag <= distance[distance_index] / vmin
        )
        axis.axvspan(
            distance[distance_index] / vmax,
            distance[distance_index] / vmin,
            color="#DDDDDD",
            alpha=0.55,
        )
        for name, curve in (
            ("reference", reference[:, distance_index]),
            *((strategy, data[strategy][:, distance_index]) for strategy in STRATEGIES),
        ):
            scale = np.max(np.abs(curve[signal]), initial=0.0)
            axis.plot(
                lag,
                curve / max(scale, np.finfo(float).eps),
                color=COLORS[name],
                linestyle="--" if name == "reference" else "-",
                linewidth=1.35 if name in {"reference", "nnws_v2_shared_oof"} else 1.0,
                label=LABELS[name],
            )
        axis.set(
            xlim=(0, min(float(lag[-1]), 1.2 * np.max(distance) / vmin)),
            ylim=(-1.18, 1.18),
            xlabel="lag time (s)",
            title=f"symmetric branch | {distance[distance_index]:g} m",
        )
        axis.grid(alpha=0.14)
        _panel(axis, label)
    axes[0].set_ylabel("per-curve normalized amplitude")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"Stacked NCF versus theoretical EGF | {band[0]:g}–{band[1]:g} Hz | gray = expected arrival",
        y=0.98,
        fontsize=10.5,
    )
    return _save(figure, output_directory / "nnws_v2_waveform_overlays")


def _column_normalize(values: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(values), axis=0, keepdims=True)
    return values / np.maximum(scale, np.finfo(float).eps)


def plot_gathers(run_directory: Path, output_directory: Path) -> list[Path]:
    _style()
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    band = (
        float(resolved["processing"]["frequency_min_hz"]),
        float(resolved["processing"]["frequency_max_hz"]),
    )
    vmin = float(resolved["processing"]["velocity_min_mps"])
    vmax = float(resolved["processing"]["velocity_max_mps"])
    data = {}
    lag = distance = reference = None
    for strategy in STRATEGIES:
        lag, distance, candidate, reference = calibrated_waveform_comparison(
            run_directory / "results.h5",
            strategy,
            "symmetric",
            frequency_range_hz=band,
        )
        data[strategy] = _column_normalize(candidate)
    assert lag is not None and distance is not None and reference is not None
    data["reference"] = _column_normalize(reference)
    panels = (
        ("reference", data["reference"]),
        *((strategy, data[strategy]) for strategy in STRATEGIES),
        ("residual", data["nnws_v2_shared_oof"] - data["reference"]),
    )
    figure, axes = plt.subplots(1, 5, figsize=(12.8, 3.7), sharey=True, constrained_layout=True)
    for axis, (name, values), label in zip(axes, panels, "abcde"):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(distance[0], distance[-1], lag[0], lag[-1]),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axis.plot(distance, distance / vmax, color="#222222", linestyle=":", linewidth=0.8)
        axis.plot(distance, distance / vmin, color="#222222", linestyle=":", linewidth=0.8)
        title = "NNWS-v2 − EGF" if name == "residual" else LABELS[name]
        axis.set(xlabel="distance (m)", title=title)
        _panel(axis, label)
    axes[0].set_ylabel("lag time (s)")
    cbar = figure.colorbar(image, ax=axes, pad=0.012, shrink=0.9)
    cbar.set_label("per-distance normalized amplitude")
    figure.suptitle(
        "Symmetric NCF gathers | dotted lines bound expected surface-wave arrivals",
        fontsize=10.5,
    )
    return _save(figure, output_directory / "nnws_v2_ncf_gathers")


def plot_dispersion(run_directory: Path, output_directory: Path) -> list[Path]:
    _style()
    reference, reference_quality = reference_dispersion_for_strategy(
        run_directory, "baseline_linear"
    )
    with h5py.File(run_directory / "results.h5", "r") as handle:
        frequency = handle["dispersion/strategies/baseline_linear/frequency_hz"][:]
        velocity = handle["dispersion/strategies/baseline_linear/velocity_mps"][:]
        theory_frequency = handle["theory/frequency_hz"][:]
        theory_velocity = handle["theory/rayleigh_mps"][:]
        energies = {
            strategy: handle[f"dispersion/strategies/{strategy}/energy"][:]
            for strategy in STRATEGIES
        }
        qualities = {
            strategy: dict(
                handle[f"quality/strategies/{strategy}/dispersion/image"].attrs
            )
            for strategy in STRATEGIES
        }
    panels = (
        ("reference", reference.energy, reference_quality, None),
        *(
            (
                strategy,
                energies[strategy],
                qualities[strategy],
                _dispersion_similarity(energies[strategy], reference.energy, velocity),
            )
            for strategy in STRATEGIES
        ),
    )
    figure, axes = plt.subplots(1, 4, figsize=(12.8, 3.8), sharex=True, sharey=True, constrained_layout=True)
    for axis, (name, energy, quality, similarity), label in zip(axes, panels, "abcd"):
        image = axis.imshow(
            energy,
            origin="lower",
            aspect="auto",
            extent=(frequency[0], frequency[-1], velocity[0], velocity[-1]),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        for mode_index in range(theory_velocity.shape[1]):
            axis.plot(
                theory_frequency,
                theory_velocity[:, mode_index],
                color="white",
                linestyle="--",
                linewidth=0.8,
            )
        title = f"{LABELS[name]}\nambiguous={float(quality['ambiguous_frequency_fraction']):.1%}"
        if similarity is not None:
            title += f", image CC={similarity['image_correlation']:.3f}"
        axis.set(xlabel="frequency (Hz)", title=title)
        _panel(axis, label)
    axes[0].set_ylabel("phase velocity (m/s)")
    cbar = figure.colorbar(image, ax=axes, pad=0.012, shrink=0.9)
    cbar.set_label("column-normalized energy")
    figure.suptitle(
        "Dispersion QC | dashed = theoretical modes | all NCF panels fail the clean-ridge gate",
        fontsize=10.5,
    )
    return _save(figure, output_directory / "nnws_v2_dispersion")


def plot_multiseed(summary_csv: Path, output_directory: Path) -> list[Path]:
    _style()
    table = np.genfromtxt(summary_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = np.atleast_1d(table)
    run_directories = sorted(
        {Path(str(value)) for value in rows["run_directory"]}
    )
    records = []
    for run_directory in run_directories:
        manifest = json.loads((run_directory / "manifest.json").read_text())
        seed = int(run_directory.name.split("_")[-1])
        for strategy in STRATEGIES:
            egf = manifest["strategy_egf_quality"][strategy]["zz_symmetric"]
            dispersion = manifest["strategy_dispersion_quality"][strategy]
            records.append(
                (
                    seed,
                    strategy,
                    egf["waveform_correlation_no_shift"],
                    egf["phase_rms_rad"],
                    egf["prearrival_ghost_energy_fraction"],
                    egf["spectral_shape_log_rms_after_global_scale"],
                    dispersion["image"]["ambiguous_frequency_fraction"],
                    dispersion["picks"]["mae_mps"],
                )
            )
    fields = (
        (2, "waveform correlation", True),
        (3, "phase RMS (rad)", False),
        (4, "pre-arrival energy fraction", False),
        (5, "spectral-shape log RMS", False),
        (6, "ambiguous frequency fraction", False),
        (7, "dispersion-pick MAE (m/s)", False),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), constrained_layout=True)
    for axis, (column, title, higher), label in zip(axes.flat, fields, "abcdef"):
        for seed in sorted({int(row[0]) for row in records}):
            values = [
                next(float(row[column]) for row in records if row[0] == seed and row[1] == strategy)
                for strategy in STRATEGIES
            ]
            axis.plot(range(3), values, color="#BBBBBB", linewidth=0.7, alpha=0.8)
            axis.scatter(
                range(3),
                values,
                color=[COLORS[strategy] for strategy in STRATEGIES],
                s=22,
                zorder=3,
            )
        medians = [
            np.median([float(row[column]) for row in records if row[1] == strategy])
            for strategy in STRATEGIES
        ]
        for index, median in enumerate(medians):
            axis.plot([index - 0.20, index + 0.20], [median, median], color="#111111", linewidth=1.2)
        axis.set_xticks(range(3), [LABELS[strategy] for strategy in STRATEGIES])
        for tick in axis.get_xticklabels():
            tick.set_rotation(12)
            tick.set_rotation_mode("anchor")
        axis.set(title=f"{title}\n{'higher' if higher else 'lower'} is favorable")
        axis.grid(alpha=0.14)
        _panel(axis, label)
    figure.suptitle(
        f"Paired robustness across {len(run_directories)} synthetic noise realizations",
        fontsize=10.5,
    )
    return _save(figure, output_directory / "nnws_v2_multiseed")


def write_gate(run_directory: Path, output_path: Path) -> Path:
    manifest = json.loads((run_directory / "manifest.json").read_text())
    candidate = manifest["strategy_egf_quality"]["nnws_v2_shared_oof"]["zz_symmetric"]
    baseline = manifest["strategy_egf_quality"]["baseline_linear"]["zz_symmetric"]
    random = manifest["strategy_egf_quality"]["nnws_v2_random_neff_control"]["zz_symmetric"]
    candidate_dispersion = manifest["strategy_dispersion_quality"]["nnws_v2_shared_oof"]["image"]
    baseline_dispersion = manifest["strategy_dispersion_quality"]["baseline_linear"]["image"]
    reference, reference_quality = reference_dispersion_for_strategy(
        run_directory, "baseline_linear"
    )
    with h5py.File(run_directory / "results.h5", "r") as handle:
        velocity = handle["dispersion/strategies/baseline_linear/velocity_mps"][:]
        candidate_energy = handle["dispersion/strategies/nnws_v2_shared_oof/energy"][:]
        baseline_energy = handle["dispersion/strategies/baseline_linear/energy"][:]
    candidate_similarity = _dispersion_similarity(candidate_energy, reference.energy, velocity)
    baseline_similarity = _dispersion_similarity(baseline_energy, reference.energy, velocity)
    checks = {
        "reference_dispersion_resolved": bool(reference_quality["clean_ridge_pass"]),
        "waveform_correlation_non_worse_than_linear": candidate["waveform_correlation_no_shift"] >= baseline["waveform_correlation_no_shift"],
        "phase_rms_non_worse_than_linear": candidate["phase_rms_rad"] <= baseline["phase_rms_rad"],
        "prearrival_energy_non_worse_than_linear": candidate["prearrival_ghost_energy_fraction"] <= baseline["prearrival_ghost_energy_fraction"],
        "spectral_shape_non_worse_than_linear": candidate["spectral_shape_log_rms_after_global_scale"] <= baseline["spectral_shape_log_rms_after_global_scale"],
        "waveform_correlation_better_than_permuted_control": candidate["waveform_correlation_no_shift"] >= random["waveform_correlation_no_shift"],
        "dispersion_ambiguity_non_worse_than_linear": candidate_dispersion["ambiguous_frequency_fraction"] <= baseline_dispersion["ambiguous_frequency_fraction"],
        "dispersion_image_correlation_non_worse_than_linear": candidate_similarity["image_correlation"] >= baseline_similarity["image_correlation"],
        "dispersion_ridge_mae_non_worse_than_linear": candidate_similarity["dominant_ridge_mae_to_egf_mps"] <= baseline_similarity["dominant_ridge_mae_to_egf_mps"],
    }
    output = {
        "rule": "pass requires a resolved reference plus non-worse waveform and dispersion performance against linear and same-Neff controls",
        "checks": checks,
        "overall_pass": all(checks.values()),
        "candidate_egf": candidate,
        "baseline_egf": baseline,
        "permuted_control_egf": random,
        "candidate_dispersion": candidate_dispersion,
        "baseline_dispersion": baseline_dispersion,
        "candidate_dispersion_similarity_to_egf": candidate_similarity,
        "baseline_dispersion_similarity_to_egf": baseline_similarity,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("output_python/passive_nnws_v2_virtual_source_600m_benchmark"),
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--seed-summary-csv", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_directory or args.run_directory / "qc"
    outputs = []
    outputs.extend(plot_source_weights(args.run_directory, output))
    outputs.extend(plot_waveforms(args.run_directory, output))
    outputs.extend(plot_gathers(args.run_directory, output))
    outputs.extend(plot_dispersion(args.run_directory, output))
    if args.seed_summary_csv is not None:
        outputs.extend(plot_multiseed(args.seed_summary_csv, output))
    gate = write_gate(args.run_directory, output / "nnws_v2_effectiveness_gate.json")
    manifest = {
        "run_directory": str(args.run_directory.resolve()),
        "seed_summary_csv": (
            str(args.seed_summary_csv.resolve())
            if args.seed_summary_csv is not None
            else None
        ),
        "figure_files": [str(path.resolve()) for path in outputs],
        "effectiveness_gate": str(gate.resolve()),
    }
    manifest_path = output / "nnws_v2_qc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(outputs)} figure files, {gate}, and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
