#!/usr/bin/env python3
"""Create auditable QC for the known-source pair-SPZ upper-bound experiment.

Figure contract
---------------
Core conclusion: known-source pair-SPZ constraints improve the stability of
the target dispersion estimate in this synthetic benchmark, but trade away
some complete-waveform EGF fidelity and therefore do not pass a general EGF
recovery gate.  Evidence chain: source geometry and window weights; stacked
waveforms; full gathers; dispersion images; paired multi-seed metrics.
Archetype: quantitative grid.  Backend: Python/matplotlib.  Exports: editable
SVG/PDF plus 300-dpi PNG.  Every plotted value comes from saved run artifacts.
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
from wavedispx.spz_selection import (
    combine_physical_and_data_weights,
    effective_window_count,
    pair_spz_kernels,
)


COLORS = {
    "reference": "#111111",
    "baseline_linear": "#6B6B6B",
    "oracle_pair_spz": "#CC79A7",
    "nnws_v2_shared_oof": "#0072B2",
    "nnws_v2_spz_random_neff_control": "#E69F00",
    "nnws_v2_spz_constrained": "#D55E00",
}
LABELS = {
    "reference": "Theoretical EGF",
    "baseline_linear": "Linear",
    "oracle_pair_spz": "Known-source SPZ only",
    "nnws_v2_shared_oof": "NNWS-v2 OOF",
    "nnws_v2_spz_random_neff_control": "NNWS-v2 + permuted SPZ",
    "nnws_v2_spz_constrained": "NNWS-v2 + known-source SPZ",
}
CORE_STRATEGIES = (
    "baseline_linear",
    "oracle_pair_spz",
    "nnws_v2_shared_oof",
    "nnws_v2_spz_random_neff_control",
    "nnws_v2_spz_constrained",
)
SEED_STRATEGIES = (
    "baseline_linear",
    "nnws_v2_shared_oof",
    "nnws_v2_spz_random_neff_control",
    "nnws_v2_spz_constrained",
)


def _style() -> None:
    configure_figure_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
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
        -0.10,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def _json_safe(value: object) -> object:
    """Convert NumPy values and non-finite metrics into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    return np.atleast_1d(table["x_m"]), np.atleast_1d(table["y_m"])


def _activity_matrix(path: Path, window_count: int, source_count: int) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    matrix = np.zeros((window_count, source_count), dtype=np.float64)
    matrix[
        np.atleast_1d(table["window_index"]).astype(int),
        np.atleast_1d(table["source_index"]).astype(int),
    ] = np.atleast_1d(table["amplitude_multiplier"])
    return matrix


def _combined_weights(
    handle: h5py.File, resolved: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    data_weights = handle["nnws_v2/weights"][:]
    physical_weights = handle["pair_spz/weights"][:]
    physical_random = handle["pair_spz/random_neff_weights"][:]
    pair_config = resolved["pair_spz"]
    assert isinstance(pair_config, dict)
    minimum_fraction = float(pair_config["minimum_effective_fraction"])
    combined = combine_physical_and_data_weights(
        data_weights,
        physical_weights,
        minimum_effective_fraction=minimum_fraction,
    )
    combined_random = combine_physical_and_data_weights(
        data_weights,
        physical_random,
        minimum_effective_fraction=minimum_fraction,
    )
    return combined, combined_random


def plot_source_weights(run_directory: Path, output_directory: Path) -> list[Path]:
    _style()
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    source_x, source_y = _read_xy(run_directory / "sources.csv")
    receiver_x, receiver_y = _read_xy(run_directory / "receivers.csv")
    with h5py.File(run_directory / "results.h5", "r") as handle:
        physical = handle["pair_spz/weights"][:]
        physical_random = handle["pair_spz/random_neff_weights"][:]
        data_weights = handle["nnws_v2/weights"][:]
        combined, combined_random = _combined_weights(handle, resolved)
        fractions = handle["pair_spz/spz_fraction"][:]
        balance = handle["pair_spz/per_window_balance"][:]
        bands = handle["pair_spz/frequency_bands_hz"][:]
        phase_tolerance = float(handle["pair_spz"].attrs["phase_tolerance_rad"])
        pair_scope = str(handle["pair_spz"].attrs["pair_scope"])
        reference_receiver = int(
            handle["pair_spz"].attrs["reference_receiver_index"]
        )
        theory_frequency = handle["theory/frequency_hz"][:]
        theory_velocity = handle["theory/rayleigh_mps"][:, 0]
        optimized_fraction = handle["pair_spz/optimized_spz_fraction"][:]
        optimized_balance = handle["pair_spz/optimized_bilateral_balance"][:]

    ambient = resolved["ambient_sources"]
    assert isinstance(ambient, dict)
    activity_path = Path(str(ambient["activity_file"]))
    activity = _activity_matrix(activity_path, physical.size, source_x.size)
    source_angle = np.mod(
        np.arctan2(source_y - np.mean(receiver_y), source_x - np.mean(receiver_x)),
        2.0 * np.pi,
    )
    source_order = np.argsort(source_angle)
    band_centres = np.mean(bands, axis=1)
    if np.any(np.diff(theory_frequency) <= 0):
        raise ValueError("saved theoretical frequencies must be strictly increasing")
    band_velocity = np.interp(band_centres, theory_frequency, theory_velocity)
    positive_kernel, negative_kernel, pair_count = pair_spz_kernels(
        source_x,
        source_y,
        receiver_x,
        receiver_y,
        band_centres,
        band_velocity,
        phase_tolerance_rad=phase_tolerance,
        pair_scope=pair_scope,
        reference_receiver_index=reference_receiver,
    )
    source_stationarity = np.mean(positive_kernel + negative_kernel, axis=1)
    source_marker_size = 26.0 + 210.0 * source_stationarity / max(
        float(np.max(source_stationarity)), np.finfo(float).eps
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.5), constrained_layout=True)
    source_plot = axes[0, 0].scatter(
        source_x,
        source_y,
        c=np.mean(activity, axis=0),
        s=source_marker_size,
        cmap="magma",
        edgecolor="#0072B2",
        linewidth=0.7,
    )
    axes[0, 0].plot(receiver_x, receiver_y, color="#222222", linewidth=1.0)
    axes[0, 0].scatter(
        receiver_x[[0, -1]], receiver_y[[0, -1]], marker="v", s=38, color="#222222"
    )
    axes[0, 0].set(
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title=f"Known source field; marker size = SPZ kernel ({pair_count} pairs)",
    )
    source_bar = figure.colorbar(source_plot, ax=axes[0, 0], pad=0.02)
    source_bar.set_label("mean source activity")

    activity_plot = axes[0, 1].imshow(
        activity[:, source_order].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        extent=(0, physical.size, 0, source_x.size),
    )
    axes[0, 1].set(
        xlabel="20-s window index",
        ylabel="source rank by azimuth",
        title="Time-varying anisotropic source amplitudes",
    )
    activity_bar = figure.colorbar(activity_plot, ax=axes[0, 1], pad=0.02)
    activity_bar.set_label("activity multiplier")

    window = np.arange(physical.size)
    for name, values in (
        ("oracle_pair_spz", physical),
        ("nnws_v2_shared_oof", data_weights),
        ("nnws_v2_spz_random_neff_control", combined_random),
        ("nnws_v2_spz_constrained", combined),
    ):
        axes[1, 0].plot(
            window,
            values,
            color=COLORS[name],
            linewidth=1.35 if name == "nnws_v2_spz_constrained" else 1.0,
            label=f"{LABELS[name]} (Neff={effective_window_count(values):.1f})",
        )
    axes[1, 0].set(
        xlabel="20-s window index",
        ylabel="mean-normalized weight",
        title="Saved and reconstructed stack weights",
    )
    axes[1, 0].legend(frameon=False, ncol=2)

    band_colors = ("#009E73", "#56B4E9", "#F0E442", "#CC79A7")
    for band_index, (minimum, maximum) in enumerate(bands):
        color = band_colors[band_index % len(band_colors)]
        axes[1, 1].plot(
            window,
            fractions[:, band_index],
            color=color,
            label=f"SPZ fraction {minimum:g}-{maximum:g} Hz",
        )
        axes[1, 1].plot(
            window,
            balance[:, band_index],
            color=color,
            linestyle="--",
            label=f"balance {minimum:g}-{maximum:g} Hz",
        )
    summary = ", ".join(
        f"{minimum:g}-{maximum:g} Hz: F={fraction:.2f}, B={bilateral:.2f}"
        for (minimum, maximum), fraction, bilateral in zip(
            bands, optimized_fraction, optimized_balance
        )
    )
    axes[1, 1].set(
        xlabel="20-s window index",
        ylabel="window diagnostic",
        ylim=(-0.03, 1.04),
        title="Pair-frequency stationary-phase diagnostics",
    )
    axes[1, 1].legend(frameon=False, ncol=2)
    axes[1, 1].text(
        0.01,
        0.02,
        f"Weighted ensemble: {summary}",
        transform=axes[1, 1].transAxes,
        fontsize=6.0,
    )
    for axis, label in zip(axes.flat, "abcd"):
        axis.grid(alpha=0.14)
        _panel(axis, label)
    figure.suptitle(
        "Known-source pair-SPZ upper bound | source field and time-weight audit",
        fontsize=10.5,
    )
    return _save(figure, output_directory / "pair_spz_sources_weights")


def _load_waveforms(
    run_directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[float, float]]:
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    processing = resolved["processing"]
    assert isinstance(processing, dict)
    band = (
        float(processing["frequency_min_hz"]),
        float(processing["frequency_max_hz"]),
    )
    data: dict[str, np.ndarray] = {}
    lag = distance = reference = None
    for strategy in CORE_STRATEGIES:
        lag, distance, candidate, reference = calibrated_waveform_comparison(
            run_directory / "results.h5",
            strategy,
            "symmetric",
            frequency_range_hz=band,
        )
        data[strategy] = candidate
    assert lag is not None and distance is not None and reference is not None
    return lag, distance, reference, data, band


def plot_waveforms(run_directory: Path, output_directory: Path) -> list[Path]:
    _style()
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    processing = resolved["processing"]
    assert isinstance(processing, dict)
    vmin = float(processing["velocity_min_mps"])
    vmax = float(processing["velocity_max_mps"])
    lag, distance, reference, data, band = _load_waveforms(run_directory)
    targets = np.asarray((150.0, 350.0, float(np.max(distance))))
    indices = np.unique(np.abs(distance[:, None] - targets[None, :]).argmin(axis=0))
    if indices.size != 3:
        raise ValueError("the QC benchmark must provide three distinct target distances")
    rows = (
        ("baseline_linear", "nnws_v2_shared_oof"),
        (
            "oracle_pair_spz",
            "nnws_v2_spz_random_neff_control",
            "nnws_v2_spz_constrained",
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.7, 6.3), sharex="col", constrained_layout=True)
    for row_index, strategies in enumerate(rows):
        for column_index, distance_index in enumerate(indices):
            axis = axes[row_index, column_index]
            signal = (lag >= distance[distance_index] / vmax) & (
                lag <= distance[distance_index] / vmin
            )
            axis.axvspan(
                distance[distance_index] / vmax,
                distance[distance_index] / vmin,
                color="#DDDDDD",
                alpha=0.55,
            )
            curves = (("reference", reference[:, distance_index]),) + tuple(
                (strategy, data[strategy][:, distance_index]) for strategy in strategies
            )
            for name, curve in curves:
                scale = np.max(np.abs(curve[signal]), initial=0.0)
                axis.plot(
                    lag,
                    curve / max(float(scale), np.finfo(float).eps),
                    color=COLORS[name],
                    linestyle="--" if name == "reference" else "-",
                    linewidth=1.35 if name in {"reference", "nnws_v2_spz_constrained"} else 1.0,
                    label=LABELS[name],
                )
            axis.set(
                xlim=(0, min(float(lag[-1]), 1.2 * float(np.max(distance)) / vmin)),
                ylim=(-1.18, 1.18),
                xlabel="lag time (s)" if row_index == 1 else None,
                ylabel="normalized amplitude" if column_index == 0 else None,
                title=(
                    ("Data-driven comparison" if row_index == 0 else "Physical constraint and control")
                    + f" | {distance[distance_index]:g} m"
                ),
            )
            axis.grid(alpha=0.14)
            axis.legend(frameon=False, loc="upper right")
            _panel(axis, "abcdef"[row_index * 3 + column_index])
    figure.suptitle(
        f"Stacked symmetric NCF versus theoretical EGF | {band[0]:g}-{band[1]:g} Hz | gray = expected arrival",
        fontsize=10.5,
    )
    return _save(figure, output_directory / "pair_spz_waveform_overlays")


def _column_normalize(values: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(values), axis=0, keepdims=True)
    return values / np.maximum(scale, np.finfo(float).eps)


def plot_gathers(run_directory: Path, output_directory: Path) -> list[Path]:
    _style()
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    processing = resolved["processing"]
    assert isinstance(processing, dict)
    vmin = float(processing["velocity_min_mps"])
    vmax = float(processing["velocity_max_mps"])
    lag, distance, reference, data, _ = _load_waveforms(run_directory)
    panels = (("reference", _column_normalize(reference)),) + tuple(
        (strategy, _column_normalize(data[strategy])) for strategy in CORE_STRATEGIES
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.6, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for axis, (name, values), label in zip(axes.flat, panels, "abcdef"):
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
        axis.set(xlabel="distance (m)", ylabel="lag time (s)", title=LABELS[name])
        _panel(axis, label)
    colorbar = figure.colorbar(image, ax=axes, pad=0.012, shrink=0.88)
    colorbar.set_label("per-distance normalized amplitude")
    figure.suptitle(
        "Symmetric NCF gathers | dotted lines bound expected surface-wave arrivals",
        fontsize=10.5,
    )
    return _save(figure, output_directory / "pair_spz_ncf_gathers")


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
            for strategy in CORE_STRATEGIES
        }
        qualities = {
            strategy: dict(
                handle[f"quality/strategies/{strategy}/dispersion/image"].attrs
            )
            for strategy in CORE_STRATEGIES
        }
    panels = (("reference", reference.energy, reference_quality, None),) + tuple(
        (
            strategy,
            energies[strategy],
            qualities[strategy],
            _dispersion_similarity(energies[strategy], reference.energy, velocity),
        )
        for strategy in CORE_STRATEGIES
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.7, 6.5), sharex=True, sharey=True, constrained_layout=True)
    for axis, (name, energy, quality, similarity), label in zip(axes.flat, panels, "abcdef"):
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
            title += f", image CC={float(similarity['image_correlation']):.3f}"
        axis.set(xlabel="frequency (Hz)", ylabel="phase velocity (m/s)", title=title)
        _panel(axis, label)
    colorbar = figure.colorbar(image, ax=axes, pad=0.012, shrink=0.88)
    colorbar.set_label("column-normalized energy")
    figure.suptitle(
        "Dispersion QC | dashed = theoretical modes | known-source SPZ is an Oracle upper bound",
        fontsize=10.5,
    )
    return _save(figure, output_directory / "pair_spz_dispersion")


def _seed_records(summary_csv: Path) -> list[tuple[object, ...]]:
    table = np.genfromtxt(
        summary_csv, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    rows = np.atleast_1d(table)
    run_directories = sorted({Path(str(value)) for value in rows["run_directory"]})
    records: list[tuple[object, ...]] = []
    for run_directory in run_directories:
        manifest = json.loads((run_directory / "manifest.json").read_text())
        seed = int(run_directory.name.split("_")[-1])
        for strategy in SEED_STRATEGIES:
            egf = manifest["strategy_egf_quality"][strategy]["zz_symmetric"]
            dispersion = manifest["strategy_dispersion_quality"][strategy]
            records.append(
                (
                    seed,
                    strategy,
                    float(egf["waveform_correlation_no_shift"]),
                    float(egf["phase_rms_rad"]),
                    float(egf["prearrival_ghost_energy_fraction"]),
                    float(egf["spectral_shape_log_rms_after_global_scale"]),
                    float(dispersion["image"]["ambiguous_frequency_fraction"]),
                    float(dispersion["picks"]["mae_mps"]),
                )
            )
    return records


def plot_multiseed(summary_csv: Path, output_directory: Path) -> list[Path]:
    _style()
    records = _seed_records(summary_csv)
    seeds = sorted({int(row[0]) for row in records})
    fields = (
        (2, "waveform correlation", True),
        (3, "phase RMS (rad)", False),
        (4, "pre-arrival energy fraction", False),
        (5, "spectral-shape log RMS", False),
        (6, "ambiguous frequency fraction", False),
        (7, "dispersion-pick MAE (m/s)", False),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.9), constrained_layout=True)
    for axis, (column, title, higher), label in zip(axes.flat, fields, "abcdef"):
        for seed in seeds:
            values = [
                next(
                    float(row[column])
                    for row in records
                    if int(row[0]) == seed and row[1] == strategy
                )
                for strategy in SEED_STRATEGIES
            ]
            axis.plot(range(len(SEED_STRATEGIES)), values, color="#BBBBBB", linewidth=0.7)
            axis.scatter(
                range(len(SEED_STRATEGIES)),
                values,
                color=[COLORS[strategy] for strategy in SEED_STRATEGIES],
                s=22,
                zorder=3,
            )
        medians = [
            np.median([float(row[column]) for row in records if row[1] == strategy])
            for strategy in SEED_STRATEGIES
        ]
        for index, median in enumerate(medians):
            axis.plot(
                [index - 0.18, index + 0.18],
                [median, median],
                color="#111111",
                linewidth=1.3,
            )
        axis.set_xticks(range(len(SEED_STRATEGIES)), [LABELS[s] for s in SEED_STRATEGIES])
        for tick in axis.get_xticklabels():
            tick.set_rotation(16)
            tick.set_rotation_mode("anchor")
            tick.set_ha("right")
        axis.set(title=f"{title}\n{'higher' if higher else 'lower'} is favorable")
        axis.grid(alpha=0.14)
        _panel(axis, label)
    figure.suptitle(
        f"Paired robustness across {len(seeds)} synthetic noise realizations | bars = medians",
        fontsize=10.5,
    )
    return _save(figure, output_directory / "pair_spz_multiseed")


def _quality_bundle(run_directory: Path, strategy: str) -> dict[str, object]:
    manifest = json.loads((run_directory / "manifest.json").read_text())
    reference, _ = reference_dispersion_for_strategy(run_directory, "baseline_linear")
    with h5py.File(run_directory / "results.h5", "r") as handle:
        velocity = handle["dispersion/strategies/baseline_linear/velocity_mps"][:]
        energy = handle[f"dispersion/strategies/{strategy}/energy"][:]
    return {
        "egf": manifest["strategy_egf_quality"][strategy]["zz_symmetric"],
        "dispersion": manifest["strategy_dispersion_quality"][strategy],
        "dispersion_similarity_to_egf": _dispersion_similarity(
            energy, reference.energy, velocity
        ),
    }


def _seed_improvement_counts(
    summary_csv: Path, candidate: str, comparator: str
) -> dict[str, int]:
    records = _seed_records(summary_csv)
    seeds = sorted({int(row[0]) for row in records})
    count = {}
    for column, name, higher in (
        (2, "waveform_correlation", True),
        (3, "phase_rms", False),
        (4, "prearrival_energy", False),
        (5, "spectral_shape", False),
        (6, "dispersion_ambiguity", False),
        (7, "dispersion_pick_mae", False),
    ):
        improvements = 0
        for seed in seeds:
            candidate_value = next(
                float(row[column])
                for row in records
                if int(row[0]) == seed and row[1] == candidate
            )
            comparator_value = next(
                float(row[column])
                for row in records
                if int(row[0]) == seed and row[1] == comparator
            )
            improvements += int(
                candidate_value >= comparator_value
                if higher
                else candidate_value <= comparator_value
            )
        count[name] = improvements
    count["seed_count"] = len(seeds)
    return count


def write_gate(
    run_directory: Path, output_path: Path, seed_summary_csv: Path | None
) -> Path:
    candidate = _quality_bundle(run_directory, "nnws_v2_spz_constrained")
    baseline = _quality_bundle(run_directory, "baseline_linear")
    random = _quality_bundle(run_directory, "nnws_v2_spz_random_neff_control")
    _, reference_quality = reference_dispersion_for_strategy(
        run_directory, "baseline_linear"
    )

    candidate_egf = candidate["egf"]
    baseline_egf = baseline["egf"]
    random_egf = random["egf"]
    assert isinstance(candidate_egf, dict)
    assert isinstance(baseline_egf, dict)
    assert isinstance(random_egf, dict)
    waveform_directions = {
        "waveform_correlation_no_shift": "higher",
        "phase_rms_rad": "lower",
        "prearrival_ghost_energy_fraction": "lower",
        "spectral_shape_log_rms_after_global_scale": "lower",
    }

    def non_worse(
        first: dict[str, object], second: dict[str, object], key: str, direction: str
    ) -> bool:
        left = float(first[key])
        right = float(second[key])
        return left >= right if direction == "higher" else left <= right

    complete_checks = {
        "reference_dispersion_resolved": bool(reference_quality["clean_ridge_pass"]),
    }
    for metric, direction in waveform_directions.items():
        complete_checks[f"{metric}_non_worse_than_linear"] = non_worse(
            candidate_egf, baseline_egf, metric, direction
        )
        complete_checks[f"{metric}_non_worse_than_permuted_spz"] = non_worse(
            candidate_egf, random_egf, metric, direction
        )

    candidate_dispersion = candidate["dispersion"]
    baseline_dispersion = baseline["dispersion"]
    random_dispersion = random["dispersion"]
    candidate_similarity = candidate["dispersion_similarity_to_egf"]
    baseline_similarity = baseline["dispersion_similarity_to_egf"]
    assert isinstance(candidate_dispersion, dict)
    assert isinstance(baseline_dispersion, dict)
    assert isinstance(random_dispersion, dict)
    assert isinstance(candidate_similarity, dict)
    assert isinstance(baseline_similarity, dict)
    dispersion_checks = {
        "reference_dispersion_resolved": bool(reference_quality["clean_ridge_pass"]),
        "ambiguity_non_worse_than_linear": float(
            candidate_dispersion["image"]["ambiguous_frequency_fraction"]
        )
        <= float(baseline_dispersion["image"]["ambiguous_frequency_fraction"]),
        "ambiguity_non_worse_than_permuted_spz": float(
            candidate_dispersion["image"]["ambiguous_frequency_fraction"]
        )
        <= float(random_dispersion["image"]["ambiguous_frequency_fraction"]),
        "pick_mae_non_worse_than_linear": float(candidate_dispersion["picks"]["mae_mps"])
        <= float(baseline_dispersion["picks"]["mae_mps"]),
        "pick_mae_non_worse_than_permuted_spz": float(
            candidate_dispersion["picks"]["mae_mps"]
        )
        <= float(random_dispersion["picks"]["mae_mps"]),
        "image_correlation_non_worse_than_linear": float(
            candidate_similarity["image_correlation"]
        )
        >= float(baseline_similarity["image_correlation"]),
        "ridge_mae_to_egf_non_worse_than_linear": float(
            candidate_similarity["dominant_ridge_mae_to_egf_mps"]
        )
        <= float(baseline_similarity["dominant_ridge_mae_to_egf_mps"]),
    }
    seed_counts: dict[str, dict[str, int]] | None = None
    if seed_summary_csv is not None:
        versus_linear = _seed_improvement_counts(
            seed_summary_csv, "nnws_v2_spz_constrained", "baseline_linear"
        )
        versus_random = _seed_improvement_counts(
            seed_summary_csv,
            "nnws_v2_spz_constrained",
            "nnws_v2_spz_random_neff_control",
        )
        seed_counts = {
            "versus_linear": versus_linear,
            "versus_permuted_spz": versus_random,
        }
        for metric in ("dispersion_ambiguity", "dispersion_pick_mae"):
            dispersion_checks[f"{metric}_improved_in_at_least_4_of_5_seeds_vs_linear"] = (
                versus_linear[metric] >= 4
            )
            dispersion_checks[
                f"{metric}_improved_in_at_least_4_of_5_seeds_vs_permuted_spz"
            ] = versus_random[metric] >= 4

    output = {
        "method_scope": (
            "Known forward-source coordinates and exact source realizations are used. "
            "This is an Oracle upper-bound experiment, not a deployable field selector."
        ),
        "complete_egf_gate": {
            "rule": "resolved reference plus every saved waveform and dispersion comparison must be non-worse",
            "checks": {**complete_checks, **dispersion_checks},
            "overall_pass": all((*complete_checks.values(), *dispersion_checks.values())),
        },
        "dispersion_only_gate": {
            "rule": "resolved reference plus non-worse ambiguity, pick, image and multi-seed comparisons",
            "checks": dispersion_checks,
            "overall_pass": all(dispersion_checks.values()),
        },
        "single_seed_quality": {
            "candidate": candidate,
            "linear": baseline,
            "permuted_spz": random,
            "reference_dispersion": reference_quality,
        },
        "multiseed_improvement_counts": seed_counts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(output), indent=2) + "\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("output_python/passive_nnws_v2_pair_spz_600m_benchmark"),
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--seed-summary-csv", type=Path, default=None)
    args = parser.parse_args()
    output_directory = args.output_directory or args.run_directory / "qc"
    output_files: list[Path] = []
    output_files.extend(plot_source_weights(args.run_directory, output_directory))
    output_files.extend(plot_waveforms(args.run_directory, output_directory))
    output_files.extend(plot_gathers(args.run_directory, output_directory))
    output_files.extend(plot_dispersion(args.run_directory, output_directory))
    if args.seed_summary_csv is not None:
        output_files.extend(plot_multiseed(args.seed_summary_csv, output_directory))
    gate = write_gate(
        args.run_directory,
        output_directory / "pair_spz_effectiveness_gate.json",
        args.seed_summary_csv,
    )
    manifest = {
        "run_directory": str(args.run_directory.resolve()),
        "seed_summary_csv": (
            str(args.seed_summary_csv.resolve())
            if args.seed_summary_csv is not None
            else None
        ),
        "figure_files": [str(path.resolve()) for path in output_files],
        "effectiveness_gate": str(gate.resolve()),
        "figure_contract": {
            "core_conclusion": (
                "The known-source SPZ constraint improves target dispersion consistency "
                "but does not improve the complete EGF without trade-offs."
            ),
            "archetype": "quantitative grid",
            "replicate_unit": "independent synthetic noise seed",
        },
    }
    manifest_path = output_directory / "pair_spz_qc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(output_files)} figure files, {gate}, and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
