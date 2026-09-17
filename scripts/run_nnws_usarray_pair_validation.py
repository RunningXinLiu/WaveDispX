#!/usr/bin/env python3
"""Independent-time-block NNWS audit on the Fan et al. USArray station pair.

This is intentionally a field-data *stability* test, not a true-Green-function
test.  NNWS is trained on 2008-01-01--10 and compared with a non-overlapping
linear reference from 2008-01-11--31.  A fixed permutation and a distribution
of permuted NNWS weights preserve the learned effective sample size while
destroying the window-to-weight association.

The preprocessing follows the paper at the level recoverable from its text:
instrument-response removal to displacement, 14--26 s bandpass, one-bit time
normalization, phase-only spectral whitening, 60-min windows, 30-min step, and
plus/minus 500 s correlations.  It is not claimed to reproduce unpublished
NoisePy parameter choices exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nnws import reported_snr, rmsr_selective_stacking, training_rms_ratio
from obspy import UTCDateTime, read, read_inventory
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import correlate

from wavedispx.nnws import NNWSConfig, train_nnws_ccfs
from wavedispx.nnws_qc import configure_figure_style
from wavedispx.nnws_v2 import NNWSV2Config, train_crossfit_proxy_weights


@dataclass(frozen=True)
class StackMetrics:
    training_snr: float
    reported_snr: float
    branch_correlation: float
    precursor_rms_ratio: float
    holdout_waveform_correlation: float
    holdout_phase_rms_rad: float
    holdout_spectral_shape_log_rms: float


def _configure_figure_style() -> None:
    configure_figure_style()
    # Kept explicit here so the standalone field-QC source remains auditable
    # even when a static checker does not follow the imported style helper.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, options in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 300})):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", **options)
        outputs.append(path)
    plt.close(figure)
    return outputs


def _phase_only_whiten(
    data: np.ndarray, sampling_rate_hz: float, frequency_band_hz: tuple[float, float]
) -> np.ndarray:
    values = np.sign(np.asarray(data, dtype=np.float64))
    spectrum = np.fft.rfft(values)
    frequency = np.fft.rfftfreq(values.size, 1.0 / sampling_rate_hz)
    selected = (frequency >= frequency_band_hz[0]) & (
        frequency <= frequency_band_hz[1]
    )
    amplitude = np.abs(spectrum)
    whitened = np.zeros_like(spectrum)
    usable = selected & (amplitude > np.finfo(float).eps)
    whitened[usable] = spectrum[usable] / amplitude[usable]
    return np.fft.irfft(whitened, n=values.size)


def _load_displacement(
    waveform_path: Path,
    inventory_path: Path,
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    frequency_band_hz: tuple[float, float],
):
    stream = read(str(waveform_path), starttime=start_time, endtime=end_time)
    stream.merge(method=1, fill_value=None)
    if len(stream) != 1 or np.ma.isMaskedArray(stream[0].data):
        raise ValueError(f"{waveform_path.name} contains a gap in the selected span")
    trace = stream[0]
    trace.trim(start_time, end_time, nearest_sample=False)
    trace.detrend("linear")
    trace.detrend("demean")
    trace.taper(max_percentage=0.005, max_length=200.0)
    trace.remove_response(
        inventory=read_inventory(str(inventory_path)),
        output="DISP",
        pre_filt=(0.01, 0.02, 0.10, 0.20),
        water_level=60,
        zero_mean=False,
        taper=False,
    )
    trace.filter(
        "bandpass",
        freqmin=frequency_band_hz[0],
        freqmax=frequency_band_hz[1],
        corners=4,
        zerophase=True,
    )
    return trace


def build_ccf_matrix(
    first,
    second,
    *,
    window_s: float,
    step_s: float,
    maximum_lag_s: float,
    frequency_band_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    sampling_rate = float(first.stats.sampling_rate)
    if not np.isclose(sampling_rate, float(second.stats.sampling_rate)):
        raise ValueError("station sampling rates differ")
    common_start = max(first.stats.starttime, second.stats.starttime)
    common_end = min(first.stats.endtime, second.stats.endtime)
    first = first.copy().trim(common_start, common_end, nearest_sample=False)
    second = second.copy().trim(common_start, common_end, nearest_sample=False)
    sample_count = min(first.stats.npts, second.stats.npts)
    first_data = np.asarray(first.data[:sample_count], dtype=np.float64)
    second_data = np.asarray(second.data[:sample_count], dtype=np.float64)
    window_samples = int(round(window_s * sampling_rate))
    step_samples = int(round(step_s * sampling_rate))
    lag_samples = int(round(maximum_lag_s * sampling_rate))
    rows: list[np.ndarray] = []
    start_times: list[float] = []
    for start_index in range(0, sample_count - window_samples + 1, step_samples):
        stop_index = start_index + window_samples
        left = first_data[start_index:stop_index]
        right = second_data[start_index:stop_index]
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            continue
        left = _phase_only_whiten(left, sampling_rate, frequency_band_hz)
        right = _phase_only_whiten(right, sampling_rate, frequency_band_hz)
        full = correlate(right, left, mode="full", method="fft")
        zero_lag = left.size - 1
        ccf = full[zero_lag - lag_samples : zero_lag + lag_samples + 1]
        scale = float(np.sqrt(np.mean(np.square(ccf))))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            continue
        rows.append(ccf / scale)
        start_times.append(float(common_start + start_index / sampling_rate))
    if not rows:
        raise ValueError("no complete CCF windows were produced")
    return np.stack(rows), np.asarray(start_times)


def _waveform_correlation(candidate: np.ndarray, reference: np.ndarray) -> float:
    first = candidate - np.mean(candidate)
    second = reference - np.mean(reference)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.vdot(first, second) / denominator) if denominator > 0 else float("nan")


def _phase_and_spectral_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    dt_s: float,
    frequency_band_hz: tuple[float, float],
) -> tuple[float, float]:
    candidate_spectrum = np.fft.rfft(candidate)
    reference_spectrum = np.fft.rfft(reference)
    frequency = np.fft.rfftfreq(candidate.size, dt_s)
    selected = (frequency >= frequency_band_hz[0]) & (
        frequency <= frequency_band_hz[1]
    )
    reference_amplitude = np.abs(reference_spectrum)
    selected &= reference_amplitude >= 0.05 * np.max(reference_amplitude, initial=0.0)
    if not np.any(selected):
        return float("nan"), float("nan")
    phase_difference = np.angle(
        candidate_spectrum[selected] * np.conj(reference_spectrum[selected])
    )
    phase_rms = float(np.sqrt(np.mean(np.square(phase_difference))))
    candidate_amplitude = np.abs(candidate_spectrum[selected])
    reference_amplitude = reference_amplitude[selected]
    scale = float(
        np.vdot(candidate_amplitude, reference_amplitude)
        / max(np.vdot(candidate_amplitude, candidate_amplitude), np.finfo(float).eps)
    )
    spectral_error = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.log(
                        np.maximum(scale * candidate_amplitude, np.finfo(float).eps)
                    )
                    - np.log(np.maximum(reference_amplitude, np.finfo(float).eps))
                )
            )
        )
    )
    return phase_rms, spectral_error


def stack_metrics(
    stack: np.ndarray,
    reference: np.ndarray,
    *,
    dt_s: float,
    tmin_s: float,
    tmax_s: float,
    tau_s: float,
    frequency_band_hz: tuple[float, float],
) -> StackMetrics:
    zero = stack.size // 2
    negative = stack[: zero + 1][::-1]
    positive = stack[zero:]
    branch_time = np.arange(positive.size) * dt_s
    signal_start = tmin_s - 2 * tau_s
    signal_stop = tmax_s + 2 * tau_s
    signal = (branch_time >= signal_start) & (branch_time <= signal_stop)
    precursor = branch_time < signal_start
    symmetric = 0.5 * (positive + negative)
    precursor_ratio = float(
        np.sqrt(np.mean(np.square(symmetric[precursor])))
        / max(np.sqrt(np.mean(np.square(symmetric[signal]))), np.finfo(float).eps)
    )
    branch_correlation = _waveform_correlation(positive[signal], negative[signal])
    phase_rms, spectral_error = _phase_and_spectral_error(
        stack, reference, dt_s=dt_s, frequency_band_hz=frequency_band_hz
    )
    return StackMetrics(
        training_snr=float(training_rms_ratio(stack, tau_s, tmin_s, tmax_s, dt_s)),
        reported_snr=float(reported_snr(stack, tau_s, tmin_s, tmax_s, dt_s)),
        branch_correlation=branch_correlation,
        precursor_rms_ratio=precursor_ratio,
        holdout_waveform_correlation=_waveform_correlation(stack, reference),
        holdout_phase_rms_rad=phase_rms,
        holdout_spectral_shape_log_rms=spectral_error,
    )


def _write_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_weights(
    matrix: np.ndarray,
    lag_s: np.ndarray,
    weights: np.ndarray,
    start_times: np.ndarray,
    history: dict[str, list[float]],
    output_directory: Path,
) -> list[Path]:
    _configure_figure_style()
    order = np.argsort(weights)[::-1]
    display = matrix[order]
    display /= np.maximum(np.max(np.abs(display), axis=1, keepdims=True), 1e-12)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    axes[0, 0].imshow(
        display,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=(lag_s[0], lag_s[-1], matrix.shape[0], 0),
    )
    axes[0, 0].set(xlabel="lag time (s)", ylabel="CCF rank", title="Training CCFs sorted by NNWS weight")
    axes[0, 1].plot(weights[order], np.arange(weights.size), color="#C44E52")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(xlabel="NNWS weight", ylabel="CCF rank", title="Exact upstream weight allocation")
    relative_day = (start_times - start_times[0]) / 86400.0
    axes[1, 0].scatter(relative_day, weights, s=8, c=weights, cmap="viridis")
    axes[1, 0].set(xlabel="day from 2008-01-01", ylabel="NNWS weight", title="Weight versus recording time")
    axes[1, 1].plot(history["total_loss"], label="Total", color="#111111")
    axes[1, 1].plot(history["snr_loss"], label="SNR loss", color="#0072B2")
    axes[1, 1].plot(history["sym_loss"], label="Symmetry loss", color="#009E73")
    axes[1, 1].set(xlabel="epoch", ylabel="loss", title="Training convergence")
    axes[1, 1].legend(frameon=False)
    for axis, label in zip(axes.flat, "abcd"):
        axis.text(-0.11, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
        axis.grid(alpha=0.15)
    figure.suptitle("USArray TA.A12A–TA.A17A | 10-day NNWS weight audit", fontsize=11)
    return _save_figure(figure, output_directory / "field_nnws_weights")


def _plot_waveforms(
    stacks: dict[str, np.ndarray],
    metrics: dict[str, StackMetrics],
    lag_s: np.ndarray,
    signal_window_s: tuple[float, float],
    output_directory: Path,
) -> list[Path]:
    _configure_figure_style()
    colors = {
        "heldout_linear": "#111111",
        "train_linear": "#5A5A5A",
        "rmsr_ss": "#E69F00",
        "random_neff": "#5875A4",
        "nnws": "#C44E52",
        "nnws_v2_random_neff": "#56B4E9",
        "nnws_v2": "#009E73",
    }
    labels = {
        "heldout_linear": "held-out 21-day linear reference",
        "train_linear": "training 10-day linear",
        "rmsr_ss": "training 10-day RMSR-SS",
        "random_neff": "fixed random Neff control",
        "nnws": "training 10-day NNWS",
        "nnws_v2_random_neff": "v2 fixed random Neff control",
        "nnws_v2": "training 10-day NNWS-v2 OOF",
    }
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.7), constrained_layout=True)
    offset = 0.0
    for name in (
        "heldout_linear",
        "train_linear",
        "rmsr_ss",
        "random_neff",
        "nnws",
        "nnws_v2_random_neff",
        "nnws_v2",
    ):
        curve = stacks[name] / max(np.max(np.abs(stacks[name])), 1e-12)
        axes[0].plot(lag_s, curve + offset, color=colors[name], label=labels[name])
        offset += 1.6
    for sign in (-1, 1):
        axes[0].axvspan(
            sign * signal_window_s[0],
            sign * signal_window_s[1],
            color="#DDDDDD",
            alpha=0.45,
        )
    axes[0].set(xlim=(-240, 240), xlabel="lag time (s)", ylabel="normalized stack + offset", title="Waveform comparison; gray = expected surface-wave window")
    axes[0].legend(frameon=False, ncol=2)

    names = (
        "train_linear",
        "rmsr_ss",
        "random_neff",
        "nnws",
        "nnws_v2_random_neff",
        "nnws_v2",
    )
    x = np.arange(len(names))
    width = 0.18
    fields = (
        ("holdout_waveform_correlation", "hold-out CC", True),
        ("branch_correlation", "causal/acausal CC", True),
        ("precursor_rms_ratio", "precursor RMS ratio", False),
        ("holdout_phase_rms_rad", "hold-out phase RMS", False),
    )
    for index, (field, label, _) in enumerate(fields):
        values = [getattr(metrics[name], field) for name in names]
        axes[1].bar(x + (index - 1.5) * width, values, width, label=label)
    axes[1].set_xticks(x, [labels[name].replace("training 10-day ", "") for name in names])
    for tick_label in axes[1].get_xticklabels():
        tick_label.set_rotation(8)
        tick_label.set_rotation_mode("anchor")
    axes[1].set(ylabel="metric value", title="Mixed-direction metrics (higher is not always better)")
    axes[1].legend(frameon=False, ncol=2)
    for axis, label in zip(axes, "ab"):
        axis.text(-0.07, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
        axis.grid(alpha=0.15)
    figure.suptitle("Field-data stability audit: improvement must transfer to held-out time", fontsize=11)
    return _save_figure(figure, output_directory / "field_nnws_waveforms_metrics")


def _plot_v2_weights(
    matrix: np.ndarray,
    lag_s: np.ndarray,
    fit,
    start_times: np.ndarray,
    output_directory: Path,
) -> list[Path]:
    """Show OOF weights, time folds, proxy CCFs, and per-fold convergence."""
    _configure_figure_style()
    order = np.argsort(fit.weights)[::-1]
    display = matrix[order].copy()
    display /= np.maximum(np.max(np.abs(display), axis=1, keepdims=True), 1e-12)
    relative_day = (start_times - start_times[0]) / 86400.0
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    axes[0, 0].imshow(
        display,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=(lag_s[0], lag_s[-1], matrix.shape[0], 0),
    )
    axes[0, 0].set(
        xlabel="lag time (s)",
        ylabel="CCF rank",
        title="Training CCFs sorted by OOF weight",
    )
    scatter = axes[0, 1].scatter(
        relative_day,
        fit.weights,
        c=fit.fold_index,
        cmap="viridis",
        s=11,
    )
    axes[0, 1].set(
        xlabel="day from 2008-01-01",
        ylabel="OOF shared weight",
        title="Continuous validation folds",
    )
    colorbar = figure.colorbar(scatter, ax=axes[0, 1], pad=0.02)
    colorbar.set_label("fold index")
    axes[1, 0].plot(lag_s, np.mean(matrix, axis=0), color="#5A5A5A", label="linear")
    axes[1, 0].plot(lag_s, fit.proxy_stack, color="#009E73", label="NNWS-v2 OOF")
    axes[1, 0].set(
        xlim=(-240, 240),
        xlabel="lag time (s)",
        ylabel="proxy amplitude",
        title="Array/pair proxy stack",
    )
    axes[1, 0].legend(frameon=False)
    for fold_index, history in enumerate(fit.fold_history):
        axes[1, 1].plot(
            history["total_loss"],
            label=f"fold {fold_index}",
            alpha=0.85,
        )
    axes[1, 1].set(
        xlabel="epoch",
        ylabel="total loss",
        title=f"Per-fold convergence; Neff={fit.effective_window_count:.1f}",
    )
    axes[1, 1].legend(frameon=False, ncol=2)
    for axis, label in zip(axes.flat, "abcd"):
        axis.text(
            -0.11,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        axis.grid(alpha=0.15)
    figure.suptitle(
        "USArray TA.A12A–TA.A17A | NNWS-v2 contiguous-block OOF audit",
        fontsize=11,
    )
    return _save_figure(figure, output_directory / "field_nnws_v2_weights")


def _plot_reliability(
    rows: list[dict[str, object]], output_directory: Path
) -> list[Path]:
    _configure_figure_style()
    metric_fields = (
        ("training_snr", "training SNR", True),
        ("branch_correlation", "causal/acausal CC", True),
        ("holdout_waveform_correlation", "held-out waveform CC", True),
        ("holdout_phase_rms_rad", "held-out phase RMS (rad)", False),
        ("holdout_spectral_shape_log_rms", "held-out spectral log RMS", False),
        ("precursor_rms_ratio", "precursor RMS ratio", False),
    )
    methods = (
        "linear",
        "rmsr_ss",
        "random_permutation",
        "nnws",
        "nnws_v2_random_permutation",
        "nnws_v2",
    )
    colors = {
        "linear": "#5A5A5A",
        "rmsr_ss": "#E69F00",
        "random_permutation": "#5875A4",
        "nnws": "#C44E52",
        "nnws_v2_random_permutation": "#56B4E9",
        "nnws_v2": "#009E73",
    }
    labels = {
        "linear": "linear",
        "rmsr_ss": "RMSR-SS",
        "random_permutation": "v1 permuted",
        "nnws": "NNWS-v1",
        "nnws_v2_random_permutation": "v2 permuted",
        "nnws_v2": "NNWS-v2 OOF",
    }
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    for axis, (field, title, higher_better), panel in zip(axes.flat, metric_fields, "abcdef"):
        for x, method in enumerate(methods):
            values = np.asarray([float(row[field]) for row in rows if row["method"] == method])
            jitter = (
                np.linspace(-0.12, 0.12, values.size)
                if values.size > 1
                else np.zeros(values.size)
            )
            axis.scatter(np.full(values.size, x) + jitter, values, s=17, alpha=0.7, color=colors[method])
            if values.size:
                axis.plot([x - 0.22, x + 0.22], [np.median(values)] * 2, color="#111111", linewidth=1.1)
        axis.set_xticks(range(len(methods)), [labels[name] for name in methods])
        for tick_label in axis.get_xticklabels():
            tick_label.set_rotation(18)
            tick_label.set_rotation_mode("anchor")
        axis.set(title=f"{title}\n{'higher' if higher_better else 'lower'} is favorable")
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold", fontsize=9)
        axis.grid(alpha=0.15)
    figure.suptitle("Optimization-target gains versus independent held-out agreement", fontsize=11)
    return _save_figure(figure, output_directory / "field_nnws_reliability")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=Path("output_python/nnws_usarray_a12a_a17a_2008/input"))
    parser.add_argument("--output-directory", type=Path, default=Path("output_python/nnws_usarray_a12a_a17a_2008"))
    parser.add_argument("--nnws-seeds", type=int, default=10)
    parser.add_argument("--nnws-v2-seeds", type=int, default=5)
    parser.add_argument("--random-repeats", type=int, default=100)
    args = parser.parse_args()
    if args.nnws_seeds < 1 or args.nnws_v2_seeds < 1 or args.random_repeats < 1:
        raise ValueError("NNWS seed counts and random-repeats must be positive")

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "ccf_cache.npz"
    frequency_band = (1.0 / 26.0, 1.0 / 14.0)
    start = UTCDateTime("2008-01-01T00:00:00")
    training_end = UTCDateTime("2008-01-11T00:00:00")
    end = UTCDateTime("2008-02-01T00:00:00")
    if cache.is_file():
        cached = np.load(cache)
        matrix = cached["matrix"]
        start_times = cached["start_times"]
    else:
        inventory_path = args.input_directory / "TA.A12A_A17A.LHZ.station.xml"
        first = _load_displacement(
            args.input_directory / "TA.A12A..LHZ.2008-01-01_2008-02-11.mseed",
            inventory_path,
            start,
            end,
            frequency_band,
        )
        second = _load_displacement(
            args.input_directory / "TA.A17A..LHZ.2008-01-01_2008-02-11.mseed",
            inventory_path,
            start,
            end,
            frequency_band,
        )
        matrix, start_times = build_ccf_matrix(
            first,
            second,
            window_s=3600.0,
            step_s=1800.0,
            maximum_lag_s=500.0,
            frequency_band_hz=frequency_band,
        )
        np.savez_compressed(cache, matrix=matrix, start_times=start_times)

    training = start_times < float(training_end)
    heldout = (start_times >= float(training_end)) & (start_times < float(end))
    train_matrix = matrix[training]
    heldout_matrix = matrix[heldout]
    if train_matrix.shape[0] < 2 or heldout_matrix.shape[0] < 2:
        raise ValueError("training or held-out CCF block is empty")

    distance_m, _, _ = gps2dist_azimuth(48.933899, -115.652603, 48.944801, -110.699402)
    tmin_s = distance_m / 4500.0
    tmax_s = distance_m / 2500.0
    tau_s = 13.0
    dt_s = 1.0
    lag_s = np.arange(-500.0, 501.0, dt_s)
    heldout_reference = np.mean(heldout_matrix, axis=0)
    linear = np.mean(train_matrix, axis=0)
    _, rmsr_sum, rmsr_mask = rmsr_selective_stacking(
        train_matrix, tau_s, tmin_s, tmax_s, dt_s
    )
    rmsr = rmsr_sum / max(int(np.count_nonzero(rmsr_mask)), 1)

    rows: list[dict[str, object]] = []
    stacks: dict[str, np.ndarray] = {
        "heldout_linear": heldout_reference,
        "train_linear": linear,
        "rmsr_ss": rmsr,
    }
    metric_map: dict[str, StackMetrics] = {}
    for method, stack in (("linear", linear), ("rmsr_ss", rmsr)):
        values = stack_metrics(
            stack,
            heldout_reference,
            dt_s=dt_s,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
            tau_s=tau_s,
            frequency_band_hz=frequency_band,
        )
        metric_map["train_linear" if method == "linear" else method] = values
        rows.append({"method": method, "seed": -1, "effective_window_count": float(train_matrix.shape[0] if method == "linear" else np.count_nonzero(rmsr_mask)), **values.__dict__})

    fixed_random_stack = None
    first_fit = None
    failure_rows: list[dict[str, object]] = []
    for seed in range(args.nnws_seeds):
        try:
            fit = train_nnws_ccfs(
                train_matrix,
                dt_s=dt_s,
                signal_window_s=(tmin_s, tmax_s),
                maximum_period_s=26.0,
                config=NNWSConfig(epochs=200, seed=seed, device="cpu", std_mode="upstream_sample"),
            )
        except (RuntimeError, ValueError) as exc:
            failure_rows.append(
                {"method": "nnws_v1", "seed": seed, "reason": str(exc)}
            )
            continue
        if first_fit is None:
            first_fit = fit
            stacks["nnws"] = fit.stack
            metric_map["nnws"] = stack_metrics(
                fit.stack,
                heldout_reference,
                dt_s=dt_s,
                tmin_s=tmin_s,
                tmax_s=tmax_s,
                tau_s=tau_s,
                frequency_band_hz=frequency_band,
            )
        nnws_metrics = stack_metrics(
            fit.stack,
            heldout_reference,
            dt_s=dt_s,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
            tau_s=tau_s,
            frequency_band_hz=frequency_band,
        )
        rows.append({"method": "nnws", "seed": seed, "effective_window_count": fit.effective_window_count, **nnws_metrics.__dict__})
        rng = np.random.default_rng(271828 + seed)
        for repeat in range(args.random_repeats):
            permuted = rng.permutation(fit.weights)
            random_stack = np.average(train_matrix, axis=0, weights=permuted)
            if seed == 0 and repeat == 0:
                fixed_random_stack = random_stack
            random_metrics = stack_metrics(
                random_stack,
                heldout_reference,
                dt_s=dt_s,
                tmin_s=tmin_s,
                tmax_s=tmax_s,
                tau_s=tau_s,
                frequency_band_hz=frequency_band,
            )
            rows.append({"method": "random_permutation", "seed": seed * args.random_repeats + repeat, "effective_window_count": fit.effective_window_count, **random_metrics.__dict__})

    if first_fit is None or fixed_random_stack is None:
        raise RuntimeError("all exact-upstream NNWS initializations failed")
    stacks["random_neff"] = fixed_random_stack
    metric_map["random_neff"] = stack_metrics(
        fixed_random_stack,
        heldout_reference,
        dt_s=dt_s,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        tau_s=tau_s,
        frequency_band_hz=frequency_band,
    )

    first_v2_fit = None
    fixed_v2_random_stack = None
    for seed in range(args.nnws_v2_seeds):
        try:
            fit_v2 = train_crossfit_proxy_weights(
                train_matrix,
                dt_s=dt_s,
                signal_window_s=(tmin_s, tmax_s),
                maximum_period_s=26.0,
                config=NNWSV2Config(
                    folds=5,
                    epochs=200,
                    seed=seed,
                    device="cpu",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            failure_rows.append(
                {"method": "nnws_v2", "seed": seed, "reason": str(exc)}
            )
            continue
        if first_v2_fit is None:
            first_v2_fit = fit_v2
            stacks["nnws_v2"] = fit_v2.proxy_stack
            metric_map["nnws_v2"] = stack_metrics(
                fit_v2.proxy_stack,
                heldout_reference,
                dt_s=dt_s,
                tmin_s=tmin_s,
                tmax_s=tmax_s,
                tau_s=tau_s,
                frequency_band_hz=frequency_band,
            )
        v2_metrics = stack_metrics(
            fit_v2.proxy_stack,
            heldout_reference,
            dt_s=dt_s,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
            tau_s=tau_s,
            frequency_band_hz=frequency_band,
        )
        rows.append(
            {
                "method": "nnws_v2",
                "seed": seed,
                "effective_window_count": fit_v2.effective_window_count,
                **v2_metrics.__dict__,
            }
        )
        rng = np.random.default_rng(314159 + seed)
        for repeat in range(args.random_repeats):
            permuted = rng.permutation(fit_v2.weights)
            random_stack = np.average(train_matrix, axis=0, weights=permuted)
            if seed == 0 and repeat == 0:
                fixed_v2_random_stack = random_stack
            random_metrics = stack_metrics(
                random_stack,
                heldout_reference,
                dt_s=dt_s,
                tmin_s=tmin_s,
                tmax_s=tmax_s,
                tau_s=tau_s,
                frequency_band_hz=frequency_band,
            )
            rows.append(
                {
                    "method": "nnws_v2_random_permutation",
                    "seed": seed * args.random_repeats + repeat,
                    "effective_window_count": fit_v2.effective_window_count,
                    **random_metrics.__dict__,
                }
            )
    if first_v2_fit is None or fixed_v2_random_stack is None:
        raise RuntimeError("all NNWS-v2 initializations failed")
    stacks["nnws_v2_random_neff"] = fixed_v2_random_stack
    metric_map["nnws_v2_random_neff"] = stack_metrics(
        fixed_v2_random_stack,
        heldout_reference,
        dt_s=dt_s,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        tau_s=tau_s,
        frequency_band_hz=frequency_band,
    )

    _write_metrics_csv(rows, output / "field_nnws_metrics.csv")
    (output / "field_nnws_failures.json").write_text(json.dumps(failure_rows, indent=2) + "\n")
    figures = []
    figures.extend(_plot_weights(train_matrix, lag_s, first_fit.weights, start_times[training], first_fit.history, output / "qc"))
    figures.extend(
        _plot_v2_weights(
            train_matrix,
            lag_s,
            first_v2_fit,
            start_times[training],
            output / "qc",
        )
    )
    figures.extend(_plot_waveforms(stacks, metric_map, lag_s, (tmin_s - 2 * tau_s, tmax_s + 2 * tau_s), output / "qc"))
    figures.extend(_plot_reliability(rows, output / "qc"))
    method_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        method_rows.setdefault(str(row["method"]), []).append(row)
    metric_names = tuple(StackMetrics.__dataclass_fields__)
    method_medians = {
        method: {
            name: float(np.median([float(row[name]) for row in values]))
            for name in metric_names
        }
        | {
            "effective_window_count": float(
                np.median([float(row["effective_window_count"]) for row in values])
            )
        }
        for method, values in method_rows.items()
    }
    linear_metrics = method_medians["linear"]
    nnws_metrics = method_medians["nnws"]
    nnws_v2_metrics = method_medians["nnws_v2"]

    def stability_checks(candidate: dict[str, float]) -> dict[str, bool]:
        return {
            "training_snr_improved": candidate["training_snr"]
            >= linear_metrics["training_snr"],
            "branch_correlation_improved": candidate["branch_correlation"]
            >= linear_metrics["branch_correlation"],
            "precursor_rms_ratio_improved": candidate["precursor_rms_ratio"]
            <= linear_metrics["precursor_rms_ratio"],
            "heldout_waveform_correlation_non_worse": candidate[
                "holdout_waveform_correlation"
            ]
            >= linear_metrics["holdout_waveform_correlation"],
            "heldout_phase_rms_non_worse": candidate["holdout_phase_rms_rad"]
            <= linear_metrics["holdout_phase_rms_rad"],
            "heldout_spectral_shape_non_worse": candidate[
                "holdout_spectral_shape_log_rms"
            ]
            <= linear_metrics["holdout_spectral_shape_log_rms"],
        }

    field_checks = stability_checks(nnws_metrics)
    field_checks_v2 = stability_checks(nnws_v2_metrics)
    summary = {
        "interpretation": "field-data stability test; no true Green function is available",
        "station_pair": "TA.A12A--TA.A17A",
        "channel": "LHZ",
        "distance_m": distance_m,
        "training_span": "2008-01-01--2008-01-10",
        "heldout_reference_span": "2008-01-11--2008-01-31",
        "training_ccf_count": int(train_matrix.shape[0]),
        "heldout_ccf_count": int(heldout_matrix.shape[0]),
        "nnws_successful_initializations": int(sum(row["method"] == "nnws" for row in rows)),
        "nnws_failed_initializations": int(
            sum(row["method"] == "nnws_v1" for row in failure_rows)
        ),
        "nnws_v2_successful_initializations": int(
            sum(row["method"] == "nnws_v2" for row in rows)
        ),
        "nnws_v2_failed_initializations": int(
            sum(row["method"] == "nnws_v2" for row in failure_rows)
        ),
        "method_medians": method_medians,
        "field_stability_gate": {
            "rule": "training-target gains must coexist with non-worse agreement to the non-overlapping held-out linear reference",
            "checks": field_checks,
            "overall_pass": all(field_checks.values()),
        },
        "field_stability_gate_v2": {
            "rule": "OOF training-target gains must coexist with non-worse agreement to the non-overlapping held-out linear reference",
            "checks": field_checks_v2,
            "overall_pass": all(field_checks_v2.values()),
        },
        "preprocessing": "response removal to displacement; 14--26 s; one-bit; phase-only whitening; 60-min windows; 30-min step; +/-500 s",
        "preprocessing_limit": "paper does not publish every NoisePy parameter; this is an explicit approximate reproduction",
        "validation_limit": "held-out linear stack is temporally independent but is not a theoretical or active-source Green function",
        "data_source": "EarthScope/IRIS FDSN, network TA, DOI 10.7914/SN/TA",
        "paper": "Fan et al. (2026), DOI 10.1029/2026JB033827",
        "figures": [str(path.resolve()) for path in figures],
    }
    (output / "field_nnws_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
