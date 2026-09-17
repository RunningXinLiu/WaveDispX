"""Controlled active/passive benchmark for array dispersion methods.

The benchmark intentionally separates source illumination from the processing
algorithm.  It includes active inline/off-axis cases, passive endfire sources
inside the stationary-phase zone, passive broadside sources outside it, and an
azimuthally diffuse passive control required for a fair SPAC/ESAC evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import time
from typing import Iterable

import h5py
import numpy as np

from .config import Layer, ModelConfig
from .processing.array_dispersion import (
    ArrayDispersionImage,
    MATLAB_KERNEL_METHODS,
    SpatialCoherency,
    canonical_array_method,
    estimate_spatial_coherency,
    method_input_contract,
    spac_dispersion,
    wavefield_dispersion,
)
from .processing.ncf import receiver_pair_ncf_branches
from .processing.theory import theoretical_dispersion


DEFAULT_METHODS = (
    "phase_shift",
    "frequency_bessel",
    "frequency_hankel",
    "fk",
    "tau_p",
    "linear_radon",
    "high_resolution_radon",
    "capon",
    "music",
    "spac",
    "espac",
)

MATLAB_KERNEL_LABELS = {
    "ps_abs": "w01 PS |E|",
    "ps_real": "w02 PS |Re|",
    "ps_real_sqrt_r": "w03 PS sqrt(r) |Re|",
    "fh_real_r": "w04 FH r·dr |Re|",
    "pfh_real_r": "w05 pFH r·dr |Re|",
    "fh_abs_r": "w06 FH r·dr |E|",
    "ps_real_pi4": "w07 PS -pi/4 |Re|",
    "pfh_abs_r": "w08 pFH r·dr |E|",
    "ps_real_pi4_sqrt_r": "w09 PS sqrt(r) -pi/4",
    "ps_imag": "w10 PS |Im|",
    "fh_imag_r": "w11 FH r·dr |Im|",
    "pfh_imag_r": "w12 pFH r·dr |Im|",
    "ps_real_raw": "w13 PS raw Re",
    "fh_real_raw_r": "w14 FH r·dr raw Re",
    "pfh_real_raw_r": "w15 pFH r·dr raw Re",
    "fh_real_pi4_r": "w16 FH r·dr +pi/4",
    "fj_real_r": "FJ J0 x Re(F)",
    "fj_imag_r": "FJ J0 x Im(F)",
    "fj_abs_r": "FJ J0 x |F|",
    "fy_real_r": "FY Y0 x Re(F)",
    "fy_imag_r": "FY Y0 x Im(F)",
    "fy_abs_r": "FY Y0 x |F|",
}
DEFAULT_CASES = (
    "active_inline",
    "active_off_spz",
    "passive_spz",
    "passive_off_spz",
    "passive_diffuse",
    "passive_oblique_off_spz",
)

FREQUENCY_MIN_HZ = 1.0
FREQUENCY_MAX_HZ = 24.0
CORE_FREQUENCY_MIN_HZ = 3.0
CORE_FREQUENCY_MAX_HZ = 10.5
VELOCITY_MIN_MPS = 180.0
VELOCITY_MAX_MPS = 420.0
LOW_FREQUENCY_APERTURE_WAVELENGTHS = 1.5
_OVERVIEW_METHODS = (
    "phase_shift",
    "frequency_bessel",
    "frequency_hankel",
    "fk",
    "tau_p",
    "high_resolution_radon",
    "spac",
    "espac",
)

BENCHMARK_MODEL = ModelConfig(
    layers=(
        Layer(
            thickness_m=10.0,
            vp_mps=380.0,
            vs_mps=220.0,
            density_kgm3=2000.0,
        ),
        Layer(
            thickness_m=math.inf,
            vp_mps=760.0,
            vs_mps=440.0,
            density_kgm3=2000.0,
        ),
    )
)

_REQUIRED_PASSIVE_METHODS = {
    "passive_spz": frozenset(
        {
            "phase_shift",
            "frequency_bessel",
            "frequency_hankel",
            "fk",
            "tau_p",
            "linear_radon",
            "high_resolution_radon",
            "capon",
            "music",
        }
    ),
    "passive_diffuse": frozenset(
        {
            "phase_shift",
            "frequency_bessel",
            "frequency_hankel",
            "spac",
            "espac",
        }
    ),
}
_PASSIVE_GATE_DEFAULT = {
    "maximum_ridge_mae_mps": 10.0,
    "minimum_support_fraction_within_12_mps": 0.85,
    "minimum_median_truth_energy": 0.80,
}
_PASSIVE_GATE_RELAXED = {
    "maximum_ridge_mae_mps": 12.0,
    "minimum_support_fraction_within_12_mps": 0.65,
    "minimum_median_truth_energy": 0.75,
}
_PASSIVE_GATE_DIFFUSE_BESSEL = {
    "maximum_ridge_mae_mps": 15.0,
    "minimum_support_fraction_within_12_mps": 0.55,
    "minimum_median_truth_energy": 0.75,
}


@dataclass(frozen=True)
class DispersionBenchmarkCase:
    name: str
    source_type: str
    illumination: str
    dt_s: float
    radial_gather: np.ndarray
    radial_distance_m: np.ndarray
    linear_gather: np.ndarray
    linear_coordinate_m: np.ndarray
    record_preview: np.ndarray
    coherency: SpatialCoherency | None
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    source_representation: str
    note: str


@dataclass(frozen=True)
class DispersionBenchmarkResult:
    output_directory: Path
    metrics_csv: Path
    results_h5: Path
    manifest_json: Path
    summary_png: Path | None
    summary_pdf: Path | None
    summary_svg: Path | None
    summary_tiff: Path | None
    method_matrix_png: Path | None
    passive_wavefield_png: Path | None
    passive_wavefield_pdf: Path | None
    passive_spac_png: Path | None
    passive_spac_pdf: Path | None
    broadband_atlas_png: Path | None
    broadband_atlas_pdf: Path | None
    broadband_atlas_svg: Path | None
    broadband_atlas_tiff: Path | None
    matlab_kernel_atlas_png: Path | None
    matlab_kernel_atlas_pdf: Path | None
    matlab_kernel_atlas_svg: Path | None
    matlab_kernel_atlas_tiff: Path | None
    rows: tuple[dict[str, object], ...]


@lru_cache(maxsize=1)
def _benchmark_theory_table() -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the model-consistent fundamental Rayleigh phase velocity."""

    frequency = np.linspace(0.75, 26.0, 2021)
    curve = theoretical_dispersion(
        BENCHMARK_MODEL,
        frequency,
        rayleigh_modes=1,
        love_modes=1,
    ).rayleigh_mps[:, 0]
    finite = np.isfinite(curve)
    if np.count_nonzero(finite) < 2:
        raise RuntimeError("benchmark layered model did not yield a Rayleigh curve")
    return frequency[finite], curve[finite]


def benchmark_phase_velocity(frequency_hz: np.ndarray) -> np.ndarray:
    """Fundamental Rayleigh phase velocity of :data:`BENCHMARK_MODEL`."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    table_frequency, table_velocity = _benchmark_theory_table()
    if not np.all(np.diff(table_frequency) > 0.0):
        raise RuntimeError("benchmark theory frequency grid must be increasing")
    return np.interp(
        frequency,
        table_frequency,
        table_velocity,
        left=table_velocity[0],
        right=table_velocity[-1],
    )


@lru_cache(maxsize=1)
def benchmark_frequency_design_limits() -> tuple[float, float]:
    """Return aperture and spatial-alias thresholds for the 240 m/10 m array."""

    frequency = np.linspace(FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ, 5001)
    velocity = benchmark_phase_velocity(frequency)
    aperture_ratio = frequency * 240.0 / velocity
    alias_ratio = 2.0 * 10.0 * frequency / velocity

    def crossing(ratio: np.ndarray, target: float) -> float:
        first = int(np.searchsorted(ratio, target))
        if first <= 0:
            return float(frequency[0])
        if first >= frequency.size:
            return float(frequency[-1])
        return float(
            np.interp(
                target,
                ratio[first - 1 : first + 1],
                frequency[first - 1 : first + 1],
            )
        )

    return crossing(
        aperture_ratio, LOW_FREQUENCY_APERTURE_WAVELENGTHS
    ), crossing(alias_ratio, 1.0)


def _band_envelope(frequency_hz: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    envelope = np.ones_like(frequency)
    low_start, low_stop = 0.6, 1.0
    high_start, high_stop = 24.0, 25.0
    low = (frequency > low_start) & (frequency < low_stop)
    high = (frequency > high_start) & (frequency < high_stop)
    envelope[frequency <= low_start] = 0.0
    envelope[low] = np.sin(
        0.5 * np.pi * (frequency[low] - low_start) / (low_stop - low_start)
    ) ** 2
    envelope[high] = np.cos(
        0.5 * np.pi * (frequency[high] - high_start) / (high_stop - high_start)
    ) ** 2
    envelope[frequency >= high_stop] = 0.0
    envelope[frequency == 0] = 0.0
    return envelope


def _active_case(name: str, *, seed: int, quick: bool) -> DispersionBenchmarkCase:
    receiver_x = np.arange(20.0, 261.0, 10.0)
    receiver_y = np.zeros_like(receiver_x)
    source_x = np.array([0.0])
    source_y = 0.0 if name == "active_inline" else 120.0
    distance = np.hypot(receiver_x, source_y)
    dt_s = 0.005
    sample_count = 2048 if not quick else 1024
    frequency = np.fft.rfftfreq(sample_count, dt_s)
    # Frequencies outside the benchmark band have zero source amplitude.  The
    # floor only keeps their unused phase calculation finite at high Nyquist.
    velocity = np.maximum(benchmark_phase_velocity(frequency), 100.0)
    delay = 0.45 + distance[None, :] / velocity[:, None]
    spectrum = (
        _band_envelope(frequency)[:, None]
        * np.exp(-2j * np.pi * frequency[:, None] * delay)
        / np.sqrt(distance[None, :])
    )
    spectrum[0] = 0.0
    if sample_count % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    gather = np.fft.irfft(spectrum, n=sample_count, axis=0)
    rng = np.random.Generator(np.random.PCG64(seed))
    gather += 0.015 * np.std(gather) * rng.standard_normal(size=gather.shape)
    if name == "active_inline":
        illumination = "inline_endfire"
        processing_distance = distance
        note = (
            "Source lies on the array extension; radial distance and line coordinate "
            "describe the same propagation path."
        )
    else:
        illumination = "off_axis_uncorrected_projected_x"
        processing_distance = receiver_x
        note = (
            "The waveform is synthesized with the true range to a source 120 m off "
            "the array, but every wavefield method receives only projected x. No "
            "source-position correction is applied in this direct-comparison stress case."
        )
    return DispersionBenchmarkCase(
        name=name,
        source_type="active",
        illumination=illumination,
        dt_s=dt_s,
        radial_gather=gather,
        radial_distance_m=processing_distance,
        linear_gather=gather,
        linear_coordinate_m=receiver_x,
        record_preview=gather,
        coherency=None,
        receiver_x_m=receiver_x,
        receiver_y_m=receiver_y,
        source_x_m=source_x,
        source_y_m=np.array([source_y]),
        source_representation="finite_point_source",
        note=note,
    )


def _passive_angles(name: str, source_count: int) -> np.ndarray:
    half = source_count // 2
    if name == "passive_spz":
        spread = np.deg2rad(np.linspace(-8.0, 8.0, half))
        return np.concatenate((spread, np.pi + spread))
    if name == "passive_oblique_off_spz":
        spread = np.deg2rad(np.linspace(-8.0, 8.0, half))
        return np.concatenate((np.pi / 6.0 + spread, 7.0 * np.pi / 6.0 + spread))
    if name == "passive_off_spz":
        spread = np.deg2rad(np.linspace(-8.0, 8.0, half))
        return np.concatenate((0.5 * np.pi + spread, 1.5 * np.pi + spread))
    return 2.0 * np.pi * np.arange(source_count, dtype=np.float64) / source_count


def _synthesize_passive_record(
    name: str,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    *,
    dt_s: float,
    window_samples: int,
    window_count: int,
    source_count: int,
    seed: int,
) -> np.ndarray:
    frequency = np.fft.rfftfreq(window_samples, dt_s)
    velocity = np.maximum(benchmark_phase_velocity(frequency), 100.0)
    wavenumber = 2.0 * np.pi * frequency / velocity
    angle = _passive_angles(name, source_count)
    projection = (
        np.cos(angle)[:, None] * receiver_x_m[None, :]
        + np.sin(angle)[:, None] * receiver_y_m[None, :]
    )
    propagation = np.exp(
        -1j * wavenumber[:, None, None] * projection[None, :, :]
    )
    envelope = _band_envelope(frequency)
    rng = np.random.Generator(np.random.PCG64(seed))
    records = []
    for _ in range(window_count):
        coefficient = (
            rng.standard_normal(size=(frequency.size, source_count))
            + 1j * rng.standard_normal(size=(frequency.size, source_count))
        ) / np.sqrt(2.0 * source_count)
        coefficient *= envelope[:, None]
        spectrum = np.einsum("fs,fsr->fr", coefficient, propagation, optimize=True)
        spectrum[0] = 0.0
        if window_samples % 2 == 0:
            spectrum[-1] = spectrum[-1].real
        segment = np.fft.irfft(spectrum, n=window_samples, axis=0)
        segment += 0.05 * np.std(segment) * rng.standard_normal(size=segment.shape)
        records.append(segment)
    return np.concatenate(records, axis=0)


def _passive_case(name: str, *, seed: int, quick: bool) -> DispersionBenchmarkCase:
    receiver_x = np.arange(0.0, 241.0, 10.0)
    receiver_y = np.zeros_like(receiver_x)
    dt_s = 0.01
    window_samples = 1024
    window_count = 12 if quick else 32
    source_count = 24 if quick else 64
    source_angle = _passive_angles(name, source_count)
    display_radius_m = 360.0
    display_center_x_m = 0.5 * (receiver_x.min() + receiver_x.max())
    source_x = display_center_x_m + display_radius_m * np.cos(source_angle)
    source_y = display_radius_m * np.sin(source_angle)
    record = _synthesize_passive_record(
        name,
        receiver_x,
        receiver_y,
        dt_s=dt_s,
        window_samples=window_samples,
        window_count=window_count,
        source_count=source_count,
        seed=seed,
    )
    window_s = window_samples * dt_s
    coherency = estimate_spatial_coherency(
        record,
        receiver_x,
        receiver_y,
        dt_s,
        window_s=window_s,
        overlap=0.0,
        frequency_min_hz=FREQUENCY_MIN_HZ,
        frequency_max_hz=FREQUENCY_MAX_HZ,
        zero_padding=1.0,
    )
    pair_ncf = receiver_pair_ncf_branches(
        record,
        receiver_x,
        receiver_y,
        dt_s,
        window_s=window_s,
        overlap=0.0,
        maximum_lag_s=2.0,
        normalization="spectral_whitening",
        reference_receiver_index=0,
    )
    # Bilateral endfire and diffuse fields are symmetric by construction.  A
    # symmetric branch also keeps the off-SPZ zero-lag contamination visible.
    gather = pair_ncf.symmetric.data
    pair_coordinate = np.where(
        pair_ncf.pair_i == 0,
        receiver_x[pair_ncf.pair_j] - receiver_x[0],
        receiver_x[pair_ncf.pair_i] - receiver_x[0],
    )
    if name == "passive_spz":
        illumination = "bilateral_endfire_inside_spz"
        note = (
            "Ambient plane waves lie within +/-8 degrees of both receiver-pair "
            "endfire directions, so their cross-correlation phase is stationary."
        )
    elif name == "passive_oblique_off_spz":
        illumination = "bilateral_oblique_30deg_outside_spz"
        note = (
            "Ambient plane waves arrive around 30 and 210 degrees (+/-8 degrees). "
            "They are outside the endfire stationary-phase zone but are not "
            "perpendicular to the line array, providing an intermediate apparent-"
            "slowness control."
        )
    elif name == "passive_off_spz":
        illumination = "bilateral_broadside_outside_spz"
        note = (
            "Ambient plane waves lie near broadside. Their pair phase is nearly "
            "constant, so the virtual-source NCF is an intentional failure case."
        )
    else:
        illumination = "azimuthally_diffuse_control"
        note = (
            "Uniform azimuthal illumination is the control in which the SPAC "
            "coefficient follows J0(2*pi*f*r/c)."
        )
    return DispersionBenchmarkCase(
        name=name,
        source_type="passive",
        illumination=illumination,
        dt_s=dt_s,
        radial_gather=gather,
        radial_distance_m=pair_ncf.distance_m,
        linear_gather=gather,
        linear_coordinate_m=pair_coordinate,
        record_preview=record[:window_samples],
        coherency=coherency,
        receiver_x_m=receiver_x,
        receiver_y_m=receiver_y,
        source_x_m=source_x,
        source_y_m=source_y,
        source_representation="far_field_direction_display_ring",
        note=note,
    )


def build_benchmark_case(
    name: str, *, seed: int = 20260830, quick: bool = False
) -> DispersionBenchmarkCase:
    """Build one deterministic controlled forward case."""

    if name not in DEFAULT_CASES:
        raise ValueError(f"unknown benchmark case {name!r}")
    case_seed = seed + DEFAULT_CASES.index(name) * 1009
    if name.startswith("active_"):
        return _active_case(name, seed=case_seed, quick=quick)
    return _passive_case(name, seed=case_seed, quick=quick)


def _expected_behavior(case_name: str, method: str) -> tuple[bool, str]:
    contract = method_input_contract(method)
    if case_name == "active_inline":
        return contract != "station_pair_coherency", "recovery_control"
    if case_name == "active_off_spz":
        return False, "uncorrected_projected_coordinate_stress"
    if case_name == "passive_spz":
        if contract == "station_pair_coherency":
            return False, "anisotropic_spac_stress"
        return True, "stationary_phase_recovery"
    if case_name == "passive_diffuse":
        return contract in {
            "radial_time_gather",
            "station_pair_coherency",
        }, "diffuse_field_control"
    return False, "off_stationary_phase_stress"


def _image_case_method(
    case: DispersionBenchmarkCase,
    method: str,
    velocity_mps: np.ndarray,
    *,
    quick: bool,
) -> ArrayDispersionImage | None:
    canonical = canonical_array_method(method)
    contract = method_input_contract(canonical)
    if contract == "station_pair_coherency":
        if case.coherency is None:
            return None
        return spac_dispersion(
            case.coherency,
            velocity_mps,
            method=canonical,
            distance_bin_m=10.0 if canonical == "spac" else None,
        )
    common = dict(
        frequency_min_hz=FREQUENCY_MIN_HZ,
        frequency_max_hz=FREQUENCY_MAX_HZ,
        method=canonical,
        zero_padding=4.0,
        distance_taper_fraction=0.25,
        frequency_block_size=12,
        radon_damping_fraction=0.05,
        radon_regularization_fraction=0.05,
        radon_iterations=24 if quick else 50,
        adaptive_diagonal_loading_fraction=0.05,
        adaptive_frequency_smoothing_bins=2,
        music_signal_count=1,
    )
    if contract == "radial_time_gather":
        return wavefield_dispersion(
            case.radial_gather,
            case.radial_distance_m,
            case.dt_s,
            velocity_mps,
            **common,
        )
    return wavefield_dispersion(
        case.linear_gather,
        case.linear_coordinate_m,
        case.dt_s,
        velocity_mps,
        **common,
    )


def _image_metrics(image: ArrayDispersionImage) -> dict[str, float | bool]:
    frequency = image.frequency_hz
    selected = (
        (frequency >= CORE_FREQUENCY_MIN_HZ)
        & (frequency <= CORE_FREQUENCY_MAX_HZ)
    )
    broad = (
        (frequency >= FREQUENCY_MIN_HZ)
        & (frequency <= FREQUENCY_MAX_HZ)
    )
    ridge = image.velocity_mps[np.argmax(image.energy[:, selected], axis=0)]
    truth = benchmark_phase_velocity(frequency[selected])
    error = ridge - truth
    truth_index = np.abs(
        image.velocity_mps[:, None] - truth[None, :]
    ).argmin(axis=0)
    truth_energy = image.energy[truth_index, np.flatnonzero(selected)]
    broad_ridge = image.velocity_mps[np.argmax(image.energy[:, broad], axis=0)]
    broad_truth = benchmark_phase_velocity(frequency[broad])
    broad_error = broad_ridge - broad_truth
    broad_truth_index = np.abs(
        image.velocity_mps[:, None] - broad_truth[None, :]
    ).argmin(axis=0)
    broad_truth_energy = image.energy[broad_truth_index, np.flatnonzero(broad)]
    return {
        "frequency_sample_count": int(np.count_nonzero(selected)),
        "metric_frequency_min_hz": CORE_FREQUENCY_MIN_HZ,
        "metric_frequency_max_hz": CORE_FREQUENCY_MAX_HZ,
        "ridge_mae_mps": float(np.sum(np.abs(error)) / error.size),
        "ridge_median_bias_mps": float(np.quantile(error, 0.5)),
        "ridge_p90_abs_error_mps": float(np.quantile(np.abs(error), 0.9)),
        "support_fraction_within_12_mps": float(
            np.count_nonzero(np.abs(error) <= 12.0) / error.size
        ),
        "median_truth_energy": float(np.quantile(truth_energy, 0.5)),
        "broadband_frequency_sample_count": int(np.count_nonzero(broad)),
        "broadband_ridge_mae_mps": float(
            np.sum(np.abs(broad_error)) / broad_error.size
        ),
        "broadband_ridge_p90_abs_error_mps": float(
            np.quantile(np.abs(broad_error), 0.9)
        ),
        "broadband_support_fraction_within_12_mps": float(
            np.count_nonzero(np.abs(broad_error) <= 12.0) / broad_error.size
        ),
        "broadband_median_truth_energy": float(
            np.quantile(broad_truth_energy, 0.5)
        ),
    }


def _passive_quality_gate(
    case_name: str,
    method: str,
) -> dict[str, float] | None:
    required = _REQUIRED_PASSIVE_METHODS.get(case_name, frozenset())
    if method not in required:
        return None
    if case_name == "passive_diffuse" and method == "frequency_bessel":
        return dict(_PASSIVE_GATE_DIFFUSE_BESSEL)
    if method in {"frequency_bessel", "frequency_hankel", "linear_radon"}:
        return dict(_PASSIVE_GATE_RELAXED)
    return dict(_PASSIVE_GATE_DEFAULT)


def _passes_quality_gate(
    metrics: dict[str, float | bool],
    gate: dict[str, float],
) -> bool:
    return bool(
        float(metrics["ridge_mae_mps"]) <= gate["maximum_ridge_mae_mps"]
        and float(metrics["support_fraction_within_12_mps"])
        >= gate["minimum_support_fraction_within_12_mps"]
        and float(metrics["median_truth_energy"])
        >= gate["minimum_median_truth_energy"]
    )


def _save_results_h5(
    path: Path,
    cases: dict[str, DispersionBenchmarkCase],
    images: dict[tuple[str, str], ArrayDispersionImage],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "wavedispx-dispersion-method-benchmark-v3"
        model_group = handle.create_group("model/layers")
        model_group.attrs["wave_type"] = "fundamental_rayleigh"
        model_group.attrs["dispersion_solver"] = "disba/dunkin"
        model_group.create_dataset(
            "thickness_m",
            data=np.array([layer.thickness_m for layer in BENCHMARK_MODEL.layers]),
        )
        model_group.create_dataset(
            "vp_mps",
            data=np.array([layer.vp_mps for layer in BENCHMARK_MODEL.layers]),
        )
        model_group.create_dataset(
            "vs_mps",
            data=np.array([layer.vs_mps for layer in BENCHMARK_MODEL.layers]),
        )
        model_group.create_dataset(
            "density_kgm3",
            data=np.array(
                [layer.density_kgm3 for layer in BENCHMARK_MODEL.layers]
            ),
        )
        for case_name, case in cases.items():
            group = handle.create_group(f"cases/{case_name}")
            group.attrs["source_type"] = case.source_type
            group.attrs["illumination"] = case.illumination
            group.attrs["note"] = case.note
            group.attrs["dt_s"] = case.dt_s
            group.attrs["source_representation"] = case.source_representation
            group.create_dataset("receiver_x_m", data=case.receiver_x_m)
            group.create_dataset("receiver_y_m", data=case.receiver_y_m)
            group.create_dataset("source_x_m", data=case.source_x_m)
            group.create_dataset("source_y_m", data=case.source_y_m)
            group.create_dataset("radial_gather", data=case.radial_gather.astype(np.float32))
            group.create_dataset("radial_distance_m", data=case.radial_distance_m)
            group.create_dataset("linear_gather", data=case.linear_gather.astype(np.float32))
            group.create_dataset("linear_coordinate_m", data=case.linear_coordinate_m)
            group.create_dataset("record_preview", data=case.record_preview.astype(np.float32))
            if case.coherency is not None:
                spac_group = group.create_group("spatial_coherency")
                spac_group.create_dataset("frequency_hz", data=case.coherency.frequency_hz)
                spac_group.create_dataset("pair_distance_m", data=case.coherency.pair_distance_m)
                spac_group.create_dataset("coherency", data=case.coherency.coherency)
                spac_group.create_dataset("weight", data=case.coherency.weight.astype(np.float32))
                spac_group.create_dataset("pair_i", data=case.coherency.pair_i)
                spac_group.create_dataset("pair_j", data=case.coherency.pair_j)
                spac_group.attrs["valid_windows"] = case.coherency.valid_windows
        for (case_name, method), image in images.items():
            group = handle.create_group(f"images/{case_name}/{method}")
            group.attrs["input_contract"] = image.input_contract
            group.create_dataset("energy", data=image.energy.astype(np.float32))
            group.create_dataset("raw_amplitude", data=image.raw_amplitude.astype(np.float32))
            group.create_dataset("frequency_hz", data=image.frequency_hz)
            group.create_dataset("velocity_mps", data=image.velocity_mps)


def _panel_label(
    axis: object,
    label: str,
    *,
    fontsize: float = 9.0,
    x: float = -0.15,
    y: float = 1.07,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _plot_layer_model(axis: object) -> None:
    depth_bottom_m = 45.0
    axis.axhspan(0.0, 10.0, color="#dceaf4", zorder=0)
    axis.axhspan(10.0, depth_bottom_m, color="#f3e4cf", zorder=0)
    depth = np.array([0.0, 10.0, 10.0, depth_bottom_m])
    vp = np.array([380.0, 380.0, 760.0, 760.0])
    vs = np.array([220.0, 220.0, 440.0, 440.0])
    axis.plot(vp, depth, color="#d95f02", linewidth=1.8, label="Vp")
    axis.plot(vs, depth, color="#1b78b4", linewidth=1.8, label="Vs")
    axis.axhline(10.0, color="0.35", linewidth=0.7, linestyle="--")
    axis.text(560.0, 5.0, "10 m layer", fontsize=6.5, ha="center", va="center")
    axis.text(
        600.0,
        28.0,
        "elastic half-space",
        fontsize=6.5,
        ha="center",
        va="center",
    )
    axis.set_xlim(150.0, 820.0)
    axis.set_ylim(depth_bottom_m, 0.0)
    axis.set_xlabel("velocity (m s⁻¹)")
    axis.set_ylabel("depth (m)")
    axis.set_title("Layered subsurface model\n(density = 2000 kg m⁻³)")
    axis.text(730.0, 42.0, "Vp", color="#d95f02", fontsize=6.5, ha="right")
    axis.text(465.0, 42.0, "Vs", color="#1b78b4", fontsize=6.5, ha="left")


def _plot_active_geometry(axis: object, cases: dict[str, DispersionBenchmarkCase]) -> None:
    inline = cases["active_inline"]
    off_axis = cases["active_off_spz"]
    axis.plot(
        inline.receiver_x_m,
        inline.receiver_y_m,
        color="0.45",
        linewidth=0.8,
        zorder=1,
    )
    axis.scatter(
        inline.receiver_x_m,
        inline.receiver_y_m,
        marker="v",
        s=18,
        color="0.1",
        label="Receiver",
        zorder=3,
    )
    axis.scatter(
        inline.source_x_m,
        inline.source_y_m,
        marker="*",
        s=80,
        color="#1b9e77",
        edgecolor="0.1",
        linewidth=0.4,
        label="Inline source",
        zorder=4,
    )
    axis.scatter(
        off_axis.source_x_m,
        off_axis.source_y_m,
        marker="*",
        s=80,
        color="#d95f02",
        edgecolor="0.1",
        linewidth=0.4,
        label="Off-axis source",
        zorder=4,
    )
    axis.set_xlim(-25.0, 280.0)
    axis.set_ylim(-25.0, 150.0)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("Active observation systems")
    axis.legend(frameon=False, fontsize=5.9, loc="upper right")


def _plot_passive_geometry(
    axis: object,
    case: DispersionBenchmarkCase,
    *,
    color: str,
    title: str,
) -> None:
    from matplotlib.patches import Wedge

    center_x = 0.5 * (case.receiver_x_m.min() + case.receiver_x_m.max())
    radius = float(
        np.quantile(np.hypot(case.source_x_m - center_x, case.source_y_m), 0.5)
    )
    if case.name == "passive_spz":
        sectors = ((-8.0, 8.0), (172.0, 188.0))
    elif case.name == "passive_oblique_off_spz":
        sectors = ((22.0, 38.0), (202.0, 218.0))
    elif case.name == "passive_off_spz":
        sectors = ((82.0, 98.0), (262.0, 278.0))
    else:
        sectors = ((0.0, 360.0),)
    for start, stop in sectors:
        axis.add_patch(
            Wedge(
                (center_x, 0.0),
                radius,
                start,
                stop,
                color=color,
                alpha=0.10,
                linewidth=0.0,
                zorder=0,
            )
        )
    axis.plot(
        case.receiver_x_m,
        case.receiver_y_m,
        color="0.45",
        linewidth=0.8,
        zorder=2,
    )
    axis.scatter(
        case.source_x_m,
        case.source_y_m,
        s=10,
        color=color,
        alpha=0.82,
        label="Noise-source direction",
        zorder=1,
    )
    axis.scatter(
        case.receiver_x_m,
        case.receiver_y_m,
        marker="v",
        s=16,
        color="0.1",
        label="Receiver",
        zorder=3,
    )
    axis.scatter(
        case.receiver_x_m[0],
        case.receiver_y_m[0],
        marker="*",
        s=55,
        color="#6a3d9a",
        edgecolor="white",
        linewidth=0.4,
        label="Virtual source",
        zorder=4,
    )
    axis.set_xlim(-275.0, 515.0)
    axis.set_ylim(-430.0, 430.0)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x (m)")
    axis.set_title(title)


def _plot_passive_error_summary(
    axis: object,
    rows: list[dict[str, object]],
) -> None:
    row_lookup = {(str(row["case"]), str(row["method"])): row for row in rows}
    labels = ("PS", "FJ", "FH", "FK", "tau-p", "HR Radon", "SPAC", "eSPAC")
    y = np.arange(len(_OVERVIEW_METHODS), dtype=np.float64)
    for case_name, color, marker in (
        ("passive_spz", "#1b9e77", "o"),
        ("passive_diffuse", "#377eb8", "s"),
    ):
        values = [
            float(row_lookup[(case_name, method)]["ridge_mae_mps"])
            for method in _OVERVIEW_METHODS
        ]
        axis.scatter(values, y, s=22, marker=marker, color=color, zorder=3)
    axis.axvline(
        _PASSIVE_GATE_DEFAULT["maximum_ridge_mae_mps"],
        color="0.45",
        linewidth=0.8,
        linestyle="--",
    )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 22.0)
    axis.set_xlabel("ridge MAE (m s⁻¹)   ● SPZ   ■ uniform")
    axis.set_title("Passive recovery by method\n(dashed gate = 10 m s⁻¹)")
    axis.grid(axis="x", color="0.88", linewidth=0.5)


def _plot_energy_panel(
    axis: object,
    image: ArrayDispersionImage,
    row: dict[str, object],
    title: str,
    *,
    metric_prefix: str = "PS ",
) -> object:
    rendered = axis.imshow(
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
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    theory_frequency = np.linspace(FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ, 500)
    axis.plot(
        theory_frequency,
        benchmark_phase_velocity(theory_frequency),
        color="#6fe7ff",
        linestyle="--",
        linewidth=1.2,
        label="Model truth",
    )
    low_limit_hz, alias_limit_hz = benchmark_frequency_design_limits()
    axis.axvspan(
        FREQUENCY_MIN_HZ,
        low_limit_hz,
        color="white",
        alpha=0.16,
        linewidth=0.0,
    )
    axis.axvspan(
        alias_limit_hz,
        FREQUENCY_MAX_HZ,
        color="white",
        alpha=0.16,
        linewidth=0.0,
    )
    axis.axvline(low_limit_hz, color="white", linewidth=0.45, linestyle=":")
    axis.axvline(alias_limit_hz, color="white", linewidth=0.45, linestyle=":")
    axis.set_xlim(FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ)
    axis.set_ylim(VELOCITY_MIN_MPS, VELOCITY_MAX_MPS)
    axis.set_xlabel("frequency (Hz)")
    axis.set_title(
        f"{title}\n{metric_prefix}core/wide MAE "
        f"{float(row['ridge_mae_mps']):.1f}/"
        f"{float(row['broadband_ridge_mae_mps']):.1f} m s⁻¹"
    )
    return rendered


def _save_figure_bundle(figure: object, base_path: Path) -> dict[str, Path]:
    outputs = {
        "png": base_path.with_suffix(".png"),
        "pdf": base_path.with_suffix(".pdf"),
        "svg": base_path.with_suffix(".svg"),
        "tiff": base_path.with_suffix(".tiff"),
    }
    figure.savefig(outputs["png"], dpi=300)
    figure.savefig(outputs["pdf"])
    figure.savefig(outputs["svg"])
    figure.savefig(outputs["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    return outputs


def _plot_passive_method_grid(
    plt: object,
    base_path: Path,
    methods: tuple[str, ...],
    images: dict[tuple[str, str], ArrayDispersionImage],
    rows: list[dict[str, object]],
    *,
    figure_title: str,
    group_headers: tuple[tuple[float, str], ...],
) -> dict[str, Path]:
    method_labels = {
        "phase_shift": "PS",
        "frequency_bessel": "FJ",
        "frequency_hankel": "FH",
        "fk": "FK",
        "tau_p": "τ-p",
        "high_resolution_radon": "HR Radon",
        "spac": "SPAC",
        "espac": "eSPAC",
    }
    passive_cases = (
        ("passive_spz", "inside SPZ"),
        ("passive_diffuse", "uniform azimuths"),
        ("passive_off_spz", "outside SPZ"),
    )
    row_lookup = {(str(row["case"]), str(row["method"])): row for row in rows}
    figure, axes = plt.subplots(
        len(passive_cases),
        len(methods),
        figsize=(7.2, 5.8),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.13,
        top=0.86,
        wspace=0.16 if len(methods) > 2 else 0.18,
        hspace=0.36,
    )
    figure.suptitle(figure_title, fontsize=10.0, y=0.985)
    for x, header in group_headers:
        figure.text(x, 0.925, header, fontsize=7.0, ha="center", va="center")

    rendered = None
    for case_index, (case_name, case_label) in enumerate(passive_cases):
        for method_index, method in enumerate(methods):
            axis = axes[case_index, method_index]
            rendered = _plot_energy_panel(
                axis,
                images[(case_name, method)],
                row_lookup[(case_name, method)],
                method_labels[method],
                metric_prefix="",
            )
            if case_index != len(passive_cases) - 1:
                axis.set_xlabel("")
                axis.set_xticklabels([])
            if method_index == 0:
                axis.set_ylabel(f"{case_label}\nphase velocity (m s⁻¹)")
            else:
                axis.set_ylabel("")
                axis.set_yticklabels([])

    panel_axes = list(axes.ravel())
    panel_ids = [chr(ord("a") + index) for index in range(len(panel_axes))]
    label_x = -0.08 if len(methods) > 2 else -0.12
    for label, axis in zip(panel_ids, panel_axes):
        _panel_label(axis, label, fontsize=7.8, x=label_x, y=1.04)

    colorbar_axis = figure.add_axes((0.36, 0.035, 0.34, 0.012))
    colorbar = figure.colorbar(rendered, cax=colorbar_axis, orientation="horizontal")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("normalized dispersion energy", fontsize=6.5, labelpad=3.0)
    colorbar.ax.tick_params(labelsize=5.8)

    figure.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        pass
    else:
        require_matplotlib_panel_alignment(
            figure,
            axes=panel_axes,
            panel_ids=panel_ids,
            json_out=base_path.with_name(f"{base_path.name}_alignment.json"),
            overlay_svg=base_path.with_name(
                f"{base_path.name}_alignment_overlay.svg"
            ),
            require_panel_labels=True,
            strict=True,
        )
    outputs = _save_figure_bundle(figure, base_path)
    plt.close(figure)
    return outputs


def _plot_passive_method_comparisons(
    plt: object,
    output_directory: Path,
    images: dict[tuple[str, str], ArrayDispersionImage],
    rows: list[dict[str, object]],
) -> dict[str, dict[str, Path]]:
    wavefield_methods = (
        "phase_shift",
        "frequency_bessel",
        "frequency_hankel",
        "fk",
        "tau_p",
        "high_resolution_radon",
    )
    spac_methods = ("spac", "espac")
    wavefield = _plot_passive_method_grid(
        plt,
        output_directory / "passive_wavefield_methods",
        wavefield_methods,
        images,
        rows,
        figure_title="Passive wavefield dispersion energy by method",
        group_headers=(
            (0.31, "radial time-gather methods"),
            (0.77, "linear time-gather methods"),
        ),
    )
    spac = _plot_passive_method_grid(
        plt,
        output_directory / "passive_spac_methods",
        spac_methods,
        images,
        rows,
        figure_title="Passive SPAC/eSPAC dispersion energy",
        group_headers=((0.535, "station-pair coherency methods"),),
    )
    return {"wavefield": wavefield, "spac": spac}


def _plot_broadband_benchmark_atlas(
    plt: object,
    base_path: Path,
    images: dict[tuple[str, str], ArrayDispersionImage],
    rows: list[dict[str, object]],
    cases: dict[str, DispersionBenchmarkCase],
) -> dict[str, Path]:
    """Put model, acquisition/source geometry, and every method in one figure."""

    method_labels = {
        "phase_shift": "PS",
        "frequency_bessel": "FJ",
        "frequency_hankel": "FH",
        "fk": "FK / Bartlett",
        "tau_p": "τ-p",
        "linear_radon": "LS Radon",
        "high_resolution_radon": "HR Radon (L1)",
        "capon": "Capon / MVDR",
        "music": "MUSIC",
        "spac": "SPAC",
        "espac": "eSPAC",
    }
    atlas_cases = (
        ("active_inline", "active · inline"),
        ("active_off_spz", "active · off-axis · x only"),
        ("passive_spz", "passive · inside SPZ"),
        ("passive_diffuse", "passive · uniform azimuths"),
        ("passive_oblique_off_spz", "passive · oblique non-SPZ"),
        ("passive_off_spz", "passive · outside SPZ"),
    )
    row_lookup = {(str(row["case"]), str(row["method"])): row for row in rows}
    figure = plt.figure(figsize=(19.0, 15.2))
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 4.15),
        left=0.055,
        right=0.992,
        bottom=0.075,
        top=0.900,
        hspace=0.19,
    )
    top_grid = outer[0].subgridspec(1, 6, wspace=0.34)
    top_axes = [figure.add_subplot(top_grid[0, index]) for index in range(6)]
    _plot_layer_model(top_axes[0])
    _plot_active_geometry(top_axes[1], cases)
    passive_geometry_plan = (
        ("passive_spz", "#1b9e77", "Passive: sources inside SPZ"),
        ("passive_diffuse", "#377eb8", "Passive: uniform azimuths"),
        (
            "passive_oblique_off_spz",
            "#984ea3",
            "Passive: oblique non-SPZ\n(30° and 210°)",
        ),
        ("passive_off_spz", "#d95f02", "Passive: sources outside SPZ"),
    )
    for axis, (case_name, color, title) in zip(
        top_axes[2:], passive_geometry_plan
    ):
        _plot_passive_geometry(axis, cases[case_name], color=color, title=title)
    top_axes[2].set_ylabel("y (m)")
    for axis in top_axes[3:]:
        axis.set_yticklabels([])

    energy_grid = outer[1].subgridspec(
        len(atlas_cases),
        len(DEFAULT_METHODS),
        wspace=0.09,
        hspace=0.18,
    )
    energy_axes: list[object] = []
    rendered = None
    for case_index, (case_name, case_label) in enumerate(atlas_cases):
        for method_index, method in enumerate(DEFAULT_METHODS):
            axis = figure.add_subplot(energy_grid[case_index, method_index])
            energy_axes.append(axis)
            image = images.get((case_name, method))
            row = row_lookup[(case_name, method)]
            if image is None:
                axis.set_facecolor("#eeeeee")
                axis.text(
                    0.5,
                    0.5,
                    "not\napplicable",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="0.35",
                    fontsize=5.4,
                )
                axis.set_xlim(FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ)
                axis.set_ylim(VELOCITY_MIN_MPS, VELOCITY_MAX_MPS)
            else:
                rendered = _plot_energy_panel(
                    axis,
                    image,
                    row,
                    method_labels[method],
                    metric_prefix="",
                )
                if case_name == "passive_oblique_off_spz":
                    apparent_frequency = np.linspace(
                        FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ, 500
                    )
                    axis.plot(
                        apparent_frequency,
                        benchmark_phase_velocity(apparent_frequency)
                        / np.cos(np.deg2rad(30.0)),
                        color="#7CFC8A",
                        linestyle="-.",
                        linewidth=0.9,
                    )
                axis.text(
                    0.97,
                    0.96,
                    f"{float(row['ridge_mae_mps']):.0f}/"
                    f"{float(row['broadband_ridge_mae_mps']):.0f}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=5.0,
                    color="white",
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.38,
                        "edgecolor": "none",
                        "pad": 0.7,
                    },
                )
            axis.set_title(method_labels[method] if case_index == 0 else "", pad=3.0)
            if method_index == 0:
                axis.set_ylabel(f"{case_label}\nphase velocity (m s⁻¹)")
            else:
                axis.set_ylabel("")
                axis.set_yticklabels([])
            if case_index == len(atlas_cases) - 1:
                axis.set_xlabel("frequency (Hz)")
            else:
                axis.set_xlabel("")
                axis.set_xticklabels([])
            axis.tick_params(length=2.0, pad=1.2)

    low_limit_hz, alias_limit_hz = benchmark_frequency_design_limits()
    figure.suptitle(
        "Broadband active/passive surface-wave dispersion benchmark",
        fontsize=14.0,
        y=0.987,
    )
    figure.text(
        0.5,
        0.952,
        "11 methods · 1–24 Hz · cyan dashed = layered-model Rayleigh truth · "
        "green dash-dot = 30° apparent velocity in oblique non-SPZ row\n"
        "active off-axis row: every wavefield method receives projected x only; "
        "no source-position correction · energy is normalized independently at each "
        "frequency (not absolute SNR)\n"
        f"white shade: aperture < {LOW_FREQUENCY_APERTURE_WAVELENGTHS:.1f} wavelengths "
        f"(<{low_limit_hz:.2f} Hz) or spatial aliasing (>{alias_limit_hz:.2f} Hz) · "
        "panel number = core/wide-band ridge MAE (m s⁻¹)",
        fontsize=7.6,
        ha="center",
        va="center",
    )
    if rendered is not None:
        colorbar_axis = figure.add_axes((0.37, 0.027, 0.28, 0.011))
        colorbar = figure.colorbar(
            rendered, cax=colorbar_axis, orientation="horizontal"
        )
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.set_label("normalized dispersion energy", fontsize=7.0, labelpad=3.0)
        colorbar.ax.tick_params(labelsize=6.0)

    figure.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        pass
    else:
        require_matplotlib_panel_alignment(
            figure,
            axes=energy_axes,
            json_out=base_path.with_name(f"{base_path.name}_alignment.json"),
            overlay_svg=base_path.with_name(
                f"{base_path.name}_alignment_overlay.svg"
            ),
            require_panel_labels=False,
            strict=True,
        )
    outputs = _save_figure_bundle(figure, base_path)
    plt.close(figure)
    return outputs


def _plot_matlab_kernel_atlas(
    plt: object,
    base_path: Path,
    images: dict[tuple[str, str], ArrayDispersionImage],
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    """Show every MATLAB PS/FH/pFH/FJ kernel mode in one comparison atlas."""

    atlas_cases = (
        ("active_inline", "active · inline"),
        ("active_off_spz", "active · off-axis · x only"),
        ("passive_spz", "passive · inside SPZ"),
        ("passive_diffuse", "passive · uniform azimuths"),
        ("passive_oblique_off_spz", "passive · oblique non-SPZ"),
        ("passive_off_spz", "passive · outside SPZ"),
    )
    row_lookup = {(str(row["case"]), str(row["method"])): row for row in rows}
    figure, axes = plt.subplots(
        len(atlas_cases),
        len(MATLAB_KERNEL_METHODS),
        figsize=(34.0, 15.2),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.997,
        bottom=0.075,
        top=0.900,
        wspace=0.09,
        hspace=0.18,
    )
    rendered = None
    for case_index, (case_name, case_label) in enumerate(atlas_cases):
        for method_index, method in enumerate(MATLAB_KERNEL_METHODS):
            axis = axes[case_index, method_index]
            row = row_lookup[(case_name, method)]
            rendered = _plot_energy_panel(
                axis,
                images[(case_name, method)],
                row,
                MATLAB_KERNEL_LABELS[method],
                metric_prefix="",
            )
            if case_name == "passive_oblique_off_spz":
                apparent_frequency = np.linspace(
                    FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ, 500
                )
                axis.plot(
                    apparent_frequency,
                    benchmark_phase_velocity(apparent_frequency)
                    / np.cos(np.deg2rad(30.0)),
                    color="#7CFC8A",
                    linestyle="-.",
                    linewidth=0.9,
                )
            axis.text(
                0.97,
                0.96,
                f"{float(row['ridge_mae_mps']):.0f}/"
                f"{float(row['broadband_ridge_mae_mps']):.0f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=5.0,
                color="white",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.38,
                    "edgecolor": "none",
                    "pad": 0.7,
                },
            )
            axis.set_title(
                MATLAB_KERNEL_LABELS[method] if case_index == 0 else "",
                pad=3.0,
            )
            if method_index == 0:
                axis.set_ylabel(f"{case_label}\nphase velocity (m s⁻¹)")
            else:
                axis.set_ylabel("")
                axis.set_yticklabels([])
            if case_index == len(atlas_cases) - 1:
                axis.set_xlabel("frequency (Hz)")
            else:
                axis.set_xlabel("")
                axis.set_xticklabels([])
            axis.tick_params(length=2.0, pad=1.2)

    low_limit_hz, alias_limit_hz = benchmark_frequency_design_limits()
    figure.suptitle(
        "MATLAB PS/FH/pFH/FJ kernel-ablation benchmark",
        fontsize=14.0,
        y=0.987,
    )
    figure.text(
        0.5,
        0.949,
        "w01–w16 reproduce PhaseShift2_modes3; FJ/FY reproduce J0/Y0 radial "
        "transforms with real, imaginary, or magnitude spectra\n"
        "cyan dashed = layered-model Rayleigh truth · green dash-dot = 30° "
        "apparent velocity · active off-axis uses projected x only · "
        "raw-real modes retain sign in HDF5 but display |raw| energy\n"
        f"white shade: aperture < {LOW_FREQUENCY_APERTURE_WAVELENGTHS:.1f} wavelengths "
        f"(<{low_limit_hz:.2f} Hz) or spatial aliasing (>{alias_limit_hz:.2f} Hz) · "
        "panel number = core/wide-band ridge MAE (m s⁻¹)",
        fontsize=7.6,
        ha="center",
        va="center",
    )
    if rendered is not None:
        colorbar_axis = figure.add_axes((0.37, 0.025, 0.28, 0.010))
        colorbar = figure.colorbar(
            rendered, cax=colorbar_axis, orientation="horizontal"
        )
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.set_label("normalized dispersion energy", fontsize=7.0, labelpad=3.0)
        colorbar.ax.tick_params(labelsize=6.0)

    panel_axes = list(axes.ravel())
    figure.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        pass
    else:
        require_matplotlib_panel_alignment(
            figure,
            axes=panel_axes,
            json_out=base_path.with_name(f"{base_path.name}_alignment.json"),
            overlay_svg=base_path.with_name(
                f"{base_path.name}_alignment_overlay.svg"
            ),
            require_panel_labels=False,
            strict=True,
        )
    outputs = _save_figure_bundle(figure, base_path)
    plt.close(figure)
    return outputs


def _plot_summary(
    path: Path,
    case_names: tuple[str, ...],
    method_names: tuple[str, ...],
    images: dict[tuple[str, str], ArrayDispersionImage],
    rows: list[dict[str, object]],
    cases: dict[str, DispersionBenchmarkCase],
) -> tuple[dict[str, Path], Path, dict[str, dict[str, Path]]] | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.3,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    row_lookup = {(str(row["case"]), str(row["method"])): row for row in rows}
    required_cases = {
        "active_inline",
        "active_off_spz",
        "passive_spz",
        "passive_diffuse",
        "passive_oblique_off_spz",
        "passive_off_spz",
    }
    if not required_cases.issubset(cases) or not set(DEFAULT_METHODS).issubset(
        method_names
    ):
        return None
    figure, axes = plt.subplots(3, 3, figsize=(7.2, 7.45), squeeze=False)
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.115,
        top=0.94,
        wspace=0.34,
        hspace=0.44,
    )
    _plot_layer_model(axes[0, 0])
    _plot_active_geometry(axes[0, 1], cases)
    _plot_passive_error_summary(axes[0, 2], rows)

    passive_plan = (
        ("passive_spz", "#1b9e77", "Passive: sources inside SPZ"),
        ("passive_diffuse", "#377eb8", "Passive: uniform azimuths"),
        ("passive_off_spz", "#d95f02", "Passive: sources outside SPZ"),
    )
    for column, (case_name, color, title) in enumerate(passive_plan):
        _plot_passive_geometry(axes[1, column], cases[case_name], color=color, title=title)
        if column == 0:
            axes[1, column].set_ylabel("y (m)")
            axes[1, column].legend(
                frameon=False, fontsize=5.8, loc="upper left", ncol=1
            )
        else:
            axes[1, column].set_yticklabels([])

    rendered = None
    energy_titles = ("SPZ passive", "uniform passive", "outside-SPZ control")
    for column, ((case_name, _color, _title), energy_title) in enumerate(
        zip(passive_plan, energy_titles)
    ):
        image = images[(case_name, "phase_shift")]
        rendered = _plot_energy_panel(
            axes[2, column],
            image,
            row_lookup[(case_name, "phase_shift")],
            energy_title,
        )
        if column == 0:
            axes[2, column].set_ylabel("phase velocity (m s⁻¹)")
        else:
            axes[2, column].set_yticklabels([])
    colorbar_axis = figure.add_axes((0.36, 0.035, 0.34, 0.012))
    colorbar = figure.colorbar(rendered, cax=colorbar_axis, orientation="horizontal")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("normalized dispersion energy", fontsize=6.5, labelpad=3.0)
    colorbar.ax.tick_params(labelsize=5.8)

    panel_axes = list(axes.ravel())
    for label, axis in zip("abcdefghi", panel_axes):
        _panel_label(axis, label)
    figure.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        pass
    else:
        require_matplotlib_panel_alignment(
            figure,
            axes=panel_axes,
            panel_ids=list("abcdefghi"),
            json_out=path.parent / "summary_alignment.json",
            overlay_svg=path.parent / "summary_alignment_overlay.svg",
            require_panel_labels=True,
            strict=True,
        )
    outputs = _save_figure_bundle(figure, path.with_suffix(""))
    plt.close(figure)

    matrix_path = path.parent / "method_matrix.png"
    matrix_figure, matrix_axes = plt.subplots(
        len(case_names),
        len(method_names),
        figsize=(18.0, 10.5),
        squeeze=False,
    )
    matrix_figure.subplots_adjust(
        left=0.075, right=0.995, bottom=0.07, top=0.94, wspace=0.12, hspace=0.25
    )
    for case_index, case_name in enumerate(case_names):
        for method_index, method in enumerate(method_names):
            axis = matrix_axes[case_index, method_index]
            image = images.get((case_name, method))
            row = row_lookup[(case_name, method)]
            if image is None:
                axis.text(0.5, 0.5, "not applicable", ha="center", va="center")
                axis.set_axis_off()
                continue
            axis.imshow(
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
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            theory_frequency = np.linspace(
                FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ, 300
            )
            axis.plot(
                theory_frequency,
                benchmark_phase_velocity(theory_frequency),
                color="#6fe7ff",
                linestyle="--",
                linewidth=0.9,
            )
            low_limit_hz, alias_limit_hz = benchmark_frequency_design_limits()
            axis.axvspan(
                FREQUENCY_MIN_HZ,
                low_limit_hz,
                color="white",
                alpha=0.16,
                linewidth=0.0,
            )
            axis.axvspan(
                alias_limit_hz,
                FREQUENCY_MAX_HZ,
                color="white",
                alpha=0.16,
                linewidth=0.0,
            )
            axis.set_xlim(FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ)
            axis.set_ylim(VELOCITY_MIN_MPS, VELOCITY_MAX_MPS)
            axis.set_title(
                f"{method}\ncore/wide "
                f"{float(row['ridge_mae_mps']):.1f}/"
                f"{float(row['broadband_ridge_mae_mps']):.1f} m/s",
                fontsize=7.0,
            )
            if method_index == 0:
                axis.set_ylabel(f"{case_name}\nvelocity (m/s)", fontsize=7.0)
            else:
                axis.set_yticklabels([])
            if case_index == len(case_names) - 1:
                axis.set_xlabel("frequency (Hz)", fontsize=7.0)
            else:
                axis.set_xticklabels([])
            axis.tick_params(labelsize=6.0)
    matrix_figure.suptitle(
        "Controlled active/passive dispersion benchmark (cyan = model truth)",
        fontsize=11,
    )
    matrix_figure.savefig(matrix_path, dpi=300)
    matrix_figure.savefig(matrix_path.with_suffix(".pdf"))
    plt.close(matrix_figure)
    passive_outputs = _plot_passive_method_comparisons(
        plt,
        path.parent,
        images,
        rows,
    )
    passive_outputs["atlas"] = _plot_broadband_benchmark_atlas(
        plt,
        path.parent / "broadband_all_methods_atlas",
        images,
        rows,
        cases,
    )
    return outputs, matrix_path, passive_outputs


def run_controlled_dispersion_benchmark(
    output_directory: str | Path,
    *,
    case_names: Iterable[str] = DEFAULT_CASES,
    method_names: Iterable[str] = DEFAULT_METHODS,
    seed: int = 20260830,
    quick: bool = False,
    make_plot: bool = True,
) -> DispersionBenchmarkResult:
    """Run, score, and persist the controlled multi-method benchmark."""

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_cases = tuple(case_names)
    selected_methods = tuple(canonical_array_method(name) for name in method_names)
    for case_name in selected_cases:
        if case_name not in DEFAULT_CASES:
            raise ValueError(f"unknown benchmark case {case_name!r}")
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("method_names contains duplicate aliases")
    velocity_step = 4.0 if quick else 2.0
    velocity = np.arange(
        VELOCITY_MIN_MPS,
        VELOCITY_MAX_MPS + velocity_step,
        velocity_step,
    )
    cases = {
        name: build_benchmark_case(name, seed=seed, quick=quick)
        for name in selected_cases
    }
    images: dict[tuple[str, str], ArrayDispersionImage] = {}
    rows: list[dict[str, object]] = []
    for case_name, case in cases.items():
        for method in selected_methods:
            expected, expectation = _expected_behavior(case_name, method)
            start = time.perf_counter()
            image = _image_case_method(case, method, velocity, quick=quick)
            elapsed = time.perf_counter() - start
            row: dict[str, object] = {
                "case": case_name,
                "source_type": case.source_type,
                "illumination": case.illumination,
                "method": method,
                "input_contract": method_input_contract(method),
                "applicable": image is not None,
                "expected_to_recover_truth": expected,
                "expectation": expectation,
                "runtime_s": elapsed,
            }
            gate = _passive_quality_gate(case_name, method)
            row["quality_gate_required"] = bool(not quick and gate is not None)
            row["quality_gate_definition"] = (
                json.dumps(gate, sort_keys=True) if gate is not None else ""
            )
            if image is None:
                row.update(
                    frequency_sample_count=0,
                    metric_frequency_min_hz=CORE_FREQUENCY_MIN_HZ,
                    metric_frequency_max_hz=CORE_FREQUENCY_MAX_HZ,
                    ridge_mae_mps=float("nan"),
                    ridge_median_bias_mps=float("nan"),
                    ridge_p90_abs_error_mps=float("nan"),
                    support_fraction_within_12_mps=float("nan"),
                    median_truth_energy=float("nan"),
                    broadband_frequency_sample_count=0,
                    broadband_ridge_mae_mps=float("nan"),
                    broadband_ridge_p90_abs_error_mps=float("nan"),
                    broadband_support_fraction_within_12_mps=float("nan"),
                    broadband_median_truth_energy=float("nan"),
                    oblique_apparent_ridge_mae_mps=float("nan"),
                    expected_recovery_pass="not_applicable",
                    quality_gate_pass="not_applicable",
                )
            else:
                images[(case_name, method)] = image
                metrics = _image_metrics(image)
                metrics["oblique_apparent_ridge_mae_mps"] = float("nan")
                if case_name == "passive_oblique_off_spz":
                    selected = (
                        (image.frequency_hz >= CORE_FREQUENCY_MIN_HZ)
                        & (image.frequency_hz <= CORE_FREQUENCY_MAX_HZ)
                    )
                    ridge = image.velocity_mps[
                        np.argmax(image.energy[:, selected], axis=0)
                    ]
                    apparent = benchmark_phase_velocity(
                        image.frequency_hz[selected]
                    ) / np.cos(np.deg2rad(30.0))
                    metrics["oblique_apparent_ridge_mae_mps"] = float(
                        np.sum(np.abs(ridge - apparent)) / ridge.size
                    )
                row.update(metrics)
                row["expected_recovery_pass"] = (
                    bool(float(row["ridge_mae_mps"]) <= 12.0)
                    if expected
                    else "diagnostic_only"
                )
                if quick and gate is not None:
                    row["quality_gate_pass"] = "full_resolution_required"
                elif gate is None:
                    row["quality_gate_pass"] = "diagnostic_only"
                else:
                    row["quality_gate_pass"] = _passes_quality_gate(metrics, gate)
            rows.append(row)

    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    h5_path = output / "benchmark.h5"
    _save_results_h5(h5_path, cases, images)
    manifest_path = output / "manifest.json"
    required_gate_rows = [
        row for row in rows if bool(row["quality_gate_required"])
    ]
    manifest = {
        "schema": "wavedispx-dispersion-method-benchmark-v3",
        "seed": seed,
        "quick": quick,
        "model": {
            "wave_type": "fundamental Rayleigh",
            "dispersion_solver": "disba/dunkin",
            "layers": [
                {
                    "thickness_m": (
                        layer.thickness_m
                        if math.isfinite(layer.thickness_m)
                        else None
                    ),
                    "half_space": not math.isfinite(layer.thickness_m),
                    "vp_mps": layer.vp_mps,
                    "vs_mps": layer.vs_mps,
                    "density_kgm3": layer.density_kgm3,
                }
                for layer in BENCHMARK_MODEL.layers
            ],
        },
        "cases": {
            name: {
                "source_type": case.source_type,
                "illumination": case.illumination,
                "receiver_count": int(case.receiver_x_m.size),
                "source_count": int(case.source_x_m.size),
                "source_representation": case.source_representation,
                "note": case.note,
            }
            for name, case in cases.items()
        },
        "methods": list(selected_methods),
        "input_contracts": {
            method: method_input_contract(method) for method in selected_methods
        },
        "matlab_kernel_suite": {
            "phase_shift_modes": "PhaseShift2_modes3 way 1-16",
            "frequency_bessel_modes": (
                "PhaseShift2_modes_FJ J0/Y0 kernels with real/imag/abs spectra"
            ),
            "methods": [
                {
                    "method": method,
                    "display_label": MATLAB_KERNEL_LABELS[method],
                }
                for method in MATLAB_KERNEL_METHODS
                if method in selected_methods
            ],
            "raw_real_note": (
                "ways 13-15 retain signed real stack in raw_amplitude; dispersion "
                "energy uses its absolute value and is therefore identical to ways "
                "2, 4, and 5 respectively"
            ),
        },
        "frequency_design": {
            "processing_band_hz": [FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ],
            "core_metric_band_hz": [
                CORE_FREQUENCY_MIN_HZ,
                CORE_FREQUENCY_MAX_HZ,
            ],
            "receiver_aperture_m": 240.0,
            "receiver_spacing_m": 10.0,
            "low_frequency_aperture_threshold_wavelengths": (
                LOW_FREQUENCY_APERTURE_WAVELENGTHS
            ),
            "low_frequency_aperture_limit_hz": (
                benchmark_frequency_design_limits()[0]
            ),
            "high_frequency_spatial_alias_limit_hz": (
                benchmark_frequency_design_limits()[1]
            ),
        },
        "truth": (
            "fundamental-mode Rayleigh phase velocity computed from the declared "
            "two-layer elastic model over 1-24 Hz"
        ),
        "quality_gates": {
            "scope": "full-resolution passive recovery controls only",
            "default": _PASSIVE_GATE_DEFAULT,
            "relaxed_frequency_bessel": _PASSIVE_GATE_RELAXED,
            "diffuse_frequency_bessel": _PASSIVE_GATE_DIFFUSE_BESSEL,
            "required_count": len(required_gate_rows),
            "all_required_passed": bool(
                required_gate_rows
                and all(row["quality_gate_pass"] is True for row in required_gate_rows)
            ),
            "results": [
                {
                    "case": row["case"],
                    "method": row["method"],
                    "passed": row["quality_gate_pass"],
                }
                for row in required_gate_rows
            ],
        },
        "spz_definition": (
            "passive_spz uses bilateral endfire plane waves within +/-8 degrees; "
            "passive_oblique_off_spz uses bilateral 30/210-degree waves within "
            "+/-8 degrees; passive_off_spz uses bilateral broadside waves within "
            "+/-8 degrees"
        ),
        "important_limit": (
            "SPAC/ESAC are evaluated on normalized station-pair cross spectra; "
            "they are not applied to active shot gathers"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = output / "summary.png"
    plotted = (
        _plot_summary(
            summary_path,
            selected_cases,
            selected_methods,
            images,
            rows,
            cases,
        )
        if make_plot
        else None
    )
    summary_outputs = plotted[0] if plotted is not None else {}
    matrix_png = plotted[1] if plotted is not None else None
    passive_outputs = plotted[2] if plotted is not None else {}
    passive_wavefield = passive_outputs.get("wavefield", {})
    passive_spac = passive_outputs.get("spac", {})
    broadband_atlas = passive_outputs.get("atlas", {})
    matlab_kernel_atlas: dict[str, Path] = {}
    if (
        make_plot
        and set(DEFAULT_CASES).issubset(selected_cases)
        and set(MATLAB_KERNEL_METHODS).issubset(selected_methods)
    ):
        import matplotlib.pyplot as plt

        matlab_kernel_atlas = _plot_matlab_kernel_atlas(
            plt,
            output / "matlab_kernel_variant_atlas",
            images,
            rows,
        )
    return DispersionBenchmarkResult(
        output_directory=output,
        metrics_csv=metrics_path,
        results_h5=h5_path,
        manifest_json=manifest_path,
        summary_png=summary_outputs.get("png"),
        summary_pdf=summary_outputs.get("pdf"),
        summary_svg=summary_outputs.get("svg"),
        summary_tiff=summary_outputs.get("tiff"),
        method_matrix_png=matrix_png,
        passive_wavefield_png=passive_wavefield.get("png"),
        passive_wavefield_pdf=passive_wavefield.get("pdf"),
        passive_spac_png=passive_spac.get("png"),
        passive_spac_pdf=passive_spac.get("pdf"),
        broadband_atlas_png=broadband_atlas.get("png"),
        broadband_atlas_pdf=broadband_atlas.get("pdf"),
        broadband_atlas_svg=broadband_atlas.get("svg"),
        broadband_atlas_tiff=broadband_atlas.get("tiff"),
        matlab_kernel_atlas_png=matlab_kernel_atlas.get("png"),
        matlab_kernel_atlas_pdf=matlab_kernel_atlas.get("pdf"),
        matlab_kernel_atlas_svg=matlab_kernel_atlas.get("svg"),
        matlab_kernel_atlas_tiff=matlab_kernel_atlas.get("tiff"),
        rows=tuple(rows),
    )


__all__ = [
    "DEFAULT_CASES",
    "DEFAULT_METHODS",
    "MATLAB_KERNEL_LABELS",
    "MATLAB_KERNEL_METHODS",
    "BENCHMARK_MODEL",
    "DispersionBenchmarkCase",
    "DispersionBenchmarkResult",
    "benchmark_phase_velocity",
    "benchmark_frequency_design_limits",
    "build_benchmark_case",
    "run_controlled_dispersion_benchmark",
]
