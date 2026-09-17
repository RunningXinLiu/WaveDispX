"""Array-method dispersion imaging with explicit input contracts.

The existing :mod:`wavedispx.processing.dispersion` module implements the
production phase-shift, frequency-Bessel, and frequency-Hankel transforms for
radial gathers.  This research module adds line-array f-k, tau-p, sparse linear
Radon, SPAC, and ESAC imaging without pretending that all methods consume the
same observable:

* phase/FJ/FH consume a time--radial-distance gather;
* f-k/tau-p/Radon/Capon/MUSIC consume a time--array-coordinate gather;
* SPAC/ESAC consume normalized station-pair cross spectra.

All functions return the same velocity-by-frequency result shape so that the
methods can be benchmarked with common ridge and image-quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import detrend
from scipy.signal.windows import tukey
from scipy.special import hankel1, jv, yv


WavefieldMethod = Literal[
    "phase_shift",
    "frequency_bessel",
    "frequency_hankel",
    "fk",
    "tau_p",
    "linear_radon",
    "high_resolution_radon",
    "capon",
    "music",
    "ps_abs",
    "ps_real",
    "ps_real_sqrt_r",
    "fh_real_r",
    "pfh_real_r",
    "fh_abs_r",
    "ps_real_pi4",
    "pfh_abs_r",
    "ps_real_pi4_sqrt_r",
    "ps_imag",
    "fh_imag_r",
    "pfh_imag_r",
    "ps_real_raw",
    "fh_real_raw_r",
    "pfh_real_raw_r",
    "fh_real_pi4_r",
    "fj_real_r",
    "fj_imag_r",
    "fj_abs_r",
    "fy_real_r",
    "fy_imag_r",
    "fy_abs_r",
]
SPACMethod = Literal["spac", "espac"]
ArrayMethod = WavefieldMethod | SPACMethod
InputContract = Literal[
    "radial_time_gather",
    "linear_time_gather",
    "station_pair_coherency",
]


MATLAB_KERNEL_METHODS: tuple[WavefieldMethod, ...] = (
    "ps_abs",
    "ps_real",
    "ps_real_sqrt_r",
    "fh_real_r",
    "pfh_real_r",
    "fh_abs_r",
    "ps_real_pi4",
    "pfh_abs_r",
    "ps_real_pi4_sqrt_r",
    "ps_imag",
    "fh_imag_r",
    "pfh_imag_r",
    "ps_real_raw",
    "fh_real_raw_r",
    "pfh_real_raw_r",
    "fh_real_pi4_r",
    "fj_real_r",
    "fj_imag_r",
    "fj_abs_r",
    "fy_real_r",
    "fy_imag_r",
    "fy_abs_r",
)

_MATLAB_KERNEL_METHOD_SET = frozenset(MATLAB_KERNEL_METHODS)


_METHOD_ALIASES: dict[str, ArrayMethod] = {
    "phase": "phase_shift",
    "phase_shift": "phase_shift",
    "ps": "phase_shift",
    "bessel": "frequency_bessel",
    "frequency_bessel": "frequency_bessel",
    "fj": "frequency_bessel",
    "hankel": "frequency_hankel",
    "frequency_hankel": "frequency_hankel",
    "fh": "frequency_hankel",
    "f-k": "fk",
    "fk": "fk",
    "tau-p": "tau_p",
    "tau_p": "tau_p",
    "taup": "tau_p",
    "damped_radon": "linear_radon",
    "least_squares_radon": "linear_radon",
    "linear_radon": "linear_radon",
    "lrt": "linear_radon",
    "radon": "linear_radon",
    "hrlrt": "high_resolution_radon",
    "high_resolution_radon": "high_resolution_radon",
    "capon": "capon",
    "mvdr": "capon",
    "music": "music",
    "ps_abs": "ps_abs",
    "ps_real": "ps_real",
    "ps_real_sqrtr": "ps_real_sqrt_r",
    "ps_real_sqrt_r": "ps_real_sqrt_r",
    "fh_real_r": "fh_real_r",
    "fh_real_current": "fh_real_r",
    "pfh_real": "pfh_real_r",
    "pfh_real_r": "pfh_real_r",
    "fh_abs_r": "fh_abs_r",
    "fh_abs_current": "fh_abs_r",
    "ps_real_pi4": "ps_real_pi4",
    "pfh_abs": "pfh_abs_r",
    "pfh_abs_r": "pfh_abs_r",
    "ps_real_pi4_sqrtr": "ps_real_pi4_sqrt_r",
    "ps_real_pi4_sqrt_r": "ps_real_pi4_sqrt_r",
    "ps_imag": "ps_imag",
    "fh_imag_r": "fh_imag_r",
    "fh_imag_current": "fh_imag_r",
    "pfh_imag": "pfh_imag_r",
    "pfh_imag_r": "pfh_imag_r",
    "ps_real_raw": "ps_real_raw",
    "fh_real_raw": "fh_real_raw_r",
    "fh_real_raw_r": "fh_real_raw_r",
    "pfh_real_raw": "pfh_real_raw_r",
    "pfh_real_raw_r": "pfh_real_raw_r",
    "fh_real_pi4": "fh_real_pi4_r",
    "fh_real_pi4_r": "fh_real_pi4_r",
    "fj_real_r": "fj_real_r",
    "fj_imag_r": "fj_imag_r",
    "fj_abs_r": "fj_abs_r",
    "fy_real_r": "fy_real_r",
    "fy_imag_r": "fy_imag_r",
    "fy_abs_r": "fy_abs_r",
    "spac": "spac",
    "esac": "espac",
    "espac": "espac",
}


@dataclass(frozen=True)
class ArrayDispersionImage:
    """Normalized dispersion energy and the unnormalized method response."""

    energy: np.ndarray
    raw_amplitude: np.ndarray
    frequency_hz: np.ndarray
    velocity_mps: np.ndarray
    method: ArrayMethod
    input_contract: InputContract


@dataclass(frozen=True)
class SpatialCoherency:
    """Pair-resolved normalized cross spectra used by SPAC and ESAC.

    ``coherency`` is frequency by pair.  ``coefficient`` is its real part,
    which is the observable fitted to ``J0(2*pi*f*r/c)``.  ``weight`` stores
    magnitude-squared coherence and is used only as a confidence weight.
    """

    frequency_hz: np.ndarray
    pair_distance_m: np.ndarray
    coherency: np.ndarray
    coefficient: np.ndarray
    weight: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    valid_windows: int


@dataclass(frozen=True)
class _PreparedWavefield:
    gather: np.ndarray
    coordinate_m: np.ndarray
    frequency_hz: np.ndarray
    spectrum: np.ndarray
    nfft: int


def canonical_array_method(method: str) -> ArrayMethod:
    """Return the canonical method name, accepting common PS/FJ/FH aliases."""

    key = str(method).strip().lower().replace(" ", "_")
    try:
        return _METHOD_ALIASES[key]
    except KeyError as exc:
        known = ", ".join(sorted(_METHOD_ALIASES))
        raise ValueError(f"unknown array dispersion method {method!r}; use one of {known}") from exc


def method_input_contract(method: str) -> InputContract:
    """Describe the observable required by one dispersion method."""

    canonical = canonical_array_method(method)
    if canonical in {
        "phase_shift",
        "frequency_bessel",
        "frequency_hankel",
    } or canonical in _MATLAB_KERNEL_METHOD_SET:
        return "radial_time_gather"
    if canonical in {
        "fk",
        "tau_p",
        "linear_radon",
        "high_resolution_radon",
        "capon",
        "music",
    }:
        return "linear_time_gather"
    return "station_pair_coherency"


def _power_of_two_at_least(value: int) -> int:
    return 1 << max(0, (int(value) - 1).bit_length())


def _normalize_columns(raw: np.ndarray) -> np.ndarray:
    energy = np.asarray(raw, dtype=np.float64).copy()
    maximum = energy.max(axis=0, initial=0.0)
    maximum[maximum == 0] = 1.0
    energy /= maximum
    return energy


def _validate_velocity(velocity_mps: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity_mps, dtype=np.float64).reshape(-1)
    if (
        velocity.size < 2
        or np.any(~np.isfinite(velocity))
        or np.any(velocity <= 0)
        or np.any(np.diff(velocity) <= 0)
    ):
        raise ValueError("velocity_mps must contain increasing finite positive values")
    return velocity


def _prepare_wavefield(
    data: np.ndarray,
    coordinate_m: np.ndarray,
    dt_s: float,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float,
    zero_padding: float,
    time_taper_fraction: float,
    radial: bool,
) -> _PreparedWavefield:
    gather = np.asarray(data, dtype=np.float64)
    coordinate = np.asarray(coordinate_m, dtype=np.float64).reshape(-1)
    if gather.ndim != 2 or gather.shape[1] != coordinate.size:
        raise ValueError("data must be time-by-coordinate and match coordinate_m")
    if (
        gather.shape[0] < 4
        or coordinate.size < 2
        or np.any(~np.isfinite(gather))
        or np.any(~np.isfinite(coordinate))
    ):
        raise ValueError("at least four finite samples and two coordinates are required")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if (
        not np.isfinite(frequency_min_hz)
        or not np.isfinite(frequency_max_hz)
        or frequency_min_hz < 0
        or frequency_max_hz <= frequency_min_hz
    ):
        raise ValueError("frequency bounds must satisfy 0 <= minimum < maximum")
    if not np.isfinite(zero_padding) or zero_padding < 1:
        raise ValueError("zero_padding must be finite and >= 1")
    if not 0 <= time_taper_fraction <= 1:
        raise ValueError("time_taper_fraction must lie in [0, 1]")
    if radial:
        keep = coordinate > 0
        gather = gather[:, keep]
        coordinate = coordinate[keep]
        if coordinate.size < 2:
            raise ValueError("radial methods require at least two positive distances")

    order = np.argsort(coordinate)
    gather = gather[:, order]
    coordinate = coordinate[order]
    unique_coordinate, inverse, count = np.unique(
        coordinate, return_inverse=True, return_counts=True
    )
    if unique_coordinate.size < 2:
        raise ValueError("at least two unique coordinates are required")
    if unique_coordinate.size != coordinate.size:
        stacked = np.zeros((gather.shape[0], unique_coordinate.size), dtype=np.float64)
        for input_index, output_index in enumerate(inverse):
            stacked[:, output_index] += gather[:, input_index]
        gather = stacked / count
        coordinate = unique_coordinate

    gather = detrend(gather, axis=0, type="linear")
    gather *= tukey(gather.shape[0], time_taper_fraction).reshape(-1, 1)
    nfft = _power_of_two_at_least(int(np.ceil(gather.shape[0] * zero_padding)))
    full_frequency = np.fft.rfftfreq(nfft, dt_s)
    selected = (
        (full_frequency > 0)
        & (full_frequency >= frequency_min_hz)
        & (full_frequency <= min(frequency_max_hz, 0.5 / dt_s))
    )
    if np.count_nonzero(selected) < 2:
        raise ValueError("selected frequency band contains fewer than two FFT samples")
    spectrum = np.fft.rfft(gather, n=nfft, axis=0)[selected]
    return _PreparedWavefield(
        gather=gather,
        coordinate_m=coordinate,
        frequency_hz=full_frequency[selected],
        spectrum=spectrum,
        nfft=nfft,
    )


def _spatial_taper(count: int, fraction: float) -> np.ndarray:
    if not 0 <= fraction <= 1:
        raise ValueError("distance_taper_fraction must lie in [0, 1]")
    if fraction == 0:
        return np.ones(count, dtype=np.float64)
    return tukey(count, fraction)


def _radial_area_weights(distance_m: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (distance_m[:-1] + distance_m[1:])
    left = max(0.0, distance_m[0] - 0.5 * (distance_m[1] - distance_m[0]))
    right = distance_m[-1] + 0.5 * (distance_m[-1] - distance_m[-2])
    edge = np.concatenate(([left], midpoint, [right]))
    return 0.5 * (edge[1:] ** 2 - edge[:-1] ** 2)


def _phase_or_fk_response(
    prepared: _PreparedWavefield,
    velocity_mps: np.ndarray,
    *,
    phase_only: bool,
    distance_taper_fraction: float,
    frequency_block_size: int,
) -> np.ndarray:
    if frequency_block_size < 1:
        raise ValueError("frequency_block_size must be positive")
    coordinate = prepared.coordinate_m - prepared.coordinate_m.min()
    spectrum = prepared.spectrum.copy()
    if phase_only:
        spectrum /= np.maximum(np.abs(spectrum), np.finfo(float).eps)
    spectrum *= _spatial_taper(coordinate.size, distance_taper_fraction)
    raw = np.zeros((velocity_mps.size, prepared.frequency_hz.size), dtype=np.float64)
    inverse_velocity = 1.0 / velocity_mps
    omega = 2.0 * np.pi * prepared.frequency_hz
    for first in range(0, omega.size, frequency_block_size):
        stop = min(omega.size, first + frequency_block_size)
        steering = np.exp(
            1j
            * omega[first:stop, None, None]
            * inverse_velocity[None, :, None]
            * coordinate[None, None, :]
        )
        raw[:, first:stop] = np.abs(
            np.einsum(
                "fvc,fc->vf", steering, spectrum[first:stop], optimize=True
            )
        )
    return raw


def _radial_transform_response(
    prepared: _PreparedWavefield,
    velocity_mps: np.ndarray,
    *,
    hankel: bool,
    component: Literal["vertical", "radial", "transverse"],
    radial_weighting: Literal["area", "uniform"],
    distance_taper_fraction: float,
    frequency_block_size: int,
) -> np.ndarray:
    if radial_weighting not in {"area", "uniform"}:
        raise ValueError("radial_weighting must be area or uniform")
    order = 1 if component == "radial" else 0
    distance = prepared.coordinate_m
    radial_weight = (
        _radial_area_weights(distance)
        if radial_weighting == "area"
        else np.ones(distance.size)
    )
    weighted = prepared.spectrum * (
        radial_weight * _spatial_taper(distance.size, distance_taper_fraction)
    )[None, :]
    raw = np.zeros((velocity_mps.size, prepared.frequency_hz.size), dtype=np.float64)
    inverse_velocity = 1.0 / velocity_mps
    omega = 2.0 * np.pi * prepared.frequency_hz
    for first in range(0, omega.size, frequency_block_size):
        stop = min(omega.size, first + frequency_block_size)
        argument = (
            omega[first:stop, None, None]
            * inverse_velocity[None, :, None]
            * distance[None, None, :]
        )
        kernel = hankel1(order, argument) if hankel else jv(order, argument)
        raw[:, first:stop] = np.abs(
            np.einsum("fvc,fc->vf", kernel, weighted[first:stop], optimize=True)
        )
    return raw


def _matlab_kernel_variant_response(
    prepared: _PreparedWavefield,
    velocity_mps: np.ndarray,
    *,
    method: str,
    distance_taper_fraction: float,
    frequency_block_size: int,
) -> np.ndarray:
    """Reproduce the PS/FH/pFH/FJ kernel variants used by the MATLAB suite.

    The 16 PS/FH/pFH modes follow ``PhaseShift2_modes3``: spectra are
    phase-only, PS uses a per-trace sum, and FH/pFH use the default ``r*dr``
    radial integration weight.  The six FJ/FY modes follow
    ``PhaseShift2_modes_FJ`` by applying J0 or Y0 to the real, imaginary, or
    magnitude part of the unwhitened spectrum with trapezoidal ``r*dr``
    integration.  Raw-real modes retain their sign in ``raw_amplitude``;
    callers take the absolute value only when constructing dispersion energy.
    """

    if method not in _MATLAB_KERNEL_METHOD_SET:
        raise ValueError(f"unknown MATLAB kernel variant {method!r}")
    if frequency_block_size < 1:
        raise ValueError("frequency_block_size must be positive")

    distance = prepared.coordinate_m
    taper = _spatial_taper(distance.size, distance_taper_fraction)
    area_weight = _radial_area_weights(distance) * taper
    inverse_velocity = 1.0 / velocity_mps
    omega = 2.0 * np.pi * prepared.frequency_hz
    epsilon = np.finfo(float).eps

    is_fj_family = method.startswith(("fj_", "fy_"))
    if is_fj_family:
        if "_real_" in method:
            field = np.real(prepared.spectrum)
        elif "_imag_" in method:
            field = np.imag(prepared.spectrum)
        else:
            field = np.abs(prepared.spectrum)
    else:
        field = prepared.spectrum / np.maximum(
            np.abs(prepared.spectrum), epsilon
        )

    raw = np.zeros((velocity_mps.size, prepared.frequency_hz.size), dtype=np.float64)
    for first in range(0, omega.size, frequency_block_size):
        stop = min(omega.size, first + frequency_block_size)
        argument = (
            omega[first:stop, None, None]
            * inverse_velocity[None, :, None]
            * distance[None, None, :]
        )
        block_field = field[first:stop]

        if is_fj_family:
            kernel = yv(0, argument) if method.startswith("fy_") else jv(0, argument)
            stack = np.einsum(
                "fvc,fc,c->vf",
                kernel,
                block_field,
                area_weight,
                optimize=True,
            )
            raw[:, first:stop] = np.abs(stack)
            continue

        if method.startswith("ps_"):
            kernel = np.exp(1j * argument)
            weight = taper
            if "sqrt_r" in method:
                weight = weight * np.sqrt(distance)
        else:
            hankel = hankel1(0, argument)
            if method.startswith("pfh_"):
                kernel = hankel / np.maximum(np.abs(hankel), epsilon)
            else:
                kernel = hankel
            weight = area_weight

        stack = np.einsum(
            "fvc,fc,c->vf",
            kernel,
            block_field,
            weight,
            optimize=True,
        )
        if method in {"ps_real_pi4", "ps_real_pi4_sqrt_r"}:
            stack *= np.exp(-1j * np.pi / 4.0)
        elif method == "fh_real_pi4_r":
            stack *= np.exp(1j * np.pi / 4.0)

        if method in {"ps_abs", "fh_abs_r", "pfh_abs_r"}:
            response = np.abs(stack)
        elif method in {"ps_imag", "fh_imag_r", "pfh_imag_r"}:
            response = np.abs(np.imag(stack))
        elif method in {"ps_real_raw", "fh_real_raw_r", "pfh_real_raw_r"}:
            response = np.real(stack)
        else:
            response = np.abs(np.real(stack))
        raw[:, first:stop] = response
    return raw


def _tau_p_response(
    prepared: _PreparedWavefield,
    velocity_mps: np.ndarray,
    dt_s: float,
    *,
    distance_taper_fraction: float,
) -> np.ndarray:
    """Time-domain slant stack followed by a tau-axis Fourier transform."""

    coordinate = prepared.coordinate_m - prepared.coordinate_m.min()
    time = np.arange(prepared.gather.shape[0], dtype=np.float64) * dt_s
    weight = _spatial_taper(coordinate.size, distance_taper_fraction)
    full_frequency = np.fft.rfftfreq(prepared.nfft, dt_s)
    selected = np.searchsorted(full_frequency, prepared.frequency_hz)
    raw = np.empty((velocity_mps.size, prepared.frequency_hz.size), dtype=np.float64)
    for velocity_index, velocity in enumerate(velocity_mps):
        stack = np.zeros(time.size, dtype=np.float64)
        for trace_index, offset in enumerate(coordinate):
            stack += weight[trace_index] * np.interp(
                time + offset / velocity,
                time,
                prepared.gather[:, trace_index],
                left=0.0,
                right=0.0,
            )
        raw[velocity_index] = np.abs(
            np.fft.rfft(stack, n=prepared.nfft)[selected]
        )
    return raw


def _complex_soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    amplitude = np.abs(values)
    scale = np.maximum(0.0, 1.0 - threshold / np.maximum(amplitude, np.finfo(float).eps))
    return scale * values


def _linear_radon_response(
    prepared: _PreparedWavefield,
    velocity_mps: np.ndarray,
    *,
    distance_taper_fraction: float,
    damping_fraction: float,
) -> np.ndarray:
    """Damped least-squares frequency-domain linear Radon inversion.

    This is the conventional L2 inverse counterpart to the sparse L1/FISTA
    high-resolution Radon implementation below.  The dual solve keeps the
    system receiver-sized even when the velocity scan is dense.
    """

    if not np.isfinite(damping_fraction) or damping_fraction < 0:
        raise ValueError("radon_damping_fraction must be finite and non-negative")
    coordinate = prepared.coordinate_m - prepared.coordinate_m.min()
    spatial_weight = _spatial_taper(coordinate.size, distance_taper_fraction)
    inverse_velocity = 1.0 / velocity_mps
    raw = np.zeros((velocity_mps.size, prepared.frequency_hz.size), dtype=np.float64)
    identity = np.eye(coordinate.size, dtype=np.complex128)
    for frequency_index, frequency in enumerate(prepared.frequency_hz):
        omega = 2.0 * np.pi * frequency
        operator = spatial_weight[:, None] * np.exp(
            -1j * omega * coordinate[:, None] * inverse_velocity[None, :]
        )
        operator /= np.maximum(
            np.linalg.norm(operator, axis=0, keepdims=True), np.finfo(float).eps
        )
        datum = prepared.spectrum[frequency_index] * spatial_weight
        gram = operator @ operator.conj().T
        scale = float(np.trace(gram).real / coordinate.size)
        system = gram + damping_fraction * max(scale, np.finfo(float).eps) * identity
        if damping_fraction == 0:
            dual = np.linalg.lstsq(system, datum, rcond=None)[0]
        else:
            dual = np.linalg.solve(system, datum)
        raw[:, frequency_index] = np.abs(operator.conj().T @ dual)
    return raw


def _adaptive_beamforming_response(
    prepared: _PreparedWavefield,
    velocity_mps: np.ndarray,
    *,
    method: Literal["capon", "music"],
    distance_taper_fraction: float,
    diagonal_loading_fraction: float,
    frequency_smoothing_bins: int,
    music_signal_count: int,
) -> np.ndarray:
    """Capon/MVDR or MUSIC imaging using locally smoothed spatial covariance."""

    if not np.isfinite(diagonal_loading_fraction) or diagonal_loading_fraction <= 0:
        raise ValueError(
            "adaptive_diagonal_loading_fraction must be finite and positive"
        )
    if frequency_smoothing_bins < 0:
        raise ValueError("adaptive_frequency_smoothing_bins must be non-negative")
    receiver_count = prepared.coordinate_m.size
    if not 1 <= music_signal_count < receiver_count:
        raise ValueError("music_signal_count must lie in [1, receiver_count)")

    coordinate = prepared.coordinate_m - prepared.coordinate_m.min()
    spatial_weight = _spatial_taper(receiver_count, distance_taper_fraction)
    inverse_velocity = 1.0 / velocity_mps
    raw = np.zeros((velocity_mps.size, prepared.frequency_hz.size), dtype=np.float64)
    identity = np.eye(receiver_count, dtype=np.complex128)
    for frequency_index, frequency in enumerate(prepared.frequency_hz):
        first = max(0, frequency_index - frequency_smoothing_bins)
        stop = min(
            prepared.frequency_hz.size,
            frequency_index + frequency_smoothing_bins + 1,
        )
        snapshots = prepared.spectrum[first:stop] * spatial_weight[None, :]
        snapshot_norm = np.linalg.norm(snapshots, axis=1, keepdims=True)
        snapshots = snapshots / np.maximum(snapshot_norm, np.finfo(float).eps)
        covariance = snapshots.T @ snapshots.conj() / max(1, snapshots.shape[0])
        covariance_scale = float(np.trace(covariance).real / receiver_count)
        covariance += (
            diagonal_loading_fraction
            * max(covariance_scale, np.finfo(float).eps)
            * identity
        )
        steering = np.exp(
            -1j
            * 2.0
            * np.pi
            * frequency
            * coordinate[:, None]
            * inverse_velocity[None, :]
        )
        steering *= spatial_weight[:, None]
        steering /= np.maximum(
            np.linalg.norm(steering, axis=0, keepdims=True), np.finfo(float).eps
        )
        if method == "capon":
            solved = np.linalg.solve(covariance, steering)
            denominator = np.real(np.sum(steering.conj() * solved, axis=0))
        else:
            _, eigenvectors = np.linalg.eigh(covariance)
            noise = eigenvectors[:, : receiver_count - music_signal_count]
            projection = noise.conj().T @ steering
            denominator = np.sum(np.abs(projection) ** 2, axis=0)
        raw[:, frequency_index] = 1.0 / np.sqrt(
            np.maximum(denominator, np.finfo(float).eps)
        )
    return raw


def _high_resolution_radon_response(
    prepared: _PreparedWavefield,
    velocity_mps: np.ndarray,
    *,
    distance_taper_fraction: float,
    regularization_fraction: float,
    iterations: int,
) -> np.ndarray:
    """Sparse frequency-domain linear Radon inversion using complex FISTA."""

    if not 0 <= regularization_fraction < 1:
        raise ValueError("radon_regularization_fraction must lie in [0, 1)")
    if iterations < 1:
        raise ValueError("radon_iterations must be positive")
    coordinate = prepared.coordinate_m - prepared.coordinate_m.min()
    spatial_weight = _spatial_taper(coordinate.size, distance_taper_fraction)
    inverse_velocity = 1.0 / velocity_mps
    raw = np.zeros((velocity_mps.size, prepared.frequency_hz.size), dtype=np.float64)
    for frequency_index, frequency in enumerate(prepared.frequency_hz):
        omega = 2.0 * np.pi * frequency
        operator = spatial_weight[:, None] * np.exp(
            -1j * omega * coordinate[:, None] * inverse_velocity[None, :]
        )
        operator /= np.maximum(
            np.linalg.norm(operator, axis=0, keepdims=True), np.finfo(float).eps
        )
        datum = prepared.spectrum[frequency_index] * spatial_weight
        datum_norm = float(np.linalg.norm(datum))
        if datum_norm <= np.finfo(float).eps:
            continue
        datum = datum / datum_norm
        # The non-zero eigenvalues of A^H A and A A^H are identical; the
        # latter is much smaller for the usual many-velocity scan.
        lipschitz = float(
            np.linalg.eigvalsh(operator @ operator.conj().T).max(initial=0.0)
        )
        if lipschitz <= np.finfo(float).eps:
            continue
        adjoint_datum = operator.conj().T @ datum
        regularization = regularization_fraction * float(
            np.abs(adjoint_datum).max(initial=0.0)
        )
        model = np.zeros(velocity_mps.size, dtype=np.complex128)
        extrapolated = model.copy()
        momentum = 1.0
        step = 1.0 / lipschitz
        for _ in range(iterations):
            gradient = operator.conj().T @ (operator @ extrapolated - datum)
            candidate = _complex_soft_threshold(
                extrapolated - step * gradient, step * regularization
            )
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
            extrapolated = candidate + (
                (momentum - 1.0) / next_momentum
            ) * (candidate - model)
            model = candidate
            momentum = next_momentum
        raw[:, frequency_index] = np.abs(model)
    return raw


def wavefield_dispersion(
    data: np.ndarray,
    coordinate_m: np.ndarray,
    dt_s: float,
    velocity_mps: np.ndarray,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float,
    method: str,
    component: Literal["vertical", "radial", "transverse"] = "vertical",
    zero_padding: float = 4.0,
    time_taper_fraction: float = 0.1,
    distance_taper_fraction: float = 0.0,
    radial_weighting: Literal["area", "uniform"] = "area",
    frequency_block_size: int = 16,
    radon_damping_fraction: float = 0.05,
    radon_regularization_fraction: float = 0.05,
    radon_iterations: int = 40,
    adaptive_diagonal_loading_fraction: float = 0.05,
    adaptive_frequency_smoothing_bins: int = 2,
    music_signal_count: int = 1,
) -> ArrayDispersionImage:
    """Image one time gather with a compatible radial or line-array method."""

    canonical = canonical_array_method(method)
    if canonical in {"spac", "espac"}:
        raise ValueError("SPAC/ESAC require spac_dispersion and station-pair coherency")
    velocity = _validate_velocity(velocity_mps)
    radial = canonical in {
        "phase_shift",
        "frequency_bessel",
        "frequency_hankel",
    } or canonical in _MATLAB_KERNEL_METHOD_SET
    prepared = _prepare_wavefield(
        data,
        coordinate_m,
        dt_s,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
        zero_padding=zero_padding,
        time_taper_fraction=time_taper_fraction,
        radial=radial,
    )
    if canonical == "phase_shift":
        raw = _phase_or_fk_response(
            prepared,
            velocity,
            phase_only=True,
            distance_taper_fraction=distance_taper_fraction,
            frequency_block_size=frequency_block_size,
        )
    elif canonical == "fk":
        raw = _phase_or_fk_response(
            prepared,
            velocity,
            phase_only=False,
            distance_taper_fraction=distance_taper_fraction,
            frequency_block_size=frequency_block_size,
        )
    elif canonical in {"frequency_bessel", "frequency_hankel"}:
        raw = _radial_transform_response(
            prepared,
            velocity,
            hankel=canonical == "frequency_hankel",
            component=component,
            radial_weighting=radial_weighting,
            distance_taper_fraction=distance_taper_fraction,
            frequency_block_size=frequency_block_size,
        )
    elif canonical in _MATLAB_KERNEL_METHOD_SET:
        raw = _matlab_kernel_variant_response(
            prepared,
            velocity,
            method=canonical,
            distance_taper_fraction=distance_taper_fraction,
            frequency_block_size=frequency_block_size,
        )
    elif canonical == "tau_p":
        raw = _tau_p_response(
            prepared,
            velocity,
            dt_s,
            distance_taper_fraction=distance_taper_fraction,
        )
    elif canonical == "linear_radon":
        raw = _linear_radon_response(
            prepared,
            velocity,
            distance_taper_fraction=distance_taper_fraction,
            damping_fraction=radon_damping_fraction,
        )
    elif canonical == "high_resolution_radon":
        raw = _high_resolution_radon_response(
            prepared,
            velocity,
            distance_taper_fraction=distance_taper_fraction,
            regularization_fraction=radon_regularization_fraction,
            iterations=radon_iterations,
        )
    else:
        raw = _adaptive_beamforming_response(
            prepared,
            velocity,
            method=canonical,
            distance_taper_fraction=distance_taper_fraction,
            diagonal_loading_fraction=adaptive_diagonal_loading_fraction,
            frequency_smoothing_bins=adaptive_frequency_smoothing_bins,
            music_signal_count=music_signal_count,
        )
    contract: InputContract = (
        "radial_time_gather" if radial else "linear_time_gather"
    )
    return ArrayDispersionImage(
        energy=_normalize_columns(np.abs(raw)),
        raw_amplitude=raw,
        frequency_hz=prepared.frequency_hz,
        velocity_mps=velocity,
        method=canonical,
        input_contract=contract,
    )


def estimate_spatial_coherency(
    data: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    dt_s: float,
    *,
    window_s: float,
    overlap: float = 0.0,
    frequency_min_hz: float,
    frequency_max_hz: float,
    zero_padding: float = 1.0,
    time_taper_fraction: float = 0.1,
) -> SpatialCoherency:
    """Estimate pair-resolved complex coherency from finite passive records."""

    traces = np.asarray(data, dtype=np.float64)
    receiver_x = np.asarray(receiver_x_m, dtype=np.float64).reshape(-1)
    receiver_y = np.asarray(receiver_y_m, dtype=np.float64).reshape(-1)
    if (
        traces.ndim != 2
        or traces.shape[1] != receiver_x.size
        or receiver_x.shape != receiver_y.shape
    ):
        raise ValueError("data must be time-by-receiver and match x/y coordinates")
    if (
        traces.shape[0] < 4
        or receiver_x.size < 2
        or np.any(~np.isfinite(traces))
        or np.any(~np.isfinite(receiver_x))
        or np.any(~np.isfinite(receiver_y))
    ):
        raise ValueError("finite records from at least two receivers are required")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be finite and positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if zero_padding < 1:
        raise ValueError("zero_padding must be >= 1")
    if not 0 <= time_taper_fraction <= 1:
        raise ValueError("time_taper_fraction must lie in [0, 1]")

    window_samples = min(
        traces.shape[0], max(4, int(round(window_s / dt_s)))
    )
    step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    window_count = max(
        1, (traces.shape[0] - window_samples) // step_samples + 1
    )
    nfft = _power_of_two_at_least(int(np.ceil(window_samples * zero_padding)))
    frequency = np.fft.rfftfreq(nfft, dt_s)
    selected = (
        (frequency > 0)
        & (frequency >= frequency_min_hz)
        & (frequency <= min(frequency_max_hz, 0.5 / dt_s))
    )
    if np.count_nonzero(selected) < 2:
        raise ValueError("selected frequency band contains fewer than two FFT samples")
    pair_i, pair_j = np.triu_indices(receiver_x.size, k=1)
    pair_distance = np.hypot(
        receiver_x[pair_j] - receiver_x[pair_i],
        receiver_y[pair_j] - receiver_y[pair_i],
    )
    keep = pair_distance > 0
    pair_i = pair_i[keep]
    pair_j = pair_j[keep]
    pair_distance = pair_distance[keep]
    if pair_distance.size == 0:
        raise ValueError("station pairs require positive separation")

    cross_sum = np.zeros((frequency.size, pair_distance.size), dtype=np.complex128)
    power_sum = np.zeros((frequency.size, receiver_x.size), dtype=np.float64)
    taper = tukey(window_samples, time_taper_fraction).reshape(-1, 1)
    valid_windows = 0
    for window_index in range(window_count):
        first = window_index * step_samples
        segment = traces[first : first + window_samples].copy()
        segment = detrend(segment, axis=0, type="linear") * taper
        if np.any(np.std(segment, axis=0) <= np.finfo(float).eps):
            continue
        spectrum = np.fft.rfft(segment, n=nfft, axis=0)
        power_sum += np.abs(spectrum) ** 2
        cross_sum += spectrum[:, pair_j] * np.conj(spectrum[:, pair_i])
        valid_windows += 1
    if valid_windows == 0:
        raise ValueError("no non-degenerate windows were available")
    denominator = np.sqrt(power_sum[:, pair_i] * power_sum[:, pair_j])
    coherency = cross_sum / np.maximum(denominator, np.finfo(float).eps)
    selected_coherency = coherency[selected]
    weight = np.clip(np.abs(selected_coherency) ** 2, 0.01, 1.0)
    return SpatialCoherency(
        frequency_hz=frequency[selected],
        pair_distance_m=pair_distance,
        coherency=selected_coherency,
        coefficient=np.real(selected_coherency),
        weight=weight,
        pair_i=pair_i,
        pair_j=pair_j,
        valid_windows=valid_windows,
    )


def bin_spatial_coherency(
    coherency: SpatialCoherency, distance_bin_m: float
) -> SpatialCoherency:
    """Azimuthally average pair coherency into conventional SPAC rings."""

    if not np.isfinite(distance_bin_m) or distance_bin_m <= 0:
        raise ValueError("distance_bin_m must be finite and positive")
    raw_bin = np.rint(coherency.pair_distance_m / distance_bin_m).astype(np.int64)
    keep = raw_bin > 0
    unique_bin, compact = np.unique(raw_bin[keep], return_inverse=True)
    if unique_bin.size == 0:
        raise ValueError("distance_bin_m is too large for every station pair")
    output = np.empty(
        (coherency.frequency_hz.size, unique_bin.size), dtype=np.complex128
    )
    output_weight = np.empty(output.shape, dtype=np.float64)
    input_coherency = coherency.coherency[:, keep]
    input_weight = coherency.weight[:, keep]
    for bin_index in range(unique_bin.size):
        selected = compact == bin_index
        weight = input_weight[:, selected]
        output[:, bin_index] = np.sum(
            weight * input_coherency[:, selected], axis=1
        ) / np.maximum(np.sum(weight, axis=1), np.finfo(float).eps)
        output_weight[:, bin_index] = np.sum(weight, axis=1)
    output_weight /= np.maximum(
        output_weight.max(axis=1, keepdims=True), np.finfo(float).eps
    )
    return SpatialCoherency(
        frequency_hz=coherency.frequency_hz.copy(),
        pair_distance_m=unique_bin.astype(np.float64) * distance_bin_m,
        coherency=output,
        coefficient=np.real(output),
        weight=np.clip(output_weight, 0.01, 1.0),
        pair_i=np.full(unique_bin.size, -1, dtype=np.int64),
        pair_j=np.full(unique_bin.size, -1, dtype=np.int64),
        valid_windows=coherency.valid_windows,
    )


def spac_dispersion(
    coherency: SpatialCoherency,
    velocity_mps: np.ndarray,
    *,
    method: str = "espac",
    distance_bin_m: float | None = None,
    misfit_floor: float = 1e-4,
) -> ArrayDispersionImage:
    """Fit the SPAC ``J0`` relation on rings (SPAC) or arbitrary pairs (ESAC)."""

    canonical = canonical_array_method(method)
    if canonical not in {"spac", "espac"}:
        raise ValueError("spac_dispersion method must be spac or espac/esac")
    velocity = _validate_velocity(velocity_mps)
    if not np.isfinite(misfit_floor) or misfit_floor <= 0:
        raise ValueError("misfit_floor must be finite and positive")
    observation = coherency
    if canonical == "spac":
        if distance_bin_m is None:
            unique_distance = np.unique(np.round(coherency.pair_distance_m, 9))
            positive_step = np.diff(unique_distance)
            positive_step = positive_step[positive_step > 0]
            if positive_step.size == 0:
                raise ValueError("SPAC requires more than one radial ring")
            distance_bin_m = float(positive_step.min())
        observation = bin_spatial_coherency(coherency, distance_bin_m)
    distance = observation.pair_distance_m
    if distance.size < 2:
        raise ValueError("SPAC/ESAC require at least two pair distances")
    raw = np.empty((velocity.size, observation.frequency_hz.size), dtype=np.float64)
    inverse_velocity = 1.0 / velocity
    for frequency_index, frequency in enumerate(observation.frequency_hz):
        predicted = jv(
            0,
            2.0
            * np.pi
            * frequency
            * inverse_velocity[:, None]
            * distance[None, :],
        )
        residual = predicted - observation.coefficient[frequency_index][None, :]
        weight = observation.weight[frequency_index][None, :]
        misfit = np.sum(weight * residual**2, axis=1) / np.maximum(
            np.sum(weight), np.finfo(float).eps
        )
        raw[:, frequency_index] = 1.0 / (misfit + misfit_floor)
    return ArrayDispersionImage(
        energy=_normalize_columns(raw),
        raw_amplitude=raw,
        frequency_hz=observation.frequency_hz.copy(),
        velocity_mps=velocity,
        method=canonical,
        input_contract="station_pair_coherency",
    )


__all__ = [
    "ArrayDispersionImage",
    "ArrayMethod",
    "InputContract",
    "MATLAB_KERNEL_METHODS",
    "SPACMethod",
    "SpatialCoherency",
    "WavefieldMethod",
    "bin_spatial_coherency",
    "canonical_array_method",
    "estimate_spatial_coherency",
    "method_input_contract",
    "spac_dispersion",
    "wavefield_dispersion",
]
