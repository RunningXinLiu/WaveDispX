"""Two-station frequency-time analysis (FTAN).

The implementation follows the physical definitions used by the Huajian Yao
``EGFAnalysisTimeFreq`` MATLAB program:

* a Gaussian narrow-band filter bank is used for group-velocity FTAN;
* the envelope maximum is mapped from time to ``U = distance / time``;
* phase velocity is mapped from the signed narrow-band waveform using the
  far-field ``pi/4`` correction, ``time = distance / c + period / 8``;
* a correlation function can first be converted to an EGF-like trace with
  ``imag(hilbert(correlation))``.

This module deliberately accepts one positive-time branch from one receiver
pair.  Splitting, orienting, and stacking causal/acausal NCF branches remains
the responsibility of the NCF workflow so that branch provenance is not lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import hilbert

from .picking import pick_dispersion_ridge


_ALPHA_DISTANCE_KM = np.array(
    [0.0, 100.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 20_000.0]
)
_ALPHA_VALUE = np.array([5.0, 8.0, 12.0, 20.0, 25.0, 35.0, 50.0, 75.0])


@dataclass(frozen=True)
class FTANImage:
    """A velocity-by-period FTAN image.

    ``amplitude`` is non-negative and normalized independently at each period.
    For a phase image, ``raw_amplitude`` retains the signed narrow-band samples
    while ``amplitude`` retains only positive crests for automatic picking.
    """

    period_s: np.ndarray
    velocity_mps: np.ndarray
    amplitude: np.ndarray
    raw_amplitude: np.ndarray
    kind: str


@dataclass(frozen=True)
class FTANCurve:
    """A full-period FTAN ridge with explicit validity information."""

    period_s: np.ndarray
    velocity_mps: np.ndarray
    normalized_amplitude: np.ndarray
    half_power_uncertainty_mps: np.ndarray
    snr: np.ndarray
    valid_mask: np.ndarray

    def as_array(self, *, valid_only: bool = False) -> np.ndarray:
        keep = self.valid_mask if valid_only else np.ones(self.period_s.size, dtype=bool)
        return np.column_stack(
            (
                self.period_s[keep],
                self.velocity_mps[keep],
                self.normalized_amplitude[keep],
                self.half_power_uncertainty_mps[keep],
                self.snr[keep],
                self.valid_mask[keep].astype(np.float64),
            )
        )


@dataclass(frozen=True)
class PairFTANResult:
    """Group- and phase-velocity FTAN products for one receiver pair."""

    time_s: np.ndarray
    processed_trace: np.ndarray
    signal_window: np.ndarray
    group_image: FTANImage
    phase_image: FTANImage
    group_curve: FTANCurve
    phase_curve: FTANCurve
    gaussian_alpha: float
    input_kind: str
    phase_method: str


def correlation_to_egf(correlation: np.ndarray) -> np.ndarray:
    """Convert a real correlation branch to an EGF-like trace.

    The sign convention matches ``EGFAnalysisTimeFreq.m``.  Callers must orient
    causal and acausal branches consistently before invoking this function.
    """

    values = _finite_trace(correlation)
    return np.imag(hilbert(values))


def phase_to_group_velocity(
    period_s: np.ndarray,
    phase_velocity_mps: np.ndarray,
) -> np.ndarray:
    """Convert a phase-velocity curve to group velocity through ``domega/dk``."""

    period = np.asarray(period_s, dtype=np.float64).reshape(-1)
    phase = np.asarray(phase_velocity_mps, dtype=np.float64).reshape(-1)
    if period.size != phase.size or period.size < 3:
        raise ValueError("period_s and phase_velocity_mps must have the same length >= 3")
    if not np.all(np.isfinite(period)) or not np.all(np.isfinite(phase)):
        raise ValueError("period_s and phase_velocity_mps must be finite")
    if np.any(period <= 0.0) or np.any(phase <= 0.0):
        raise ValueError("periods and phase velocities must be positive")
    if np.any(np.diff(period) <= 0.0):
        raise ValueError("period_s must be strictly increasing")
    omega = 2.0 * np.pi / period
    wavenumber = omega / phase
    dk_domega = np.gradient(wavenumber, omega, edge_order=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        group = 1.0 / dk_domega
    group[~np.isfinite(group) | (group <= 0.0)] = np.nan
    return group


def pair_ftan(
    trace: np.ndarray,
    *,
    dt_s: float,
    distance_m: float,
    frequency_range_hz: tuple[float, float],
    group_velocity_range_mps: tuple[float, float],
    phase_velocity_range_mps: tuple[float, float],
    periods_s: np.ndarray | None = None,
    period_count: int = 96,
    velocity_step_mps: float = 2.0,
    input_kind: str = "egf",
    phase_method: str = "time_domain",
    gaussian_alpha: float | None = None,
    signal_taper_periods: float = 1.0,
    group_window_periods: float = 6.0,
    minimum_group_window_s: float = 0.4,
    phase_minus_group_range_mps: tuple[float, float] | None = None,
    phase_velocity_anchor: tuple[float, float] | None = None,
    phase_group_tolerance_mps: float | None | Literal["auto"] = "auto",
    minimum_snr: float = 2.0,
    group_picker_smoothness: float = 2.0,
    phase_picker_smoothness: float = 2.0,
    maximum_jump_mps: float = 40.0,
    phase_maximum_jump_mps: float | None = None,
    minimum_pick_amplitude: float = 0.05,
    minimum_phase_pick_amplitude: float = 0.0,
) -> PairFTANResult:
    """Compute group and phase FTAN for one positive-time receiver-pair trace.

    Parameters are in SI units.  ``phase_minus_group_range_mps`` is an optional
    fundamental-mode prior on ``c - U``.  It is disabled by default because it
    is not valid for every mode or medium; when used, it affects the phase
    image and therefore must be reported with the result.
    """

    waveform = _finite_trace(trace)
    if waveform.size < 32:
        raise ValueError("trace must contain at least 32 samples")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("distance_m must be positive and finite")
    frequency_min, frequency_max = _positive_range(
        frequency_range_hz, "frequency_range_hz"
    )
    if frequency_max >= 0.5 / dt_s:
        raise ValueError("frequency_range_hz must lie below Nyquist")
    group_min, group_max = _positive_range(
        group_velocity_range_mps, "group_velocity_range_mps"
    )
    phase_min, phase_max = _positive_range(
        phase_velocity_range_mps, "phase_velocity_range_mps"
    )
    if not np.isfinite(velocity_step_mps) or velocity_step_mps <= 0.0:
        raise ValueError("velocity_step_mps must be positive and finite")
    if period_count < 3:
        raise ValueError("period_count must be at least 3")
    if input_kind not in {"egf", "correlation"}:
        raise ValueError("input_kind must be 'egf' or 'correlation'")
    if phase_method not in {"time_domain", "frequency_domain"}:
        raise ValueError("phase_method must be 'time_domain' or 'frequency_domain'")
    if phase_group_tolerance_mps not in {None, "auto"} and (
        not isinstance(phase_group_tolerance_mps, (int, float))
        or not np.isfinite(phase_group_tolerance_mps)
        or phase_group_tolerance_mps <= 0.0
    ):
        raise ValueError(
            "phase_group_tolerance_mps must be 'auto', None, or a positive finite value"
        )
    if minimum_snr < 0.0:
        raise ValueError("minimum_snr must be non-negative")
    if phase_maximum_jump_mps is not None and (
        not np.isfinite(phase_maximum_jump_mps)
        or phase_maximum_jump_mps <= 0.0
    ):
        raise ValueError("phase_maximum_jump_mps must be positive and finite")

    if periods_s is None:
        periods = np.linspace(1.0 / frequency_max, 1.0 / frequency_min, period_count)
    else:
        periods = np.asarray(periods_s, dtype=np.float64).reshape(-1)
        if periods.size < 3 or not np.all(np.isfinite(periods)):
            raise ValueError("periods_s must contain at least three finite values")
        if np.any(np.diff(periods) <= 0.0):
            raise ValueError("periods_s must be strictly increasing")
        if periods[0] < 1.0 / frequency_max or periods[-1] > 1.0 / frequency_min:
            raise ValueError("periods_s must lie inside frequency_range_hz")

    processed = correlation_to_egf(waveform) if input_kind == "correlation" else waveform.copy()
    processed = processed - np.mean(processed)
    time = np.arange(processed.size, dtype=np.float64) * dt_s
    signal_start_s = distance_m / group_max
    signal_end_s = distance_m / group_min
    if signal_start_s >= time[-1]:
        raise ValueError("trace ends before the fastest requested group arrival")
    signal_end_s = min(signal_end_s, time[-1])
    if signal_end_s <= signal_start_s:
        raise ValueError("trace does not cover the requested group-velocity window")
    taper_s = signal_taper_periods * float(np.max(periods))
    taper_s = min(taper_s, 0.25 * (signal_end_s - signal_start_s))
    # Keep the complete requested arrival interval at unit weight.  The taper
    # lies outside it; tapering inside the interval biases the fastest/slowest
    # group arrivals toward the middle of the search range.
    signal_window = _cosine_boxcar(
        time,
        signal_start_s - taper_s,
        signal_end_s + taper_s,
        taper_s,
    )

    alpha = (
        float(np.interp(distance_m / 1_000.0, _ALPHA_DISTANCE_KM, _ALPHA_VALUE))
        if gaussian_alpha is None
        else float(gaussian_alpha)
    )
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("gaussian_alpha must be positive and finite")

    group_velocity = _regular_axis(group_min, group_max, velocity_step_mps)
    phase_velocity = _regular_axis(phase_min, phase_max, velocity_step_mps)
    nfft = _next_power_of_two(2 * processed.size)
    filtered_signal = _gaussian_filter_bank(
        processed * signal_window, dt_s, periods, alpha, nfft
    )
    envelope = np.abs(hilbert(filtered_signal, axis=1))
    group_raw = _time_to_velocity(envelope, time, distance_m, group_velocity)
    group_amplitude = _normalize_columns(group_raw)

    filtered_full = _gaussian_filter_bank(processed, dt_s, periods, alpha, nfft)
    full_envelope = np.abs(hilbert(filtered_full, axis=1))
    snr = _period_snr(
        envelope,
        full_envelope,
        time,
        signal_start_s,
        signal_end_s,
        taper_s,
    )
    group_curve = _pick_ftan_curve(
        group_amplitude,
        periods,
        group_velocity,
        snr,
        minimum_snr=minimum_snr,
        smoothness=group_picker_smoothness,
        maximum_jump_mps=maximum_jump_mps,
        minimum_amplitude=minimum_pick_amplitude,
    )

    group_reference = _fill_curve(group_curve.velocity_mps)
    if phase_method == "time_domain":
        phase_signed = np.zeros((phase_velocity.size, periods.size), dtype=np.float64)
        for period_index, period in enumerate(periods):
            center_s = distance_m / group_reference[period_index]
            width_s = max(group_window_periods * period, minimum_group_window_s)
            local_window = _cosine_boxcar(
                time,
                center_s - 0.5 * width_s,
                center_s + 0.5 * width_s,
                min(period, 0.25 * width_s),
            )
            filtered = _gaussian_filter_bank(
                processed * signal_window * local_window,
                dt_s,
                periods[period_index : period_index + 1],
                alpha,
                nfft,
            )[0]
            phase_time = distance_m / phase_velocity + period / 8.0
            phase_signed[:, period_index] = np.interp(
                phase_time, time, filtered, left=0.0, right=0.0
            )
        phase_signed = _signed_column_scale(phase_signed)
    else:
        phase_signed = _frequency_domain_phase_image(
            processed * signal_window,
            dt_s,
            periods,
            distance_m,
            phase_velocity,
            nfft,
        )
    phase_amplitude = _normalize_columns(np.clip(phase_signed, 0.0, None))
    if phase_minus_group_range_mps is not None:
        difference_min, difference_max = phase_minus_group_range_mps
        if not np.isfinite(difference_min) or not np.isfinite(difference_max):
            raise ValueError("phase_minus_group_range_mps must be finite")
        if difference_max <= difference_min:
            raise ValueError("phase_minus_group_range_mps must be increasing")
        difference = phase_velocity[:, None] - group_reference[None, :]
        phase_amplitude = np.where(
            (difference >= difference_min) & (difference <= difference_max),
            phase_amplitude,
            0.0,
        )
    effective_phase_group_tolerance = phase_group_tolerance_mps
    if effective_phase_group_tolerance == "auto":
        effective_phase_group_tolerance = 12.0 if phase_method == "time_domain" else 100.0
    phase_curve = _pick_phase_curve(
        phase_amplitude,
        periods,
        phase_velocity,
        snr,
        group_reference,
        anchor=phase_velocity_anchor,
        group_tolerance_mps=effective_phase_group_tolerance,
        minimum_snr=minimum_snr,
        smoothness=phase_picker_smoothness,
        maximum_jump_mps=(
            maximum_jump_mps
            if phase_maximum_jump_mps is None
            else phase_maximum_jump_mps
        ),
        minimum_amplitude=minimum_phase_pick_amplitude,
    )

    return PairFTANResult(
        time_s=time,
        processed_trace=processed,
        signal_window=signal_window,
        group_image=FTANImage(
            period_s=periods,
            velocity_mps=group_velocity,
            amplitude=group_amplitude,
            raw_amplitude=group_raw,
            kind="group",
        ),
        phase_image=FTANImage(
            period_s=periods,
            velocity_mps=phase_velocity,
            amplitude=phase_amplitude,
            raw_amplitude=phase_signed,
            kind="phase",
        ),
        group_curve=group_curve,
        phase_curve=phase_curve,
        gaussian_alpha=alpha,
        input_kind=input_kind,
        phase_method=phase_method,
    )


def _finite_trace(trace: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("trace must be a non-empty finite one-dimensional array")
    return values


def _positive_range(values: tuple[float, float], name: str) -> tuple[float, float]:
    lower, upper = (float(values[0]), float(values[1]))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise ValueError(f"{name} must be a positive increasing pair")
    return lower, upper


def _regular_axis(lower: float, upper: float, step: float) -> np.ndarray:
    count = int(np.floor((upper - lower) / step + 0.5))
    axis = lower + step * np.arange(count + 1, dtype=np.float64)
    if axis[-1] < upper - 0.25 * step:
        axis = np.append(axis, upper)
    else:
        axis[-1] = upper
    return axis


def _next_power_of_two(value: int) -> int:
    return 1 << int(np.ceil(np.log2(max(2, value))))


def _cosine_boxcar(
    time_s: np.ndarray,
    start_s: float,
    end_s: float,
    taper_s: float,
) -> np.ndarray:
    start = max(float(start_s), float(time_s[0]))
    end = min(float(end_s), float(time_s[-1]))
    window = np.zeros(time_s.size, dtype=np.float64)
    if end <= start:
        return window
    inside = (time_s >= start) & (time_s <= end)
    window[inside] = 1.0
    taper = min(max(float(taper_s), 0.0), 0.5 * (end - start))
    if taper > 0.0:
        leading = (time_s >= start) & (time_s < start + taper)
        trailing = (time_s > end - taper) & (time_s <= end)
        window[leading] = 0.5 * (
            1.0 - np.cos(np.pi * (time_s[leading] - start) / taper)
        )
        window[trailing] = 0.5 * (
            1.0 - np.cos(np.pi * (end - time_s[trailing]) / taper)
        )
    return window


def _gaussian_filter_bank(
    trace: np.ndarray,
    dt_s: float,
    periods_s: np.ndarray,
    alpha: float,
    nfft: int,
) -> np.ndarray:
    spectrum = np.fft.rfft(trace, n=nfft)
    frequency = np.fft.rfftfreq(nfft, dt_s)
    center = 1.0 / periods_s
    response = np.exp(
        -alpha * ((frequency[None, :] - center[:, None]) / center[:, None]) ** 2
    )
    filtered = np.fft.irfft(response * spectrum[None, :], n=nfft, axis=1)
    return filtered[:, : trace.size]


def _time_to_velocity(
    period_by_time: np.ndarray,
    time_s: np.ndarray,
    distance_m: float,
    velocity_mps: np.ndarray,
) -> np.ndarray:
    image = np.zeros((velocity_mps.size, period_by_time.shape[0]), dtype=np.float64)
    travel_time = distance_m / velocity_mps
    for period_index, values in enumerate(period_by_time):
        image[:, period_index] = np.interp(
            travel_time, time_s, values, left=0.0, right=0.0
        )
    return image


def _frequency_domain_phase_image(
    trace: np.ndarray,
    dt_s: float,
    periods_s: np.ndarray,
    distance_m: float,
    velocity_mps: np.ndarray,
    nfft: int,
) -> np.ndarray:
    """MATLAB-compatible frequency-domain far-field phase coherence image."""

    spectrum = np.fft.rfft(trace, n=nfft)
    frequency_axis = np.fft.rfftfreq(nfft, dt_s)
    center_frequency = 1.0 / periods_s
    interpolated = np.interp(center_frequency, frequency_axis, spectrum.real) + 1j * np.interp(
        center_frequency, frequency_axis, spectrum.imag
    )
    observed_phase = np.angle(interpolated)
    kr = (
        2.0
        * np.pi
        * center_frequency[None, :]
        * distance_m
        / velocity_mps[:, None]
    )
    expected_phase = -(kr + np.pi / 4.0)
    return np.cos(expected_phase - observed_phase[None, :])


def _normalize_columns(image: np.ndarray) -> np.ndarray:
    maximum = np.max(image, axis=0, keepdims=True)
    return np.divide(image, maximum, out=np.zeros_like(image), where=maximum > 0.0)


def _signed_column_scale(image: np.ndarray) -> np.ndarray:
    maximum = np.max(np.abs(image), axis=0, keepdims=True)
    return np.divide(image, maximum, out=np.zeros_like(image), where=maximum > 0.0)


def _period_snr(
    signal_envelope: np.ndarray,
    full_envelope: np.ndarray,
    time_s: np.ndarray,
    signal_start_s: float,
    signal_end_s: float,
    taper_s: float,
) -> np.ndarray:
    signal = (time_s >= signal_start_s) & (time_s <= signal_end_s)
    noise_start = signal_end_s + max(taper_s, 0.0)
    noise = time_s >= noise_start
    # Avoid a wrapped negative-time branch in FFT-generated synthetic traces.
    if np.count_nonzero(noise) >= 16:
        noise_index = np.flatnonzero(noise)
        noise[noise_index[noise_index.size // 2 :]] = False
    if np.count_nonzero(noise) < 16:
        noise = time_s < max(0.0, signal_start_s - max(taper_s, 0.0))
    if np.count_nonzero(noise) < 16:
        return np.full(signal_envelope.shape[0], np.nan)
    numerator = np.max(signal_envelope[:, signal], axis=1)
    denominator = np.mean(full_envelope[:, noise], axis=1)
    floor = np.finfo(np.float64).eps * np.maximum(
        np.max(full_envelope, axis=1), 1.0
    )
    return numerator / np.maximum(denominator, floor)


def _pick_ftan_curve(
    amplitude: np.ndarray,
    period_s: np.ndarray,
    velocity_mps: np.ndarray,
    snr: np.ndarray,
    *,
    minimum_snr: float,
    smoothness: float,
    maximum_jump_mps: float,
    minimum_amplitude: float,
) -> FTANCurve:
    if not np.isfinite(maximum_jump_mps) or maximum_jump_mps <= 0.0:
        raise ValueError("maximum_jump_mps must be positive and finite")
    dv = np.median(np.diff(velocity_mps)) if velocity_mps.size > 1 else 1.0
    # ``pick_dispersion_ridge`` expresses its penalty in velocity-grid cells.
    # Rescale here so the public FTAN smoothness is tied to the physical maximum
    # jump and is therefore stable when ``velocity_step_mps`` changes.
    grid_smoothness = smoothness * (dv / maximum_jump_mps) ** 2
    picks = pick_dispersion_ridge(
        amplitude,
        period_s,
        velocity_mps,
        smoothness=grid_smoothness,
        maximum_jump_mps=maximum_jump_mps,
        minimum_energy=0.0,
    )
    picked_velocity = np.full(period_s.size, np.nan)
    picked_amplitude = np.full(period_s.size, np.nan)
    picked_uncertainty = np.full(period_s.size, np.nan)
    index = np.searchsorted(period_s, picks.frequency_hz)
    picked_velocity[index] = picks.phase_velocity_mps
    picked_amplitude[index] = picks.normalized_energy
    picked_uncertainty[index] = picks.half_power_uncertainty_mps
    snr_valid = np.isfinite(snr) & (snr >= minimum_snr)
    valid = (
        np.isfinite(picked_velocity)
        & np.isfinite(picked_amplitude)
        & (picked_amplitude > minimum_amplitude)
        & snr_valid
    )
    return FTANCurve(
        period_s=period_s.copy(),
        velocity_mps=picked_velocity,
        normalized_amplitude=picked_amplitude,
        half_power_uncertainty_mps=picked_uncertainty,
        snr=snr.copy(),
        valid_mask=valid,
    )


def _fill_curve(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if not np.any(valid):
        raise ValueError("group FTAN produced no finite ridge")
    index = np.arange(values.size)
    return np.interp(index, index[valid], values[valid])


def _pick_phase_curve(
    amplitude: np.ndarray,
    period_s: np.ndarray,
    velocity_mps: np.ndarray,
    snr: np.ndarray,
    group_reference_mps: np.ndarray,
    *,
    anchor: tuple[float, float] | None,
    group_tolerance_mps: float | None,
    minimum_snr: float,
    smoothness: float,
    maximum_jump_mps: float,
    minimum_amplitude: float,
) -> FTANCurve:
    """Track one phase crest while enforcing phase/group differential physics.

    Group velocity constrains the slope of a phase curve but cannot determine
    its integer cycle.  A phase-velocity anchor therefore selects the absolute
    branch.  Without one, an illustrative ridge is returned but deliberately
    marked invalid rather than presented as an absolute phase-velocity pick.
    """

    if group_tolerance_mps is not None and (
        not np.isfinite(group_tolerance_mps) or group_tolerance_mps <= 0.0
    ):
        raise ValueError("phase_group_tolerance_mps must be positive and finite or None")
    period_count = period_s.size
    velocity_count = velocity_mps.size
    score = np.full((velocity_count, period_count), -np.inf)
    back_pointer = np.zeros((velocity_count, period_count), dtype=np.int32)
    score[:, 0] = np.log(np.maximum(amplitude[:, 0], np.finfo(float).eps))

    anchor_index: int | None = None
    anchor_velocity_index: int | None = None
    if anchor is not None:
        anchor_period, anchor_velocity = float(anchor[0]), float(anchor[1])
        if not np.isfinite(anchor_period) or not np.isfinite(anchor_velocity):
            raise ValueError("phase_velocity_anchor must be finite")
        if not period_s[0] <= anchor_period <= period_s[-1]:
            raise ValueError("phase_velocity_anchor period lies outside the FTAN period range")
        if not velocity_mps[0] <= anchor_velocity <= velocity_mps[-1]:
            raise ValueError("phase_velocity_anchor velocity lies outside the phase range")
        anchor_index = int(np.argmin(np.abs(period_s - anchor_period)))
        anchor_velocity_index = int(np.argmin(np.abs(velocity_mps - anchor_velocity)))
        if anchor_index == 0:
            score[:, 0] = -np.inf
            score[anchor_velocity_index, 0] = np.log(
                max(amplitude[anchor_velocity_index, 0], np.finfo(float).eps)
            )

    previous_velocity = velocity_mps[None, :]
    current_velocity = velocity_mps[:, None]
    jump = current_velocity - previous_velocity
    for period_index in range(1, period_count):
        delta_period = period_s[period_index] - period_s[period_index - 1]
        center_period = 0.5 * (period_s[period_index] + period_s[period_index - 1])
        center_velocity = 0.5 * (current_velocity + previous_velocity)
        derivative = jump / delta_period
        denominator = 1.0 + center_period * derivative / center_velocity
        with np.errstate(divide="ignore", invalid="ignore"):
            implied_group = center_velocity / denominator
        transition = smoothness * (jump / maximum_jump_mps) ** 2
        if group_tolerance_mps is not None:
            target_group = 0.5 * (
                group_reference_mps[period_index]
                + group_reference_mps[period_index - 1]
            )
            transition += ((implied_group - target_group) / group_tolerance_mps) ** 2
        transition[(np.abs(jump) > maximum_jump_mps) | (denominator <= 0.0)] = np.inf
        candidate = score[:, period_index - 1][None, :] - transition
        previous = np.argmax(candidate, axis=1)
        score[:, period_index] = (
            np.log(np.maximum(amplitude[:, period_index], np.finfo(float).eps))
            + candidate[np.arange(velocity_count), previous]
        )
        back_pointer[:, period_index] = previous
        if anchor_index == period_index and anchor_velocity_index is not None:
            keep_score = score[anchor_velocity_index, period_index]
            score[:, period_index] = -np.inf
            score[anchor_velocity_index, period_index] = keep_score

    ridge_index = np.zeros(period_count, dtype=np.int32)
    ridge_index[-1] = int(np.argmax(score[:, -1]))
    for period_index in range(period_count - 1, 0, -1):
        ridge_index[period_index - 1] = back_pointer[
            ridge_index[period_index], period_index
        ]
    column = np.arange(period_count)
    picked_velocity = velocity_mps[ridge_index]
    picked_amplitude = amplitude[ridge_index, column]
    uncertainty = np.zeros(period_count, dtype=np.float64)
    for period_index, center in enumerate(ridge_index):
        above = amplitude[:, period_index] >= 0.5 * picked_amplitude[period_index]
        left = int(center)
        right = int(center)
        while left > 0 and above[left - 1]:
            left -= 1
        while right + 1 < velocity_count and above[right + 1]:
            right += 1
        uncertainty[period_index] = 0.5 * (velocity_mps[right] - velocity_mps[left])
    valid = (
        (anchor is not None)
        & np.isfinite(snr)
        & (snr >= minimum_snr)
        & (picked_amplitude > minimum_amplitude)
    )
    return FTANCurve(
        period_s=period_s.copy(),
        velocity_mps=picked_velocity,
        normalized_amplitude=picked_amplitude,
        half_power_uncertainty_mps=uncertainty,
        snr=snr.copy(),
        valid_mask=np.asarray(valid, dtype=bool),
    )
