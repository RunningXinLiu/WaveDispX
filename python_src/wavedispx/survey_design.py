"""Model-aware frequency, velocity, and record-window recommendations.

The DWM kernel solves the layered system at discrete complex frequencies.  A
finite record is still required whenever those responses are transformed to a
waveform or used to synthesize finite ambient-noise windows.  This module keeps
that numerical period separate from the source band and from the requested
dispersion display window.

Recommendations never overwrite explicit configuration values.  They are used
by preflight diagnostics and by the active ``record_duration_s = 0`` automatic
mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ActiveSourceConfig, ModelConfig, RunConfig
from .noise import source_spectral_amplitude
from .processing.theory import TheoreticalDispersion, theoretical_dispersion


SOURCE_POWER_THRESHOLD = 0.01  # -20 dB relative power
APERTURE_WAVELENGTHS = 2.0
VELOCITY_LOWER_MARGIN = 0.70
VELOCITY_UPPER_MARGIN = 1.30
MINIMUM_PHYSICAL_FREQUENCY_BINS = 20.0


@dataclass(frozen=True)
class SurveyDesign:
    """Recommended observation and imaging ranges for one frozen case."""

    frequency_hz: np.ndarray
    source_power: np.ndarray
    theory: TheoreticalDispersion
    useful_source_min_hz: float
    useful_source_max_hz: float
    recommended_frequency_min_hz: float
    recommended_frequency_max_hz: float
    recommended_frequency_step_hz: float
    recommended_velocity_min_mps: float
    recommended_velocity_max_mps: float
    recommended_velocity_step_mps: float
    minimum_group_velocity_mps: float
    maximum_group_velocity_mps: float
    recommended_active_record_duration_s: float
    recommended_passive_green_duration_s: float
    recommended_passive_window_s: float
    recommended_passive_maximum_lag_s: float


def model_group_velocity_bounds(
    model: ModelConfig,
    frequency_min_hz: float,
    frequency_max_hz: float,
    *,
    rayleigh_modes: int = 4,
    love_modes: int = 0,
) -> tuple[float, float]:
    """Return finite group-velocity bounds over an explicit frequency band."""

    if frequency_min_hz <= 0.0 or frequency_max_hz <= frequency_min_hz:
        raise ValueError("frequency bounds must be positive and ordered")
    frequency = np.linspace(frequency_min_hz, frequency_max_hz, 240)
    theory = theoretical_dispersion(
        model,
        frequency,
        rayleigh_modes=rayleigh_modes,
        love_modes=max(1, love_modes),
        backend="disba",
        compute_group=True,
    )
    arrays = [theory.rayleigh_group_mps]
    if love_modes > 0:
        arrays.append(theory.love_group_mps[:, :love_modes])
    group = np.column_stack(arrays)
    maximum_plausible = 1.5 * max(layer.vp_mps for layer in model.layers)
    finite = group[
        np.isfinite(group) & (group > 0.0) & (group <= maximum_plausible)
    ]
    if finite.size == 0:
        return (
            0.8 * min(layer.vs_mps for layer in model.layers),
            max(layer.vs_mps for layer in model.layers),
        )
    return float(np.min(finite)), float(np.max(finite))


def recommended_active_record_duration(
    model: ModelConfig,
    source: ActiveSourceConfig,
    maximum_source_receiver_distance_m: float,
    *,
    frequency_min_hz: float | None = None,
    frequency_max_hz: float | None = None,
    rayleigh_modes: int = 4,
    love_modes: int = 0,
) -> tuple[float, float, float]:
    """Return recommended duration plus the group-velocity bounds used.

    The returned duration describes the saved record after removal of the DWM
    exponential-window delay.  It includes farthest slow-group arrival, source
    support, and two periods at the lowest imaged frequency.
    """

    if frequency_min_hz is None:
        frequency = np.linspace(
            max(0.02, source.maximum_frequency_hz / 1000.0),
            source.maximum_frequency_hz,
            480,
        )
        power = (
            np.ones_like(frequency)
            if source.wavelet == "unit_impulse"
            else _active_ricker_power(frequency, source.dominant_frequency_hz)
        )
        supported = np.flatnonzero(power >= SOURCE_POWER_THRESHOLD)
        frequency_min_hz = float(
            frequency[supported[0]] if supported.size else source.dominant_frequency_hz / 4.0
        )
    if frequency_max_hz is None:
        frequency_max_hz = source.maximum_frequency_hz
    frequency_max_hz = min(frequency_max_hz, source.maximum_frequency_hz)
    minimum_group, maximum_group = model_group_velocity_bounds(
        model,
        frequency_min_hz,
        frequency_max_hz,
        rayleigh_modes=rayleigh_modes,
        love_modes=love_modes,
    )
    travel_s = maximum_source_receiver_distance_m / minimum_group
    source_duration_s = (
        source.dt_s
        if source.wavelet == "unit_impulse"
        else 4.0 / source.dominant_frequency_hz
    )
    duration_s = (
        1.15 * travel_s + source_duration_s + 2.0 / frequency_min_hz
    )
    duration_s = max(
        duration_s,
        MINIMUM_PHYSICAL_FREQUENCY_BINS
        / (frequency_max_hz - frequency_min_hz),
    )
    return float(duration_s), minimum_group, maximum_group


def _active_ricker_power(frequency_hz: np.ndarray, dominant_hz: float) -> np.ndarray:
    ratio = np.asarray(frequency_hz, dtype=np.float64) / float(dominant_hz)
    # The Ricker amplitude spectrum is proportional to f^2 exp(-(f/f0)^2).
    # Squaring and normalizing at f=f0 gives the expression below.
    return ratio**4 * np.exp(2.0 * (1.0 - ratio**2))


def _finite_row_extrema(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(values.shape[0], np.nan, dtype=np.float64)
    maximum = np.full(values.shape[0], np.nan, dtype=np.float64)
    for index, row in enumerate(values):
        finite = row[np.isfinite(row) & (row > 0.0)]
        if finite.size:
            minimum[index] = float(np.min(finite))
            maximum[index] = float(np.max(finite))
    return minimum, maximum


def _largest_true_run(mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return mask
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, split)
    longest = max(runs, key=lambda item: item.size)
    result = np.zeros_like(mask, dtype=bool)
    result[longest] = True
    return result


def _round_down(value: float, step: float) -> float:
    return float(np.floor(value / step) * step)


def _round_up(value: float, step: float) -> float:
    return float(np.ceil(value / step) * step)


def _nice_step(value: float) -> float:
    """Round a positive sampling interval to a stable 1/2/5 engineering step."""

    value = max(float(value), np.finfo(float).eps)
    exponent = float(10.0 ** np.floor(np.log10(value)))
    scaled = value / exponent
    factor = 1.0 if scaled <= 1.0 else 2.0 if scaled <= 2.0 else 5.0
    return factor * exponent


def design_survey_window(
    config: RunConfig,
    *,
    receiver_aperture_m: float,
    effective_spacing_m: float,
    maximum_source_receiver_distance_m: float,
    maximum_gauge_length_m: float | None = None,
) -> SurveyDesign:
    """Design an auditable f-v window and finite-record requirement.

    The frequency recommendation is the largest contiguous interval that has
    source power above -20 dB, at least two wavelengths of aperture, no simple
    spatial alias under the effective radial sampling, and (for DAS) remains
    below the first finite-gauge zero for the slowest retained mode.
    """

    if config.mode == "active":
        source = config.active_source
        lower = max(0.02, source.maximum_frequency_hz / 1000.0)
        upper = source.maximum_frequency_hz
        frequency = np.linspace(lower, upper, 480)
        if source.wavelet == "unit_impulse":
            source_power = np.ones_like(frequency)
        else:
            source_power = _active_ricker_power(
                frequency, source.dominant_frequency_hz
            )
    else:
        noise = config.noise
        lower = max(0.02, noise.minimum_frequency_hz)
        upper = noise.maximum_frequency_hz
        frequency = np.linspace(lower, upper, 480)
        amplitude = source_spectral_amplitude(frequency, noise)
        source_power = amplitude**2
    peak = float(np.max(source_power, initial=0.0))
    if peak > 0.0:
        source_power = source_power / peak

    theory = theoretical_dispersion(
        config.model,
        frequency,
        rayleigh_modes=config.processing.rayleigh_modes,
        love_modes=config.processing.love_modes,
        backend="disba",
        compute_group=True,
    )
    use_love = (
        config.mode == "active"
        and config.active_source.component in {"horizontal", "both"}
    )
    phase = (
        np.column_stack((theory.rayleigh_mps, theory.love_mps))
        if use_love
        else theory.rayleigh_mps
    )
    phase_min, phase_max = _finite_row_extrema(phase)
    finite_phase = np.isfinite(phase_min) & np.isfinite(phase_max)
    source_ok = source_power >= SOURCE_POWER_THRESHOLD
    aperture_ok = (
        finite_phase
        & (receiver_aperture_m * frequency / phase_max >= APERTURE_WAVELENGTHS)
    )
    alias_ok = finite_phase & (
        frequency <= phase_min / (2.0 * effective_spacing_m)
    )
    if maximum_gauge_length_m is None:
        gauge_ok = np.ones_like(frequency, dtype=bool)
    else:
        gauge_ok = finite_phase & (
            frequency < phase_min / maximum_gauge_length_m
        )
    recommended_mask = _largest_true_run(
        source_ok & finite_phase & aperture_ok & alias_ok & gauge_ok
    )
    if not np.any(recommended_mask):
        recommended_mask = _largest_true_run(source_ok & finite_phase)
    if not np.any(recommended_mask):
        recommended_mask = finite_phase
    if not np.any(recommended_mask):
        raise ValueError("the model produced no finite dispersion roots in the source band")

    useful_index = np.flatnonzero(source_ok)
    if useful_index.size:
        useful_min = float(frequency[useful_index[0]])
        useful_max = float(frequency[useful_index[-1]])
    else:
        useful_min = float(frequency[0])
        useful_max = float(frequency[-1])
    selected = np.flatnonzero(recommended_mask)
    recommended_fmin = float(frequency[selected[0]])
    recommended_fmax = float(frequency[selected[-1]])

    selected_phase = phase[recommended_mask]
    finite_selected_phase = selected_phase[
        np.isfinite(selected_phase) & (selected_phase > 0.0)
    ]
    step = config.processing.velocity_step_mps
    recommended_vmin = max(
        step,
        _round_down(
            VELOCITY_LOWER_MARGIN * float(np.min(finite_selected_phase)), step
        ),
    )
    recommended_vmax = _round_up(
        VELOCITY_UPPER_MARGIN * float(np.max(finite_selected_phase)), step
    )

    timing_love_modes = (
        config.processing.love_modes
        if (
            (config.mode == "active" and config.active_source.component in {"horizontal", "both"})
            or (
                config.mode == "passive"
                and config.ambient_sources.component == "three_component"
            )
        )
        else 0
    )
    minimum_group, maximum_group = model_group_velocity_bounds(
        config.model,
        config.processing.frequency_min_hz,
        config.processing.frequency_max_hz,
        rayleigh_modes=config.processing.rayleigh_modes,
        love_modes=timing_love_modes,
    )

    travel_s = maximum_source_receiver_distance_m / minimum_group
    spectral_tail_s = 2.0 / config.processing.frequency_min_hz
    if config.mode == "active":
        source = config.active_source
        source_duration_s = (
            source.dt_s if source.wavelet == "unit_impulse" else 4.0 / source.dominant_frequency_hz
        )
    else:
        source_duration_s = 0.0
    active_record_s = 1.15 * travel_s + source_duration_s + spectral_tail_s
    passive_green_s = 1.15 * travel_s + spectral_tail_s
    maximum_pair_lag_s = receiver_aperture_m / minimum_group
    passive_window_s = max(
        passive_green_s,
        2.0 * maximum_pair_lag_s + spectral_tail_s,
        MINIMUM_PHYSICAL_FREQUENCY_BINS
        / (
            config.processing.frequency_max_hz
            - config.processing.frequency_min_hz
        ),
    )
    active_record_s = max(
        active_record_s,
        MINIMUM_PHYSICAL_FREQUENCY_BINS
        / (
            config.processing.frequency_max_hz
            - config.processing.frequency_min_hz
        ),
    )
    image_period_s = (
        active_record_s if config.mode == "active" else passive_window_s
    )
    physical_df_hz = 1.0 / image_period_s
    image_bandwidth_hz = recommended_fmax - recommended_fmin
    recommended_df_hz = _nice_step(
        min(0.25 * physical_df_hz, image_bandwidth_hz / 120.0)
    )
    velocity_bandwidth_mps = recommended_vmax - recommended_vmin
    representative_velocity_mps = 0.5 * (
        recommended_vmin + recommended_vmax
    )
    aperture_velocity_resolution_mps = (
        representative_velocity_mps**2
        / max(recommended_fmax * receiver_aperture_m, np.finfo(float).eps)
    )
    recommended_dv_mps = _nice_step(
        max(
            0.5,
            min(
                velocity_bandwidth_mps / 250.0,
                0.5 * aperture_velocity_resolution_mps,
            ),
        )
    )
    recommended_maximum_lag_s = 1.3 * maximum_pair_lag_s
    if config.nnws.enabled or config.nnws_v2.enabled:
        recommended_maximum_lag_s = max(
            recommended_maximum_lag_s,
            receiver_aperture_m / recommended_vmin
            + 4.0 / recommended_fmin,
        )
    return SurveyDesign(
        frequency_hz=frequency,
        source_power=source_power,
        theory=theory,
        useful_source_min_hz=useful_min,
        useful_source_max_hz=useful_max,
        recommended_frequency_min_hz=recommended_fmin,
        recommended_frequency_max_hz=recommended_fmax,
        recommended_frequency_step_hz=recommended_df_hz,
        recommended_velocity_min_mps=recommended_vmin,
        recommended_velocity_max_mps=recommended_vmax,
        recommended_velocity_step_mps=recommended_dv_mps,
        minimum_group_velocity_mps=minimum_group,
        maximum_group_velocity_mps=maximum_group,
        recommended_active_record_duration_s=active_record_s,
        recommended_passive_green_duration_s=passive_green_s,
        recommended_passive_window_s=passive_window_s,
        recommended_passive_maximum_lag_s=recommended_maximum_lag_s,
    )
