"""Ambient-source PSDs and finite-window stochastic source spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import tukey

from .config import NoiseConfig


@dataclass(frozen=True)
class MixedNoiseWindow:
    record: np.ndarray
    source_coefficients: np.ndarray | None
    source_events: np.ndarray | None = None


SOURCE_EVENT_DTYPE = np.dtype(
    [
        ("source_index", np.int64),
        ("time_in_window_s", np.float64),
        ("weight", np.float64),
    ]
)

SOURCE_EVENT_3C_DTYPE = np.dtype(
    [
        ("source_index", np.int64),
        ("component_index", np.int64),
        ("time_in_window_s", np.float64),
        ("weight", np.float64),
    ]
)


def add_independent_receiver_noise(
    record: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent band-limited receiver noise before correlation.

    The configured fraction is the RMS of each receiver/component noise trace
    relative to the median non-zero RMS of the mixed signal traces.  A value of
    zero is an exact no-op.  The noise stream uses a generator independent from
    the ambient-source realization so changing receiver noise does not change
    the source sequence.
    """
    signal = np.asarray(record, dtype=np.float64)
    if signal.ndim not in (2, 3):
        raise ValueError("record must be time-by-receiver or time-by-receiver-by-component")
    if np.any(~np.isfinite(signal)):
        raise ValueError("record contains non-finite values")
    fraction = float(config.receiver_noise_rms_fraction)
    if fraction == 0.0:
        return signal

    trace_rms = np.sqrt(np.mean(signal**2, axis=0))
    positive = trace_rms[np.isfinite(trace_rms) & (trace_rms > 0.0)]
    if not positive.size:
        return signal
    target_rms = fraction * float(np.median(positive))

    white = rng.standard_normal(signal.shape)
    spectrum = np.fft.rfft(white, axis=0)
    frequency = np.fft.rfftfreq(signal.shape[0], config.dt_s)
    inside = (
        (frequency >= config.minimum_frequency_hz)
        & (frequency <= config.maximum_frequency_hz)
    )
    spectrum[~inside] = 0.0
    band_limited = np.fft.irfft(spectrum, n=signal.shape[0], axis=0)
    noise_rms = np.sqrt(np.mean(band_limited**2, axis=0))
    scale = np.divide(
        target_rms,
        noise_rms,
        out=np.zeros_like(noise_rms),
        where=noise_rms > 0.0,
    )
    return signal + band_limited * scale[None, ...]


def _stochastic_source_realization(
    frequency_hz: np.ndarray,
    source_count: int,
    config: NoiseConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source spectra plus discrete Poisson events, when applicable."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    amplitude = source_spectral_amplitude(frequency, config)
    coefficient_count = source_count if config.independent_sources else 1
    events: list[tuple[int, float, float]] = []
    if config.temporal_model == "stationary_gaussian":
        coefficients = np.zeros((coefficient_count, frequency.size), dtype=np.complex128)
        active = amplitude > 0
        coefficients[:, active] = (
            rng.standard_normal((coefficient_count, int(np.count_nonzero(active))))
            + 1j * rng.standard_normal((coefficient_count, int(np.count_nonzero(active))))
        ) / np.sqrt(2.0)
    elif config.temporal_model == "poisson_ricker":
        coefficients = np.zeros((coefficient_count, frequency.size), dtype=np.complex128)
        for source_index in range(coefficient_count):
            event_count = rng.poisson(config.event_rate_hz * config.window_s)
            if event_count == 0:
                continue
            event_time = rng.uniform(0.0, config.window_s, event_count)
            event_weight = rng.choice((-1.0, 1.0), event_count) * rng.uniform(
                0.5, 1.0, event_count
            )
            coefficients[source_index] = np.sum(
                event_weight[:, None]
                * np.exp(-2j * np.pi * event_time[:, None] * frequency[None, :]),
                axis=0,
            )
            catalog_source = source_index if config.independent_sources else -1
            events.extend(
                (catalog_source, float(time_s), float(weight))
                for time_s, weight in zip(event_time, event_weight)
            )
    else:
        raise ValueError(f"unsupported temporal model: {config.temporal_model}")
    coefficients *= amplitude
    if not config.independent_sources:
        coefficients = np.repeat(coefficients, source_count, axis=0)
    event_array = np.asarray(events, dtype=SOURCE_EVENT_DTYPE)
    return coefficients, event_array


def synthesize_mixed_noise_window_band(
    transfer_spectrum: np.ndarray,
    transfer_frequency_hz: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator,
    *,
    return_source_coefficients: bool = False,
    return_source_events: bool = False,
    source_block_size: int | None = None,
    source_activity_weights: np.ndarray | None = None,
) -> MixedNoiseWindow:
    """Mix a compact band-limited receiver/source transfer tensor."""
    transfer = np.asarray(transfer_spectrum)
    frequency = np.asarray(transfer_frequency_hz, dtype=np.float64).reshape(-1)
    if transfer.ndim != 3 or transfer.shape[0] != frequency.size:
        raise ValueError("transfer_spectrum must be frequency-by-receiver-by-source")
    window_samples = int(round(config.window_s / config.dt_s))
    full_frequency = np.fft.rfftfreq(window_samples, config.dt_s)
    frequency_index = np.rint(frequency * window_samples * config.dt_s).astype(int)
    if np.any(frequency_index < 0) or np.any(frequency_index >= full_frequency.size):
        raise ValueError("transfer frequencies fall outside the rFFT grid")
    if not np.allclose(full_frequency[frequency_index], frequency, rtol=0, atol=1e-12):
        raise ValueError("transfer frequencies must lie exactly on the rFFT grid")
    coefficients, source_events = _stochastic_source_realization(
        frequency, transfer.shape[2], config, rng
    )
    if source_activity_weights is not None:
        activity = np.asarray(source_activity_weights, dtype=np.float64).reshape(-1)
        if (
            activity.size != transfer.shape[2]
            or np.any(~np.isfinite(activity))
            or np.any(activity < 0)
        ):
            raise ValueError(
                "source_activity_weights must be one finite non-negative value per source"
            )
        coefficients *= activity[:, None]
    if source_block_size is None:
        source_block_size = transfer.shape[2]
    if source_block_size < 1:
        raise ValueError("source_block_size must be >= 1")
    mixed_band = np.zeros(
        (frequency.size, transfer.shape[1]), dtype=np.complex128
    )
    for first_source in range(0, transfer.shape[2], source_block_size):
        last_source = min(transfer.shape[2], first_source + source_block_size)
        mixed_band += np.einsum(
            "frs,sf->fr",
            transfer[:, :, first_source:last_source],
            coefficients[first_source:last_source],
            optimize=True,
        )
    mixed_spectrum = np.zeros(
        (full_frequency.size, transfer.shape[1]), dtype=np.complex128
    )
    mixed_spectrum[frequency_index] = mixed_band
    mixed_spectrum[0] = mixed_spectrum[0].real
    if window_samples % 2 == 0:
        mixed_spectrum[-1] = mixed_spectrum[-1].real
    record = np.fft.irfft(mixed_spectrum, n=window_samples, axis=0)
    return MixedNoiseWindow(
        record,
        coefficients if return_source_coefficients else None,
        source_events if return_source_events else None,
    )


def synthesize_mixed_noise_window_3c_band(
    transfer_spectrum: np.ndarray,
    transfer_frequency_hz: np.ndarray,
    source_receiver_azimuth_rad: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator,
    *,
    return_source_coefficients: bool = False,
    return_source_events: bool = False,
    source_block_size: int | None = None,
    source_activity_weights: np.ndarray | None = None,
) -> MixedNoiseWindow:
    """Mix physical global X/Y/Z random forces through five local Green terms.

    ``transfer_spectrum`` is ``(frequency, receiver, source, 5)`` ordered as
    ``RR, TT, ZR, RZ, ZZ`` where the first letter is receiver component and the
    second is source-force orientation in the local source-receiver frame.
    The returned finite record is ``(time, receiver, 3)`` in global X/Y/Z.
    """
    transfer = np.asarray(transfer_spectrum)
    frequency = np.asarray(transfer_frequency_hz, dtype=np.float64).reshape(-1)
    azimuth = np.asarray(source_receiver_azimuth_rad, dtype=np.float64)
    if transfer.ndim != 4 or transfer.shape[0] != frequency.size or transfer.shape[3] != 5:
        raise ValueError(
            "transfer_spectrum must be frequency-by-receiver-by-source-by-5"
        )
    if azimuth.shape != transfer.shape[1:3] or np.any(~np.isfinite(azimuth)):
        raise ValueError("source_receiver_azimuth_rad must be receiver-by-source")
    window_samples = int(round(config.window_s / config.dt_s))
    full_frequency = np.fft.rfftfreq(window_samples, config.dt_s)
    frequency_index = np.rint(frequency * window_samples * config.dt_s).astype(int)
    if np.any(frequency_index < 0) or np.any(frequency_index >= full_frequency.size):
        raise ValueError("transfer frequencies fall outside the rFFT grid")
    if not np.allclose(full_frequency[frequency_index], frequency, rtol=0, atol=1e-12):
        raise ValueError("transfer frequencies must lie exactly on the rFFT grid")

    source_count = transfer.shape[2]
    flat_coefficients, flat_events = _stochastic_source_realization(
        frequency, 3 * source_count, config, rng
    )
    coefficients = flat_coefficients.reshape(source_count, 3, frequency.size)
    if source_activity_weights is not None:
        activity = np.asarray(source_activity_weights, dtype=np.float64).reshape(-1)
        if (
            activity.size != source_count
            or np.any(~np.isfinite(activity))
            or np.any(activity < 0)
        ):
            raise ValueError(
                "source_activity_weights must be one finite non-negative value per source"
            )
        coefficients *= activity[:, None, None]
    if source_block_size is None:
        source_block_size = source_count
    if source_block_size < 1:
        raise ValueError("source_block_size must be >= 1")
    mixed_band = np.zeros(
        (frequency.size, transfer.shape[1], 3), dtype=np.complex128
    )
    cosine = np.cos(azimuth)
    sine = np.sin(azimuth)
    for first_source in range(0, source_count, source_block_size):
        last_source = min(source_count, first_source + source_block_size)
        block = transfer[:, :, first_source:last_source]
        c = cosine[:, first_source:last_source]
        s = sine[:, first_source:last_source]
        coefficient = coefficients[first_source:last_source]
        fx = coefficient[:, 0]
        fy = coefficient[:, 1]
        fz = coefficient[:, 2]
        radial_force = c[:, :, None] * fx[None, :, :] + s[:, :, None] * fy[None, :, :]
        transverse_force = -s[:, :, None] * fx[None, :, :] + c[:, :, None] * fy[None, :, :]
        # Keep source azimuth inside the summation: every source has its own
        # local cylindrical basis, even after all sources are mixed.
        mixed_band[:, :, 0] += (
            np.einsum("frs,rsf,rs->fr", block[..., 0], radial_force, c, optimize=True)
            - np.einsum(
                "frs,rsf,rs->fr", block[..., 1], transverse_force, s, optimize=True
            )
            + np.einsum("frs,sf,rs->fr", block[..., 3], fz, c, optimize=True)
        )
        mixed_band[:, :, 1] += (
            np.einsum("frs,rsf,rs->fr", block[..., 0], radial_force, s, optimize=True)
            + np.einsum(
                "frs,rsf,rs->fr", block[..., 1], transverse_force, c, optimize=True
            )
            + np.einsum("frs,sf,rs->fr", block[..., 3], fz, s, optimize=True)
        )
        mixed_band[:, :, 2] += (
            np.einsum("frs,rsf->fr", block[..., 2], radial_force, optimize=True)
            + np.einsum("frs,sf->fr", block[..., 4], fz, optimize=True)
        )
    mixed_spectrum = np.zeros(
        (full_frequency.size, transfer.shape[1], 3), dtype=np.complex128
    )
    mixed_spectrum[frequency_index] = mixed_band
    mixed_spectrum[0] = mixed_spectrum[0].real
    if window_samples % 2 == 0:
        mixed_spectrum[-1] = mixed_spectrum[-1].real
    record = np.fft.irfft(mixed_spectrum, n=window_samples, axis=0)

    source_events = None
    if return_source_events:
        source_events = np.empty(flat_events.size, dtype=SOURCE_EVENT_3C_DTYPE)
        flat_index = flat_events["source_index"]
        independent = flat_index >= 0
        source_events["source_index"] = -1
        source_events["component_index"] = -1
        source_events["source_index"][independent] = flat_index[independent] // 3
        source_events["component_index"][independent] = flat_index[independent] % 3
        source_events["time_in_window_s"] = flat_events["time_in_window_s"]
        source_events["weight"] = flat_events["weight"]
    return MixedNoiseWindow(
        record.astype(np.float64, copy=False),
        coefficients if return_source_coefficients else None,
        source_events,
    )


def source_spectral_amplitude(frequency_hz: np.ndarray, config: NoiseConfig) -> np.ndarray:
    """Return a peak-normalized amplitude spectrum whose square is the PSD."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    inside = (
        (frequency >= config.minimum_frequency_hz)
        & (frequency <= config.maximum_frequency_hz)
    )
    amplitude = np.zeros(frequency.size, dtype=np.float64)
    if config.psd_shape == "ricker":
        ratio = frequency[inside] / config.dominant_frequency_hz
        amplitude[inside] = ratio**2 * np.exp(1.0 - ratio**2)
    elif config.psd_shape == "flat_band":
        count = int(np.count_nonzero(inside))
        if count:
            amplitude[inside] = tukey(count, 0.2)
    else:
        table = np.loadtxt(config.psd_file, delimiter=",", skiprows=1, ndmin=2)
        if table.shape[1] < 2:
            raise ValueError("PSD CSV must contain frequency_hz,power columns")
        power = np.interp(frequency, table[:, 0], table[:, 1], left=0.0, right=0.0)
        amplitude = np.sqrt(np.maximum(power, 0.0))
        amplitude[~inside] = 0.0
    peak = amplitude.max(initial=0.0)
    if peak:
        amplitude /= peak
    return amplitude * config.source_amplitude


def stochastic_source_coefficients(
    frequency_hz: np.ndarray,
    source_count: int,
    config: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one finite-window source spectrum, source by frequency."""
    coefficients, _ = _stochastic_source_realization(
        frequency_hz, source_count, config, rng
    )
    return coefficients


def synthesize_mixed_noise_window(
    transfer_spectrum: np.ndarray,
    frequency_hz: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator,
    *,
    return_source_coefficients: bool = False,
    return_source_events: bool = False,
) -> MixedNoiseWindow:
    """Propagate and mix all sources before returning receiver records.

    ``transfer_spectrum`` is frequency-by-receiver-by-source on the complete
    non-negative rFFT grid.  The output contains no source dimension.
    """
    transfer = np.asarray(transfer_spectrum)
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    if transfer.ndim != 3 or transfer.shape[0] != frequency.size:
        raise ValueError("transfer_spectrum must be frequency-by-receiver-by-source")
    window_samples = int(round(config.window_s / config.dt_s))
    expected_frequency = np.fft.rfftfreq(window_samples, config.dt_s)
    if frequency.shape != expected_frequency.shape or not np.allclose(
        frequency, expected_frequency, rtol=0, atol=1e-12
    ):
        raise ValueError("frequency_hz must be the complete rFFT grid for noise.window_s")
    coefficients, source_events = _stochastic_source_realization(
        frequency, transfer.shape[2], config, rng
    )
    mixed_spectrum = np.einsum(
        "frs,sf->fr", transfer, coefficients, optimize=True
    )
    mixed_spectrum[0] = mixed_spectrum[0].real
    if window_samples % 2 == 0:
        mixed_spectrum[-1] = mixed_spectrum[-1].real
    record = np.fft.irfft(mixed_spectrum, n=window_samples, axis=0)
    return MixedNoiseWindow(
        record.astype(np.float64, copy=False),
        coefficients if return_source_coefficients else None,
        source_events if return_source_events else None,
    )
