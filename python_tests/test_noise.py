from pathlib import Path
from dataclasses import replace

import numpy as np

from wavedispx.config import load_config
from wavedispx.noise import (
    add_independent_receiver_noise,
    source_spectral_amplitude,
    stochastic_source_coefficients,
    synthesize_mixed_noise_window,
    synthesize_mixed_noise_window_3c_band,
    synthesize_mixed_noise_window_band,
)


ROOT = Path(__file__).resolve().parents[1]


def test_receiver_noise_is_reproducible_band_limited_and_has_requested_rms():
    config = load_config(ROOT / "examples/passive_quick.toml").noise
    noisy_config = replace(config, receiver_noise_rms_fraction=0.25)
    sample_count = round(noisy_config.window_s / noisy_config.dt_s)
    time = np.arange(sample_count) * noisy_config.dt_s
    record = np.stack(
        [
            np.sin(2.0 * np.pi * noisy_config.dominant_frequency_hz * time),
            0.5 * np.sin(2.0 * np.pi * noisy_config.dominant_frequency_hz * time),
        ],
        axis=1,
    )
    first = add_independent_receiver_noise(
        record, noisy_config, np.random.default_rng(91)
    )
    second = add_independent_receiver_noise(
        record, noisy_config, np.random.default_rng(91)
    )
    np.testing.assert_array_equal(first, second)

    added = first - record
    signal_rms = np.sqrt(np.mean(record**2, axis=0))
    expected = 0.25 * np.median(signal_rms)
    np.testing.assert_allclose(
        np.sqrt(np.mean(added**2, axis=0)), expected, rtol=1e-12, atol=1e-12
    )
    frequency = np.fft.rfftfreq(sample_count, noisy_config.dt_s)
    spectrum = np.abs(np.fft.rfft(added, axis=0))
    outside = (frequency < noisy_config.minimum_frequency_hz) | (
        frequency > noisy_config.maximum_frequency_hz
    )
    assert np.max(spectrum[outside], initial=0.0) < 1e-10


def test_zero_receiver_noise_is_an_exact_no_op():
    config = load_config(ROOT / "examples/passive_quick.toml").noise
    record = np.arange(24, dtype=float).reshape(8, 3)
    output = add_independent_receiver_noise(record, config, np.random.default_rng(2))
    np.testing.assert_array_equal(output, record)


def test_ricker_psd_is_banded_and_peaks_near_dominant_frequency():
    config = load_config(ROOT / "examples/passive_two_layer.toml").noise
    frequency = np.linspace(0, 30, 601)
    amplitude = source_spectral_amplitude(frequency, config)
    assert np.all(amplitude[frequency < 3] == 0)
    assert np.all(amplitude[frequency > 18.8] == 0)
    assert abs(frequency[np.argmax(amplitude)] - 12.0) <= 0.05


def test_stationary_coefficients_are_reproducible():
    config = load_config(ROOT / "examples/passive_two_layer.toml").noise
    frequency = np.linspace(0, 100, 201)
    first = stochastic_source_coefficients(frequency, 4, config, np.random.default_rng(9))
    second = stochastic_source_coefficients(frequency, 4, config, np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 201)


def test_all_sources_are_mixed_before_output():
    config = load_config(ROOT / "examples/passive_two_layer.toml").noise
    small = replace(config, window_s=2.0, duration_s=2.0)
    samples = round(small.window_s / small.dt_s)
    frequency = np.fft.rfftfreq(samples, small.dt_s)
    transfer = np.ones((frequency.size, 3, 4), dtype=np.complex128)
    result = synthesize_mixed_noise_window(
        transfer,
        frequency,
        small,
        np.random.default_rng(11),
        return_source_coefficients=True,
    )
    assert result.record.shape == (samples, 3)
    assert result.source_coefficients.shape == (4, frequency.size)
    np.testing.assert_allclose(result.record[:, 0], result.record[:, 1])


def test_poisson_model_generates_finite_spectra():
    config = load_config(ROOT / "examples/passive_two_layer.toml").noise
    poisson = replace(config, temporal_model="poisson_ricker", event_rate_hz=1.0)
    frequency = np.linspace(0, 30, 121)
    coefficients = stochastic_source_coefficients(
        frequency, 3, poisson, np.random.default_rng(3)
    )
    assert coefficients.shape == (3, 121)
    assert np.all(np.isfinite(coefficients))
    assert np.any(np.abs(coefficients) > 0)


def test_poisson_window_can_return_reproducible_event_times():
    config = load_config(ROOT / "examples/passive_two_layer.toml").noise
    poisson = replace(
        config,
        temporal_model="poisson_ricker",
        event_rate_hz=1.0,
        window_s=2.0,
        duration_s=2.0,
    )
    sample_count = round(poisson.window_s / poisson.dt_s)
    frequency = np.fft.rfftfreq(sample_count, poisson.dt_s)
    selected = (frequency >= poisson.minimum_frequency_hz) & (
        frequency <= poisson.maximum_frequency_hz
    )
    transfer = np.ones((selected.sum(), 2, 3), dtype=np.complex128)
    first = synthesize_mixed_noise_window_band(
        transfer,
        frequency[selected],
        poisson,
        np.random.default_rng(17),
        return_source_events=True,
    )
    second = synthesize_mixed_noise_window_band(
        transfer,
        frequency[selected],
        poisson,
        np.random.default_rng(17),
        return_source_events=True,
    )
    np.testing.assert_array_equal(first.source_events, second.source_events)
    assert first.source_events.dtype.names == (
        "source_index",
        "time_in_window_s",
        "weight",
    )
    assert np.all(first.source_events["time_in_window_s"] >= 0)
    assert np.all(first.source_events["time_in_window_s"] < poisson.window_s)


def test_band_limited_mixing_matches_full_grid_mixing():
    config = load_config(ROOT / "examples/passive_two_layer.toml").noise
    small = replace(config, window_s=2.0, duration_s=2.0)
    sample_count = round(small.window_s / small.dt_s)
    frequency = np.fft.rfftfreq(sample_count, small.dt_s)
    selected = (frequency >= small.minimum_frequency_hz) & (
        frequency <= small.maximum_frequency_hz
    )
    rng = np.random.default_rng(21)
    full_transfer = rng.normal(size=(frequency.size, 3, 4)) + 1j * rng.normal(
        size=(frequency.size, 3, 4)
    )
    full_transfer[~selected] = 0
    full = synthesize_mixed_noise_window(
        full_transfer, frequency, small, np.random.default_rng(5)
    )
    band = synthesize_mixed_noise_window_band(
        full_transfer[selected], frequency[selected], small, np.random.default_rng(5)
    )
    np.testing.assert_allclose(band.record, full.record, rtol=1e-12, atol=1e-12)


def test_source_activity_weights_modulate_the_exact_mixed_realization():
    config = load_config(ROOT / "examples/passive_quick.toml").noise
    sample_count = round(config.window_s / config.dt_s)
    full_frequency = np.fft.rfftfreq(sample_count, config.dt_s)
    selected = (
        (full_frequency >= config.minimum_frequency_hz)
        & (full_frequency <= config.maximum_frequency_hz)
    )
    frequency = full_frequency[selected]
    transfer = np.ones((frequency.size, 2, 3), dtype=np.complex128)
    weights = np.array([1.0, 0.0, 0.25])
    baseline = synthesize_mixed_noise_window_band(
        transfer,
        frequency,
        config,
        np.random.default_rng(81),
        return_source_coefficients=True,
    )
    modulated = synthesize_mixed_noise_window_band(
        transfer,
        frequency,
        config,
        np.random.default_rng(81),
        return_source_coefficients=True,
        source_activity_weights=weights,
    )
    np.testing.assert_allclose(
        modulated.source_coefficients,
        baseline.source_coefficients * weights[:, None],
        rtol=0,
        atol=0,
    )
    assert not np.array_equal(modulated.record, baseline.record)


def test_three_component_mixer_rotates_full_green_tensor_consistently():
    config = load_config(ROOT / "examples/passive_quick.toml").noise
    small = replace(config, window_s=2.0, duration_s=2.0)
    sample_count = round(small.window_s / small.dt_s)
    full_frequency = np.fft.rfftfreq(sample_count, small.dt_s)
    selected = (full_frequency >= small.minimum_frequency_hz) & (
        full_frequency <= small.maximum_frequency_hz
    )
    frequency = full_frequency[selected]
    transfer = np.empty((frequency.size, 2, 1, 5), dtype=np.complex128)
    transfer[..., 0] = 1.0
    transfer[..., 1] = 2.0
    transfer[..., 2] = 3.0
    transfer[..., 3] = 4.0
    transfer[..., 4] = 5.0
    azimuth = np.array([[0.0], [0.5 * np.pi]])
    result = synthesize_mixed_noise_window_3c_band(
        transfer,
        frequency,
        azimuth,
        small,
        np.random.default_rng(8),
        return_source_coefficients=True,
    )
    fx, fy, fz = result.source_coefficients[0]
    recovered = np.fft.rfft(result.record, axis=0)[selected]
    expected = np.empty_like(recovered)
    expected[:, 0, 0] = fx + 4.0 * fz
    expected[:, 0, 1] = 2.0 * fy
    expected[:, 0, 2] = 3.0 * fx + 5.0 * fz
    expected[:, 1, 0] = 2.0 * fx
    expected[:, 1, 1] = fy + 4.0 * fz
    expected[:, 1, 2] = 3.0 * fy + 5.0 * fz
    np.testing.assert_allclose(recovered, expected, rtol=1e-11, atol=1e-11)
