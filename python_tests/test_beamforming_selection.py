from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from wavedispx.beamforming_selection import (
    PhaseOnlyBeamformer,
    angular_target_weights,
    angular_uniformity_weights,
    circular_distance_deg,
)
from wavedispx.config import load_config
from wavedispx.noise import synthesize_mixed_noise_window_band
from wavedispx.processing import receiver_pair_ncf_branches


ROOT = Path(__file__).resolve().parents[1]


def _sunflower_array(count: int = 13, radius_m: float = 45.0):
    index = np.arange(count, dtype=np.float64)
    radius = radius_m * np.sqrt(index / (count - 1))
    angle = np.deg2rad(index * 137.507764)
    return radius * np.cos(angle), radius * np.sin(angle)


def _plane_wave(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    backazimuth_deg: float,
    velocity_mps: float,
    dt_s: float = 0.01,
    duration_s: float = 25.0,
    noise_fraction: float = 0.18,
) -> np.ndarray:
    time = np.arange(round(duration_s / dt_s)) * dt_s
    direction = np.array(
        [np.cos(np.deg2rad(backazimuth_deg)), np.sin(np.deg2rad(backazimuth_deg))]
    )
    projected = x_m * direction[0] + y_m * direction[1]
    record = np.zeros((time.size, x_m.size), dtype=np.float64)
    for frequency, phase in zip((4.5, 6.5, 8.5, 10.5), (0.2, 1.1, 2.0, 2.7)):
        record += np.sin(
            2.0
            * np.pi
            * frequency
            * (time[:, None] + projected[None, :] / velocity_mps)
            + phase
        )
    rng = np.random.default_rng(20260815 + int(round(backazimuth_deg)))
    record += noise_fraction * np.std(record) * rng.standard_normal(record.shape)
    return record


def test_phase_only_beamformer_recovers_registered_plane_waves() -> None:
    x_m, y_m = _sunflower_array()
    beamformer = PhaseOnlyBeamformer(
        x_m,
        y_m,
        dt_s=0.01,
        window_samples=2500,
    )
    errors = []
    for backazimuth_deg, velocity_mps in ((12.0, 200.0), (75.0, 280.0), (168.0, 360.0)):
        estimate = beamformer.estimate(
            _plane_wave(
                x_m,
                y_m,
                backazimuth_deg=backazimuth_deg,
                velocity_mps=velocity_mps,
            )
        )
        errors.append(float(circular_distance_deg(estimate.peak_azimuth_deg, backazimuth_deg)))
        assert abs(estimate.peak_velocity_mps - velocity_mps) <= 40.0
        assert np.isclose(np.sum(estimate.angular_power), 1.0)
    assert np.median(errors) <= 5.0
    assert np.quantile(errors, 0.9) <= 10.0


def test_uniformity_optimizer_reduces_directional_excess_with_neff_floor() -> None:
    rng = np.random.default_rng(44)
    window_count = 80
    azimuth_count = 72
    power = rng.uniform(0.8, 1.2, (window_count, azimuth_count))
    power[:40, 10] += 40.0

    result = angular_uniformity_weights(power)

    assert result.success
    assert result.weighted_misfit < result.unweighted_misfit
    assert result.effective_window_count >= 0.50 * window_count - 1e-8
    assert np.mean(result.weights[:40]) < np.mean(result.weights[40:])
    assert np.isclose(np.mean(result.weights), 1.0)


def test_uniformity_optimizer_leaves_identical_uniform_windows_unchanged() -> None:
    power = np.ones((20, 72), dtype=np.float64)

    result = angular_uniformity_weights(power)

    np.testing.assert_allclose(result.weights, 1.0, atol=1e-10)
    assert result.weighted_misfit < 1e-28


def test_angular_target_optimizer_upweights_registered_target_windows() -> None:
    rng = np.random.default_rng(441)
    window_count = 80
    azimuth_count = 72
    power = rng.uniform(0.8, 1.2, (window_count, azimuth_count))
    power[:40, 0] += 40.0
    target = np.zeros(azimuth_count)
    target[[0, 1, -1]] = 1.0

    result = angular_target_weights(power, target)

    assert result.success
    assert result.weighted_misfit < result.unweighted_misfit
    assert result.effective_window_count >= 0.50 * window_count - 1e-8
    assert np.mean(result.weights[:40]) > np.mean(result.weights[40:])
    assert np.isclose(np.mean(result.weights), 1.0)


def test_target_pair_branch_sign_convention_is_fixed_by_source_side() -> None:
    dt_s = 0.01
    time = np.arange(400) * dt_s
    pulse = np.exp(-0.5 * np.square((time - 1.0) / 0.05))

    def shifted(delay_samples: int) -> np.ndarray:
        return np.pad(pulse, (delay_samples, 0))[: pulse.size]

    source_at_zero_deg = np.column_stack((shifted(60), pulse))
    source_at_180_deg = np.column_stack((pulse, shifted(60)))
    kwargs = {
        "receiver_x_m": np.array([-300.0, 300.0]),
        "receiver_y_m": np.zeros(2),
        "dt_s": dt_s,
        "window_s": 4.0,
        "overlap": 0.0,
        "maximum_lag_s": 1.0,
        "normalization": "none",
    }
    east = receiver_pair_ncf_branches(source_at_zero_deg, **kwargs)
    west = receiver_pair_ncf_branches(source_at_180_deg, **kwargs)

    east_positive_peak = east.positive.lag_time_s[
        np.argmax(np.abs(east.positive.data[:, 0]))
    ]
    west_negative_peak = west.negative.lag_time_s[
        np.argmax(np.abs(west.negative.data[:, 0]))
    ]
    assert np.isclose(east_positive_peak, 0.60, atol=dt_s)
    assert np.isclose(west_negative_peak, 0.60, atol=dt_s)
    assert np.max(np.abs(east.positive.data[:, 0])) > 5 * np.max(
        np.abs(east.negative.data[:, 0])
    )
    assert np.max(np.abs(west.negative.data[:, 0])) > 5 * np.max(
        np.abs(west.positive.data[:, 0])
    )


def test_adding_auxiliary_receivers_preserves_target_record_and_event_catalog() -> None:
    base = load_config(ROOT / "examples/passive_quick.toml").noise
    config = replace(
        base,
        temporal_model="poisson_ricker",
        duration_s=4.0,
        window_s=4.0,
        event_rate_hz=0.8,
    )
    sample_count = round(config.window_s / config.dt_s)
    full_frequency = np.fft.rfftfreq(sample_count, config.dt_s)
    selected = (full_frequency >= config.minimum_frequency_hz) & (
        full_frequency <= config.maximum_frequency_hz
    )
    frequency = full_frequency[selected]
    rng = np.random.default_rng(81)
    target_transfer = rng.standard_normal((frequency.size, 2, 4)) + 1j * rng.standard_normal(
        (frequency.size, 2, 4)
    )
    expanded_transfer = np.concatenate(
        (
            target_transfer,
            rng.standard_normal((frequency.size, 5, 4))
            + 1j * rng.standard_normal((frequency.size, 5, 4)),
        ),
        axis=1,
    )

    target = synthesize_mixed_noise_window_band(
        target_transfer,
        frequency,
        config,
        np.random.default_rng(9001),
        return_source_events=True,
    )
    expanded = synthesize_mixed_noise_window_band(
        expanded_transfer,
        frequency,
        config,
        np.random.default_rng(9001),
        return_source_events=True,
    )

    np.testing.assert_allclose(expanded.record[:, :2], target.record, rtol=0, atol=0)
    np.testing.assert_array_equal(expanded.source_events, target.source_events)
