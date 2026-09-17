from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from wavedispx.processing import (
    correlation_to_egf,
    pair_ftan,
    phase_to_group_velocity,
)
from wavedispx.processing.ftan import _pick_phase_curve


def _synthetic_constant_speed_egf(
    *,
    dt_s: float = 0.002,
    sample_count: int = 16_384,
    distance_m: float = 600.0,
    velocity_mps: float = 300.0,
) -> np.ndarray:
    frequency = np.fft.rfftfreq(sample_count, dt_s)
    amplitude = np.zeros_like(frequency)
    flat = (frequency >= 1.0) & (frequency <= 16.0)
    low_taper = (frequency >= 0.5) & (frequency < 1.0)
    high_taper = (frequency > 16.0) & (frequency <= 20.0)
    amplitude[flat] = 1.0
    amplitude[low_taper] = np.sin(
        0.5 * np.pi * (frequency[low_taper] - 0.5) / 0.5
    ) ** 2
    amplitude[high_taper] = np.cos(
        0.5 * np.pi * (frequency[high_taper] - 16.0) / 4.0
    ) ** 2
    spectrum = amplitude * np.exp(
        -1j * 2.0 * np.pi * frequency * distance_m / velocity_mps
    )
    return np.fft.irfft(spectrum, n=sample_count)[: sample_count // 2]


def _dominant_group_error(
    result,
    theory_velocity_mps: np.ndarray | float,
) -> np.ndarray:
    ridge = result.group_image.velocity_mps[
        np.argmax(result.group_image.amplitude, axis=0)
    ]
    error = np.abs(ridge - theory_velocity_mps)
    edge = max(1, int(np.floor(0.08 * error.size)))
    return error[edge:-edge]


def _synthetic_dispersive_egf(
    *,
    dt_s: float = 0.002,
    sample_count: int = 16_384,
    distance_m: float = 600.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequency = np.fft.rfftfreq(sample_count, dt_s)
    phase_velocity = 220.0 + 160.0 / (1.0 + (frequency / 4.0) ** 2)
    amplitude = np.zeros_like(frequency)
    flat = (frequency >= 1.0) & (frequency <= 16.0)
    low_taper = (frequency >= 0.5) & (frequency < 1.0)
    high_taper = (frequency > 16.0) & (frequency <= 20.0)
    amplitude[flat] = 1.0
    amplitude[low_taper] = np.sin(
        0.5 * np.pi * (frequency[low_taper] - 0.5) / 0.5
    ) ** 2
    amplitude[high_taper] = np.cos(
        0.5 * np.pi * (frequency[high_taper] - 16.0) / 4.0
    ) ** 2
    phase = -(2.0 * np.pi * frequency * distance_m / phase_velocity + np.pi / 4.0)
    spectrum = amplitude * np.exp(1j * phase)
    spectrum[0] = 0.0
    trace = np.fft.irfft(spectrum, n=sample_count)

    theory_frequency = np.linspace(2.0, 12.0, 501)
    theory_phase = 220.0 + 160.0 / (1.0 + (theory_frequency / 4.0) ** 2)
    order = np.argsort(1.0 / theory_frequency)
    theory_period = (1.0 / theory_frequency)[order]
    theory_phase = theory_phase[order]
    theory_group = phase_to_group_velocity(theory_period, theory_phase)
    return trace[: sample_count // 2], theory_period, theory_phase, theory_group


def test_correlation_to_egf_matches_matlab_hilbert_sign() -> None:
    sample = np.arange(2_048)
    correlation = np.cos(2.0 * np.pi * 0.08 * sample)

    converted = correlation_to_egf(correlation)

    expected = np.sin(2.0 * np.pi * 0.08 * sample)
    assert np.corrcoef(converted[64:-64], expected[64:-64])[0, 1] > 0.999


def test_phase_to_group_velocity_recovers_nondispersive_speed() -> None:
    period = np.linspace(0.08, 0.5, 101)
    phase = np.full(period.size, 300.0)

    group = phase_to_group_velocity(period, phase)

    np.testing.assert_allclose(group, 300.0, atol=1e-8)


@pytest.mark.parametrize("polarity", [1.0, -1.0])
@pytest.mark.parametrize("input_kind", ["egf", "correlation"])
@pytest.mark.parametrize(
    ("distance_m", "gaussian_alpha"),
    [(300.0, 20.0), (600.0, 40.0), (900.0, 80.0)],
)
def test_group_energy_meets_gate0_for_constant_speed_without_a_tracked_pick(
    polarity: float,
    input_kind: str,
    distance_m: float,
    gaussian_alpha: float,
) -> None:
    egf = polarity * _synthetic_constant_speed_egf(distance_m=distance_m)
    trace = -np.imag(hilbert(egf)) if input_kind == "correlation" else egf
    result = pair_ftan(
        trace,
        dt_s=0.002,
        distance_m=distance_m,
        frequency_range_hz=(2.0, 12.0),
        group_velocity_range_mps=(200.0, 400.0),
        phase_velocity_range_mps=(200.0, 400.0),
        period_count=96,
        velocity_step_mps=2.0,
        input_kind=input_kind,
        minimum_snr=0.0,
        gaussian_alpha=gaussian_alpha,
        phase_method="frequency_domain",
    )

    error = _dominant_group_error(result, 300.0)
    assert np.median(error) <= 2.0
    assert np.percentile(error, 90.0) <= 4.0


def test_group_energy_meets_gate0_for_known_dispersion_without_a_tracked_pick() -> None:
    trace, theory_period, _, theory_group = _synthetic_dispersive_egf()
    result = pair_ftan(
        trace,
        dt_s=0.002,
        distance_m=600.0,
        frequency_range_hz=(2.0, 12.0),
        group_velocity_range_mps=(140.0, 360.0),
        phase_velocity_range_mps=(200.0, 400.0),
        period_count=96,
        velocity_step_mps=2.0,
        minimum_snr=0.0,
        gaussian_alpha=40.0,
        phase_method="frequency_domain",
    )

    expected = np.interp(result.group_image.period_s, theory_period, theory_group)
    error = _dominant_group_error(result, expected)
    assert np.median(error) <= 2.0
    assert np.percentile(error, 90.0) <= 4.0


def test_pair_ftan_recovers_known_group_and_phase_curves() -> None:
    trace, theory_period, theory_phase, theory_group = _synthetic_dispersive_egf()
    result = pair_ftan(
        trace,
        dt_s=0.002,
        distance_m=600.0,
        frequency_range_hz=(2.0, 12.0),
        group_velocity_range_mps=(170.0, 330.0),
        phase_velocity_range_mps=(210.0, 370.0),
        period_count=96,
        velocity_step_mps=2.0,
        minimum_snr=1.0,
        gaussian_alpha=80.0,
        phase_minus_group_range_mps=(0.0, 100.0),
        phase_velocity_anchor=(0.45, float(np.interp(0.45, theory_period, theory_phase))),
    )

    group_valid = result.group_curve.valid_mask
    phase_valid = result.phase_curve.valid_mask
    assert np.count_nonzero(group_valid) >= 80
    assert np.count_nonzero(phase_valid) >= 80

    group_period = result.group_curve.period_s[group_valid]
    expected_group = np.interp(group_period, theory_period, theory_group)
    group_error = np.abs(result.group_curve.velocity_mps[group_valid] - expected_group)
    phase_period = result.phase_curve.period_s[phase_valid]
    expected_phase = np.interp(phase_period, theory_period, theory_phase)
    phase_error = np.abs(result.phase_curve.velocity_mps[phase_valid] - expected_phase)

    assert np.nanmedian(group_error) < 12.0
    assert np.nanpercentile(group_error, 90) < 25.0
    assert np.nanmedian(phase_error) < 10.0
    assert np.nanpercentile(phase_error, 90) < 22.0


def test_pair_ftan_rejects_frequency_above_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        pair_ftan(
            np.ones(128),
            dt_s=0.1,
            distance_m=100.0,
            frequency_range_hz=(1.0, 6.0),
            group_velocity_range_mps=(100.0, 300.0),
            phase_velocity_range_mps=(100.0, 300.0),
        )


def test_pair_ftan_rejects_nonpositive_phase_jump_limit() -> None:
    with pytest.raises(ValueError, match="phase_maximum_jump_mps"):
        pair_ftan(
            np.ones(128),
            dt_s=0.01,
            distance_m=100.0,
            frequency_range_hz=(2.0, 10.0),
            group_velocity_range_mps=(100.0, 400.0),
            phase_velocity_range_mps=(100.0, 500.0),
            phase_maximum_jump_mps=0.0,
        )


def test_phase_pick_without_cycle_anchor_is_explicitly_invalid() -> None:
    trace, _, _, _ = _synthetic_dispersive_egf()

    result = pair_ftan(
        trace,
        dt_s=0.002,
        distance_m=600.0,
        frequency_range_hz=(2.0, 12.0),
        group_velocity_range_mps=(170.0, 330.0),
        phase_velocity_range_mps=(210.0, 370.0),
        period_count=24,
        velocity_step_mps=4.0,
        minimum_snr=1.0,
        gaussian_alpha=80.0,
    )

    assert np.count_nonzero(result.group_curve.valid_mask) > 0
    assert not np.any(result.phase_curve.valid_mask)


def test_frequency_domain_phase_ftan_recovers_the_absolute_branch_with_anchor() -> None:
    trace, theory_period, theory_phase, _ = _synthetic_dispersive_egf()
    result = pair_ftan(
        trace,
        dt_s=0.002,
        distance_m=600.0,
        frequency_range_hz=(2.0, 12.0),
        group_velocity_range_mps=(170.0, 330.0),
        phase_velocity_range_mps=(210.0, 370.0),
        period_count=72,
        velocity_step_mps=2.0,
        minimum_snr=1.0,
        gaussian_alpha=80.0,
        phase_method="frequency_domain",
        phase_minus_group_range_mps=(0.0, 100.0),
        phase_velocity_anchor=(0.45, float(np.interp(0.45, theory_period, theory_phase))),
    )

    valid = result.phase_curve.valid_mask
    expected = np.interp(result.phase_curve.period_s[valid], theory_period, theory_phase)
    error = np.abs(result.phase_curve.velocity_mps[valid] - expected)
    assert np.median(error) < 4.0
    assert np.percentile(error, 90.0) < 9.0


def test_biased_group_ridge_does_not_pull_phase_pick_off_its_phase_crests() -> None:
    period = np.linspace(1.0 / 12.0, 0.5, 48)
    velocity = np.arange(120.0, 501.0, 2.0)
    frequency = 1.0 / period
    phase_truth = 220.0 + 160.0 / (1.0 + (frequency / 4.0) ** 2)
    group_truth = phase_to_group_velocity(period, phase_truth)
    biased_group_pick = group_truth + 30.0
    distance_m = 600.0
    phase_image = np.clip(
        np.cos(
            2.0
            * np.pi
            * distance_m
            / period[None, :]
            * (1.0 / velocity[:, None] - 1.0 / phase_truth[None, :])
        ),
        0.0,
        None,
    )

    picked = _pick_phase_curve(
        phase_image,
        period,
        velocity,
        np.full(period.size, 100.0),
        biased_group_pick,
        anchor=(0.45, float(np.interp(0.45, period, phase_truth))),
        group_tolerance_mps=100.0,
        minimum_snr=2.0,
        smoothness=2.0,
        maximum_jump_mps=40.0,
        minimum_amplitude=0.0,
    )
    phase_cycle_error = np.abs(
        distance_m
        / period
        * (1.0 / picked.velocity_mps - 1.0 / phase_truth)
    )

    assert np.max(phase_cycle_error) < 0.1
