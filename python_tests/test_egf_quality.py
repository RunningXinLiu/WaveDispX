import numpy as np

from wavedispx.egf_reference import PassiveEgfReference
from wavedispx.processing import NCFGather
from wavedispx.quality import egf_equivalence_metrics


def _reference_and_ncf(delay_samples: int = 0):
    dt = 0.01
    time = np.arange(256) * dt
    reference_signal = (
        np.exp(-((time - 0.4) / 0.08) ** 2)
        * np.cos(2.0 * np.pi * 8.0 * (time - 0.4))
    )
    candidate = np.roll(reference_signal, delay_samples)
    if delay_samples:
        candidate[:delay_samples] = 0.0
    reference = PassiveEgfReference(
        time_s=time,
        distance_m=np.array([40.0]),
        components=reference_signal[:, None, None],
        component_names=("zz",),
        pair_i=np.array([0]),
        pair_j=np.array([1]),
        pair_distance_index=np.array([0]),
        source_time_function="unit_impulse",
        calibration_operator="identity",
        delay_removed_s=0.0,
        maximum_frequency_hz=20.0,
        build_seconds=0.0,
    )
    ncf = NCFGather(
        data=candidate[:, None],
        lag_time_s=time,
        distance_m=np.array([40.0]),
        pair_count=np.array([1]),
        valid_windows=10,
    )
    return reference, ncf


def test_egf_metrics_are_exact_for_an_identical_unshifted_waveform():
    reference, ncf = _reference_and_ncf()
    metrics = egf_equivalence_metrics(
        ncf,
        reference,
        component_name="zz",
        frequency_range_hz=(4.0, 14.0),
        minimum_velocity_mps=50.0,
        maximum_velocity_mps=500.0,
        correlation_normalization="none",
    )
    assert metrics["waveform_correlation_no_shift"] > 1.0 - 1e-14
    assert metrics["normalized_l2_unit_norm"] < 1e-14
    assert metrics["scaled_waveform_l2"] < 1e-14
    assert metrics["phase_rms_rad"] < 1e-14
    assert abs(metrics["median_envelope_peak_time_bias_s"]) < 1e-14
    assert not metrics["absolute_amplitude_available"]
    assert np.isnan(metrics["absolute_amplitude_log_rms"])


def test_egf_metrics_do_not_hide_a_time_shift_with_alignment():
    reference, ncf = _reference_and_ncf(delay_samples=3)
    metrics = egf_equivalence_metrics(
        ncf,
        reference,
        component_name="zz",
        frequency_range_hz=(4.0, 14.0),
        minimum_velocity_mps=50.0,
        maximum_velocity_mps=500.0,
        correlation_normalization="spectral_whitening",
    )
    assert metrics["median_envelope_peak_time_bias_s"] == 0.03
    assert metrics["waveform_correlation_no_shift"] < 0.5
    assert metrics["phase_rms_rad"] > 0.5
    assert metrics["time_shift_optimized"] is False


def test_interferometric_derivative_operator_recovers_reference_phase():
    reference, ncf = _reference_and_ncf()
    reference_signal = reference.components[:, 0, 0]
    integrated = -np.cumsum(reference_signal) * 0.01
    derivative_ncf = NCFGather(
        integrated[:, None],
        ncf.lag_time_s,
        ncf.distance_m,
        ncf.pair_count,
        ncf.valid_windows,
    )
    calibrated_reference = PassiveEgfReference(
        **{
            **reference.__dict__,
            "calibration_operator": "negative_time_derivative_of_ncf",
        }
    )
    metrics = egf_equivalence_metrics(
        derivative_ncf,
        calibrated_reference,
        component_name="zz",
        frequency_range_hz=(4.0, 14.0),
        minimum_velocity_mps=50.0,
        maximum_velocity_mps=500.0,
        correlation_normalization="none",
    )
    assert metrics["waveform_correlation_no_shift"] > 0.95
    assert metrics["phase_rms_rad"] < 0.3
