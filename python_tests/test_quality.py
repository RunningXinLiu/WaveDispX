import numpy as np

from wavedispx.processing import DispersionImage, DispersionPicks
from wavedispx.quality import (
    dispersion_quality_metrics,
    pick_error_metrics,
    theoretical_mode_support_metrics,
)


def test_quality_metrics_accept_a_clean_theoretical_ridge():
    frequency = np.linspace(5, 15, 51)
    velocity = np.arange(180.0, 361.0, 2.0)
    theory = 320.0 - 7.0 * (frequency - 5.0)
    energy = np.exp(
        -0.5 * ((velocity[:, None] - theory[None, :]) / 4.0) ** 2
    )
    image = DispersionImage(energy, energy, frequency, velocity, "phase_shift")
    metrics = dispersion_quality_metrics(
        image, frequency, theory, frequency_range_hz=(5, 15)
    )
    assert metrics["clean_ridge_pass"]
    assert metrics["dominant_ridge_fraction"] == 1.0
    picks = DispersionPicks(frequency, theory, np.ones_like(frequency), np.ones_like(frequency))
    errors = pick_error_metrics(picks, frequency, theory)
    assert errors["mae_mps"] == 0.0


def test_mode_support_reports_each_theoretical_branch():
    frequency = np.linspace(5, 15, 51)
    velocity = np.arange(180.0, 401.0, 2.0)
    theory = np.column_stack(
        (320.0 - 7.0 * (frequency - 5.0), 380.0 - 5.0 * (frequency - 5.0))
    )
    energy = np.maximum.reduce(
        [
            np.exp(-0.5 * ((velocity[:, None] - theory[:, mode][None, :]) / 4.0) ** 2)
            for mode in range(2)
        ]
    )
    image = DispersionImage(energy, energy, frequency, velocity, "phase_shift")
    metrics = theoretical_mode_support_metrics(image, frequency, theory)
    assert metrics["mode_0_median_energy"] > 0.9
    assert metrics["mode_1_fraction_ge_0p35"] == 1.0
