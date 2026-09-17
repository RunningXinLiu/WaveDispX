from dataclasses import replace
from pathlib import Path

import numpy as np

from wavedispx.config import load_config
from wavedispx.geometry import build_geometry
from wavedispx.source_selection import (
    build_oracle_axis_selection,
    score_oracle_axis_window,
)


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_axis_score_distinguishes_balanced_endfire_and_off_axis_power():
    frequency = np.array([4.0, 6.0, 8.0])
    source_x = np.array([-20.0, 20.0, 0.0])
    source_y = np.array([0.0, 0.0, 20.0])
    receiver_x = np.array([-5.0, 5.0])
    receiver_y = np.zeros(2)
    coefficients = np.zeros((3, frequency.size), dtype=np.complex128)
    coefficients[:2] = 1.0

    balanced = score_oracle_axis_window(
        coefficients,
        frequency,
        source_x,
        source_y,
        receiver_x,
        receiver_y,
        frequency_bands_hz=((3.0, 7.0), (7.0, 9.0)),
        stationary_phase_half_angle_deg=15.0,
        hard_min_endfire_fraction=0.8,
        hard_min_bidirectional_balance=0.8,
    )
    np.testing.assert_allclose(balanced.endfire_fraction, 1.0)
    np.testing.assert_allclose(balanced.bidirectional_balance, 1.0)
    np.testing.assert_allclose(balanced.off_axis_fraction, 0.0)
    np.testing.assert_allclose(balanced.effective_source_count, 2.0)
    assert balanced.soft_weight == 1.0
    assert balanced.hard_selected

    coefficients[:2] = 0.0
    coefficients[2] = 1.0
    off_axis = score_oracle_axis_window(
        coefficients,
        frequency,
        source_x,
        source_y,
        receiver_x,
        receiver_y,
        frequency_bands_hz=((3.0, 9.0),),
        stationary_phase_half_angle_deg=15.0,
        hard_min_endfire_fraction=0.8,
        hard_min_bidirectional_balance=0.8,
    )
    np.testing.assert_allclose(off_axis.endfire_fraction, 0.0)
    np.testing.assert_allclose(off_axis.off_axis_fraction, 1.0)
    assert off_axis.soft_weight == 0.0
    assert not off_axis.hard_selected


def test_oracle_selection_plan_has_exact_random_budget_and_neff_controls():
    base = load_config(ROOT / "examples/passive_source_selection_quick.toml")
    config = replace(
        base,
        noise=replace(base.noise, duration_s=120.0),
    )
    geometry = build_geometry(config.receivers, config.ambient_sources)
    sample_count = int(round(config.noise.window_s / config.noise.dt_s))
    full_frequency = np.fft.rfftfreq(sample_count, config.noise.dt_s)
    selected = (
        (full_frequency >= config.noise.minimum_frequency_hz)
        & (full_frequency <= config.noise.maximum_frequency_hz)
    )

    plan = build_oracle_axis_selection(config, geometry, full_frequency[selected])

    assert plan.scope == "array_axis_approximation"
    assert plan.endfire_fraction.shape == (12, 2)
    assert plan.bidirectional_balance.shape == (12, 2)
    assert plan.correlated_source_risk is False
    assert 0 < np.count_nonzero(plan.hard_selected) < plan.window_index.size
    weights = plan.strategy_weights
    assert set(weights) == {
        "oracle_hard_mask",
        "oracle_soft_weight",
        "random_budget_control",
        "random_neff_control",
    }
    assert weights["oracle_hard_mask"].sum() == weights["random_budget_control"].sum()
    soft = weights["oracle_soft_weight"]
    random_soft = weights["random_neff_control"]
    np.testing.assert_allclose(soft.sum(), random_soft.sum(), rtol=1e-15)
    np.testing.assert_allclose(
        np.square(soft).sum(), np.square(random_soft).sum(), rtol=1e-15
    )


def test_source_selection_config_defaults_off_and_example_enables_oracle():
    default = load_config(ROOT / "examples/passive_quick.toml")
    enabled = load_config(ROOT / "examples/passive_source_selection_quick.toml")
    assert not default.source_selection.enabled
    assert enabled.source_selection.enabled
    assert enabled.source_selection.objective == "stationary_phase"
    assert enabled.source_selection.frequency_bands_hz == (
        (3.5, 6.5),
        (6.5, 9.5),
    )


def test_anisotropic_activity_schedule_produces_predeclared_good_windows():
    config = load_config(
        ROOT / "examples/passive_source_selection_anisotropic_quick.toml"
    )
    geometry = build_geometry(config.receivers, config.ambient_sources)
    sample_count = int(round(config.noise.window_s / config.noise.dt_s))
    full_frequency = np.fft.rfftfreq(sample_count, config.noise.dt_s)
    selected = (
        (full_frequency >= config.noise.minimum_frequency_hz)
        & (full_frequency <= config.noise.maximum_frequency_hz)
    )
    plan = build_oracle_axis_selection(config, geometry, full_frequency[selected])
    expected_good = np.flatnonzero(
        np.isin(np.arange(60) % 6, np.array([0, 3, 5]))
    )
    expected_bad = np.setdiff1d(np.arange(60), expected_good)
    np.testing.assert_array_equal(np.flatnonzero(plan.hard_selected), expected_good)
    assert np.all(plan.soft_weight[expected_good] > 0.8)
    np.testing.assert_allclose(plan.soft_weight[expected_bad], 0.0)
