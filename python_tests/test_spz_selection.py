from dataclasses import replace
from pathlib import Path

import numpy as np

from wavedispx.config import load_config
from wavedispx.geometry import build_geometry
from wavedispx.spz_selection import (
    PairSPZConfig,
    blend_physical_and_data_weights,
    build_oracle_pair_spz_selection,
    combine_physical_and_data_weights,
    effective_window_count,
    optimize_one_sided_pair_spz_weights,
    optimize_pair_spz_weights,
    pair_spz_kernels,
)


ROOT = Path(__file__).resolve().parents[1]


def test_pair_spz_kernel_prefers_sources_beyond_pair_endpoints():
    positive, negative, pair_count = pair_spz_kernels(
        np.array((-20.0, 20.0, 0.0)),
        np.array((0.0, 0.0, 20.0)),
        np.array((-5.0, 5.0)),
        np.zeros(2),
        np.array((5.0,)),
        np.array((100.0,)),
        phase_tolerance_rad=np.pi / 2,
    )
    assert pair_count == 1
    summed = (positive + negative)[:, 0]
    np.testing.assert_allclose(summed[:2], 1.0, atol=1e-12)
    assert summed[2] < 0.15
    assert positive[:2, 0].sum() == 1.0
    assert negative[:2, 0].sum() == 1.0
    assert negative[0, 0] > positive[0, 0]
    assert positive[1, 0] > negative[1, 0]


def test_pair_spz_optimizer_combines_complementary_one_sided_windows():
    positive = np.array(((10.0,), (0.0,), (0.2,), (0.2,)))
    negative = np.array(((0.0,), (10.0,), (0.2,), (0.2,)))
    off = np.array(((0.0,), (0.0,), (10.0,), (10.0,)))
    config = PairSPZConfig(
        frequency_bands_hz=((4.0, 6.0),),
        minimum_effective_fraction=0.5,
        maximum_iterations=100,
    )
    weights, success, _ = optimize_pair_spz_weights(
        positive, negative, off, config=config
    )
    assert success
    assert np.min(weights[:2]) > np.max(weights[2:])
    assert effective_window_count(weights) >= 2.0 - 1e-9
    weighted_positive = float(weights @ positive[:, 0])
    weighted_negative = float(weights @ negative[:, 0])
    np.testing.assert_allclose(weighted_positive, weighted_negative, rtol=1e-6)


def test_one_sided_pair_spz_optimizer_rejects_opposite_branch_windows():
    target = np.array((10.0, 8.0, 0.1, 0.1))
    opposite = np.array((0.0, 0.0, 10.0, 8.0))
    off = np.array((0.0, 0.2, 0.0, 0.2))
    config = PairSPZConfig(
        frequency_bands_hz=((4.0, 6.0),),
        minimum_effective_fraction=0.5,
        maximum_iterations=100,
    )
    weights, success, _ = optimize_one_sided_pair_spz_weights(
        target, opposite, off, config=config
    )
    assert success
    assert np.min(weights[:2]) > np.max(weights[2:])
    assert effective_window_count(weights) >= 2.0 - 1e-9


def test_combined_weights_preserve_requested_effective_count_floor():
    data = np.array((8.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    physical = np.array((1.0, 1.0, 8.0, 1.0, 1.0, 1.0))
    combined = combine_physical_and_data_weights(
        data, physical, minimum_effective_fraction=0.75
    )
    np.testing.assert_allclose(np.mean(combined), 1.0)
    assert effective_window_count(combined) >= 4.5 - 1e-9


def test_log_space_blend_has_exact_endpoints_and_validates_fraction():
    data = np.asarray((0.5, 1.0, 2.0, 0.75))
    physical = np.asarray((2.0, 1.5, 0.5, 1.0))
    data_endpoint = blend_physical_and_data_weights(
        data,
        physical,
        physical_fraction=0.0,
        minimum_effective_fraction=0.25,
    )
    physical_endpoint = blend_physical_and_data_weights(
        data,
        physical,
        physical_fraction=1.0,
        minimum_effective_fraction=0.25,
    )
    np.testing.assert_allclose(data_endpoint, data / np.mean(data))
    np.testing.assert_allclose(physical_endpoint, physical / np.mean(physical))
    with np.testing.assert_raises_regex(ValueError, "physical_fraction"):
        blend_physical_and_data_weights(
            data,
            physical,
            physical_fraction=1.1,
            minimum_effective_fraction=0.25,
        )


def test_oracle_pair_spz_plan_replays_windows_and_has_exact_neff_control():
    base = load_config(ROOT / "examples/passive_source_selection_quick.toml")
    config = replace(base, noise=replace(base.noise, duration_s=60.0))
    geometry = build_geometry(config.receivers, config.ambient_sources)
    sample_count = int(round(config.noise.window_s / config.noise.dt_s))
    frequency = np.fft.rfftfreq(sample_count, config.noise.dt_s)
    selected = (frequency >= config.noise.minimum_frequency_hz) & (
        frequency <= config.noise.maximum_frequency_hz
    )
    plan = build_oracle_pair_spz_selection(
        config,
        geometry,
        frequency[selected],
        config=PairSPZConfig(
            frequency_bands_hz=config.source_selection.frequency_bands_hz,
            maximum_iterations=50,
        ),
    )
    assert plan.scope == "pair_frequency_stationary_phase_known_source"
    assert plan.weights.shape == (6,)
    assert plan.band_weights.shape == (6, 2)
    assert plan.band_random_neff_weights.shape == (6, 2)
    assert plan.band_effective_window_count.shape == (2,)
    assert plan.positive_branch_band_weights.shape == (6, 2)
    assert plan.negative_branch_band_weights.shape == (6, 2)
    assert plan.branch_band_effective_window_count.shape == (2, 2)
    assert plan.pair_positive_branch_weights.shape == (6, plan.pair_count)
    assert plan.pair_negative_branch_weights.shape == (6, plan.pair_count)
    assert plan.pair_branch_effective_window_count.shape == (2, plan.pair_count)
    assert plan.pair_positive_branch_band_weights.shape == (
        6,
        2,
        plan.pair_count,
    )
    assert plan.pair_negative_branch_band_weights.shape == (
        6,
        2,
        plan.pair_count,
    )
    assert plan.pair_branch_band_effective_window_count.shape == (
        2,
        2,
        plan.pair_count,
    )
    assert plan.spz_fraction.shape == (6, 2)
    assert plan.pair_count == geometry.receiver_x_m.size - 1
    np.testing.assert_allclose(
        np.sort(plan.weights), np.sort(plan.random_neff_weights)
    )
    for band_index in range(2):
        np.testing.assert_allclose(
            np.sort(plan.band_weights[:, band_index]),
            np.sort(plan.band_random_neff_weights[:, band_index]),
        )
        np.testing.assert_allclose(
            np.sort(plan.positive_branch_band_weights[:, band_index]),
            np.sort(plan.positive_branch_band_random_weights[:, band_index]),
        )
        np.testing.assert_allclose(
            np.sort(plan.negative_branch_band_weights[:, band_index]),
            np.sort(plan.negative_branch_band_random_weights[:, band_index]),
        )
    assert plan.effective_window_count >= 3.0 - 1e-9
    for pair_index in range(plan.pair_count):
        np.testing.assert_allclose(
            np.sort(plan.pair_positive_branch_weights[:, pair_index]),
            np.sort(plan.pair_positive_branch_random_weights[:, pair_index]),
        )
        np.testing.assert_allclose(
            np.sort(plan.pair_negative_branch_weights[:, pair_index]),
            np.sort(plan.pair_negative_branch_random_weights[:, pair_index]),
        )
        for band_index in range(2):
            np.testing.assert_allclose(
                np.sort(
                    plan.pair_positive_branch_band_weights[
                        :, band_index, pair_index
                    ]
                ),
                np.sort(
                    plan.pair_positive_branch_band_random_weights[
                        :, band_index, pair_index
                    ]
                ),
            )
            np.testing.assert_allclose(
                np.sort(
                    plan.pair_negative_branch_band_weights[
                        :, band_index, pair_index
                    ]
                ),
                np.sort(
                    plan.pair_negative_branch_band_random_weights[
                        :, band_index, pair_index
                    ]
                ),
            )
