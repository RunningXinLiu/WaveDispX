import numpy as np

from wavedispx.pair_tau_p_selection import (
    fit_pair_crossfit_tau_p_weights,
    joint_diffuse_prearrival_hard_weights,
)
from wavedispx.processing import NCFPairBranches, NCFPairGather


def _window(slowness_s_per_m: float, seed: int) -> NCFPairBranches:
    dt_s = 0.01
    lag = np.arange(201) * dt_s
    distance = np.asarray((2.0, 4.0, 6.0, 8.0))
    rng = np.random.default_rng(seed)
    data = np.column_stack(
        [
            np.exp(-0.5 * np.square((lag - slowness_s_per_m * value) / 0.08))
            * np.cos(2.0 * np.pi * 2.0 * (lag - slowness_s_per_m * value))
            + 0.01 * rng.standard_normal(lag.size)
            for value in distance
        ]
    )
    gather = NCFPairGather(
        data=data,
        lag_time_s=lag,
        pair_i=np.zeros(distance.size, dtype=np.int64),
        pair_j=np.arange(1, distance.size + 1, dtype=np.int64),
        distance_m=distance,
        valid_windows=1,
    )
    return NCFPairBranches(gather, gather, gather)


def test_pair_tau_p_crossfit_prefers_training_supported_slowness():
    good = 0.10
    bad = 0.045
    slowness = (good, bad, good, good, good, bad, good, good)
    result = fit_pair_crossfit_tau_p_weights(
        [_window(value, index) for index, value in enumerate(slowness)],
        dt_s=0.01,
        frequency_bands_hz=((1.0, 3.0),),
        minimum_velocity_mps=8.0,
        maximum_velocity_mps=30.0,
        phase_tolerance_rad=np.pi,
        folds=4,
        minimum_effective_fraction=0.5,
        slowness_count=64,
    )
    assert result.positive_band_weights.shape == (8, 1, 4)
    assert result.negative_band_weights.shape == (8, 1, 4)
    assert result.directional_imbalance.shape == (8, 1)
    assert result.diffuse_soft_band_weights.shape == (8, 1)
    assert result.diffuse_hard_band_weights.shape == (8, 1)
    assert np.count_nonzero(result.diffuse_hard_band_weights[:, 0]) == 4
    assert result.diffuse_hard_effective_window_count[0] == 4.0
    assert result.fold_index.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert np.allclose(result.reference_slowness_s_per_m, good, atol=0.015)
    good_index = np.asarray(slowness) == good
    bad_index = ~good_index
    assert np.median(result.positive_band_weights[good_index]) > np.median(
        result.positive_band_weights[bad_index]
    )
    assert np.min(result.positive_effective_window_count) >= 4.0 - 1e-10
    assert np.min(result.negative_effective_window_count) >= 4.0 - 1e-10


def test_pair_tau_p_rejects_single_distance_gather():
    window = _window(0.1, 0)
    gather = NCFPairGather(
        data=window.positive.data[:, :1],
        lag_time_s=window.positive.lag_time_s,
        pair_i=window.positive.pair_i[:1],
        pair_j=window.positive.pair_j[:1],
        distance_m=window.positive.distance_m[:1],
        valid_windows=1,
    )
    with np.testing.assert_raises_regex(ValueError, "multi-distance"):
        fit_pair_crossfit_tau_p_weights(
            [NCFPairBranches(gather, gather, gather)] * 2,
            dt_s=0.01,
            frequency_bands_hz=((1.0, 3.0),),
            minimum_velocity_mps=8.0,
            maximum_velocity_mps=30.0,
            phase_tolerance_rad=np.pi,
            folds=2,
            minimum_effective_fraction=0.5,
        )


def test_joint_hard_selector_balances_direction_and_prearrival_ranks():
    first_band = np.asarray((0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0))
    imbalance = np.column_stack((first_band, first_band[::-1]))
    prearrival = np.repeat(
        np.asarray((7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0))[:, None],
        3,
        axis=1,
    )

    weights = joint_diffuse_prearrival_hard_weights(
        imbalance,
        prearrival,
        prearrival,
        retention_fraction=0.5,
        directional_fraction=0.75,
    )

    assert weights.shape == (8, 2)
    np.testing.assert_array_equal(np.sum(weights, axis=0), (4.0, 4.0))
    assert np.mean(imbalance[weights[:, 0] > 0, 0]) < np.mean(imbalance[:, 0])
    direction_only = np.argsort(imbalance[:, 0], kind="stable")[:4]
    assert np.mean(prearrival[weights[:, 0] > 0]) < np.mean(prearrival[direction_only])
