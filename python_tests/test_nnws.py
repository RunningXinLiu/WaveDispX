import numpy as np
import pytest

from wavedispx.nnws import (
    NNWSConfig,
    apply_nnws_weights_to_windows,
    ccf_matrix_from_branches,
    pair_nnws_eligibility,
    stack_pair_ncf_windows,
    stack_pair_ncf_windows_frequency_weighted,
    train_nnws_ccfs,
    train_pairwise_nnws_strategy,
    train_nnws_window_strategy,
)
from wavedispx.nnws_v2 import (
    NNWSV2Config,
    apply_pair_weights_within_frequency_mask,
    blend_frequency_and_pair_weights,
    blend_shared_weights_with_directional_guard,
    build_moveout_aligned_proxy,
    enforce_effective_window_floor,
    train_shared_crossfit_nnws_v2,
)
from wavedispx.passive import _extract_pair_window_ncf
from wavedispx.processing import (
    NCFBranches,
    NCFGather,
    NCFPairBranches,
    NCFPairGather,
    bin_pair_ncf_branches,
)


def _window_branches(scale: float, *, asymmetric: bool = False):
    lag = np.arange(21, dtype=np.float64)
    distance = np.array([100.0, 200.0])
    pair_count = np.array([4, 3])
    pulse = np.exp(-0.5 * ((lag - 7.0) / 1.2) ** 2)
    positive_data = np.column_stack((scale * pulse, 2.0 * scale * pulse))
    negative_data = positive_data.copy()
    if asymmetric:
        negative_data[:, 0] += 3.0 * np.exp(-0.5 * ((lag - 3.0) / 0.8) ** 2)
        negative_data[:, 1] += 6.0 * np.exp(-0.5 * ((lag - 3.0) / 0.8) ** 2)

    def gather(data):
        return NCFGather(data, lag, distance, pair_count, valid_windows=1)

    return {
        "zz": NCFBranches(
            positive=gather(positive_data),
            negative=gather(negative_data),
            symmetric=gather(0.5 * (positive_data + negative_data)),
        )
    }


def _pair_window(scale: float, *, contaminate: bool = False):
    lag = np.arange(21, dtype=np.float64)
    pair_i = np.array([0, 0, 1])
    pair_j = np.array([1, 2, 2])
    distance = np.array([60.0, 62.0, 120.0])
    arrival = distance / 15.0
    positive = np.column_stack(
        [scale * np.exp(-0.5 * ((lag - time) / 1.0) ** 2) for time in arrival]
    )
    negative = positive.copy()
    if contaminate:
        negative[:, 1] += 2.5 * np.exp(-0.5 * ((lag - 2.0) / 0.7) ** 2)

    def gather(data):
        return NCFPairGather(
            data=data,
            lag_time_s=lag,
            pair_i=pair_i,
            pair_j=pair_j,
            distance_m=distance,
            valid_windows=1,
        )

    return NCFPairBranches(
        positive=gather(positive),
        negative=gather(negative),
        symmetric=gather(0.5 * (positive + negative)),
    )


def test_ccf_matrix_reconstructs_conventional_full_lag_order():
    windows = [_window_branches(1.0), _window_branches(2.0, asymmetric=True)]
    matrix, lag, distance = ccf_matrix_from_branches(
        windows, component="zz", distance_index=0
    )
    assert matrix.shape == (2, 41)
    np.testing.assert_array_equal(lag, np.arange(-20.0, 21.0))
    assert distance == 100.0
    np.testing.assert_allclose(matrix[0, :20], matrix[0, 21:][::-1])
    assert matrix[1, 17] > matrix[1, 23]


def test_applying_nnws_weights_uses_existing_auditable_accumulator():
    windows = [_window_branches(1.0), _window_branches(3.0)]
    result = apply_nnws_weights_to_windows(windows, np.array([1.0, 0.5]))
    np.testing.assert_allclose(
        result.component_branches["zz"].positive.data[:, 0],
        (1.0 * windows[0]["zz"].positive.data[:, 0]
         + 0.5 * windows[1]["zz"].positive.data[:, 0]) / 1.5,
    )
    assert result.statistics.sum_weight == 1.5
    assert result.statistics.effective_window_count == pytest.approx(1.8)


def test_upstream_nnws_training_is_reproducible_and_reports_neff():
    pytest.importorskip("torch")
    windows = [
        _window_branches(1.0 + 0.05 * index, asymmetric=index >= 6)
        for index in range(8)
    ]
    matrix, _, _ = ccf_matrix_from_branches(
        windows, component="zz", distance_index=0
    )
    options = dict(
        dt_s=1.0,
        signal_window_s=(6.0, 9.0),
        maximum_period_s=2.0,
        config=NNWSConfig(epochs=3, seed=7, device="cpu"),
    )
    first = train_nnws_ccfs(matrix, **options)
    second = train_nnws_ccfs(matrix, **options)
    np.testing.assert_allclose(first.weights, second.weights, rtol=0, atol=0)
    np.testing.assert_allclose(
        first.stack,
        np.average(matrix, axis=0, weights=first.weights),
        rtol=1e-6,
        atol=1e-7,
    )
    assert 1.0 <= first.effective_window_count <= len(windows)
    assert len(first.history["total_loss"]) == 3


def test_window_strategy_keeps_array_proxy_scope_explicit():
    pytest.importorskip("torch")
    windows = [
        _window_branches(1.0 + 0.1 * index, asymmetric=index >= 2)
        for index in range(4)
    ]
    result = train_nnws_window_strategy(
        windows,
        component="zz",
        distance_index=0,
        dt_s=1.0,
        signal_window_s=(6.0, 9.0),
        maximum_period_s=2.0,
        config=NNWSConfig(epochs=2, seed=3, device="cpu"),
    )
    assert result.scope == "array_proxy_from_one_distance_bin"
    assert result.calibration_distance_m == 100.0
    assert result.strategy_stack.statistics.windows_seen == 4


def test_low_diversity_upstream_failure_is_rejected_before_accumulation():
    pytest.importorskip("torch")
    windows = [_window_branches(1.0 + 0.1 * index) for index in range(4)]
    matrix, _, _ = ccf_matrix_from_branches(
        windows, component="zz", distance_index=0
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        train_nnws_ccfs(
            matrix,
            dt_s=1.0,
            signal_window_s=(6.0, 9.0),
            maximum_period_s=2.0,
            config=NNWSConfig(epochs=1, seed=3, device="cpu"),
        )


def test_population_epsilon_std_ablation_stays_finite_for_low_diversity_ccfs():
    pytest.importorskip("torch")
    windows = [_window_branches(1.0 + 0.1 * index) for index in range(4)]
    matrix, _, _ = ccf_matrix_from_branches(
        windows, component="zz", distance_index=0
    )
    result = train_nnws_ccfs(
        matrix,
        dt_s=1.0,
        signal_window_s=(6.0, 9.0),
        maximum_period_s=2.0,
        config=NNWSConfig(
            epochs=5,
            seed=3,
            device="cpu",
            std_mode="population_eps",
        ),
    )
    assert np.isfinite(result.weights).all()
    assert np.isfinite(result.history["total_loss"]).all()
    assert 1.0 <= result.effective_window_count <= len(windows)


def test_pair_window_unit_weights_reproduce_the_linear_pair_stack():
    windows = [_pair_window(1.0), _pair_window(2.0), _pair_window(4.0)]
    weights = np.ones((len(windows), windows[0].distance_m.size))
    result = stack_pair_ncf_windows(windows, weights)

    for branch_name in ("positive", "negative", "symmetric"):
        expected = np.mean(
            [getattr(window, branch_name).data for window in windows], axis=0
        )
        np.testing.assert_allclose(
            getattr(result.branches, branch_name).data,
            expected,
            rtol=0,
            atol=0,
        )
    np.testing.assert_allclose(result.statistics.effective_window_count, 3.0)
    assert result.statistics.sufficient.all()
    assert result.branches.valid_windows == 3


def test_extract_pair_window_ncf_preserves_one_pair_before_distance_binning():
    windows = [_pair_window(1.0), _pair_window(2.0), _pair_window(4.0)]
    extracted = _extract_pair_window_ncf(
        windows,
        reference_receiver_index=0,
        target_receiver_index=2,
    )
    assert extracted is not None
    assert (extracted.pair_i, extracted.pair_j) == (0, 2)
    assert extracted.distance_m == 62.0
    assert extracted.symmetric.shape == (3, 21)
    np.testing.assert_allclose(
        extracted.symmetric,
        np.stack([window.symmetric.data[:, 1] for window in windows]).astype(
            np.float32
        ),
        rtol=0,
        atol=0,
    )


def test_frequency_weighted_pair_stack_matches_scalar_when_bands_agree():
    windows = [_pair_window(1.0), _pair_window(2.0), _pair_window(4.0)]
    scalar = np.asarray((1.0, 0.5, 2.0))
    scalar_result = stack_pair_ncf_windows(
        windows,
        np.repeat(scalar[:, None], windows[0].distance_m.size, axis=1),
    )
    frequency_result = stack_pair_ncf_windows_frequency_weighted(
        windows,
        np.repeat(scalar[:, None], 2, axis=1),
        np.asarray((0.1, 0.4)),
        dt_s=1.0,
    )
    for branch_name in ("positive", "negative", "symmetric"):
        np.testing.assert_allclose(
            getattr(frequency_result.branches, branch_name).data,
            getattr(scalar_result.branches, branch_name).data,
            rtol=1e-12,
            atol=1e-12,
        )


def test_frequency_weighted_pair_stack_accepts_branch_specific_weights():
    windows = [_pair_window(1.0), _pair_window(2.0), _pair_window(4.0)]
    positive = np.asarray((1.0, 0.5, 2.0))
    negative = np.asarray((2.0, 0.5, 1.0))
    result = stack_pair_ncf_windows_frequency_weighted(
        windows,
        np.repeat(positive[:, None], 2, axis=1),
        np.asarray((0.1, 0.4)),
        dt_s=1.0,
        negative_band_weights=np.repeat(negative[:, None], 2, axis=1),
    )
    expected_positive = np.average(
        [window.positive.data for window in windows],
        axis=0,
        weights=positive,
    )
    expected_negative = np.average(
        [window.negative.data for window in windows],
        axis=0,
        weights=negative,
    )
    np.testing.assert_allclose(
        result.branches.positive.data, expected_positive, atol=1e-12
    )
    np.testing.assert_allclose(
        result.branches.negative.data, expected_negative, atol=1e-12
    )
    np.testing.assert_allclose(
        result.branches.symmetric.data,
        0.5 * (expected_positive + expected_negative),
        atol=1e-12,
    )


def test_pair_stack_accepts_pair_and_branch_specific_weights():
    windows = [_pair_window(1.0), _pair_window(2.0), _pair_window(4.0)]
    positive = np.asarray(
        ((1.0, 2.0, 1.0), (0.5, 0.5, 1.0), (2.0, 1.0, 1.0))
    )
    negative = positive[::-1]
    result = stack_pair_ncf_windows(
        windows, positive, negative_weights=negative
    )
    positive_cube = np.stack([window.positive.data for window in windows])
    negative_cube = np.stack([window.negative.data for window in windows])
    expected_positive = np.einsum("wp,wlp->lp", positive, positive_cube)
    expected_positive /= positive.sum(axis=0)[None, :]
    expected_negative = np.einsum("wp,wlp->lp", negative, negative_cube)
    expected_negative /= negative.sum(axis=0)[None, :]
    np.testing.assert_allclose(result.branches.positive.data, expected_positive)
    np.testing.assert_allclose(result.branches.negative.data, expected_negative)
    np.testing.assert_allclose(
        result.branches.symmetric.data,
        0.5 * (expected_positive + expected_negative),
    )


def test_frequency_stack_accepts_pair_branch_band_weights():
    windows = [_pair_window(1.0), _pair_window(2.0), _pair_window(4.0)]
    positive = np.asarray(
        ((1.0, 2.0, 1.0), (0.5, 0.5, 1.0), (2.0, 1.0, 1.0))
    )
    negative = positive[::-1]
    scalar = stack_pair_ncf_windows(
        windows, positive, negative_weights=negative
    )
    frequency = stack_pair_ncf_windows_frequency_weighted(
        windows,
        np.repeat(positive[:, None, :], 2, axis=1),
        np.asarray((0.1, 0.4)),
        dt_s=1.0,
        negative_band_weights=np.repeat(negative[:, None, :], 2, axis=1),
    )
    for branch_name in ("positive", "negative", "symmetric"):
        np.testing.assert_allclose(
            getattr(frequency.branches, branch_name).data,
            getattr(scalar.branches, branch_name).data,
            atol=1e-12,
        )


def test_frequency_stack_preserves_zero_weight_hard_mask():
    windows = [
        _pair_window(1.0),
        _pair_window(2.0),
        _pair_window(4.0),
        _pair_window(8.0),
    ]
    hard_mask = np.asarray((1.0, 0.0, 1.0, 0.0))
    result = stack_pair_ncf_windows_frequency_weighted(
        windows,
        np.repeat(hard_mask[:, None], 2, axis=1),
        np.asarray((0.1, 0.4)),
        dt_s=1.0,
        minimum_effective_windows=2.0,
    )
    for branch_name in ("positive", "negative", "symmetric"):
        expected = np.mean(
            [getattr(windows[index], branch_name).data for index in (0, 2)],
            axis=0,
        )
        np.testing.assert_allclose(
            getattr(result.branches, branch_name).data,
            expected,
            rtol=0,
            atol=1e-12,
        )
    np.testing.assert_array_equal(result.statistics.nonzero_window_count, 2)
    np.testing.assert_allclose(result.statistics.effective_window_count, 2.0)
    assert result.branches.valid_windows == 2
    assert result.statistics.sufficient.all()


def test_nnws_v2_final_weights_enforce_configured_neff_floor():
    weights = np.asarray((10.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    constrained = enforce_effective_window_floor(weights, 0.75)
    effective = constrained.sum() ** 2 / np.square(constrained).sum()
    np.testing.assert_allclose(np.mean(constrained), 1.0)
    assert effective >= 4.5 - 1e-10
    assert np.linalg.norm(constrained - weights / np.mean(weights)) > 0
    windows = [_pair_window(float(index + 1)) for index in range(6)]
    stacked = stack_pair_ncf_windows(
        windows,
        np.repeat(constrained[:, None], windows[0].distance_m.size, axis=1),
        minimum_effective_windows=4.5,
    )
    assert stacked.statistics.sufficient.all()


def test_v2_directional_guard_overrides_misdirected_shared_weights_and_keeps_neff():
    shared = np.asarray((2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 0.5, 0.5))
    directional = np.asarray(
        (
            (0.25, 0.30),
            (0.30, 0.25),
            (2.00, 1.80),
            (1.80, 2.00),
            (0.25, 0.30),
            (0.30, 0.25),
            (2.00, 1.80),
            (1.80, 2.00),
        )
    )

    guarded = blend_shared_weights_with_directional_guard(
        shared,
        directional,
        directional_fraction=0.8,
        minimum_effective_fraction=0.5,
    )

    assert guarded.shape == directional.shape
    assert np.all(
        np.mean(guarded[[2, 3, 6, 7]], axis=0)
        > 2.0 * np.mean(guarded[[0, 1, 4, 5]], axis=0)
    )
    for band in range(guarded.shape[1]):
        effective = guarded[:, band].sum() ** 2 / np.square(
            guarded[:, band]
        ).sum()
        assert effective >= 4.0 - 1e-10
        assert np.mean(guarded[:, band]) == pytest.approx(1.0)


def test_v2_directional_guard_preserves_pair_branch_frequency_shape():
    shared = np.asarray((1.5, 0.5, 1.5, 0.5, 1.5, 0.5))
    directional = np.ones((6, 2, 3), dtype=np.float64)
    directional[1::2] = 3.0

    guarded = blend_shared_weights_with_directional_guard(
        shared,
        directional,
        directional_fraction=0.8,
        minimum_effective_fraction=0.5,
    )

    assert guarded.shape == directional.shape
    assert np.all(np.mean(guarded[1::2], axis=0) > np.mean(guarded[::2], axis=0))
    effective = np.square(np.sum(guarded, axis=0)) / np.sum(
        np.square(guarded), axis=0
    )
    assert np.all(effective >= 3.0 - 1e-10)
    np.testing.assert_allclose(np.mean(guarded, axis=0), 1.0)


def test_frequency_pair_blend_preserves_neff_and_broadcasts_axes():
    frequency = np.asarray(
        ((2.0, 0.5), (2.0, 0.5), (0.5, 2.0), (0.5, 2.0))
    )
    pair = np.asarray(
        ((2.0, 0.5, 1.0), (0.5, 2.0, 1.0), (2.0, 0.5, 1.0), (0.5, 2.0, 1.0))
    )

    blended = blend_frequency_and_pair_weights(
        frequency,
        pair,
        frequency_fraction=0.7,
        minimum_effective_fraction=0.5,
    )

    assert blended.shape == (4, 2, 3)
    np.testing.assert_allclose(np.mean(blended, axis=0), 1.0)
    effective = np.square(np.sum(blended, axis=0)) / np.sum(
        np.square(blended), axis=0
    )
    assert np.all(effective >= 2.0 - 1e-10)


def test_pair_reweighting_never_reintroduces_masked_windows():
    mask = np.asarray(
        ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 1.0))
    )
    pair = np.asarray(
        ((10.0, 1.0), (1.0, 10.0), (10.0, 1.0), (1.0, 10.0), (1.0, 1.0))
    )

    weights = apply_pair_weights_within_frequency_mask(
        mask,
        pair,
        minimum_effective_support_fraction=0.75,
    )

    assert weights.shape == (5, 2, 2)
    for band_index in range(mask.shape[1]):
        np.testing.assert_array_equal(
            weights[mask[:, band_index] == 0.0, band_index],
            0.0,
        )
        support_count = np.count_nonzero(mask[:, band_index])
        effective = np.square(np.sum(weights[:, band_index], axis=0)) / np.sum(
            np.square(weights[:, band_index]), axis=0
        )
        assert np.all(effective >= 0.75 * support_count - 1e-10)


def test_pair_binning_excludes_failed_pairs_without_linear_fallback():
    windows = [_pair_window(1.0), _pair_window(2.0), _pair_window(4.0)]
    weights = np.ones((len(windows), windows[0].distance_m.size))
    weights[:, 0] = 0.0
    pair_stack = stack_pair_ncf_windows(windows, weights)
    binned = bin_pair_ncf_branches(pair_stack.branches, distance_bin_m=60.0)

    assert not pair_stack.statistics.sufficient[0]
    np.testing.assert_array_equal(binned.symmetric.pair_count, [1, 1])
    assert np.isfinite(binned.symmetric.data).all()


def test_pairwise_nnws_trains_each_pair_and_returns_binned_ncf():
    pytest.importorskip("torch")
    windows = [
        _pair_window(1.0 + 0.05 * index, contaminate=index >= 5)
        for index in range(8)
    ]
    result = train_pairwise_nnws_strategy(
        windows,
        dt_s=1.0,
        distance_bin_m=60.0,
        minimum_velocity_mps=10.0,
        maximum_velocity_mps=20.0,
        maximum_period_s=2.0,
        config=NNWSConfig(
            epochs=3,
            seed=11,
            device="cpu",
            std_mode="population_eps",
        ),
    )

    assert result.scope == "pair_resolved"
    assert result.pair_stack.statistics.weights.shape == (8, 3)
    assert np.isfinite(result.pair_stack.statistics.weights).all()
    assert result.pair_stack.statistics.sufficient.all()
    assert result.failure_reason == ("", "", "")
    assert result.eligibility_reason == ("", "", "")
    assert result.eligible.all()
    np.testing.assert_array_equal(result.binned_branches.symmetric.pair_count, [2, 1])
    np.testing.assert_array_equal(
        result.common_linear_binned.symmetric.distance_m,
        result.binned_branches.symmetric.distance_m,
    )
    np.testing.assert_array_equal(
        result.random_control_binned.symmetric.distance_m,
        result.binned_branches.symmetric.distance_m,
    )
    np.testing.assert_array_equal(
        result.hybrid_full_aperture_binned.symmetric.distance_m,
        result.binned_branches.symmetric.distance_m,
    )
    successful = result.pair_stack.statistics.sufficient
    np.testing.assert_allclose(
        result.common_linear_stack.statistics.weights[:, successful], 1.0
    )
    for pair_index in np.flatnonzero(successful):
        np.testing.assert_allclose(
            np.sort(result.random_control_stack.statistics.weights[:, pair_index]),
            np.sort(result.pair_stack.statistics.weights[:, pair_index]),
        )
        np.testing.assert_allclose(
            result.hybrid_full_aperture_stack.statistics.weights[:, pair_index],
            result.pair_stack.statistics.weights[:, pair_index],
        )
        np.testing.assert_allclose(
            result.hybrid_random_control_stack.statistics.weights[:, pair_index],
            result.random_control_stack.statistics.weights[:, pair_index],
        )


def test_nnws_v2_uses_contiguous_oof_shared_weights_and_exact_neff_control():
    pytest.importorskip("torch")
    windows = [
        _pair_window(1.0 + 0.05 * index, contaminate=index >= 4)
        for index in range(8)
    ]
    proxy, pair_mask, distance, signal_window = build_moveout_aligned_proxy(
        windows,
        dt_s=1.0,
        minimum_velocity_mps=10.0,
        maximum_velocity_mps=20.0,
        maximum_period_s=2.0,
    )
    assert proxy.shape == (8, 41)
    assert pair_mask.all()
    assert distance == pytest.approx(62.0)
    assert signal_window[0] < signal_window[1]

    result = train_shared_crossfit_nnws_v2(
        windows,
        dt_s=1.0,
        distance_bin_m=60.0,
        minimum_velocity_mps=10.0,
        maximum_velocity_mps=20.0,
        maximum_period_s=2.0,
        config=NNWSV2Config(
            folds=2,
            epochs=2,
            seed=19,
            device="cpu",
            minimum_effective_fraction=0.5,
        ),
        random_control_seed=23,
    )
    assert result.scope == "array_shared_moveout_proxy_contiguous_oof"
    np.testing.assert_array_equal(result.fit.fold_index, [0, 0, 0, 0, 1, 1, 1, 1])
    assert result.fit.effective_window_count >= 4.0
    assert result.fit.sufficient
    weights = result.pair_stack.statistics.weights
    np.testing.assert_allclose(
        weights, np.repeat(weights[:, :1], weights.shape[1], axis=1)
    )
    for pair_index in range(weights.shape[1]):
        np.testing.assert_allclose(
            np.sort(result.random_control_stack.statistics.weights[:, pair_index]),
            np.sort(weights[:, pair_index]),
        )
    np.testing.assert_array_equal(
        result.binned_branches.symmetric.distance_m,
        result.random_control_binned.symmetric.distance_m,
    )


def test_pair_eligibility_separates_geometry_from_training_failure():
    eligible, reasons = pair_nnws_eligibility(
        np.array([20.0, 60.0, 540.0]),
        lag_time_s=np.arange(61, dtype=np.float64),
        dt_s=1.0,
        minimum_velocity_mps=10.0,
        maximum_velocity_mps=20.0,
        maximum_period_s=2.0,
    )

    np.testing.assert_array_equal(eligible, [False, True, False])
    assert "pre-signal" in reasons[0]
    assert reasons[1] == ""
    assert "remote-tail" in reasons[2]
