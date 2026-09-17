from pathlib import Path

import numpy as np
import h5py
import pytest

from wavedispx.config import load_config
from wavedispx.processing import (
    linear_array_ncf,
    pick_dispersion_ridge,
    receiver_array_3c_ncf,
    receiver_array_ncf,
    surface_wave_dispersion,
    theoretical_dispersion,
)
from wavedispx.processing.ncf import (
    bin_pair_ncf_branches,
    linear_array_ncf_branches,
    receiver_pair_ncf_branches,
    receiver_array_3c_ncf_branches,
    receiver_array_ncf_branches,
)


ROOT = Path(__file__).resolve().parents[1]


def test_linear_fft_ncf_returns_expected_pair_counts():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(800, 5))
    result = linear_array_ncf(
        data,
        np.arange(5) * 2.0,
        0.01,
        window_s=4.0,
        overlap=0.5,
        maximum_lag_s=1.0,
        normalization="none",
    )
    np.testing.assert_array_equal(result.pair_count, [4, 3, 2, 1])
    np.testing.assert_allclose(result.distance_m, [2, 4, 6, 8])
    assert result.data.shape == (101, 4)
    assert result.valid_windows == 3


def test_linear_ncf_branches_preserve_the_existing_symmetric_result():
    rng = np.random.default_rng(71)
    data = rng.normal(size=(500, 6))
    receiver_x = np.arange(6, dtype=float) * 4.0
    options = dict(
        window_s=2.5,
        overlap=0.0,
        maximum_lag_s=0.6,
        normalization="none",
    )

    expected = linear_array_ncf(data, receiver_x, 0.01, **options)
    branches = linear_array_ncf_branches(data, receiver_x, 0.01, **options)

    np.testing.assert_allclose(branches.symmetric.data, expected.data, rtol=0, atol=0)
    np.testing.assert_allclose(
        branches.symmetric.data,
        0.5 * (branches.positive.data + branches.negative.data),
        rtol=0,
        atol=1e-15,
    )
    assert branches.positive.valid_windows == expected.valid_windows
    assert branches.negative.valid_windows == expected.valid_windows


def test_receiver_array_ncf_preserves_exact_line_fast_path():
    rng = np.random.default_rng(17)
    data = rng.normal(size=(600, 6))
    receiver_x = np.arange(6, dtype=float) * 3.0
    order = np.array([3, 0, 5, 1, 4, 2])
    expected = linear_array_ncf(
        data,
        receiver_x,
        0.01,
        window_s=3.0,
        overlap=0.0,
        maximum_lag_s=0.8,
        normalization="none",
    )
    actual = receiver_array_ncf(
        data[:, order],
        receiver_x[order],
        np.zeros(6),
        0.01,
        distance_bin_m=3.0,
        window_s=3.0,
        overlap=0.0,
        maximum_lag_s=0.8,
        normalization="none",
    )
    np.testing.assert_array_equal(actual.distance_m, expected.distance_m)
    np.testing.assert_array_equal(actual.pair_count, expected.pair_count)
    np.testing.assert_allclose(actual.data, expected.data, rtol=0, atol=0)


def test_receiver_array_ncf_bins_all_irregular_2d_pairs():
    rng = np.random.default_rng(23)
    data = rng.normal(size=(400, 4))
    result = receiver_array_ncf(
        data,
        np.array([0.0, 1.0, 0.0, 2.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        0.01,
        distance_bin_m=1.0,
        window_s=4.0,
        overlap=0.0,
        maximum_lag_s=0.5,
        normalization="none",
    )
    np.testing.assert_array_equal(result.distance_m, [1.0, 2.0])
    np.testing.assert_array_equal(result.pair_count, [4, 2])
    assert int(result.pair_count.sum()) == 6
    assert result.data.shape == (51, 2)
    assert result.valid_windows == 1


def test_pair_resolved_scalar_ncf_bins_back_to_existing_array_result():
    rng = np.random.default_rng(231)
    data = rng.normal(size=(400, 4))
    receiver_x = np.array([0.0, 1.0, 0.0, 2.0])
    receiver_y = np.array([0.0, 0.0, 1.0, 0.0])
    options = dict(
        window_s=4.0,
        overlap=0.0,
        maximum_lag_s=0.5,
        normalization="none",
    )
    pairs = receiver_pair_ncf_branches(
        data, receiver_x, receiver_y, 0.01, **options
    )
    actual = bin_pair_ncf_branches(pairs, distance_bin_m=1.0)
    expected = receiver_array_ncf_branches(
        data,
        receiver_x,
        receiver_y,
        0.01,
        distance_bin_m=1.0,
        **options,
    )

    np.testing.assert_array_equal(pairs.pair_i, [0, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(pairs.pair_j, [1, 2, 3, 2, 3, 3])
    np.testing.assert_allclose(
        pairs.distance_m,
        np.hypot(
            receiver_x[pairs.pair_j] - receiver_x[pairs.pair_i],
            receiver_y[pairs.pair_j] - receiver_y[pairs.pair_i],
        ),
    )
    for branch_name in ("positive", "negative", "symmetric"):
        pair_branch = getattr(pairs, branch_name)
        assert pair_branch.data.shape == (51, 6)
        np.testing.assert_allclose(
            getattr(actual, branch_name).data,
            getattr(expected, branch_name).data,
            rtol=1e-13,
            atol=1e-15,
        )
    np.testing.assert_array_equal(actual.symmetric.pair_count, [4, 2])


def test_unit_weight_pair_bins_match_the_existing_sorted_line_fast_path():
    rng = np.random.default_rng(232)
    data = rng.normal(size=(500, 6))
    receiver_x = np.arange(6, dtype=np.float64) * 4.0
    options = dict(
        window_s=2.5,
        overlap=0.0,
        maximum_lag_s=0.6,
        normalization="none",
    )
    pairs = receiver_pair_ncf_branches(
        data, receiver_x, np.zeros(6), 0.01, **options
    )
    actual = bin_pair_ncf_branches(pairs, distance_bin_m=4.0)
    expected = linear_array_ncf_branches(data, receiver_x, 0.01, **options)

    for branch_name in ("positive", "negative", "symmetric"):
        np.testing.assert_allclose(
            getattr(actual, branch_name).data,
            getattr(expected, branch_name).data,
            rtol=2e-13,
            atol=2e-15,
        )
    np.testing.assert_array_equal(actual.symmetric.pair_count, [5, 4, 3, 2, 1])


def test_pair_resolved_ncf_can_be_restricted_to_one_reference_receiver():
    rng = np.random.default_rng(233)
    data = rng.normal(size=(400, 5))
    receiver_x = np.arange(5, dtype=np.float64) * 10.0
    pairs = receiver_pair_ncf_branches(
        data,
        receiver_x,
        np.zeros(5),
        0.01,
        window_s=4.0,
        overlap=0.0,
        maximum_lag_s=0.5,
        normalization="none",
        reference_receiver_index=2,
    )

    np.testing.assert_array_equal(pairs.pair_i, [0, 1, 2, 2])
    np.testing.assert_array_equal(pairs.pair_j, [2, 2, 3, 4])
    np.testing.assert_allclose(pairs.distance_m, [20.0, 10.0, 10.0, 20.0])
    assert pairs.symmetric.data.shape == (51, 4)


def test_irregular_and_three_component_ncf_expose_all_branches_in_one_result():
    rng = np.random.default_rng(73)
    receiver_x = np.array([0.0, 3.0, 0.0, 4.0])
    receiver_y = np.array([0.0, 0.0, 4.0, 3.0])
    scalar = rng.normal(size=(400, 4))
    three_component = rng.normal(size=(400, 4, 3))
    options = dict(
        distance_bin_m=1.0,
        window_s=4.0,
        overlap=0.0,
        maximum_lag_s=0.5,
        normalization="none",
    )

    expected_scalar = receiver_array_ncf(
        scalar, receiver_x, receiver_y, 0.01, **options
    )
    scalar_branches = receiver_array_ncf_branches(
        scalar, receiver_x, receiver_y, 0.01, **options
    )
    np.testing.assert_allclose(
        scalar_branches.symmetric.data, expected_scalar.data, rtol=0, atol=0
    )

    expected_3c = receiver_array_3c_ncf(
        three_component, receiver_x, receiver_y, 0.01, **options
    )
    component_branches = receiver_array_3c_ncf_branches(
        three_component, receiver_x, receiver_y, 0.01, **options
    )
    assert set(component_branches) == {"rr", "tt", "zz"}
    for name, branches in component_branches.items():
        np.testing.assert_allclose(
            branches.symmetric.data, expected_3c[name].data, rtol=0, atol=0
        )


def test_three_component_line_ncf_matches_three_scalar_fast_paths():
    rng = np.random.default_rng(29)
    data = rng.normal(size=(500, 6, 3))
    receiver_x = np.arange(6, dtype=float) * 4.0
    options = dict(
        window_s=2.5,
        overlap=0.0,
        maximum_lag_s=0.6,
        normalization="none",
    )
    actual = receiver_array_3c_ncf(
        data,
        receiver_x,
        np.zeros(6),
        0.01,
        distance_bin_m=4.0,
        **options,
    )
    for name, component in (("rr", 0), ("tt", 1), ("zz", 2)):
        expected = linear_array_ncf(
            data[:, :, component], receiver_x, 0.01, **options
        )
        np.testing.assert_allclose(actual[name].data, expected.data, rtol=0, atol=0)


def test_three_component_array_rotates_horizontal_records_to_pair_axis():
    rng = np.random.default_rng(37)
    data = np.zeros((400, 2, 3))
    data[:, :, 0] = 0.1 * rng.normal(size=(400, 2))
    data[:, :, 1] = rng.normal(size=(400, 2))
    actual = receiver_array_3c_ncf(
        data,
        np.array([0.0, 0.0]),
        np.array([0.0, 5.0]),
        0.01,
        distance_bin_m=5.0,
        window_s=4.0,
        overlap=0.0,
        maximum_lag_s=0.5,
        normalization="none",
    )
    expected_radial = receiver_array_ncf(
        data[:, :, 1],
        np.array([0.0, 0.0]),
        np.array([0.0, 5.0]),
        0.01,
        distance_bin_m=5.0,
        window_s=4.0,
        overlap=0.0,
        maximum_lag_s=0.5,
        normalization="none",
    )
    np.testing.assert_allclose(actual["rr"].data, expected_radial.data, rtol=1e-14)


def test_phase_shift_and_picker_recover_known_velocity():
    dt = 0.005
    time = np.arange(1000) * dt
    distance = np.arange(5.0, 205.0, 5.0)
    true_velocity = 280.0
    frequencies = np.array([7.0, 9.0, 11.0, 13.0])
    data = sum(
        np.cos(2 * np.pi * frequency * (time[:, None] - distance[None, :] / true_velocity))
        for frequency in frequencies
    )
    velocity = np.arange(180.0, 381.0, 2.0)
    image = surface_wave_dispersion(
        data,
        distance,
        dt,
        velocity,
        frequency_min_hz=6.0,
        frequency_max_hz=14.0,
        method="phase_shift",
        zero_padding=4,
        distance_taper_fraction=0.5,
    )
    picks = pick_dispersion_ridge(
        image.energy,
        image.frequency_hz,
        image.velocity_mps,
        frequency_range_hz=(6.5, 13.5),
        velocity_range_mps=(240, 320),
        smoothness=0.2,
        maximum_jump_mps=20,
        minimum_energy=0.1,
    )
    assert np.median(np.abs(picks.phase_velocity_mps - true_velocity)) <= 2.0


def test_dispersion_explicit_frequency_step_controls_output_grid_only():
    dt = 0.01
    time = np.arange(400) * dt
    distance = np.arange(5.0, 55.0, 5.0)
    data = np.cos(2.0 * np.pi * 8.0 * (time[:, None] - distance[None, :] / 250.0))
    image = surface_wave_dispersion(
        data,
        distance,
        dt,
        np.arange(180.0, 321.0, 5.0),
        frequency_min_hz=5.0,
        frequency_max_hz=12.0,
        frequency_step_hz=0.2,
        zero_padding=1.0,
    )
    np.testing.assert_allclose(np.diff(image.frequency_hz), 0.2, atol=1.0e-12)
    assert image.frequency_hz[0] == pytest.approx(5.0)
    assert image.frequency_hz[-1] <= 12.0


def test_dispersion_averages_duplicate_radial_distances():
    rng = np.random.default_rng(41)
    unique_gather = rng.normal(size=(256, 3))
    duplicated_gather = unique_gather[:, [0, 1, 1, 2]]
    velocity = np.arange(180.0, 321.0, 10.0)
    options = dict(
        frequency_min_hz=4.0,
        frequency_max_hz=18.0,
        method="phase_shift",
        zero_padding=2,
    )
    expected = surface_wave_dispersion(
        unique_gather, np.array([5.0, 10.0, 20.0]), 0.01, velocity, **options
    )
    actual = surface_wave_dispersion(
        duplicated_gather,
        np.array([5.0, 10.0, 10.0, 20.0]),
        0.01,
        velocity,
        **options,
    )
    np.testing.assert_allclose(actual.energy, expected.energy, rtol=0, atol=0)


def test_theoretical_curve_for_reference_model():
    config = load_config(ROOT / "examples/passive_two_layer.toml")
    theory = theoretical_dispersion(config.model, np.linspace(5, 18, 14), rayleigh_modes=2, love_modes=2)
    fundamental = theory.rayleigh_mps[:, 0]
    assert np.all(np.isfinite(fundamental))
    assert 190 < fundamental[-1] < 230
    assert fundamental[0] > fundamental[-1]


def test_disba_theory_can_return_group_velocity_for_record_design():
    config = load_config(ROOT / "examples/passive_two_layer.toml")
    theory = theoretical_dispersion(
        config.model,
        np.linspace(3.0, 18.8, 40),
        rayleigh_modes=3,
        love_modes=3,
        compute_group=True,
    )
    assert theory.rayleigh_group_mps is not None
    assert theory.love_group_mps is not None
    fundamental = theory.rayleigh_group_mps[:, 0]
    assert np.all(np.isfinite(fundamental))
    assert np.all(fundamental > 0.0)


def test_ncf_matches_matlab_golden():
    with h5py.File(ROOT / "matlab_reference/golden_dwm_kernels.h5") as reference:
        group = reference["processing_ncf"]
        mixed = group["mixed_record"][...].T
        receiver_x = group["receiver_x_m"][...].reshape(-1)
        expected = group["gather"][...].T
        dt = float(group["dt_s"][...].reshape(-1)[0])
    result = linear_array_ncf(
        mixed,
        receiver_x,
        dt,
        window_s=4.0,
        overlap=0.5,
        maximum_lag_s=1.0,
        normalization="none",
    )
    np.testing.assert_allclose(result.data, expected, rtol=1e-10, atol=1e-12)


def test_phase_shift_matches_matlab_golden():
    with h5py.File(ROOT / "matlab_reference/golden_dwm_kernels.h5") as reference:
        group = reference["processing_dispersion"]
        wavefield = group["wavefield"][...].T
        distance = group["distance_m"][...].reshape(-1)
        velocity = group["velocity_mps"][...].reshape(-1)
        expected_frequency = group["frequency_hz"][...].reshape(-1)
        expected_energy = group["energy"][...].T
        dt = float(group["dt_s"][...].reshape(-1)[0])
    image = surface_wave_dispersion(
        wavefield,
        distance,
        dt,
        velocity,
        frequency_min_hz=6.0,
        frequency_max_hz=14.0,
        method="phase_shift",
        zero_padding=4,
        distance_taper_fraction=0.5,
    )
    np.testing.assert_allclose(image.frequency_hz, expected_frequency, rtol=0, atol=1e-14)
    np.testing.assert_allclose(image.energy, expected_energy, rtol=1e-9, atol=1e-11)


def test_parallel_hankel_lookup_matches_exact_special_functions():
    rng = np.random.default_rng(31)
    data = rng.normal(size=(256, 12))
    distance = np.arange(1.0, 13.0)
    velocity = np.arange(160.0, 341.0, 10.0)
    options = dict(
        frequency_min_hz=4.0,
        frequency_max_hz=20.0,
        method="hankel",
        component="vertical",
        zero_padding=2,
        frequency_block_size=4,
    )
    exact = surface_wave_dispersion(
        data, distance, 0.01, velocity, radial_kernel_lookup_step=0.0, **options
    )
    accelerated = surface_wave_dispersion(
        data, distance, 0.01, velocity, radial_kernel_lookup_step=0.002, **options
    )
    np.testing.assert_allclose(accelerated.energy, exact.energy, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(
        np.argmax(accelerated.energy, axis=0), np.argmax(exact.energy, axis=0)
    )
