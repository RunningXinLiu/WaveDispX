import numpy as np
import pytest
from scipy.special import jv

from wavedispx.processing.array_dispersion import (
    MATLAB_KERNEL_METHODS,
    SpatialCoherency,
    canonical_array_method,
    method_input_contract,
    spac_dispersion,
    wavefield_dispersion,
)


def _plane_wave_gather(true_velocity_mps: float = 250.0):
    dt_s = 0.002
    time_s = np.arange(0.0, 2.5, dt_s)
    coordinate_m = np.arange(10.0, 170.0, 5.0)
    frequency_hz = 10.0
    arrival_s = 0.35 + coordinate_m / true_velocity_mps
    argument = np.pi * frequency_hz * (time_s[:, None] - arrival_s[None, :])
    gather = (1.0 - 2.0 * argument**2) * np.exp(-argument**2)
    return gather, coordinate_m, dt_s


def _median_ridge(image, lower_hz=6.0, upper_hz=14.0):
    selected = (
        (image.frequency_hz >= lower_hz) & (image.frequency_hz <= upper_hz)
    )
    return float(
        np.median(image.velocity_mps[np.argmax(image.energy[:, selected], axis=0)])
    )


def test_method_aliases_preserve_distinct_input_contracts():
    assert canonical_array_method("PS") == "phase_shift"
    assert canonical_array_method("FJ") == "frequency_bessel"
    assert canonical_array_method("FH") == "frequency_hankel"
    assert canonical_array_method("taup") == "tau_p"
    assert canonical_array_method("radon") == "linear_radon"
    assert canonical_array_method("hrlrt") == "high_resolution_radon"
    assert canonical_array_method("mvdr") == "capon"
    assert canonical_array_method("ESAC") == "espac"
    assert canonical_array_method("PS_real_pi4_sqrtR") == "ps_real_pi4_sqrt_r"
    assert canonical_array_method("FH_real_current") == "fh_real_r"
    assert method_input_contract("fh") == "radial_time_gather"
    assert method_input_contract("fk") == "linear_time_gather"
    assert method_input_contract("music") == "linear_time_gather"
    assert method_input_contract("spac") == "station_pair_coherency"
    with pytest.raises(ValueError, match="unknown array dispersion method"):
        canonical_array_method("not-a-method")


@pytest.mark.parametrize(
    ("method", "tolerance_mps"),
    [
        ("phase_shift", 4.0),
        ("fk", 4.0),
        ("tau_p", 6.0),
        ("linear_radon", 8.0),
        ("high_resolution_radon", 8.0),
        ("capon", 8.0),
        ("music", 8.0),
    ],
)
def test_line_array_methods_recover_a_known_plane_wave(method, tolerance_mps):
    gather, coordinate_m, dt_s = _plane_wave_gather()
    velocity_mps = np.arange(180.0, 321.0, 2.0)
    image = wavefield_dispersion(
        gather,
        coordinate_m,
        dt_s,
        velocity_mps,
        frequency_min_hz=4.0,
        frequency_max_hz=16.0,
        method=method,
        zero_padding=2.0,
        distance_taper_fraction=0.25,
        radon_iterations=30,
    )
    assert image.energy.shape == (velocity_mps.size, image.frequency_hz.size)
    assert np.all(np.isfinite(image.energy))
    assert np.allclose(image.energy.max(axis=0), 1.0)
    assert abs(_median_ridge(image) - 250.0) <= tolerance_mps


def test_frequency_bessel_and_hankel_return_finite_radial_images():
    gather, distance_m, dt_s = _plane_wave_gather()
    velocity_mps = np.arange(180.0, 321.0, 4.0)
    for method in ("frequency_bessel", "frequency_hankel"):
        image = wavefield_dispersion(
            gather,
            distance_m,
            dt_s,
            velocity_mps,
            frequency_min_hz=5.0,
            frequency_max_hz=15.0,
            method=method,
            zero_padding=1.0,
            radial_weighting="uniform",
        )
        assert image.input_contract == "radial_time_gather"
        assert image.energy.shape == (velocity_mps.size, image.frequency_hz.size)
        assert np.all(np.isfinite(image.energy))
        assert np.allclose(image.energy.max(axis=0), 1.0)


def test_all_matlab_kernel_variants_return_finite_radial_images():
    gather, distance_m, dt_s = _plane_wave_gather()
    velocity_mps = np.arange(180.0, 321.0, 8.0)
    for method in MATLAB_KERNEL_METHODS:
        image = wavefield_dispersion(
            gather,
            distance_m,
            dt_s,
            velocity_mps,
            frequency_min_hz=5.0,
            frequency_max_hz=15.0,
            method=method,
            zero_padding=1.0,
            distance_taper_fraction=0.0,
        )
        assert image.method == method
        assert image.input_contract == "radial_time_gather"
        assert image.energy.shape == (velocity_mps.size, image.frequency_hz.size)
        assert np.all(np.isfinite(image.energy))
        assert np.allclose(image.energy.max(axis=0), 1.0)


@pytest.mark.parametrize(
    ("projected", "signed"),
    [
        ("ps_real", "ps_real_raw"),
        ("fh_real_r", "fh_real_raw_r"),
        ("pfh_real_r", "pfh_real_raw_r"),
    ],
)
def test_raw_real_modes_have_the_same_energy_as_absolute_real_modes(
    projected, signed
):
    gather, distance_m, dt_s = _plane_wave_gather()
    velocity_mps = np.arange(180.0, 321.0, 6.0)
    common = dict(
        data=gather,
        coordinate_m=distance_m,
        dt_s=dt_s,
        velocity_mps=velocity_mps,
        frequency_min_hz=5.0,
        frequency_max_hz=15.0,
        zero_padding=1.0,
        distance_taper_fraction=0.0,
    )
    projected_image = wavefield_dispersion(method=projected, **common)
    signed_image = wavefield_dispersion(method=signed, **common)
    np.testing.assert_allclose(projected_image.energy, signed_image.energy)
    assert np.any(signed_image.raw_amplitude < 0.0)


def test_spac_and_espac_recover_an_exact_bessel_coherency():
    frequency_hz = np.linspace(4.0, 12.0, 33)
    distance_m = np.arange(10.0, 121.0, 10.0)
    true_velocity_mps = 280.0 - 3.0 * (frequency_hz - 4.0)
    coefficient = jv(
        0,
        2.0
        * np.pi
        * frequency_hz[:, None]
        * distance_m[None, :]
        / true_velocity_mps[:, None],
    )
    coherency = SpatialCoherency(
        frequency_hz=frequency_hz,
        pair_distance_m=distance_m,
        coherency=coefficient.astype(np.complex128),
        coefficient=coefficient,
        weight=np.ones_like(coefficient),
        pair_i=np.arange(distance_m.size),
        pair_j=np.arange(distance_m.size) + 1,
        valid_windows=20,
    )
    velocity_mps = np.arange(220.0, 321.0, 2.0)
    for method in ("spac", "espac"):
        image = spac_dispersion(
            coherency,
            velocity_mps,
            method=method,
            distance_bin_m=10.0,
        )
        ridge = image.velocity_mps[np.argmax(image.energy, axis=0)]
        assert image.input_contract == "station_pair_coherency"
        assert float(np.median(np.abs(ridge - true_velocity_mps))) <= 1.0


def test_spac_cannot_be_called_through_the_time_gather_api():
    gather, coordinate_m, dt_s = _plane_wave_gather()
    with pytest.raises(ValueError, match="require spac_dispersion"):
        wavefield_dispersion(
            gather,
            coordinate_m,
            dt_s,
            np.arange(200.0, 301.0, 5.0),
            frequency_min_hz=5.0,
            frequency_max_hz=15.0,
            method="spac",
        )
