from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from wavedispx.app_service import load_template_mapping
from wavedispx.config import ConfigError, config_from_mapping, load_config
from wavedispx.das import (
    apply_finite_gauge_operator,
    build_das_geometry,
    load_das_geometry_csv,
    normalized_gauge_response,
)
from wavedispx.runner import run


ROOT = Path(__file__).resolve().parents[1]


def test_rigid_translation_has_zero_das_strain():
    geometry = build_das_geometry(
        np.array([0.0, 10.0]), np.zeros(2), 4.0, 0.0
    )
    minus = np.ones((32, 2, 2), dtype=float)
    plus = minus.copy()
    record = apply_finite_gauge_operator(
        minus, plus, geometry, dt_s=0.01, measurement="strain"
    )
    np.testing.assert_allclose(record.data, 0.0, atol=0.0)
    assert record.units == "1"


def test_uniform_axial_extension_recovers_strain():
    geometry = build_das_geometry(
        np.array([-3.0, 7.0]), np.zeros(2), 5.0, 0.0
    )
    strain = 2.5e-6
    sample_count = 20
    minus = np.zeros((sample_count, 2, 2))
    plus = np.zeros_like(minus)
    minus[..., 0] = strain * geometry.endpoint_minus_xy_m[:, 0]
    plus[..., 0] = strain * geometry.endpoint_plus_xy_m[:, 0]
    record = apply_finite_gauge_operator(
        minus, plus, geometry, dt_s=0.02, measurement="strain"
    )
    np.testing.assert_allclose(record.data, strain, rtol=1e-12, atol=1e-15)


def test_reversing_fiber_azimuth_preserves_axial_strain_polarity():
    """An unoriented fibre axis is 180-degree periodic for axial strain."""
    sample_count = 8
    strain = 4.0e-6
    outputs = []
    for azimuth_deg in (0.0, 180.0):
        geometry = build_das_geometry(
            np.array([2.0]), np.array([0.0]), 6.0, azimuth_deg
        )
        minus = np.zeros((sample_count, 1, 2))
        plus = np.zeros_like(minus)
        minus[:, 0, 0] = strain * geometry.endpoint_minus_xy_m[0, 0]
        plus[:, 0, 0] = strain * geometry.endpoint_plus_xy_m[0, 0]
        outputs.append(
            apply_finite_gauge_operator(
                minus,
                plus,
                geometry,
                dt_s=0.01,
                measurement="strain",
            ).data
        )
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-12, atol=1e-15)


def test_strain_rate_uses_numpy_rfft_derivative_convention():
    sample_count = 256
    dt_s = 0.01
    bin_index = 7
    frequency_hz = bin_index / (sample_count * dt_s)
    time_s = np.arange(sample_count) * dt_s
    strain = 3.0e-6 * np.sin(2.0 * np.pi * frequency_hz * time_s)
    expected = (
        3.0e-6
        * 2.0
        * np.pi
        * frequency_hz
        * np.cos(2.0 * np.pi * frequency_hz * time_s)
    )
    geometry = build_das_geometry(np.array([0.0]), np.array([0.0]), 8.0, 0.0)
    minus = np.zeros((sample_count, 1, 2))
    plus = np.zeros_like(minus)
    minus[:, 0, 0] = strain * geometry.endpoint_minus_xy_m[0, 0]
    plus[:, 0, 0] = strain * geometry.endpoint_plus_xy_m[0, 0]
    record = apply_finite_gauge_operator(
        minus, plus, geometry, dt_s=dt_s, measurement="strain_rate"
    )
    np.testing.assert_allclose(record.data[:, 0], expected, rtol=1e-11, atol=1e-12)
    assert record.units == "s^-1"
    assert "derivative=+iomega" in record.time_convention


def test_finite_gauge_response_has_theoretical_first_zero():
    gauge_length_m = 10.0
    response = normalized_gauge_response(2.0 * np.pi / gauge_length_m, gauge_length_m)
    assert abs(response) < 1e-12


def test_channel_file_supports_turning_and_variable_gauge_geometry():
    geometry = load_das_geometry_csv(
        ROOT / "examples/das_channels_bent_example.csv"
    )
    assert geometry.geometry_mode == "channel_file"
    assert geometry.channel_count == 15
    assert geometry.channel_id[0] == "D001"
    assert geometry.channel_id[-1] == "D015"
    assert np.isclose(geometry.gauge_length_by_channel_m.min(), 6.0)
    assert np.isclose(geometry.gauge_length_by_channel_m.max(), 8.0)
    assert np.isnan(geometry.gauge_length_m)
    assert np.isnan(geometry.fiber_azimuth_deg)

    # Isotropic in-plane dilation has the same axial strain for every tangent
    # and verifies the per-channel divisor rather than a global median gauge.
    strain = 3.0e-6
    minus = np.repeat(
        (strain * geometry.endpoint_minus_xy_m)[None, :, :], 4, axis=0
    )
    plus = np.repeat(
        (strain * geometry.endpoint_plus_xy_m)[None, :, :], 4, axis=0
    )
    record = apply_finite_gauge_operator(
        minus, plus, geometry, dt_s=0.01, measurement="strain"
    )
    np.testing.assert_allclose(record.data, strain, rtol=1e-12, atol=1e-15)


def test_channel_file_accepts_explicit_endpoints(tmp_path):
    path = tmp_path / "channels.csv"
    path.write_text(
        "channel_id,x_m,y_m,endpoint_minus_x_m,endpoint_minus_y_m,"
        "endpoint_plus_x_m,endpoint_plus_y_m\n"
        "A,0,0,-2,0,2,0\n"
        "B,5,5,5,2,5,8\n",
        encoding="utf-8",
    )
    geometry = load_das_geometry_csv(path)
    np.testing.assert_allclose(geometry.gauge_length_by_channel_m, (4.0, 6.0))
    np.testing.assert_allclose(
        geometry.tangent_xy, ((1.0, 0.0), (0.0, 1.0)), atol=1e-15
    )


def test_channel_file_rejects_center_not_at_endpoint_midpoint(tmp_path):
    path = tmp_path / "bad_channels.csv"
    path.write_text(
        "x_m,y_m,endpoint_minus_x_m,endpoint_minus_y_m,"
        "endpoint_plus_x_m,endpoint_plus_y_m\n"
        "1,0,-2,0,2,0\n"
        "5,5,5,2,5,8\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="midpoint"):
        load_das_geometry_csv(path)


@pytest.mark.parametrize(
    "path",
    (
        "examples/active_das_line_quick.toml",
        "examples/active_das_grid_quick.toml",
        "examples/active_das_channel_file_quick.toml",
        "examples/passive_das_line_quick.toml",
        "examples/passive_das_grid_quick.toml",
        "examples/passive_das_channel_file_quick.toml",
    ),
)
def test_das_examples_are_valid(path):
    config = load_config(ROOT / path)
    assert config.das.enabled
    assert config.processing.dispersion_method == "phase_shift"


def test_das_rejects_unvalidated_hankel_imaging():
    mapping = load_template_mapping(ROOT / "examples/active_das_line_quick.toml")
    invalid = deepcopy(mapping)
    invalid["processing"]["dispersion_method"] = "hankel"
    with pytest.raises(ConfigError, match="DAS v1 requires"):
        config_from_mapping(invalid, base_directory=ROOT / "examples")


def test_passive_das_rejects_unvalidated_three_component_forces():
    mapping = load_template_mapping(ROOT / "examples/passive_das_line_quick.toml")
    invalid = deepcopy(mapping)
    invalid["ambient_sources"]["component"] = "three_component"
    with pytest.raises(ConfigError, match="passive DAS v1"):
        config_from_mapping(invalid, base_directory=ROOT / "examples")


def test_channel_file_mode_requires_receiver_file_layout():
    mapping = load_template_mapping(ROOT / "examples/active_das_line_quick.toml")
    invalid = deepcopy(mapping)
    invalid["das"]["geometry_mode"] = "channel_file"
    with pytest.raises(ConfigError, match="receiver.*layout='file'"):
        config_from_mapping(invalid, base_directory=ROOT / "examples")


def test_active_das_cli_runner_writes_measurement_contract(tmp_path):
    """Guard the direct CLI/runner path as well as the staged workflow path."""
    config = replace(
        load_config(ROOT / "examples/active_das_line_quick.toml"),
        output_directory=tmp_path / "active_das",
    )
    artifacts = run(config)
    with h5py.File(artifacts.results_h5) as results:
        assert results["das/data"].shape[1] == config.receivers.count
        assert results["das"].attrs["measurement"] == "strain_rate"
        assert results["das"].attrs["operator_version"] == (
            "wavedispx-finite-gauge-v1"
        )
        assert "das" in results["dispersion/components"]


def test_active_channel_file_runner_writes_per_channel_geometry(tmp_path):
    config = replace(
        load_config(ROOT / "examples/active_das_channel_file_quick.toml"),
        output_directory=tmp_path / "active_das_channel_file",
    )
    artifacts = run(config)
    assert (artifacts.output_directory / "das_channels.csv").is_file()
    roundtrip = load_das_geometry_csv(
        artifacts.output_directory / "das_channels.csv"
    )
    assert roundtrip.channel_id[0] == "D001"
    np.testing.assert_allclose(
        roundtrip.gauge_length_by_channel_m,
        load_das_geometry_csv(
            ROOT / "examples/das_channels_bent_example.csv"
        ).gauge_length_by_channel_m,
    )
    with h5py.File(artifacts.results_h5) as results:
        das = results["das"]
        assert das.attrs["geometry_mode"] == "channel_file"
        assert not das.attrs["uniform_gauge_length"]
        assert not das.attrs["uniform_fiber_azimuth"]
        assert das["gauge_length_by_channel_m"].shape == (15,)
        assert das["fiber_azimuth_by_channel_deg"].shape == (15,)
        assert das["channel_id"].asstr()[0] == "D001"
