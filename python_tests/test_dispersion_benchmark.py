import csv
import json

import h5py
import numpy as np

from wavedispx.dispersion_benchmark import (
    MATLAB_KERNEL_METHODS,
    build_benchmark_case,
    run_controlled_dispersion_benchmark,
)


def test_quick_benchmark_preserves_not_applicable_and_measured_rows(tmp_path):
    result = run_controlled_dispersion_benchmark(
        tmp_path,
        case_names=("active_inline", "passive_diffuse"),
        method_names=("phase_shift", "spac"),
        quick=True,
        make_plot=False,
    )
    assert result.metrics_csv.is_file()
    assert result.results_h5.is_file()
    assert result.manifest_json.is_file()
    with result.metrics_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    lookup = {(row["case"], row["method"]): row for row in rows}
    assert lookup[("active_inline", "phase_shift")]["applicable"] == "True"
    assert lookup[("active_inline", "spac")]["applicable"] == "False"
    assert lookup[("passive_diffuse", "phase_shift")]["applicable"] == "True"
    assert lookup[("passive_diffuse", "spac")]["applicable"] == "True"


def test_benchmark_persists_subsurface_model_and_source_receiver_geometry(tmp_path):
    result = run_controlled_dispersion_benchmark(
        tmp_path,
        case_names=("active_inline", "passive_spz"),
        method_names=("phase_shift",),
        quick=True,
        make_plot=False,
    )
    with h5py.File(result.results_h5) as handle:
        assert handle.attrs["schema"] == "wavedispx-dispersion-method-benchmark-v3"
        assert handle["model/layers/thickness_m"].shape == (2,)
        assert handle["model/layers/vp_mps"].shape == (2,)
        assert handle["model/layers/vs_mps"].shape == (2,)
        assert handle["model/layers/density_kgm3"].shape == (2,)
        for case_name in ("active_inline", "passive_spz"):
            group = handle[f"cases/{case_name}"]
            assert group["receiver_x_m"].shape == group["receiver_y_m"].shape
            assert group["source_x_m"].shape == group["source_y_m"].shape
            assert group["receiver_x_m"].size >= 2
            assert group["source_x_m"].size >= 1


def test_active_off_axis_uses_projected_x_without_source_position_correction(tmp_path):
    case = build_benchmark_case("active_off_spz", quick=True)
    true_range = np.hypot(
        case.receiver_x_m - case.source_x_m[0],
        case.receiver_y_m - case.source_y_m[0],
    )
    np.testing.assert_allclose(case.radial_distance_m, case.linear_coordinate_m)
    np.testing.assert_allclose(case.radial_distance_m, case.receiver_x_m)
    assert not np.allclose(case.radial_distance_m, true_range)
    assert "No source-position correction" in case.note
    result = run_controlled_dispersion_benchmark(
        tmp_path,
        case_names=("active_off_spz",),
        method_names=("phase_shift", "fk"),
        quick=True,
        make_plot=False,
    )
    for row in result.rows:
        assert row["expected_to_recover_truth"] is False
        assert row["expectation"] == "uncorrected_projected_coordinate_stress"
        assert float(row["ridge_mae_mps"]) > 30.0


def test_quick_benchmark_uses_declared_broad_frequency_band(tmp_path):
    result = run_controlled_dispersion_benchmark(
        tmp_path,
        case_names=("passive_diffuse",),
        method_names=("phase_shift",),
        quick=True,
        make_plot=False,
    )
    with h5py.File(result.results_h5) as handle:
        frequency = handle["images/passive_diffuse/phase_shift/frequency_hz"][:]
    assert frequency[0] <= 1.1
    assert frequency[-1] >= 23.9
    row = result.rows[0]
    assert row["metric_frequency_min_hz"] == 3.0
    assert row["metric_frequency_max_hz"] == 10.5
    assert int(row["broadband_frequency_sample_count"]) > int(
        row["frequency_sample_count"]
    )


def test_matlab_kernel_suite_is_persisted_with_signed_raw_diagnostics(tmp_path):
    assert len(MATLAB_KERNEL_METHODS) == 22
    result = run_controlled_dispersion_benchmark(
        tmp_path,
        case_names=("active_inline",),
        method_names=("ps_real", "ps_real_raw", "fj_real_r"),
        quick=True,
        make_plot=False,
    )
    with h5py.File(result.results_h5) as handle:
        projected = handle["images/active_inline/ps_real/energy"][:]
        signed = handle["images/active_inline/ps_real_raw/energy"][:]
        signed_raw = handle[
            "images/active_inline/ps_real_raw/raw_amplitude"
        ][:]
    np.testing.assert_allclose(projected, signed)
    assert np.any(signed_raw < 0.0)
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    methods = {
        row["method"] for row in manifest["matlab_kernel_suite"]["methods"]
    }
    assert methods == {"ps_real", "ps_real_raw", "fj_real_r"}


def test_oblique_passive_case_is_outside_spz_but_not_broadside(tmp_path):
    case = build_benchmark_case("passive_oblique_off_spz", quick=True)
    center_x = 0.5 * (case.receiver_x_m.min() + case.receiver_x_m.max())
    angle_deg = np.rad2deg(
        np.mod(
            np.arctan2(case.source_y_m, case.source_x_m - center_x),
            2.0 * np.pi,
        )
    )
    distance_to_oblique_axis = np.minimum(
        np.abs(angle_deg - 30.0), np.abs(angle_deg - 210.0)
    )
    assert np.max(distance_to_oblique_axis) <= 8.01
    assert np.all(np.abs(np.cos(np.deg2rad(angle_deg))) > 0.75)
    result = run_controlled_dispersion_benchmark(
        tmp_path,
        case_names=("passive_oblique_off_spz",),
        method_names=("fk",),
        quick=True,
        make_plot=False,
    )
    row = result.rows[0]
    assert float(row["ridge_mae_mps"]) > 30.0
    assert float(row["oblique_apparent_ridge_mae_mps"]) < 12.0


def test_full_passive_spz_and_diffuse_controls_pass_declared_quality_gates(tmp_path):
    result = run_controlled_dispersion_benchmark(
        tmp_path,
        case_names=("passive_spz", "passive_diffuse"),
        method_names=("phase_shift", "spac"),
        quick=False,
        make_plot=False,
    )
    lookup = {(row["case"], row["method"]): row for row in result.rows}
    required = (
        ("passive_spz", "phase_shift"),
        ("passive_diffuse", "phase_shift"),
        ("passive_diffuse", "spac"),
    )
    for key in required:
        row = lookup[key]
        assert row["quality_gate_required"] is True
        assert row["quality_gate_pass"] is True
        assert float(row["median_truth_energy"]) >= 0.80
        assert float(row["support_fraction_within_12_mps"]) >= 0.85
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["quality_gates"]["all_required_passed"] is True
