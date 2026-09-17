from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from wavedispx.config import load_config
from wavedispx.runner import run


ROOT = Path(__file__).resolve().parents[1]


def test_active_quick_end_to_end_writes_complete_artifacts(tmp_path):
    config = replace(
        load_config(ROOT / "examples/active_quick.toml"),
        output_directory=tmp_path / "active",
    )
    artifacts = run(config)
    assert artifacts.summary_png.is_file()
    assert (artifacts.output_directory / "model.csv").is_file()
    assert (artifacts.output_directory / "receivers.csv").is_file()
    assert (artifacts.output_directory / "sources.csv").is_file()
    with h5py.File(artifacts.results_h5) as results:
        assert results.attrs["mode"] == "active"
        assert results.attrs["forward_solver_domain"] == "frequency-wavenumber"
        assert results.attrs["dispersion_product"] == "windowed_waveform_image"
        assert results.attrs["record_duration_mode"] == "automatic"
        assert np.isclose(
            results.attrs["record_frequency_step_hz"],
            1.0
            / (
                results.attrs["record_sample_count"]
                * results.attrs["record_dt_s"]
            ),
        )
        assert results.attrs["dwm_frequency_step_hz"] < results.attrs[
            "record_frequency_step_hz"
        ]
        assert results["shot/components"].shape[2] == 5
        assert results["dispersion/rayleigh/energy"].shape[0] > 10


def test_passive_quick_end_to_end_is_mixed_and_streaming(tmp_path):
    config = replace(
        load_config(ROOT / "examples/passive_quick.toml"),
        output_directory=tmp_path / "passive",
    )
    artifacts = run(config)
    assert artifacts.summary_png.is_file()
    with h5py.File(artifacts.results_h5) as results:
        assert results.attrs["mode"] == "passive"
        assert results.attrs["forward_solver_domain"] == "frequency-wavenumber"
        assert (
            results.attrs["dispersion_product"]
            == "finite_record_ncf_waveform_image"
        )
        assert results.attrs["correlation_mode"] == "mixed finite-record"
        assert results.attrs["independent_window_count"] == 6
        assert np.isclose(
            results.attrs["ncf_frequency_step_hz"],
            1.0 / results.attrs["noise_independent_window_s"],
        )
        assert results.attrs["noise_total_duration_s"] > results.attrs[
            "noise_independent_window_s"
        ]
        assert results.attrs["radial_interpolation_relative_l2"] < 0.02
        assert "vertical_spectrum" not in results["transfer"]
        assert results["preview/mixed_record"].shape[1] == 41
        assert results["ncf/gather"].shape[1] == 40
        baseline = results["ncf/strategies/baseline_linear"]
        assert baseline.attrs["sum_weight"] == 6.0
        assert baseline.attrs["effective_window_count"] == 6.0
        assert baseline.attrs["sufficient"]
        assert set(baseline["components/zz"]) == {
            "positive",
            "negative",
            "symmetric",
        }
        assert baseline["components/zz/symmetric"].shape == results["ncf/gather"].shape
        reference = results["egf_reference"]
        assert reference.attrs["reference_type"] == "active_interreceiver_green_function"
        assert reference.attrs["source_time_function"] == "unit_impulse"
        assert (
            reference.attrs["calibration_operator"]
            == "negative_time_derivative_of_ncf"
        )
        assert tuple(reference.attrs["component_names"]) == ("zz",)
        assert reference["components"].shape[1:] == (40, 1)
        assert reference["pair_i"].size == 820
        assert reference["pair_j"].shape == reference["pair_i"].shape
        assert reference["pair_distance_index"].shape == reference["pair_i"].shape


def test_passive_three_component_end_to_end_writes_tensor_products(tmp_path):
    base = load_config(ROOT / "examples/passive_quick.toml")
    config = replace(
        base,
        ambient_sources=replace(base.ambient_sources, component="three_component"),
        noise=replace(base.noise, duration_s=20.0),
        output_directory=tmp_path / "passive_3c",
    )
    artifacts = run(config)
    assert (artifacts.output_directory / "passive_components.png").is_file()
    with h5py.File(artifacts.results_h5) as results:
        assert results["preview/mixed_record"].ndim == 3
        assert results["preview/mixed_record"].shape[2] == 3
        assert set(results["ncf/components"]) == {"rr", "tt", "zz"}
        assert set(results["dispersion/components"]) == {"rr", "tt", "zz"}
        assert "rayleigh_mode_support" in results["quality"]


def test_oracle_source_selection_writes_traceable_audit_and_strategies(tmp_path):
    config = replace(
        load_config(ROOT / "examples/passive_source_selection_quick.toml"),
        output_directory=tmp_path / "source_selection",
    )
    artifacts = run(config)
    assert (artifacts.output_directory / "source_selection_windows.csv").is_file()
    with h5py.File(artifacts.results_h5) as results:
        selection = results["source_selection"]
        assert selection.attrs["scope"] == "array_axis_approximation"
        assert selection.attrs["method_version"] == "oracle_axis_stationary_phase_v1"
        assert not selection.attrs["correlated_source_risk"]
        assert selection["window_index"].shape == (6,)
        assert selection["endfire_fraction"].shape == (6, 2)
        assert set(selection["weights"]) == {
            "oracle_hard_mask",
            "oracle_soft_weight",
            "random_budget_control",
            "random_neff_control",
        }
        assert set(results["ncf/strategies"]) == {
            "baseline_linear",
            "oracle_hard_mask",
            "oracle_soft_weight",
            "random_budget_control",
            "random_neff_control",
        }
        assert set(results["dispersion/strategies"]) == set(results["ncf/strategies"])
        baseline_quality = results[
            "quality/strategies/baseline_linear/egf/zz_symmetric"
        ]
        assert not baseline_quality.attrs["absolute_amplitude_available"]
        assert not baseline_quality.attrs["time_shift_optimized"]
        assert "phase_rms_rad" in baseline_quality.attrs


def test_pairwise_nnws_writes_pair_and_window_resolved_audit(tmp_path):
    base = load_config(ROOT / "examples/passive_nnws_pairwise_quick.toml")
    config = replace(
        base,
        nnws=replace(base.nnws, epochs=2),
        output_directory=tmp_path / "nnws_pairwise",
    )
    artifacts = run(config)
    assert (artifacts.output_directory / "nnws_pair_audit.csv").is_file()
    with h5py.File(artifacts.results_h5) as results:
        audit = results["nnws_pairwise"]
        assert audit.attrs["scope"] == "pair_resolved"
        assert audit.attrs["pair_scope"] == "all"
        assert audit.attrs["reference_receiver_index"] == 0
        assert audit.attrs["std_mode"] == "population_eps"
        assert audit.attrs["pair_coverage"] == 1.0
        assert audit.attrs["eligible_pair_count"] == 3
        assert audit.attrs["ineligible_pair_count"] == 3
        assert audit["weights"].shape == (4, 6)
        assert audit["pair_i"].shape == (6,)
        assert audit["window_start_time_s"].shape == (4,)
        assert audit["sufficient"][:].sum() == 3
        assert audit["eligible"][:].sum() == 3
        assert sum(bool(reason) for reason in audit["eligibility_reason"][:]) == 3
        assert set(results["ncf/strategies"]) == {
            "baseline_linear",
            "nnws_common_linear",
            "nnws_hybrid_full_aperture",
            "nnws_hybrid_random_neff_control",
            "nnws_random_neff_control",
            "nnws_pairwise",
        }
        assert audit["controls/common_linear/weights"].shape == (4, 6)
        assert audit["controls/random_neff/weights"].shape == (4, 6)
        assert audit["controls/hybrid_full_aperture/weights"].shape == (4, 6)
        assert audit["controls/hybrid_random_neff/weights"].shape == (4, 6)
        ineligible = ~audit["eligible"][:].astype(bool)
        np.testing.assert_allclose(
            audit["controls/hybrid_full_aperture/weights"][:, ineligible],
            1.0,
        )
        np.testing.assert_allclose(
            audit["controls/hybrid_random_neff/weights"][:, ineligible],
            1.0,
        )
        strategy = results["ncf/strategies/nnws_pairwise"]
        assert strategy.attrs["sufficient"]
        assert strategy["distance_m"].shape == (2,)
        for control_name in (
            "nnws_common_linear",
            "nnws_random_neff_control",
        ):
            assert (
                results[f"ncf/strategies/{control_name}/distance_m"][:]
                == strategy["distance_m"][:]
            ).all()
        baseline_distance = results["ncf/strategies/baseline_linear/distance_m"][:]
        for hybrid_name in (
            "nnws_hybrid_full_aperture",
            "nnws_hybrid_random_neff_control",
        ):
            np.testing.assert_array_equal(
                results[f"ncf/strategies/{hybrid_name}/distance_m"][:],
                baseline_distance,
            )
            assert results[f"ncf/strategies/{hybrid_name}"].attrs["sufficient"]
        assert set(strategy["components/zz"]) == {
            "positive",
            "negative",
            "symmetric",
        }


def test_nnws_v2_writes_crossfit_and_shared_weight_audit(tmp_path):
    base = load_config(ROOT / "examples/passive_nnws_pairwise_quick.toml")
    config = replace(
        base,
        nnws=replace(base.nnws, enabled=False),
        nnws_v2=replace(
            base.nnws_v2,
            enabled=True,
            folds=2,
            epochs=2,
            minimum_effective_fraction=0.5,
            device="cpu",
            pair_scope="all",
        ),
        output_directory=tmp_path / "nnws_v2",
    )
    artifacts = run(config)
    assert (artifacts.output_directory / "nnws_v2_window_audit.csv").is_file()
    assert (
        artifacts.output_directory / "pair_crossfit_tau_p_spz_audit.csv"
    ).is_file()
    with h5py.File(artifacts.results_h5) as results:
        audit = results["nnws_v2"]
        assert audit.attrs["scope"] == (
            "array_shared_moveout_proxy_contiguous_oof"
        )
        assert audit.attrs["pair_scope"] == "all"
        assert audit.attrs["folds"] == 2
        assert audit["weights"].shape == (4,)
        assert audit["fold_index"].shape == (4,)
        assert set(np.unique(audit["fold_index"][:])) == {0, 1}
        assert audit["proxy_pair_mask"].shape == (6,)
        assert {
            "baseline_linear",
            "nnws_v2_random_neff_control",
            "nnws_v2_shared_oof",
            "nnws_v2_tau_p_directional_guard_frequency",
            "nnws_v2_tau_p_directional_guard_frequency_permuted_time_control",
            "nnws_v2_tau_p_branch_guard_frequency",
            "nnws_v2_tau_p_branch_guard_frequency_permuted_time_control",
            "nnws_v2_tau_p_diffuse_prearrival_frequency",
            "nnws_v2_tau_p_diffuse_prearrival_frequency_permuted_time_control",
            "pair_crossfit_tau_p_diffuse_prearrival_hard_frequency",
            "pair_crossfit_tau_p_diffuse_prearrival_hard_frequency_"
            "permuted_time_control",
            "pair_crossfit_tau_p_spz_frequency",
            "pair_crossfit_tau_p_spz_frequency_permuted_time_control",
        }.issubset(set(results["ncf/strategies"]))
        tau_p = results["pair_crossfit_tau_p_spz"]
        assert tau_p.attrs["scope"] == (
            "pair_branch_frequency_contiguous_crossfit_tau_p_data_only"
        )
        assert np.isclose(tau_p.attrs["nnws_v2_directional_fraction"], 0.8)
        expected_tau_p_shape = (
            4,
            len(config.source_selection.frequency_bands_hz),
            6,
        )
        assert tau_p["positive_band_weights"].shape == expected_tau_p_shape
        assert tau_p["negative_band_weights"].shape == expected_tau_p_shape
        assert tau_p["nnws_v2_directional_guard_band_weights"].shape == (
            expected_tau_p_shape[:2]
        )
        assert tau_p["nnws_v2_positive_branch_guard_band_weights"].shape == (
            expected_tau_p_shape
        )
        assert tau_p["nnws_v2_negative_branch_guard_band_weights"].shape == (
            expected_tau_p_shape
        )
        assert tau_p[
            "nnws_v2_positive_branch_guard_control_band_weights"
        ].shape == expected_tau_p_shape
        assert tau_p[
            "nnws_v2_negative_branch_guard_control_band_weights"
        ].shape == expected_tau_p_shape
        assert tau_p[
            "nnws_v2_positive_diffuse_prearrival_band_weights"
        ].shape == expected_tau_p_shape
        assert tau_p[
            "nnws_v2_negative_diffuse_prearrival_band_weights"
        ].shape == expected_tau_p_shape
        assert np.isclose(
            tau_p.attrs["nnws_v2_diffuse_prearrival_frequency_fraction"],
            0.7,
        )
        assert tau_p["diffuse_prearrival_hard_band_weights"].shape == (
            expected_tau_p_shape[:2]
        )
        assert np.isclose(
            tau_p.attrs["diffuse_prearrival_directional_fraction"],
            0.75,
        )
        np.testing.assert_allclose(
            np.sort(audit["weights"][:]),
            np.sort(audit["random_control_weights"][:]),
        )
        assert results["ncf/strategies/nnws_v2_shared_oof"].attrs[
            "sufficient"
        ]


def test_pair_spz_and_nnws_v2_combination_writes_physical_audit(tmp_path):
    base = load_config(ROOT / "examples/passive_nnws_pairwise_quick.toml")
    config = replace(
        base,
        nnws=replace(base.nnws, enabled=False),
        nnws_v2=replace(
            base.nnws_v2,
            enabled=True,
            folds=2,
            epochs=2,
            device="cpu",
            pair_scope="all",
        ),
        pair_spz=replace(
            base.pair_spz,
            enabled=True,
            maximum_iterations=20,
            pair_scope="all",
        ),
        output_directory=tmp_path / "pair_spz",
    )
    artifacts = run(config)
    assert (artifacts.output_directory / "pair_spz_window_audit.csv").is_file()
    with h5py.File(artifacts.results_h5) as results:
        audit = results["pair_spz"]
        assert audit.attrs["scope"] == (
            "pair_frequency_stationary_phase_known_source"
        )
        assert audit.attrs["pair_scope"] == "all"
        assert audit["weights"].shape == (4,)
        assert audit["nnws_v2_spz_constrained_weights"].shape == (4,)
        assert set(results["ncf/strategies"]) == {
            "baseline_linear",
            "oracle_pair_spz",
            "oracle_pair_spz_random_neff_control",
                "nnws_v2_random_neff_control",
                "nnws_v2_shared_oof",
                "nnws_v2_tau_p_directional_guard_frequency",
                "nnws_v2_tau_p_directional_guard_frequency_permuted_time_control",
                "nnws_v2_tau_p_branch_guard_frequency",
                "nnws_v2_tau_p_branch_guard_frequency_permuted_time_control",
                "nnws_v2_tau_p_diffuse_prearrival_frequency",
                "nnws_v2_tau_p_diffuse_prearrival_frequency_permuted_time_control",
                "pair_crossfit_tau_p_diffuse_prearrival_hard_frequency",
                "pair_crossfit_tau_p_diffuse_prearrival_hard_frequency_"
                "permuted_time_control",
                "nnws_v2_spz_blend_010",
                "nnws_v2_spz_blend_020",
                "nnws_v2_spz_blend_030",
                "nnws_v2_spz_blend_040",
                "nnws_v2_spz_frequency_blend_010",
                "nnws_v2_spz_frequency_blend_020",
                "nnws_v2_spz_frequency_blend_030",
                "nnws_v2_spz_frequency_blend_040",
                "nnws_v2_spz_frequency_blend_050",
                "nnws_v2_spz_phase_hybrid_010",
                "nnws_v2_spz_phase_hybrid_020",
                "nnws_v2_spz_phase_hybrid_030",
                "nnws_v2_spz_phase_hybrid_040",
                "nnws_v2_spz_phase_hybrid_050",
                "nnws_v2_spz_phase_hybrid_permuted_time_control_020",
                "nnws_v2_spz_v2_amplitude_phase_hybrid_020",
                "nnws_v2_spz_v2_amplitude_phase_hybrid_permuted_time_control_020",
                "nnws_v2_spz_branch_frequency_blend_020",
                "nnws_v2_spz_branch_frequency_blend_010",
                "nnws_v2_spz_branch_phase_hybrid_010",
                "nnws_v2_spz_branch_phase_hybrid_permuted_time_control_010",
                "nnws_v2_spz_branch_phase_hybrid_020",
                "nnws_v2_spz_branch_v2_amplitude_phase_hybrid_020",
                "nnws_v2_spz_branch_frequency_permuted_time_control_020",
                "nnws_v2_spz_branch_phase_hybrid_permuted_time_control_020",
                "nnws_v2_spz_branch_v2_amplitude_phase_hybrid_permuted_time_control_020",
                "supervised_egf_oracle_upper_bound",
                "supervised_egf_oracle_permuted_time_control",
                "supervised_pair_egf_oracle_upper_bound",
                "supervised_pair_egf_oracle_permuted_time_control",
                "pair_crossfit_coherence",
                "pair_crossfit_coherence_permuted_time_control",
                "pair_prearrival_reweighting",
                "pair_prearrival_reweighting_permuted_time_control",
                "pair_crossfit_tau_p_spz_frequency",
                "pair_crossfit_tau_p_spz_frequency_permuted_time_control",
                "pair_crossfit_tau_p_diffuse_soft_frequency",
                "pair_crossfit_tau_p_diffuse_soft_frequency_permuted_time_control",
                "pair_crossfit_tau_p_diffuse_hard_frequency",
                "pair_crossfit_tau_p_diffuse_hard_frequency_permuted_time_control",
                "oracle_pair_branch_spz",
                "oracle_pair_branch_spz_random_neff_control",
                "nnws_v2_pair_branch_spz_blend_020",
                "nnws_v2_pair_branch_spz_blend_permuted_time_control_020",
                "oracle_pair_branch_spz_frequency",
                "oracle_pair_branch_spz_frequency_random_neff_control",
                "nnws_v2_pair_branch_spz_frequency_blend_020",
                "nnws_v2_pair_branch_spz_frequency_blend_permuted_time_control_020",
                "linear_spz_frequency_blend_010",
                "linear_spz_frequency_blend_020",
                "linear_spz_frequency_blend_030",
                "linear_spz_frequency_blend_040",
                "linear_spz_frequency_blend_050",
                "nnws_v2_spz_random_neff_control",
                "nnws_v2_spz_constrained",
            }
        np.testing.assert_allclose(
            np.sort(audit["weights"][:]),
            np.sort(audit["random_neff_weights"][:]),
        )
        phase_hybrid_weights = audit[
            "nnws_v2_spz_phase_hybrid_band_weights"
        ][:]
        phase_hybrid_control = audit[
            "nnws_v2_spz_phase_hybrid_permuted_time_control_band_weights"
        ][:]
        np.testing.assert_allclose(
            np.sort(phase_hybrid_weights, axis=0),
            np.sort(phase_hybrid_control, axis=0),
        )
        for strategy in (
            "oracle_pair_spz",
            "nnws_v2_spz_constrained",
        ):
            assert results[f"ncf/strategies/{strategy}"].attrs["sufficient"]
