from dataclasses import replace
from pathlib import Path
import tomllib

import numpy as np

from wavedispx.passive import accumulate_passive_correlation, simulate_passive
from wavedispx.workflow import WaveDispXWorkflow


ROOT = Path(__file__).resolve().parents[1]


def test_cell_mapping_and_toml_build_the_same_case():
    path = ROOT / "examples/passive_quick.toml"
    with path.open("rb") as stream:
        mapping = tomllib.load(stream)
    from_toml = WaveDispXWorkflow.from_toml(path)
    from_cells = WaveDispXWorkflow.from_mapping(
        mapping, base_directory=path.parent
    )
    assert from_cells.fingerprint == from_toml.fingerprint
    assert from_cells.config.to_dict()["model"] == from_toml.config.to_dict()["model"]


def test_staged_passive_workflow_matches_end_to_end_function():
    path = ROOT / "examples/passive_quick.toml"
    workflow = WaveDispXWorkflow.from_toml(path)
    staged = workflow.run_all(save=False)
    direct = simulate_passive(workflow.config)
    np.testing.assert_allclose(
        workflow.correlation.ncf.data, direct.ncf.data, rtol=0, atol=0
    )
    baseline = workflow.correlation.strategy_stacks["baseline_linear"]
    assert baseline.statistics.sum_weight == 6.0
    assert baseline.statistics.effective_window_count == 6.0
    assert baseline.statistics.sufficient
    np.testing.assert_allclose(
        baseline.component_branches["zz"].symmetric.data,
        workflow.correlation.ncf.data,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        direct.strategy_stacks["baseline_linear"].component_branches["zz"].positive.data,
        workflow.correlation.strategy_stacks["baseline_linear"].component_branches["zz"].positive.data,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        staged.dispersion.energy, direct.dispersion.energy, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        staged.picks.phase_velocity_mps,
        direct.picks.phase_velocity_mps,
        rtol=0,
        atol=0,
    )


def test_passive_correlation_accumulates_multiple_weights_in_one_realization():
    workflow = WaveDispXWorkflow.from_toml(ROOT / "examples/passive_quick.toml")
    transfer = workflow.compute_forward()
    alternating = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    correlation = accumulate_passive_correlation(
        workflow.config,
        workflow.geometry,
        transfer,
        strategy_weights_by_window={
            "alternating_mask": alternating,
            "empty_mask": np.zeros_like(alternating),
        },
    )

    assert set(correlation.strategy_stacks) == {
        "baseline_linear",
        "alternating_mask",
        "empty_mask",
    }
    assert correlation.strategy_stacks["alternating_mask"].statistics.sum_weight == 3.0
    assert correlation.strategy_stacks["alternating_mask"].statistics.effective_window_count == 3.0
    empty = correlation.strategy_stacks["empty_mask"]
    assert not empty.statistics.sufficient
    assert np.isnan(empty.component_branches["zz"].symmetric.data).all()


def test_enabled_oracle_selection_runs_all_budget_controls_in_one_simulation():
    workflow = WaveDispXWorkflow.from_toml(
        ROOT / "examples/passive_source_selection_quick.toml"
    )
    correlation = workflow.simulate_observations()

    assert correlation.source_selection is not None
    assert set(correlation.strategy_stacks) == {
        "baseline_linear",
        "oracle_hard_mask",
        "oracle_soft_weight",
        "random_budget_control",
        "random_neff_control",
    }
    hard = correlation.strategy_stacks["oracle_hard_mask"].statistics
    random_hard = correlation.strategy_stacks["random_budget_control"].statistics
    assert hard.nonzero_window_count == random_hard.nonzero_window_count
    soft = correlation.strategy_stacks["oracle_soft_weight"].statistics
    random_soft = correlation.strategy_stacks["random_neff_control"].statistics
    assert np.isclose(soft.effective_window_count, random_soft.effective_window_count)

    analysis = workflow.compute_dispersion()
    assert set(analysis.strategy_images) == set(correlation.strategy_stacks)
    assert set(analysis.strategy_dispersion_quality) == set(correlation.strategy_stacks)
    assert set(analysis.strategy_egf_quality) == set(correlation.strategy_stacks)
    baseline_egf = analysis.strategy_egf_quality["baseline_linear"]
    assert set(baseline_egf) == {"zz_positive", "zz_negative", "zz_symmetric"}
    assert not baseline_egf["zz_symmetric"]["absolute_amplitude_available"]
    assert baseline_egf["zz_symmetric"]["time_shift_optimized"] is False


def test_isotropic_oracle_beats_random_budget_on_controlled_anisotropic_case():
    workflow = WaveDispXWorkflow.from_toml(
        ROOT / "examples/passive_source_selection_anisotropic_quick.toml"
    )
    analysis = workflow.compute_dispersion()
    oracle = analysis.strategy_egf_quality["oracle_hard_mask"]["zz_symmetric"]
    random = analysis.strategy_egf_quality["random_budget_control"]["zz_symmetric"]
    baseline = analysis.strategy_egf_quality["baseline_linear"]["zz_symmetric"]

    assert oracle["waveform_correlation_no_shift"] > random["waveform_correlation_no_shift"]
    assert oracle["waveform_correlation_no_shift"] > baseline["waveform_correlation_no_shift"]
    assert oracle["phase_rms_rad"] < random["phase_rms_rad"]
    assert oracle["prearrival_ghost_energy_fraction"] < random[
        "prearrival_ghost_energy_fraction"
    ]


def test_processing_edit_invalidates_only_analysis_cache():
    workflow = WaveDispXWorkflow.from_toml(ROOT / "examples/passive_quick.toml")
    workflow.compute_dispersion()
    transfer = workflow.transfer
    correlation = workflow.correlation
    changed_processing = replace(
        workflow.config.processing,
        picker_smoothness=workflow.config.processing.picker_smoothness + 0.05,
    )
    invalidated = workflow.configure(
        replace(workflow.config, processing=changed_processing)
    )
    assert invalidated == ["analysis"]
    assert workflow.transfer is transfer
    assert workflow.correlation is correlation
    assert workflow.passive_analysis is None


def test_passive_forward_builds_and_caches_active_reference():
    base = WaveDispXWorkflow.from_toml(ROOT / "examples/passive_quick.toml").config
    config = replace(
        base,
        receivers=replace(base.receivers, start_x_m=0.0, stop_x_m=15.0),
    )
    workflow = WaveDispXWorkflow(config)

    transfer = workflow.compute_forward()
    reference = workflow.egf_reference

    assert transfer is workflow.transfer
    assert reference is not None
    assert workflow.compute_forward() is transfer
    assert workflow.egf_reference is reference

    changed_noise = replace(config.noise, seed=config.noise.seed + 1)
    invalidated = workflow.configure(replace(config, noise=changed_noise))
    assert invalidated == ["observations", "analysis"]
    assert workflow.transfer is transfer
    assert workflow.egf_reference is reference


def test_irregular_receiver_workflow_reaches_dispersion():
    workflow = WaveDispXWorkflow.from_toml(
        ROOT / "examples/passive_irregular_quick.toml"
    )
    analysis = workflow.run_all(save=False)
    assert workflow.geometry.receiver_x_m.size == 21
    assert workflow.correlation.ncf.pair_count.sum() == 210
    assert analysis.dispersion.energy.shape[1] > 10
    assert np.all(np.diff(workflow.correlation.ncf.distance_m) > 0)


def test_active_irregular_receivers_reach_both_dispersion_images():
    workflow = WaveDispXWorkflow.from_toml(
        ROOT / "examples/active_irregular_quick.toml"
    )
    analysis = workflow.run_all(save=False)
    assert workflow.shot.receiver_x_m.size == 21
    assert np.any(workflow.shot.receiver_y_m != 0)
    assert np.all(workflow.shot.distance_m > 0)
    assert analysis.rayleigh.energy.shape[1] > 10
    assert analysis.love.energy.shape == analysis.rayleigh.energy.shape
