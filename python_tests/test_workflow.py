from dataclasses import replace
from pathlib import Path
import tomllib

import numpy as np

from wavedispx.passive import simulate_passive
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
    np.testing.assert_allclose(
        staged.dispersion.energy, direct.dispersion.energy, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        staged.picks.phase_velocity_mps,
        direct.picks.phase_velocity_mps,
        rtol=0,
        atol=0,
    )


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
