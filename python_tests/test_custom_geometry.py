from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from wavedispx.app_service import load_template_mapping
from wavedispx.workflow import WaveDispXWorkflow
from wavedispx.source_activity import load_source_activity_matrix
from wavedispx.experiment import run_experiment, ExperimentRequest

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "examples/custom_geometry"


@pytest.mark.parametrize("name", ["east_line", "clusters_array", "switching_array"])
def test_nonuniform_input_package(name):
    w = WaveDispXWorkflow.from_toml(INPUTS / f"passive_{name}.toml")
    assert w.review_inputs().ready
    assert len(w.geometry.source_x_m) == 12
    assert len(w.geometry.receiver_x_m) == (41 if name == "east_line" else 25)
    weights = load_source_activity_matrix(w.config.ambient_sources, 6)
    if name == "east_line":
        assert np.all(w.geometry.source_x_m > w.geometry.receiver_x_m.max())
    else:
        assert np.sum(w.geometry.source_x_m > 0) == 8
    if name == "switching_array":
        np.testing.assert_array_equal(weights[:3, :8], 1)
        np.testing.assert_array_equal(weights[3:, :8], .2)
        np.testing.assert_array_equal(weights[:3, 8:], .2)
        np.testing.assert_array_equal(weights[3:, 8:], 1)
    else:
        np.testing.assert_array_equal(weights, 1)


def test_worker_count_preserves_active_waveform_and_energy():
    m = load_template_mapping(ROOT / "examples/active_vector_studio.toml")
    results = []
    for workers in (1, 2):
        cfg = deepcopy(m)
        cfg["run"]["workers"] = workers
        w = WaveDispXWorkflow.from_mapping(cfg, base_directory=ROOT / "examples")
        r = run_experiment(w, ExperimentRequest(product="energy", waves="both"))
        results.append((w.shot.vector_xyz, r.images))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-12, atol=1e-14)
    for wave in ("rayleigh", "love"):
        a, b = results[0][1][wave], results[1][1][wave]
        np.testing.assert_array_equal(a.frequency_hz, b.frequency_hz)
        np.testing.assert_array_equal(a.velocity_mps, b.velocity_mps)
        np.testing.assert_allclose(a.energy, b.energy, rtol=1e-12, atol=1e-14)
