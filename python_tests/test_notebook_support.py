from copy import deepcopy
import json
from pathlib import Path

from wavedispx.config import ReceiverFileConfig
from wavedispx.notebook_support import (
    input_error_markdown,
    parameter_reference_markdown,
    prepare_notebook_case,
    required_failure_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def _notebook_parameters(name: str) -> dict:
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "PARAMETERS = {" in "".join(cell["source"])
    )
    namespace: dict = {}
    exec(compile(source, name, "exec"), namespace)
    return namespace["PARAMETERS"]


def test_active_notebook_parameters_support_receiver_and_shot_files(tmp_path):
    parameters = deepcopy(_notebook_parameters("01_active_quickstart.ipynb"))
    parameters["receivers"] = {
        "layout": "file",
        "coordinates_file": "examples/observation_system/receivers.csv",
    }
    parameters["shot_csv"] = "examples/observation_system/shots.csv"
    parameters["shot_id"] = "S002"
    parameters["run"]["output_directory"] = str(tmp_path / "active")
    prepared = prepare_notebook_case(ROOT, "active", parameters)
    assert isinstance(prepared.config.receivers, ReceiverFileConfig)
    assert prepared.config.receivers.count == 21
    assert prepared.config.active_source.x_m == -120.0
    assert prepared.config.active_source.y_m == 0.0


def test_passive_notebook_parameters_support_explicit_geometry_files(tmp_path):
    parameters = deepcopy(_notebook_parameters("02_passive_quickstart.ipynb"))
    parameters["receivers"] = {
        "layout": "file",
        "coordinates_file": "examples/observation_system/receivers.csv",
        "distance_bin_m": 5.0,
    }
    parameters["ambient_sources"] = {
        "layout": "file",
        "count": 48,
        "seed": 42,
        "coordinates_file": "examples/observation_system/noise_sources.csv",
        "angular_sampling": "stratified",
        "component": "vertical",
    }
    parameters["run"]["output_directory"] = str(tmp_path / "passive")
    prepared = prepare_notebook_case(ROOT, "passive", parameters)
    assert isinstance(prepared.config.receivers, ReceiverFileConfig)
    assert prepared.config.receivers.count == 21
    assert prepared.config.ambient_sources.layout == "file"
    assert prepared.workflow.review_inputs().geometry.source_x_m.size == 48


def test_parameter_reference_and_required_failure_are_actionable(tmp_path):
    active_reference = parameter_reference_markdown("active")
    passive_reference = parameter_reference_markdown("passive")
    assert "shot_csv / shot_id" in active_reference
    assert "processing.picker_minimum_energy" in active_reference
    assert "noise.temporal_model" in passive_reference
    assert "ambient_sources.coordinates_file" in passive_reference

    parameters = deepcopy(_notebook_parameters("01_active_quickstart.ipynb"))
    parameters["active_source"]["x_m"] = 5.0
    parameters["run"]["output_directory"] = str(tmp_path / "blocked")
    review = prepare_notebook_case(ROOT, "active", parameters).workflow.review_inputs()
    explanation = required_failure_markdown(review)
    assert "active_source.x_m,y_m" in explanation
    assert "当前值" in explanation
    assert "怎么改" in explanation
    input_explanation = input_error_markdown(
        ValueError("noise frequencies must satisfy minimum < dominant <= maximum")
    )
    assert "噪声时间模型与 PSD" in input_explanation
    assert "怎么改" in input_explanation
    assert "PSD最低频率 < PSD主频" in input_explanation
