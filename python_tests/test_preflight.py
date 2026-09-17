import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wavedispx.config import load_config
from wavedispx.preflight import (
    build_preflight_review,
    checks_markdown,
    plot_preflight_review,
    save_preflight_review,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("example", "receiver_count", "source_count"),
    (
        ("active_two_layer.toml", 200, 1),
        ("passive_two_layer.toml", 121, 384),
    ),
)
def test_formal_examples_pass_preflight(example, receiver_count, source_count):
    review = build_preflight_review(load_config(ROOT / "examples" / example))
    assert review.ready
    assert review.geometry.receiver_x_m.size == receiver_count
    assert review.geometry.source_x_m.size == source_count
    assert np.isclose(review.source_power.max(), 1.0)
    assert all(
        check.passed for check in review.checks if check.level == "required"
    )
    assert "Temporal Nyquist" in checks_markdown(review)
    assert review.diagnostics["minimum_theoretical_group_velocity_mps"] > 0.0
    assert review.diagnostics["recommended_frequency_max_hz"] > review.diagnostics[
        "recommended_frequency_min_hz"
    ]


def test_passive_preflight_saves_review_without_forward_modelling(tmp_path):
    review = build_preflight_review(
        load_config(ROOT / "examples/passive_two_layer.toml")
    )
    figure = plot_preflight_review(review)
    assert len(figure.axes) == 6
    plt.close(figure)
    artifacts = save_preflight_review(review, tmp_path)
    assert all(path.is_file() for path in artifacts.values())
    report = json.loads(artifacts["report"].read_text())
    assert report["ready"]
    assert report["diagnostics"]["source_count"] == 384
    assert "results.h5" not in {path.name for path in tmp_path.iterdir()}


@pytest.mark.parametrize(
    ("example", "expected_spacing", "sampling_basis"),
    (
        (
            "manual_active_array.toml",
            0.9107408385430062,
            "median unique source-receiver radial gap",
        ),
        (
            "manual_passive_array.toml",
            5.0,
            "radial pair-distance bin",
        ),
    ),
)
def test_array_preflight_uses_radial_sampling_seen_by_dispersion_imaging(
    example, expected_spacing, sampling_basis
):
    review = build_preflight_review(load_config(ROOT / "examples" / example))
    assert review.ready
    assert np.isclose(
        review.diagnostics["receiver_effective_spacing_m"], expected_spacing
    )
    assert review.diagnostics["spatial_sampling_basis"] == sampling_basis
    alias_check = next(
        check for check in review.checks if check.name == "Spatial alias estimate"
    )
    assert alias_check.passed


def test_required_preflight_failure_names_parameter_and_correction(tmp_path):
    text = (ROOT / "examples/active_quick.toml").read_text()
    text = text.replace("x_m = 0.0\ny_m = 0.0", "x_m = 5.0\ny_m = 0.0")
    path = tmp_path / "source_on_receiver.toml"
    path.write_text(text)
    review = build_preflight_review(load_config(path))
    failure = next(
        check
        for check in review.checks
        if check.name == "Minimum source-receiver distance"
    )
    assert not failure.passed
    assert failure.level == "required"
    assert "active_source.x_m,y_m" in failure.parameter_path
    assert "移动" in failure.guidance
    markdown = checks_markdown(review)
    assert "Parameter" in markdown
    assert "How to fix" in markdown


def test_preflight_exposes_model_aware_range_outside_requested_crop():
    review = build_preflight_review(load_config(ROOT / "examples/active_quick.toml"))
    assert review.theory.frequency_hz[0] < review.config.processing.frequency_min_hz
    assert review.theory.frequency_hz[-1] > review.config.processing.frequency_max_hz
    assert any(
        check.name == "Processing frequency coverage" for check in review.checks
    )
    assert any(check.name == "Velocity scan coverage" for check in review.checks)
    assert "physical_frequency_step_hz" in review.diagnostics


def test_explicit_short_active_record_is_a_required_failure(tmp_path):
    text = (ROOT / "examples/active_quick.toml").read_text()
    text = text.replace(
        "record_duration_s = 0.0             # 0 = model/group-velocity automatic design",
        "record_duration_s = 0.5",
    )
    path = tmp_path / "short_active_record.toml"
    path.write_text(text)
    review = build_preflight_review(load_config(path))
    failure = next(
        check for check in review.checks if check.name == "Active record duration"
    )
    assert not failure.passed
    assert failure.level == "required"
    assert not review.ready


def test_short_passive_window_is_checked_against_group_and_green_time(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text = text.replace("window_s = 10.0", "window_s = 2.0")
    path = tmp_path / "short_passive_window.toml"
    path.write_text(text)
    review = build_preflight_review(load_config(path))
    failure = next(
        check for check in review.checks if check.name == "Correlation window"
    )
    assert not failure.passed
    assert failure.level == "required"


@pytest.mark.parametrize(
    ("example", "receiver_count", "source_count"),
    (
        ("manual_active_line.toml", 200, 1),
        ("manual_active_array.toml", 441, 1),
        ("manual_passive_line.toml", 121, 384),
        ("manual_passive_array.toml", 441, 384),
    ),
)
def test_html_manual_cases_are_runnable_after_preflight(
    example, receiver_count, source_count
):
    review = build_preflight_review(load_config(ROOT / "examples" / example))
    assert review.ready
    assert review.geometry.receiver_x_m.size == receiver_count
    assert review.geometry.source_x_m.size == source_count


def test_notebook_defaults_to_safe_no_forward_mode():
    notebook_path = ROOT / "notebooks/preflight_review.ipynb"
    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(notebook_path), "exec")
    assert "CONFIRM_AND_RUN = False" in code
    assert "if CONFIRM_AND_RUN:" in code
    assert "build_preflight_review" in code


def test_workflow_index_routes_to_small_task_specific_notebooks():
    notebook = json.loads((ROOT / "notebooks/WaveDispX_workflow.ipynb").read_text())
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "01_active_quickstart.ipynb" in markdown
    assert "02_passive_quickstart.ipynb" in markdown
    assert "03_methods_and_validation.ipynb" in markdown
    assert "wavedispx-ui" in markdown


@pytest.mark.parametrize(
    ("name", "stage_marker"),
    (
        ("01_active_quickstart.ipynb", "RUN_FORWARD_STAGE = False"),
        ("02_passive_quickstart.ipynb", "RUN_NCF_STAGE = False"),
    ),
)
def test_quickstart_notebooks_are_executed_and_safely_gated(name, stage_marker):
    notebook_path = ROOT / "notebooks" / name
    notebook = json.loads(notebook_path.read_text())
    errors = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{notebook_path}#cell-{index}", "exec")
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert not errors
    assert all("id" in cell for cell in notebook["cells"])
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert "INPUT_TOML = None" in code
    assert "PARAMETERS = {" in code
    assert "RECEIVER_PRESETS" in code
    assert "prepare_notebook_case" in code
    assert "parameter_reference_markdown" in code
    assert "required_failure_markdown" in code
    assert "plot_preflight_review" in code
    assert stage_marker in code
    assert "RUN_DISPERSION_STAGE = False" in code
    assert "RUN_ALL_NOW = False" in code
    assert "workflow.run_all(save=True)" in code
    assert "Image(filename=str(artifacts.summary_png))" in code


def test_methods_notebook_is_executed_and_documents_both_imaging_families():
    path = ROOT / "notebooks/03_methods_and_validation.ipynb"
    notebook = json.loads(path.read_text())
    content = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert not [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert "E_{PS}(f,c)" in content
    assert "FJ/Bessel" in content
    assert "Hankel" in content
    assert "Poisson" in content
    assert "stationary_gaussian" in content
    assert "noise_events.csv" in content
    assert "逐噪声源先做理想 Green 互相关" in content
    assert "plot_receiver_layout_comparison" in content
