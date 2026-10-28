"""Interaction regressions; numerical acceptance is run separately on CPU."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from wavedispx.desktop.studio import StudioWindow
from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit
from wavedispx.app_service import load_template_mapping
from wavedispx.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    w = StudioWindow(ROOT / "examples/active_quick.toml")
    errors = []
    monkeypatch.setattr(w, "_show_error", lambda *args: errors.append(args))
    w.test_errors = errors
    yield w
    w._running = False
    w.close()


def test_integer_literal_does_not_make_physical_parameter_integer_only(window):
    mapping = window.parameter_page.mapping()
    mapping["geometry"]["receivers"]["spacing_m"] = 5
    window._load_mapping(mapping, ROOT / "examples", "integer literals")
    field = window.compact.findChild(QLineEdit, "geometry.receivers.spacing_m")
    field.setText("2.5")
    field.editingFinished.emit()
    assert not window.test_errors
    assert window.parameter_page.mapping()["geometry"]["receivers"]["spacing_m"] == 2.5


def test_passive_love_uses_existing_tt_image(window):
    mapping = load_template_mapping(ROOT / "examples/passive_quick.toml")
    mapping["ambient_sources"]["component"] = "three_component"
    window._load_mapping(mapping, ROOT / "examples", "passive")
    assert window.validate_case(navigate=False)
    frequency = np.array([4., 5.])
    image = SimpleNamespace(frequency_hz=frequency, velocity_mps=np.array([200., 300.]),
                            energy=np.array([[.2, .3], [.9, 1.]]))
    theory = SimpleNamespace(rayleigh_mps=np.array([[210.], [200.]]),
                             love_mps=np.array([[280.], [270.]]))
    window.workflow.passive_analysis = SimpleNamespace(dispersion=image, theory=theory,
                                                       component_images={"tt": image})
    window.result_tabs.setCurrentWidget(window.love_panel)
    window.render_results()
    assert window.love_panel._figure is not None
    np.testing.assert_array_equal(window.love_panel._figure.axes[0].collections[0].get_array(), image.energy)


def test_logs_can_be_opened_during_computation(window):
    window._running = True
    window.select_page("run")
    assert window.run_page.isVisible()


def test_save_as_preserves_relative_geometry_target(window, monkeypatch, tmp_path):
    window._load_mapping(load_template_mapping(ROOT / "examples/active_das_channel_file_quick.toml"),
                         ROOT / "examples", "CSV")
    destination = tmp_path / "elsewhere.toml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(destination), "TOML"))
    window.save_toml()
    assert not window.test_errors
    restored = load_config(destination)
    assert restored.receivers.coordinates_file == ROOT / "examples/das_channels_bent_example.csv"


def test_save_button_does_not_start_unrequested_forward(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window.thread_pool, "start", lambda worker: calls.append(worker))
    window.start_stage("save")
    window._running = False
    assert not calls


def test_invalid_import_preserves_existing_case(window):
    before = window.parameter_page.mapping()
    window._load_mapping({"run": {"mode": "active"}}, ROOT, "incomplete")
    assert window.test_errors
    assert window.parameter_page.mapping() == before


def test_reopening_logs_reuses_visible_page(window):
    window.select_page("run")
    original = window.run_page.parentWidget()
    original.close()
    window.select_page("run")
    assert window.run_page.parentWidget() is original
    assert window.run_page.isVisible()


def test_sampling_summary_distinguishes_passive_total_window_and_lag(window):
    mapping = load_template_mapping(ROOT / "examples/passive_quick.toml")
    window._load_mapping(mapping, ROOT / "examples", "passive")
    assert window.validate_case(navigate=False)
    for term in ("总记录 60", "独立窗 10", "NCF lag", "扫描 df"):
        assert term in window.sampling_summary.text()


def test_csv_receiver_and_shot_import_keep_actual_coordinates(window, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    source = ROOT / "examples/observation_system/receivers.csv"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(source), "CSV"))
    window._parameter_action("receiver_file")
    mapping = window.parameter_page.mapping()
    assert mapping["geometry"]["receivers"]["coordinates_file"] == str(source)
    source = ROOT / "examples/observation_system/shots.csv"
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: (args[3][0], True))
    window._parameter_action("shot_file")
    assert not window.test_errors
    assert window.validate_case(navigate=False)


def test_poisson_controls_and_file_layout_alias_are_not_misrepresented(window):
    from PySide6.QtWidgets import QComboBox
    mapping = load_template_mapping(ROOT / "examples/passive_quick.toml")
    mapping["ambient_sources"] = dict(component="vertical", layout="road", count=24,
                                      x_min_m=-500., x_max_m=500., y_m=300., seed=42)
    window._load_mapping(mapping, ROOT / "examples", "road")
    combo = window.compact.findChild(QComboBox, "ambient_sources.layout")
    assert combo.currentData() == "road"
    window.set_value(("noise", "temporal_model"), "poisson_ricker", "stationary_gaussian")
    assert window.compact.findChild(QLineEdit, "noise.event_rate_hz") is not None


@pytest.mark.parametrize("filename", [
    "active_quick", "manual_active_array", "passive_quick", "passive_grid_quick",
    "active_das_line_quick", "active_das_grid_quick", "passive_das_line_quick", "passive_das_grid_quick",
])
def test_eight_routes_preview_and_editable_axes(app, filename):
    w = StudioWindow(ROOT / f"examples/{filename}.toml")
    try:
        assert w._review.ready
        for key in ("frequency_min_hz", "frequency_max_hz", "frequency_step_hz",
                    "velocity_min_mps", "velocity_max_mps", "velocity_step_mps"):
            assert w.compact.findChild(QLineEdit, "processing." + key)
        assert w.parameter_page.mapping()["processing"]["theory_solver"] == "qedisp"
    finally:
        w.close()
