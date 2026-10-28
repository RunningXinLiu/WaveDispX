"""Operational state and display integrity for the Studio redesign."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication, QLineEdit
from wavedispx.desktop.studio import StudioWindow
from wavedispx.app_service import load_template_mapping

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("action", ["source_annulus", "source_line", "source_file"])
def test_source_layout_keeps_physics_and_activity(window, monkeypatch, action):
    from wavedispx.desktop.app import QFileDialog
    m = load_template_mapping(ROOT / "examples/passive_hvsr_studio.toml")
    m["ambient_sources"]["activity_file"] = str(ROOT / "examples/source_activity_anisotropic_quick.csv")
    window._load_mapping(m, ROOT / "examples", "test")
    noise_before = window.parameter_page.mapping()["noise"]
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k:
                        (str(ROOT / "examples/observation_system/noise_sources.csv"), "CSV"))
    window._parameter_action(action)
    source = window.parameter_page.mapping()["ambient_sources"]
    for key in ("component", "polarization", "horizontal_power_fraction", "horizontal_azimuth_deg", "activity_file", "seed"):
        assert source[key] == m["ambient_sources"][key]
    assert source["count"] == (48 if action == "source_file" else 24)
    if action != "source_annulus":
        assert "inner_radius_m" not in source
    assert window.parameter_page.mapping()["noise"] == noise_before


def test_default_window_is_normal_and_fits_screen(window):
    window.show_windowed()
    QApplication.processEvents()
    available = window.screen().availableGeometry()
    assert not window.isFullScreen() and not window.isMaximized()
    assert window.width() <= available.width()
    assert window.height() < available.height()


def test_example_entry_loads_energy_without_hvsr_or_computing(window):
    window.load_studio_example("passive_vector_energy_studio.toml")
    assert window.product_combo.currentData() == "energy_theory"
    assert window.workflow.config.ambient_sources.polarization == "independent_hv"
    assert window.workflow.correlation is None
    assert window.result_tabs.indexOf(window.hvsr_panel) < 0
    window._set_ui_state("running")
    assert not window.examples_button.isEnabled()


def test_runtime_controls_edit_integers_and_keep_physics(window, monkeypatch):
    errors = []
    monkeypatch.setattr(window, "_show_error", lambda *args: errors.append(args))
    before = window.parameter_page.mapping()
    window.compact.setCurrentIndex(window.compact.count()-1)
    for key, value in (("workers", "2"), ("frequency_block_size", "4"), ("source_block_size", "3")):
        field = window.compact.findChild(QLineEdit, "run."+key)
        assert field is not None and field.toolTip()
        field.setText(value)
        field.editingFinished.emit()
        assert window.parameter_page.mapping()["run"][key] == int(value)
        assert type(window.parameter_page.mapping()["run"][key]) is int
    after = window.parameter_page.mapping()
    for key in ("model", "geometry", "active_source", "processing"):
        assert before[key] == after[key]
    for key, bad in (("workers", "1.5"), ("workers", "-1"), ("source_block_size", "0")):
        accepted = window.parameter_page.mapping()["run"][key]
        window.set_value(("run", key), bad, accepted)
        assert window.parameter_page.mapping()["run"][key] == accepted
    assert len(errors) == 3


def test_source_activity_import_and_remove(window, monkeypatch):
    from wavedispx.desktop.studio import QFileDialog
    window.load_studio_example("custom_geometry/passive_clusters_array.toml")
    initial = window.parameter_page.mapping()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k:
                        (str(ROOT / "examples/custom_geometry/source_activity_switching.csv"), "CSV"))
    window._parameter_action("activity_file")
    assert window.parameter_page.mapping()["ambient_sources"]["activity_file"].endswith("source_activity_switching.csv")
    assert any("移除源活动" in a.text() for a in window.import_button.menu().actions())
    window._parameter_action("clear_activity_file")
    assert window.parameter_page.mapping() == initial


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    w = StudioWindow(ROOT / "examples/active_quick.toml")
    w.show()
    app.processEvents()
    yield w
    w._running = False
    w.close()


def test_distinct_primary_action_and_states(window):
    assert window.compute_button.property("role") == "primary"
    assert window.state_chip.property("tone") == "ready"
    assert not window.export_button.isEnabled()
    window._set_ui_state("running")
    for button in (window.compute_button, window.preview_button, window.auto_button,
                   window.import_button, window.save_button, window.open_button):
        assert not button.isEnabled()
    assert window.progress.isVisible()
    window._set_ui_state("error")
    assert window.preview_button.isEnabled()
    assert not window.progress.isVisible()
    assert "处理" in window.state_chip.text()


def test_dirty_immediately_hides_old_canvas(window):
    canvas = window.model_panel._canvas
    toolbar = window.model_panel._toolbar
    window.set_value(("processing", "frequency_step_hz"), ".15", 0.)
    assert canvas.isHidden() and toolbar.isHidden()
    assert window.model_panel._figure is None
    assert window.state_chip.property("tone") == "dirty"
    assert not window.export_button.isEnabled()


def test_collapse_does_not_change_input_or_compute(window):
    before = window.parameter_page.mapping()
    window.parameters_toggle.click()
    assert window.compact.isHidden()
    window.parameters_toggle.click()
    assert not window.compact.isHidden()
    assert window.parameter_page.mapping() == before
    assert window.workflow.active_analysis is None


def test_csv_menu_dispatch_and_das_dependency(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_parameter_action", lambda action: calls.append(action))
    window.import_button.menu().actions()[0].trigger()
    assert calls == ["receiver_file"]
    field = window.compact.findChild(QLineEdit, "das.gauge_length_m")
    assert not field.isEnabled()
    window.set_value(("das", "enabled"), "true", False)
    assert window.compact.findChild(QLineEdit, "das.gauge_length_m").isEnabled()
    assert "DAS" in window.case_badge.text()


def test_new_template_cannot_keep_old_result_state(window):
    window._set_ui_state("result")
    window._load_mapping(load_template_mapping(ROOT / "examples/passive_quick.toml"),
                         ROOT / "examples", "test")
    assert window.state_chip.property("tone") == "dirty"
    assert not window.export_button.isEnabled()
    assert "被动源" in window.case_badge.text()
    assert any("PSD" in a.text() for a in window.import_button.menu().actions())


def test_theory_ui_ignores_invalid_acquisition_and_preserves_tabs(window):
    window.product_combo.setCurrentIndex(window.product_combo.findData("theory"))
    window.waves_combo.setCurrentIndex(window.waves_combo.findData("love"))
    window.set_value(("active_source", "dt_s"), "-1", 0.)
    assert window.validate_case(navigate=False)
    assert window._theory_context.config.mode == "theory"
    assert window.result_tabs.indexOf(window.rayleigh_panel) >= 0
    assert window.result_tabs.indexOf(window.love_panel) >= 0


def test_hvsr_selector_and_source_controls(window):
    m=load_template_mapping(ROOT/"examples/passive_quick.toml")
    m["experiment"]={"product":"hvsr"}
    window._load_mapping(m,ROOT/"examples","test")
    assert window.result_tabs.indexOf(window.hvsr_panel)>=0
    window.set_value(("ambient_sources","component"),"three_component","")
    window.set_value(("ambient_sources","polarization"),"directed_hv","")
    field=window.compact.findChild(QLineEdit,"ambient_sources.horizontal_power_fraction")
    assert field is not None and "功率" in field.toolTip()


def test_hvsr_is_hidden_opt_in_and_can_return_to_main_target(window):
    assert window.product_combo.findData("hvsr") == -1
    assert window.product_combo.findData("energy_hvsr") == -1
    assert window.result_tabs.indexOf(window.hvsr_panel) == -1
    assert window.hvsr_controls.isHidden()
    window.product_combo.setCurrentIndex(window.product_combo.findData("energy"))
    window.extra_analysis_actions["hvsr"].trigger()
    assert window.parameter_page.mapping()["experiment"]["product"] == "hvsr"
    assert window.result_tabs.indexOf(window.hvsr_panel) >= 0
    assert window.compact.findChild(QLineEdit,"experiment.hvsr_smoothing_b") is not None
    window.extra_analysis_actions["off"].trigger()
    assert window.parameter_page.mapping()["experiment"]["product"] == "energy"
    assert window.product_combo.findData("hvsr") == -1
    assert window.result_tabs.indexOf(window.hvsr_panel) == -1
    # deleteLater old controls can remain QObject children until events flush;
    # only the current tab pages are user-visible inputs.
    assert not any(window.compact.widget(i).findChild(QLineEdit,"experiment.hvsr_smoothing_b")
                   for i in range(window.compact.count()))
    assert not any("HVSR" in window.compact.tabText(i) for i in range(window.compact.count()))
    assert window.hvsr_controls.isHidden()
    window._set_ui_state("running")
    assert not window.extra_analysis_button.isEnabled()


def test_minimal_theory_file_does_not_destroy_full_ui(window,monkeypatch):
    m=load_template_mapping(ROOT/"examples/active_quick.toml")
    m.pop("geometry")
    m["experiment"]={"product":"theory"}
    previous=window.parameter_page.mapping()
    errors=[]
    monkeypatch.setattr(window,"_show_error",lambda *args:errors.append(args))
    window._load_mapping(m,ROOT/"examples","test")
    assert errors and window.parameter_page.mapping()==previous


def test_result_export_state_after_real_cpu_result(window):
    result = window.workflow.compute_dispersion()
    before = result.rayleigh_combined.energy.copy()
    window._stage_succeeded("analysis", result)
    assert window.state_chip.property("tone") == "result"
    assert window.export_button.isEnabled()
    window.parameters_toggle.click()
    window.overlay.setChecked(False)
    import numpy as np
    np.testing.assert_array_equal(before, result.rayleigh_combined.energy)
