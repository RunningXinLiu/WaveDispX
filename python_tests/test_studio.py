"""Small UI checks, no large data or output runs."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication
from wavedispx.desktop.studio import StudioWindow

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_studio_parameters_and_visibility():
    app = QApplication.instance() or QApplication([])
    w = StudioWindow(ROOT / "examples/active_quick.toml")
    w.resize(1728, 1080)
    w.show()
    app.processEvents()
    assert w.parameter_page.mapping()["processing"]["theory_solver"] == "qedisp"
    assert w.model_panel.isVisible() and w.geometry_panel.isVisible()
    assert w.result_tabs.count() == 2
    w.wave_toggle.setChecked(True)
    assert w.result_tabs.count() == 3
    w.wave_toggle.setChecked(False)
    assert w.result_tabs.count() == 2
    w.set_value(("processing", "frequency_step_hz"), "0.15", 0.0)
    assert w.parameter_page.mapping()["processing"]["frequency_step_hz"] == .15
    assert w._dirty
    w.close()


@pytest.mark.parametrize("template", ["passive_quick", "active_das_grid_quick"])
def test_studio_passive_and_das_preview(template):
    from PySide6.QtWidgets import QLineEdit
    w = StudioWindow(ROOT / f"examples/{template}.toml")
    assert w._review.ready
    assert w.workflow.config.processing.theory_solver == "qedisp"
    assert w.compact.findChild(QLineEdit, "processing.frequency_step_hz")
    if template.startswith("passive"):
        assert w.compact.findChild(QLineEdit, "noise.maximum_lag_s")
        assert w.compact.findChild(QLineEdit, "noise.duration_s")
    else:
        assert w._review.das_geometry.channel_count == 121
        edit = w.compact.findChild(QLineEdit, "das.gauge_length_m")
        assert "约束" in edit.toolTip() or "必须" in edit.toolTip()
        w.set_value(("das", "gauge_length_m"), "12.0", 0.0)
        assert w.validate_case(navigate=False)
        assert w.workflow.config.das.gauge_length_m == 12.0
    w.close()


def test_studio_reverts_invalid_text_and_invalidates_stale_results(monkeypatch):
    from PySide6.QtWidgets import QLineEdit
    w = StudioWindow(ROOT / "examples/active_quick.toml")
    errors = []
    monkeypatch.setattr(w, "_show_error", lambda *args: errors.append(args))
    old = w.parameter_page.mapping()["processing"]["velocity_step_mps"]
    w.set_value(("processing", "velocity_step_mps"), "bad", 0.0)
    assert errors
    edit = w.compact.findChild(QLineEdit, "processing.velocity_step_mps")
    assert float(edit.text()) == old
    w.set_value(("model", "layers", 0, "vs_mps"), "-1", 0.0)
    assert not w.validate_case(navigate=False)
    assert w._dirty and w.model_panel._figure is None
    w.close()


def test_studio_active_computation_qed_and_overlay(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = StudioWindow(ROOT / "examples/active_quick.toml")
    assert w.validate_case(navigate=False)
    result = w.workflow.compute_dispersion()
    before = result.rayleigh_combined.energy.copy()
    w.render_results()
    assert w.rayleigh_panel._figure is not None
    w.overlay.setChecked(False)
    import numpy as np
    np.testing.assert_array_equal(before, result.rayleigh_combined.energy)
    assert not w.rayleigh_panel._figure.axes[0].lines
    w.show_waveforms(True)
    assert w.wave_panel._figure is not None
    w.close()
