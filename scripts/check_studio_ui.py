"""Capture a real CPU/QED quick result without saving numerical bulk output."""
from pathlib import Path
import time
import hashlib
import json
import subprocess
import sys
import traceback
import argparse
from PySide6.QtWidgets import QApplication, QToolTip, QLineEdit
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from wavedispx.desktop.studio import StudioWindow
from wavedispx.app_service import toml_dumps
from wavedispx.processing import find_qedisp_forward
from wavedispx.core import _dwm_core

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ROOT / "docs/studio")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = QApplication([])
w = StudioWindow(ROOT / "examples/active_quick.toml")
w.showFullScreen()
started = time.monotonic()

def capture_impl():
    if w._running:
        if time.monotonic() - started > 90:
            raise TimeoutError("UI CPU smoke run exceeded 90 seconds")
        QTimer.singleShot(200, capture)
        return
    assert w.workflow.active_analysis is not None
    w.notice.setText("真实CPU主动quick结果 · QED理论曲线 · 未保存大型波形")
    app.processEvents()
    path = args.output / "active_cpu_qed.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    w.grab().save(str(path))
    energy = w.workflow.active_analysis.rayleigh_combined.energy
    before = hashlib.sha256(energy.tobytes()).hexdigest()
    w.overlay.setChecked(False)
    app.processEvents()
    w.grab().save(str(path.with_name("energy_without_overlay.png")))
    w.overlay.setChecked(True)
    after = hashlib.sha256(energy.tobytes()).hexdigest()
    assert before == after
    w.compact.setCurrentIndex(1)
    app.processEvents()
    w.grab().save(str(path.with_name("geometry_controls.png")))
    w.compact.setCurrentIndex(2)
    app.processEvents()
    w.grab().save(str(path.with_name("source_controls.png")))
    field = w.compact.findChild(QLineEdit, "active_source.maximum_frequency_hz")
    QToolTip.showText(field.mapToGlobal(field.rect().center()), field.toolTip(), field)
    QTest.qWait(250)
    # A tooltip is a separate top-level window; capture its real rendered widget.
    for widget in app.topLevelWidgets():
        if widget.isVisible() and widget.objectName() == "qtooltip_label":
            widget.grab().save(str(path.with_name("cutoff_tooltip.png")))
    QToolTip.hideText()
    w.compact.setCurrentIndex(3)
    app.processEvents()
    w.grab().save(str(path.with_name("imaging_controls.png")))
    w.parameters_toggle.click()
    app.processEvents()
    w.grab().save(str(path.with_name("focused_results.png")))
    w.parameters_toggle.click()
    w.wave_toggle.setChecked(True)
    w.result_tabs.setCurrentWidget(w.wave_panel)
    app.processEvents()
    w.grab().save(str(path.with_name("waveforms.png")))
    config_path = path.with_name("checked_input.toml")
    config_path.write_text(toml_dumps(w.parameter_page.mapping()), encoding="utf-8")
    files = [ROOT / "python_src/wavedispx/desktop/studio.py", ROOT / "python_src/wavedispx/desktop/studio_style.py", Path(__file__).resolve(),
             ROOT / "python_src/wavedispx/processing/qedisp.py", ROOT / "python_src/wavedispx/forward.py",
             ROOT / "python_src/wavedispx/workflow.py", Path(_dwm_core.__file__),
             Path(find_qedisp_forward()), ROOT / "examples/active_quick.toml", config_path]
    report = {
        "scope": "UI smoke only; actual active quick CPU/QED, no convergence certification",
        "python": sys.executable,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree": "contains pre-existing uncommitted classroom/QED work; HEAD alone is not the running version",
        "energy_sha256_before_display": before, "energy_sha256_after_display": after,
        "shape": list(energy.shape), "fullscreen": w.isFullScreen(),
        "logical_window": [w.width(), w.height()], "device_pixel_ratio": w.devicePixelRatioF(),
        "files_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    path.with_name("UI_VALIDATION.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    w.set_value(("processing", "frequency_step_hz"), "0.15", 0.0)
    app.processEvents()
    w.grab().save(str(path.with_name("modified_input.png")))
    from wavedispx.app_service import load_template_mapping
    w._load_mapping(load_template_mapping(ROOT / "examples/passive_das_grid_quick.toml"), ROOT / "examples", "Passive DAS preview only")
    w.validate_case(navigate=False)
    w.compact.setCurrentIndex(2)
    app.processEvents()
    w.grab().save(str(path.with_name("passive_source_controls.png")))
    w.close()
    app.quit()

def capture():
    try:
        capture_impl()
    except Exception:
        traceback.print_exc()
        app.exit(1)

def run():
    app.processEvents()
    w.grab().save(str(args.output / "initial.png"))
    w.compute_button.click()
    app.processEvents()
    w.grab().save(str(args.output / "running.png"))
    QTimer.singleShot(200, capture)

QTimer.singleShot(500, run)
raise SystemExit(app.exec())
