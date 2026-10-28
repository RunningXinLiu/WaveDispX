"""Check native window controls and explicit geometry examples without forwarding."""
from pathlib import Path
import argparse
import hashlib
import json
from wavedispx.desktop.studio import StudioWindow
from PySide6.QtWidgets import QApplication, QLineEdit
from PySide6.QtTest import QTest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = QApplication([])
    w = StudioWindow()
    errors = []
    w._show_error = lambda *a: errors.append(str(a))
    w.show_windowed()
    QTest.qWait(300)
    w.compact.setCurrentIndex(w.compact.count()-1)
    workers = w.compact.findChild(QLineEdit, "run.workers")
    workers.setText("2")
    workers.editingFinished.emit()
    assert w.validate_case(navigate=False)
    assert w.workflow.config.workers == 2
    QTest.qWait(250)
    assert w.grab().save(str(args.output/"compute_options.png"))
    report = {"scope": "native Qt input/preview only, no automatic forward", "workers": 2,
              "window_normal": not w.isFullScreen() and not w.isMaximized(), "cases": {}}
    for name in ("east_line", "clusters_array", "switching_array"):
        w.load_studio_example(f"custom_geometry/passive_{name}.toml")
        assert not errors, errors
        assert w._review.ready and w.workflow.correlation is None
        w.compact.setCurrentIndex(2)
        QTest.qWait(250)
        for panel in (w.model_panel, w.geometry_panel):
            panel._canvas.draw()
        assert w.grab().save(str(args.output/(name+".png")))
        g = w.workflow.geometry
        report["cases"][name] = {"sources": len(g.source_x_m), "receivers": len(g.receiver_x_m),
                                  "activity_file": str(w.workflow.config.ambient_sources.activity_file)}
    report["files_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT/"python_src/wavedispx/desktop/studio.py", ROOT/"python_src/wavedispx/desktop/help_text.py", Path(__file__).resolve()]
                              + list((ROOT/"examples/custom_geometry").glob("*.toml"))
                              + list((ROOT/"examples/custom_geometry").glob("*.csv"))}
    (args.output/"UI_VALIDATION.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    w.close()
    app.processEvents()


if __name__ == "__main__":
    main()
