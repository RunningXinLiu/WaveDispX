"""Real small CPU/QED runs through Qt worker buttons; no bulk output saved."""
from pathlib import Path
import argparse
import hashlib
import json
import time
from PySide6.QtWidgets import QApplication, QLineEdit
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtTest import QTest
from wavedispx.desktop.studio import StudioWindow
from wavedispx.app_service import load_template_mapping

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--skip-hvsr",action="store_true",help="Active/passive R/L and theory recheck; no HVSR acceptance claim")
    parser.add_argument("--keep-open",action="store_true",help="Leave the checked window open for manual use")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    app=QApplication([])
    w=StudioWindow(ROOT/"examples/active_quick.toml")
    errors=[]
    w._show_error=lambda *args: errors.append(str(args))
    w.show_windowed()
    QTest.qWait(500)
    assert not w.isFullScreen() and not w.isMaximized()
    report={"scope":"real Qt worker CPU integration only, no convergence claim","cases":{},
            "window_state": "normal", "logical_window": [w.width(),w.height()],
            "files_sha256": {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (ROOT/"python_src/wavedispx/desktop/studio.py", ROOT/"python_src/wavedispx/desktop/app.py",
                                       ROOT/"python_src/wavedispx/app_service.py", Path(__file__).resolve())}}

    def run(name):
        w.compute_button.click()
        # A native event loop releases the GIL. Repeated QTest.qWait can starve
        # Python-heavy QRunnable work and falsely report a UI compute timeout.
        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(50)
        poll.timeout.connect(lambda: loop.quit() if not w._running else None)
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        poll.start()
        timeout.start(90000)
        if w._running:
            loop.exec()
        poll.stop()
        timeout.stop()
        assert not w._running and not errors, errors
        assert w.experiment_result is not None
        report["cases"][name]={"statuses":w.experiment_result.statuses,"elapsed_s":w.experiment_result.elapsed_s}
        report["cases"][name]["energy_sha256"] = {
            wave: hashlib.sha256(im.energy.tobytes()).hexdigest()
            for wave, im in w.experiment_result.images.items()}

    def capture(name,panel):
        before = {wave:hashlib.sha256(im.energy.tobytes()).hexdigest()
                  for wave,im in w.experiment_result.images.items()}
        w.result_tabs.setCurrentWidget(panel)
        w.render_results()
        QTest.qWait(150)
        assert w.grab().save(str(args.output/(name+".png")))
        assert before == {wave:hashlib.sha256(im.energy.tobytes()).hexdigest()
                          for wave,im in w.experiment_result.images.items()}

    m=load_template_mapping(ROOT/"examples/active_vector_studio.toml")
    w._load_mapping(m,ROOT/"examples","Vector source UI acceptance")
    run("active_vector")
    w.compact.setCurrentIndex(2)
    capture("active_vector",w.rayleigh_panel)
    capture("active_vector_love",w.love_panel)
    field=w.compact.findChild(QLineEdit,"active_source.force_inclination_deg")
    field.setText("90")
    field.editingFinished.emit()
    run("vertical_vector")
    capture("unavailable_love",w.love_panel)
    assert "love" not in w.experiment_result.images

    w.product_combo.setCurrentIndex(w.product_combo.findData("theory"))
    w.waves_combo.setCurrentIndex(w.waves_combo.findData("love"))
    run("theory_only")
    assert w._experiment_workflow.shot is None
    capture("theory_love",w.love_panel)

    w.load_studio_example("passive_vector_energy_studio.toml")
    run("passive_vector_energy")
    assert w.experiment_result.hvsr is None
    assert set(w.experiment_result.images)=={"rayleigh","love"}
    w.compact.setCurrentIndex(2)
    capture("passive_rayleigh",w.rayleigh_panel)
    capture("passive_love",w.love_panel)
    w.show_studio_guide()
    QTest.qWait(250)
    w.dialogs[-1].grab().save(str(args.output/"guide.png"))
    w.dialogs[-1].close()

    if args.skip_hvsr:
        report["scope"]="active/passive R/L and theory Qt check; no HVSR or convergence claim"
        (args.output/"UI_PRODUCTS.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        if args.keep_open:
            return app.exec()
        w.close()
        return

    w._load_mapping(load_template_mapping(ROOT/"examples/passive_hvsr_studio.toml"),ROOT/"examples","HVSR UI acceptance")
    run("hvsr")
    assert w.experiment_result.hvsr.processed_windows==6
    w.compact.setCurrentIndex(2)
    w.hvsr_station.setCurrentIndex(20)
    capture("hvsr",w.hvsr_panel)
    w.wave_toggle.setChecked(True)
    capture("hvsr_raw_xyz",w.wave_panel)
    w.hvsr_station.setCurrentIndex(10)
    assert "Station 10" in w.wave_panel._figure.axes[0].get_title()
    report.update(logical_window=[w.width(),w.height()],device_pixel_ratio=w.devicePixelRatioF(),
                  files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [ROOT/"python_src/wavedispx/desktop/studio.py",Path(__file__),
                              ROOT/"examples/active_vector_studio.toml",ROOT/"examples/passive_hvsr_studio.toml"]})
    (args.output/"UI_PRODUCTS.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    w.close()
    app.processEvents()


if __name__=="__main__": main()
