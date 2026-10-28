"""Display the saved server HVSR example; never regenerate noise for screenshots."""
import argparse
from dataclasses import fields
from pathlib import Path
import hashlib
import json
import numpy as np
from wavedispx.desktop.studio import StudioWindow
from wavedispx.hvsr import HVSRResult
from wavedispx.geometry import Geometry
from wavedispx.experiment import ExperimentRequest, ExperimentResult
from wavedispx.app_service import load_template_mapping
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QPoint, Qt
from PySide6.QtTest import QTest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"docs/studio/hvsr_multistation_20261028"
INPUT=ROOT/"docs/studio/products_v1_20261028/server_final/manifests/passive_directed_h/experiment_1793169186538740127/input.toml"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--keep-open",action="store_true")
    parser.add_argument("--output",type=Path,default=DATA/"screenshots")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    app=QApplication([])
    w=StudioWindow(ROOT/"examples/passive_hvsr_studio.toml")
    mapping=load_template_mapping(INPUT)
    # Runtime-only local destination. Original server input remains immutable.
    mapping["run"]["output_directory"]=str(ROOT/"output_python/hvsr_interactive_demo")
    w._load_mapping(mapping,INPUT.parent,"读取已保存的服务器HVSR结果；未重新正演")
    assert w.validate_case(navigate=False)
    with np.load(DATA/"hvsr_display_only.npz",allow_pickle=False) as z:
        values={f.name:z["hvsr_"+f.name].copy() for f in fields(HVSRResult)}
        for key in ("processed_windows","smoothing_b","vertical_floor_ratio"):
            values[key]=values[key].item()
        for key in ("receiver_x_m","receiver_y_m","source_x_m","source_y_m"):
            np.testing.assert_allclose(getattr(w.workflow.geometry,key),z["geometry_"+key],rtol=0,atol=1e-10)
        # Display the exact saved coordinates, not regenerated libm roundoff.
        w.workflow.geometry=Geometry(**{f.name:z["geometry_"+f.name].copy() for f in fields(Geometry)})
    h=HVSRResult(**values)
    def digest():
        return hashlib.sha256(b"".join(np.asarray(getattr(h,f.name)).tobytes() for f in fields(h))).hexdigest()
    before=digest()
    w._experiment_workflow=w.workflow
    result=ExperimentResult(ExperimentRequest(product="hvsr"),hvsr=h,statuses={"rayleigh":"本例仅HVSR","love":"本例仅HVSR"})
    w._stage_succeeded("experiment",result)
    w.setWindowTitle("WaveDispX Studio · 多台站HVSR · 已保存服务器算例")
    w.show_windowed()

    def capture():
        try:
            w.parameters_toggle.setChecked(True)
            w._select_hvsr_station(20)
            note="已保存算例｜41台站 / 24噪声源 / 60 s；只读取HVSR统计，无新增正演；短记录非收敛认证。"
            w.notice.setText(note)
            def shot(name):
                QTest.qWait(250)
                w.model_panel._canvas.draw()
                w.geometry_panel._canvas.draw()
                w.hvsr_panel._canvas.draw()
                assert w.grab().save(str(args.output/(name+"_window.png")))
                w.hvsr_panel._figure.savefig(args.output/(name+"_plot.png"),dpi=150)
            shot("single")
            # A real Qt mouse click on the receiver canvas, with Retina scaling.
            canvas=w.geometry_panel._canvas
            ax=w.geometry_panel._figure.axes[0]
            g=w.workflow.geometry
            x,y=ax.transData.transform((g.receiver_x_m[10],g.receiver_y_m[10]))
            ratio=canvas.devicePixelRatioF()
            QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,
                             QPoint(round(x/ratio),round(canvas.height()-y/ratio)))
            assert w.hvsr_station.currentIndex()==10
            w.hvsr_view.setCurrentIndex(w.hvsr_view.findData("overlay"))
            shot("overlay")
            w.hvsr_view.setCurrentIndex(w.hvsr_view.findData("heatmap"))
            shot("heatmap")
            assert before==digest()
            report={"scope":"read-only multi-station display; no new forward calculation", "input":str(INPUT),
                "snapshot_sha256":hashlib.sha256((DATA/"hvsr_display_only.npz").read_bytes()).hexdigest(),
                "hvsr_arrays_sha256_before":before,"hvsr_arrays_sha256_after":digest(),
                "qt_geometry_click_station":10,"selected_overlay_stations":w._hvsr_selected,
                "logical_window":[w.width(),w.height()],"device_pixel_ratio":w.devicePixelRatioF(),
                "runtime_output_override":mapping["run"]["output_directory"],
                "files_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [ROOT/"python_src/wavedispx/desktop/studio.py",ROOT/"python_src/wavedispx/desktop/hvsr_view.py",Path(__file__)]}}
            (args.output/"VALIDATION.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
            w.hvsr_view.setCurrentIndex(w.hvsr_view.findData("overlay"))
            w.notice.setText(note+" 点左侧接收器看单站，或切换全台站热图。")
            if not args.keep_open:
                w.close()
                app.quit()
        except Exception:
            import traceback
            traceback.print_exc()
            app.exit(1)
    QTimer.singleShot(800,capture)
    raise SystemExit(app.exec())


if __name__=="__main__": main()
