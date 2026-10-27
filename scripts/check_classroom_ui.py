#!/usr/bin/env python3
"""Exercise backend selection and lazy ideal in a real Qt window; keep v1 QC intact."""
import hashlib
import json
from pathlib import Path
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from wavedispx.classroom.app import ClassroomWindow
from wavedispx.classroom.plotting import export_result

out=Path(__file__).resolve().parents[1]/'docs'/'classroom'/'qc_v2'
out.mkdir(parents=True,exist_ok=True)
app=QApplication(sys.argv)
window=ClassroomWindow(); window.show()
window.presets.setCurrentIndex(1)
state={'phase':'curves'}


def curve_hash():
    return {w:hashlib.sha256(window.result['curves'][w].tobytes()).hexdigest() for w in ('rayleigh','love')}


def capture(name):
    for canvas in window.canvases.values(): canvas.draw()
    window.grab().save(str(out/name))


def inspect():
    if window.busy or window.result is None: return
    if state['phase']=='curves':
        assert window.result['response'] is None and not window.energy.isChecked()
        state['before_ideal']=curve_hash()
        capture('desktop_curves.png')
        export_result(out/'qedisp_curves.zip',window.result)
        state['phase']='ideal'
        window.energy.setChecked(True)
    elif state['phase']=='ideal':
        assert window.result['response'] is not None
        state['after_ideal']=curve_hash()
        assert state['before_ideal']==state['after_ideal']
        capture('desktop_rayleigh_ideal.png')
        window.tabs.setCurrentIndex(1)
        capture('desktop_love_ideal.png')
        export_result(out/'qedisp_ideal.zip',window.result)
        state['phase']='compact'
        window.resize(1200,780)
    elif state['phase']=='compact':
        capture('desktop_compact.png')
        window.resize(1440,900)
        window.energy.setChecked(False)
        window.backend.setCurrentIndex(1)
        state['phase']='cps'
        window.start()
    elif state['phase']=='cps':
        assert window.result['backend']=='cps' and window.result['response'] is None
        capture('desktop_cps.png')
        export_result(out/'cps_curves.zip',window.result)
        (out/'ui_checks.json').write_text(json.dumps(state,indent=2))
        timer.stop(); app.quit()

window.run_button.click()
timer=QTimer(); timer.timeout.connect(inspect); timer.start(500)
QTimer.singleShot(180000,lambda:app.exit(2))
sys.exit(app.exec())
