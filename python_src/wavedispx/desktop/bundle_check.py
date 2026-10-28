"""Exercise packaged Studio UI, native backends, raw arrays and saved results."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
from .runtime import resource_root
from .bundle_cases import CASES, case_mapping, arrays, compare


def main(output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'platform':platform.platform(),'cases':{}}
    window=None
    try:
        from .studio import StudioWindow
        from wavedispx.classroom.runtime import configure_ui_font
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QEventLoop,QTimer
        app=QApplication.instance() or QApplication([])
        configure_ui_font(app)
        root=resource_root()
        reference=root/'studio_reference'
        if not reference.exists(): reference=root/'packaging/studio/reference'
        reference_files=json.loads((reference/'FILES.json').read_text(encoding='utf-8'))['cases']
        window=StudioWindow()
        window.show_windowed()
        window.resize(1600,960)  # Fixed QC client size even on offscreen CI screens.
        window.about_button.click()
        app.processEvents()
        assert 'Pan Lei' in window._about_dialog.browser.toPlainText()
        assert 'https://runningxinliu.github.io/' in window.author_link.text()
        window._about_dialog.grab().save(str(output/'about.png'))
        window._about_dialog.close()
        if getattr(sys,'frozen',False):
            from PySide6.QtGui import QRawFont
            font=QRawFont.fromFont(app.font())
            assert all(font.supportsCharacter(ord(c)) for c in '频散波场模型密度来源计算'), 'Missing Chinese font'
            report['font']=font.familyName()
        errors=[]
        window._show_error=lambda *a:errors.append(str(a))
        original=window._stage_failed
        def failed(*a):
            errors.append(str(a))
            original(*a)
        window._stage_failed=failed
        def settle():
            loop=QEventLoop()
            poll=QTimer()
            poll.timeout.connect(lambda:loop.quit() if not window._running else None)
            poll.start(30)
            timeout=QTimer()
            timeout.setSingleShot(True)
            timeout.timeout.connect(loop.quit)
            timeout.start(600000)
            loop.exec()
            poll.stop();timeout.stop()
            assert not window._running,'Packaged calculation exceeded 600s'
            assert not errors,errors
        for name in CASES:
            start=time.monotonic()
            mapping,base=case_mapping(root,name,output/'学生 结果'/name)
            window._load_mapping(mapping,base,'打包验收：'+name)
            assert window.validate_case(navigate=False),name
            window.compute_button.click()
            settle()
            result=window.experiment_result
            assert result is not None,name
            actual=arrays(window.workflow,result)
            entry=reference_files[name]
            path=reference/entry['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'], 'Baseline hash'
            with np.load(path,allow_pickle=False) as expected:
                baseline={k:expected[k] for k in expected.files}
            metrics=compare(actual,baseline)
            window.render_results()
            window.wave_toggle.setChecked(name!='hvsr')
            app.processEvents()
            window.grab().save(str(output/(name+'.png')))
            if 'love' in result.images:
                window.result_tabs.setCurrentIndex(1)
                app.processEvents()
                window.grab().save(str(output/(name+'_love.png')))
            window.start_stage('save')
            settle()
            saved=list((output/'学生 结果'/name).glob('experiment_*/manifest.json'))
            assert len(saved)==1,'Saved result missing'
            manifest=json.loads(saved[0].read_text(encoding='utf-8'))
            assert hashlib.sha256(saved[0].with_name('arrays.npz').read_bytes()).hexdigest()==manifest['arrays_sha256']
            from wavedispx.app_service import parse_toml_bytes
            reloaded=parse_toml_bytes(saved[0].with_name('input.toml').read_bytes())
            assert reloaded['model']==mapping['model']
            from wavedispx.config import load_config
            from wavedispx.geometry import receiver_positions
            restored=load_config(saved[0].with_name('input.toml'))
            for a,b in zip(receiver_positions(restored.receivers),receiver_positions(window.workflow.config.receivers)):
                np.testing.assert_array_equal(a,b)
            for path,digest in manifest.get('external_inputs_sha256',{}).items():
                assert hashlib.sha256((saved[0].parent/path).read_bytes()).hexdigest()==digest
            report['cases'][name]={'passed':True,'elapsed_s':time.monotonic()-start,'relative_L2':metrics,'statuses':result.statuses,'manifest':manifest}
            print('PASS',name,flush=True)
        # Confirm the default compatible entry can calculate as well.
        window.load_studio_example('active_quick.toml')
        window.compute_button.click()
        settle()
        assert window.workflow.active_analysis is not None
        report['legacy_active_default']='passed'
        report['passed']=True
    except Exception:
        report['traceback']=traceback.format_exc()
        print(report['traceback'],flush=True)
    finally:
        if window is not None and not window._running:
            window.close()
        (output/'self_test.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return 0 if report['passed'] else 1
