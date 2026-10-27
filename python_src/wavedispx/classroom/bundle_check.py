"""Run the shipped GUI, bundled solvers and ideal operator on a clean machine."""
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback

import numpy as np

from .runtime import resource_root


def main(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'platform': platform.platform(), 'passed': False, 'checks': {}}
    old_temp = tempfile.tempdir
    try:
        # Exercise subprocess cwd handling with a non-ASCII, space-containing path.
        work = output / '学生 临时计算'
        work.mkdir(exist_ok=True)
        tempfile.tempdir = str(work)
        from .app import ClassroomWindow
        from .plotting import export_result
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        root = resource_root()
        reference = root/'reference'
        if not reference.is_dir():
            reference = root/'packaging/classroom/reference'
        expected = np.load(reference/'baseline.npz')
        window = ClassroomWindow()
        if getattr(__import__('sys'), 'frozen', False):
            from PySide6.QtGui import QRawFont
            font = QRawFont.fromFont(window.font())
            assert font.isValid(), 'Invalid UI font'
            missing = [c for c in '频散曲线模型密度计算后端VpVs0123456789ρ' if not font.supportsCharacter(ord(c))]
            assert not missing, f'Missing UI font glyphs: {missing}'
            report['checks']['ui_font'] = {'family': font.familyName(), 'glyph_coverage_passed': True}
        errors = []

        def failed(message):
            window.unlock()
            errors.append(message)

        window.failed = failed
        window.presets.setCurrentIndex(1)
        window.nc.setValue(len(expected['velocity_mps']))
        window.show()

        def settle():
            deadline = time.monotonic()+180
            while window.busy and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.02)
            app.processEvents()
            if window.busy:
                raise TimeoutError('GUI calculation exceeded 180 seconds')
            if errors:
                raise RuntimeError(errors[-1])

        def capture(name):
            for canvas in window.canvases.values():
                canvas.draw()
            app.processEvents()
            if not window.grab().save(str(output/name)):
                raise RuntimeError('Could not save GUI screenshot')

        assert not window.energy.isChecked()
        for index, backend in enumerate(('qedisp', 'cps')):
            window.backend.setCurrentIndex(index)
            window.run_button.click()
            settle()
            result = window.result
            assert result is not None and result['backend'] == backend
            assert result['response'] is None
            errors_mps = {}
            for wave in ('rayleigh', 'love'):
                actual = result['curves'][wave]
                baseline = expected[f'{backend}_{wave}']
                np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(baseline))
                np.testing.assert_allclose(actual, baseline, atol=.002, rtol=2e-6, equal_nan=True)
                errors_mps[wave] = float(np.nanmax(abs(actual-baseline)))
            report['checks'][backend] = {'max_error_mps': errors_mps, 'binaries': result['binaries']}
            capture(backend+'_curves.png')
            export_result(output/(backend+'_curves.zip'), result)

        before = {w: a.copy() for w, a in window.result['curves'].items()}
        window.energy.setChecked(True)
        settle()
        assert window.result['response'] is not None
        for wave in ('rayleigh', 'love'):
            np.testing.assert_array_equal(before[wave], window.result['curves'][wave])
            np.testing.assert_allclose(window.result['response'][wave], expected['ideal_'+wave],
                                       rtol=1e-7, atol=1e-18)
        capture('rayleigh_ideal.png')
        window.tabs.setCurrentIndex(1)
        capture('love_ideal.png')
        export_result(output/'cps_ideal.zip', window.result)
        window.energy.setChecked(False)
        for wave in before:
            np.testing.assert_array_equal(before[wave], window.result['curves'][wave])
        window.resize(1200, 780)
        capture('compact_curves.png')
        report['checks']['ideal'] = {'curves_unchanged': True, 'complex_response_matches': True}
        report['checks']['unicode_temp_directory'] = True
        report['checks']['frozen'] = bool(getattr(__import__('sys'), 'frozen', False))
        report['passed'] = True
        window.close()
        return 0
    except Exception:
        report['error'] = traceback.format_exc()
        return 1
    finally:
        tempfile.tempdir = old_temp
        (output/'self_test.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
