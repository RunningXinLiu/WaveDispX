# Build with CLASSROOM_NATIVE=/absolute/cmake-build/solvers.
import os
import sys
from pathlib import Path
root = Path(SPECPATH).resolve().parents[1]
native = Path(os.environ['CLASSROOM_NATIVE']).resolve()
names = ['forward','sprep96','sdisp96','sdpsrf96']
suffix = '.exe' if os.name == 'nt' else ''
binaries = []
for name in names:
    path = native/(name+suffix)
    if not path.is_file(): raise RuntimeError(f'Missing solver {path}')
    binaries.append((str(path), 'solvers'))
a = Analysis([str(root/'packaging/classroom/entry.py')],
    pathex=[str(root/'python_src')], binaries=binaries,
    datas=[(str(root/'packaging/classroom/reference'),'reference'),
           (str(root/'packaging/classroom/fonts'),'fonts'),
           (str(root/'docs/classroom/two_layer.csv'),'examples'),
           (str(root/'docs/classroom/low_velocity_layer.csv'),'examples'),
           (str(root/'docs/classroom/halfspace.csv'),'examples'),
           (str(root/'docs/classroom/README_zh.md'),'docs'),
           (str(root/'packaging/classroom/vendor/SOURCE_MANIFEST.json'),'provenance'),
           (str(root/'packaging/classroom/build_manifest.json'),'provenance'),
           (str(root/'packaging/classroom/vendor/qed/LICENSE'),'licenses/QEDispInv')],
    hiddenimports=['matplotlib.backends.backend_qtagg'],
    excludes=['torch','tensorflow','streamlit','IPython','notebook','pandas','numba','disba',
              'PyQt5','PyQt6','PySide2','tkinter','pytest','wavedispx.app_service',
              'wavedispx.workflow','wavedispx.processing.theory'],
    noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz,a.scripts,[],exclude_binaries=True,name='WaveDispX-Classroom',
          debug=False,strip=False,upx=False,console=False,disable_windowed_traceback=False)
coll = COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='WaveDispX-Classroom')
if sys.platform == 'darwin':
    app = BUNDLE(coll,name='WaveDispX-Classroom.app',bundle_identifier='org.wavedispx.classroom',
                 info_plist={'NSHighResolutionCapable':True})
