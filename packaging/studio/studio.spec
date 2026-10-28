import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata
root = Path(SPECPATH).resolve().parents[1]
native = Path(os.environ['STUDIO_NATIVE']).resolve()
names = ['forward.exe', 'qedisp_group.dll'] if os.name == 'nt' else ['forward', 'libqedisp_group.dylib' if sys.platform == 'darwin' else 'libqedisp_group.so']
binaries = []
for name in names:
    path = native/name
    if not path.is_file(): raise RuntimeError(f'Missing solver {path}')
    binaries.append((str(path), 'solvers'))
# disba uses cached numba functions: include its source for cache locators.
dd, db, dh = collect_all('disba', include_py_files=True)
datas = dd + [
    (str(root/'examples'), 'examples'),
    (str(root/'docs/studio'), 'docs/studio'),
    (str(root/'packaging/studio/README_zh.md'), 'docs'),
    (str(root/'packaging/studio/reference'), 'studio_reference'),
    (str(root/'packaging/classroom/fonts'), 'fonts'),
    (str(root/'packaging/classroom/vendor/SOURCE_MANIFEST.json'), 'provenance'),
    (str(root/'packaging/studio/build_manifest.json'), 'provenance'),
    (str(root/'packaging/classroom/THIRD_PARTY_zh.md'), 'licenses'),
    (str(root/'packaging/classroom/vendor/qed/LICENSE'), 'licenses/QEDispInv'),
    (str(root/'Readme.txt'), 'licenses/OriginalProgram'),
]
# Saved experiment provenance hashes the actual Python source files.
datas += [(str(p), str(p.parent.relative_to(root/'python_src')))
          for p in (root/'python_src/wavedispx').rglob('*.py')]
for package in ('numpy','scipy','matplotlib','PySide6','shiboken6','h5py','disba','numba','llvmlite'):
    datas += copy_metadata(package)
a = Analysis([str(root/'packaging/studio/entry.py')], pathex=[str(root/'python_src')],
    binaries=binaries+db, datas=datas,
    hiddenimports=dh+collect_submodules('wavedispx.processing')+['wavedispx._dwm_core','matplotlib.backends.backend_qtagg'],
    excludes=['torch','tensorflow','streamlit','IPython','notebook','PyQt5','PyQt6','PySide2','tkinter','pytest'],
    noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz,a.scripts,[],exclude_binaries=True,name='WaveDispX-Studio',
          debug=False,strip=False,upx=False,console=False,disable_windowed_traceback=False)
coll = COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='WaveDispX-Studio')
if sys.platform == 'darwin':
    app = BUNDLE(coll,name='WaveDispX-Studio.app',bundle_identifier='io.github.runningxinliu.wavedispx.studio',
                 info_plist={'NSHighResolutionCapable':True})
