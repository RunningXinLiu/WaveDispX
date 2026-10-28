"""Freeze Studio in an isolated native build; preserve exact build provenance."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
if os.name == 'nt':
    mingw = Path(os.environ['CLASSROOM_MINGW_BIN'])
    if not (mingw/'libgfortran-5.dll').is_file(): raise RuntimeError('Missing native runtime')
    os.environ['PATH'] = str(mingw)+os.pathsep+os.environ.get('PATH','')
manifest = {
    'status': 'Studio unsigned trial candidate; not universal scientific validation',
    'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'platform': platform.platform(), 'python': sys.version,
    'dependencies': {p: importlib.metadata.version(p) for p in ('numpy','scipy','h5py','matplotlib','PySide6','pyinstaller','disba','numba')},
    'sources_sha256': {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('python_src/wavedispx','packaging/studio','cpp','examples')
        for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.toml','.csv','.spec','.txt')},
    'native_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(os.environ['STUDIO_NATIVE']).iterdir() if p.is_file()},
}
(root/'packaging/studio/build_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--clean',str(root/'packaging/studio/studio.spec')],cwd=root,check=True)
