"""Studio bundle paths only; no changes to physical or numerical settings."""
from copy import deepcopy
import os
from pathlib import Path
import sys


def resource_root():
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[3]))


def runtime_mapping(mapping, base):
    """Built-in bundle templates save outside the read-only application folder."""
    result = deepcopy(mapping)
    if not getattr(sys, 'frozen', False):
        return result
    base = Path(base).resolve()
    builtins = (resource_root()/'examples').resolve()
    if not base.is_relative_to(builtins):
        return result  # User-supplied TOML paths remain explicit and unchanged.
    raw = result.get('run', {}).get('output_directory')
    if raw and not Path(raw).is_absolute():
        from PySide6.QtCore import QStandardPaths
        documents = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        target = Path(documents or Path.home())/'WaveDispX'/'results'
        # Use the existing output basename, not unchecked user path fragments.
        result['run']['output_directory'] = str(target/Path(raw).name)
    return result


def prepare_runtime():
    if not getattr(sys, 'frozen', False):
        return
    from wavedispx.classroom.runtime import prepare_runtime as prepare_native
    prepare_native()
    root = resource_root()
    forward = root/'solvers'/('forward.exe' if os.name == 'nt' else 'forward')
    libname = 'qedisp_group.dll' if os.name == 'nt' else 'libqedisp_group.dylib' if sys.platform == 'darwin' else 'libqedisp_group.so'
    group = root/'solvers'/libname
    for p in (forward, group, root/'examples/active_quick.toml'):
        if not p.is_file():
            raise FileNotFoundError(f'发行包缺少必要文件：{p.name}')
    os.environ['QEDISPINV_FORWARD'] = str(forward)
    os.environ['QEDISPINV_GROUP_LIBRARY'] = str(group)
    from PySide6.QtCore import QStandardPaths
    cache = Path(QStandardPaths.writableLocation(QStandardPaths.GenericCacheLocation))/'WaveDispX'/'numba'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['NUMBA_CACHE_DIR'] = str(cache)
