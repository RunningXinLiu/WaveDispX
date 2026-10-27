"""Resolve solver resources without user PATH configuration in a frozen app."""
import os
from pathlib import Path
import sys


def resource_root():
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[3]))


def bundled_solver(name):
    if not getattr(sys, 'frozen', False):
        return None
    base = resource_root() / 'solvers'
    path = base / ('forward' if name == 'qedisp' else name)
    if os.name == 'nt':
        path = path.with_suffix('.exe')
    if not path.is_file():
        raise FileNotFoundError(f'发行包缺少内置求解器：{path.name}')
    return path


def prepare_runtime():
    if getattr(sys, 'frozen', False):
        # DLLs collected alongside Python extensions must also be available to
        # separately executed native solvers. No user environment is modified.
        root = resource_root()
        os.environ['PATH'] = os.pathsep.join([str(root/'solvers'), str(root), os.environ.get('PATH','')])
