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


def configure_ui_font(app):
    """Use the shipped Chinese font, independent of OS language packs."""
    if app.property('classroom_font_family'):
        return
    font = resource_root()/'fonts/NotoSansSC.ttf'
    if not font.is_file():
        if getattr(sys, 'frozen', False):
            raise FileNotFoundError('发行包缺少中文字体 NotoSansSC.ttf')
        return  # Development checkout can use its existing system fonts.
    from PySide6.QtGui import QFont, QFontDatabase
    identifier = QFontDatabase.addApplicationFont(str(font))
    families = QFontDatabase.applicationFontFamilies(identifier)
    if identifier < 0 or not families:
        raise RuntimeError('无法载入内置中文字体。')
    app.setFont(QFont(families[0], 10))
    app.setProperty('classroom_font_family', families[0])
