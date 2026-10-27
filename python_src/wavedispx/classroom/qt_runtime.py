"""Keep wheel Qt plugins with wheel Qt libraries in mixed Conda environments."""
import os
from pathlib import Path
import sys


def configure_qt():
    os.environ['QT_API']='pyside6'
    if getattr(sys, 'frozen', False):
        # PyInstaller's PySide6 runtime hook sets the bundled plugin locations.
        return
    if sys.platform == 'darwin':
        import PySide6
        plugins=Path(PySide6.__file__).parent/'Qt'/'plugins'
        # Conda's qt6 plugin directory can otherwise override the wheel's plugin
        # discovery at QApplication construction and load a second Qt runtime.
        if (plugins/'platforms'/'libqcocoa.dylib').is_file():
            os.environ['QT_PLUGIN_PATH']=str(plugins)
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(plugins/'platforms')
