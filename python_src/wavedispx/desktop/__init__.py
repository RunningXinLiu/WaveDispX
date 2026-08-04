"""Native PySide6 desktop workbench for WaveDispX.

The ``seisloc`` environment contains the pip PySide6 wheel (with bundled Qt)
and a separate Conda Qt used by OpenCV.  On macOS, an active ``CONDA_PREFIX``
can make Qt discover both copies and abort before a window is created.  The
interpreter and ``sys.path`` are already fixed when this package is imported,
so removing only the activation-marker variables before importing PySide6 is
safe and keeps the wheel on its matching bundled Qt runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _prepare_qt_runtime() -> None:
    if sys.platform != "darwin":
        return
    prefix = Path(sys.prefix)
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    bundled_qt = any(
        (prefix / "lib" / python_dir / "site-packages" / "PySide6" / "Qt" / "lib").glob(
            "QtCore.framework"
        )
    )
    conda_qt = any((prefix / "lib").glob("libQt6Core*.dylib"))
    if bundled_qt and conda_qt:
        for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER"):
            os.environ.pop(key, None)


_prepare_qt_runtime()

__all__ = ["launch"]


def launch() -> int:
    """Launch the desktop workbench and return the Qt exit code."""
    from .launcher import main

    return main()
