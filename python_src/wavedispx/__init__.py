"""WaveDispX wavefield simulation and dispersion-analysis package."""

__version__ = "0.2.0"

from .config import ConfigError, RunConfig, config_from_mapping, load_config
from .app_service import CaseInputError, CaseSession
from .das import (
    DASGeometry,
    DASRecord,
    build_das_geometry,
    load_das_geometry_csv,
)
from .workflow import WaveDispXWorkflow

__all__ = [
    "ConfigError",
    "CaseInputError",
    "CaseSession",
    "DASGeometry",
    "DASRecord",
    "WaveDispXWorkflow",
    "RunConfig",
    "build_das_geometry",
    "load_das_geometry_csv",
    "config_from_mapping",
    "load_config",
]
