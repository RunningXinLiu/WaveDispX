"""WaveDispX wavefield simulation and dispersion-analysis package."""

__version__ = "0.2.0"

from .config import ConfigError, RunConfig, config_from_mapping, load_config
from .app_service import CaseInputError, CaseSession
from .workflow import WaveDispXWorkflow

__all__ = [
    "ConfigError",
    "CaseInputError",
    "CaseSession",
    "WaveDispXWorkflow",
    "RunConfig",
    "config_from_mapping",
    "load_config",
]
