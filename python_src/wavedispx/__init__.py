"""WaveDispX wavefield simulation and dispersion-analysis package.

Public exports are lazy so the classroom does not import unrelated workflows.
"""
from importlib import import_module

__version__ = "0.2.0"

_EXPORTS = {'ConfigError': '.config', 'RunConfig': '.config', 'config_from_mapping': '.config', 'load_config': '.config', 'CaseInputError': '.app_service', 'CaseSession': '.app_service', 'DASGeometry': '.das', 'DASRecord': '.das', 'build_das_geometry': '.das', 'load_das_geometry_csv': '.das', 'WaveDispXWorkflow': '.workflow'}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
