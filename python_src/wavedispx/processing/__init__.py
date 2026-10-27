"""Ambient-noise correlation, dispersion imaging, picking, and theory.

Public exports are lazy so the classroom does not import unrelated workflows.
"""
from importlib import import_module

_EXPORTS = {'ArrayDispersionImage': '.array_dispersion', 'MATLAB_KERNEL_METHODS': '.array_dispersion', 'SpatialCoherency': '.array_dispersion', 'bin_spatial_coherency': '.array_dispersion', 'canonical_array_method': '.array_dispersion', 'estimate_spatial_coherency': '.array_dispersion', 'method_input_contract': '.array_dispersion', 'spac_dispersion': '.array_dispersion', 'wavefield_dispersion': '.array_dispersion', 'DispersionImage': '.dispersion', 'combine_dispersion_images': '.dispersion', 'surface_wave_dispersion': '.dispersion', 'FTANCurve': '.ftan', 'FTANImage': '.ftan', 'PairFTANResult': '.ftan', 'correlation_to_egf': '.ftan', 'pair_ftan': '.ftan', 'phase_to_group_velocity': '.ftan', 'NCFBranches': '.ncf', 'NCFGather': '.ncf', 'NCFPairBranches': '.ncf', 'NCFPairGather': '.ncf', 'bin_pair_ncf_branches': '.ncf', 'linear_array_ncf': '.ncf', 'linear_array_ncf_branches': '.ncf', 'receiver_array_3c_ncf': '.ncf', 'receiver_array_3c_ncf_branches': '.ncf', 'receiver_array_ncf': '.ncf', 'receiver_array_ncf_branches': '.ncf', 'receiver_pair_ncf_branches': '.ncf', 'DispersionPicks': '.picking', 'pick_dispersion_ridge': '.picking', 'QEDispForwardResult': '.qedisp', 'QEDispKernels': '.qedisp', 'find_qedisp_forward': '.qedisp', 'find_qedisp_group_library': '.qedisp', 'run_qedisp_forward': '.qedisp', 'TheoreticalDispersion': '.theory', 'theoretical_dispersion': '.theory'}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
