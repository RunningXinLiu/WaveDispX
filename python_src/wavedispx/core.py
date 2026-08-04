"""Public typed wrappers for the compiled discrete-wavenumber (DWM) kernels."""

from __future__ import annotations

import numpy as np

try:
    from . import _dwm_core
except ImportError as exc:  # pragma: no cover - exercised only before building
    _dwm_core = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_core():
    if _dwm_core is None:
        raise RuntimeError(
            "the WaveDispX C++ forward core is not built; run `python -m pip install -e .`"
        ) from _IMPORT_ERROR


def global_stiffness(
    vp_mps: np.ndarray,
    vs_mps: np.ndarray,
    damp_p: np.ndarray,
    damp_s: np.ndarray,
    density_kgm3: np.ndarray,
    thickness_m: np.ndarray,
    wavenumber_radpm: float,
    omega_radps: complex,
) -> tuple[np.ndarray, np.ndarray]:
    _require_core()
    return _dwm_core.global_stiffness(
        np.ascontiguousarray(vp_mps, dtype=np.float64),
        np.ascontiguousarray(vs_mps, dtype=np.float64),
        np.ascontiguousarray(damp_p, dtype=np.float64),
        np.ascontiguousarray(damp_s, dtype=np.float64),
        np.ascontiguousarray(density_kgm3, dtype=np.float64),
        np.ascontiguousarray(thickness_m, dtype=np.float64),
        float(wavenumber_radpm),
        complex(omega_radps),
    )


def radial_transform_lookup(
    weighted_spectrum: np.ndarray,
    frequency_hz: np.ndarray,
    distance_m: np.ndarray,
    velocity_mps: np.ndarray,
    lookup_kernel: np.ndarray,
    *,
    lookup_minimum: float,
    lookup_step: float,
    workers: int = 0,
) -> np.ndarray:
    """Evaluate an FJ/Hankel transform using a controlled kernel lookup table."""
    _require_core()
    return _dwm_core.radial_transform_lookup(
        np.ascontiguousarray(weighted_spectrum, dtype=np.complex128),
        np.ascontiguousarray(frequency_hz, dtype=np.float64),
        np.ascontiguousarray(distance_m, dtype=np.float64),
        np.ascontiguousarray(velocity_mps, dtype=np.float64),
        np.ascontiguousarray(lookup_kernel, dtype=np.complex128),
        float(lookup_minimum),
        float(lookup_step),
        int(workers),
    )


def vertical_fk_response(
    vp_mps: np.ndarray,
    vs_mps: np.ndarray,
    damp_p: np.ndarray,
    damp_s: np.ndarray,
    density_kgm3: np.ndarray,
    thickness_m: np.ndarray,
    *,
    sample_count: int,
    dt_s: float,
    maximum_distance_m: float,
    delay_s: float,
    maximum_frequency_hz: float,
    workers: int = 0,
) -> dict[str, np.ndarray]:
    _require_core()
    return _dwm_core.fk_response(
        np.ascontiguousarray(vp_mps, dtype=np.float64),
        np.ascontiguousarray(vs_mps, dtype=np.float64),
        np.ascontiguousarray(damp_p, dtype=np.float64),
        np.ascontiguousarray(damp_s, dtype=np.float64),
        np.ascontiguousarray(density_kgm3, dtype=np.float64),
        np.ascontiguousarray(thickness_m, dtype=np.float64),
        int(sample_count),
        float(dt_s),
        float(maximum_distance_m),
        float(delay_s),
        float(maximum_frequency_hz),
        int(workers),
        False,
        True,
    )


def full_fk_response(
    vp_mps: np.ndarray,
    vs_mps: np.ndarray,
    damp_p: np.ndarray,
    damp_s: np.ndarray,
    density_kgm3: np.ndarray,
    thickness_m: np.ndarray,
    *,
    sample_count: int,
    dt_s: float,
    maximum_distance_m: float,
    delay_s: float,
    maximum_frequency_hz: float,
    workers: int = 0,
    include_horizontal: bool = True,
    include_vertical: bool = True,
) -> dict[str, np.ndarray]:
    _require_core()
    if not include_horizontal and not include_vertical:
        raise ValueError("at least one load component must be enabled")
    return _dwm_core.fk_response(
        np.ascontiguousarray(vp_mps, dtype=np.float64),
        np.ascontiguousarray(vs_mps, dtype=np.float64),
        np.ascontiguousarray(damp_p, dtype=np.float64),
        np.ascontiguousarray(damp_s, dtype=np.float64),
        np.ascontiguousarray(density_kgm3, dtype=np.float64),
        np.ascontiguousarray(thickness_m, dtype=np.float64),
        int(sample_count),
        float(dt_s),
        float(maximum_distance_m),
        float(delay_s),
        float(maximum_frequency_hz),
        int(workers),
        bool(include_horizontal),
        bool(include_vertical),
    )
