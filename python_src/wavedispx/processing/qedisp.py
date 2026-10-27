"""Subprocess adapter for the QEDispInv surface-wave forward solver.

QEDispInv remains an independent LGPL-3.0 program.  WaveDispX writes its
documented text inputs, invokes the ``forward`` executable, and parses the
documented text/HDF5 outputs; no QEDispInv source or library is linked here.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

from ..config import ModelConfig


@dataclass(frozen=True)
class QEDispKernels:
    """QEDispInv phase-velocity sensitivity kernels for one wave type.

    ``vp`` and ``vs`` are dimensionless.  ``density`` retains QEDispInv's
    native derivative unit.  Columns follow ``frequency_hz`` and ``mode``.
    """

    frequency_hz: np.ndarray
    phase_velocity_mps: np.ndarray
    mode: np.ndarray
    depth_m: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    density: np.ndarray


@dataclass(frozen=True)
class QEDispForwardResult:
    """QEDispInv phase/group curves and optional sensitivity kernels.

    Group velocities are populated only when ``compute_group=True``.  They
    come from QEDispInv's native ``sregn96``/``slegn96`` energy-integral
    routines, not from numerical differentiation of the phase curves.
    """

    frequency_hz: np.ndarray
    rayleigh_mps: np.ndarray
    love_mps: np.ndarray
    executable: Path
    rayleigh_kernels: QEDispKernels | None = None
    love_kernels: QEDispKernels | None = None
    rayleigh_group_mps: np.ndarray | None = None
    love_group_mps: np.ndarray | None = None


def find_qedisp_forward(executable: str | Path | None = None) -> Path:
    """Resolve a QEDispInv ``forward`` executable without silent fallback."""

    candidates: list[Path] = []
    if executable is not None:
        candidates.append(Path(executable).expanduser())
    else:
        environment = os.environ.get("QEDISPINV_FORWARD")
        if environment:
            candidates.append(Path(environment).expanduser())
        project_root = Path(__file__).resolve().parents[3]
        candidates.append(project_root / ".wavedispx-tools" / "QEDispInv" / "bin" / "forward")

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    searched = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(
        "QEDispInv forward executable was not found or is not executable; "
        f"searched: {searched}. Set QEDISPINV_FORWARD or pass qedisp_forward=."
    )


def find_qedisp_group_library(
    library: str | Path | None = None,
    *,
    forward_executable: str | Path | None = None,
) -> Path:
    """Resolve the local QEDispInv native group-velocity helper library."""

    candidates: list[Path] = []
    if library is not None:
        candidates.append(Path(library).expanduser())
    else:
        environment = os.environ.get("QEDISPINV_GROUP_LIBRARY")
        if environment:
            candidates.append(Path(environment).expanduser())
        forward = find_qedisp_forward(forward_executable)
        candidates.extend(
            (
                forward.parent.parent / "lib" / "libqedisp_group.dylib",
                forward.parent.parent / "lib" / "libqedisp_group.so",
            )
        )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    searched = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(
        "QEDispInv native group-velocity helper was not found; "
        f"searched: {searched}. Re-run scripts/install_qedispinv_local.py "
        "or set QEDISPINV_GROUP_LIBRARY."
    )


def run_qedisp_forward(
    model: ModelConfig,
    frequency_hz: np.ndarray,
    *,
    rayleigh_modes: int = 4,
    love_modes: int = 4,
    executable: str | Path | None = None,
    compute_kernels: bool = False,
    compute_group: bool = False,
    group_library: str | Path | None = None,
    timeout_s: float = 120.0,
) -> QEDispForwardResult:
    """Compute QEDispInv Rayleigh/Love curves at exact requested frequencies.

    Missing roots are represented by ``NaN`` rather than interpolated across a
    mode gap.  QEDispInv prints frequencies to five decimal places, so inputs
    that collide after that rounding are rejected explicitly.
    """

    frequency = _validated_frequency(frequency_hz)
    if rayleigh_modes < 0 or love_modes < 0:
        raise ValueError("rayleigh_modes and love_modes must be non-negative")
    if rayleigh_modes == 0 and love_modes == 0:
        raise ValueError("at least one Rayleigh or Love mode is required")
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive and finite")
    forward = find_qedisp_forward(executable)

    with tempfile.TemporaryDirectory(prefix="wavedispx-qedisp-") as temporary:
        work = Path(temporary)
        model_path = work / "model.txt"
        config_path = work / "config.toml"
        _write_model(model_path, model)
        config_path.write_text(
            "[forward]\n"
            f'file_model = "{model_path.name}"\n'
            f"fmin = {frequency[0]:.12g}\n"
            f"fmax = {frequency[-1]:.12g}\n"
            f"nf = {frequency.size}\n"
        )

        rayleigh, rayleigh_kernel = _run_wave_type(
            forward,
            work,
            config_path,
            model_path,
            frequency,
            rayleigh_modes,
            sh=False,
            compute_kernels=compute_kernels,
            timeout_s=timeout_s,
        )
        love, love_kernel = _run_wave_type(
            forward,
            work,
            config_path,
            model_path,
            frequency,
            love_modes,
            sh=True,
            compute_kernels=compute_kernels,
            timeout_s=timeout_s,
        )

    rayleigh_group = None
    love_group = None
    if compute_group:
        native_library = find_qedisp_group_library(
            group_library,
            forward_executable=forward,
        )
        rayleigh_group = _native_group_curves(
            model,
            frequency,
            rayleigh,
            native_library,
            love=False,
        )
        love_group = _native_group_curves(
            model,
            frequency,
            love,
            native_library,
            love=True,
        )

    return QEDispForwardResult(
        frequency_hz=frequency,
        rayleigh_mps=rayleigh,
        love_mps=love,
        executable=forward,
        rayleigh_kernels=rayleigh_kernel,
        love_kernels=love_kernel,
        rayleigh_group_mps=rayleigh_group,
        love_group_mps=love_group,
    )


def _native_group_curves(
    model: ModelConfig,
    frequency_hz: np.ndarray,
    phase_velocity_mps: np.ndarray,
    library: Path,
    *,
    love: bool,
) -> np.ndarray:
    """Evaluate QEDispInv's native energy-integral group velocity."""

    phase = np.asarray(phase_velocity_mps, dtype=np.float64)
    group = np.full_like(phase, np.nan)
    if phase.size == 0:
        return group

    native = ctypes.CDLL(str(library))
    double_pointer = ctypes.POINTER(ctypes.c_double)
    thickness = np.asarray(
        [
            0.0 if np.isinf(layer.thickness_m) else layer.thickness_m / 1000.0
            for layer in model.layers
        ],
        dtype=np.float64,
    )
    vp = np.asarray([layer.vp_mps / 1000.0 for layer in model.layers], dtype=np.float64)
    vs = np.asarray([layer.vs_mps / 1000.0 for layer in model.layers], dtype=np.float64)
    density = np.asarray(
        [layer.density_kgm3 / 1000.0 for layer in model.layers], dtype=np.float64
    )
    layer_count = len(model.layers)

    if love:
        function = native.slegn96_double_
        function.argtypes = [
            double_pointer,
            double_pointer,
            double_pointer,
            ctypes.c_int,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            ctypes.c_int,
        ]
        work_count = 5
    else:
        function = native.sregn96_double_
        function.argtypes = [
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            ctypes.c_int,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            ctypes.c_int,
        ]
        work_count = 8
    function.restype = None

    for frequency_index, frequency in enumerate(frequency_hz):
        for mode in range(phase.shape[1]):
            phase_mps = phase[frequency_index, mode]
            if not np.isfinite(phase_mps):
                continue
            period = ctypes.c_double(1.0 / float(frequency))
            phase_kmps = ctypes.c_double(float(phase_mps) / 1000.0)
            group_kmps = ctypes.c_double(np.nan)
            work = [np.zeros(layer_count, dtype=np.float64) for _ in range(work_count)]
            if love:
                function(
                    thickness.ctypes.data_as(double_pointer),
                    vs.ctypes.data_as(double_pointer),
                    density.ctypes.data_as(double_pointer),
                    layer_count,
                    ctypes.byref(period),
                    ctypes.byref(phase_kmps),
                    ctypes.byref(group_kmps),
                    *(values.ctypes.data_as(double_pointer) for values in work),
                    0,
                )
            else:
                function(
                    thickness.ctypes.data_as(double_pointer),
                    vp.ctypes.data_as(double_pointer),
                    vs.ctypes.data_as(double_pointer),
                    density.ctypes.data_as(double_pointer),
                    layer_count,
                    ctypes.byref(period),
                    ctypes.byref(phase_kmps),
                    ctypes.byref(group_kmps),
                    *(values.ctypes.data_as(double_pointer) for values in work),
                    0,
                )
            value_mps = 1000.0 * group_kmps.value
            if np.isfinite(value_mps) and value_mps > 0.0:
                group[frequency_index, mode] = value_mps
    return group


def _validated_frequency(values: np.ndarray) -> np.ndarray:
    frequency = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        frequency.size == 0
        or not np.all(np.isfinite(frequency))
        or np.any(frequency <= 0.0)
        or np.any(np.diff(frequency) <= 0.0)
    ):
        raise ValueError("frequency_hz must be finite, positive, and strictly increasing")
    rounded = np.round(frequency, 5)
    if np.unique(rounded).size != frequency.size:
        raise ValueError(
            "frequency_hz contains values that collide at QEDispInv's 5-decimal output precision"
        )
    return frequency


def _write_model(path: Path, model: ModelConfig) -> None:
    rows: list[str] = []
    depth_km = 0.0
    for index, layer in enumerate(model.layers, start=1):
        rows.append(
            f"{index:d} {depth_km:.12g} {layer.density_kgm3 / 1000.0:.12g} "
            f"{layer.vs_mps / 1000.0:.12g} {layer.vp_mps / 1000.0:.12g}"
        )
        if np.isfinite(layer.thickness_m):
            depth_km += layer.thickness_m / 1000.0
    path.write_text("\n".join(rows) + "\n")


def _run_wave_type(
    executable: Path,
    work: Path,
    config_path: Path,
    model_path: Path,
    frequency: np.ndarray,
    mode_count: int,
    *,
    sh: bool,
    compute_kernels: bool,
    timeout_s: float,
) -> tuple[np.ndarray, QEDispKernels | None]:
    curves = np.full((frequency.size, mode_count), np.nan, dtype=np.float64)
    if mode_count == 0:
        return curves, None

    label = "love" if sh else "rayleigh"
    target_path = work / f"{label}_target.txt"
    output_path = work / f"{label}_disp.txt"
    rows = [
        f"{value:.12g} 0 {mode:d}"
        for mode in range(mode_count)
        for value in frequency
    ]
    target_path.write_text("\n".join(rows) + "\n")
    command = [
        str(executable),
        "--config",
        config_path.name,
        "--model",
        model_path.name,
        "--disp",
        target_path.name,
        "--mode",
        str(mode_count - 1),
        "--out",
        output_path.name,
    ]
    if sh:
        command.append("--sh")
    if compute_kernels:
        command.append("--compute_kernel")
    try:
        completed = subprocess.run(
            command,
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stdout = getattr(exc, "stdout", "") or ""
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(
            f"QEDispInv {label} forward run failed: {stdout[-1000:]} {stderr[-1000:]}"
        ) from exc

    if not output_path.is_file():
        raise RuntimeError(
            f"QEDispInv {label} run completed without output: {completed.stdout[-1000:]}"
        )
    output = _load_numeric_table(output_path, column_count=3)
    lookup = {float(value): index for index, value in enumerate(np.round(frequency, 5))}
    for output_frequency, velocity_kmps, raw_mode in output:
        mode = int(round(raw_mode))
        index = lookup.get(float(np.round(output_frequency, 5)))
        if index is None or not 0 <= mode < mode_count:
            raise RuntimeError(
                f"QEDispInv returned an unexpected {label} row: "
                f"{output_frequency}, {velocity_kmps}, {raw_mode}"
            )
        curves[index, mode] = 1000.0 * velocity_kmps

    kernels = None
    if compute_kernels:
        kernel_path = work / "kernel.h5"
        if not kernel_path.is_file():
            raise RuntimeError(f"QEDispInv {label} did not produce kernel.h5")
        kernels = _load_kernels(kernel_path)
        kernel_path.unlink()
    return curves, kernels


def _load_numeric_table(path: Path, *, column_count: int) -> np.ndarray:
    if path.stat().st_size == 0:
        return np.empty((0, column_count), dtype=np.float64)
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.ndim != 2 or values.shape[1] != column_count:
        raise RuntimeError(f"unexpected QEDispInv table shape in {path}: {values.shape}")
    return values


def _load_kernels(path: Path) -> QEDispKernels:
    with h5py.File(path) as handle:
        dispersion = np.asarray(handle["disp"], dtype=np.float64)
        vp = np.asarray(handle["kvp"], dtype=np.float64)
        vs = np.asarray(handle["kvs"], dtype=np.float64)
        density = np.asarray(handle["krho"], dtype=np.float64)
        depth_m = 1000.0 * np.asarray(handle["z"], dtype=np.float64).reshape(-1)
    return QEDispKernels(
        dispersion[:, 0],
        1000.0 * dispersion[:, 1],
        dispersion[:, 2].astype(np.int64),
        depth_m,
        vp,
        vs,
        density,
    )
