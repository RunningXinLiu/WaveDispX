#!/usr/bin/env python3
"""Export a WaveDispX dispersion image and QEDisp truth to one NPZ file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_IMAGE_PATHS = {
    "rayleigh": (
        "dispersion/rayleigh_combined",
        "dispersion/rayleigh",
    ),
    "love": ("dispersion/love",),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export column-normalized dispersion energy, native axes, and "
            "frequency-by-mode theoretical phase velocities for picker evaluation."
        )
    )
    parser.add_argument("results_h5", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--wave-type", choices=("rayleigh", "love"), default="rayleigh")
    parser.add_argument(
        "--image-path",
        help=(
            "HDF5 group containing energy/frequency_hz/velocity_mps; by default "
            "Rayleigh prefers dispersion/rayleigh_combined and Love uses dispersion/love"
        ),
    )
    parser.add_argument(
        "--allow-non-qedisp",
        action="store_true",
        help="allow exporting truth produced by a solver other than QEDispInv",
    )
    return parser


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _resolve_image_path(handle: h5py.File, wave_type: str, requested: str | None) -> str:
    candidates = (requested,) if requested else DEFAULT_IMAGE_PATHS[wave_type]
    for candidate in candidates:
        if candidate and candidate in handle:
            group = handle[candidate]
            if all(name in group for name in ("energy", "frequency_hz", "velocity_mps")):
                return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise KeyError(f"no usable dispersion image group found; checked: {joined}")


def export_benchmark(
    results_h5: Path,
    output_npz: Path,
    *,
    wave_type: str,
    image_path: str | None,
    allow_non_qedisp: bool,
) -> dict[str, object]:
    results_h5 = results_h5.expanduser().resolve()
    output_npz = output_npz.expanduser().resolve()
    with h5py.File(results_h5, "r") as handle:
        solver = _text(handle["theory"].attrs.get("solver", ""))
        if not allow_non_qedisp and not solver.startswith("QEDispInv/"):
            raise ValueError(
                f"expected QEDispInv truth, found {solver!r}; use --allow-non-qedisp to override"
            )
        selected_image_path = _resolve_image_path(handle, wave_type, image_path)
        image = handle[selected_image_path]
        energy = np.asarray(image["energy"], dtype=np.float32)
        raw_amplitude = (
            np.asarray(image["raw_amplitude"], dtype=np.float32)
            if "raw_amplitude" in image
            else energy.copy()
        )
        frequency_hz = np.asarray(image["frequency_hz"], dtype=np.float64)
        velocity_mps = np.asarray(image["velocity_mps"], dtype=np.float64)
        theory_frequency_hz = np.asarray(handle["theory/frequency_hz"], dtype=np.float64)
        theory_mps = np.asarray(
            handle[f"theory/{wave_type}_mps"], dtype=np.float64
        )
        forward_solver = _text(handle.attrs.get("forward_solver", ""))
        run_mode = _text(handle.attrs.get("mode", ""))
        wavedispx_version = _text(handle.attrs.get("wavedispx_version", ""))

    expected_shape = (velocity_mps.size, frequency_hz.size)
    if energy.shape != expected_shape:
        raise ValueError(f"energy shape {energy.shape} does not match axes {expected_shape}")
    if theory_mps.ndim != 2 or theory_mps.shape[0] != frequency_hz.size:
        raise ValueError(
            "theory must be frequency-by-mode and share the dispersion frequency count"
        )
    if theory_frequency_hz.shape != frequency_hz.shape or not np.allclose(
        theory_frequency_hz, frequency_hz, rtol=0.0, atol=1e-10
    ):
        raise ValueError("theory and dispersion frequency grids are not identical")
    if not np.all(np.isfinite(energy)):
        raise ValueError("dispersion energy contains non-finite values")
    if raw_amplitude.shape != expected_shape or not np.all(np.isfinite(raw_amplitude)):
        raise ValueError("raw dispersion amplitude is invalid or does not match the axes")
    column_coherence = np.max(raw_amplitude, axis=0)
    if not np.all(np.diff(frequency_hz) > 0.0) or not np.all(
        np.diff(velocity_mps) > 0.0
    ):
        raise ValueError("frequency and velocity axes must be strictly increasing")

    metadata = {
        "schema": "wavedispx-picker-benchmark-v1",
        "source_results_h5": str(results_h5),
        "wave_type": wave_type,
        "image_path": selected_image_path,
        "energy_shape": list(energy.shape),
        "energy_axis_order": ["velocity", "frequency"],
        "frequency_order": "ascending",
        "velocity_order": "ascending",
        "energy_normalization": "each frequency column has maximum 1 when nonzero",
        "raw_amplitude_available": bool(np.any(raw_amplitude != energy)),
        "column_coherence_definition": "maximum raw imaging amplitude over velocity",
        "theory_shape": list(theory_mps.shape),
        "theory_axis_order": ["frequency", "mode"],
        "theory_missing_root": "NaN; exclude with theory_valid_mask_fm",
        "theory_solver": solver,
        "forward_solver": forward_solver,
        "run_mode": run_mode,
        "wavedispx_version": wavedispx_version,
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        energy_vf=energy,
        raw_amplitude_vf=raw_amplitude,
        column_coherence_f=column_coherence.astype(np.float32),
        frequency_hz=frequency_hz,
        velocity_mps=velocity_mps,
        theory_phase_velocity_mps_fm=theory_mps,
        theory_valid_mask_fm=np.isfinite(theory_mps),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def main() -> int:
    args = _parser().parse_args()
    metadata = export_benchmark(
        args.results_h5,
        args.output_npz,
        wave_type=args.wave_type,
        image_path=args.image_path,
        allow_non_qedisp=args.allow_non_qedisp,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"output_npz: {args.output_npz.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
