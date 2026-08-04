"""Command-line interface for WaveDispX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .config import ConfigError, load_config
from .geometry import build_geometry, receiver_positions, save_geometry_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wavedispx",
        description=(
            "Active/passive wavefield simulation, ambient-noise correlation, "
            "and dispersion analysis"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate and summarize a TOML input")
    validate.add_argument("config", type=Path)
    validate.add_argument("--json", action="store_true", help="print machine-readable JSON")
    geometry = subparsers.add_parser("geometry", help="materialize source/receiver coordinates")
    geometry.add_argument("config", type=Path)
    geometry.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run", help="run one validated active/passive case")
    run.add_argument("config", type=Path)
    return parser


def _summary(config) -> dict[str, object]:
    receiver_x, receiver_y = receiver_positions(config.receivers)
    receiver_aperture = float(
        np.hypot(
            receiver_x[:, None] - receiver_x[None, :],
            receiver_y[:, None] - receiver_y[None, :],
        ).max()
    )
    result: dict[str, object] = {
        "name": config.name,
        "mode": config.mode,
        "layers": len(config.model.layers),
        "receivers": config.receivers.count,
        "receiver_layout": config.receivers.layout if hasattr(config.receivers, "layout") else "line",
        "receiver_aperture_m": receiver_aperture,
        "frequency_band_hz": [config.processing.frequency_min_hz, config.processing.frequency_max_hz],
        "dispersion_method": config.processing.dispersion_method,
        "output_directory": str(config.output_directory),
    }
    if config.mode == "passive":
        result.update(
            sources=config.ambient_sources.count,
            source_layout=config.ambient_sources.layout,
            duration_s=config.noise.duration_s,
            window_s=config.noise.window_s,
            noise_psd=config.noise.psd_shape,
        )
    else:
        result.update(
            dominant_frequency_hz=config.active_source.dominant_frequency_hz,
            source_component=config.active_source.component,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "validate":
            summary = _summary(config)
            if args.json:
                print(json.dumps(summary, indent=2, ensure_ascii=False))
            else:
                print("Configuration valid")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
            return 0
        if args.command == "run":
            from .runner import run

            artifacts = run(config)
            print(f"output: {artifacts.output_directory}")
            print(f"results: {artifacts.results_h5}")
            print(f"summary: {artifacts.summary_png}")
            return 0
        if config.mode == "passive":
            geometry = build_geometry(config.receivers, config.ambient_sources)
        else:
            receiver_x, receiver_y = receiver_positions(config.receivers)
            from .geometry import Geometry
            geometry = Geometry(receiver_x, receiver_y,
                                source_x_m=np.array([config.active_source.x_m]),
                                source_y_m=np.array([config.active_source.y_m]))
        receiver_path, source_path = save_geometry_csv(geometry, args.output)
        print(f"receivers: {receiver_path}")
        print(f"sources: {source_path}")
        return 0
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
