"""Reproducible source and receiver geometry generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    AmbientSourceConfig,
    ReceiverFileConfig,
    ReceiverGridConfig,
    ReceiverLineConfig,
)


@dataclass(frozen=True)
class Geometry:
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray


def _load_xy_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if table.dtype.names is None or not {"x_m", "y_m"}.issubset(table.dtype.names):
        raise ValueError("coordinate CSV must contain x_m,y_m headers")
    x = np.atleast_1d(table["x_m"]).astype(np.float64)
    y = np.atleast_1d(table["y_m"]).astype(np.float64)
    if x.size < 1 or x.size != y.size or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("coordinate CSV must contain finite x_m,y_m rows")
    return x, y


def receiver_positions(
    config: ReceiverLineConfig | ReceiverGridConfig | ReceiverFileConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(config, ReceiverLineConfig):
        x = config.start_x_m + config.spacing_m * np.arange(
            config.count, dtype=np.float64
        )
        y = np.full_like(x, config.y_m)
        return x, y
    if isinstance(config, ReceiverGridConfig):
        x_axis = config.start_x_m + config.spacing_x_m * np.arange(
            config.count_x, dtype=np.float64
        )
        y_axis = config.start_y_m + config.spacing_y_m * np.arange(
            config.count_y, dtype=np.float64
        )
        x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
        return x_grid.reshape(-1), y_grid.reshape(-1)
    x, y = _load_xy_csv(config.coordinates_file)
    if x.size != config.count:
        raise ValueError(
            f"receiver coordinate CSV has {x.size} rows but count={config.count}"
        )
    return x, y


def receiver_line(config: ReceiverLineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible line-array helper."""
    if not isinstance(config, ReceiverLineConfig):
        raise ValueError("receiver_line requires layout='line'; use receiver_positions")
    return receiver_positions(config)


def ambient_source_positions(config: AmbientSourceConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    count = config.count
    if config.layout == "annulus":
        inner = float(config.inner_radius_m)
        outer = float(config.outer_radius_m)
        radius = np.sqrt(inner**2 + (outer**2 - inner**2) * rng.random(count))
        if config.angular_sampling == "random":
            angle = 2 * np.pi * rng.random(count)
        elif config.angular_sampling == "uniform":
            angle = 2 * np.pi * np.arange(count) / count + 2 * np.pi * rng.random()
        else:
            angle = 2 * np.pi * (np.arange(count) + rng.random(count)) / count
            angle += 2 * np.pi * rng.random()
        return radius * np.cos(angle), radius * np.sin(angle)
    if config.layout in {"road", "line"}:
        x = rng.uniform(float(config.x_min_m), float(config.x_max_m), count)
        y = np.full(count, float(config.y_m))
        return x, y

    x, y = _load_xy_csv(config.coordinates_file)
    if x.size != count:
        raise ValueError(
            f"source coordinate CSV has {x.size} rows but count={count}"
        )
    return x, y


def build_geometry(
    receivers: ReceiverLineConfig | ReceiverGridConfig | ReceiverFileConfig,
    sources: AmbientSourceConfig,
) -> Geometry:
    receiver_x, receiver_y = receiver_positions(receivers)
    source_x, source_y = ambient_source_positions(sources)
    return Geometry(receiver_x, receiver_y, source_x, source_y)


def save_geometry_csv(geometry: Geometry, output_directory: str | Path) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    receiver_path = output / "receivers.csv"
    source_path = output / "sources.csv"
    np.savetxt(receiver_path, np.column_stack((geometry.receiver_x_m, geometry.receiver_y_m)),
               delimiter=",", header="x_m,y_m", comments="")
    np.savetxt(source_path, np.column_stack((geometry.source_x_m, geometry.source_y_m)),
               delimiter=",", header="x_m,y_m", comments="")
    return receiver_path, source_path
