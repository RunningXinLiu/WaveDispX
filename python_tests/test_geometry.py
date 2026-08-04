from pathlib import Path

import numpy as np

from wavedispx.config import load_config
from wavedispx.geometry import build_geometry


ROOT = Path(__file__).resolve().parents[1]


def test_annular_geometry_is_reproducible_and_in_bounds():
    config = load_config(ROOT / "examples/passive_two_layer.toml")
    first = build_geometry(config.receivers, config.ambient_sources)
    second = build_geometry(config.receivers, config.ambient_sources)
    np.testing.assert_array_equal(first.source_x_m, second.source_x_m)
    np.testing.assert_array_equal(first.source_y_m, second.source_y_m)
    radius = np.hypot(first.source_x_m, first.source_y_m)
    assert np.all(radius >= 800)
    assert np.all(radius <= 1200)
    assert first.receiver_x_m.size == 121
    assert first.receiver_x_m[0] == -300
    assert first.receiver_x_m[-1] == 300


def test_annular_radius_is_area_uniform():
    config = load_config(ROOT / "examples/passive_two_layer.toml")
    geometry = build_geometry(config.receivers, config.ambient_sources)
    radius_squared = geometry.source_x_m**2 + geometry.source_y_m**2
    normalized_mean = (radius_squared.mean() - 800**2) / (1200**2 - 800**2)
    assert abs(normalized_mean - 0.5) < 0.05

