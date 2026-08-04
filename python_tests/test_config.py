from pathlib import Path

import pytest

from wavedispx.config import (
    ConfigError,
    ReceiverFileConfig,
    ReceiverGridConfig,
    load_config,
)
from wavedispx.geometry import receiver_positions


ROOT = Path(__file__).resolve().parents[1]


def test_passive_example_is_valid():
    config = load_config(ROOT / "examples/passive_two_layer.toml")
    assert config.mode == "passive"
    assert config.receivers.count == 121
    assert config.ambient_sources.count == 384
    assert config.noise.duration_s == 1800
    assert config.model.layers[-1].thickness_m == float("inf")


def test_active_example_is_valid():
    config = load_config(ROOT / "examples/active_two_layer.toml")
    assert config.mode == "active"
    assert config.receivers.count == 200
    assert config.active_source.dominant_frequency_hz == 20


def test_irregular_receiver_file_example_is_valid():
    config = load_config(ROOT / "examples/passive_irregular_quick.toml")
    assert isinstance(config.receivers, ReceiverFileConfig)
    assert config.receivers.count == 21
    assert config.receivers.distance_bin_m == 10.0
    receiver_x, receiver_y = receiver_positions(config.receivers)
    assert receiver_x.shape == receiver_y.shape == (21,)
    assert not (receiver_y == receiver_y[0]).all()


def test_receiver_grid_example_materializes_full_2d_array():
    config = load_config(ROOT / "examples/passive_grid_quick.toml")
    assert isinstance(config.receivers, ReceiverGridConfig)
    assert config.receivers.count_x == 11
    assert config.receivers.count_y == 11
    assert config.receivers.count == 121
    receiver_x, receiver_y = receiver_positions(config.receivers)
    assert receiver_x.shape == receiver_y.shape == (121,)
    assert len(set(receiver_x)) == 11
    assert len(set(receiver_y)) == 11


def test_rejects_processing_above_noise_band(tmp_path):
    text = (ROOT / "examples/passive_two_layer.toml").read_text()
    text = text.replace("frequency_max_hz = 18.8\nvelocity_min", "frequency_max_hz = 19.0\nvelocity_min")
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="must not exceed noise"):
        load_config(path)


def test_rejects_unknown_horizontal_only_ambient_load(tmp_path):
    text = (ROOT / "examples/passive_two_layer.toml").read_text()
    text = text.replace('component = "vertical"', 'component = "horizontal"')
    path = tmp_path / "horizontal_ambient.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="vertical.*three_component"):
        load_config(path)


def test_accepts_physical_three_component_ambient_load(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text = text.replace('component = "vertical"', 'component = "three_component"')
    path = tmp_path / "three_component.toml"
    path.write_text(text)
    assert load_config(path).ambient_sources.component == "three_component"


def test_rejects_misleading_overlap_for_independent_synthetic_windows(tmp_path):
    text = (ROOT / "examples/passive_two_layer.toml").read_text()
    text = text.replace("overlap = 0.0", "overlap = 0.5")
    path = tmp_path / "overlap.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="must currently be 0"):
        load_config(path)


def test_passive_receiver_file_requires_distance_bin(tmp_path):
    coordinate_path = tmp_path / "receivers.csv"
    coordinate_path.write_text("x_m,y_m\n0,0\n10,2\n")
    text = (ROOT / "examples/passive_quick.toml").read_text()
    first = text.index("[geometry.receivers]")
    last = text.index("\n[ambient_sources]", first)
    text = (
        text[:first]
        + '[geometry.receivers]\nlayout = "file"\ncoordinates_file = "receivers.csv"\n'
        + text[last:]
    )
    path = tmp_path / "missing_bin.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="requires geometry.receivers.distance_bin_m"):
        load_config(path)


def test_receiver_file_rejects_duplicate_coordinates(tmp_path):
    coordinate_path = tmp_path / "receivers.csv"
    coordinate_path.write_text("x_m,y_m\n0,1\n0,1\n")
    text = (ROOT / "examples/active_quick.toml").read_text()
    first = text.index("[geometry.receivers]")
    last = text.index("\n[active_source]", first)
    text = (
        text[:first]
        + '[geometry.receivers]\nlayout = "file"\ncoordinates_file = "receivers.csv"\n'
        + text[last:]
    )
    path = tmp_path / "duplicate_receivers.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="duplicate x_m,y_m"):
        load_config(path)
