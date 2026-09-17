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
    assert config.noise.receiver_noise_rms_fraction == 0.0
    assert config.model.layers[-1].thickness_m == float("inf")
    assert not config.nnws.enabled
    assert config.noise.maximum_lag_s is None
    assert config.processing.frequency_step_hz is None


def test_explicit_display_df_and_passive_lag_are_validated(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text = text.replace(
        "velocity_step_mps = 2.0",
        "velocity_step_mps = 2.0\nfrequency_step_hz = 0.05",
    ).replace("window_s = 10.0", "window_s = 10.0\nmaximum_lag_s = 2.0")
    path = tmp_path / "explicit_grid.toml"
    path.write_text(text)
    config = load_config(path)
    assert config.processing.frequency_step_hz == 0.05
    assert config.noise.maximum_lag_s == 2.0

    path.write_text(text.replace("maximum_lag_s = 2.0", "maximum_lag_s = 10.0"))
    with pytest.raises(ConfigError, match="maximum_lag_s"):
        load_config(path)


def test_v3_pair_ftan_receiver_is_a_distinct_valid_receiver():
    config = load_config(ROOT / "examples/passive_ncf_selection_v3_biased.toml")
    assert config.nnws_v2.reference_receiver_index == 0
    assert config.nnws_v2.pair_ftan_receiver_index == 60


def test_active_example_is_valid():
    config = load_config(ROOT / "examples/active_two_layer.toml")
    assert config.mode == "active"
    assert config.receivers.count == 200
    assert config.active_source.dominant_frequency_hz == 20
    assert config.processing.theory_solver == "disba"


def test_qedisp_theory_solver_is_explicit_and_validated(tmp_path):
    config = load_config(ROOT / "examples/active_broadband_2_12hz_validation.toml")
    assert config.processing.theory_solver == "qedisp"

    text = (ROOT / "examples/active_two_layer.toml").read_text()
    text = text.replace(
        'dispersion_method = "hankel"',
        'dispersion_method = "hankel"\ntheory_solver = "unknown"',
    )
    path = tmp_path / "unknown_theory_solver.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="theory_solver"):
        load_config(path)


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


def test_pairwise_nnws_research_config_is_explicit_and_validated(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text += """

[nnws]
enabled = true
epochs = 3
learning_rate = 0.001
snr_weight = 0.8
symmetry_weight = 0.2
std_weight = 0.01
std_mode = "population_eps"
seed = 17
device = "cpu"
minimum_effective_windows = 2.0
minimum_pair_coverage = 1.0
maximum_cache_mb = 64.0
pair_scope = "reference_receiver"
reference_receiver_index = 2
"""
    path = tmp_path / "nnws.toml"
    path.write_text(text)
    config = load_config(path)
    assert config.nnws.enabled
    assert config.nnws.epochs == 3
    assert config.nnws.std_mode == "population_eps"
    assert config.nnws.minimum_pair_coverage == 1.0
    assert config.nnws.random_control_seed == 271828
    assert config.nnws.pair_scope == "reference_receiver"
    assert config.nnws.reference_receiver_index == 2


def test_pairwise_nnws_rejects_missing_reference_receiver(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text += """

[nnws]
enabled = true
pair_scope = "reference_receiver"
reference_receiver_index = 999
"""
    path = tmp_path / "nnws_bad_reference.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="identify an existing receiver"):
        load_config(path)


def test_pairwise_nnws_rejects_unsupported_three_component_input(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text = text.replace('component = "vertical"', 'component = "three_component"')
    text += "\n[nnws]\nenabled = true\n"
    path = tmp_path / "nnws_3c.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="currently supports vertical"):
        load_config(path)


def test_nnws_v2_research_config_is_explicit_and_validated(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text += """

[nnws_v2]
enabled = true
folds = 2
epochs = 3
learning_rate = 0.002
arrival_weight = 0.50
precursor_weight = 0.25
symmetry_weight = 0.05
effective_count_weight = 0.20
temporal_smoothness_weight = 0.01
minimum_weight = 0.10
minimum_effective_fraction = 0.60
seed = 19
random_control_seed = 23
device = "cpu"
maximum_cache_mb = 64.0
pair_scope = "reference_receiver"
reference_receiver_index = 1
"""
    path = tmp_path / "nnws_v2.toml"
    path.write_text(text)
    config = load_config(path)
    assert config.nnws_v2.enabled
    assert config.nnws_v2.folds == 2
    assert config.nnws_v2.epochs == 3
    assert config.nnws_v2.minimum_effective_fraction == 0.60
    assert config.nnws_v2.reference_receiver_index == 1


def test_nnws_v2_rejects_missing_reference_receiver(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text += """

[nnws_v2]
enabled = true
pair_scope = "reference_receiver"
reference_receiver_index = 999
"""
    path = tmp_path / "nnws_v2_bad_reference.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="identify an existing receiver"):
        load_config(path)


def test_pair_spz_research_config_is_explicit_and_validated(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text += """

[pair_spz]
enabled = true
phase_tolerance_rad = 1.2
off_spz_weight = 0.5
bilateral_balance_weight = 0.35
effective_count_weight = 0.15
minimum_weight = 0.1
minimum_effective_fraction = 0.6
maximum_iterations = 50
random_seed = 31
pair_scope = "reference_receiver"
reference_receiver_index = 1
"""
    path = tmp_path / "pair_spz.toml"
    path.write_text(text)
    config = load_config(path)
    assert config.pair_spz.enabled
    assert config.pair_spz.phase_tolerance_rad == 1.2
    assert config.pair_spz.minimum_effective_fraction == 0.6
    assert config.pair_spz.reference_receiver_index == 1


def test_pair_spz_and_nnws_v2_require_matching_pair_scope(tmp_path):
    text = (ROOT / "examples/passive_quick.toml").read_text()
    text += """

[nnws_v2]
enabled = true
pair_scope = "all"

[pair_spz]
enabled = true
pair_scope = "reference_receiver"
reference_receiver_index = 0
"""
    path = tmp_path / "pair_scope_mismatch.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="must use the same pair scope"):
        load_config(path)
