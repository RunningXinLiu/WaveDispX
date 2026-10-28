from copy import deepcopy
from pathlib import Path
import tomllib

import pytest

from wavedispx.app_service import (
    apply_automatic_parameters,
    CaseInputError,
    CaseSession,
    explain_input_error,
    parse_shot_csv,
    parse_toml_bytes,
    reset_automatic_parameters,
    stage_uploaded_file,
    toml_dumps,
)
from wavedispx.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _mapping(name: str = "passive_quick.toml"):
    return parse_toml_bytes((ROOT / "examples" / name).read_bytes())


def test_toml_round_trip_preserves_valid_case(tmp_path):
    mapping = _mapping()
    text = toml_dumps(mapping)
    assert tomllib.loads(text) == mapping
    session = CaseSession(tmp_path)
    config, invalidated = session.apply_mapping(tomllib.loads(text))
    assert config.mode == "passive"
    assert invalidated == ["forward", "observations", "analysis"]


@pytest.mark.parametrize("name", ("active_quick.toml", "passive_quick.toml"))
def test_automatic_parameters_are_materialized_and_remain_editable(name):
    mapping = _mapping(name)
    automatic, diagnostics = apply_automatic_parameters(
        mapping, base_directory=ROOT
    )
    processing = automatic["processing"]
    assert processing["frequency_step_hz"] > 0.0
    assert processing["velocity_step_mps"] > 0.0
    assert processing["frequency_min_hz"] == pytest.approx(
        diagnostics["recommended_frequency_min_hz"]
    )
    if automatic["run"]["mode"] == "active":
        assert automatic["active_source"]["record_duration_s"] > 0.0
    else:
        assert 0.0 < automatic["noise"]["maximum_lag_s"] < automatic["noise"]["window_s"]
        assert automatic["noise"]["duration_s"] >= 600.0
        assert automatic["noise"]["duration_s"] / automatic["noise"]["window_s"] >= 60
        assert automatic["ambient_sources"] == mapping["ambient_sources"]
    from wavedispx.config import config_from_mapping
    from wavedispx.preflight import build_preflight_review
    assert build_preflight_review(
        config_from_mapping(automatic, base_directory=ROOT)
    ).ready

    automatic["processing"]["frequency_step_hz"] *= 2.0
    assert automatic["processing"]["frequency_step_hz"] != mapping["processing"].get(
        "frequency_step_hz", 0.0
    )


def test_reset_automatic_parameters_does_not_replace_model_or_geometry():
    mapping = _mapping("passive_quick.toml")
    template = deepcopy(mapping)
    mapping["model"]["layers"][0]["vs_mps"] = 211.0
    mapping["geometry"]["receivers"]["stop_x_m"] = 210.0
    mapping["processing"]["frequency_step_hz"] = 0.037
    mapping["noise"]["maximum_lag_s"] = 1.5
    mapping["noise"]["duration_s"] = 1200.0
    restored = reset_automatic_parameters(mapping, template)
    assert restored["model"]["layers"][0]["vs_mps"] == 211.0
    assert restored["geometry"]["receivers"]["stop_x_m"] == 210.0
    assert restored["processing"]["frequency_step_hz"] == 0.0
    assert restored["noise"]["maximum_lag_s"] == 0.0
    assert restored["noise"]["duration_s"] == template["noise"]["duration_s"]


def test_short_aperture_does_not_hide_source_supported_low_frequencies():
    automatic, diagnostics = apply_automatic_parameters(_mapping("active_quick.toml"), base_directory=ROOT)
    assert automatic["processing"]["frequency_min_hz"] < 5.0
    assert automatic["processing"]["frequency_max_hz"] > 17.0


def test_sampling_recommendation_rounds_down():
    from wavedispx.survey_design import _nice_step
    for bound in (0.018, 0.03, 0.09, 0.3, 0.8, 1.9, 3.0, 9.0):
        assert 0 < _nice_step(bound) <= bound


def test_toml_round_trip_preserves_inline_and_nested_arrays():
    mapping = _mapping("passive_source_selection_quick.toml")
    text = toml_dumps(mapping)
    restored = tomllib.loads(text)
    assert restored == mapping
    assert "frequency_bands_hz = [[3.5, 6.5], [6.5, 9.5]]" in text


def test_upload_staging_is_content_addressed_and_sanitized(tmp_path):
    first = stage_uploaded_file(
        tmp_path,
        "../../receivers.csv",
        b"x_m,y_m\n0,0\n1,0\n",
        allowed_suffixes=(".csv",),
    )
    second = stage_uploaded_file(
        tmp_path,
        "receivers.csv",
        b"x_m,y_m\n0,0\n2,0\n",
        allowed_suffixes=(".csv",),
    )
    assert first.parent == second.parent == (tmp_path / "uploads").resolve()
    assert first.name != second.name
    assert ".." not in first.name


def test_upload_rejects_empty_or_wrong_suffix(tmp_path):
    with pytest.raises(CaseInputError, match="类型不支持"):
        stage_uploaded_file(
            tmp_path, "receivers.txt", b"x_m,y_m\n0,0\n", allowed_suffixes=(".csv",)
        )
    with pytest.raises(CaseInputError, match="为空"):
        stage_uploaded_file(
            tmp_path, "receivers.csv", b"", allowed_suffixes=(".csv",)
        )


def test_shot_csv_supports_named_multi_shot_selection():
    shots = parse_shot_csv(
        b"shot_id,x_m,y_m\nS001,0,0\nS002,-120.5,4\nS003,130,-2\n"
    )
    assert [(shot.shot_id, shot.x_m, shot.y_m) for shot in shots] == [
        ("S001", 0.0, 0.0),
        ("S002", -120.5, 4.0),
        ("S003", 130.0, -2.0),
    ]


def test_shot_csv_assigns_ids_and_rejects_duplicate_ids():
    shots = parse_shot_csv(b"x_m,y_m\n0,0\n1,2\n")
    assert [shot.shot_id for shot in shots] == ["shot_001", "shot_002"]
    with pytest.raises(CaseInputError, match="shot_id"):
        parse_shot_csv(b"shot_id,x_m,y_m\nS001,0,0\nS001,1,2\n")


def test_same_coordinate_path_with_new_content_invalidates_forward(tmp_path):
    receiver_file = tmp_path / "receivers.csv"
    receiver_file.write_text("x_m,y_m\n5,0\n10,0\n15,0\n20,0\n")
    mapping = _mapping("active_quick.toml")
    mapping["geometry"]["receivers"] = {
        "layout": "file",
        "coordinates_file": str(receiver_file),
    }
    session = CaseSession(tmp_path)
    session.apply_mapping(mapping)
    original = session.workflow.fingerprint

    receiver_file.write_text("x_m,y_m\n5,0\n10,1\n15,0\n20,-1\n")
    _, invalidated = session.apply_mapping(deepcopy(mapping))
    assert session.workflow.fingerprint != original
    assert invalidated == ["forward", "observations", "analysis"]


def test_configuration_error_explanation_points_to_field_and_fix():
    issue = explain_input_error(
        ValueError("processing.frequency_max_hz must not exceed noise.maximum_frequency_hz")
    )
    assert "频散计算范围" in issue.location
    assert "处理最高频率" in issue.field_path
    assert "PSD最高频率" in issue.suggestion
    assert issue.technical_message.startswith("processing.frequency_max_hz")


def test_required_upload_error_explanation_is_actionable():
    issue = explain_input_error(CaseInputError("请选择炮点 Shot CSV"))
    assert "主动源与子波" in issue.location
    assert issue.field_path == "Shot CSV"
    assert "shot_id,x_m,y_m" in issue.suggestion


@pytest.mark.parametrize("message, field, suggestion", [
    ("source activity rows contain invalid indexes or amplitudes", "activity_file", "source_index"),
    ("weighted polarization v1 supports stationary_gaussian only", "polarization", "stationary_gaussian"),
    ("horizontal_power_fraction must be in [0,1]", "horizontal_power_fraction", "0–1"),
    ("non-default force direction/amplitude requires active_source.component='vector'", "force_azimuth", "vector"),
    ("active DAS v1 requires active_source.component='vertical'", "das.enabled", "尚未实现"),
    ("节点HVSR需要被动three_component的XYZ观测且关闭DAS", "experiment.product", "关闭附加分析"),
])
def test_source_product_errors_have_chinese_location_and_repair(message, field, suggestion):
    issue = explain_input_error(ValueError(message))
    assert field in issue.field_path
    assert suggestion in issue.suggestion
    assert issue.technical_message == message


def test_saved_ui_case_rewrites_uploaded_coordinates_to_portable_paths(tmp_path):
    receiver_file = tmp_path / "uploaded_receivers.csv"
    receiver_file.write_text("x_m,y_m\n5,0\n10,0\n15,0\n20,0\n")
    mapping = _mapping("active_quick.toml")
    mapping["run"]["output_directory"] = str(tmp_path / "result")
    mapping["geometry"]["receivers"] = {
        "layout": "file",
        "coordinates_file": str(receiver_file),
    }
    session = CaseSession(tmp_path)
    session.apply_mapping(mapping)
    artifacts = session.workflow.run_all(save=True)
    portable_path = artifacts.output_directory / "input.toml"
    portable = tomllib.loads(portable_path.read_text())
    assert portable["geometry"]["receivers"]["coordinates_file"] == "receivers.csv"
    assert load_config(portable_path).receivers.count == 4
