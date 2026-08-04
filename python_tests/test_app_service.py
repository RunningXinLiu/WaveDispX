from copy import deepcopy
from pathlib import Path
import tomllib

import pytest

from wavedispx.app_service import (
    CaseInputError,
    CaseSession,
    explain_input_error,
    parse_shot_csv,
    parse_toml_bytes,
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
