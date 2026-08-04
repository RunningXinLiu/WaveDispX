from dataclasses import replace
from pathlib import Path

import h5py

from wavedispx.config import load_config
from wavedispx.runner import run


ROOT = Path(__file__).resolve().parents[1]


def test_active_quick_end_to_end_writes_complete_artifacts(tmp_path):
    config = replace(
        load_config(ROOT / "examples/active_quick.toml"),
        output_directory=tmp_path / "active",
    )
    artifacts = run(config)
    assert artifacts.summary_png.is_file()
    assert (artifacts.output_directory / "model.csv").is_file()
    assert (artifacts.output_directory / "receivers.csv").is_file()
    assert (artifacts.output_directory / "sources.csv").is_file()
    with h5py.File(artifacts.results_h5) as results:
        assert results.attrs["mode"] == "active"
        assert results["shot/components"].shape[2] == 5
        assert results["dispersion/rayleigh/energy"].shape[0] > 10


def test_passive_quick_end_to_end_is_mixed_and_streaming(tmp_path):
    config = replace(
        load_config(ROOT / "examples/passive_quick.toml"),
        output_directory=tmp_path / "passive",
    )
    artifacts = run(config)
    assert artifacts.summary_png.is_file()
    with h5py.File(artifacts.results_h5) as results:
        assert results.attrs["mode"] == "passive"
        assert results.attrs["correlation_mode"] == "mixed finite-record"
        assert results.attrs["independent_window_count"] == 6
        assert results.attrs["radial_interpolation_relative_l2"] < 0.02
        assert "vertical_spectrum" not in results["transfer"]
        assert results["preview/mixed_record"].shape[1] == 41
        assert results["ncf/gather"].shape[1] == 40


def test_passive_three_component_end_to_end_writes_tensor_products(tmp_path):
    base = load_config(ROOT / "examples/passive_quick.toml")
    config = replace(
        base,
        ambient_sources=replace(base.ambient_sources, component="three_component"),
        noise=replace(base.noise, duration_s=20.0),
        output_directory=tmp_path / "passive_3c",
    )
    artifacts = run(config)
    assert (artifacts.output_directory / "passive_components.png").is_file()
    with h5py.File(artifacts.results_h5) as results:
        assert results["preview/mixed_record"].ndim == 3
        assert results["preview/mixed_record"].shape[2] == 3
        assert set(results["ncf/components"]) == {"rr", "tt", "zz"}
        assert set(results["dispersion/components"]) == {"rr", "tt", "zz"}
        assert "rayleigh_mode_support" in results["quality"]
