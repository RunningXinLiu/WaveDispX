"""Stable result layout for reproducible active/passive runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import resource
import shutil
import sys
from typing import Any

import h5py
import numpy as np

from . import __version__
from .active import ActiveShotGather
from .config import RunConfig
from .geometry import Geometry, save_geometry_csv
from .processing import DispersionImage, DispersionPicks, TheoreticalDispersion


@dataclass(frozen=True)
class RunArtifacts:
    output_directory: Path
    results_h5: Path
    summary_png: Path
    manifest_json: Path


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _runtime_metadata() -> dict[str, str]:
    return {
        "application": "WaveDispX",
        "wavedispx_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "forward_solver": "discrete-wavenumber method (DWM)",
        "compiled_core": "C++17/pybind11",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def prepare_output(config: RunConfig) -> Path:
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    if config.config_path is not None:
        shutil.copy2(config.config_path, output / "input.toml")
    (output / "resolved_config.json").write_text(
        json.dumps(_json_safe(config.to_dict()), ensure_ascii=False, indent=2) + "\n"
    )
    model_rows = np.array(
        [
            (
                layer.thickness_m,
                layer.vp_mps,
                layer.vs_mps,
                layer.density_kgm3,
                layer.qp,
                layer.qs,
            )
            for layer in config.model.layers
        ]
    )
    np.savetxt(
        output / "model.csv",
        model_rows,
        delimiter=",",
        header="thickness_m,vp_mps,vs_mps,density_kgm3,qp,qs",
        comments="",
    )
    return output


def save_picks(path: Path, picks: DispersionPicks) -> None:
    np.savetxt(
        path,
        picks.as_array(),
        delimiter=",",
        header=(
            "frequency_hz,phase_velocity_mps,normalized_energy,"
            "half_power_uncertainty_mps"
        ),
        comments="",
    )


def _save_convergence(path: Path, convergence: list[dict[str, object]]) -> None:
    rows = np.array(
        [
            (
                item["duration_s"],
                item["window_count"],
                item["ncf_relative_l2_to_final"],
                item["dispersion_relative_l2_to_final"],
                item["pick_mae_to_final_mps"],
                item["quality"]["picks"]["mae_mps"],
                float(item["quality"]["image"]["clean_ridge_pass"]),
            )
            for item in convergence
        ],
        dtype=np.float64,
    )
    np.savetxt(
        path,
        rows,
        delimiter=",",
        header=(
            "duration_s,window_count,ncf_relative_l2_to_final,"
            "dispersion_relative_l2_to_final,pick_mae_to_final_mps,"
            "pick_mae_to_theory_mps,clean_ridge_pass"
        ),
        comments="",
    )


def save_active_results(
    config: RunConfig,
    geometry: Geometry,
    shot: ActiveShotGather,
    rayleigh: DispersionImage,
    love: DispersionImage,
    rayleigh_picks: DispersionPicks,
    love_picks: DispersionPicks,
    theory: TheoreticalDispersion,
    quality: dict[str, dict[str, float | bool]],
    *,
    timings: dict[str, float],
    summary_png: Path,
    rayleigh_combined: DispersionImage | None = None,
    rayleigh_combined_picks: DispersionPicks | None = None,
    component_images: dict[str, DispersionImage] | None = None,
    method_comparison: dict[str, DispersionImage] | None = None,
) -> RunArtifacts:
    output = prepare_output(config)
    save_geometry_csv(geometry, output)
    save_picks(output / "picked_rayleigh.csv", rayleigh_picks)
    save_picks(output / "picked_love.csv", love_picks)
    if rayleigh_combined_picks is not None:
        save_picks(output / "picked_rayleigh_combined.csv", rayleigh_combined_picks)
    results_path = output / "results.h5"
    with h5py.File(results_path, "w") as handle:
        handle.attrs["application"] = "WaveDispX"
        handle.attrs["mode"] = "active"
        handle.attrs["forward_solver"] = "discrete-wavenumber method (DWM)"
        handle.attrs["dispersion_method"] = config.processing.dispersion_method
        handle.attrs["wavedispx_version"] = __version__
        handle.attrs["peak_rss_bytes"] = _peak_rss_bytes()
        for name, value in timings.items():
            handle.attrs[f"timing_{name}_s"] = value
        quality_group = handle.create_group("quality")
        for branch, metrics in quality.items():
            branch_group = quality_group.create_group(branch)
            for name, value in metrics.items():
                branch_group.attrs[name] = value
        shot_group = handle.create_group("shot")
        shot_group.create_dataset("time_s", data=shot.time_s)
        shot_group.create_dataset("receiver_x_m", data=shot.receiver_x_m)
        shot_group.create_dataset("receiver_y_m", data=shot.receiver_y_m)
        shot_group.create_dataset("distance_m", data=shot.distance_m)
        shot_group.create_dataset(
            "components", data=shot.components.astype(np.float32), compression="gzip", shuffle=True
        )
        for name, image in (("rayleigh", rayleigh), ("love", love)):
            group = handle.create_group(f"dispersion/{name}")
            group.create_dataset("frequency_hz", data=image.frequency_hz)
            group.create_dataset("velocity_mps", data=image.velocity_mps)
            group.create_dataset("energy", data=image.energy.astype(np.float32), compression="gzip")
        if rayleigh_combined is not None:
            group = handle.create_group("dispersion/rayleigh_combined")
            group.create_dataset("frequency_hz", data=rayleigh_combined.frequency_hz)
            group.create_dataset("velocity_mps", data=rayleigh_combined.velocity_mps)
            group.create_dataset(
                "energy", data=rayleigh_combined.energy.astype(np.float32), compression="gzip"
            )
        for parent, images in (
            ("dispersion/components", component_images or {}),
            ("dispersion/method_comparison", method_comparison or {}),
        ):
            for name, image in images.items():
                group = handle.create_group(f"{parent}/{name}")
                group.attrs["method"] = image.method
                group.create_dataset("frequency_hz", data=image.frequency_hz)
                group.create_dataset("velocity_mps", data=image.velocity_mps)
                group.create_dataset(
                    "energy", data=image.energy.astype(np.float32), compression="gzip"
                )
        theory_group = handle.create_group("theory")
        theory_group.create_dataset("frequency_hz", data=theory.frequency_hz)
        theory_group.create_dataset("rayleigh_mps", data=theory.rayleigh_mps)
        theory_group.create_dataset("love_mps", data=theory.love_mps)
    manifest = {
        "mode": "active",
        "input": "input.toml",
        "resolved_config": "resolved_config.json",
        "model": "model.csv",
        "receivers": "receivers.csv",
        "sources": "sources.csv",
        "results": "results.h5",
        "rayleigh_picks": "picked_rayleigh.csv",
        "love_picks": "picked_love.csv",
        "rayleigh_combined_picks": (
            "picked_rayleigh_combined.csv" if rayleigh_combined_picks is not None else None
        ),
        "summary": summary_png.name,
        "component_comparison": "active_components.png" if component_images else None,
        "method_comparison": (
            "active_method_comparison.png"
            if method_comparison is not None and len(method_comparison) > 1
            else None
        ),
        "timings_seconds": timings,
        "peak_rss_bytes": _peak_rss_bytes(),
        "quality": quality,
        "runtime": _runtime_metadata(),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return RunArtifacts(output, results_path, summary_png, manifest_path)


def save_passive_results(
    config: RunConfig,
    simulation,
    *,
    summary_png: Path,
) -> RunArtifacts:
    output = prepare_output(config)
    save_geometry_csv(simulation.geometry, output)
    save_picks(output / "picked_rayleigh.csv", simulation.picks)
    _save_convergence(output / "convergence.csv", simulation.convergence)
    noise_events_path = None
    if simulation.noise_event_catalog is not None:
        noise_events_path = output / "noise_events.csv"
        three_component_events = simulation.noise_event_catalog.shape[1] == 8
        event_header = (
            "window_index,source_index,source_component_index,source_x_m,source_y_m,"
            "time_in_window_s,event_time_s,weight"
            if three_component_events
            else "window_index,source_index,source_x_m,source_y_m,"
            "time_in_window_s,event_time_s,weight"
        )
        event_format = (
            ["%d", "%d", "%d", "%.9g", "%.9g", "%.12g", "%.12g", "%.12g"]
            if three_component_events
            else ["%d", "%d", "%.9g", "%.9g", "%.12g", "%.12g", "%.12g"]
        )
        np.savetxt(
            noise_events_path,
            simulation.noise_event_catalog,
            delimiter=",",
            header=event_header,
            comments="",
            fmt=event_format,
        )
    results_path = output / "results.h5"
    with h5py.File(results_path, "w") as handle:
        handle.attrs["application"] = "WaveDispX"
        handle.attrs["mode"] = "passive"
        handle.attrs["forward_solver"] = "discrete-wavenumber method (DWM)"
        handle.attrs["correlation_mode"] = "mixed finite-record"
        handle.attrs["wavedispx_version"] = __version__
        handle.attrs["independent_window_count"] = simulation.independent_window_count
        handle.attrs["transfer_bytes"] = simulation.transfer.bytes
        handle.attrs["response_sample_count"] = simulation.transfer.response_sample_count
        handle.attrs["radial_grid_count"] = simulation.transfer.radial_grid_count
        handle.attrs["radial_interpolation_step_m"] = simulation.transfer.radial_step_m
        handle.attrs["radial_interpolation_relative_l2"] = (
            simulation.transfer.interpolation_relative_l2
        )
        handle.attrs["peak_rss_bytes"] = _peak_rss_bytes()
        for name, value in simulation.timings.items():
            handle.attrs[f"timing_{name}_s"] = value
        quality_group = handle.create_group("quality")
        for branch, metrics in simulation.quality.items():
            branch_group = quality_group.create_group(branch)
            for name, value in metrics.items():
                branch_group.attrs[name] = value
        convergence_group = handle.create_group("convergence")
        for index, item in enumerate(simulation.convergence):
            checkpoint = convergence_group.create_group(str(index))
            for name in (
                "duration_s",
                "window_count",
                "ncf_relative_l2_to_final",
                "dispersion_relative_l2_to_final",
                "pick_mae_to_final_mps",
            ):
                checkpoint.attrs[name] = item[name]
            for branch, metrics in item["quality"].items():
                branch_group = checkpoint.create_group(branch)
                for name, value in metrics.items():
                    branch_group.attrs[name] = value
        preview = handle.create_group("preview")
        preview.create_dataset("time_s", data=simulation.preview_time_s)
        preview.create_dataset(
            "mixed_record",
            data=simulation.preview_record,
            compression="gzip",
            shuffle=True,
        )
        if simulation.noise_event_catalog is not None:
            events = handle.create_group("noise_events")
            event_names = (
                (
                    "window_index",
                    "source_index",
                    "source_component_index",
                    "source_x_m",
                    "source_y_m",
                    "time_in_window_s",
                    "event_time_s",
                    "weight",
                )
                if simulation.noise_event_catalog.shape[1] == 8
                else (
                    "window_index",
                    "source_index",
                    "source_x_m",
                    "source_y_m",
                    "time_in_window_s",
                    "event_time_s",
                    "weight",
                )
            )
            for column_index, name in enumerate(event_names):
                data = simulation.noise_event_catalog[:, column_index]
                if name in {"window_index", "source_index", "source_component_index"}:
                    data = data.astype(np.int64)
                events.create_dataset(name, data=data)
        ncf = handle.create_group("ncf")
        ncf.create_dataset("lag_time_s", data=simulation.ncf.lag_time_s)
        ncf.create_dataset("distance_m", data=simulation.ncf.distance_m)
        ncf.create_dataset("pair_count", data=simulation.ncf.pair_count)
        ncf.create_dataset(
            "gather", data=simulation.ncf.data.astype(np.float32), compression="gzip"
        )
        for name, component_ncf in simulation.component_ncfs.items():
            component_group = handle.create_group(f"ncf/components/{name}")
            component_group.create_dataset(
                "gather",
                data=component_ncf.data.astype(np.float32),
                compression="gzip",
            )
        dispersion = handle.create_group("dispersion/rayleigh")
        dispersion.create_dataset("frequency_hz", data=simulation.dispersion.frequency_hz)
        dispersion.create_dataset("velocity_mps", data=simulation.dispersion.velocity_mps)
        dispersion.create_dataset(
            "energy", data=simulation.dispersion.energy.astype(np.float32), compression="gzip"
        )
        for parent, images in (
            ("dispersion/components", simulation.component_images),
            ("dispersion/method_comparison", simulation.method_comparison),
        ):
            for name, image in images.items():
                group = handle.create_group(f"{parent}/{name}")
                group.attrs["method"] = image.method
                group.create_dataset("frequency_hz", data=image.frequency_hz)
                group.create_dataset("velocity_mps", data=image.velocity_mps)
                group.create_dataset(
                    "energy", data=image.energy.astype(np.float32), compression="gzip"
                )
        theory = handle.create_group("theory")
        theory.create_dataset("frequency_hz", data=simulation.theory.frequency_hz)
        theory.create_dataset("rayleigh_mps", data=simulation.theory.rayleigh_mps)
        theory.create_dataset("love_mps", data=simulation.theory.love_mps)
        transfer = handle.create_group("transfer")
        transfer.create_dataset("frequency_hz", data=simulation.transfer.frequency_hz)
        if config.store_source_responses:
            transfer.create_dataset(
                (
                    "full_green_spectrum"
                    if simulation.transfer.spectrum.ndim == 4
                    else "vertical_spectrum"
                ),
                data=simulation.transfer.spectrum,
                compression="gzip",
                shuffle=True,
            )
    manifest = {
        "mode": "passive",
        "correlation_mode": "all sources mixed before finite-record correlation",
        "input": "input.toml",
        "resolved_config": "resolved_config.json",
        "model": "model.csv",
        "receivers": "receivers.csv",
        "sources": "sources.csv",
        "results": "results.h5",
        "rayleigh_picks": "picked_rayleigh.csv",
        "convergence_csv": "convergence.csv",
        "noise_events": noise_events_path.name if noise_events_path else None,
        "summary": summary_png.name,
        "component_comparison": (
            "passive_components.png" if len(simulation.component_images) > 1 else None
        ),
        "method_comparison": (
            "passive_method_comparison.png"
            if len(simulation.method_comparison) > 1
            else None
        ),
        "ambient_source_component": config.ambient_sources.component,
        "independent_window_count": simulation.independent_window_count,
        "transfer_bytes": simulation.transfer.bytes,
        "radial_grid_count": simulation.transfer.radial_grid_count,
        "radial_interpolation_step_m": simulation.transfer.radial_step_m,
        "radial_interpolation_relative_l2": simulation.transfer.interpolation_relative_l2,
        "stored_source_transfer": config.store_source_responses,
        "timings_seconds": simulation.timings,
        "peak_rss_bytes": _peak_rss_bytes(),
        "quality": simulation.quality,
        "convergence": simulation.convergence,
        "runtime": _runtime_metadata(),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return RunArtifacts(output, results_path, summary_png, manifest_path)
