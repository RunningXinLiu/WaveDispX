"""Stable result layout for reproducible active/passive runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import resource
import shutil
import sys
import tomllib
from typing import Any

import h5py
import numpy as np

from . import __version__
from .active import ActiveShotGather
from .config import RunConfig
from .geometry import Geometry, save_geometry_csv
from .processing import DispersionImage, DispersionPicks, TheoreticalDispersion
from .spz_selection import (
    blend_physical_and_data_weights,
    combine_physical_and_data_weights,
)


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


def _write_portable_input(config: RunConfig, output: Path) -> None:
    """Rewrite file-backed inputs to the canonical copies saved in output."""
    if config.config_path is None:
        return
    from .app_service import toml_dumps

    with config.config_path.open("rb") as stream:
        mapping = tomllib.load(stream)
    receivers = mapping.get("geometry", {}).get("receivers", {})
    if receivers.get("layout") == "file":
        receivers["coordinates_file"] = (
            "das_channels.csv"
            if config.das.enabled and config.das.geometry_mode == "channel_file"
            else "receivers.csv"
        )
    ambient = mapping.get("ambient_sources", {})
    if ambient.get("layout") == "file":
        ambient["coordinates_file"] = "sources.csv"
    noise = mapping.get("noise", {})
    if noise.get("psd_shape") == "file" and config.noise is not None:
        destination = output / "noise_psd.csv"
        source = config.noise.psd_file
        if source is not None and source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        noise["psd_file"] = "noise_psd.csv"
    (output / "input.toml").write_text(toml_dumps(mapping), encoding="utf-8")


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


def _write_das_geometry(group: h5py.Group, geometry) -> None:
    """Persist the complete per-channel geometry without losing legacy attrs."""
    uniform_gauge = geometry.gauge_length_m
    uniform_azimuth = geometry.fiber_azimuth_deg
    group.attrs["geometry_mode"] = geometry.geometry_mode
    group.attrs["uniform_gauge_length"] = bool(np.isfinite(uniform_gauge))
    group.attrs["uniform_fiber_azimuth"] = bool(np.isfinite(uniform_azimuth))
    group.attrs["gauge_length_m"] = uniform_gauge
    group.attrs["fiber_azimuth_deg"] = uniform_azimuth
    group.attrs["gauge_length_min_m"] = float(
        np.min(geometry.gauge_length_by_channel_m)
    )
    group.attrs["gauge_length_max_m"] = float(
        np.max(geometry.gauge_length_by_channel_m)
    )
    group.create_dataset("channel_center_xy_m", data=geometry.channel_center_xy_m)
    group.create_dataset("endpoint_minus_xy_m", data=geometry.endpoint_minus_xy_m)
    group.create_dataset("endpoint_plus_xy_m", data=geometry.endpoint_plus_xy_m)
    group.create_dataset("tangent_xy", data=geometry.tangent_xy)
    group.create_dataset(
        "gauge_length_by_channel_m", data=geometry.gauge_length_by_channel_m
    )
    group.create_dataset(
        "fiber_azimuth_by_channel_deg",
        data=geometry.fiber_azimuth_by_channel_deg,
    )
    group.create_dataset(
        "channel_id",
        data=np.asarray(geometry.channel_id, dtype=h5py.string_dtype("utf-8")),
    )


def _das_geometry_manifest(geometry) -> dict[str, object]:
    uniform_gauge = geometry.gauge_length_m
    uniform_azimuth = geometry.fiber_azimuth_deg
    return {
        "geometry_mode": geometry.geometry_mode,
        "channel_count": geometry.channel_count,
        "gauge_length_m": (
            float(uniform_gauge) if np.isfinite(uniform_gauge) else None
        ),
        "gauge_length_range_m": [
            float(np.min(geometry.gauge_length_by_channel_m)),
            float(np.max(geometry.gauge_length_by_channel_m)),
        ],
        "fiber_azimuth_deg": (
            float(uniform_azimuth) if np.isfinite(uniform_azimuth) else None
        ),
        "per_channel_geometry_datasets": [
            "channel_id",
            "channel_center_xy_m",
            "endpoint_minus_xy_m",
            "endpoint_plus_xy_m",
            "tangent_xy",
            "gauge_length_by_channel_m",
            "fiber_azimuth_by_channel_deg",
        ],
    }


def _save_das_channels_csv(path: Path, geometry) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "channel_id",
                "x_m",
                "y_m",
                "endpoint_minus_x_m",
                "endpoint_minus_y_m",
                "endpoint_plus_x_m",
                "endpoint_plus_y_m",
                "gauge_length_m",
                "fiber_azimuth_deg",
            )
        )
        for index, channel_id in enumerate(geometry.channel_id):
            writer.writerow(
                (
                    channel_id,
                    *geometry.channel_center_xy_m[index],
                    *geometry.endpoint_minus_xy_m[index],
                    *geometry.endpoint_plus_xy_m[index],
                    geometry.gauge_length_by_channel_m[index],
                    geometry.fiber_azimuth_by_channel_deg[index],
                )
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
    if shot.das_record is not None:
        save_picks(output / "picked_das.csv", rayleigh_picks)
        _save_das_channels_csv(
            output / "das_channels.csv", shot.das_record.geometry
        )
    save_picks(output / "picked_love.csv", love_picks)
    if rayleigh_combined_picks is not None:
        save_picks(output / "picked_rayleigh_combined.csv", rayleigh_combined_picks)
    results_path = output / "results.h5"
    record_fft_period_s = shot.time_s.size * config.active_source.dt_s
    record_span_s = (
        float(shot.time_s[-1] - shot.time_s[0]) if shot.time_s.size > 1 else 0.0
    )
    green_delay_s = 4.0 / config.active_source.dominant_frequency_hz
    green_delay_samples = int(green_delay_s / config.active_source.dt_s)
    dwm_period_s = (
        shot.time_s.size + green_delay_samples
    ) * config.active_source.dt_s
    with h5py.File(results_path, "w") as handle:
        handle.attrs["application"] = "WaveDispX"
        handle.attrs["mode"] = "active"
        handle.attrs["forward_solver"] = "discrete-wavenumber method (DWM)"
        handle.attrs["forward_solver_domain"] = "frequency-wavenumber"
        handle.attrs["dispersion_product"] = "windowed_waveform_image"
        handle.attrs["dispersion_method"] = config.processing.dispersion_method
        handle.attrs["record_duration_mode"] = (
            "automatic"
            if config.active_source.record_duration_s is None
            else "explicit"
        )
        handle.attrs["record_sample_count"] = shot.time_s.size
        handle.attrs["record_dt_s"] = config.active_source.dt_s
        handle.attrs["record_span_s"] = record_span_s
        handle.attrs["record_fft_period_s"] = record_fft_period_s
        handle.attrs["record_frequency_step_hz"] = 1.0 / record_fft_period_s
        handle.attrs["dispersion_display_frequency_step_hz"] = (
            float(np.median(np.diff(rayleigh.frequency_hz)))
            if rayleigh.frequency_hz.size > 1
            else np.nan
        )
        handle.attrs["dwm_synthetic_period_s"] = dwm_period_s
        handle.attrs["dwm_frequency_step_hz"] = 1.0 / dwm_period_s
        handle.attrs["dwm_complex_frequency_shift_radps"] = 2.0 * np.pi / dwm_period_s
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
        if shot.das_record is not None:
            record = shot.das_record
            das_group = handle.create_group("das")
            das_group.attrs["measurement"] = record.measurement
            das_group.attrs["units"] = record.units
            das_group.attrs["operator_version"] = record.operator_version
            das_group.attrs["time_convention"] = record.time_convention
            das_group.attrs["input_observable"] = record.input_observable
            das_group.create_dataset("time_s", data=record.time_s)
            das_group.create_dataset(
                "data",
                data=record.data.astype(np.float32),
                compression="gzip",
                shuffle=True,
            )
            _write_das_geometry(das_group, record.geometry)
        for name, image in (("rayleigh", rayleigh), ("love", love)):
            group = handle.create_group(f"dispersion/{name}")
            group.create_dataset("frequency_hz", data=image.frequency_hz)
            group.create_dataset("velocity_mps", data=image.velocity_mps)
            group.create_dataset("energy", data=image.energy.astype(np.float32), compression="gzip")
            group.create_dataset(
                "raw_amplitude",
                data=image.raw_amplitude.astype(np.float32),
                compression="gzip",
            )
        if rayleigh_combined is not None:
            group = handle.create_group("dispersion/rayleigh_combined")
            group.create_dataset("frequency_hz", data=rayleigh_combined.frequency_hz)
            group.create_dataset("velocity_mps", data=rayleigh_combined.velocity_mps)
            group.create_dataset(
                "energy", data=rayleigh_combined.energy.astype(np.float32), compression="gzip"
            )
            group.create_dataset(
                "raw_amplitude",
                data=rayleigh_combined.raw_amplitude.astype(np.float32),
                compression="gzip",
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
                group.create_dataset(
                    "raw_amplitude",
                    data=image.raw_amplitude.astype(np.float32),
                    compression="gzip",
                )
        theory_group = handle.create_group("theory")
        theory_group.attrs["solver"] = theory.solver
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
        "das_picks": "picked_das.csv" if shot.das_record is not None else None,
        "das_channels": "das_channels.csv" if shot.das_record is not None else None,
        "love_picks": "picked_love.csv",
        "rayleigh_combined_picks": (
            "picked_rayleigh_combined.csv" if rayleigh_combined_picks is not None else None
        ),
        "summary": summary_png.name,
        "forward_solver_domain": "frequency-wavenumber",
        "dispersion_product": "windowed_waveform_image",
        "record_duration": {
            "mode": (
                "automatic"
                if config.active_source.record_duration_s is None
                else "explicit"
            ),
            "sample_count": int(shot.time_s.size),
            "dt_s": config.active_source.dt_s,
            "span_s": record_span_s,
            "fft_period_s": record_fft_period_s,
            "frequency_step_hz": 1.0 / record_fft_period_s,
        },
        "dispersion_grid": {
            "display_frequency_step_hz": (
                float(np.median(np.diff(rayleigh.frequency_hz)))
                if rayleigh.frequency_hz.size > 1
                else None
            ),
            "velocity_step_mps": config.processing.velocity_step_mps,
        },
        "dwm_complex_frequency": {
            "synthetic_period_s": dwm_period_s,
            "frequency_step_hz": 1.0 / dwm_period_s,
            "shift_radps": 2.0 * np.pi / dwm_period_s,
        },
        "component_comparison": "active_components.png" if component_images else None,
        "method_comparison": (
            "active_method_comparison.png"
            if method_comparison is not None and len(method_comparison) > 1
            else None
        ),
        "timings_seconds": timings,
        "peak_rss_bytes": _peak_rss_bytes(),
        "das": (
            {
                "operator_version": shot.das_record.operator_version,
                "measurement": shot.das_record.measurement,
                "time_convention": shot.das_record.time_convention,
                **_das_geometry_manifest(shot.das_record.geometry),
            }
            if shot.das_record is not None
            else None
        ),
        "quality": quality,
        "runtime": _runtime_metadata(),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _write_portable_input(config, output)
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
    if config.das.enabled:
        save_picks(output / "picked_das.csv", simulation.picks)
        _save_das_channels_csv(
            output / "das_channels.csv", simulation.transfer.das_geometry
        )
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
    source_selection_path = None
    if simulation.source_selection is not None:
        selection = simulation.source_selection
        source_selection_path = output / "source_selection_windows.csv"
        columns = [
            selection.window_index,
            selection.start_time_s,
            selection.end_time_s,
            selection.hard_selected.astype(np.int64),
            selection.soft_weight,
            selection.random_budget_selected.astype(np.int64),
            selection.random_neff_weight,
        ]
        names = [
            "window_index",
            "start_time_s",
            "end_time_s",
            "oracle_hard_mask",
            "oracle_soft_weight",
            "random_budget_control",
            "random_neff_control",
        ]
        for band_index in range(selection.frequency_bands_hz.shape[0]):
            for metric_name in (
                "total_power",
                "endfire_fraction",
                "bidirectional_balance",
                "off_axis_fraction",
                "azimuth_entropy",
                "effective_source_count",
            ):
                columns.append(getattr(selection, metric_name)[:, band_index])
                names.append(f"{metric_name}_band_{band_index}")
        np.savetxt(
            source_selection_path,
            np.column_stack(columns),
            delimiter=",",
            header=",".join(names),
            comments="",
            fmt="%.12g",
        )
    nnws_pair_audit_path = None
    if simulation.pairwise_nnws is not None:
        nnws_result = simulation.pairwise_nnws
        pair_branches = nnws_result.pair_stack.branches
        pair_statistics = nnws_result.pair_stack.statistics
        nnws_pair_audit_path = output / "nnws_pair_audit.csv"
        with nnws_pair_audit_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "pair_index",
                    "receiver_i",
                    "receiver_j",
                    "distance_m",
                    "eligible",
                    "eligibility_reason",
                    "sufficient",
                    "nonzero_window_count",
                    "sum_weight",
                    "sum_weight_squared",
                    "effective_window_count",
                    "training_snr",
                    "reported_snr",
                    "symmetry_ratio",
                    "final_loss",
                    "failure_reason",
                )
            )
            for pair_index in range(pair_branches.distance_m.size):
                writer.writerow(
                    (
                        pair_index,
                        int(pair_branches.pair_i[pair_index]),
                        int(pair_branches.pair_j[pair_index]),
                        float(pair_branches.distance_m[pair_index]),
                        int(nnws_result.eligible[pair_index]),
                        nnws_result.eligibility_reason[pair_index],
                        int(pair_statistics.sufficient[pair_index]),
                        int(pair_statistics.nonzero_window_count[pair_index]),
                        float(pair_statistics.sum_weight[pair_index]),
                        float(pair_statistics.sum_weight_squared[pair_index]),
                        float(pair_statistics.effective_window_count[pair_index]),
                        float(nnws_result.training_snr[pair_index]),
                        float(nnws_result.reported_snr[pair_index]),
                        float(nnws_result.symmetry_ratio[pair_index]),
                        float(nnws_result.final_loss[pair_index]),
                        nnws_result.failure_reason[pair_index],
                    )
                )
    nnws_v2_window_audit_path = None
    if simulation.nnws_v2 is not None:
        fit = simulation.nnws_v2.fit
        random_weights = (
            simulation.nnws_v2.random_control_stack.statistics.weights[:, 0]
        )
        window_index = np.arange(fit.weights.size, dtype=np.int64)
        nnws_v2_window_audit_path = output / "nnws_v2_window_audit.csv"
        np.savetxt(
            nnws_v2_window_audit_path,
            np.column_stack(
                (
                    window_index,
                    window_index * config.noise.window_s,
                    (window_index + 1) * config.noise.window_s,
                    fit.fold_index,
                    fit.weights,
                    random_weights,
                )
            ),
            delimiter=",",
            header=(
                "window_index,start_time_s,end_time_s,fold_index,"
                "nnws_v2_shared_oof_weight,random_neff_control_weight"
            ),
            comments="",
            fmt="%.12g",
        )
    supervised_oracle_window_audit_path = None
    if simulation.supervised_egf_oracle is not None:
        oracle = simulation.supervised_egf_oracle
        window_index = np.arange(oracle.positive.size, dtype=np.int64)
        permutation = np.random.default_rng(
            config.nnws_v2.random_control_seed + 31
        ).permutation(window_index.size)
        supervised_oracle_window_audit_path = (
            output / "supervised_egf_oracle_window_audit.csv"
        )
        np.savetxt(
            supervised_oracle_window_audit_path,
            np.column_stack(
                (
                    window_index,
                    window_index * config.noise.window_s,
                    (window_index + 1) * config.noise.window_s,
                    oracle.positive,
                    oracle.negative,
                    oracle.positive[permutation],
                    oracle.negative[permutation],
                )
            ),
            delimiter=",",
            header=(
                "window_index,start_time_s,end_time_s,positive_branch_weight,"
                "negative_branch_weight,permuted_positive_branch_weight,"
                "permuted_negative_branch_weight"
            ),
            comments="",
            fmt="%.12g",
        )
    supervised_pair_oracle_audit_path = None
    if simulation.supervised_pair_egf_oracle is not None:
        oracle = simulation.supervised_pair_egf_oracle
        window_count, pair_count = oracle.positive.shape
        permutation = np.random.default_rng(
            config.nnws_v2.random_control_seed + 31
        ).permutation(window_count)
        window_index = np.repeat(np.arange(window_count), pair_count)
        pair_index = np.tile(np.arange(pair_count), window_count)
        supervised_pair_oracle_audit_path = (
            output / "supervised_pair_egf_oracle_weight_audit.csv"
        )
        np.savetxt(
            supervised_pair_oracle_audit_path,
            np.column_stack(
                (
                    window_index,
                    pair_index,
                    oracle.pair_i[pair_index],
                    oracle.pair_j[pair_index],
                    oracle.positive.reshape(-1),
                    oracle.negative.reshape(-1),
                    oracle.positive[permutation].reshape(-1),
                    oracle.negative[permutation].reshape(-1),
                )
            ),
            delimiter=",",
            header=(
                "window_index,pair_index,pair_i,pair_j,positive_branch_weight,"
                "negative_branch_weight,permuted_positive_branch_weight,"
                "permuted_negative_branch_weight"
            ),
            comments="",
            fmt="%.12g",
        )
    pair_crossfit_coherence_audit_path = None
    if simulation.pair_crossfit_coherence is not None:
        selection = simulation.pair_crossfit_coherence
        window_count, pair_count = selection.positive_weights.shape
        window_index = np.repeat(np.arange(window_count), pair_count)
        pair_index = np.tile(np.arange(pair_count), window_count)
        pair_crossfit_coherence_audit_path = (
            output / "pair_crossfit_coherence_audit.csv"
        )
        np.savetxt(
            pair_crossfit_coherence_audit_path,
            np.column_stack(
                (
                    window_index,
                    pair_index,
                    selection.pair_i[pair_index],
                    selection.pair_j[pair_index],
                    selection.fold_index[window_index],
                    selection.positive_weights.reshape(-1),
                    selection.negative_weights.reshape(-1),
                    selection.positive_score.reshape(-1),
                    selection.negative_score.reshape(-1),
                    selection.positive_arrival_correlation.reshape(-1),
                    selection.negative_arrival_correlation.reshape(-1),
                    selection.positive_phase_coherence.reshape(-1),
                    selection.negative_phase_coherence.reshape(-1),
                    selection.positive_prearrival_fraction.reshape(-1),
                    selection.negative_prearrival_fraction.reshape(-1),
                    selection.positive_prearrival_weights.reshape(-1),
                    selection.negative_prearrival_weights.reshape(-1),
                )
            ),
            delimiter=",",
            header=(
                "window_index,pair_index,pair_i,pair_j,fold_index,"
                "positive_weight,negative_weight,positive_score,negative_score,"
                "positive_arrival_correlation,negative_arrival_correlation,"
                "positive_phase_coherence,negative_phase_coherence,"
                "positive_prearrival_fraction,negative_prearrival_fraction"
                ",positive_prearrival_weight,negative_prearrival_weight"
            ),
            comments="",
            fmt="%.12g",
        )
    pair_tau_p_audit_path = None
    if simulation.pair_tau_p_selection is not None:
        selection = simulation.pair_tau_p_selection
        window_count, band_count, pair_count = (
            selection.positive_band_weights.shape
        )
        window_index = np.repeat(
            np.arange(window_count), band_count * pair_count
        )
        band_index = np.tile(
            np.repeat(np.arange(band_count), pair_count), window_count
        )
        pair_index = np.tile(np.arange(pair_count), window_count * band_count)
        permutation = np.random.default_rng(
            config.nnws_v2.random_control_seed + 47
        ).permutation(window_count)
        directional_guard = (
            np.full((window_count, band_count), np.nan)
            if selection.nnws_v2_directional_guard_band_weights is None
            else selection.nnws_v2_directional_guard_band_weights
        )
        directional_guard_control = (
            np.full((window_count, band_count), np.nan)
            if selection.nnws_v2_directional_guard_control_band_weights is None
            else selection.nnws_v2_directional_guard_control_band_weights
        )
        joint_diffuse_prearrival_hard = (
            np.full((window_count, band_count), np.nan)
            if selection.diffuse_prearrival_hard_band_weights is None
            else selection.diffuse_prearrival_hard_band_weights
        )
        positive_branch_guard = (
            np.full((window_count, band_count, pair_count), np.nan)
            if selection.nnws_v2_positive_branch_guard_band_weights is None
            else selection.nnws_v2_positive_branch_guard_band_weights
        )
        negative_branch_guard = (
            np.full((window_count, band_count, pair_count), np.nan)
            if selection.nnws_v2_negative_branch_guard_band_weights is None
            else selection.nnws_v2_negative_branch_guard_band_weights
        )
        positive_branch_guard_control = (
            np.full((window_count, band_count, pair_count), np.nan)
            if selection.nnws_v2_positive_branch_guard_control_band_weights is None
            else selection.nnws_v2_positive_branch_guard_control_band_weights
        )
        negative_branch_guard_control = (
            np.full((window_count, band_count, pair_count), np.nan)
            if selection.nnws_v2_negative_branch_guard_control_band_weights is None
            else selection.nnws_v2_negative_branch_guard_control_band_weights
        )
        positive_diffuse_prearrival = (
            np.full((window_count, band_count, pair_count), np.nan)
            if selection.nnws_v2_positive_diffuse_prearrival_band_weights is None
            else selection.nnws_v2_positive_diffuse_prearrival_band_weights
        )
        negative_diffuse_prearrival = (
            np.full((window_count, band_count, pair_count), np.nan)
            if selection.nnws_v2_negative_diffuse_prearrival_band_weights is None
            else selection.nnws_v2_negative_diffuse_prearrival_band_weights
        )
        positive_diffuse_prearrival_control = (
            np.full((window_count, band_count, pair_count), np.nan)
            if selection.nnws_v2_positive_diffuse_prearrival_control_band_weights
            is None
            else selection.nnws_v2_positive_diffuse_prearrival_control_band_weights
        )
        negative_diffuse_prearrival_control = (
            np.full((window_count, band_count, pair_count), np.nan)
            if selection.nnws_v2_negative_diffuse_prearrival_control_band_weights
            is None
            else selection.nnws_v2_negative_diffuse_prearrival_control_band_weights
        )
        pair_tau_p_audit_path = output / "pair_crossfit_tau_p_spz_audit.csv"
        np.savetxt(
            pair_tau_p_audit_path,
            np.column_stack(
                (
                    window_index,
                    band_index,
                    selection.band_center_hz[band_index],
                    pair_index,
                    selection.pair_i[pair_index],
                    selection.pair_j[pair_index],
                    selection.fold_index[window_index],
                    selection.directional_imbalance[
                        window_index, band_index
                    ],
                    selection.diffuse_soft_band_weights[
                        window_index, band_index
                    ],
                    selection.diffuse_hard_band_weights[
                        window_index, band_index
                    ],
                    joint_diffuse_prearrival_hard[window_index, band_index],
                    directional_guard[window_index, band_index],
                    directional_guard_control[window_index, band_index],
                    positive_branch_guard.reshape(-1),
                    negative_branch_guard.reshape(-1),
                    positive_branch_guard_control.reshape(-1),
                    negative_branch_guard_control.reshape(-1),
                    positive_diffuse_prearrival.reshape(-1),
                    negative_diffuse_prearrival.reshape(-1),
                    positive_diffuse_prearrival_control.reshape(-1),
                    negative_diffuse_prearrival_control.reshape(-1),
                    selection.positive_spz_score.reshape(-1),
                    selection.negative_spz_score.reshape(-1),
                    selection.positive_band_weights.reshape(-1),
                    selection.negative_band_weights.reshape(-1),
                    selection.positive_band_weights[permutation].reshape(-1),
                    selection.negative_band_weights[permutation].reshape(-1),
                )
            ),
            delimiter=",",
            header=(
                "window_index,band_index,band_center_hz,pair_index,pair_i,pair_j,"
                "fold_index,directional_imbalance,diffuse_soft_weight,"
                "diffuse_hard_weight,diffuse_prearrival_hard_weight,"
                "nnws_v2_directional_guard_weight,"
                "permuted_nnws_v2_directional_guard_weight,"
                "nnws_v2_positive_branch_guard_weight,"
                "nnws_v2_negative_branch_guard_weight,"
                "permuted_nnws_v2_positive_branch_guard_weight,"
                "permuted_nnws_v2_negative_branch_guard_weight,"
                "nnws_v2_positive_diffuse_prearrival_weight,"
                "nnws_v2_negative_diffuse_prearrival_weight,"
                "permuted_nnws_v2_positive_diffuse_prearrival_weight,"
                "permuted_nnws_v2_negative_diffuse_prearrival_weight,"
                "positive_spz_score,negative_spz_score,positive_weight,"
                "negative_weight,permuted_positive_weight,permuted_negative_weight"
            ),
            comments="",
            fmt="%.12g",
        )
    pair_spz_window_audit_path = None
    pair_spz_pair_weight_audit_path = None
    if simulation.pair_spz_selection is not None:
        selection = simulation.pair_spz_selection
        combined_weights = (
            combine_physical_and_data_weights(
                simulation.nnws_v2.fit.weights,
                selection.weights,
                minimum_effective_fraction=max(
                    config.nnws_v2.minimum_effective_fraction,
                    config.pair_spz.minimum_effective_fraction,
                ),
            )
            if simulation.nnws_v2 is not None
            else None
        )
        phase_hybrid_band_weights = None
        phase_hybrid_control_band_weights = None
        if simulation.nnws_v2 is not None:
            phase_hybrid_band_weights = np.column_stack(
                [
                    blend_physical_and_data_weights(
                        simulation.nnws_v2.fit.weights,
                        selection.band_weights[:, band_index],
                        physical_fraction=0.20,
                        minimum_effective_fraction=max(
                            config.nnws_v2.minimum_effective_fraction,
                            config.pair_spz.minimum_effective_fraction,
                        ),
                    )
                    for band_index in range(selection.frequency_bands_hz.shape[0])
                ]
            )
            permutation = np.random.default_rng(
                config.pair_spz.random_seed + 2
            ).permutation(selection.window_index.size)
            phase_hybrid_control_band_weights = phase_hybrid_band_weights[
                permutation
            ]
        columns = [
            selection.window_index,
            selection.start_time_s,
            selection.end_time_s,
            selection.weights,
            selection.random_neff_weights,
        ]
        names = [
            "window_index",
            "start_time_s",
            "end_time_s",
            "oracle_pair_spz_weight",
            "oracle_pair_spz_random_neff_weight",
        ]
        if combined_weights is not None:
            columns.append(combined_weights)
            names.append("nnws_v2_spz_constrained_weight")
        for band_index in range(selection.frequency_bands_hz.shape[0]):
            if phase_hybrid_band_weights is not None:
                columns.extend(
                    (
                        phase_hybrid_band_weights[:, band_index],
                        phase_hybrid_control_band_weights[:, band_index],
                    )
                )
                names.extend(
                    (
                        f"nnws_v2_spz_phase_hybrid_weight_band_{band_index}",
                        "nnws_v2_spz_phase_hybrid_permuted_time_control_weight_"
                        f"band_{band_index}",
                    )
                )
            columns.extend(
                (
                    selection.band_weights[:, band_index],
                    selection.band_random_neff_weights[:, band_index],
                    selection.positive_branch_band_weights[:, band_index],
                    selection.negative_branch_band_weights[:, band_index],
                    selection.positive_branch_band_random_weights[:, band_index],
                    selection.negative_branch_band_random_weights[:, band_index],
                )
            )
            names.extend(
                (
                    f"oracle_pair_spz_weight_band_{band_index}",
                    f"oracle_pair_spz_random_neff_weight_band_{band_index}",
                    f"oracle_pair_spz_positive_branch_weight_band_{band_index}",
                    f"oracle_pair_spz_negative_branch_weight_band_{band_index}",
                    "oracle_pair_spz_positive_branch_random_weight_"
                    f"band_{band_index}",
                    "oracle_pair_spz_negative_branch_random_weight_"
                    f"band_{band_index}",
                )
            )
            for name in (
                "positive_spz_power",
                "negative_spz_power",
                "off_spz_power",
                "spz_fraction",
                "per_window_balance",
            ):
                columns.append(getattr(selection, name)[:, band_index])
                names.append(f"{name}_band_{band_index}")
        pair_spz_window_audit_path = output / "pair_spz_window_audit.csv"
        np.savetxt(
            pair_spz_window_audit_path,
            np.column_stack(columns),
            delimiter=",",
            header=",".join(names),
            comments="",
            fmt="%.12g",
        )
        window_count, pair_count = selection.pair_positive_branch_weights.shape
        window_index = np.repeat(np.arange(window_count), pair_count)
        pair_index = np.tile(np.arange(pair_count), window_count)
        pair_spz_pair_weight_audit_path = (
            output / "pair_spz_pair_weight_audit.csv"
        )
        np.savetxt(
            pair_spz_pair_weight_audit_path,
            np.column_stack(
                (
                    window_index,
                    pair_index,
                    selection.pair_i[pair_index],
                    selection.pair_j[pair_index],
                    selection.pair_positive_branch_weights.reshape(-1),
                    selection.pair_negative_branch_weights.reshape(-1),
                    selection.pair_positive_branch_random_weights.reshape(-1),
                    selection.pair_negative_branch_random_weights.reshape(-1),
                )
            ),
            delimiter=",",
            header=(
                "window_index,pair_index,pair_i,pair_j,positive_branch_weight,"
                "negative_branch_weight,random_positive_branch_weight,"
                "random_negative_branch_weight"
            ),
            comments="",
            fmt="%.12g",
        )
    results_path = output / "results.h5"
    passive_window_s = config.noise.window_s
    passive_total_duration_s = config.noise.duration_s
    dwm_period_s = simulation.transfer.response_sample_count * config.noise.dt_s
    with h5py.File(results_path, "w") as handle:
        handle.attrs["application"] = "WaveDispX"
        handle.attrs["mode"] = "passive"
        handle.attrs["forward_solver"] = "discrete-wavenumber method (DWM)"
        handle.attrs["forward_solver_domain"] = "frequency-wavenumber"
        handle.attrs["dispersion_product"] = "finite_record_ncf_waveform_image"
        handle.attrs["correlation_mode"] = "mixed finite-record"
        handle.attrs["noise_total_duration_s"] = passive_total_duration_s
        handle.attrs["noise_independent_window_s"] = passive_window_s
        handle.attrs["ncf_frequency_step_hz"] = 1.0 / passive_window_s
        handle.attrs["ncf_maximum_lag_s"] = float(
            simulation.ncf.lag_time_s[-1]
        )
        handle.attrs["dispersion_display_frequency_step_hz"] = (
            float(np.median(np.diff(simulation.dispersion.frequency_hz)))
            if simulation.dispersion.frequency_hz.size > 1
            else np.nan
        )
        handle.attrs["dwm_synthetic_period_s"] = dwm_period_s
        handle.attrs["dwm_frequency_step_hz"] = 1.0 / dwm_period_s
        handle.attrs["dwm_complex_frequency_shift_radps"] = 2.0 * np.pi / dwm_period_s
        handle.attrs["wavedispx_version"] = __version__
        handle.attrs["independent_window_count"] = simulation.independent_window_count
        handle.attrs["transfer_bytes"] = simulation.transfer.bytes
        handle.attrs["egf_reference_bytes"] = (
            simulation.egf_reference.bytes
            if simulation.egf_reference is not None
            else 0
        )
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
        strategy_quality_group = quality_group.create_group("strategies")
        for strategy_name, sections in simulation.strategy_dispersion_quality.items():
            strategy_group = strategy_quality_group.create_group(strategy_name)
            dispersion_quality_group = strategy_group.create_group("dispersion")
            for section_name, metrics in sections.items():
                section_group = dispersion_quality_group.create_group(section_name)
                for name, value in metrics.items():
                    section_group.attrs[name] = value
            egf_quality_group = strategy_group.create_group("egf")
            for comparison_name, metrics in simulation.strategy_egf_quality.get(
                strategy_name, {}
            ).items():
                comparison_group = egf_quality_group.create_group(comparison_name)
                for name, value in metrics.items():
                    comparison_group.attrs[name] = value
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
        if simulation.source_selection is not None:
            selection = simulation.source_selection
            selection_group = handle.create_group("source_selection")
            selection_group.attrs["scope"] = selection.scope
            selection_group.attrs["method_version"] = selection.method_version
            selection_group.attrs["objective"] = config.source_selection.objective
            selection_group.attrs["correlated_source_risk"] = (
                selection.correlated_source_risk
            )
            selection_group.attrs["stationary_phase_half_angle_deg"] = (
                config.source_selection.stationary_phase_half_angle_deg
            )
            selection_group.attrs["hard_min_endfire_fraction"] = (
                config.source_selection.hard_min_endfire_fraction
            )
            selection_group.attrs["hard_min_bidirectional_balance"] = (
                config.source_selection.hard_min_bidirectional_balance
            )
            selection_group.attrs["hard_min_azimuth_entropy"] = (
                config.source_selection.hard_min_azimuth_entropy
            )
            selection_group.create_dataset(
                "frequency_bands_hz", data=selection.frequency_bands_hz
            )
            for name in (
                "window_index",
                "start_time_s",
                "end_time_s",
                "total_power",
                "endfire_fraction",
                "bidirectional_balance",
                "off_axis_fraction",
                "azimuth_entropy",
                "effective_source_count",
                "hard_selected",
                "soft_weight",
                "random_budget_selected",
                "random_neff_weight",
            ):
                selection_group.create_dataset(name, data=getattr(selection, name))
            weights_group = selection_group.create_group("weights")
            for name, weights in selection.strategy_weights.items():
                weights_group.create_dataset(name, data=weights)
        if simulation.pairwise_nnws is not None:
            nnws_result = simulation.pairwise_nnws
            pair_branches = nnws_result.pair_stack.branches
            pair_statistics = nnws_result.pair_stack.statistics
            nnws_group = handle.create_group("nnws_pairwise")
            nnws_group.attrs["scope"] = nnws_result.scope
            nnws_group.attrs["pair_scope"] = config.nnws.pair_scope
            nnws_group.attrs["reference_receiver_index"] = (
                config.nnws.reference_receiver_index
            )
            nnws_group.attrs["strategy_name"] = "nnws_pairwise"
            nnws_group.attrs["std_mode"] = config.nnws.std_mode
            nnws_group.attrs["epochs"] = config.nnws.epochs
            nnws_group.attrs["learning_rate"] = config.nnws.learning_rate
            nnws_group.attrs["snr_weight"] = config.nnws.snr_weight
            nnws_group.attrs["symmetry_weight"] = config.nnws.symmetry_weight
            nnws_group.attrs["std_weight"] = config.nnws.std_weight
            nnws_group.attrs["seed"] = config.nnws.seed
            nnws_group.attrs["random_control_seed"] = (
                config.nnws.random_control_seed
            )
            nnws_group.attrs["device"] = config.nnws.device
            nnws_group.attrs["minimum_effective_windows"] = (
                config.nnws.minimum_effective_windows
            )
            nnws_group.attrs["minimum_pair_coverage"] = (
                config.nnws.minimum_pair_coverage
            )
            eligible_count = int(np.count_nonzero(nnws_result.eligible))
            successful_count = int(
                np.count_nonzero(
                    nnws_result.eligible & pair_statistics.sufficient
                )
            )
            nnws_group.attrs["eligible_pair_count"] = eligible_count
            nnws_group.attrs["ineligible_pair_count"] = int(
                nnws_result.eligible.size - eligible_count
            )
            nnws_group.attrs["successful_pair_count"] = successful_count
            nnws_group.attrs["pair_coverage"] = (
                successful_count / eligible_count if eligible_count else 0.0
            )
            nnws_group.attrs["scalar_strategy_statistics"] = (
                "medians across sufficient receiver pairs; exact per-pair values "
                "are stored in this group"
            )
            nnws_group.create_dataset("pair_i", data=pair_branches.pair_i)
            nnws_group.create_dataset("pair_j", data=pair_branches.pair_j)
            nnws_group.create_dataset(
                "distance_m", data=pair_branches.distance_m
            )
            nnws_group.create_dataset(
                "signal_window_start_s",
                data=(
                    pair_branches.distance_m
                    / config.processing.velocity_max_mps
                ),
            )
            nnws_group.create_dataset(
                "signal_window_end_s",
                data=(
                    pair_branches.distance_m
                    / config.processing.velocity_min_mps
                ),
            )
            window_index = np.arange(
                pair_statistics.weights.shape[0], dtype=np.int64
            )
            nnws_group.create_dataset("window_index", data=window_index)
            nnws_group.create_dataset(
                "window_start_time_s", data=window_index * config.noise.window_s
            )
            nnws_group.create_dataset(
                "window_end_time_s",
                data=(window_index + 1) * config.noise.window_s,
            )
            nnws_group.create_dataset(
                "weights",
                data=pair_statistics.weights.astype(np.float32),
                compression="gzip",
                shuffle=True,
            )
            nnws_group.create_dataset("eligible", data=nnws_result.eligible)
            nnws_group.create_dataset(
                "eligibility_reason",
                data=np.asarray(
                    nnws_result.eligibility_reason,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
            )
            for name in (
                "sum_weight",
                "sum_weight_squared",
                "effective_window_count",
                "nonzero_window_count",
                "sufficient",
            ):
                nnws_group.create_dataset(name, data=getattr(pair_statistics, name))
            for name in (
                "training_snr",
                "reported_snr",
                "symmetry_ratio",
                "final_loss",
            ):
                nnws_group.create_dataset(name, data=getattr(nnws_result, name))
            nnws_group.create_dataset(
                "failure_reason",
                data=np.asarray(
                    nnws_result.failure_reason,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
            )
            controls_group = nnws_group.create_group("controls")
            for control_name, control_stack in (
                ("common_linear", nnws_result.common_linear_stack),
                ("random_neff", nnws_result.random_control_stack),
                (
                    "hybrid_full_aperture",
                    nnws_result.hybrid_full_aperture_stack,
                ),
                (
                    "hybrid_random_neff",
                    nnws_result.hybrid_random_control_stack,
                ),
            ):
                control_group = controls_group.create_group(control_name)
                control_statistics = control_stack.statistics
                control_group.create_dataset(
                    "weights",
                    data=control_statistics.weights.astype(np.float32),
                    compression="gzip",
                    shuffle=True,
                )
                for name in (
                    "sum_weight",
                    "sum_weight_squared",
                    "effective_window_count",
                    "nonzero_window_count",
                    "sufficient",
                ):
                    control_group.create_dataset(
                        name, data=getattr(control_statistics, name)
                    )
        if simulation.nnws_v2 is not None:
            nnws_v2_result = simulation.nnws_v2
            fit = nnws_v2_result.fit
            nnws_v2_group = handle.create_group("nnws_v2")
            nnws_v2_group.attrs["scope"] = nnws_v2_result.scope
            nnws_v2_group.attrs["pair_scope"] = config.nnws_v2.pair_scope
            nnws_v2_group.attrs["reference_receiver_index"] = (
                config.nnws_v2.reference_receiver_index
            )
            nnws_v2_group.attrs["strategy_name"] = "nnws_v2_shared_oof"
            for name in (
                "folds",
                "epochs",
                "learning_rate",
                "arrival_weight",
                "precursor_weight",
                "symmetry_weight",
                "effective_count_weight",
                "temporal_smoothness_weight",
                "minimum_weight",
                "minimum_effective_fraction",
                "seed",
                "random_control_seed",
                "device",
            ):
                nnws_v2_group.attrs[name] = getattr(config.nnws_v2, name)
            nnws_v2_group.attrs["effective_window_count"] = (
                fit.effective_window_count
            )
            nnws_v2_group.attrs["sufficient"] = fit.sufficient
            nnws_v2_group.attrs["proxy_distance_m"] = fit.proxy_distance_m
            nnws_v2_group.attrs["signal_window_start_s"] = fit.signal_window_s[0]
            nnws_v2_group.attrs["signal_window_end_s"] = fit.signal_window_s[1]
            nnws_v2_group.create_dataset("weights", data=fit.weights)
            nnws_v2_group.create_dataset("fold_index", data=fit.fold_index)
            nnws_v2_group.create_dataset(
                "proxy_pair_mask", data=fit.proxy_pair_mask
            )
            nnws_v2_group.create_dataset(
                "proxy_matrix",
                data=fit.proxy_matrix.astype(np.float32),
                compression="gzip",
                shuffle=True,
            )
            nnws_v2_group.create_dataset(
                "proxy_stack", data=fit.proxy_stack.astype(np.float32)
            )
            nnws_v2_group.create_dataset(
                "random_control_weights",
                data=(
                    nnws_v2_result.random_control_stack.statistics.weights[:, 0]
                ),
            )
            history_group = nnws_v2_group.create_group("fold_history")
            for fold_index, history in enumerate(fit.fold_history):
                fold_group = history_group.create_group(str(fold_index))
                for name, values in history.items():
                    fold_group.create_dataset(name, data=np.asarray(values))
        if simulation.pair_spz_selection is not None:
            selection = simulation.pair_spz_selection
            spz_group = handle.create_group("pair_spz")
            spz_group.attrs["scope"] = selection.scope
            spz_group.attrs["method_version"] = selection.method_version
            spz_group.attrs["pair_scope"] = config.pair_spz.pair_scope
            spz_group.attrs["reference_receiver_index"] = (
                config.pair_spz.reference_receiver_index
            )
            spz_group.attrs["pair_count"] = selection.pair_count
            spz_group.attrs["phase_tolerance_rad"] = (
                config.pair_spz.phase_tolerance_rad
            )
            spz_group.attrs["effective_window_count"] = (
                selection.effective_window_count
            )
            spz_group.attrs["optimization_success"] = (
                selection.optimization_success
            )
            spz_group.attrs["optimization_message"] = (
                selection.optimization_message
            )
            for name in (
                "frequency_bands_hz",
                "window_index",
                "start_time_s",
                "end_time_s",
                "positive_spz_power",
                "negative_spz_power",
                "off_spz_power",
                "spz_fraction",
                "per_window_balance",
                "weights",
                "random_neff_weights",
                "band_weights",
                "band_random_neff_weights",
                "band_effective_window_count",
                "positive_branch_band_weights",
                "negative_branch_band_weights",
                "positive_branch_band_random_weights",
                "negative_branch_band_random_weights",
                "branch_band_effective_window_count",
                "pair_positive_branch_weights",
                "pair_negative_branch_weights",
                "pair_positive_branch_random_weights",
                "pair_negative_branch_random_weights",
                "pair_branch_effective_window_count",
                "pair_positive_branch_band_weights",
                "pair_negative_branch_band_weights",
                "pair_positive_branch_band_random_weights",
                "pair_negative_branch_band_random_weights",
                "pair_branch_band_effective_window_count",
                "pair_i",
                "pair_j",
                "optimized_spz_fraction",
                "optimized_bilateral_balance",
            ):
                spz_group.create_dataset(name, data=getattr(selection, name))
            if simulation.nnws_v2 is not None:
                combined_weights = combine_physical_and_data_weights(
                    simulation.nnws_v2.fit.weights,
                    selection.weights,
                    minimum_effective_fraction=max(
                        config.nnws_v2.minimum_effective_fraction,
                        config.pair_spz.minimum_effective_fraction,
                    ),
                )
                spz_group.create_dataset(
                    "nnws_v2_spz_constrained_weights", data=combined_weights
                )
                phase_hybrid_band_weights = np.column_stack(
                    [
                        blend_physical_and_data_weights(
                            simulation.nnws_v2.fit.weights,
                            selection.band_weights[:, band_index],
                            physical_fraction=0.20,
                            minimum_effective_fraction=max(
                                config.nnws_v2.minimum_effective_fraction,
                                config.pair_spz.minimum_effective_fraction,
                            ),
                        )
                        for band_index in range(
                            selection.frequency_bands_hz.shape[0]
                        )
                    ]
                )
                permutation = np.random.default_rng(
                    config.pair_spz.random_seed + 2
                ).permutation(selection.window_index.size)
                spz_group.create_dataset(
                    "nnws_v2_spz_phase_hybrid_band_weights",
                    data=phase_hybrid_band_weights,
                )
                spz_group.create_dataset(
                    "nnws_v2_spz_phase_hybrid_permuted_time_control_band_weights",
                    data=phase_hybrid_band_weights[permutation],
                )
        if simulation.supervised_egf_oracle is not None:
            oracle = simulation.supervised_egf_oracle
            oracle_group = handle.create_group("supervised_egf_oracle")
            oracle_group.attrs["scope"] = oracle.scope
            oracle_group.attrs["positive_effective_window_count"] = (
                oracle.positive_effective_window_count
            )
            oracle_group.attrs["negative_effective_window_count"] = (
                oracle.negative_effective_window_count
            )
            oracle_group.create_dataset("positive_weights", data=oracle.positive)
            oracle_group.create_dataset("negative_weights", data=oracle.negative)
        if simulation.supervised_pair_egf_oracle is not None:
            oracle = simulation.supervised_pair_egf_oracle
            oracle_group = handle.create_group("supervised_pair_egf_oracle")
            oracle_group.attrs["scope"] = oracle.scope
            for name in (
                "positive",
                "negative",
                "positive_effective_window_count",
                "negative_effective_window_count",
                "pair_i",
                "pair_j",
                "positive_fit_success",
                "negative_fit_success",
            ):
                oracle_group.create_dataset(name, data=getattr(oracle, name))
        if simulation.pair_crossfit_coherence is not None:
            selection = simulation.pair_crossfit_coherence
            selection_group = handle.create_group("pair_crossfit_coherence")
            selection_group.attrs["scope"] = selection.scope
            for name in (
                "positive_weights",
                "negative_weights",
                "positive_score",
                "negative_score",
                "positive_arrival_correlation",
                "negative_arrival_correlation",
                "positive_phase_coherence",
                "negative_phase_coherence",
                "positive_prearrival_fraction",
                "negative_prearrival_fraction",
                "positive_prearrival_weights",
                "negative_prearrival_weights",
                "fold_index",
                "positive_effective_window_count",
                "negative_effective_window_count",
                "positive_prearrival_effective_window_count",
                "negative_prearrival_effective_window_count",
                "pair_i",
                "pair_j",
            ):
                selection_group.create_dataset(name, data=getattr(selection, name))
        if simulation.pair_tau_p_selection is not None:
            selection = simulation.pair_tau_p_selection
            selection_group = handle.create_group("pair_crossfit_tau_p_spz")
            selection_group.attrs["scope"] = selection.scope
            if selection.nnws_v2_directional_fraction is not None:
                selection_group.attrs["nnws_v2_directional_fraction"] = (
                    selection.nnws_v2_directional_fraction
                )
            if selection.nnws_v2_diffuse_prearrival_frequency_fraction is not None:
                selection_group.attrs[
                    "nnws_v2_diffuse_prearrival_frequency_fraction"
                ] = selection.nnws_v2_diffuse_prearrival_frequency_fraction
            if selection.diffuse_prearrival_directional_fraction is not None:
                selection_group.attrs[
                    "diffuse_prearrival_directional_fraction"
                ] = selection.diffuse_prearrival_directional_fraction
            for name in (
                "band_center_hz",
                "slowness_s_per_m",
                "reference_slowness_s_per_m",
                "positive_band_weights",
                "negative_band_weights",
                "positive_spz_score",
                "negative_spz_score",
                "directional_imbalance",
                "diffuse_soft_band_weights",
                "diffuse_hard_band_weights",
                "diffuse_prearrival_hard_band_weights",
                "diffuse_soft_effective_window_count",
                "diffuse_hard_effective_window_count",
                "positive_effective_window_count",
                "negative_effective_window_count",
                "fold_index",
                "pair_i",
                "pair_j",
            ):
                selection_group.create_dataset(name, data=getattr(selection, name))
            for name in (
                "nnws_v2_directional_guard_band_weights",
                "nnws_v2_directional_guard_control_band_weights",
                "nnws_v2_positive_branch_guard_band_weights",
                "nnws_v2_negative_branch_guard_band_weights",
                "nnws_v2_positive_branch_guard_control_band_weights",
                "nnws_v2_negative_branch_guard_control_band_weights",
                "nnws_v2_positive_diffuse_prearrival_band_weights",
                "nnws_v2_negative_diffuse_prearrival_band_weights",
                "nnws_v2_positive_diffuse_prearrival_control_band_weights",
                "nnws_v2_negative_diffuse_prearrival_control_band_weights",
            ):
                value = getattr(selection, name)
                if value is not None:
                    selection_group.create_dataset(name, data=value)
        if simulation.pair_window_ncf is not None:
            pair_window = simulation.pair_window_ncf
            pair_group = handle.create_group("pair_window_ncf")
            pair_group.attrs["pair_i"] = pair_window.pair_i
            pair_group.attrs["pair_j"] = pair_window.pair_j
            pair_group.attrs["distance_m"] = pair_window.distance_m
            pair_group.attrs["window_count"] = pair_window.symmetric.shape[0]
            pair_group.attrs["provenance"] = "before distance-bin averaging"
            pair_group.create_dataset("lag_time_s", data=pair_window.lag_time_s)
            for branch_name in ("positive", "negative", "symmetric"):
                pair_group.create_dataset(
                    branch_name,
                    data=getattr(pair_window, branch_name),
                    compression="gzip",
                    shuffle=True,
                )
        ncf = handle.create_group("ncf")
        ncf.create_dataset("lag_time_s", data=simulation.ncf.lag_time_s)
        ncf.create_dataset("distance_m", data=simulation.ncf.distance_m)
        ncf.create_dataset("pair_count", data=simulation.ncf.pair_count)
        ncf.create_dataset(
            "gather", data=simulation.ncf.data.astype(np.float32), compression="gzip"
        )
        ncf.attrs["default_strategy"] = "baseline_linear"
        ncf.attrs["default_branch"] = "symmetric"
        for name, component_ncf in simulation.component_ncfs.items():
            component_group = handle.create_group(f"ncf/components/{name}")
            component_group.create_dataset(
                "gather",
                data=component_ncf.data.astype(np.float32),
                compression="gzip",
            )
        for strategy_name, stack in simulation.strategy_stacks.items():
            strategy_group = handle.create_group(f"ncf/strategies/{strategy_name}")
            statistics = stack.statistics
            strategy_group.attrs["windows_seen"] = statistics.windows_seen
            strategy_group.attrs["nonzero_window_count"] = (
                statistics.nonzero_window_count
            )
            strategy_group.attrs["sum_weight"] = statistics.sum_weight
            strategy_group.attrs["sum_weight_squared"] = (
                statistics.sum_weight_squared
            )
            strategy_group.attrs["effective_window_count"] = (
                statistics.effective_window_count
            )
            strategy_group.attrs["sufficient"] = statistics.sufficient
            first_branches = next(iter(stack.component_branches.values()))
            strategy_group.create_dataset(
                "lag_time_s", data=first_branches.symmetric.lag_time_s
            )
            strategy_group.create_dataset(
                "distance_m", data=first_branches.symmetric.distance_m
            )
            strategy_group.create_dataset(
                "pair_count", data=first_branches.symmetric.pair_count
            )
            for component_name, branches in stack.component_branches.items():
                component_group = strategy_group.create_group(
                    f"components/{component_name}"
                )
                for branch_name in ("positive", "negative", "symmetric"):
                    component_group.create_dataset(
                        branch_name,
                        data=getattr(branches, branch_name).data.astype(np.float32),
                        compression="gzip",
                    )
        dispersion = handle.create_group("dispersion/rayleigh")
        dispersion.create_dataset("frequency_hz", data=simulation.dispersion.frequency_hz)
        dispersion.create_dataset("velocity_mps", data=simulation.dispersion.velocity_mps)
        dispersion.create_dataset(
            "energy", data=simulation.dispersion.energy.astype(np.float32), compression="gzip"
        )
        dispersion.create_dataset(
            "raw_amplitude",
            data=simulation.dispersion.raw_amplitude.astype(np.float32),
            compression="gzip",
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
                group.create_dataset(
                    "raw_amplitude",
                    data=image.raw_amplitude.astype(np.float32),
                    compression="gzip",
                )
        for strategy_name, image in simulation.strategy_images.items():
            strategy_group = handle.create_group(
                f"dispersion/strategies/{strategy_name}"
            )
            strategy_group.attrs["method"] = image.method
            strategy_group.create_dataset("frequency_hz", data=image.frequency_hz)
            strategy_group.create_dataset("velocity_mps", data=image.velocity_mps)
            strategy_group.create_dataset(
                "energy", data=image.energy.astype(np.float32), compression="gzip"
            )
            strategy_group.create_dataset(
                "raw_amplitude",
                data=image.raw_amplitude.astype(np.float32),
                compression="gzip",
            )
            for component_name, component_image in (
                simulation.strategy_component_images[strategy_name].items()
            ):
                component_group = strategy_group.create_group(
                    f"components/{component_name}"
                )
                component_group.attrs["method"] = component_image.method
                component_group.create_dataset(
                    "energy",
                    data=component_image.energy.astype(np.float32),
                    compression="gzip",
                )
                component_group.create_dataset(
                    "raw_amplitude",
                    data=component_image.raw_amplitude.astype(np.float32),
                    compression="gzip",
                )
        theory = handle.create_group("theory")
        theory.attrs["solver"] = simulation.theory.solver
        theory.create_dataset("frequency_hz", data=simulation.theory.frequency_hz)
        theory.create_dataset("rayleigh_mps", data=simulation.theory.rayleigh_mps)
        theory.create_dataset("love_mps", data=simulation.theory.love_mps)
        if simulation.egf_reference is not None:
            reference = handle.create_group("egf_reference")
            reference.attrs["reference_type"] = "active_interreceiver_green_function"
            reference.attrs["source_time_function"] = (
                simulation.egf_reference.source_time_function
            )
            reference.attrs["calibration_operator"] = (
                simulation.egf_reference.calibration_operator
            )
            reference.attrs["component_names"] = np.asarray(
                simulation.egf_reference.component_names,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            reference.attrs["component_convention"] = (
                "receiver_response_due_to_source_load"
            )
            reference.attrs["time_unit"] = "s"
            reference.attrs["distance_unit"] = "m"
            reference.attrs["amplitude_unit"] = "DWM displacement per unit impulse load"
            reference.attrs["delay_removed_s"] = simulation.egf_reference.delay_removed_s
            reference.attrs["maximum_frequency_hz"] = (
                simulation.egf_reference.maximum_frequency_hz
            )
            reference.create_dataset("time_s", data=simulation.egf_reference.time_s)
            reference.create_dataset(
                "distance_m", data=simulation.egf_reference.distance_m
            )
            reference.create_dataset("pair_i", data=simulation.egf_reference.pair_i)
            reference.create_dataset("pair_j", data=simulation.egf_reference.pair_j)
            reference.create_dataset(
                "pair_distance_index",
                data=simulation.egf_reference.pair_distance_index,
            )
            reference.create_dataset(
                "components",
                data=simulation.egf_reference.components.astype(np.float32),
                compression="gzip",
                shuffle=True,
            )
        elif config.das.enabled:
            handle.attrs["egf_reference_status"] = (
                "not_computed: DAS-DAS active interreceiver reference is not yet implemented"
            )
        if simulation.transfer.das_geometry is not None:
            geometry = simulation.transfer.das_geometry
            das_group = handle.create_group("das")
            das_group.attrs["measurement"] = config.das.measurement
            das_group.attrs["units"] = (
                "1" if config.das.measurement == "strain" else "s^-1"
            )
            das_group.attrs["operator_version"] = "wavedispx-finite-gauge-v1"
            das_group.attrs["time_convention"] = (
                "numpy_rfft_forward_exp(-iwt); derivative=+iomega"
            )
            das_group.attrs["input_observable"] = "horizontal_displacement_xy"
            das_group.create_dataset("time_s", data=simulation.preview_time_s)
            das_group.create_dataset(
                config.das.measurement,
                data=simulation.preview_record.astype(np.float32),
                compression="gzip",
                shuffle=True,
            )
            _write_das_geometry(das_group, geometry)
        transfer = handle.create_group("transfer")
        transfer.create_dataset("frequency_hz", data=simulation.transfer.frequency_hz)
        if config.store_source_responses:
            transfer.create_dataset(
                (
                    "das_spectrum"
                    if config.das.enabled
                    else (
                        "full_green_spectrum"
                        if simulation.transfer.spectrum.ndim == 4
                        else "vertical_spectrum"
                    )
                ),
                data=simulation.transfer.spectrum,
                compression="gzip",
                shuffle=True,
            )
    manifest = {
        "mode": "passive",
        "correlation_mode": "all sources mixed before finite-record correlation",
        "forward_solver_domain": "frequency-wavenumber",
        "dispersion_product": "finite_record_ncf_waveform_image",
        "record_duration": {
            "total_s": passive_total_duration_s,
            "independent_window_s": passive_window_s,
            "independent_window_count": simulation.independent_window_count,
            "physical_frequency_step_hz": 1.0 / passive_window_s,
            "ncf_maximum_lag_s": float(simulation.ncf.lag_time_s[-1]),
        },
        "dispersion_grid": {
            "display_frequency_step_hz": (
                float(np.median(np.diff(simulation.dispersion.frequency_hz)))
                if simulation.dispersion.frequency_hz.size > 1
                else None
            ),
            "velocity_step_mps": config.processing.velocity_step_mps,
        },
        "dwm_complex_frequency": {
            "synthetic_period_s": dwm_period_s,
            "frequency_step_hz": 1.0 / dwm_period_s,
            "shift_radps": 2.0 * np.pi / dwm_period_s,
        },
        "input": "input.toml",
        "resolved_config": "resolved_config.json",
        "model": "model.csv",
        "receivers": "receivers.csv",
        "sources": "sources.csv",
        "results": "results.h5",
        "rayleigh_picks": "picked_rayleigh.csv",
        "das_picks": "picked_das.csv" if config.das.enabled else None,
        "das_channels": "das_channels.csv" if config.das.enabled else None,
        "convergence_csv": "convergence.csv",
        "noise_events": noise_events_path.name if noise_events_path else None,
        "source_selection_windows": (
            source_selection_path.name if source_selection_path else None
        ),
        "nnws_pair_audit": (
            nnws_pair_audit_path.name if nnws_pair_audit_path else None
        ),
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
        "receiver_noise_rms_fraction": config.noise.receiver_noise_rms_fraction,
        "receiver_noise_seed": config.noise.receiver_noise_seed,
        "independent_window_count": simulation.independent_window_count,
        "transfer_bytes": simulation.transfer.bytes,
        "egf_reference_bytes": (
            simulation.egf_reference.bytes
            if simulation.egf_reference is not None
            else 0
        ),
        "egf_reference_source_time_function": (
            simulation.egf_reference.source_time_function
            if simulation.egf_reference is not None
            else None
        ),
        "egf_reference_calibration_operator": (
            simulation.egf_reference.calibration_operator
            if simulation.egf_reference is not None
            else None
        ),
        "egf_reference_status": (
            "available"
            if simulation.egf_reference is not None
            else "DAS-DAS reference not yet implemented"
        ),
        "das": (
            {
                "operator_version": "wavedispx-finite-gauge-v1",
                "measurement": config.das.measurement,
                "time_convention": (
                    "numpy_rfft_forward_exp(-iwt); derivative=+iomega"
                ),
                **_das_geometry_manifest(simulation.transfer.das_geometry),
            }
            if config.das.enabled
            else None
        ),
        "source_selection": (
            {
                "scope": simulation.source_selection.scope,
                "method_version": simulation.source_selection.method_version,
                "correlated_source_risk": (
                    simulation.source_selection.correlated_source_risk
                ),
            }
            if simulation.source_selection is not None
            else None
        ),
        "nnws_pairwise": (
            {
                "scope": simulation.pairwise_nnws.scope,
                "pair_scope": config.nnws.pair_scope,
                "reference_receiver_index": config.nnws.reference_receiver_index,
                "std_mode": config.nnws.std_mode,
                "pair_count": int(
                    simulation.pairwise_nnws.pair_stack.branches.distance_m.size
                ),
                "eligible_pair_count": int(
                    np.count_nonzero(simulation.pairwise_nnws.eligible)
                ),
                "ineligible_pair_count": int(
                    np.count_nonzero(~simulation.pairwise_nnws.eligible)
                ),
                "pair_coverage": (
                    float(
                        np.count_nonzero(
                            simulation.pairwise_nnws.eligible
                            & simulation.pairwise_nnws.pair_stack.statistics.sufficient
                        )
                        / np.count_nonzero(simulation.pairwise_nnws.eligible)
                    )
                    if np.any(simulation.pairwise_nnws.eligible)
                    else 0.0
                ),
                "failed_pair_count": int(
                    np.count_nonzero(
                        simulation.pairwise_nnws.eligible
                        & ~simulation.pairwise_nnws.pair_stack.statistics.sufficient
                    )
                ),
            }
            if simulation.pairwise_nnws is not None
            else None
        ),
        "nnws_v2": (
            {
                "scope": simulation.nnws_v2.scope,
                "pair_scope": config.nnws_v2.pair_scope,
                "reference_receiver_index": (
                    config.nnws_v2.reference_receiver_index
                ),
                "folds": config.nnws_v2.folds,
                "effective_window_count": (
                    simulation.nnws_v2.fit.effective_window_count
                ),
                "minimum_effective_fraction": (
                    config.nnws_v2.minimum_effective_fraction
                ),
                "sufficient": simulation.nnws_v2.fit.sufficient,
                "proxy_distance_m": simulation.nnws_v2.fit.proxy_distance_m,
                "window_audit_csv": (
                    nnws_v2_window_audit_path.name
                    if nnws_v2_window_audit_path is not None
                    else None
                ),
            }
            if simulation.nnws_v2 is not None
            else None
        ),
        "pair_spz": (
            {
                "scope": simulation.pair_spz_selection.scope,
                "method_version": simulation.pair_spz_selection.method_version,
                "pair_scope": config.pair_spz.pair_scope,
                "reference_receiver_index": (
                    config.pair_spz.reference_receiver_index
                ),
                "pair_count": simulation.pair_spz_selection.pair_count,
                "effective_window_count": (
                    simulation.pair_spz_selection.effective_window_count
                ),
                "optimized_spz_fraction": (
                    simulation.pair_spz_selection.optimized_spz_fraction.tolist()
                ),
                "optimized_bilateral_balance": (
                    simulation.pair_spz_selection.optimized_bilateral_balance.tolist()
                ),
                "band_effective_window_count": (
                    simulation.pair_spz_selection.band_effective_window_count.tolist()
                ),
                "optimization_success": (
                    simulation.pair_spz_selection.optimization_success
                ),
                "window_audit_csv": (
                    pair_spz_window_audit_path.name
                    if pair_spz_window_audit_path is not None
                    else None
                ),
                "pair_weight_audit_csv": (
                    pair_spz_pair_weight_audit_path.name
                    if pair_spz_pair_weight_audit_path is not None
                    else None
                ),
                "minimum_pair_branch_effective_window_count": float(
                    np.min(
                        simulation.pair_spz_selection.
                        pair_branch_effective_window_count
                    )
                ),
                "minimum_pair_branch_band_effective_window_count": float(
                    np.min(
                        simulation.pair_spz_selection.
                        pair_branch_band_effective_window_count
                    )
                ),
            }
            if simulation.pair_spz_selection is not None
            else None
        ),
        "supervised_egf_oracle": (
            {
                "scope": simulation.supervised_egf_oracle.scope,
                "positive_effective_window_count": (
                    simulation.supervised_egf_oracle.positive_effective_window_count
                ),
                "negative_effective_window_count": (
                    simulation.supervised_egf_oracle.negative_effective_window_count
                ),
                "window_audit_csv": (
                    supervised_oracle_window_audit_path.name
                    if supervised_oracle_window_audit_path is not None
                    else None
                ),
                "deployable": False,
            }
            if simulation.supervised_egf_oracle is not None
            else None
        ),
        "supervised_pair_egf_oracle": (
            {
                "scope": simulation.supervised_pair_egf_oracle.scope,
                "minimum_positive_effective_window_count": float(
                    np.min(
                        simulation.supervised_pair_egf_oracle.
                        positive_effective_window_count
                    )
                ),
                "minimum_negative_effective_window_count": float(
                    np.min(
                        simulation.supervised_pair_egf_oracle.
                        negative_effective_window_count
                    )
                ),
                "pair_count": int(
                    simulation.supervised_pair_egf_oracle.pair_i.size
                ),
                "failed_positive_pair_count": int(
                    np.count_nonzero(
                        ~simulation.supervised_pair_egf_oracle.
                        positive_fit_success
                    )
                ),
                "failed_negative_pair_count": int(
                    np.count_nonzero(
                        ~simulation.supervised_pair_egf_oracle.
                        negative_fit_success
                    )
                ),
                "weight_audit_csv": (
                    supervised_pair_oracle_audit_path.name
                    if supervised_pair_oracle_audit_path is not None
                    else None
                ),
                "deployable": False,
            }
            if simulation.supervised_pair_egf_oracle is not None
            else None
        ),
        "pair_crossfit_coherence": (
            {
                "scope": simulation.pair_crossfit_coherence.scope,
                "pair_count": int(
                    simulation.pair_crossfit_coherence.pair_i.size
                ),
                "minimum_positive_effective_window_count": float(
                    np.min(
                        simulation.pair_crossfit_coherence.
                        positive_effective_window_count
                    )
                ),
                "minimum_negative_effective_window_count": float(
                    np.min(
                        simulation.pair_crossfit_coherence.
                        negative_effective_window_count
                    )
                ),
                "window_pair_audit_csv": (
                    pair_crossfit_coherence_audit_path.name
                    if pair_crossfit_coherence_audit_path is not None
                    else None
                ),
                "uses_theoretical_egf": False,
                "uses_true_source_coordinates": False,
            }
            if simulation.pair_crossfit_coherence is not None
            else None
        ),
        "pair_crossfit_tau_p_spz": (
            {
                "scope": simulation.pair_tau_p_selection.scope,
                "pair_count": int(simulation.pair_tau_p_selection.pair_i.size),
                "band_count": int(
                    simulation.pair_tau_p_selection.band_center_hz.size
                ),
                "minimum_positive_effective_window_count": float(
                    np.min(
                        simulation.pair_tau_p_selection.
                        positive_effective_window_count
                    )
                ),
                "minimum_negative_effective_window_count": float(
                    np.min(
                        simulation.pair_tau_p_selection.
                        negative_effective_window_count
                    )
                ),
                "minimum_diffuse_soft_effective_window_count": float(
                    np.min(
                        simulation.pair_tau_p_selection.
                        diffuse_soft_effective_window_count
                    )
                ),
                "minimum_diffuse_hard_effective_window_count": float(
                    np.min(
                        simulation.pair_tau_p_selection.
                        diffuse_hard_effective_window_count
                    )
                ),
                "window_pair_band_audit_csv": (
                    pair_tau_p_audit_path.name
                    if pair_tau_p_audit_path is not None
                    else None
                ),
                "uses_theoretical_egf": False,
                "uses_true_source_coordinates": False,
                "validation_status": "failed_complete_egf_gate",
            }
            if simulation.pair_tau_p_selection is not None
            else None
        ),
        "ncf_strategies": {
            name: {
                "windows_seen": stack.statistics.windows_seen,
                "nonzero_window_count": stack.statistics.nonzero_window_count,
                "sum_weight": stack.statistics.sum_weight,
                "sum_weight_squared": stack.statistics.sum_weight_squared,
                "effective_window_count": stack.statistics.effective_window_count,
                "sufficient": stack.statistics.sufficient,
            }
            for name, stack in simulation.strategy_stacks.items()
        },
        "strategy_egf_quality": simulation.strategy_egf_quality,
        "strategy_dispersion_quality": simulation.strategy_dispersion_quality,
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
    _write_portable_input(config, output)
    return RunArtifacts(output, results_path, summary_png, manifest_path)
