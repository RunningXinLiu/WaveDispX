"""Staged streaming finite-record ambient-noise modelling and processing."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .config import ReceiverLineConfig, RunConfig
from .forward import (
    compute_full_fk_green,
    compute_vertical_fk_green,
    synthesize_full_receivers,
    synthesize_vertical_receivers,
)
from .geometry import Geometry, build_geometry
from .noise import (
    source_spectral_amplitude,
    synthesize_mixed_noise_window_3c_band,
    synthesize_mixed_noise_window_band,
)
from .processing import (
    DispersionImage,
    DispersionPicks,
    NCFGather,
    TheoreticalDispersion,
    combine_dispersion_images,
    linear_array_ncf,
    pick_dispersion_ridge,
    receiver_array_ncf,
    receiver_array_3c_ncf,
    surface_wave_dispersion,
    theoretical_dispersion,
)
from .quality import (
    dispersion_quality_metrics,
    pick_error_metrics,
    theoretical_mode_support_metrics,
)


@dataclass(frozen=True)
class PassiveTransfer:
    frequency_hz: np.ndarray
    spectrum: np.ndarray
    response_sample_count: int
    delay_samples: int
    radial_grid_count: int
    radial_step_m: float
    interpolation_relative_l2: float
    component_names: tuple[str, ...]
    build_seconds: float

    @property
    def bytes(self) -> int:
        return self.spectrum.nbytes


@dataclass(frozen=True)
class PassiveCorrelation:
    preview_record: np.ndarray
    preview_time_s: np.ndarray
    ncf: NCFGather
    component_ncfs: dict[str, NCFGather]
    checkpoint_gathers: dict[int, np.ndarray]
    checkpoint_component_gathers: dict[str, dict[int, np.ndarray]]
    checkpoint_durations_s: dict[int, float]
    noise_event_catalog: np.ndarray | None
    independent_window_count: int
    correlation_seconds: float


@dataclass(frozen=True)
class PassiveAnalysis:
    dispersion: DispersionImage
    picks: DispersionPicks
    theory: TheoreticalDispersion
    quality: dict[str, dict[str, float | bool]]
    component_images: dict[str, DispersionImage]
    method_comparison: dict[str, DispersionImage]
    convergence: list[dict[str, object]]
    imaging_seconds: float


@dataclass(frozen=True)
class PassiveSimulation:
    geometry: Geometry
    transfer: PassiveTransfer
    preview_record: np.ndarray
    preview_time_s: np.ndarray
    ncf: NCFGather
    component_ncfs: dict[str, NCFGather]
    dispersion: DispersionImage
    picks: DispersionPicks
    theory: TheoreticalDispersion
    quality: dict[str, dict[str, float | bool]]
    component_images: dict[str, DispersionImage]
    method_comparison: dict[str, DispersionImage]
    convergence: list[dict[str, object]]
    noise_event_catalog: np.ndarray | None
    independent_window_count: int
    timings: dict[str, float]


def build_passive_transfer(config: RunConfig, geometry: Geometry) -> PassiveTransfer:
    """Compute the vertical or full 3-C source-receiver transfer tensor."""
    if config.mode != "passive" or config.noise is None:
        raise ValueError("build_passive_transfer requires a passive configuration")
    noise = config.noise
    start = time.perf_counter()
    distance = np.hypot(
        geometry.receiver_x_m[:, None] - geometry.source_x_m[None, :],
        geometry.receiver_y_m[:, None] - geometry.source_y_m[None, :],
    )
    if np.any(distance <= 0):
        raise ValueError("ambient sources may not coincide with receivers")
    maximum_distance = float(np.ceil(distance.max()))
    reference_frequency = max(
        noise.minimum_frequency_hz, 0.25 * noise.dominant_frequency_hz
    )
    delay_s = 4.0 / reference_frequency
    response_duration = (
        2.0
        * maximum_distance
        / min(layer.vs_mps for layer in config.model.layers)
        / 0.862
        + 2.0 / reference_frequency
        + delay_s
    )
    response_sample_count = int(np.floor(response_duration / noise.dt_s)) + 1
    if response_sample_count % 2:
        response_sample_count += 1
    delay_samples = int(delay_s / noise.dt_s)
    three_component = config.ambient_sources.component == "three_component"
    green_options = dict(
        sample_count=response_sample_count,
        dt_s=noise.dt_s,
        maximum_distance_m=maximum_distance,
        delay_s=delay_s,
        maximum_frequency_hz=noise.maximum_frequency_hz,
        workers=config.workers,
    )
    fk_green = (
        compute_full_fk_green(config.model, **green_options)
        if three_component
        else compute_vertical_fk_green(config.model, **green_options)
    )
    window_samples = int(round(noise.window_s / noise.dt_s))
    full_frequency = np.fft.rfftfreq(window_samples, noise.dt_s)
    amplitude = source_spectral_amplitude(full_frequency, noise)
    frequency_index = np.flatnonzero(amplitude > 0)
    transfer_frequency = full_frequency[frequency_index]
    receiver_count = geometry.receiver_x_m.size
    source_count = geometry.source_x_m.size
    transfer_shape = (transfer_frequency.size, receiver_count, source_count)
    if three_component:
        transfer_shape += (5,)
    transfer = np.empty(transfer_shape, dtype=np.complex64)
    radial_step = config.radial_interpolation_step_m
    radial_start = max(
        radial_step, np.floor(distance.min() / radial_step) * radial_step
    )
    radial_stop = np.ceil(distance.max() / radial_step) * radial_step
    radial_count = int(round((radial_stop - radial_start) / radial_step)) + 1
    radial_grid = radial_start + radial_step * np.arange(radial_count)
    fastest_velocity = 1.5 * max(layer.vp_mps for layer in config.model.layers)
    radial_shape = (transfer_frequency.size, radial_count)
    if three_component:
        radial_shape += (5,)
    radial_spectrum = np.empty(radial_shape, dtype=np.complex128)
    radial_block_size = max(32, config.frequency_block_size * 8)
    for first_radius in range(0, radial_count, radial_block_size):
        last_radius = min(radial_count, first_radius + radial_block_size)
        radial_block = radial_grid[first_radius:last_radius]
        if three_component:
            response = synthesize_full_receivers(
                fk_green,
                radial_block,
                sample_count=response_sample_count,
                dt_s=noise.dt_s,
                distance_block_size=radial_block_size,
            )
            impulse = np.stack(
                (
                    response.horizontal_radial,
                    response.horizontal_tangential,
                    response.horizontal_vertical,
                    response.vertical_radial,
                    response.vertical_vertical,
                ),
                axis=2,
            )[delay_samples:]
        else:
            response = synthesize_vertical_receivers(
                fk_green,
                radial_block,
                sample_count=response_sample_count,
                dt_s=noise.dt_s,
                distance_block_size=radial_block_size,
                include_radial=False,
            )
            impulse = response.vertical[delay_samples:]
        if impulse.shape[0] > window_samples:
            raise ValueError(
                "noise.window_s is shorter than the farthest-source Green response; "
                f"use at least {impulse.shape[0] * noise.dt_s:.3f} s"
            )
        for column, offset in enumerate(radial_block):
            mute_samples = int(offset / fastest_velocity / noise.dt_s)
            impulse[:mute_samples, column] = 0.0
        radial_spectrum[:, first_radius:last_radius] = np.fft.rfft(
            impulse, n=window_samples, axis=0
        )[frequency_index]
    flattened_distance = distance.reshape(-1)
    for frequency_position in range(transfer_frequency.size):
        if three_component:
            for component_index in range(5):
                values = radial_spectrum[frequency_position, :, component_index]
                interpolated = np.interp(
                    flattened_distance, radial_grid, values.real
                ) + 1j * np.interp(flattened_distance, radial_grid, values.imag)
                transfer[frequency_position, :, :, component_index] = interpolated.reshape(
                    receiver_count, source_count
                )
        else:
            values = radial_spectrum[frequency_position]
            interpolated = np.interp(
                flattened_distance, radial_grid, values.real
            ) + 1j * np.interp(flattened_distance, radial_grid, values.imag)
            transfer[frequency_position] = interpolated.reshape(
                receiver_count, source_count
            )
    check_count = min(16, flattened_distance.size)
    check_index = np.linspace(
        0, flattened_distance.size - 1, check_count, dtype=int
    )
    check_distance = flattened_distance[check_index]
    if three_component:
        check_response = synthesize_full_receivers(
            fk_green,
            check_distance,
            sample_count=response_sample_count,
            dt_s=noise.dt_s,
            distance_block_size=check_count,
        )
        check_impulse = np.stack(
            (
                check_response.horizontal_radial,
                check_response.horizontal_tangential,
                check_response.horizontal_vertical,
                check_response.vertical_radial,
                check_response.vertical_vertical,
            ),
            axis=2,
        )[delay_samples:]
    else:
        check_response = synthesize_vertical_receivers(
            fk_green,
            check_distance,
            sample_count=response_sample_count,
            dt_s=noise.dt_s,
            distance_block_size=check_count,
            include_radial=False,
        )
        check_impulse = check_response.vertical[delay_samples:]
    for column, offset in enumerate(check_distance):
        mute_samples = int(offset / fastest_velocity / noise.dt_s)
        check_impulse[:mute_samples, column] = 0.0
    exact_check = np.fft.rfft(check_impulse, n=window_samples, axis=0)[
        frequency_index
    ]
    interpolated_check = (
        transfer.reshape(transfer_frequency.size, -1, 5)[:, check_index]
        if three_component
        else transfer.reshape(transfer_frequency.size, -1)[:, check_index]
    )
    interpolation_relative_l2 = float(
        np.linalg.norm(interpolated_check - exact_check)
        / max(np.linalg.norm(exact_check), np.finfo(float).eps)
    )
    return PassiveTransfer(
        transfer_frequency,
        transfer,
        response_sample_count,
        delay_samples,
        radial_count,
        radial_step,
        interpolation_relative_l2,
        ("rr", "tt", "zr", "rz", "zz") if three_component else ("zz",),
        time.perf_counter() - start,
    )


def accumulate_passive_correlation(
    config: RunConfig,
    geometry: Geometry,
    transfer: PassiveTransfer,
) -> PassiveCorrelation:
    """Generate finite mixed records and stream receiver-only NCF stacking."""
    if config.mode != "passive" or config.noise is None:
        raise ValueError("accumulate_passive_correlation requires a passive case")
    noise = config.noise
    duration_ratio = noise.duration_s / noise.window_s
    window_count = int(round(duration_ratio))
    if not np.isclose(duration_ratio, window_count, rtol=0, atol=1e-10):
        raise ValueError("noise.duration_s must be an integer multiple of noise.window_s")
    aperture = np.hypot(
        geometry.receiver_x_m[:, None] - geometry.receiver_x_m[None, :],
        geometry.receiver_y_m[:, None] - geometry.receiver_y_m[None, :],
    ).max()
    maximum_lag_s = 1.3 * aperture / (
        0.90 * min(layer.vs_mps for layer in config.model.layers)
    )
    if maximum_lag_s >= noise.window_s:
        raise ValueError(
            f"noise.window_s must exceed the required {maximum_lag_s:.3f} s maximum lag"
        )
    rng = np.random.default_rng(noise.seed)
    three_component = transfer.spectrum.ndim == 4
    source_receiver_azimuth = np.arctan2(
        geometry.receiver_y_m[:, None] - geometry.source_y_m[None, :],
        geometry.receiver_x_m[:, None] - geometry.source_x_m[None, :],
    )
    cumulative_gathers: dict[str, np.ndarray] = {}
    ncf_templates: dict[str, NCFGather] = {}
    preview_windows = min(
        window_count, max(1, int(round(60.0 / noise.window_s)))
    )
    preview = []
    event_catalog_rows: list[np.ndarray] = []
    checkpoint_durations = sorted(
        {
            duration
            for duration in (60.0, 600.0, noise.duration_s)
            if duration <= noise.duration_s
            and np.isclose(
                duration / noise.window_s, round(duration / noise.window_s)
            )
        }
    )
    checkpoint_windows = {
        int(round(duration / noise.window_s)): duration
        for duration in checkpoint_durations
    }
    component_snapshots: dict[str, dict[int, np.ndarray]] = {}
    correlation_start = time.perf_counter()
    for window_index in range(window_count):
        if three_component:
            realization = synthesize_mixed_noise_window_3c_band(
                transfer.spectrum,
                transfer.frequency_hz,
                source_receiver_azimuth,
                noise,
                rng,
                return_source_events=noise.temporal_model == "poisson_ricker",
                source_block_size=config.source_block_size,
            )
        else:
            realization = synthesize_mixed_noise_window_band(
                transfer.spectrum,
                transfer.frequency_hz,
                noise,
                rng,
                return_source_events=noise.temporal_model == "poisson_ricker",
                source_block_size=config.source_block_size,
            )
        mixed = realization.record
        if realization.source_events is not None and realization.source_events.size:
            source_index = realization.source_events["source_index"]
            source_x = np.full(source_index.size, np.nan, dtype=np.float64)
            source_y = np.full(source_index.size, np.nan, dtype=np.float64)
            independent = source_index >= 0
            source_x[independent] = geometry.source_x_m[source_index[independent]]
            source_y[independent] = geometry.source_y_m[source_index[independent]]
            time_in_window = realization.source_events["time_in_window_s"]
            columns = [
                np.full(source_index.size, window_index, dtype=np.float64),
                source_index.astype(np.float64),
            ]
            if "component_index" in realization.source_events.dtype.names:
                columns.append(
                    realization.source_events["component_index"].astype(np.float64)
                )
            columns.extend(
                (
                    source_x,
                    source_y,
                    time_in_window,
                    window_index * noise.window_s + time_in_window,
                    realization.source_events["weight"],
                )
            )
            event_catalog_rows.append(np.column_stack(columns))
        if window_index < preview_windows:
            preview.append(mixed.astype(np.float32))
        correlation_options = dict(
            window_s=noise.window_s,
            overlap=0.0,
            maximum_lag_s=maximum_lag_s,
            normalization=config.processing.correlation_normalization,
            branch="symmetric",
        )
        if three_component:
            distance_bin = (
                config.receivers.spacing_m
                if isinstance(config.receivers, ReceiverLineConfig)
                else config.receivers.distance_bin_m
            )
            window_ncfs = receiver_array_3c_ncf(
                mixed,
                geometry.receiver_x_m,
                geometry.receiver_y_m,
                noise.dt_s,
                distance_bin_m=distance_bin,
                **correlation_options,
            )
        elif not isinstance(config.receivers, ReceiverLineConfig):
            window_ncfs = {
                "zz": receiver_array_ncf(
                    mixed,
                    geometry.receiver_x_m,
                    geometry.receiver_y_m,
                    noise.dt_s,
                    distance_bin_m=config.receivers.distance_bin_m,
                    **correlation_options,
                )
            }
        else:
            window_ncfs = {
                "zz": linear_array_ncf(
                    mixed, geometry.receiver_x_m, noise.dt_s, **correlation_options
                )
            }
        for name, window_ncf in window_ncfs.items():
            if name not in cumulative_gathers:
                cumulative_gathers[name] = np.zeros_like(window_ncf.data)
                ncf_templates[name] = window_ncf
                component_snapshots[name] = {}
            cumulative_gathers[name] += window_ncf.data
        completed_windows = window_index + 1
        if completed_windows in checkpoint_windows:
            for name, cumulative in cumulative_gathers.items():
                component_snapshots[name][completed_windows] = (
                    cumulative / completed_windows
                ).copy()
    component_ncfs = {
        name: NCFGather(
            cumulative / window_count,
            ncf_templates[name].lag_time_s,
            ncf_templates[name].distance_m,
            ncf_templates[name].pair_count,
            window_count,
        )
        for name, cumulative in cumulative_gathers.items()
    }
    ncf = component_ncfs["zz"]
    preview_record = np.concatenate(preview, axis=0)
    preview_time = np.arange(preview_record.shape[0]) * noise.dt_s
    event_catalog = (
        np.concatenate(event_catalog_rows, axis=0)
        if event_catalog_rows
        else (
            np.empty((0, 8 if three_component else 7), dtype=np.float64)
            if noise.temporal_model == "poisson_ricker"
            else None
        )
    )
    return PassiveCorrelation(
        preview_record,
        preview_time,
        ncf=ncf,
        component_ncfs=component_ncfs,
        checkpoint_gathers=component_snapshots["zz"],
        checkpoint_component_gathers=component_snapshots,
        checkpoint_durations_s=checkpoint_windows,
        noise_event_catalog=event_catalog,
        independent_window_count=window_count,
        correlation_seconds=time.perf_counter() - correlation_start,
    )


def analyze_passive_correlation(
    config: RunConfig, correlation: PassiveCorrelation
) -> PassiveAnalysis:
    """Image, pick, compare with theory, and quantify duration convergence."""
    if config.mode != "passive" or config.noise is None:
        raise ValueError("analyze_passive_correlation requires a passive case")
    start = time.perf_counter()
    processing = config.processing
    velocity_count = int(
        round(
            (processing.velocity_max_mps - processing.velocity_min_mps)
            / processing.velocity_step_mps
        )
    )
    velocity = processing.velocity_min_mps + processing.velocity_step_mps * np.arange(
        velocity_count + 1
    )

    imaging_options = dict(
        frequency_min_hz=processing.frequency_min_hz,
        frequency_max_hz=processing.frequency_max_hz,
        zero_padding=processing.zero_padding,
        distance_taper_fraction=processing.distance_taper_fraction,
        frequency_block_size=config.frequency_block_size,
        radial_kernel_lookup_step=processing.radial_kernel_lookup_step,
        workers=config.workers,
    )
    pick_options = dict(
        frequency_range_hz=(processing.frequency_min_hz, processing.frequency_max_hz),
        velocity_range_mps=(
            processing.velocity_min_mps,
            processing.velocity_max_mps,
        ),
        smoothness=processing.picker_smoothness,
        maximum_jump_mps=processing.picker_maximum_jump_mps,
        minimum_energy=processing.picker_minimum_energy,
    )

    def image_gather(
        gather: np.ndarray, component: str, method: str
    ) -> DispersionImage:
        return surface_wave_dispersion(
            gather,
            correlation.ncf.distance_m,
            config.noise.dt_s,
            velocity,
            method=method,
            component=component,
            **imaging_options,
        )

    def analyze_gathers(gathers: dict[str, np.ndarray], method: str):
        component_images = {
            "zz": image_gather(gathers["zz"], "vertical", method)
        }
        if "rr" in gathers:
            component_images["rr"] = image_gather(gathers["rr"], "radial", method)
        if "tt" in gathers:
            component_images["tt"] = image_gather(
                gathers["tt"], "transverse", method
            )
        image = (
            combine_dispersion_images(
                [component_images["zz"], component_images["rr"]]
            )
            if "rr" in component_images
            else component_images["zz"]
        )
        image_picks = pick_dispersion_ridge(
            image.energy, image.frequency_hz, velocity, **pick_options
        )
        image_theory = theoretical_dispersion(
            config.model,
            image.frequency_hz,
            rayleigh_modes=processing.rayleigh_modes,
            love_modes=processing.love_modes,
        )
        image_quality = {
            "image": dispersion_quality_metrics(
                image,
                image_theory.frequency_hz,
                image_theory.rayleigh_mps,
                frequency_range_hz=(
                    processing.frequency_min_hz,
                    processing.frequency_max_hz,
                ),
            ),
            "picks": pick_error_metrics(
                image_picks,
                image_theory.frequency_hz,
                image_theory.rayleigh_mps[:, 0],
            ),
        }
        for name, component_image in component_images.items():
            branch = "love_mps" if name == "tt" else "rayleigh_mps"
            image_quality[f"{name}_image"] = dispersion_quality_metrics(
                component_image,
                image_theory.frequency_hz,
                getattr(image_theory, branch),
                frequency_range_hz=(
                    processing.frequency_min_hz,
                    processing.frequency_max_hz,
                ),
            )
        return image, image_picks, image_theory, image_quality, component_images

    analyses = {
        count: analyze_gathers(
            {
                name: snapshots[count]
                for name, snapshots in correlation.checkpoint_component_gathers.items()
            },
            processing.dispersion_method,
        )
        for count in sorted(correlation.checkpoint_gathers)
    }
    final_count = correlation.independent_window_count
    dispersion, picks, theory, quality, component_images = analyses[final_count]
    method_comparison = {processing.dispersion_method: dispersion}
    if processing.compare_methods:
        final_gathers = {
            name: gather.data for name, gather in correlation.component_ncfs.items()
        }
        for method in ("phase_shift", "bessel", "hankel"):
            if method == processing.dispersion_method:
                continue
            method_comparison[method] = analyze_gathers(final_gathers, method)[0]
    quality["rayleigh_mode_support"] = theoretical_mode_support_metrics(
        dispersion, theory.frequency_hz, theory.rayleigh_mps
    )
    for method, image in method_comparison.items():
        quality[f"rayleigh_{method}_mode_support"] = theoretical_mode_support_metrics(
            image, theory.frequency_hz, theory.rayleigh_mps
        )
    convergence: list[dict[str, object]] = []
    final_ncf_norm = max(
        np.sqrt(
            sum(
                np.linalg.norm(gather.data) ** 2
                for gather in correlation.component_ncfs.values()
            )
        ),
        np.finfo(float).eps,
    )
    final_image_norm = max(np.linalg.norm(dispersion.energy), np.finfo(float).eps)
    for count, (image, image_picks, _, image_quality, _) in analyses.items():
        ncf_difference = np.sqrt(
            sum(
                np.linalg.norm(
                    snapshots[count] - correlation.component_ncfs[name].data
                )
                ** 2
                for name, snapshots in correlation.checkpoint_component_gathers.items()
            )
        )
        convergence.append(
            {
                "duration_s": correlation.checkpoint_durations_s[count],
                "window_count": count,
                "ncf_relative_l2_to_final": float(
                    ncf_difference / final_ncf_norm
                ),
                "dispersion_relative_l2_to_final": float(
                    np.linalg.norm(image.energy - dispersion.energy) / final_image_norm
                ),
                "pick_mae_to_final_mps": float(
                    np.mean(
                        np.abs(
                            image_picks.phase_velocity_mps
                            - picks.phase_velocity_mps
                        )
                    )
                ),
                "quality": image_quality,
            }
        )
    return PassiveAnalysis(
        dispersion=dispersion,
        picks=picks,
        theory=theory,
        quality=quality,
        component_images=component_images,
        method_comparison=method_comparison,
        convergence=convergence,
        imaging_seconds=time.perf_counter() - start,
    )


def simulate_passive(config: RunConfig) -> PassiveSimulation:
    """Compatibility one-call API composed from the public notebook stages."""
    if config.mode != "passive" or config.ambient_sources is None:
        raise ValueError("simulate_passive requires a passive configuration")
    overall_start = time.perf_counter()
    geometry = build_geometry(config.receivers, config.ambient_sources)
    transfer = build_passive_transfer(config, geometry)
    correlation = accumulate_passive_correlation(config, geometry, transfer)
    analysis = analyze_passive_correlation(config, correlation)
    timings = {
        "transfer": transfer.build_seconds,
        "correlation": correlation.correlation_seconds,
        "imaging": analysis.imaging_seconds,
        "total": time.perf_counter() - overall_start,
    }
    return PassiveSimulation(
        geometry=geometry,
        transfer=transfer,
        preview_record=correlation.preview_record,
        preview_time_s=correlation.preview_time_s,
        ncf=correlation.ncf,
        component_ncfs=correlation.component_ncfs,
        dispersion=analysis.dispersion,
        picks=analysis.picks,
        theory=analysis.theory,
        quality=analysis.quality,
        component_images=analysis.component_images,
        method_comparison=analysis.method_comparison,
        convergence=analysis.convergence,
        noise_event_catalog=correlation.noise_event_catalog,
        independent_window_count=correlation.independent_window_count,
        timings=timings,
    )
