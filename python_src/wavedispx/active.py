"""WaveDispX active-source shot gathers."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .config import (
    ActiveSourceConfig,
    DASConfig,
    ModelConfig,
    ProcessingConfig,
    ReceiverFileConfig,
    ReceiverGridConfig,
    ReceiverLineConfig,
    RunConfig,
)
from .das import (
    DASRecord,
    apply_finite_gauge_operator,
    build_configured_das_geometry,
)
from .forward import compute_full_fk_green, synthesize_full_receivers
from .geometry import receiver_positions
from .processing import (
    DispersionImage,
    DispersionPicks,
    TheoreticalDispersion,
    combine_dispersion_images,
    pick_dispersion_ridge,
    surface_wave_dispersion,
    theoretical_dispersion,
)
from .quality import (
    dispersion_quality_metrics,
    pick_error_metrics,
    theoretical_mode_support_metrics,
)
from .survey_design import recommended_active_record_duration


@dataclass(frozen=True)
class ActiveShotGather:
    time_s: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    distance_m: np.ndarray
    components: np.ndarray
    das_record: DASRecord | None = None
    vector_xyz: np.ndarray | None = None
    force_xyz: np.ndarray | None = None

    @property
    def horizontal_radial(self) -> np.ndarray:
        return self.components[:, :, 0]

    @property
    def horizontal_tangential(self) -> np.ndarray:
        return self.components[:, :, 1]

    @property
    def horizontal_vertical(self) -> np.ndarray:
        return self.components[:, :, 2]

    @property
    def vertical_radial(self) -> np.ndarray:
        return self.components[:, :, 3]

    @property
    def vertical_vertical(self) -> np.ndarray:
        return self.components[:, :, 4]


@dataclass(frozen=True)
class ActiveAnalysis:
    rayleigh: DispersionImage
    rayleigh_combined: DispersionImage
    love: DispersionImage
    rayleigh_picks: DispersionPicks
    rayleigh_combined_picks: DispersionPicks
    love_picks: DispersionPicks
    theory: TheoreticalDispersion
    quality: dict[str, dict[str, float | bool]]
    component_images: dict[str, DispersionImage]
    method_comparison: dict[str, DispersionImage]
    imaging_seconds: float


def ricker_wavelet(time_s: np.ndarray, dominant_frequency_hz: float) -> np.ndarray:
    time = np.asarray(time_s, dtype=np.float64)
    peak_time = 2.0 / dominant_frequency_hz
    argument = np.pi * dominant_frequency_hz * (time - peak_time)
    return (1.0 - 2.0 * argument**2) * np.exp(-argument**2)


def active_source_wavelet(
    time_s: np.ndarray,
    wavelet: str,
    dominant_frequency_hz: float,
) -> np.ndarray:
    """Return the configured discrete source time function.

    ``unit_impulse`` is an ideal synthetic calibration source. Its discrete
    Fourier amplitude is flat, so it can test the model, receiver sampling,
    and dispersion imager without a Ricker spectrum hiding either band edge.
    """
    time = np.asarray(time_s, dtype=np.float64)
    if wavelet == "ricker":
        return ricker_wavelet(time, dominant_frequency_hz)
    if wavelet == "unit_impulse":
        result = np.zeros_like(time)
        if result.size:
            result[0] = 1.0
        return result
    raise ValueError(f"unsupported active-source wavelet: {wavelet}")


def _linear_convolve_wavelet(
    wavefield: np.ndarray,
    dt_s: float,
    wavelet_name: str,
    dominant_frequency_hz: float,
) -> np.ndarray:
    if wavelet_name == "unit_impulse":
        return wavefield.copy()
    sample_count = wavefield.shape[0]
    nfft = 1 << (2 * sample_count - 2).bit_length()
    time = np.arange(sample_count, dtype=np.float64) * dt_s
    wavelet = active_source_wavelet(
        time, wavelet_name, dominant_frequency_hz
    )
    wavelet_spectrum = np.fft.fft(wavelet, nfft).reshape(-1, 1, 1)
    result = np.fft.ifft(
        np.fft.fft(wavefield, nfft, axis=0) * wavelet_spectrum, axis=0
    ).real
    return result[:sample_count]


def simulate_active_shot(
    model: ModelConfig,
    source: ActiveSourceConfig,
    receivers: ReceiverLineConfig | ReceiverGridConfig | ReceiverFileConfig,
    *,
    das: DASConfig | None = None,
    processing: ProcessingConfig | None = None,
    workers: int = 0,
    distance_block_size: int = 128,
) -> ActiveShotGather:
    receiver_x, receiver_y = receiver_positions(receivers)
    distance = np.hypot(receiver_x - source.x_m, receiver_y - source.y_m)
    if np.any(distance == 0):
        raise ValueError("point-load response is undefined at zero source-receiver distance")
    das_geometry = None
    endpoint_distance = np.empty(0, dtype=np.float64)
    if das is not None and das.enabled:
        if source.component != "vertical":
            raise ValueError("active DAS v1 requires a vertical point-force source")
        das_geometry = build_configured_das_geometry(
            receiver_x,
            receiver_y,
            geometry_mode=das.geometry_mode,
            gauge_length_m=das.gauge_length_m,
            fiber_azimuth_deg=das.fiber_azimuth_deg,
            receiver_coordinates_file=getattr(
                receivers, "coordinates_file", None
            ),
        )
        endpoint_xy = das_geometry.endpoint_xy_m
        endpoint_distance = np.hypot(
            endpoint_xy[:, 0] - source.x_m, endpoint_xy[:, 1] - source.y_m
        )
        if np.any(endpoint_distance == 0):
            raise ValueError(
                "point-load response is undefined when an active source lies on a DAS endpoint"
            )
    maximum_distance = float(
        max(distance.max(), endpoint_distance.max(initial=0.0))
    )
    peak_delay = 2.0 / source.dominant_frequency_hz
    green_delay = 2.0 * peak_delay
    if processing is None:
        # Preserve the original low-level API/MATLAB regression contract.
        # Production runners always pass ProcessingConfig and therefore use
        # the model-aware group-arrival design below.
        recommended_duration = (
            2.0
            * maximum_distance
            / min(layer.vs_mps for layer in model.layers)
            / 0.862
            + peak_delay
        )
    else:
        recommended_duration, _, _ = recommended_active_record_duration(
            model,
            source,
            maximum_distance,
            frequency_min_hz=processing.frequency_min_hz,
            frequency_max_hz=processing.frequency_max_hz,
            rayleigh_modes=processing.rayleigh_modes,
            love_modes=(
                processing.love_modes
                if source.component in {"horizontal", "both", "vector"}
                else 0
            ),
        )
    output_duration = (
        recommended_duration
        if source.record_duration_s is None
        else source.record_duration_s
    )
    if output_duration + source.dt_s < recommended_duration:
        raise ValueError(
            "active_source.record_duration_s is shorter than the model-aware "
            f"group-arrival requirement; use at least {recommended_duration:.3f} s "
            "or set it to 0/omit it for automatic design"
        )
    total_duration = output_duration + green_delay
    sample_count = int(np.ceil(total_duration / source.dt_s))
    if sample_count % 2:
        sample_count += 1 if processing is not None else -1
    fk_green = compute_full_fk_green(
        model,
        sample_count=sample_count,
        dt_s=source.dt_s,
        maximum_distance_m=maximum_distance,
        delay_s=green_delay,
        maximum_frequency_hz=source.maximum_frequency_hz,
        workers=workers,
    )
    synthesis_distance = (
        np.concatenate((distance, endpoint_distance))
        if endpoint_distance.size
        else distance
    )
    all_green = synthesize_full_receivers(
        fk_green,
        synthesis_distance,
        sample_count=sample_count,
        dt_s=source.dt_s,
        distance_block_size=distance_block_size,
    )
    receiver_count = distance.size
    fields = np.stack(
        (
            all_green.horizontal_radial[:, :receiver_count],
            all_green.horizontal_tangential[:, :receiver_count],
            all_green.horizontal_vertical[:, :receiver_count],
            all_green.vertical_radial[:, :receiver_count],
            all_green.vertical_vertical[:, :receiver_count],
        ),
        axis=2,
    )
    if source.component == "vertical":
        fields[:, :, :3] = 0.0
    elif source.component == "horizontal":
        fields[:, :, 3:] = 0.0
    convolved = _linear_convolve_wavelet(
        fields,
        source.dt_s,
        source.wavelet,
        source.dominant_frequency_hz,
    )
    delay_samples = int(green_delay / source.dt_s)
    output = convolved[delay_samples:]
    time = np.arange(output.shape[0], dtype=np.float64) * source.dt_s
    das_record = None
    if das_geometry is not None and das is not None:
        endpoint_radial = all_green.vertical_radial[:, receiver_count:]
        endpoint_xy = das_geometry.endpoint_xy_m
        endpoint_azimuth = np.arctan2(
            endpoint_xy[:, 1] - source.y_m,
            endpoint_xy[:, 0] - source.x_m,
        )
        endpoint_displacement = np.stack(
            (
                endpoint_radial * np.cos(endpoint_azimuth)[None, :],
                endpoint_radial * np.sin(endpoint_azimuth)[None, :],
            ),
            axis=2,
        )
        endpoint_convolved = _linear_convolve_wavelet(
            endpoint_displacement,
            source.dt_s,
            source.wavelet,
            source.dominant_frequency_hz,
        )[delay_samples:]
        channel_count = das_geometry.channel_count
        das_record = apply_finite_gauge_operator(
            endpoint_convolved[:, :channel_count],
            endpoint_convolved[:, channel_count:],
            das_geometry,
            dt_s=source.dt_s,
            measurement=das.measurement,
        )
    xyz = force = None
    if source.component == "vector":
        from .source_vector import force_vector, project_green_to_xyz
        force = force_vector(source.force_amplitude, source.force_azimuth_deg, source.force_inclination_deg)
        xyz = project_green_to_xyz(output, np.arctan2(receiver_y-source.y_m, receiver_x-source.x_m), force)
    return ActiveShotGather(time, receiver_x, receiver_y, distance, output, das_record, xyz, force)


def analyze_active_shot(config: RunConfig, shot: ActiveShotGather) -> ActiveAnalysis:
    """Image and pick one active shot without saving artifacts."""
    if config.mode != "active" or config.active_source is None:
        raise ValueError("analyze_active_shot requires an active configuration")
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
    common = dict(
        frequency_min_hz=processing.frequency_min_hz,
        frequency_max_hz=processing.frequency_max_hz,
        frequency_step_hz=processing.frequency_step_hz,
        zero_padding=processing.zero_padding,
        distance_taper_fraction=processing.distance_taper_fraction,
        frequency_block_size=config.frequency_block_size,
        radial_kernel_lookup_step=processing.radial_kernel_lookup_step,
        workers=config.workers,
    )
    component_data = {
        "rr": (shot.horizontal_radial, "radial"),
        "tt": (shot.horizontal_tangential, "transverse"),
        "zr": (shot.horizontal_vertical, "vertical"),
        "rz": (shot.vertical_radial, "radial"),
        "zz": (shot.vertical_vertical, "vertical"),
    }
    if shot.vector_xyz is not None:
        from .source_vector import project_green_to_rtz
        az = np.arctan2(shot.receiver_y_m-config.active_source.y_m, shot.receiver_x_m-config.active_source.x_m)
        r,t,z = np.moveaxis(project_green_to_rtz(shot.components,az,shot.force_xyz),-1,0)
        component_data = {"rr": (r, "radial"), "tt": (t, "transverse"), "zz": (z, "vertical")}
    if config.das.enabled:
        if shot.das_record is None:
            raise ValueError("DAS is enabled but the active shot has no DAS record")
        component_data["das"] = (shot.das_record.data, "vertical")

    def image_component(name: str, method: str) -> DispersionImage:
        data, component = component_data[name]
        return surface_wave_dispersion(
            data,
            shot.distance_m,
            config.active_source.dt_s,
            velocity,
            component=component,
            method=method,
            **common,
        )

    component_images = {
        name: image_component(name, processing.dispersion_method)
        for name in component_data
    }
    rayleigh = component_images["das"] if config.das.enabled else component_images["zz"]
    love = component_images["tt"]
    if shot.vector_xyz is not None:
        rayleigh_names = ("rr", "zz")
    elif config.das.enabled:
        rayleigh_names = ("das",)
    elif config.active_source.component == "vertical":
        rayleigh_names = ("zz", "rz")
    elif config.active_source.component == "horizontal":
        rayleigh_names = ("rr", "zr")
    else:
        rayleigh_names = ("rr", "zr", "rz", "zz")
    rayleigh_combined = combine_dispersion_images(
        [component_images[name] for name in rayleigh_names]
    )
    method_comparison = {processing.dispersion_method: rayleigh_combined}
    if processing.compare_methods:
        for method in ("phase_shift", "bessel", "hankel"):
            if method == processing.dispersion_method:
                continue
            method_comparison[method] = combine_dispersion_images(
                [image_component(name, method) for name in rayleigh_names]
            )
    pick_options = dict(
        frequency_range_hz=(
            processing.frequency_min_hz,
            processing.frequency_max_hz,
        ),
        velocity_range_mps=(
            processing.velocity_min_mps,
            processing.velocity_max_mps,
        ),
        smoothness=processing.picker_smoothness,
        maximum_jump_mps=processing.picker_maximum_jump_mps,
        minimum_energy=processing.picker_minimum_energy,
    )
    rayleigh_picks = pick_dispersion_ridge(
        rayleigh.energy, rayleigh.frequency_hz, velocity, **pick_options
    )
    rayleigh_combined_picks = pick_dispersion_ridge(
        rayleigh_combined.energy,
        rayleigh_combined.frequency_hz,
        velocity,
        **pick_options,
    )
    love_picks = pick_dispersion_ridge(
        love.energy, love.frequency_hz, velocity, **pick_options
    )
    theory = theoretical_dispersion(
        config.model,
        rayleigh.frequency_hz,
        rayleigh_modes=processing.rayleigh_modes,
        love_modes=processing.love_modes,
        backend=processing.theory_solver,
    )
    quality = {
        "rayleigh_image": dispersion_quality_metrics(
            rayleigh,
            theory.frequency_hz,
            theory.rayleigh_mps,
            frequency_range_hz=(
                processing.frequency_min_hz,
                processing.frequency_max_hz,
            ),
        ),
        "rayleigh_picks": pick_error_metrics(
            rayleigh_picks, theory.frequency_hz, theory.rayleigh_mps[:, 0]
        ),
        "rayleigh_combined_image": dispersion_quality_metrics(
            rayleigh_combined,
            theory.frequency_hz,
            theory.rayleigh_mps,
            frequency_range_hz=(
                processing.frequency_min_hz,
                processing.frequency_max_hz,
            ),
        ),
        "rayleigh_combined_picks": pick_error_metrics(
            rayleigh_combined_picks,
            theory.frequency_hz,
            theory.rayleigh_mps[:, 0],
        ),
        "love_image": dispersion_quality_metrics(
            love,
            theory.frequency_hz,
            theory.love_mps,
            frequency_range_hz=(
                processing.frequency_min_hz,
                processing.frequency_max_hz,
            ),
        ),
        "love_picks": pick_error_metrics(
            love_picks, theory.frequency_hz, theory.love_mps[:, 0]
        ),
    }
    quality["rayleigh_combined_mode_support"] = theoretical_mode_support_metrics(
        rayleigh_combined, theory.frequency_hz, theory.rayleigh_mps
    )
    for method, image in method_comparison.items():
        quality[f"rayleigh_{method}_mode_support"] = theoretical_mode_support_metrics(
            image, theory.frequency_hz, theory.rayleigh_mps
        )
    return ActiveAnalysis(
        rayleigh=rayleigh,
        rayleigh_combined=rayleigh_combined,
        love=love,
        rayleigh_picks=rayleigh_picks,
        rayleigh_combined_picks=rayleigh_combined_picks,
        love_picks=love_picks,
        theory=theory,
        quality=quality,
        component_images=component_images,
        method_comparison=method_comparison,
        imaging_seconds=time.perf_counter() - start,
    )
