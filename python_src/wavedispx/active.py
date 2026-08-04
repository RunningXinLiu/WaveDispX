"""WaveDispX active-source shot gathers."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .config import (
    ActiveSourceConfig,
    ModelConfig,
    ReceiverFileConfig,
    ReceiverGridConfig,
    ReceiverLineConfig,
    RunConfig,
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


@dataclass(frozen=True)
class ActiveShotGather:
    time_s: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    distance_m: np.ndarray
    components: np.ndarray

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


def _linear_convolve_wavelet(
    wavefield: np.ndarray, dt_s: float, dominant_frequency_hz: float
) -> np.ndarray:
    sample_count = wavefield.shape[0]
    nfft = 1 << (2 * sample_count - 2).bit_length()
    time = np.arange(sample_count, dtype=np.float64) * dt_s
    wavelet = ricker_wavelet(time, dominant_frequency_hz)
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
    workers: int = 0,
    distance_block_size: int = 128,
) -> ActiveShotGather:
    receiver_x, receiver_y = receiver_positions(receivers)
    distance = np.hypot(receiver_x - source.x_m, receiver_y - source.y_m)
    if np.any(distance == 0):
        raise ValueError("point-load response is undefined at zero source-receiver distance")
    peak_delay = 2.0 / source.dominant_frequency_hz
    green_delay = 2.0 * peak_delay
    total_duration = (
        2.0 * distance.max() / min(layer.vs_mps for layer in model.layers) / 0.862
        + peak_delay
        + green_delay
    )
    sample_count = int(np.ceil(total_duration / source.dt_s))
    if sample_count % 2:
        sample_count -= 1
    fk_green = compute_full_fk_green(
        model,
        sample_count=sample_count,
        dt_s=source.dt_s,
        maximum_distance_m=float(distance.max()),
        delay_s=green_delay,
        maximum_frequency_hz=source.maximum_frequency_hz,
        workers=workers,
    )
    receiver_green = synthesize_full_receivers(
        fk_green,
        distance,
        sample_count=sample_count,
        dt_s=source.dt_s,
        distance_block_size=distance_block_size,
    )
    fields = np.stack(
        (
            receiver_green.horizontal_radial,
            receiver_green.horizontal_tangential,
            receiver_green.horizontal_vertical,
            receiver_green.vertical_radial,
            receiver_green.vertical_vertical,
        ),
        axis=2,
    )
    if source.component == "vertical":
        fields[:, :, :3] = 0.0
    elif source.component == "horizontal":
        fields[:, :, 3:] = 0.0
    convolved = _linear_convolve_wavelet(
        fields, source.dt_s, source.dominant_frequency_hz
    )
    delay_samples = int(green_delay / source.dt_s)
    output = convolved[delay_samples:]
    time = np.arange(output.shape[0], dtype=np.float64) * source.dt_s
    return ActiveShotGather(time, receiver_x, receiver_y, distance, output)


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
    rayleigh = component_images["zz"]
    love = component_images["tt"]
    if config.active_source.component == "vertical":
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
