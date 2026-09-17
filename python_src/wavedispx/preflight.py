"""Fast, no-forward preflight review for active/passive configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import matplotlib

if "IPython" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .active import active_source_wavelet
from .config import ReceiverLineConfig, RunConfig
from .das import DASGeometry, build_configured_das_geometry
from .geometry import Geometry, build_geometry, receiver_positions, save_geometry_csv
from .noise import source_spectral_amplitude, stochastic_source_coefficients
from .processing import TheoreticalDispersion, theoretical_dispersion
from .survey_design import design_survey_window


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    value: str
    criterion: str
    passed: bool
    level: str = "recommended"
    parameter_path: str = ""
    guidance: str = ""


@dataclass(frozen=True)
class PreflightReview:
    config: RunConfig
    geometry: Geometry
    das_geometry: DASGeometry | None
    theory: TheoreticalDispersion
    source_time_s: np.ndarray
    source_waveform: np.ndarray
    spectrum_frequency_hz: np.ndarray
    source_power: np.ndarray
    checks: tuple[PreflightCheck, ...]
    diagnostics: dict[str, float | int | str]

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks if check.level == "required")


def _geometry(config: RunConfig) -> Geometry:
    if config.mode == "passive":
        return build_geometry(config.receivers, config.ambient_sources)
    receiver_x, receiver_y = receiver_positions(config.receivers)
    return Geometry(
        receiver_x,
        receiver_y,
        np.array([config.active_source.x_m], dtype=np.float64),
        np.array([config.active_source.y_m], dtype=np.float64),
    )


def _maximum_azimuth_gap_deg(geometry: Geometry) -> float:
    center_x = float(np.mean(geometry.receiver_x_m))
    center_y = float(np.mean(geometry.receiver_y_m))
    angle = np.mod(
        np.arctan2(
            geometry.source_y_m - center_y,
            geometry.source_x_m - center_x,
        ),
        2.0 * np.pi,
    )
    if angle.size < 2:
        return 360.0
    angle = np.sort(angle)
    gaps = np.diff(np.concatenate((angle, angle[:1] + 2.0 * np.pi)))
    return float(np.degrees(gaps.max()))


def _source_preview(config: RunConfig) -> tuple[np.ndarray, ...]:
    if config.mode == "active":
        source = config.active_source
        duration = 6.0 / source.dominant_frequency_hz
        time_s = np.arange(int(np.ceil(duration / source.dt_s)) + 1) * source.dt_s
        waveform = active_source_wavelet(
            time_s, source.wavelet, source.dominant_frequency_hz
        )
        frequency = np.fft.rfftfreq(time_s.size, source.dt_s)
        power = np.abs(np.fft.rfft(waveform)) ** 2
    else:
        noise = config.noise
        sample_count = int(round(noise.window_s / noise.dt_s))
        time_s = np.arange(sample_count) * noise.dt_s
        frequency = np.fft.rfftfreq(sample_count, noise.dt_s)
        amplitude = source_spectral_amplitude(frequency, noise)
        power = amplitude**2
        coefficients = stochastic_source_coefficients(
            frequency,
            1,
            noise,
            np.random.default_rng(noise.seed),
        )[0]
        coefficients[0] = coefficients[0].real
        if sample_count % 2 == 0:
            coefficients[-1] = coefficients[-1].real
        waveform = np.fft.irfft(coefficients, n=sample_count)
    peak = float(np.max(power, initial=0.0))
    if peak > 0:
        power = power / peak
    waveform_peak = float(np.max(np.abs(waveform), initial=0.0))
    if waveform_peak > 0:
        waveform = waveform / waveform_peak
    return time_s, waveform, frequency, power


def build_preflight_review(config: RunConfig) -> PreflightReview:
    """Build all preflight inputs without running the wavefield solver."""
    geometry = _geometry(config)
    processing = config.processing
    time_s, waveform, spectrum_frequency, source_power = _source_preview(config)
    receiver_pair_distance = np.hypot(
        geometry.receiver_x_m[:, None] - geometry.receiver_x_m[None, :],
        geometry.receiver_y_m[:, None] - geometry.receiver_y_m[None, :],
    )
    receiver_aperture = float(receiver_pair_distance.max())
    source_receiver_distance = np.hypot(
        geometry.receiver_x_m[:, None] - geometry.source_x_m[None, :],
        geometry.receiver_y_m[:, None] - geometry.source_y_m[None, :],
    )
    if isinstance(config.receivers, ReceiverLineConfig):
        effective_spacing = config.receivers.spacing_m
        spatial_sampling_basis = "receiver line spacing"
    elif config.mode == "passive":
        # Passive 2-D/file geometries are converted to a radial NCF gather on
        # the explicitly requested pair-distance grid.  That radial bin width,
        # rather than the Cartesian nearest-neighbour station spacing, controls
        # the spatial sampling seen by phase-shift/FJ imaging.
        effective_spacing = float(config.receivers.distance_bin_m)
        spatial_sampling_basis = "radial pair-distance bin"
    else:
        # Active 2-D/file layouts are imaged against their actual radial
        # source-receiver offsets.  Use the median unique radial-offset gap as
        # a robust irregular-sampling estimate; Cartesian station spacing can
        # be much larger while the radial offsets remain densely sampled.
        radial_offset = np.unique(np.round(source_receiver_distance.ravel(), 9))
        radial_gap = np.diff(radial_offset)
        positive_gap = radial_gap[radial_gap > 0]
        if positive_gap.size:
            effective_spacing = float(np.median(positive_gap))
        else:
            nearest = receiver_pair_distance.copy()
            np.fill_diagonal(nearest, np.inf)
            effective_spacing = float(np.max(np.min(nearest, axis=1)))
        spatial_sampling_basis = "median unique source-receiver radial gap"
    das_geometry = None
    maximum_gauge = None
    maximum_design_distance = float(source_receiver_distance.max())
    if config.das.enabled:
        das_geometry = build_configured_das_geometry(
            geometry.receiver_x_m,
            geometry.receiver_y_m,
            geometry_mode=config.das.geometry_mode,
            gauge_length_m=config.das.gauge_length_m,
            fiber_azimuth_deg=config.das.fiber_azimuth_deg,
            receiver_coordinates_file=getattr(
                config.receivers, "coordinates_file", None
            ),
        )
        endpoint_xy = das_geometry.endpoint_xy_m
        endpoint_source_distance = np.hypot(
            endpoint_xy[:, 0, None] - geometry.source_x_m[None, :],
            endpoint_xy[:, 1, None] - geometry.source_y_m[None, :],
        )
        maximum_design_distance = max(
            maximum_design_distance, float(endpoint_source_distance.max())
        )
        maximum_gauge = float(
            np.max(das_geometry.gauge_length_by_channel_m)
        )
    design = design_survey_window(
        config,
        receiver_aperture_m=receiver_aperture,
        effective_spacing_m=effective_spacing,
        maximum_source_receiver_distance_m=maximum_design_distance,
        maximum_gauge_length_m=maximum_gauge,
    )
    theory = (
        design.theory
        if processing.theory_solver == "disba"
        else theoretical_dispersion(
            config.model,
            design.frequency_hz,
            rayleigh_modes=processing.rayleigh_modes,
            love_modes=processing.love_modes,
            backend=processing.theory_solver,
        )
    )
    all_theory = theory.rayleigh_mps
    if config.mode == "active" and config.active_source.component in {"horizontal", "both"}:
        all_theory = np.column_stack((all_theory, theory.love_mps))
    requested_frequency = (
        (theory.frequency_hz >= processing.frequency_min_hz)
        & (theory.frequency_hz <= processing.frequency_max_hz)
    )
    finite_velocity = all_theory[requested_frequency]
    finite_velocity = finite_velocity[np.isfinite(finite_velocity)]
    minimum_phase_velocity = float(
        finite_velocity.min(initial=min(layer.vs_mps for layer in config.model.layers))
    )
    fundamental = theory.rayleigh_mps[:, 0]
    finite_fundamental = np.isfinite(fundamental)
    fundamental_at_fmin = (
        np.interp(
            processing.frequency_min_hz,
            theory.frequency_hz[finite_fundamental],
            fundamental[finite_fundamental],
        )
        if np.any(finite_fundamental)
        else np.nan
    )
    if not np.isfinite(fundamental_at_fmin):
        fundamental_at_fmin = max(layer.vs_mps for layer in config.model.layers)
    wavelength_count = float(
        receiver_aperture * processing.frequency_min_hz / fundamental_at_fmin
    )
    spatial_alias_hz = float(
        minimum_phase_velocity / (2.0 * effective_spacing)
    )
    dt_s = config.active_source.dt_s if config.mode == "active" else config.noise.dt_s
    temporal_nyquist_hz = 0.5 / dt_s
    source_maximum_hz = (
        config.active_source.maximum_frequency_hz
        if config.mode == "active"
        else config.noise.maximum_frequency_hz
    )
    samples_per_shortest_period = float(1.0 / (dt_s * source_maximum_hz))
    frequency_range_covered = (
        processing.frequency_min_hz
        <= design.recommended_frequency_min_hz + 1.0e-9
        and processing.frequency_max_hz
        >= design.recommended_frequency_max_hz - 1.0e-9
    )
    velocity_range_covered = (
        processing.velocity_min_mps
        <= design.recommended_velocity_min_mps + 1.0e-9
        and processing.velocity_max_mps
        >= design.recommended_velocity_max_mps - 1.0e-9
    )
    if config.mode == "active":
        actual_record_duration_s = (
            design.recommended_active_record_duration_s
            if config.active_source.record_duration_s is None
            else config.active_source.record_duration_s
        )
        observation_window_s = actual_record_duration_s
        internal_period_s = actual_record_duration_s + (
            4.0 / config.active_source.dominant_frequency_hz
        )
    else:
        actual_record_duration_s = config.noise.duration_s
        observation_window_s = config.noise.window_s
        reference_frequency = max(
            config.noise.minimum_frequency_hz,
            0.25 * config.noise.dominant_frequency_hz,
        )
        internal_period_s = (
            design.recommended_passive_green_duration_s
            + 4.0 / reference_frequency
        )
    # The output-image resolution is controlled by the saved active gather or
    # one passive correlation window.  The DWM complex-frequency grid has its
    # own (usually different) synthetic period and must not be reported as the
    # observation's physical frequency resolution.
    physical_frequency_step_hz = 1.0 / observation_window_s
    observation_sample_count = max(2, int(np.ceil(observation_window_s / dt_s)))
    if config.mode == "passive":
        # The image FFT sees the retained positive/symmetric NCF, not the
        # entire random window used to estimate it.
        retained_lag = (design.recommended_passive_maximum_lag_s
                        if config.noise.maximum_lag_s is None
                        else config.noise.maximum_lag_s)
        observation_sample_count = int(round(retained_lag / dt_s)) + 1
    padded_sample_count = 1 << max(
        0,
        (int(np.ceil(observation_sample_count * processing.zero_padding)) - 1).bit_length(),
    )
    automatic_display_frequency_step_hz = 1.0 / (padded_sample_count * dt_s)
    display_frequency_step_hz = (
        automatic_display_frequency_step_hz
        if processing.frequency_step_hz is None
        else processing.frequency_step_hz
    )
    dwm_frequency_step_hz = 1.0 / internal_period_s
    maximum_recommended_df_hz = (
        processing.frequency_max_hz - processing.frequency_min_hz
    ) / 20.0
    checks = [
        PreflightCheck(
            "Temporal Nyquist",
            f"{temporal_nyquist_hz:.2f} Hz",
            f"> source cutoff {source_maximum_hz:.2f} Hz",
            temporal_nyquist_hz > source_maximum_hz,
            "required",
            (
                "active_source.dt_s / maximum_frequency_hz"
                if config.mode == "active"
                else "noise.dt_s / maximum_frequency_hz"
            ),
            "减小 dt，或降低源/噪声最高频率，使最高频率严格小于 1/(2dt)。",
        ),
        PreflightCheck(
            "Samples per shortest period",
            f"{samples_per_shortest_period:.1f}",
            ">= 8 recommended",
            samples_per_shortest_period >= 8.0,
            parameter_path=(
                "active_source.dt_s / maximum_frequency_hz"
                if config.mode == "active"
                else "noise.dt_s / maximum_frequency_hz"
            ),
            guidance="建议减小 dt，使最高频率对应的最短周期至少有 8 个采样点。",
        ),
        PreflightCheck(
            "Spatial alias estimate",
            f"{spatial_alias_hz:.2f} Hz",
            f">= processing maximum {processing.frequency_max_hz:.2f} Hz",
            spatial_alias_hz >= processing.frequency_max_hz,
            "required",
            "geometry.receivers spacing / processing.frequency_max_hz",
            "减小接收点间距或径向距离箱，或降低处理最高频率，直到空间混叠上限覆盖处理频带。",
        ),
        PreflightCheck(
            "Aperture at minimum frequency",
            f"{wavelength_count:.2f} wavelengths",
            ">= 2 wavelengths recommended",
            wavelength_count >= 2.0,
            parameter_path="geometry.receivers aperture / processing.frequency_min_hz",
            guidance="建议扩大阵列孔径，或提高处理最低频率，使孔径至少覆盖约 2 个波长。",
        ),
        PreflightCheck(
            "Minimum source-receiver distance",
            f"{source_receiver_distance.min():.2f} m",
            "> 0 m",
            bool(source_receiver_distance.min() > 0),
            "required",
            (
                "active_source.x_m,y_m / geometry.receivers"
                if config.mode == "active"
                else "ambient_sources coordinates / geometry.receivers"
            ),
            "移动与接收器重合的源或炮点；源—接收器最小距离必须大于 0 m。",
        ),
        PreflightCheck(
            "Processing frequency coverage",
            f"{processing.frequency_min_hz:.2f}–{processing.frequency_max_hz:.2f} Hz",
            (
                f"cover model/source suggestion "
                f"{design.recommended_frequency_min_hz:.2f}–"
                f"{design.recommended_frequency_max_hz:.2f} Hz"
            ),
            frequency_range_covered,
            parameter_path="processing.frequency_min_hz / frequency_max_hz",
            guidance=(
                "根据源谱、理论多模态、孔径和空间采样扩大或移动处理频带；"
                "若有意只研究子频带，可保留设置并在结果中声明裁剪。"
            ),
        ),
        PreflightCheck(
            "Velocity scan coverage",
            f"{processing.velocity_min_mps:.1f}–{processing.velocity_max_mps:.1f} m/s",
            (
                f"cover theory + margin suggestion "
                f"{design.recommended_velocity_min_mps:.1f}–"
                f"{design.recommended_velocity_max_mps:.1f} m/s"
            ),
            velocity_range_covered,
            parameter_path="processing.velocity_min_mps / velocity_max_mps",
            guidance=(
                "扩大速度扫描窗，使理论分支上下仍有背景能量边界；"
                "不要只围着基阶理论线裁图。"
            ),
        ),
        PreflightCheck(
            "Physical frequency resolution",
            f"df≈{physical_frequency_step_hz:.4f} Hz",
            f"<= {maximum_recommended_df_hz:.4f} Hz (>=20 bins across band)",
            physical_frequency_step_hz <= maximum_recommended_df_hz,
            parameter_path=(
                "active_source.record_duration_s"
                if config.mode == "active"
                else "noise.window_s"
            ),
            guidance=(
                "增大真实记录/独立窗长以减小物理 df；processing.zero_padding "
                "只能加密显示采样，不能提高真实频率分辨率。"
            ),
        ),
    ]
    diagnostics: dict[str, float | int | str] = {
        "mode": config.mode,
        "receiver_count": geometry.receiver_x_m.size,
        "source_count": geometry.source_x_m.size,
        "receiver_aperture_m": receiver_aperture,
        "receiver_effective_spacing_m": effective_spacing,
        "spatial_sampling_basis": spatial_sampling_basis,
        "minimum_source_receiver_distance_m": float(source_receiver_distance.min()),
        "maximum_source_receiver_distance_m": float(source_receiver_distance.max()),
        "temporal_nyquist_hz": temporal_nyquist_hz,
        "spatial_alias_estimate_hz": spatial_alias_hz,
        "aperture_wavelengths_at_fmin": wavelength_count,
        "forward_solver_domain": "frequency-wavenumber DWM with optional time synthesis",
        "range_design_theory_solver": design.theory.solver,
        "useful_source_frequency_min_hz": design.useful_source_min_hz,
        "useful_source_frequency_max_hz": design.useful_source_max_hz,
        "recommended_frequency_min_hz": design.recommended_frequency_min_hz,
        "recommended_frequency_max_hz": design.recommended_frequency_max_hz,
        "recommended_velocity_min_mps": design.recommended_velocity_min_mps,
        "recommended_velocity_max_mps": design.recommended_velocity_max_mps,
        "minimum_theoretical_group_velocity_mps": design.minimum_group_velocity_mps,
        "maximum_theoretical_group_velocity_mps": design.maximum_group_velocity_mps,
        "physical_frequency_step_hz": physical_frequency_step_hz,
        "display_frequency_step_hz": display_frequency_step_hz,
        "recommended_display_frequency_step_hz": (
            design.recommended_frequency_step_hz
        ),
        "observation_fft_period_s": observation_window_s,
        "dwm_synthetic_period_s": internal_period_s,
        "dwm_frequency_step_hz": dwm_frequency_step_hz,
        "dwm_complex_frequency_shift_radps": 2.0 * np.pi * dwm_frequency_step_hz,
        "record_duration_s": actual_record_duration_s,
        "recommended_velocity_step_mps": design.recommended_velocity_step_mps,
    }
    if config.mode == "active":
        recommended_record = design.recommended_active_record_duration_s
        checks.append(
            PreflightCheck(
                "Active record duration",
                f"{actual_record_duration_s:.3f} s",
                f">= model/group-arrival design {recommended_record:.3f} s",
                actual_record_duration_s + dt_s >= recommended_record,
                "required",
                "active_source.record_duration_s",
                (
                    f"把主动记录时长设为至少 {recommended_record:.3f} s；"
                    "设为 0 或省略时由模型、群速度、最大偏移和最低处理频率自动设计。"
                ),
            )
        )
        diagnostics.update(
            recommended_active_record_duration_s=recommended_record,
            active_record_duration_mode=(
                "automatic"
                if config.active_source.record_duration_s is None
                else "explicit"
            ),
        )
    else:
        actual_maximum_lag_s = (
            design.recommended_passive_maximum_lag_s
            if config.noise.maximum_lag_s is None
            else config.noise.maximum_lag_s
        )
        checks.append(
            PreflightCheck(
                "Passive NCF maximum lag",
                f"{actual_maximum_lag_s:.3f} s",
                (
                    f">= group-arrival suggestion "
                    f"{design.recommended_passive_maximum_lag_s:.3f} s and < window"
                ),
                (
                    actual_maximum_lag_s + dt_s
                    >= design.recommended_passive_maximum_lag_s
                    and actual_maximum_lag_s < config.noise.window_s
                ),
                "required",
                "noise.maximum_lag_s / noise.window_s",
                (
                    "增大 maximum_lag_s 以包含最慢面波到时，同时保持它小于 window_s；"
                    "设为 0 或省略时由模型群速度和阵列孔径自动设计。"
                ),
            )
        )
        diagnostics.update(
            passive_total_record_duration_s=config.noise.duration_s,
            passive_independent_window_s=config.noise.window_s,
            recommended_passive_green_duration_s=(
                design.recommended_passive_green_duration_s
            ),
            recommended_passive_window_s=design.recommended_passive_window_s,
            passive_maximum_lag_s=actual_maximum_lag_s,
            ncf_fft_period_s=observation_sample_count * dt_s,
            ncf_physical_frequency_step_hz=1.0 / (observation_sample_count * dt_s),
            recommended_passive_maximum_lag_s=(
                design.recommended_passive_maximum_lag_s
            ),
            passive_maximum_lag_mode=(
                "automatic" if config.noise.maximum_lag_s is None else "explicit"
            ),
        )
    if config.das.enabled:
        assert das_geometry is not None
        endpoint_xy = das_geometry.endpoint_xy_m
        endpoint_source_distance = np.hypot(
            endpoint_xy[:, 0, None] - geometry.source_x_m[None, :],
            endpoint_xy[:, 1, None] - geometry.source_y_m[None, :],
        )
        maximum_gauge = float(np.max(das_geometry.gauge_length_by_channel_m))
        minimum_gauge = float(np.min(das_geometry.gauge_length_by_channel_m))
        first_zero_min_hz = minimum_phase_velocity / maximum_gauge
        checks.extend(
            (
                PreflightCheck(
                    "DAS endpoint-source distance",
                    f"{endpoint_source_distance.min():.2f} m",
                    "> 0 m",
                    bool(endpoint_source_distance.min() > 0),
                    "required",
                    (
                        "geometry.receivers.coordinates_file / source coordinates"
                        if config.das.geometry_mode == "channel_file"
                        else "das.gauge_length_m / das.fiber_azimuth_deg / source coordinates"
                    ),
                    "移动源、道中心或修改标距/光纤方位，避免点源恰好落在任一标距端点。",
                ),
                PreflightCheck(
                    "DAS first finite-gauge zero",
                    f"from {first_zero_min_hz:.2f} Hz over the velocity scan",
                    "processing band preferably below the first zero",
                    processing.frequency_max_hz < first_zero_min_hz,
                    parameter_path=(
                        "geometry.receivers.coordinates_file / processing.frequency_max_hz"
                        if config.das.geometry_mode == "channel_file"
                        else "das.gauge_length_m / processing.frequency_max_hz"
                    ),
                    guidance=(
                        "标距的 sinc 响应会在 f=c/Lg 附近出现物理零点；若目标频带穿过该区，"
                        "减小 gauge_length_m 或把该零点作为 DAS 灵敏度缺口解释。"
                    ),
                ),
            )
        )
        diagnostics.update(
            das_measurement=config.das.measurement,
            das_geometry_mode=config.das.geometry_mode,
            das_gauge_length_min_m=minimum_gauge,
            das_gauge_length_max_m=maximum_gauge,
            das_fiber_azimuth_min_deg=float(
                np.min(das_geometry.fiber_azimuth_by_channel_deg)
            ),
            das_fiber_azimuth_max_deg=float(
                np.max(das_geometry.fiber_azimuth_by_channel_deg)
            ),
            das_minimum_endpoint_source_distance_m=float(
                endpoint_source_distance.min()
            ),
            das_first_gauge_zero_minimum_hz=first_zero_min_hz,
        )
    if config.mode == "passive":
        minimum_vs = min(layer.vs_mps for layer in config.model.layers)
        required_lag_s = receiver_aperture / design.minimum_group_velocity_mps
        recommended_window_s = design.recommended_passive_window_s
        maximum_gap = _maximum_azimuth_gap_deg(geometry)
        radial_step_limit = minimum_vs / (20.0 * config.noise.maximum_frequency_hz)
        checks.extend(
            (
                PreflightCheck(
                    "Correlation window",
                    f"{config.noise.window_s:.2f} s",
                    (
                        f">= group-arrival/Green design "
                        f"{recommended_window_s:.2f} s"
                    ),
                    config.noise.window_s >= recommended_window_s,
                    "required",
                    "noise.window_s",
                    (
                        f"把独立窗长增大到至少 {recommended_window_s:.2f} s；"
                        "它同时覆盖最远噪声源 Green 尾部和最慢群速度的台间传播。"
                    ),
                ),
                PreflightCheck(
                    "Ambient source count",
                    str(config.ambient_sources.count),
                    ">= 192 recommended for diffuse annular illumination",
                    config.ambient_sources.count >= 192
                    or config.ambient_sources.layout != "annulus",
                    parameter_path="ambient_sources.count",
                    guidance="源环建议增加到至少 192 个源；正式收敛计算通常使用 192–384 个。",
                ),
                PreflightCheck(
                    "Maximum source azimuth gap",
                    f"{maximum_gap:.2f}°",
                    "<= 10° recommended for annular illumination",
                    maximum_gap <= 10.0
                    or config.ambient_sources.layout != "annulus",
                    parameter_path="ambient_sources.count / angular_sampling",
                    guidance="增加噪声源数量并使用 stratified 或 uniform 方位采样，减少照明空缺。",
                ),
                PreflightCheck(
                    "Radial Green interpolation step",
                    f"{config.radial_interpolation_step_m:.3f} m",
                    f"<= {radial_step_limit:.3f} m",
                    config.radial_interpolation_step_m <= radial_step_limit,
                    "required",
                    "run.radial_interpolation_step_m",
                    f"把“径向 Green 插值”减小到不超过 {radial_step_limit:.3f} m。",
                ),
                PreflightCheck(
                    "Independent noise windows",
                    str(int(round(config.noise.duration_s / config.noise.window_s))),
                    ">= 30 for a useful convergence check; 60+ preferred",
                    config.noise.duration_s / config.noise.window_s >= 30.0,
                    parameter_path="noise.duration_s / noise.window_s",
                    guidance=(
                        "增加总记录时长以增加独立窗数；总时长改善统计收敛，"
                        "不能替代单窗对最慢传播和 Green 尾部的覆盖。"
                    ),
                ),
            )
        )
        diagnostics.update(
            ambient_source_component=config.ambient_sources.component,
            required_maximum_lag_s=required_lag_s,
            recommended_passive_green_duration_s=(
                design.recommended_passive_green_duration_s
            ),
            recommended_passive_window_s=recommended_window_s,
            recommended_passive_duration_s=(
                max(60, int(np.ceil(600.0 / max(config.noise.window_s, recommended_window_s))))
                * max(config.noise.window_s, recommended_window_s)
            ),
            passive_duration_recommendation_basis="candidate: >=60 independent windows and >=600 s; verify by doubling",
            maximum_source_azimuth_gap_deg=maximum_gap,
            radial_interpolation_limit_m=radial_step_limit,
            independent_window_count=int(round(config.noise.duration_s / config.noise.window_s)),
        )
    return PreflightReview(
        config,
        geometry,
        das_geometry,
        theory,
        time_s,
        waveform,
        spectrum_frequency,
        source_power,
        tuple(checks),
        diagnostics,
    )


def checks_markdown(review: PreflightReview) -> str:
    lines = [
        "| Status | Check | Value | Criterion | Parameter | How to fix |",
        "|---|---|---:|---|---|---|",
    ]
    for check in review.checks:
        status = "✅ PASS" if check.passed else ("❌ FAIL" if check.level == "required" else "⚠️ WARN")
        lines.append(
            f"| {status} | {check.name} | {check.value} | {check.criterion} | "
            f"{check.parameter_path} | {check.guidance} |"
        )
    return "\n".join(lines)


def _layer_steps(config: RunConfig, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite_depth = sum(
        layer.thickness_m
        for layer in config.model.layers
        if np.isfinite(layer.thickness_m)
    )
    maximum_depth = max(40.0, 4.0 * finite_depth)
    depth = []
    expanded = []
    current = 0.0
    for layer, value in zip(config.model.layers, values):
        bottom = maximum_depth if np.isinf(layer.thickness_m) else current + layer.thickness_m
        depth.extend((current, bottom))
        expanded.extend((value, value))
        current = bottom
    return np.asarray(expanded), np.asarray(depth)


def plot_preflight_review(review: PreflightReview):
    """Return a six-panel, notebook-displayable preflight figure."""
    config = review.config
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    vp = np.array([layer.vp_mps for layer in config.model.layers])
    vs = np.array([layer.vs_mps for layer in config.model.layers])
    density = np.array([layer.density_kgm3 for layer in config.model.layers])
    vp_step, depth = _layer_steps(config, vp)
    vs_step, _ = _layer_steps(config, vs)
    rho_step, rho_depth = _layer_steps(config, density)
    axes[0, 0].plot(vp_step, depth, "r-", linewidth=2, label="$V_P$")
    axes[0, 0].plot(vs_step, depth, "b-", linewidth=2, label="$V_S$")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(xlabel="Velocity (m/s)", ylabel="Depth (m)", title="Velocity model")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(rho_step, rho_depth, "k-", linewidth=2)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(xlabel="Density (kg/m³)", ylabel="Depth (m)", title="Density model")
    axes[0, 1].grid(alpha=0.25)
    geometry = review.geometry
    axes[0, 2].scatter(
        geometry.source_x_m,
        geometry.source_y_m,
        marker="*",
        s=42 if geometry.source_x_m.size > 1 else 180,
        c="red",
        label=f"Sources (n={geometry.source_x_m.size})",
    )
    axes[0, 2].scatter(
        geometry.receiver_x_m,
        geometry.receiver_y_m,
        marker="v",
        s=20,
        label=f"Receivers (n={geometry.receiver_x_m.size})",
    )
    if review.das_geometry is not None:
        das_geometry = review.das_geometry
        segments = np.stack(
            (das_geometry.endpoint_minus_xy_m, das_geometry.endpoint_plus_xy_m),
            axis=1,
        )
        axes[0, 2].add_collection(
            LineCollection(
                segments,
                colors="tab:orange",
                linewidths=1.2,
                alpha=0.8,
                label=f"DAS gauges ({das_geometry.gauge_label})",
            )
        )
        stride = max(1, das_geometry.channel_count // 80)
        endpoint = das_geometry.endpoint_xy_m[::stride]
        axes[0, 2].scatter(
            endpoint[:, 0], endpoint[:, 1], s=5, c="tab:orange", alpha=0.7
        )
    axes[0, 2].set(xlabel="x (m)", ylabel="y (m)", title="Source-receiver geometry")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.25)
    if geometry.source_x_m.size > 1:
        axes[0, 2].set_aspect("equal", adjustable="datalim")
    preview_limit = review.source_time_s[-1]
    if config.mode == "passive":
        preview_limit = min(5.0, preview_limit)
    preview_mask = review.source_time_s <= preview_limit
    axes[1, 0].plot(
        review.source_time_s[preview_mask],
        review.source_waveform[preview_mask],
        "k-",
        linewidth=0.9,
    )
    axes[1, 0].set(
        xlabel="Time (s)",
        ylabel="Normalized amplitude",
        title="Active wavelet" if config.mode == "active" else "Example stochastic source (first 5 s)",
    )
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(review.spectrum_frequency_hz, review.source_power, "k-", linewidth=2)
    axes[1, 1].fill_between(
        review.spectrum_frequency_hz,
        0,
        review.source_power,
        color="tab:blue",
        alpha=0.25,
    )
    axes[1, 1].axvspan(
        config.processing.frequency_min_hz,
        config.processing.frequency_max_hz,
        color="tab:orange",
        alpha=0.18,
        label="Processing band",
    )
    axes[1, 1].set_xlim(
        0,
        min(
            review.spectrum_frequency_hz[-1],
            1.25
            * max(
                config.processing.frequency_max_hz,
                config.active_source.dominant_frequency_hz
                if config.mode == "active"
                else config.noise.maximum_frequency_hz,
            ),
        ),
    )
    axes[1, 1].set(
        xlabel="Frequency (Hz)",
        ylabel="Normalized power",
        title="Wavelet power" if config.mode == "active" else "Expected ambient-source PSD",
    )
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    for mode_index in range(review.theory.rayleigh_mps.shape[1]):
        axes[1, 2].plot(
            review.theory.frequency_hz,
            review.theory.rayleigh_mps[:, mode_index],
            "-",
            color="tab:blue",
            linewidth=1.7 if mode_index == 0 else 0.9,
            alpha=1.0 if mode_index == 0 else 0.65,
            label="Rayleigh" if mode_index == 0 else None,
        )
    if config.mode == "active" and config.active_source.component in {"horizontal", "both"}:
        for mode_index in range(review.theory.love_mps.shape[1]):
            axes[1, 2].plot(
                review.theory.frequency_hz,
                review.theory.love_mps[:, mode_index],
                "--",
                color="tab:red",
                linewidth=1.7 if mode_index == 0 else 0.9,
                alpha=1.0 if mode_index == 0 else 0.65,
                label="Love" if mode_index == 0 else None,
            )
    axes[1, 2].axhspan(
        config.processing.velocity_min_mps,
        config.processing.velocity_max_mps,
        color="tab:green",
        alpha=0.08,
        label="Velocity scan",
    )
    axes[1, 2].axvspan(
        config.processing.frequency_min_hz,
        config.processing.frequency_max_hz,
        color="tab:orange",
        alpha=0.06,
        label="Frequency scan",
    )
    axes[1, 2].axhline(
        config.processing.velocity_min_mps,
        color="tab:green",
        linewidth=0.8,
    )
    axes[1, 2].axhline(
        config.processing.velocity_max_mps,
        color="tab:green",
        linewidth=0.8,
    )
    axes[1, 2].axvline(
        config.processing.frequency_min_hz,
        color="tab:orange",
        linewidth=0.8,
    )
    axes[1, 2].axvline(
        config.processing.frequency_max_hz,
        color="tab:orange",
        linewidth=0.8,
    )
    recommended_fmin = float(
        review.diagnostics["recommended_frequency_min_hz"]
    )
    recommended_fmax = float(
        review.diagnostics["recommended_frequency_max_hz"]
    )
    recommended_vmin = float(
        review.diagnostics["recommended_velocity_min_mps"]
    )
    recommended_vmax = float(
        review.diagnostics["recommended_velocity_max_mps"]
    )
    axes[1, 2].axvline(
        recommended_fmin,
        color="tab:purple",
        linestyle="--",
        linewidth=1.1,
        label="Model/source/array suggestion",
    )
    axes[1, 2].axvline(
        recommended_fmax,
        color="tab:purple",
        linestyle="--",
        linewidth=1.1,
    )
    axes[1, 2].axhline(
        recommended_vmin,
        color="tab:purple",
        linestyle="--",
        linewidth=1.1,
    )
    axes[1, 2].axhline(
        recommended_vmax,
        color="tab:purple",
        linestyle="--",
        linewidth=1.1,
    )
    axes[1, 2].set(
        xlabel="Frequency (Hz)",
        ylabel="Phase velocity (m/s)",
        title="Model-aware theory and requested f-v window",
    )
    finite_theory = review.theory.rayleigh_mps[
        np.isfinite(review.theory.rayleigh_mps)
    ]
    display_velocity_min = config.processing.velocity_min_mps
    display_velocity_max = config.processing.velocity_max_mps
    if finite_theory.size:
        display_velocity_min = min(
            display_velocity_min, 0.9 * float(np.min(finite_theory))
        )
        display_velocity_max = max(
            display_velocity_max, 1.1 * float(np.max(finite_theory))
        )
    display_velocity_min = min(display_velocity_min, 0.95 * recommended_vmin)
    display_velocity_max = max(display_velocity_max, 1.05 * recommended_vmax)
    axes[1, 2].set_ylim(
        display_velocity_min,
        display_velocity_max,
    )
    axes[1, 2].set_xlim(
        float(review.theory.frequency_hz[0]),
        float(review.theory.frequency_hz[-1]),
    )
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.25)
    required_failures = sum(
        not check.passed and check.level == "required" for check in review.checks
    )
    warnings = sum(
        not check.passed and check.level != "required" for check in review.checks
    )
    status = "READY" if review.ready else "BLOCKED"
    figure.suptitle(
        f"WaveDispX preflight | {config.name} | {config.mode} | {status} | "
        f"required failures={required_failures}, warnings={warnings}",
        fontsize=15,
    )
    return figure


def save_preflight_review(
    review: PreflightReview,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save preview figure, checks, coordinates, and source preview arrays."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    figure = plot_preflight_review(review)
    figure_path = output / "preflight.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    receiver_path, source_path = save_geometry_csv(review.geometry, output)
    source_preview_path = output / "source_preview.csv"
    np.savetxt(
        source_preview_path,
        np.column_stack((review.source_time_s, review.source_waveform)),
        delimiter=",",
        header="time_s,normalized_amplitude",
        comments="",
    )
    report_path = output / "preflight_report.json"
    report = {
        "name": review.config.name,
        "mode": review.config.mode,
        "ready": review.ready,
        "diagnostics": review.diagnostics,
        "checks": [asdict(check) for check in review.checks],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {
        "figure": figure_path,
        "report": report_path,
        "receivers": receiver_path,
        "sources": source_path,
        "source_preview": source_preview_path,
    }
