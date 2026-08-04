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
import numpy as np

from .active import ricker_wavelet
from .config import ReceiverLineConfig, RunConfig
from .geometry import Geometry, build_geometry, receiver_positions, save_geometry_csv
from .noise import source_spectral_amplitude, stochastic_source_coefficients
from .processing import TheoreticalDispersion, theoretical_dispersion


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
        waveform = ricker_wavelet(time_s, source.dominant_frequency_hz)
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
    theory_frequency = np.linspace(
        max(processing.frequency_min_hz, 0.05),
        processing.frequency_max_hz,
        240,
    )
    theory = theoretical_dispersion(
        config.model,
        theory_frequency,
        rayleigh_modes=processing.rayleigh_modes,
        love_modes=processing.love_modes,
    )
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
    all_theory = theory.rayleigh_mps
    if config.mode == "active" and config.active_source.component in {"horizontal", "both"}:
        all_theory = np.column_stack((all_theory, theory.love_mps))
    finite_velocity = all_theory[np.isfinite(all_theory)]
    minimum_phase_velocity = float(
        finite_velocity.min(initial=min(layer.vs_mps for layer in config.model.layers))
    )
    fundamental_at_fmin = theory.rayleigh_mps[0, 0]
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
    }
    if config.mode == "passive":
        minimum_vs = min(layer.vs_mps for layer in config.model.layers)
        required_lag_s = 1.3 * receiver_aperture / (0.9 * minimum_vs)
        maximum_gap = _maximum_azimuth_gap_deg(geometry)
        radial_step_limit = minimum_vs / (20.0 * config.noise.maximum_frequency_hz)
        checks.extend(
            (
                PreflightCheck(
                    "Correlation window",
                    f"{config.noise.window_s:.2f} s",
                    f"> required lag {required_lag_s:.2f} s",
                    config.noise.window_s > required_lag_s,
                    "required",
                    "noise.window_s",
                    f"把独立窗长增大到大于 {required_lag_s:.2f} s，并保证不超过总记录时长。",
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
            )
        )
        diagnostics.update(
            ambient_source_component=config.ambient_sources.component,
            required_maximum_lag_s=required_lag_s,
            maximum_source_azimuth_gap_deg=maximum_gap,
            radial_interpolation_limit_m=radial_step_limit,
            independent_window_count=int(round(config.noise.duration_s / config.noise.window_s)),
        )
    return PreflightReview(
        config,
        geometry,
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
    axes[1, 2].set(
        xlabel="Frequency (Hz)",
        ylabel="Phase velocity (m/s)",
        title="Theory within requested processing window",
    )
    axes[1, 2].set_ylim(
        config.processing.velocity_min_mps,
        config.processing.velocity_max_mps,
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
