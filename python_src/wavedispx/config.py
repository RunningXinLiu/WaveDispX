"""Typed TOML configuration for active and ambient-noise simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import math
import tomllib
from typing import Any, Literal


class ConfigError(ValueError):
    """Raised when a user-facing configuration value is invalid."""


def _table(data: dict[str, Any], key: str, *, required: bool = True) -> dict[str, Any]:
    value = data.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a TOML table")
    return value


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be numeric") from exc
    lower_ok = number >= 0 if allow_zero else number > 0
    if not math.isfinite(number) or not lower_ok:
        relation = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"{name} must be finite and {relation}")
    return number


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if number != value or number < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return number


@dataclass(frozen=True)
class Layer:
    thickness_m: float
    vp_mps: float
    vs_mps: float
    density_kgm3: float
    qp: float = math.inf
    qs: float = math.inf

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "Layer":
        prefix = f"model.layers[{index}]"
        thickness = float(data.get("thickness_m", math.nan))
        if not (math.isfinite(thickness) and thickness > 0) and not math.isinf(thickness):
            raise ConfigError(f"{prefix}.thickness_m must be > 0 or inf")
        vp = _positive(data.get("vp_mps"), f"{prefix}.vp_mps")
        vs = _positive(data.get("vs_mps"), f"{prefix}.vs_mps")
        density = _positive(data.get("density_kgm3"), f"{prefix}.density_kgm3")
        if vp <= vs:
            raise ConfigError(f"{prefix}.vp_mps must exceed vs_mps")
        qp = float(data.get("qp", math.inf))
        qs = float(data.get("qs", math.inf))
        if (not math.isinf(qp) and qp <= 0) or (not math.isinf(qs) and qs <= 0):
            raise ConfigError(f"{prefix}.qp and qs must be > 0 or inf")
        return cls(thickness, vp, vs, density, qp, qs)

    @property
    def damp_p(self) -> float:
        return 0.0 if math.isinf(self.qp) else 1.0 / (2.0 * self.qp)

    @property
    def damp_s(self) -> float:
        return 0.0 if math.isinf(self.qs) else 1.0 / (2.0 * self.qs)


@dataclass(frozen=True)
class ModelConfig:
    layers: tuple[Layer, ...]

    @classmethod
    def from_root(cls, data: dict[str, Any]) -> "ModelConfig":
        model = _table(data, "model")
        raw_layers = model.get("layers")
        if not isinstance(raw_layers, list) or not raw_layers:
            raise ConfigError("[[model.layers]] must contain at least one layer")
        layers = tuple(Layer.from_dict(item, i) for i, item in enumerate(raw_layers))
        if not math.isinf(layers[-1].thickness_m):
            raise ConfigError("the last model layer must have thickness_m = inf (half-space)")
        if any(math.isinf(layer.thickness_m) for layer in layers[:-1]):
            raise ConfigError("only the final model layer may have infinite thickness")
        return cls(layers)


@dataclass(frozen=True)
class ReceiverFileConfig:
    coordinates_file: Path
    count: int
    distance_bin_m: float | None = None
    layout: Literal["file"] = "file"


@dataclass(frozen=True)
class ReceiverGridConfig:
    start_x_m: float
    stop_x_m: float
    spacing_x_m: float
    start_y_m: float
    stop_y_m: float
    spacing_y_m: float
    distance_bin_m: float | None = None
    layout: Literal["grid"] = "grid"

    @property
    def count_x(self) -> int:
        return round((self.stop_x_m - self.start_x_m) / self.spacing_x_m) + 1

    @property
    def count_y(self) -> int:
        return round((self.stop_y_m - self.start_y_m) / self.spacing_y_m) + 1

    @property
    def count(self) -> int:
        return self.count_x * self.count_y


@dataclass(frozen=True)
class ReceiverLineConfig:
    start_x_m: float
    stop_x_m: float
    spacing_m: float
    y_m: float = 0.0
    layout: Literal["line"] = "line"

    @classmethod
    def from_root(
        cls, data: dict[str, Any], base_dir: Path | None = None
    ) -> "ReceiverLineConfig | ReceiverGridConfig | ReceiverFileConfig":
        geometry = _table(data, "geometry")
        receivers = geometry.get("receivers")
        if not isinstance(receivers, dict):
            raise ConfigError("[geometry.receivers] is required")
        layout = str(receivers.get("layout", "line")).lower()
        if layout == "file":
            raw_path = receivers.get("coordinates_file")
            if not raw_path:
                raise ConfigError(
                    "geometry.receivers.coordinates_file is required for layout='file'"
                )
            path = ((base_dir or Path.cwd()) / str(raw_path)).resolve()
            if not path.is_file():
                raise ConfigError(f"receiver coordinate file does not exist: {path}")
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if not rows or not {"x_m", "y_m"}.issubset(rows[0]):
                raise ConfigError(
                    "receiver coordinate CSV must have x_m,y_m headers and at least one row"
                )
            positions: list[tuple[float, float]] = []
            for row_index, row in enumerate(rows):
                try:
                    x = float(row["x_m"])
                    y = float(row["y_m"])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"receiver coordinate row {row_index + 2} is not numeric"
                    ) from exc
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ConfigError(
                        f"receiver coordinate row {row_index + 2} must be finite"
                    )
                positions.append((x, y))
            if len(rows) < 2:
                raise ConfigError("at least two receivers are required")
            if len(set(positions)) != len(positions):
                raise ConfigError("receiver coordinate CSV contains duplicate x_m,y_m rows")
            raw_bin = receivers.get("distance_bin_m")
            distance_bin = (
                _positive(raw_bin, "geometry.receivers.distance_bin_m")
                if raw_bin is not None
                else None
            )
            return ReceiverFileConfig(path, len(rows), distance_bin)
        if layout in {"grid", "array"}:
            start_x = float(receivers.get("start_x_m", math.nan))
            stop_x = float(receivers.get("stop_x_m", math.nan))
            start_y = float(receivers.get("start_y_m", math.nan))
            stop_y = float(receivers.get("stop_y_m", math.nan))
            spacing_x = _positive(
                receivers.get("spacing_x_m"), "geometry.receivers.spacing_x_m"
            )
            spacing_y = _positive(
                receivers.get("spacing_y_m"), "geometry.receivers.spacing_y_m"
            )
            if not all(
                math.isfinite(value)
                for value in (start_x, stop_x, start_y, stop_y)
            ):
                raise ConfigError("receiver grid coordinates must be finite")
            if stop_x <= start_x or stop_y <= start_y:
                raise ConfigError(
                    "receiver grid requires start_x_m < stop_x_m and "
                    "start_y_m < stop_y_m"
                )
            intervals_x = (stop_x - start_x) / spacing_x
            intervals_y = (stop_y - start_y) / spacing_y
            if not math.isclose(
                intervals_x, round(intervals_x), rel_tol=0, abs_tol=1e-9
            ) or not math.isclose(
                intervals_y, round(intervals_y), rel_tol=0, abs_tol=1e-9
            ):
                raise ConfigError(
                    "receiver grid spans must be integer multiples of their spacing"
                )
            raw_bin = receivers.get("distance_bin_m")
            distance_bin = (
                _positive(raw_bin, "geometry.receivers.distance_bin_m")
                if raw_bin is not None
                else None
            )
            return ReceiverGridConfig(
                start_x,
                stop_x,
                spacing_x,
                start_y,
                stop_y,
                spacing_y,
                distance_bin,
            )
        if layout != "line":
            raise ConfigError(
                "geometry.receivers.layout must be 'line', 'grid', or 'file'"
            )
        start = float(receivers.get("start_x_m", math.nan))
        stop = float(receivers.get("stop_x_m", math.nan))
        spacing = _positive(receivers.get("spacing_m"), "geometry.receivers.spacing_m")
        y = float(receivers.get("y_m", 0.0))
        if not all(math.isfinite(item) for item in (start, stop, y)):
            raise ConfigError("receiver coordinates must be finite")
        if stop <= start:
            raise ConfigError("geometry.receivers.stop_x_m must exceed start_x_m")
        intervals = (stop - start) / spacing
        if not math.isclose(intervals, round(intervals), rel_tol=0, abs_tol=1e-9):
            raise ConfigError("receiver span must be an integer multiple of spacing_m")
        return cls(start, stop, spacing, y)

    @property
    def count(self) -> int:
        return round((self.stop_x_m - self.start_x_m) / self.spacing_m) + 1


@dataclass(frozen=True)
class ActiveSourceConfig:
    x_m: float
    y_m: float
    component: Literal["vertical", "horizontal", "both", "vector"]
    wavelet: Literal["ricker", "unit_impulse"]
    dominant_frequency_hz: float
    maximum_frequency_hz: float
    dt_s: float
    record_duration_s: float | None = None
    force_amplitude: float = 1.0
    force_azimuth_deg: float = 0.0
    force_inclination_deg: float = 90.0

    @classmethod
    def from_root(cls, data: dict[str, Any]) -> "ActiveSourceConfig":
        source = _table(data, "active_source")
        component = str(source.get("component", "vertical")).lower()
        if component not in {"vertical", "horizontal", "both", "vector"}:
            raise ConfigError("active_source.component must be vertical, horizontal, both, or vector")
        force_amplitude = _positive(source.get("force_amplitude", 1.0), "active_source.force_amplitude")
        azimuth = float(source.get("force_azimuth_deg", 0.0))
        inclination = float(source.get("force_inclination_deg", 90.0))
        if not math.isfinite(azimuth) or not math.isfinite(inclination) or not -90 <= inclination <= 90:
            raise ConfigError("force_azimuth_deg must be finite; force_inclination_deg must be within [-90,90]")
        if component != "vector" and (force_amplitude != 1.0 or azimuth != 0.0 or inclination != 90.0):
            raise ConfigError("non-default force direction/amplitude requires active_source.component='vector'")
        wavelet = str(source.get("wavelet", "ricker")).lower()
        if wavelet not in {"ricker", "unit_impulse"}:
            raise ConfigError(
                "active_source.wavelet must be 'ricker' or 'unit_impulse'"
            )
        x = float(source.get("x_m", 0.0))
        y = float(source.get("y_m", 0.0))
        if not math.isfinite(x) or not math.isfinite(y):
            raise ConfigError("active source coordinates must be finite")
        dominant = _positive(source.get("dominant_frequency_hz"), "active_source.dominant_frequency_hz")
        maximum = _positive(source.get("maximum_frequency_hz"), "active_source.maximum_frequency_hz")
        dt = _positive(source.get("dt_s"), "active_source.dt_s")
        raw_record_duration = source.get("record_duration_s", 0.0)
        record_duration = _positive(
            raw_record_duration,
            "active_source.record_duration_s",
            allow_zero=True,
        )
        if record_duration == 0.0:
            record_duration = None
        if maximum < dominant:
            raise ConfigError("active_source.maximum_frequency_hz must be >= dominant_frequency_hz")
        if maximum >= 0.5 / dt:
            raise ConfigError("active_source.maximum_frequency_hz must be below the temporal Nyquist")
        return cls(
            x,
            y,
            component,
            wavelet,
            dominant,
            maximum,
            dt,
            record_duration,
            force_amplitude,
            azimuth,
            inclination,
        )


@dataclass(frozen=True)
class AmbientSourceConfig:
    layout: Literal["annulus", "road", "line", "file"]
    count: int
    seed: int
    angular_sampling: Literal["random", "stratified", "uniform"]
    component: Literal["vertical", "three_component"]
    inner_radius_m: float | None = None
    outer_radius_m: float | None = None
    x_min_m: float | None = None
    x_max_m: float | None = None
    y_m: float | None = None
    coordinates_file: Path | None = None
    activity_file: Path | None = None
    polarization: str = "legacy_xyz"
    horizontal_power_fraction: float = 2.0 / 3.0
    horizontal_azimuth_deg: float = 0.0

    @classmethod
    def from_root(cls, data: dict[str, Any], base_dir: Path) -> "AmbientSourceConfig":
        source = _table(data, "ambient_sources")
        layout = str(source.get("layout", "annulus")).lower()
        if layout not in {"annulus", "road", "line", "file"}:
            raise ConfigError("ambient_sources.layout must be annulus, road, line, or file")
        count = _integer(source.get("count"), "ambient_sources.count")
        seed = _integer(source.get("seed", 42), "ambient_sources.seed", minimum=0)
        angular = str(source.get("angular_sampling", "stratified")).lower()
        if angular not in {"random", "stratified", "uniform"}:
            raise ConfigError("ambient_sources.angular_sampling must be random, stratified, or uniform")
        component = str(source.get("component", "vertical")).lower()
        aliases = {"3c": "three_component", "three-component": "three_component"}
        component = aliases.get(component, component)
        if component not in {"vertical", "three_component"}:
            raise ConfigError(
                "ambient_sources.component must be 'vertical' or 'three_component'"
            )

        kwargs: dict[str, Any] = {}
        polarization = str(source.get("polarization", "legacy_xyz"))
        if polarization not in {"legacy_xyz", "independent_hv", "directed_hv"}:
            raise ConfigError("ambient_sources.polarization must be legacy_xyz, independent_hv, or directed_hv")
        fraction = float(source.get("horizontal_power_fraction", 2.0 / 3.0))
        azimuth = float(source.get("horizontal_azimuth_deg", 0.0))
        if not math.isfinite(fraction) or not 0 <= fraction <= 1 or not math.isfinite(azimuth):
            raise ConfigError("horizontal_power_fraction must be in [0,1]; horizontal_azimuth_deg must be finite")
        if polarization != "legacy_xyz" and component != "three_component":
            raise ConfigError("source polarization weights require ambient_sources.component='three_component' (XYZ receivers)")
        if polarization == "legacy_xyz" and (fraction != 2.0 / 3.0 or azimuth != 0):
            raise ConfigError("legacy_xyz does not apply weights/direction; choose independent_hv or directed_hv")
        if polarization == "independent_hv" and azimuth != 0:
            raise ConfigError("independent_hv has no preferred azimuth; choose directed_hv")
        if polarization != "legacy_xyz" and not bool(data.get("noise", {}).get("independent_sources", True)):
            raise ConfigError("weighted source polarization v1 requires noise.independent_sources=true")
        if polarization != "legacy_xyz" and data.get("noise", {}).get("temporal_model", "stationary_gaussian") != "stationary_gaussian":
            raise ConfigError("weighted polarization v1 supports stationary_gaussian only; event-force catalogs are not yet implemented")
        kwargs.update(polarization=polarization, horizontal_power_fraction=fraction, horizontal_azimuth_deg=azimuth)
        if layout == "annulus":
            inner = _positive(source.get("inner_radius_m"), "ambient_sources.inner_radius_m", allow_zero=True)
            outer = _positive(source.get("outer_radius_m"), "ambient_sources.outer_radius_m")
            if outer <= inner:
                raise ConfigError("ambient_sources.outer_radius_m must exceed inner_radius_m")
            kwargs.update(inner_radius_m=inner, outer_radius_m=outer)
        elif layout in {"road", "line"}:
            x_min = float(source.get("x_min_m", math.nan))
            x_max = float(source.get("x_max_m", math.nan))
            y = float(source.get("y_m", math.nan))
            if not all(math.isfinite(item) for item in (x_min, x_max, y)) or x_max <= x_min:
                raise ConfigError("road/line sources require finite x_min_m < x_max_m and y_m")
            kwargs.update(x_min_m=x_min, x_max_m=x_max, y_m=y)
        else:
            raw_path = source.get("coordinates_file")
            if not raw_path:
                raise ConfigError("file source layout requires ambient_sources.coordinates_file")
            path = (base_dir / str(raw_path)).resolve()
            if not path.is_file():
                raise ConfigError(f"source coordinate file does not exist: {path}")
            kwargs["coordinates_file"] = path
        raw_activity_path = source.get("activity_file")
        if raw_activity_path:
            activity_path = (base_dir / str(raw_activity_path)).resolve()
            if not activity_path.is_file():
                raise ConfigError(
                    f"source activity file does not exist: {activity_path}"
                )
            kwargs["activity_file"] = activity_path
        return cls(layout, count, seed, angular, component, **kwargs)


@dataclass(frozen=True)
class NoiseConfig:
    temporal_model: Literal["stationary_gaussian", "poisson_ricker"]
    seed: int
    duration_s: float
    dt_s: float
    window_s: float
    overlap: float
    psd_shape: Literal["ricker", "flat_band", "file"]
    dominant_frequency_hz: float
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    source_amplitude: float
    independent_sources: bool
    event_rate_hz: float
    psd_file: Path | None = None
    receiver_noise_rms_fraction: float = 0.0
    receiver_noise_seed: int = 271828
    maximum_lag_s: float | None = None

    @classmethod
    def from_root(cls, data: dict[str, Any], base_dir: Path) -> "NoiseConfig":
        noise = _table(data, "noise")
        temporal = str(noise.get("temporal_model", "stationary_gaussian")).lower()
        if temporal not in {"stationary_gaussian", "poisson_ricker"}:
            raise ConfigError("noise.temporal_model must be stationary_gaussian or poisson_ricker")
        seed = _integer(noise.get("seed", 314159), "noise.seed", minimum=0)
        duration = _positive(noise.get("duration_s"), "noise.duration_s")
        dt = _positive(noise.get("dt_s"), "noise.dt_s")
        window = _positive(noise.get("window_s"), "noise.window_s")
        overlap = float(noise.get("overlap", 0.0))
        if overlap != 0.0:
            raise ConfigError(
                "noise.overlap must currently be 0: synthetic source windows are "
                "independent and are stacked by the streaming correlator"
            )
        if window > duration:
            raise ConfigError("noise.window_s must not exceed duration_s")
        psd_shape = str(noise.get("psd_shape", "ricker")).lower()
        if psd_shape not in {"ricker", "flat_band", "file"}:
            raise ConfigError("noise.psd_shape must be ricker, flat_band, or file")
        fmin = _positive(noise.get("minimum_frequency_hz"), "noise.minimum_frequency_hz", allow_zero=True)
        fmax = _positive(noise.get("maximum_frequency_hz"), "noise.maximum_frequency_hz")
        dominant = _positive(noise.get("dominant_frequency_hz", 0.5 * (fmin + fmax)), "noise.dominant_frequency_hz")
        if not fmin < dominant <= fmax:
            raise ConfigError("noise frequencies must satisfy minimum < dominant <= maximum")
        if fmax >= 0.5 / dt:
            raise ConfigError("noise.maximum_frequency_hz must be below temporal Nyquist")
        amplitude = _positive(noise.get("source_amplitude", 1.0), "noise.source_amplitude")
        independent = noise.get("independent_sources", True)
        if not isinstance(independent, bool):
            raise ConfigError("noise.independent_sources must be true or false")
        event_rate = _positive(
            noise.get("event_rate_hz", 0.4), "noise.event_rate_hz", allow_zero=False
        )
        receiver_noise_fraction = _positive(
            noise.get("receiver_noise_rms_fraction", 0.0),
            "noise.receiver_noise_rms_fraction",
            allow_zero=True,
        )
        receiver_noise_seed = _integer(
            noise.get("receiver_noise_seed", seed + 104729),
            "noise.receiver_noise_seed",
            minimum=0,
        )
        raw_maximum_lag = noise.get("maximum_lag_s", 0.0)
        maximum_lag = _positive(
            raw_maximum_lag,
            "noise.maximum_lag_s",
            allow_zero=True,
        )
        if maximum_lag == 0.0:
            maximum_lag = None
        elif maximum_lag >= window:
            raise ConfigError("noise.maximum_lag_s must be smaller than noise.window_s")
        psd_file = None
        if psd_shape == "file":
            raw_path = noise.get("psd_file")
            if not raw_path:
                raise ConfigError("noise.psd_file is required when psd_shape='file'")
            psd_file = (base_dir / str(raw_path)).resolve()
            if not psd_file.is_file():
                raise ConfigError(f"PSD file does not exist: {psd_file}")
        return cls(
            temporal,
            seed,
            duration,
            dt,
            window,
            overlap,
            psd_shape,
            dominant,
            fmin,
            fmax,
            amplitude,
            independent,
            event_rate,
            psd_file,
            receiver_noise_fraction,
            receiver_noise_seed,
            maximum_lag,
        )


@dataclass(frozen=True)
class DASConfig:
    """Finite-gauge DAS measurement settings; disabled for legacy cases."""

    enabled: bool
    geometry_mode: Literal["uniform", "channel_file"]
    gauge_length_m: float
    measurement: Literal["strain", "strain_rate"]
    fiber_azimuth_deg: float

    @classmethod
    def from_root(cls, data: dict[str, Any]) -> "DASConfig":
        values = _table(data, "das", required=False)
        enabled = values.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError("das.enabled must be true or false")
        geometry_mode = str(values.get("geometry_mode", "uniform")).lower()
        if geometry_mode not in {"uniform", "channel_file"}:
            raise ConfigError(
                "das.geometry_mode must be 'uniform' or 'channel_file'"
            )
        gauge_length = _positive(
            values.get("gauge_length_m", 10.0), "das.gauge_length_m"
        )
        measurement = str(values.get("measurement", "strain_rate")).lower()
        if measurement not in {"strain", "strain_rate"}:
            raise ConfigError("das.measurement must be 'strain' or 'strain_rate'")
        azimuth = float(values.get("fiber_azimuth_deg", 0.0))
        if not math.isfinite(azimuth):
            raise ConfigError("das.fiber_azimuth_deg must be finite")
        return cls(  # type: ignore[arg-type]
            enabled, geometry_mode, gauge_length, measurement, azimuth
        )


@dataclass(frozen=True)
class SourceSelectionConfig:
    enabled: bool
    method: Literal["oracle_axis"]
    objective: Literal["stationary_phase", "isotropic_egf"]
    frequency_bands_hz: tuple[tuple[float, float], ...]
    stationary_phase_half_angle_deg: float
    hard_min_endfire_fraction: float
    hard_min_bidirectional_balance: float
    hard_min_azimuth_entropy: float
    random_seed: int

    @classmethod
    def from_root(
        cls,
        data: dict[str, Any],
        noise: NoiseConfig | None,
        processing: "ProcessingConfig",
    ) -> "SourceSelectionConfig":
        selection = _table(data, "source_selection", required=False)
        enabled = selection.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError("source_selection.enabled must be true or false")
        method = str(selection.get("method", "oracle_axis")).lower()
        if method != "oracle_axis":
            raise ConfigError("source_selection.method currently supports only oracle_axis")
        objective = str(
            selection.get("objective", "stationary_phase")
        ).lower()
        if objective not in {"stationary_phase", "isotropic_egf"}:
            raise ConfigError(
                "source_selection.objective must be stationary_phase or isotropic_egf"
            )
        raw_bands = selection.get(
            "frequency_bands_hz",
            [[processing.frequency_min_hz, processing.frequency_max_hz]],
        )
        if not isinstance(raw_bands, list) or not raw_bands:
            raise ConfigError("source_selection.frequency_bands_hz must be a non-empty array")
        bands: list[tuple[float, float]] = []
        for index, raw_band in enumerate(raw_bands):
            if not isinstance(raw_band, list) or len(raw_band) != 2:
                raise ConfigError(
                    f"source_selection.frequency_bands_hz[{index}] must contain [min, max]"
                )
            lower = _positive(
                raw_band[0],
                f"source_selection.frequency_bands_hz[{index}][0]",
                allow_zero=True,
            )
            upper = _positive(
                raw_band[1], f"source_selection.frequency_bands_hz[{index}][1]"
            )
            if upper <= lower:
                raise ConfigError("source-selection band maximum must exceed minimum")
            if noise is not None and (enabled or "frequency_bands_hz" in selection) and (
                lower < noise.minimum_frequency_hz
                or upper > noise.maximum_frequency_hz
            ):
                raise ConfigError("source-selection bands must lie inside the noise band")
            bands.append((lower, upper))
        half_angle = _positive(
            selection.get("stationary_phase_half_angle_deg", 20.0),
            "source_selection.stationary_phase_half_angle_deg",
        )
        if half_angle > 90.0:
            raise ConfigError("source_selection.stationary_phase_half_angle_deg must be <= 90")
        endfire = float(selection.get("hard_min_endfire_fraction", 0.6))
        balance = float(selection.get("hard_min_bidirectional_balance", 0.3))
        entropy = float(selection.get("hard_min_azimuth_entropy", 0.7))
        if (
            not 0 <= endfire <= 1
            or not 0 <= balance <= 1
            or not 0 <= entropy <= 1
        ):
            raise ConfigError("source-selection hard thresholds must lie between 0 and 1")
        random_seed = _integer(
            selection.get("random_seed", 271828),
            "source_selection.random_seed",
            minimum=0,
        )
        return cls(
            enabled,
            "oracle_axis",
            objective,
            tuple(bands),
            half_angle,
            endfire,
            balance,
            entropy,
            random_seed,
        )


@dataclass(frozen=True)
class NNWSResearchConfig:
    """Optional pair-resolved NNWS research workflow; disabled by default."""

    enabled: bool
    epochs: int
    learning_rate: float
    snr_weight: float
    symmetry_weight: float
    std_weight: float
    std_mode: Literal["upstream_sample", "population_eps", "disabled"]
    seed: int
    random_control_seed: int
    device: Literal["auto", "cpu", "cuda", "mps"]
    minimum_effective_windows: float
    minimum_pair_coverage: float
    maximum_cache_mb: float
    pair_scope: Literal["all", "reference_receiver"]
    reference_receiver_index: int

    @classmethod
    def from_root(cls, data: dict[str, Any]) -> "NNWSResearchConfig":
        values = _table(data, "nnws", required=False)
        enabled = values.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError("nnws.enabled must be true or false")
        epochs = _integer(values.get("epochs", 200), "nnws.epochs")
        learning_rate = _positive(
            values.get("learning_rate", 1e-3), "nnws.learning_rate"
        )
        loss_weights = []
        for key, default in (
            ("snr_weight", 0.8),
            ("symmetry_weight", 0.2),
            ("std_weight", 0.01),
        ):
            number = float(values.get(key, default))
            if not math.isfinite(number) or number < 0:
                raise ConfigError(f"nnws.{key} must be finite and >= 0")
            loss_weights.append(number)
        if loss_weights[0] + loss_weights[1] <= 0:
            raise ConfigError("nnws requires a positive SNR or symmetry loss weight")
        std_mode = str(values.get("std_mode", "upstream_sample")).lower()
        if std_mode not in {"upstream_sample", "population_eps", "disabled"}:
            raise ConfigError(
                "nnws.std_mode must be upstream_sample, population_eps, or disabled"
            )
        seed = _integer(values.get("seed", 0), "nnws.seed", minimum=0)
        random_control_seed = _integer(
            values.get("random_control_seed", 271828),
            "nnws.random_control_seed",
            minimum=0,
        )
        device = str(values.get("device", "auto")).lower()
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigError("nnws.device must be auto, cpu, cuda, or mps")
        minimum_effective_windows = _positive(
            values.get("minimum_effective_windows", 2.0),
            "nnws.minimum_effective_windows",
        )
        minimum_pair_coverage = float(values.get("minimum_pair_coverage", 1.0))
        if not math.isfinite(minimum_pair_coverage) or not (
            0 < minimum_pair_coverage <= 1
        ):
            raise ConfigError("nnws.minimum_pair_coverage must lie in (0, 1]")
        maximum_cache_mb = _positive(
            values.get("maximum_cache_mb", 512.0), "nnws.maximum_cache_mb"
        )
        pair_scope = str(values.get("pair_scope", "all")).lower()
        if pair_scope not in {"all", "reference_receiver"}:
            raise ConfigError("nnws.pair_scope must be all or reference_receiver")
        reference_receiver_index = _integer(
            values.get("reference_receiver_index", 0),
            "nnws.reference_receiver_index",
            minimum=0,
        )
        return cls(
            enabled=enabled,
            epochs=epochs,
            learning_rate=learning_rate,
            snr_weight=loss_weights[0],
            symmetry_weight=loss_weights[1],
            std_weight=loss_weights[2],
            std_mode=std_mode,  # type: ignore[arg-type]
            seed=seed,
            random_control_seed=random_control_seed,
            device=device,  # type: ignore[arg-type]
            minimum_effective_windows=minimum_effective_windows,
            minimum_pair_coverage=minimum_pair_coverage,
            maximum_cache_mb=maximum_cache_mb,
            pair_scope=pair_scope,  # type: ignore[arg-type]
            reference_receiver_index=reference_receiver_index,
        )


@dataclass(frozen=True)
class NNWSV2ResearchConfig:
    """Independent experimental shared-weight, cross-fitted NNWS-v2."""

    enabled: bool
    folds: int
    epochs: int
    learning_rate: float
    arrival_weight: float
    precursor_weight: float
    symmetry_weight: float
    effective_count_weight: float
    temporal_smoothness_weight: float
    minimum_weight: float
    minimum_effective_fraction: float
    seed: int
    random_control_seed: int
    device: Literal["auto", "cpu", "cuda", "mps"]
    pair_scope: Literal["all", "reference_receiver"]
    reference_receiver_index: int
    pair_ftan_receiver_index: int | None
    maximum_cache_mb: float

    @classmethod
    def from_root(cls, data: dict[str, Any]) -> "NNWSV2ResearchConfig":
        values = _table(data, "nnws_v2", required=False)
        enabled = values.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError("nnws_v2.enabled must be true or false")
        folds = _integer(values.get("folds", 5), "nnws_v2.folds", minimum=2)
        epochs = _integer(values.get("epochs", 200), "nnws_v2.epochs")
        learning_rate = _positive(
            values.get("learning_rate", 1e-3), "nnws_v2.learning_rate"
        )
        loss_values = []
        for key, default in (
            ("arrival_weight", 0.55),
            ("precursor_weight", 0.20),
            ("symmetry_weight", 0.05),
            ("effective_count_weight", 0.20),
            ("temporal_smoothness_weight", 0.02),
        ):
            number = float(values.get(key, default))
            if not math.isfinite(number) or number < 0:
                raise ConfigError(f"nnws_v2.{key} must be finite and >= 0")
            loss_values.append(number)
        if sum(loss_values[:4]) <= 0:
            raise ConfigError("nnws_v2 requires a positive physical or Neff loss")
        minimum_weight = float(values.get("minimum_weight", 0.05))
        if not math.isfinite(minimum_weight) or not 0 <= minimum_weight < 1:
            raise ConfigError("nnws_v2.minimum_weight must lie in [0, 1)")
        minimum_effective_fraction = float(
            values.get("minimum_effective_fraction", 0.5)
        )
        if not math.isfinite(minimum_effective_fraction) or not (
            0 < minimum_effective_fraction <= 1
        ):
            raise ConfigError(
                "nnws_v2.minimum_effective_fraction must lie in (0, 1]"
            )
        seed = _integer(values.get("seed", 0), "nnws_v2.seed", minimum=0)
        random_control_seed = _integer(
            values.get("random_control_seed", 271828),
            "nnws_v2.random_control_seed",
            minimum=0,
        )
        device = str(values.get("device", "auto")).lower()
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigError("nnws_v2.device must be auto, cpu, cuda, or mps")
        pair_scope = str(values.get("pair_scope", "reference_receiver")).lower()
        if pair_scope not in {"all", "reference_receiver"}:
            raise ConfigError("nnws_v2.pair_scope must be all or reference_receiver")
        reference_receiver_index = _integer(
            values.get("reference_receiver_index", 0),
            "nnws_v2.reference_receiver_index",
            minimum=0,
        )
        raw_pair_ftan_receiver_index = values.get("pair_ftan_receiver_index")
        pair_ftan_receiver_index = (
            None
            if raw_pair_ftan_receiver_index is None
            else _integer(
                raw_pair_ftan_receiver_index,
                "nnws_v2.pair_ftan_receiver_index",
                minimum=0,
            )
        )
        maximum_cache_mb = _positive(
            values.get("maximum_cache_mb", 512.0), "nnws_v2.maximum_cache_mb"
        )
        return cls(
            enabled=enabled,
            folds=folds,
            epochs=epochs,
            learning_rate=learning_rate,
            arrival_weight=loss_values[0],
            precursor_weight=loss_values[1],
            symmetry_weight=loss_values[2],
            effective_count_weight=loss_values[3],
            temporal_smoothness_weight=loss_values[4],
            minimum_weight=minimum_weight,
            minimum_effective_fraction=minimum_effective_fraction,
            seed=seed,
            random_control_seed=random_control_seed,
            device=device,  # type: ignore[arg-type]
            pair_scope=pair_scope,  # type: ignore[arg-type]
            reference_receiver_index=reference_receiver_index,
            pair_ftan_receiver_index=pair_ftan_receiver_index,
            maximum_cache_mb=maximum_cache_mb,
        )


@dataclass(frozen=True)
class PairSPZResearchConfig:
    """Known-source, pair/frequency stationary-phase upper-bound experiment."""

    enabled: bool
    phase_tolerance_rad: float
    off_spz_weight: float
    bilateral_balance_weight: float
    effective_count_weight: float
    minimum_weight: float
    minimum_effective_fraction: float
    maximum_iterations: int
    random_seed: int
    pair_scope: Literal["all", "reference_receiver"]
    reference_receiver_index: int

    @classmethod
    def from_root(cls, data: dict[str, Any]) -> "PairSPZResearchConfig":
        values = _table(data, "pair_spz", required=False)
        enabled = values.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError("pair_spz.enabled must be true or false")
        phase_tolerance_rad = _positive(
            values.get("phase_tolerance_rad", math.pi / 2.0),
            "pair_spz.phase_tolerance_rad",
        )
        loss_values = []
        for key, default in (
            ("off_spz_weight", 0.55),
            ("bilateral_balance_weight", 0.30),
            ("effective_count_weight", 0.15),
        ):
            number = float(values.get(key, default))
            if not math.isfinite(number) or number < 0:
                raise ConfigError(f"pair_spz.{key} must be finite and >= 0")
            loss_values.append(number)
        if sum(loss_values) <= 0:
            raise ConfigError("pair_spz requires at least one positive loss weight")
        minimum_weight = float(values.get("minimum_weight", 0.05))
        if not math.isfinite(minimum_weight) or not 0 <= minimum_weight < 1:
            raise ConfigError("pair_spz.minimum_weight must lie in [0, 1)")
        minimum_effective_fraction = float(
            values.get("minimum_effective_fraction", 0.5)
        )
        if not math.isfinite(minimum_effective_fraction) or not (
            0 < minimum_effective_fraction <= 1
        ):
            raise ConfigError(
                "pair_spz.minimum_effective_fraction must lie in (0, 1]"
            )
        maximum_iterations = _integer(
            values.get("maximum_iterations", 500),
            "pair_spz.maximum_iterations",
        )
        random_seed = _integer(
            values.get("random_seed", 161803),
            "pair_spz.random_seed",
            minimum=0,
        )
        pair_scope = str(values.get("pair_scope", "reference_receiver")).lower()
        if pair_scope not in {"all", "reference_receiver"}:
            raise ConfigError("pair_spz.pair_scope must be all or reference_receiver")
        reference_receiver_index = _integer(
            values.get("reference_receiver_index", 0),
            "pair_spz.reference_receiver_index",
            minimum=0,
        )
        return cls(
            enabled=enabled,
            phase_tolerance_rad=phase_tolerance_rad,
            off_spz_weight=loss_values[0],
            bilateral_balance_weight=loss_values[1],
            effective_count_weight=loss_values[2],
            minimum_weight=minimum_weight,
            minimum_effective_fraction=minimum_effective_fraction,
            maximum_iterations=maximum_iterations,
            random_seed=random_seed,
            pair_scope=pair_scope,  # type: ignore[arg-type]
            reference_receiver_index=reference_receiver_index,
        )


@dataclass(frozen=True)
class ProcessingConfig:
    correlation_normalization: Literal["spectral_whitening", "one_bit", "none"]
    dispersion_method: Literal["phase_shift", "bessel", "hankel"]
    theory_solver: Literal["disba", "qedisp"]
    frequency_min_hz: float
    frequency_max_hz: float
    velocity_min_mps: float
    velocity_max_mps: float
    velocity_step_mps: float
    frequency_step_hz: float | None
    distance_taper_fraction: float
    zero_padding: float
    radial_kernel_lookup_step: float
    picker_smoothness: float
    picker_maximum_jump_mps: float
    picker_minimum_energy: float
    rayleigh_modes: int
    love_modes: int
    compare_methods: bool

    @classmethod
    def from_root(cls, data: dict[str, Any]) -> "ProcessingConfig":
        processing = _table(data, "processing")
        normalization = str(processing.get("correlation_normalization", "spectral_whitening")).lower()
        if normalization not in {"spectral_whitening", "one_bit", "none"}:
            raise ConfigError("processing.correlation_normalization is invalid")
        method = str(processing.get("dispersion_method", "phase_shift")).lower()
        if method not in {"phase_shift", "bessel", "hankel"}:
            raise ConfigError("processing.dispersion_method must be phase_shift, bessel, or hankel")
        theory_solver = str(processing.get("theory_solver", "disba")).lower()
        if theory_solver not in {"disba", "qedisp"}:
            raise ConfigError("processing.theory_solver must be disba or qedisp")
        fmin = _positive(processing.get("frequency_min_hz"), "processing.frequency_min_hz", allow_zero=True)
        fmax = _positive(processing.get("frequency_max_hz"), "processing.frequency_max_hz")
        vmin = _positive(processing.get("velocity_min_mps"), "processing.velocity_min_mps")
        vmax = _positive(processing.get("velocity_max_mps"), "processing.velocity_max_mps")
        vstep = _positive(processing.get("velocity_step_mps"), "processing.velocity_step_mps")
        raw_frequency_step = processing.get("frequency_step_hz", 0.0)
        frequency_step = _positive(
            raw_frequency_step,
            "processing.frequency_step_hz",
            allow_zero=True,
        )
        if frequency_step == 0.0:
            frequency_step = None
        taper = float(processing.get("distance_taper_fraction", 0.5))
        zero_padding = _positive(
            processing.get("zero_padding", 8.0), "processing.zero_padding"
        )
        radial_kernel_lookup_step = _positive(
            processing.get("radial_kernel_lookup_step", 0.002),
            "processing.radial_kernel_lookup_step",
            allow_zero=True,
        )
        picker_smoothness = _positive(
            processing.get("picker_smoothness", 0.3),
            "processing.picker_smoothness",
            allow_zero=True,
        )
        picker_jump = _positive(
            processing.get("picker_maximum_jump_mps", 25.0),
            "processing.picker_maximum_jump_mps",
        )
        picker_energy = float(processing.get("picker_minimum_energy", 0.05))
        if fmax <= fmin or vmax <= vmin:
            raise ConfigError("processing maximum values must exceed minimum values")
        if not 0 <= taper <= 1:
            raise ConfigError("processing.distance_taper_fraction must be between 0 and 1")
        if not 0 <= picker_energy <= 1:
            raise ConfigError("processing.picker_minimum_energy must be between 0 and 1")
        rayleigh = _integer(processing.get("rayleigh_modes", 4), "processing.rayleigh_modes")
        love = _integer(processing.get("love_modes", 4), "processing.love_modes")
        compare_methods = processing.get("compare_methods", False)
        if not isinstance(compare_methods, bool):
            raise ConfigError("processing.compare_methods must be true or false")
        return cls(normalization, method, theory_solver, fmin, fmax, vmin, vmax, vstep,
                   frequency_step,
                   taper, zero_padding, radial_kernel_lookup_step,
                   picker_smoothness, picker_jump,
                   picker_energy, rayleigh, love, compare_methods)


@dataclass(frozen=True)
class RunConfig:
    name: str
    mode: Literal["active", "passive"]
    output_directory: Path
    workers: int
    frequency_block_size: int
    source_block_size: int
    radial_interpolation_step_m: float
    store_source_responses: bool
    model: ModelConfig
    receivers: ReceiverLineConfig | ReceiverGridConfig | ReceiverFileConfig
    das: DASConfig
    processing: ProcessingConfig
    source_selection: SourceSelectionConfig
    nnws: NNWSResearchConfig
    nnws_v2: NNWSV2ResearchConfig
    pair_spz: PairSPZResearchConfig
    active_source: ActiveSourceConfig | None = None
    ambient_sources: AmbientSourceConfig | None = None
    noise: NoiseConfig | None = None
    config_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("output_directory", "config_path"):
            if result[key] is not None:
                result[key] = str(result[key])
        ambient = result.get("ambient_sources")
        if isinstance(ambient, dict) and ambient.get("coordinates_file") is not None:
            ambient["coordinates_file"] = str(ambient["coordinates_file"])
        if isinstance(ambient, dict) and ambient.get("activity_file") is not None:
            ambient["activity_file"] = str(ambient["activity_file"])
        noise = result.get("noise")
        if isinstance(noise, dict) and noise.get("psd_file") is not None:
            noise["psd_file"] = str(noise["psd_file"])
        receivers = result.get("receivers")
        if isinstance(receivers, dict) and receivers.get("coordinates_file") is not None:
            receivers["coordinates_file"] = str(receivers["coordinates_file"])
        return result


def config_from_mapping(
    data: dict[str, Any],
    *,
    base_directory: str | Path | None = None,
    config_path: str | Path | None = None,
) -> RunConfig:
    """Build the same validated configuration from notebook cell mappings."""
    base_dir = Path(base_directory or Path.cwd()).expanduser().resolve()
    resolved_config_path = (
        Path(config_path).expanduser().resolve() if config_path is not None else None
    )
    run = _table(data, "run")
    default_name = resolved_config_path.stem if resolved_config_path else "notebook_case"
    name = str(run.get("name", default_name)).strip()
    if not name:
        raise ConfigError("run.name must not be empty")
    mode = str(run.get("mode", "")).lower()
    if mode not in {"active", "passive"}:
        raise ConfigError("run.mode must be 'active' or 'passive'")
    raw_output = str(run.get("output_directory", f"output_python/{name}"))
    output = (base_dir / raw_output).resolve()
    workers = _integer(run.get("workers", 0), "run.workers", minimum=0)
    frequency_block = _integer(run.get("frequency_block_size", 16), "run.frequency_block_size")
    source_block = _integer(run.get("source_block_size", 16), "run.source_block_size")
    radial_step = _positive(
        run.get("radial_interpolation_step_m", 0.5),
        "run.radial_interpolation_step_m",
    )
    store_responses = run.get("store_source_responses", False)
    if not isinstance(store_responses, bool):
        raise ConfigError("run.store_source_responses must be true or false")

    model = ModelConfig.from_root(data)
    receivers = ReceiverLineConfig.from_root(data, base_dir)
    das = DASConfig.from_root(data)
    processing = ProcessingConfig.from_root(data)
    active = ActiveSourceConfig.from_root(data) if mode == "active" else None
    ambient = AmbientSourceConfig.from_root(data, base_dir) if mode == "passive" else None
    noise = NoiseConfig.from_root(data, base_dir) if mode == "passive" else None
    source_selection = SourceSelectionConfig.from_root(data, noise, processing)
    nnws = NNWSResearchConfig.from_root(data)
    nnws_v2 = NNWSV2ResearchConfig.from_root(data)
    pair_spz = PairSPZResearchConfig.from_root(data)

    sample_dt = active.dt_s if active else noise.dt_s  # type: ignore[union-attr]
    if processing.frequency_max_hz >= 0.5 / sample_dt:
        raise ConfigError("processing.frequency_max_hz must be below temporal Nyquist")
    if mode == "passive" and processing.frequency_max_hz > noise.maximum_frequency_hz:  # type: ignore[union-attr]
        raise ConfigError("processing.frequency_max_hz must not exceed noise.maximum_frequency_hz")

    if mode == "passive":
        if source_selection.enabled and not isinstance(receivers, ReceiverLineConfig):
            raise ConfigError(
                "source_selection.method='oracle_axis' currently requires line receivers"
            )
        if not isinstance(receivers, ReceiverLineConfig) and receivers.distance_bin_m is None:
            raise ConfigError(
                "passive receiver layout='grid'/'file' requires "
                "geometry.receivers.distance_bin_m"
            )
        maximum_recommended_step = min(layer.vs_mps for layer in model.layers) / (
            20.0 * noise.maximum_frequency_hz
        )
        if radial_step > maximum_recommended_step:
            raise ConfigError(
                "run.radial_interpolation_step_m is too coarse; use <= "
                f"{maximum_recommended_step:.3f} m (5% of minimum wavelength)"
            )
        if nnws.enabled and ambient.component != "vertical":  # type: ignore[union-attr]
            raise ConfigError("pair-resolved NNWS currently supports vertical ambient sources")
        if nnws_v2.enabled and ambient.component != "vertical":  # type: ignore[union-attr]
            raise ConfigError("NNWS-v2 currently supports vertical ambient sources")
        if pair_spz.enabled and ambient.component != "vertical":  # type: ignore[union-attr]
            raise ConfigError("pair-SPZ currently supports vertical ambient sources")
        if (
            nnws.enabled
            and nnws.pair_scope == "reference_receiver"
            and nnws.reference_receiver_index >= receivers.count
        ):
            raise ConfigError(
                "nnws.reference_receiver_index must identify an existing receiver"
            )
        if (
            nnws_v2.enabled
            and nnws_v2.pair_scope == "reference_receiver"
            and nnws_v2.reference_receiver_index >= receivers.count
        ):
            raise ConfigError(
                "nnws_v2.reference_receiver_index must identify an existing receiver"
            )
        if nnws_v2.pair_ftan_receiver_index is not None:
            if not nnws_v2.enabled or nnws_v2.pair_scope != "reference_receiver":
                raise ConfigError(
                    "nnws_v2.pair_ftan_receiver_index requires enabled "
                    "reference_receiver scope"
                )
            if nnws_v2.pair_ftan_receiver_index >= receivers.count:
                raise ConfigError(
                    "nnws_v2.pair_ftan_receiver_index must identify an existing receiver"
                )
            if (
                nnws_v2.pair_ftan_receiver_index
                == nnws_v2.reference_receiver_index
            ):
                raise ConfigError(
                    "nnws_v2.pair_ftan_receiver_index must differ from the reference receiver"
                )
        if (
            pair_spz.enabled
            and pair_spz.pair_scope == "reference_receiver"
            and pair_spz.reference_receiver_index >= receivers.count
        ):
            raise ConfigError(
                "pair_spz.reference_receiver_index must identify an existing receiver"
            )
        if (
            nnws.enabled
            and nnws_v2.enabled
            and (
                nnws.pair_scope != nnws_v2.pair_scope
                or nnws.reference_receiver_index
                != nnws_v2.reference_receiver_index
            )
        ):
            raise ConfigError(
                "nnws and nnws_v2 must use the same pair scope when both are enabled"
            )
        if pair_spz.enabled and nnws_v2.enabled and (
            pair_spz.pair_scope != nnws_v2.pair_scope
            or pair_spz.reference_receiver_index
            != nnws_v2.reference_receiver_index
        ):
            raise ConfigError(
                "pair_spz and nnws_v2 must use the same pair scope when combined"
            )
    elif nnws.enabled or nnws_v2.enabled or pair_spz.enabled:
        raise ConfigError("NNWS/SPZ research methods are available only in passive mode")
    if das.enabled:
        if das.geometry_mode == "channel_file" and not isinstance(
            receivers, ReceiverFileConfig
        ):
            raise ConfigError(
                "das.geometry_mode='channel_file' requires "
                "geometry.receivers.layout='file'; the receiver CSV must include "
                "x_m,y_m plus endpoint_* columns or per-channel "
                "gauge_length_m,fiber_azimuth_deg"
            )
        if das.geometry_mode == "channel_file":
            from .das import load_das_geometry_csv

            try:
                load_das_geometry_csv(receivers.coordinates_file)  # type: ignore[union-attr]
            except (OSError, ValueError) as exc:
                raise ConfigError(f"invalid DAS channel CSV: {exc}") from exc
        if processing.dispersion_method != "phase_shift":
            raise ConfigError(
                "DAS v1 requires processing.dispersion_method='phase_shift'; "
                "FJ/Bessel and Hankel DAS kernels are not yet validated"
            )
        if processing.compare_methods:
            raise ConfigError(
                "DAS v1 requires processing.compare_methods=false because only "
                "the phase-shift imaging path is validated"
            )
        if mode == "active" and active.component != "vertical":  # type: ignore[union-attr]
            raise ConfigError(
                "active DAS v1 requires active_source.component='vertical'; "
                "horizontal source azimuth is not defined by the current input schema"
            )
        if mode == "passive" and (
            source_selection.enabled
            or nnws.enabled
            or nnws_v2.enabled
            or pair_spz.enabled
        ):
            raise ConfigError(
                "DAS v1 does not yet support source_selection/NNWS/pair-SPZ; "
                "disable those experimental component-specific workflows"
            )
        if mode == "passive" and ambient.component != "vertical":  # type: ignore[union-attr]
            raise ConfigError(
                "passive DAS v1 currently requires ambient_sources.component='vertical'; "
                "three-component stochastic source forces will be added after the "
                "vertical-force DAS-DAS reference is validated"
            )
    return RunConfig(
        name=name,
        mode=mode,
        output_directory=output,
        workers=workers,
        frequency_block_size=frequency_block,
        source_block_size=source_block,
        radial_interpolation_step_m=radial_step,
        store_source_responses=store_responses,
        model=model,
        receivers=receivers,
        das=das,
        processing=processing,
        source_selection=source_selection,
        nnws=nnws,
        nnws_v2=nnws_v2,
        pair_spz=pair_spz,
        active_source=active,
        ambient_sources=ambient,
        noise=noise,
        config_path=resolved_config_path,
    )


def load_config(path: str | Path) -> RunConfig:
    """Load and validate one active or passive TOML input file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)
    return config_from_mapping(
        data,
        base_directory=config_path.parent,
        config_path=config_path,
    )
