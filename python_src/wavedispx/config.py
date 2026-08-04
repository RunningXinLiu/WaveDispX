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
    component: Literal["vertical", "horizontal", "both"]
    wavelet: Literal["ricker"]
    dominant_frequency_hz: float
    maximum_frequency_hz: float
    dt_s: float

    @classmethod
    def from_root(cls, data: dict[str, Any]) -> "ActiveSourceConfig":
        source = _table(data, "active_source")
        component = str(source.get("component", "vertical")).lower()
        if component not in {"vertical", "horizontal", "both"}:
            raise ConfigError("active_source.component must be vertical, horizontal, or both")
        wavelet = str(source.get("wavelet", "ricker")).lower()
        if wavelet != "ricker":
            raise ConfigError("active_source.wavelet currently supports only 'ricker'")
        x = float(source.get("x_m", 0.0))
        y = float(source.get("y_m", 0.0))
        if not math.isfinite(x) or not math.isfinite(y):
            raise ConfigError("active source coordinates must be finite")
        dominant = _positive(source.get("dominant_frequency_hz"), "active_source.dominant_frequency_hz")
        maximum = _positive(source.get("maximum_frequency_hz"), "active_source.maximum_frequency_hz")
        dt = _positive(source.get("dt_s"), "active_source.dt_s")
        if maximum < dominant:
            raise ConfigError("active_source.maximum_frequency_hz must be >= dominant_frequency_hz")
        if maximum >= 0.5 / dt:
            raise ConfigError("active_source.maximum_frequency_hz must be below the temporal Nyquist")
        return cls(x, y, component, wavelet, dominant, maximum, dt)


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
        psd_file = None
        if psd_shape == "file":
            raw_path = noise.get("psd_file")
            if not raw_path:
                raise ConfigError("noise.psd_file is required when psd_shape='file'")
            psd_file = (base_dir / str(raw_path)).resolve()
            if not psd_file.is_file():
                raise ConfigError(f"PSD file does not exist: {psd_file}")
        return cls(temporal, seed, duration, dt, window, overlap, psd_shape, dominant,
                   fmin, fmax, amplitude, independent, event_rate, psd_file)


@dataclass(frozen=True)
class ProcessingConfig:
    correlation_normalization: Literal["spectral_whitening", "one_bit", "none"]
    dispersion_method: Literal["phase_shift", "bessel", "hankel"]
    frequency_min_hz: float
    frequency_max_hz: float
    velocity_min_mps: float
    velocity_max_mps: float
    velocity_step_mps: float
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
        fmin = _positive(processing.get("frequency_min_hz"), "processing.frequency_min_hz", allow_zero=True)
        fmax = _positive(processing.get("frequency_max_hz"), "processing.frequency_max_hz")
        vmin = _positive(processing.get("velocity_min_mps"), "processing.velocity_min_mps")
        vmax = _positive(processing.get("velocity_max_mps"), "processing.velocity_max_mps")
        vstep = _positive(processing.get("velocity_step_mps"), "processing.velocity_step_mps")
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
        return cls(normalization, method, fmin, fmax, vmin, vmax, vstep,
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
    processing: ProcessingConfig
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
    processing = ProcessingConfig.from_root(data)
    active = ActiveSourceConfig.from_root(data) if mode == "active" else None
    ambient = AmbientSourceConfig.from_root(data, base_dir) if mode == "passive" else None
    noise = NoiseConfig.from_root(data, base_dir) if mode == "passive" else None

    sample_dt = active.dt_s if active else noise.dt_s  # type: ignore[union-attr]
    if processing.frequency_max_hz >= 0.5 / sample_dt:
        raise ConfigError("processing.frequency_max_hz must be below temporal Nyquist")
    if mode == "passive" and processing.frequency_max_hz > noise.maximum_frequency_hz:  # type: ignore[union-attr]
        raise ConfigError("processing.frequency_max_hz must not exceed noise.maximum_frequency_hz")

    if mode == "passive":
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
    return RunConfig(name, mode, output, workers, frequency_block, source_block,
                     radial_step, store_responses, model, receivers, processing, active,
                     ambient, noise, resolved_config_path)


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
