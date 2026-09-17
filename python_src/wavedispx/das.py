"""Finite-gauge distributed acoustic sensing (DAS) measurement operators.

The WaveDispX elastic solver returns displacement at mathematical point
receivers.  DAS is not another source type: it is a spatial measurement of
the displacement field.  Version 1 therefore evaluates displacement at both
ends of every straight gauge and applies the endpoint-difference operator

    epsilon_tt(t) = ((u_plus(t) - u_minus(t)) dot t_hat) / L_g.

``strain_rate`` applies a time derivative after the finite-gauge difference.
NumPy's rFFT convention is used explicitly: the forward transform contains
``exp(-i omega t)`` and a time derivative is multiplication by ``+i omega``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


DASMeasurement = Literal["strain", "strain_rate"]

OPERATOR_VERSION = "wavedispx-finite-gauge-v1"
TIME_CONVENTION = "numpy_rfft_forward_exp(-iwt); derivative=+iomega"


@dataclass(frozen=True)
class DASGeometry:
    """Straight horizontal finite-gauge channels at arbitrary XY centres.

    Uniform line/grid geometries and channel-file geometries share this
    contract.  Gauge length and tangent are stored per channel so a real cable
    can turn and change gauge length without changing the forward interface.
    """

    channel_center_xy_m: np.ndarray
    endpoint_minus_xy_m: np.ndarray
    endpoint_plus_xy_m: np.ndarray
    tangent_xy: np.ndarray
    gauge_length_by_channel_m: np.ndarray
    fiber_azimuth_by_channel_deg: np.ndarray
    channel_id: tuple[str, ...]
    geometry_mode: Literal["uniform", "channel_file"] = "uniform"

    @property
    def channel_count(self) -> int:
        return int(self.channel_center_xy_m.shape[0])

    @property
    def endpoint_xy_m(self) -> np.ndarray:
        """Return endpoints ordered as all minus then all plus channels."""
        return np.concatenate(
            (self.endpoint_minus_xy_m, self.endpoint_plus_xy_m), axis=0
        )

    @property
    def gauge_length_m(self) -> float:
        """Return the uniform gauge length, or NaN for per-channel lengths."""
        first = float(self.gauge_length_by_channel_m[0])
        return first if np.allclose(self.gauge_length_by_channel_m, first) else float("nan")

    @property
    def fiber_azimuth_deg(self) -> float:
        """Return the uniform fibre azimuth, or NaN for a turning cable."""
        angle = np.deg2rad(self.fiber_azimuth_by_channel_deg)
        # Axial strain is 180-degree periodic, so compare doubled directions.
        reference = np.exp(2j * angle[0])
        if np.allclose(np.exp(2j * angle), reference):
            return float(self.fiber_azimuth_by_channel_deg[0])
        return float("nan")

    @property
    def gauge_label(self) -> str:
        minimum = float(np.min(self.gauge_length_by_channel_m))
        maximum = float(np.max(self.gauge_length_by_channel_m))
        if np.isclose(minimum, maximum):
            return f"L={minimum:g} m"
        return f"L={minimum:g}–{maximum:g} m"


@dataclass(frozen=True)
class DASRecord:
    """Finite-gauge scalar DAS record with its complete measurement contract."""

    time_s: np.ndarray
    data: np.ndarray
    geometry: DASGeometry
    measurement: DASMeasurement
    units: str
    operator_version: str = OPERATOR_VERSION
    time_convention: str = TIME_CONVENTION
    input_observable: str = "horizontal_displacement_xy"


def build_das_geometry(
    channel_x_m: np.ndarray,
    channel_y_m: np.ndarray,
    gauge_length_m: float,
    fiber_azimuth_deg: float,
) -> DASGeometry:
    """Build gauge endpoints independently of channel spacing or layout.

    This uniform constructor makes parallel straight gauge segments for any
    line/grid/file receiver centres.  Turning or variable-gauge cables use
    :func:`load_das_geometry_csv`, which provides one tangent and gauge length
    per channel under the same downstream record contract.
    """
    x = np.asarray(channel_x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(channel_y_m, dtype=np.float64).reshape(-1)
    gauge = float(gauge_length_m)
    azimuth = float(fiber_azimuth_deg)
    if x.size == 0 or x.size != y.size:
        raise ValueError("DAS channel x/y coordinates must be non-empty and aligned")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("DAS channel coordinates must be finite")
    if not np.isfinite(gauge) or gauge <= 0.0:
        raise ValueError("gauge_length_m must be finite and positive")
    if not np.isfinite(azimuth):
        raise ValueError("fiber_azimuth_deg must be finite")
    angle = np.deg2rad(azimuth)
    tangent = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    centers = np.column_stack((x, y))
    channel_tangent = np.repeat(tangent[None, :], x.size, axis=0)
    half_gauge = 0.5 * gauge * channel_tangent
    return DASGeometry(
        channel_center_xy_m=centers,
        endpoint_minus_xy_m=centers - half_gauge,
        endpoint_plus_xy_m=centers + half_gauge,
        tangent_xy=channel_tangent,
        gauge_length_by_channel_m=np.full(x.size, gauge, dtype=np.float64),
        fiber_azimuth_by_channel_deg=np.full(x.size, azimuth, dtype=np.float64),
        channel_id=tuple(f"channel_{index + 1:06d}" for index in range(x.size)),
        geometry_mode="uniform",
    )


def _float_column(
    rows: list[dict[str, str]], name: str, path: Path
) -> np.ndarray:
    values = []
    for row_index, row in enumerate(rows, start=2):
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"DAS channel CSV {path} row {row_index} has invalid {name}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"DAS channel CSV {path} row {row_index} {name} must be finite"
            )
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def load_das_geometry_csv(
    path: str | Path,
    *,
    expected_center_x_m: np.ndarray | None = None,
    expected_center_y_m: np.ndarray | None = None,
) -> DASGeometry:
    """Load per-channel DAS geometry from the receiver coordinate CSV.

    The CSV always contains ``x_m,y_m`` channel centres and at least one of two schemas:

    - explicit endpoints: ``endpoint_minus_x_m``, ``endpoint_minus_y_m``,
      ``endpoint_plus_x_m``, ``endpoint_plus_y_m``;
    - per-channel design: ``gauge_length_m,fiber_azimuth_deg``.

    ``channel_id`` is optional.  File row order is the receiver/channel order.
    If both schemas are present, explicit endpoints are authoritative; this
    makes the exported ``das_channels.csv`` directly reusable as input.
    """
    resolved = Path(path).expanduser().resolve()
    with resolved.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    if len(rows) < 2 or not {"x_m", "y_m"}.issubset(fieldnames):
        raise ValueError(
            "DAS channel CSV must contain x_m,y_m and at least two rows"
        )
    endpoint_fields = {
        "endpoint_minus_x_m",
        "endpoint_minus_y_m",
        "endpoint_plus_x_m",
        "endpoint_plus_y_m",
    }
    design_fields = {"gauge_length_m", "fiber_azimuth_deg"}
    has_endpoints = endpoint_fields.issubset(fieldnames)
    has_design = design_fields.issubset(fieldnames)
    if not has_endpoints and not has_design:
        raise ValueError(
            "DAS channel CSV must contain a geometry schema: either "
            "four endpoint_* columns or gauge_length_m,fiber_azimuth_deg"
        )
    center_x = _float_column(rows, "x_m", resolved)
    center_y = _float_column(rows, "y_m", resolved)
    centers = np.column_stack((center_x, center_y))
    if expected_center_x_m is not None or expected_center_y_m is not None:
        if expected_center_x_m is None or expected_center_y_m is None:
            raise ValueError("both expected DAS centre coordinate arrays are required")
        expected = np.column_stack(
            (
                np.asarray(expected_center_x_m, dtype=np.float64).reshape(-1),
                np.asarray(expected_center_y_m, dtype=np.float64).reshape(-1),
            )
        )
        if expected.shape != centers.shape or not np.allclose(
            expected, centers, rtol=0.0, atol=1.0e-8
        ):
            raise ValueError(
                "DAS channel CSV centres must match receiver x_m,y_m in row order"
            )
    if has_endpoints:
        minus = np.column_stack(
            (
                _float_column(rows, "endpoint_minus_x_m", resolved),
                _float_column(rows, "endpoint_minus_y_m", resolved),
            )
        )
        plus = np.column_stack(
            (
                _float_column(rows, "endpoint_plus_x_m", resolved),
                _float_column(rows, "endpoint_plus_y_m", resolved),
            )
        )
        delta = plus - minus
        gauge = np.linalg.norm(delta, axis=1)
        if np.any(gauge <= 0.0):
            raise ValueError("DAS channel endpoints must define positive gauge lengths")
        midpoint = 0.5 * (minus + plus)
        tolerance = np.maximum(1.0e-8, 1.0e-8 * gauge)[:, None]
        if np.any(np.abs(midpoint - centers) > tolerance):
            raise ValueError(
                "DAS channel x_m,y_m must equal the midpoint of its two endpoints"
            )
        tangent = delta / gauge[:, None]
        azimuth = np.degrees(np.arctan2(tangent[:, 1], tangent[:, 0]))
    else:
        gauge = _float_column(rows, "gauge_length_m", resolved)
        if np.any(gauge <= 0.0):
            raise ValueError("DAS channel gauge_length_m must be positive")
        azimuth = _float_column(rows, "fiber_azimuth_deg", resolved)
        angle = np.deg2rad(azimuth)
        tangent = np.column_stack((np.cos(angle), np.sin(angle)))
        half = 0.5 * gauge[:, None] * tangent
        minus = centers - half
        plus = centers + half
    identifiers = tuple(
        (str(row.get("channel_id", "")).strip() or f"channel_{index + 1:06d}")
        for index, row in enumerate(rows)
    )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("DAS channel_id values must be unique")
    return DASGeometry(
        channel_center_xy_m=centers,
        endpoint_minus_xy_m=minus,
        endpoint_plus_xy_m=plus,
        tangent_xy=tangent,
        gauge_length_by_channel_m=gauge,
        fiber_azimuth_by_channel_deg=azimuth,
        channel_id=identifiers,
        geometry_mode="channel_file",
    )


def build_configured_das_geometry(
    channel_x_m: np.ndarray,
    channel_y_m: np.ndarray,
    *,
    geometry_mode: str,
    gauge_length_m: float,
    fiber_azimuth_deg: float,
    receiver_coordinates_file: str | Path | None = None,
) -> DASGeometry:
    """Build either uniform gauges or per-channel geometry from a receiver CSV."""
    if geometry_mode == "uniform":
        return build_das_geometry(
            channel_x_m, channel_y_m, gauge_length_m, fiber_azimuth_deg
        )
    if geometry_mode != "channel_file":
        raise ValueError("DAS geometry_mode must be 'uniform' or 'channel_file'")
    if receiver_coordinates_file is None:
        raise ValueError(
            "DAS geometry_mode='channel_file' requires receiver layout='file'"
        )
    return load_das_geometry_csv(
        receiver_coordinates_file,
        expected_center_x_m=channel_x_m,
        expected_center_y_m=channel_y_m,
    )


def normalized_gauge_response(
    wavenumber_along_fiber_rad_m: np.ndarray | float | complex,
    gauge_length_m: np.ndarray | float,
) -> np.ndarray:
    """Return ``sinc(k_t L_g / 2)`` relative to point axial strain."""
    gauge = np.asarray(gauge_length_m, dtype=np.float64)
    if np.any(~np.isfinite(gauge)) or np.any(gauge <= 0.0):
        raise ValueError("gauge_length_m values must be finite and positive")
    argument = 0.5 * np.asarray(wavenumber_along_fiber_rad_m) * gauge
    result = np.ones_like(argument, dtype=np.result_type(argument, complex))
    regular = np.abs(argument) > 1.0e-8
    result[regular] = np.sin(argument[regular]) / argument[regular]
    if np.any(~regular):
        small = argument[~regular]
        result[~regular] = 1.0 - small**2 / 6.0 + small**4 / 120.0
    return np.real_if_close(result)


def finite_gauge_strain_transfer(
    wavenumber_along_fiber_rad_m: np.ndarray | float | complex,
    gauge_length_m: float,
) -> np.ndarray:
    """Return the endpoint-difference strain transfer ``2i sin(kL/2)/L``."""
    wavenumber = np.asarray(wavenumber_along_fiber_rad_m)
    return 1j * wavenumber * normalized_gauge_response(
        wavenumber, gauge_length_m
    )


def temporal_derivative_transfer(frequency_hz: np.ndarray | float) -> np.ndarray:
    """Return ``+i omega`` for NumPy's forward rFFT ``exp(-i omega t)``."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("frequency_hz must be finite and non-negative")
    return 2j * np.pi * frequency


def gauge_response_on_velocity_grid(
    frequency_hz: np.ndarray,
    phase_velocity_mps: np.ndarray,
    gauge_length_m: float,
) -> np.ndarray:
    """Absolute finite-gauge sensitivity for propagation along the fibre."""
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    velocity = np.asarray(phase_velocity_mps, dtype=np.float64).reshape(-1)
    if (
        frequency.size == 0
        or velocity.size == 0
        or np.any(~np.isfinite(frequency))
        or np.any(~np.isfinite(velocity))
        or np.any(frequency <= 0.0)
        or np.any(velocity <= 0.0)
    ):
        raise ValueError("frequency and phase velocity grids must be positive")
    wavenumber = 2.0 * np.pi * frequency[:, None] / velocity[None, :]
    return np.abs(normalized_gauge_response(wavenumber, gauge_length_m)).astype(
        np.float64
    )


def apply_finite_gauge_operator(
    endpoint_minus_displacement_xy: np.ndarray,
    endpoint_plus_displacement_xy: np.ndarray,
    geometry: DASGeometry,
    *,
    dt_s: float,
    measurement: DASMeasurement,
) -> DASRecord:
    """Convert endpoint horizontal displacement into strain or strain rate.

    Arrays must be ``(time, channel, 2)`` in global X/Y coordinates.  Applying
    the spatial difference before the temporal derivative keeps the operator
    identical for active records and mixed passive-noise windows.
    """
    minus = np.asarray(endpoint_minus_displacement_xy, dtype=np.float64)
    plus = np.asarray(endpoint_plus_displacement_xy, dtype=np.float64)
    if minus.shape != plus.shape or minus.ndim != 3 or minus.shape[2] != 2:
        raise ValueError(
            "endpoint displacement arrays must share shape (time, channel, 2)"
        )
    if minus.shape[1] != geometry.channel_count:
        raise ValueError("endpoint displacement channel count does not match geometry")
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if measurement not in {"strain", "strain_rate"}:
        raise ValueError("measurement must be 'strain' or 'strain_rate'")

    axial_difference = np.einsum(
        "tci,ci->tc", plus - minus, geometry.tangent_xy, optimize=True
    )
    data = axial_difference / geometry.gauge_length_by_channel_m[None, :]
    if measurement == "strain_rate":
        frequency = np.fft.rfftfreq(data.shape[0], dt)
        spectrum = np.fft.rfft(data, axis=0)
        spectrum *= temporal_derivative_transfer(frequency)[:, None]
        data = np.fft.irfft(spectrum, n=data.shape[0], axis=0)
    time_s = np.arange(data.shape[0], dtype=np.float64) * dt
    return DASRecord(
        time_s=time_s,
        data=data,
        geometry=geometry,
        measurement=measurement,
        units="1" if measurement == "strain" else "s^-1",
    )
