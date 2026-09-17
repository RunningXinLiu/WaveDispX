"""Phase-only source-azimuth estimation and fixed angular-target weighting.

The routines in this module never consume source coordinates, source-event
labels, a Green function, or a theoretical dispersion curve.  They convert a
small auxiliary array's waveform snapshots into an observed backazimuth power
distribution and optionally weight time windows toward a declared angular target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .spz_selection import effective_window_count, enforce_effective_count_floor


@dataclass(frozen=True)
class BeamPowerEstimate:
    """One window's normalized backazimuth power and peak location."""

    azimuth_deg: np.ndarray
    velocity_mps: np.ndarray
    angular_power: np.ndarray
    velocity_azimuth_power: np.ndarray
    peak_azimuth_deg: float
    peak_velocity_mps: float
    valid_snapshot_count: int


@dataclass(frozen=True)
class AngularUniformityWeights:
    """Non-negative angular-target weights and auditable optimization metrics."""

    weights: np.ndarray
    success: bool
    message: str
    iterations: int
    unweighted_misfit: float
    weighted_misfit: float
    effective_window_count: float


class PhaseOnlyBeamformer:
    """Reusable broadband plane-wave beamformer for equal-length records.

    Azimuth is the backazimuth from the array centre toward the source.  A
    source at azimuth ``beta`` generates a plane-wave spectrum proportional to
    ``exp(+i 2*pi*f*r dot u(beta)/v)``.  The stored steering vectors apply the
    conjugate phase and therefore peak at ``beta`` rather than the propagation
    direction ``beta + 180 degrees``.
    """

    def __init__(
        self,
        receiver_x_m: np.ndarray,
        receiver_y_m: np.ndarray,
        *,
        dt_s: float,
        window_samples: int,
        snapshot_s: float = 4.0,
        overlap: float = 0.5,
        frequencies_hz: np.ndarray | None = None,
        azimuth_deg: np.ndarray | None = None,
        velocity_mps: np.ndarray | None = None,
    ) -> None:
        x = np.asarray(receiver_x_m, dtype=np.float64).reshape(-1)
        y = np.asarray(receiver_y_m, dtype=np.float64).reshape(-1)
        if x.size < 3 or x.shape != y.shape or np.any(~np.isfinite(x + y)):
            raise ValueError("beamforming requires at least three finite 2-D receivers")
        if not np.isfinite(dt_s) or dt_s <= 0 or window_samples < 8:
            raise ValueError("dt_s and window_samples are invalid")
        if not np.isfinite(snapshot_s) or snapshot_s <= 0 or not 0 <= overlap < 1:
            raise ValueError("snapshot_s and overlap are invalid")
        snapshot_samples = int(round(snapshot_s / dt_s))
        if snapshot_samples < 8 or snapshot_samples > window_samples:
            raise ValueError("snapshot_s does not fit inside the record")
        step_samples = int(round(snapshot_samples * (1.0 - overlap)))
        if step_samples < 1:
            raise ValueError("overlap leaves no positive snapshot step")
        starts = np.arange(
            0,
            window_samples - snapshot_samples + 1,
            step_samples,
            dtype=np.int64,
        )
        if starts.size < 1:
            raise ValueError("no complete snapshots fit inside the record")

        requested_frequency = np.asarray(
            np.arange(4.5, 12.0, 1.0)
            if frequencies_hz is None
            else frequencies_hz,
            dtype=np.float64,
        ).reshape(-1)
        azimuth = np.asarray(
            np.arange(0.0, 360.0, 5.0) if azimuth_deg is None else azimuth_deg,
            dtype=np.float64,
        ).reshape(-1)
        velocity = np.asarray(
            np.arange(160.0, 441.0, 40.0) if velocity_mps is None else velocity_mps,
            dtype=np.float64,
        ).reshape(-1)
        if (
            requested_frequency.size < 1
            or azimuth.size < 4
            or velocity.size < 1
            or np.any(~np.isfinite(requested_frequency))
            or np.any(~np.isfinite(azimuth))
            or np.any(~np.isfinite(velocity))
            or np.any(requested_frequency <= 0)
            or np.any(velocity <= 0)
        ):
            raise ValueError("beamforming search grids are invalid")
        if np.unique(np.mod(azimuth, 360.0)).size != azimuth.size:
            raise ValueError("azimuth grid must not contain duplicate directions")

        snapshot_frequency = np.fft.rfftfreq(snapshot_samples, dt_s)
        frequency_index = np.asarray(
            [int(np.argmin(np.abs(snapshot_frequency - value))) for value in requested_frequency],
            dtype=np.int64,
        )
        actual_frequency = snapshot_frequency[frequency_index]
        tolerance = 0.5 / (snapshot_samples * dt_s) + 1e-12
        if np.any(np.abs(actual_frequency - requested_frequency) > tolerance):
            raise ValueError("requested frequency is not represented by the snapshot FFT")
        if np.unique(frequency_index).size != frequency_index.size:
            raise ValueError("requested frequencies collapse onto duplicate FFT bins")

        centred_x = x - float(np.mean(x))
        centred_y = y - float(np.mean(y))
        angle_rad = np.deg2rad(np.mod(azimuth, 360.0))
        projected_distance = (
            np.cos(angle_rad)[:, None] * centred_x[None, :]
            + np.sin(angle_rad)[:, None] * centred_y[None, :]
        )
        delay = projected_distance[None, :, :] / velocity[:, None, None]
        steering = np.exp(
            -2j
            * np.pi
            * actual_frequency[:, None, None, None]
            * delay[None, :, :, :]
        ).astype(np.complex64)

        self.receiver_x_m = x
        self.receiver_y_m = y
        self.dt_s = float(dt_s)
        self.window_samples = int(window_samples)
        self.snapshot_samples = snapshot_samples
        self.snapshot_starts = starts
        self.frequency_index = frequency_index
        self.frequency_hz = actual_frequency
        self.azimuth_deg = np.mod(azimuth, 360.0)
        self.velocity_mps = velocity
        self._taper = np.hanning(snapshot_samples).astype(np.float64)
        self._steering = steering

    def estimate(self, record: np.ndarray) -> BeamPowerEstimate:
        """Estimate one record's phase-only broadband backazimuth power."""

        values = np.asarray(record, dtype=np.float64)
        expected_shape = (self.window_samples, self.receiver_x_m.size)
        if values.shape != expected_shape or np.any(~np.isfinite(values)):
            raise ValueError(f"record must have finite shape {expected_shape}")
        snapshots = np.stack(
            [
                values[start : start + self.snapshot_samples]
                for start in self.snapshot_starts
            ],
            axis=0,
        )
        snapshots = snapshots - np.mean(snapshots, axis=1, keepdims=True)
        spectra = np.fft.rfft(snapshots * self._taper[None, :, None], axis=1)
        selected = spectra[:, self.frequency_index, :]
        amplitude = np.abs(selected)
        scale = np.max(amplitude, axis=2, keepdims=True)
        valid = scale[..., 0] > np.finfo(float).eps
        phase = np.zeros_like(selected, dtype=np.complex128)
        np.divide(
            selected,
            amplitude,
            out=phase,
            where=amplitude > np.maximum(scale * 1e-8, np.finfo(float).eps),
        )
        velocity_azimuth_power = np.zeros(
            (self.velocity_mps.size, self.azimuth_deg.size), dtype=np.float64
        )
        valid_frequency_count = 0
        for frequency_position in range(self.frequency_hz.size):
            snapshot_mask = valid[:, frequency_position]
            if not np.any(snapshot_mask):
                continue
            aligned = np.einsum(
                "vam,sm->sva",
                self._steering[frequency_position],
                phase[snapshot_mask, frequency_position],
                optimize=True,
            )
            velocity_azimuth_power += np.mean(
                np.abs(aligned) ** 2 / self.receiver_x_m.size**2,
                axis=0,
            )
            valid_frequency_count += 1
        if valid_frequency_count == 0:
            raise ValueError("record contains no finite beamforming energy")
        velocity_azimuth_power /= valid_frequency_count
        angular_power = np.max(velocity_azimuth_power, axis=0)
        total = float(np.sum(angular_power))
        if not np.isfinite(total) or total <= np.finfo(float).eps:
            raise ValueError("beamforming power is empty")
        angular_power /= total
        peak_velocity_index, peak_azimuth_index = np.unravel_index(
            int(np.argmax(velocity_azimuth_power)), velocity_azimuth_power.shape
        )
        return BeamPowerEstimate(
            azimuth_deg=self.azimuth_deg.copy(),
            velocity_mps=self.velocity_mps.copy(),
            angular_power=angular_power,
            velocity_azimuth_power=velocity_azimuth_power,
            peak_azimuth_deg=float(self.azimuth_deg[peak_azimuth_index]),
            peak_velocity_mps=float(self.velocity_mps[peak_velocity_index]),
            valid_snapshot_count=int(np.count_nonzero(np.any(valid, axis=1))),
        )


def angular_uniformity_weights(
    angular_power_by_window: np.ndarray,
    *,
    regularization: float = 0.05,
    minimum_effective_fraction: float = 0.50,
    maximum_iterations: int = 500,
) -> AngularUniformityWeights:
    """Weight windows so their observed angular power approaches uniformity."""

    power = np.asarray(angular_power_by_window, dtype=np.float64)
    if power.ndim != 2:
        raise ValueError("angular_power_by_window must be a matrix")
    target = np.full(power.shape[1], 1.0 / power.shape[1])
    return angular_target_weights(
        power,
        target,
        regularization=regularization,
        minimum_effective_fraction=minimum_effective_fraction,
        maximum_iterations=maximum_iterations,
    )


def angular_target_weights(
    angular_power_by_window: np.ndarray,
    target_angular_power: np.ndarray,
    *,
    regularization: float = 0.05,
    minimum_effective_fraction: float = 0.50,
    maximum_iterations: int = 500,
) -> AngularUniformityWeights:
    """Weight windows toward one fixed, normalized angular target.

    The target is external to the optimizer.  This routine does not inspect a
    Green function, a correlation trace, a source coordinate, or a dispersion
    curve; callers are responsible for declaring whether the supplied angular
    power is an observable beam estimate or a truth-leaking diagnostic.
    """

    power = np.asarray(angular_power_by_window, dtype=np.float64)
    if (
        power.ndim != 2
        or power.shape[0] < 2
        or power.shape[1] < 4
        or np.any(~np.isfinite(power))
        or np.any(power < 0)
    ):
        raise ValueError("angular_power_by_window must be a finite non-negative matrix")
    row_sum = np.sum(power, axis=1)
    if np.any(row_sum <= np.finfo(float).eps):
        raise ValueError("every window must contain positive angular power")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and non-negative")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    normalized = power / row_sum[:, None]
    window_count, azimuth_count = normalized.shape
    target = np.asarray(target_angular_power, dtype=np.float64).reshape(-1)
    if (
        target.size != azimuth_count
        or np.any(~np.isfinite(target))
        or np.any(target < 0.0)
        or float(np.sum(target)) <= np.finfo(float).eps
    ):
        raise ValueError(
            "target_angular_power must be one finite non-negative value per azimuth"
        )
    target = target / np.sum(target)

    def misfit(weights: np.ndarray) -> float:
        aggregate = normalized.T @ weights / window_count
        difference = aggregate - target
        return float(azimuth_count * np.dot(difference, difference))

    def objective(weights: np.ndarray) -> float:
        return misfit(weights) + regularization * float(
            np.mean(np.square(weights - 1.0))
        )

    def gradient(weights: np.ndarray) -> np.ndarray:
        aggregate = normalized.T @ weights / window_count
        difference = aggregate - target
        return (
            2.0 * azimuth_count / window_count * (normalized @ difference)
            + 2.0 * regularization / window_count * (weights - 1.0)
        )

    result = minimize(
        objective,
        np.ones(window_count, dtype=np.float64),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * window_count,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - window_count),
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"maxiter": int(maximum_iterations), "ftol": 1e-12, "disp": False},
    )
    if np.any(~np.isfinite(result.x)) or float(np.sum(result.x)) <= np.finfo(float).eps:
        raise RuntimeError(f"angular uniformity optimization failed: {result.message}")
    raw_weights = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    raw_weights /= np.mean(raw_weights)
    weights = enforce_effective_count_floor(
        raw_weights, minimum_effective_fraction
    )
    return AngularUniformityWeights(
        weights=weights,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        unweighted_misfit=misfit(np.ones(window_count, dtype=np.float64)),
        weighted_misfit=misfit(weights),
        effective_window_count=effective_window_count(weights),
    )


def circular_distance_deg(first_deg: float | np.ndarray, second_deg: float) -> np.ndarray:
    """Return unsigned circular separation in degrees."""

    difference = np.mod(np.asarray(first_deg, dtype=np.float64) - second_deg + 180.0, 360.0)
    return np.abs(difference - 180.0)
