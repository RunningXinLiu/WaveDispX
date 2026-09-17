"""Oracle source-illumination scoring for controlled forward experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .geometry import Geometry
from .noise import stochastic_source_coefficients
from .source_activity import load_source_activity_matrix


@dataclass(frozen=True)
class OracleWindowScore:
    total_power: np.ndarray
    endfire_fraction: np.ndarray
    bidirectional_balance: np.ndarray
    off_axis_fraction: np.ndarray
    azimuth_entropy: np.ndarray
    effective_source_count: np.ndarray
    soft_weight: float
    hard_selected: bool


@dataclass(frozen=True)
class OracleSelectionPlan:
    scope: str
    method_version: str
    frequency_bands_hz: np.ndarray
    window_index: np.ndarray
    start_time_s: np.ndarray
    end_time_s: np.ndarray
    total_power: np.ndarray
    endfire_fraction: np.ndarray
    bidirectional_balance: np.ndarray
    off_axis_fraction: np.ndarray
    azimuth_entropy: np.ndarray
    effective_source_count: np.ndarray
    hard_selected: np.ndarray
    soft_weight: np.ndarray
    random_budget_selected: np.ndarray
    random_neff_weight: np.ndarray
    correlated_source_risk: bool

    @property
    def strategy_weights(self) -> dict[str, np.ndarray]:
        return {
            "oracle_hard_mask": self.hard_selected.astype(np.float64),
            "oracle_soft_weight": self.soft_weight,
            "random_budget_control": self.random_budget_selected.astype(np.float64),
            "random_neff_control": self.random_neff_weight,
        }


def _array_axis(
    receiver_x_m: np.ndarray, receiver_y_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    receiver = np.column_stack(
        (
            np.asarray(receiver_x_m, dtype=np.float64).reshape(-1),
            np.asarray(receiver_y_m, dtype=np.float64).reshape(-1),
        )
    )
    if receiver.shape[0] < 2 or np.any(~np.isfinite(receiver)):
        raise ValueError("at least two finite receiver coordinates are required")
    centered = receiver - receiver.mean(axis=0)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= 0:
        raise ValueError("receiver coordinates do not define an array axis")
    if singular_values.size > 1 and singular_values[1] > 1e-8 * singular_values[0]:
        raise ValueError("oracle axis scoring requires a linear receiver array")
    axis = right[0]
    # Fix the otherwise arbitrary SVD sign for reproducible positive/negative labels.
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    return receiver.mean(axis=0), axis


def score_oracle_axis_window(
    source_coefficients: np.ndarray,
    frequency_hz: np.ndarray,
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    *,
    frequency_bands_hz: tuple[tuple[float, float], ...],
    stationary_phase_half_angle_deg: float,
    hard_min_endfire_fraction: float,
    hard_min_bidirectional_balance: float,
    objective: str = "stationary_phase",
    hard_min_azimuth_entropy: float = 0.0,
) -> OracleWindowScore:
    """Score one known source realization without reading receiver waveforms."""
    coefficients = np.asarray(source_coefficients)
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    source_x = np.asarray(source_x_m, dtype=np.float64).reshape(-1)
    source_y = np.asarray(source_y_m, dtype=np.float64).reshape(-1)
    if coefficients.ndim == 2:
        source_power_spectrum = np.abs(coefficients) ** 2
    elif coefficients.ndim == 3:
        source_power_spectrum = np.sum(np.abs(coefficients) ** 2, axis=1)
    else:
        raise ValueError("source_coefficients must be source-by-frequency or source-by-3-by-frequency")
    if (
        source_power_spectrum.shape != (source_x.size, frequency.size)
        or source_x.shape != source_y.shape
        or np.any(~np.isfinite(source_x))
        or np.any(~np.isfinite(source_y))
    ):
        raise ValueError("source coefficients, frequencies, and source coordinates do not match")

    midpoint, axis = _array_axis(receiver_x_m, receiver_y_m)
    relative = np.column_stack((source_x, source_y)) - midpoint
    radius = np.linalg.norm(relative, axis=1)
    if np.any(radius <= 0):
        raise ValueError("oracle source positions may not coincide with the array midpoint")
    direction = relative / radius[:, None]
    axial_cosine = direction @ axis
    cosine_limit = np.cos(np.deg2rad(stationary_phase_half_angle_deg))
    positive_endfire = axial_cosine >= cosine_limit
    negative_endfire = axial_cosine <= -cosine_limit
    endfire = positive_endfire | negative_endfire
    azimuth = np.arctan2(direction[:, 1], direction[:, 0])
    azimuth_bin = np.floor((azimuth + np.pi) / (2.0 * np.pi) * 12).astype(int) % 12

    band_count = len(frequency_bands_hz)
    total_power = np.zeros(band_count, dtype=np.float64)
    endfire_fraction = np.zeros(band_count, dtype=np.float64)
    balance = np.zeros(band_count, dtype=np.float64)
    off_axis = np.zeros(band_count, dtype=np.float64)
    entropy = np.zeros(band_count, dtype=np.float64)
    effective_count = np.zeros(band_count, dtype=np.float64)
    for band_index, (minimum, maximum) in enumerate(frequency_bands_hz):
        selected_frequency = (frequency >= minimum) & (frequency <= maximum)
        if not np.any(selected_frequency):
            raise ValueError(
                f"source-selection band [{minimum}, {maximum}] has no frequency samples"
            )
        source_power = np.sum(source_power_spectrum[:, selected_frequency], axis=1)
        total = float(np.sum(source_power))
        total_power[band_index] = total
        if total <= 0:
            continue
        positive_power = float(np.sum(source_power[positive_endfire]))
        negative_power = float(np.sum(source_power[negative_endfire]))
        endfire_power = positive_power + negative_power
        endfire_fraction[band_index] = endfire_power / total
        off_axis[band_index] = max(0.0, 1.0 - endfire_fraction[band_index])
        if endfire_power > 0:
            balance[band_index] = 2.0 * min(positive_power, negative_power) / endfire_power
        denominator = float(np.sum(source_power**2))
        if denominator > 0:
            effective_count[band_index] = total * total / denominator
        bin_power = np.bincount(azimuth_bin, weights=source_power, minlength=12)
        probability = bin_power[bin_power > 0] / total
        entropy[band_index] = float(
            -np.sum(probability * np.log(probability)) / np.log(12.0)
        )

    if objective == "stationary_phase":
        per_band_quality = endfire_fraction * balance
        hard_selected = bool(
            np.all(endfire_fraction >= hard_min_endfire_fraction)
            and np.all(balance >= hard_min_bidirectional_balance)
        )
    elif objective == "isotropic_egf":
        per_band_quality = entropy * balance
        hard_selected = bool(
            np.all(entropy >= hard_min_azimuth_entropy)
            and np.all(balance >= hard_min_bidirectional_balance)
        )
    else:
        raise ValueError("objective must be stationary_phase or isotropic_egf")
    soft_weight = float(np.mean(per_band_quality))
    return OracleWindowScore(
        total_power,
        endfire_fraction,
        balance,
        off_axis,
        entropy,
        effective_count,
        soft_weight,
        hard_selected,
    )


def build_oracle_axis_selection(
    config: RunConfig,
    geometry: Geometry,
    frequency_hz: np.ndarray,
) -> OracleSelectionPlan:
    """Precompute auditable weights from the exact source RNG stream.

    The source RNG is replayed from its seed without propagating waveforms.  The
    passive simulator then uses a fresh generator with the same seed, so these
    scores describe the exact same source realizations used by every strategy.
    """
    if (
        config.mode != "passive"
        or config.noise is None
        or config.ambient_sources is None
        or not config.source_selection.enabled
    ):
        raise ValueError("build_oracle_axis_selection requires enabled passive source selection")
    noise = config.noise
    selection = config.source_selection
    duration_ratio = noise.duration_s / noise.window_s
    window_count = int(round(duration_ratio))
    if not np.isclose(duration_ratio, window_count, rtol=0, atol=1e-10):
        raise ValueError("noise.duration_s must be an integer multiple of noise.window_s")
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    band_count = len(selection.frequency_bands_hz)
    metric_shape = (window_count, band_count)
    total_power = np.zeros(metric_shape)
    endfire_fraction = np.zeros(metric_shape)
    balance = np.zeros(metric_shape)
    off_axis = np.zeros(metric_shape)
    entropy = np.zeros(metric_shape)
    effective_count = np.zeros(metric_shape)
    soft_weight = np.zeros(window_count)
    hard_selected = np.zeros(window_count, dtype=bool)
    rng = np.random.default_rng(noise.seed)
    source_activity = load_source_activity_matrix(
        config.ambient_sources, window_count
    )
    three_component = config.ambient_sources.component == "three_component"
    coefficient_source_count = geometry.source_x_m.size * (3 if three_component else 1)
    for window_index in range(window_count):
        coefficients = stochastic_source_coefficients(
            frequency,
            coefficient_source_count,
            noise,
            rng,
        )
        if three_component:
            coefficients = coefficients.reshape(geometry.source_x_m.size, 3, frequency.size)
            coefficients *= source_activity[window_index, :, None, None]
        else:
            coefficients *= source_activity[window_index, :, None]
        score = score_oracle_axis_window(
            coefficients,
            frequency,
            geometry.source_x_m,
            geometry.source_y_m,
            geometry.receiver_x_m,
            geometry.receiver_y_m,
            frequency_bands_hz=selection.frequency_bands_hz,
            stationary_phase_half_angle_deg=selection.stationary_phase_half_angle_deg,
            hard_min_endfire_fraction=selection.hard_min_endfire_fraction,
            hard_min_bidirectional_balance=selection.hard_min_bidirectional_balance,
            objective=selection.objective,
            hard_min_azimuth_entropy=selection.hard_min_azimuth_entropy,
        )
        total_power[window_index] = score.total_power
        endfire_fraction[window_index] = score.endfire_fraction
        balance[window_index] = score.bidirectional_balance
        off_axis[window_index] = score.off_axis_fraction
        entropy[window_index] = score.azimuth_entropy
        effective_count[window_index] = score.effective_source_count
        soft_weight[window_index] = score.soft_weight
        hard_selected[window_index] = score.hard_selected

    control_rng = np.random.default_rng(selection.random_seed)
    random_order = control_rng.permutation(window_count)
    random_budget = np.zeros(window_count, dtype=bool)
    random_budget[random_order[: int(np.count_nonzero(hard_selected))]] = True
    random_neff_weight = soft_weight[control_rng.permutation(window_count)]
    window_index = np.arange(window_count, dtype=np.int64)
    start_time = window_index.astype(np.float64) * noise.window_s
    return OracleSelectionPlan(
        scope="array_axis_approximation",
        method_version=f"oracle_axis_{selection.objective}_v1",
        frequency_bands_hz=np.asarray(selection.frequency_bands_hz, dtype=np.float64),
        window_index=window_index,
        start_time_s=start_time,
        end_time_s=start_time + noise.window_s,
        total_power=total_power,
        endfire_fraction=endfire_fraction,
        bidirectional_balance=balance,
        off_axis_fraction=off_axis,
        azimuth_entropy=entropy,
        effective_source_count=effective_count,
        hard_selected=hard_selected,
        soft_weight=soft_weight,
        random_budget_selected=random_budget,
        random_neff_weight=random_neff_weight,
        correlated_source_risk=not noise.independent_sources,
    )
