"""Pair-aware stationary-phase illumination weights for controlled experiments.

The selector in this module is deliberately an oracle: it reads the known
forward-model source coordinates and replays the source random stream.  It is
therefore an upper-bound experiment for a future beamforming estimator, not a
field-data selector by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import RunConfig
from .geometry import Geometry
from .noise import stochastic_source_coefficients
from .processing import theoretical_dispersion
from .source_activity import load_source_activity_matrix


@dataclass(frozen=True)
class PairSPZConfig:
    frequency_bands_hz: tuple[tuple[float, float], ...]
    phase_tolerance_rad: float = np.pi / 2.0
    off_spz_weight: float = 0.55
    bilateral_balance_weight: float = 0.30
    effective_count_weight: float = 0.15
    minimum_weight: float = 0.05
    minimum_effective_fraction: float = 0.50
    maximum_iterations: int = 500
    random_seed: int = 161803
    pair_scope: str = "reference_receiver"
    reference_receiver_index: int = 0


@dataclass(frozen=True)
class PairSPZSelectionPlan:
    scope: str
    method_version: str
    frequency_bands_hz: np.ndarray
    window_index: np.ndarray
    start_time_s: np.ndarray
    end_time_s: np.ndarray
    positive_spz_power: np.ndarray
    negative_spz_power: np.ndarray
    off_spz_power: np.ndarray
    spz_fraction: np.ndarray
    per_window_balance: np.ndarray
    weights: np.ndarray
    random_neff_weights: np.ndarray
    band_weights: np.ndarray
    band_random_neff_weights: np.ndarray
    band_effective_window_count: np.ndarray
    positive_branch_band_weights: np.ndarray
    negative_branch_band_weights: np.ndarray
    positive_branch_band_random_weights: np.ndarray
    negative_branch_band_random_weights: np.ndarray
    branch_band_effective_window_count: np.ndarray
    pair_positive_branch_weights: np.ndarray
    pair_negative_branch_weights: np.ndarray
    pair_positive_branch_random_weights: np.ndarray
    pair_negative_branch_random_weights: np.ndarray
    pair_branch_effective_window_count: np.ndarray
    pair_positive_branch_band_weights: np.ndarray
    pair_negative_branch_band_weights: np.ndarray
    pair_positive_branch_band_random_weights: np.ndarray
    pair_negative_branch_band_random_weights: np.ndarray
    pair_branch_band_effective_window_count: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    effective_window_count: float
    optimized_spz_fraction: np.ndarray
    optimized_bilateral_balance: np.ndarray
    pair_count: int
    optimization_success: bool
    optimization_message: str

    @property
    def strategy_weights(self) -> dict[str, np.ndarray]:
        return {
            "oracle_pair_spz": self.weights,
            "oracle_pair_spz_random_neff_control": self.random_neff_weights,
        }


def effective_window_count(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    denominator = float(np.sum(np.square(values)))
    return (
        float(np.sum(values) ** 2 / denominator)
        if denominator > np.finfo(float).eps
        else 0.0
    )


def enforce_effective_count_floor(
    weights: np.ndarray, minimum_effective_fraction: float
) -> np.ndarray:
    """Blend non-negative weights toward uniform until the Neff floor is met."""
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        values.size < 2
        or np.any(~np.isfinite(values))
        or np.any(values < 0)
        or not 0 < minimum_effective_fraction <= 1
    ):
        raise ValueError("weights and minimum_effective_fraction are invalid")
    if float(np.sum(values)) <= np.finfo(float).eps:
        raise ValueError("weights must have positive total mass")
    values = values / np.mean(values)
    target = minimum_effective_fraction * values.size
    if effective_window_count(values) >= target:
        return values
    uniform = np.ones_like(values)
    lower = 0.0
    upper = 1.0
    for _ in range(60):
        fraction = 0.5 * (lower + upper)
        candidate = uniform + fraction * (values - uniform)
        if effective_window_count(candidate) >= target:
            lower = fraction
        else:
            upper = fraction
    result = uniform + lower * (values - uniform)
    return result / np.mean(result)


def combine_physical_and_data_weights(
    data_weights: np.ndarray,
    physical_weights: np.ndarray,
    *,
    minimum_effective_fraction: float,
) -> np.ndarray:
    """Use an equal log-space blend, then enforce the common Neff floor."""
    return blend_physical_and_data_weights(
        data_weights,
        physical_weights,
        physical_fraction=0.5,
        minimum_effective_fraction=minimum_effective_fraction,
    )


def blend_physical_and_data_weights(
    data_weights: np.ndarray,
    physical_weights: np.ndarray,
    *,
    physical_fraction: float,
    minimum_effective_fraction: float,
) -> np.ndarray:
    """Interpolate positive weights in log space and enforce a Neff floor."""
    data = np.asarray(data_weights, dtype=np.float64).reshape(-1)
    physical = np.asarray(physical_weights, dtype=np.float64).reshape(-1)
    if data.shape != physical.shape or np.any(data <= 0) or np.any(physical <= 0):
        raise ValueError("data and physical weights must be positive and aligned")
    if not np.isfinite(physical_fraction) or not 0 <= physical_fraction <= 1:
        raise ValueError("physical_fraction must lie in [0, 1]")
    combined = np.exp(
        (1.0 - physical_fraction) * np.log(data)
        + physical_fraction * np.log(physical)
    )
    return enforce_effective_count_floor(combined, minimum_effective_fraction)


def _validate_config(config: PairSPZConfig, receiver_count: int) -> None:
    if not config.frequency_bands_hz:
        raise ValueError("at least one SPZ frequency band is required")
    if not np.isfinite(config.phase_tolerance_rad) or config.phase_tolerance_rad <= 0:
        raise ValueError("phase_tolerance_rad must be finite and positive")
    losses = np.asarray(
        (
            config.off_spz_weight,
            config.bilateral_balance_weight,
            config.effective_count_weight,
        ),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(losses)) or np.any(losses < 0) or losses.sum() <= 0:
        raise ValueError("SPZ loss weights must be non-negative with positive sum")
    if not 0 <= config.minimum_weight < 1:
        raise ValueError("minimum_weight must lie in [0, 1)")
    if not 0 < config.minimum_effective_fraction <= 1:
        raise ValueError("minimum_effective_fraction must lie in (0, 1]")
    if config.maximum_iterations < 1 or config.random_seed < 0:
        raise ValueError("maximum_iterations must be positive and seed non-negative")
    if config.pair_scope not in {"all", "reference_receiver"}:
        raise ValueError("pair_scope must be all or reference_receiver")
    if not 0 <= config.reference_receiver_index < receiver_count:
        raise ValueError("reference_receiver_index must identify a receiver")


def _receiver_pairs(
    receiver_count: int, pair_scope: str, reference_receiver_index: int
) -> tuple[np.ndarray, np.ndarray]:
    pair_i, pair_j = np.triu_indices(receiver_count, k=1)
    if pair_scope == "reference_receiver":
        keep = (pair_i == reference_receiver_index) | (
            pair_j == reference_receiver_index
        )
        pair_i = pair_i[keep]
        pair_j = pair_j[keep]
    return pair_i, pair_j


def pair_spz_kernels(
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    frequency_hz: np.ndarray,
    phase_velocity_mps: np.ndarray,
    *,
    phase_tolerance_rad: float,
    pair_scope: str = "reference_receiver",
    reference_receiver_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return source-frequency kernels for the two stationary-phase branches."""
    positive, negative, pair_i, _ = pair_spz_kernels_resolved(
        source_x_m,
        source_y_m,
        receiver_x_m,
        receiver_y_m,
        frequency_hz,
        phase_velocity_mps,
        phase_tolerance_rad=phase_tolerance_rad,
        pair_scope=pair_scope,
        reference_receiver_index=reference_receiver_index,
    )
    return np.mean(positive, axis=0), np.mean(negative, axis=0), int(pair_i.size)


def pair_spz_kernels_resolved(
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    frequency_hz: np.ndarray,
    phase_velocity_mps: np.ndarray,
    *,
    phase_tolerance_rad: float,
    pair_scope: str = "reference_receiver",
    reference_receiver_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pair-resolved source-frequency kernels for both lag branches."""
    source = np.column_stack(
        (
            np.asarray(source_x_m, dtype=np.float64).reshape(-1),
            np.asarray(source_y_m, dtype=np.float64).reshape(-1),
        )
    )
    receiver = np.column_stack(
        (
            np.asarray(receiver_x_m, dtype=np.float64).reshape(-1),
            np.asarray(receiver_y_m, dtype=np.float64).reshape(-1),
        )
    )
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    velocity = np.asarray(phase_velocity_mps, dtype=np.float64).reshape(-1)
    if (
        source.shape[0] < 1
        or receiver.shape[0] < 2
        or frequency.size < 1
        or velocity.shape != frequency.shape
        or np.any(~np.isfinite(source))
        or np.any(~np.isfinite(receiver))
        or np.any(~np.isfinite(frequency))
        or np.any(frequency <= 0)
        or np.any(~np.isfinite(velocity))
        or np.any(velocity <= 0)
        or not np.isfinite(phase_tolerance_rad)
        or phase_tolerance_rad <= 0
    ):
        raise ValueError("SPZ kernel inputs are invalid")
    pair_i, pair_j = _receiver_pairs(
        receiver.shape[0], pair_scope, reference_receiver_index
    )
    first_distance = np.linalg.norm(
        source[None, :, :] - receiver[pair_i, None, :], axis=2
    )
    second_distance = np.linalg.norm(
        source[None, :, :] - receiver[pair_j, None, :], axis=2
    )
    pair_distance = np.linalg.norm(receiver[pair_j] - receiver[pair_i], axis=1)
    path_difference = second_distance - first_distance
    path_mismatch = np.maximum(
        pair_distance[:, None] - np.abs(path_difference), 0.0
    )
    phase_error = (
        2.0
        * np.pi
        * frequency[None, None, :]
        * path_mismatch[:, :, None]
        / velocity[None, None, :]
    )
    kernel = np.exp(-0.5 * np.square(phase_error / phase_tolerance_rad))
    # receiver_pair_ncf_branches uses conj(X_i) * X_j: a source closer to j
    # (negative path_difference) appears on the positive-lag branch.
    positive = kernel * (path_difference[:, :, None] < 0)
    negative = kernel * (path_difference[:, :, None] >= 0)
    return positive, negative, pair_i, pair_j


def optimize_pair_spz_weights(
    positive_power: np.ndarray,
    negative_power: np.ndarray,
    off_power: np.ndarray,
    *,
    config: PairSPZConfig,
) -> tuple[np.ndarray, bool, str]:
    """Optimize one ensemble weight vector for SPZ fraction and bilateral balance."""
    positive = np.asarray(positive_power, dtype=np.float64)
    negative = np.asarray(negative_power, dtype=np.float64)
    off = np.asarray(off_power, dtype=np.float64)
    if (
        positive.ndim != 2
        or positive.shape != negative.shape
        or positive.shape != off.shape
        or positive.shape[0] < 2
        or np.any(~np.isfinite(positive))
        or np.any(~np.isfinite(negative))
        or np.any(~np.isfinite(off))
        or np.any(positive < 0)
        or np.any(negative < 0)
        or np.any(off < 0)
    ):
        raise ValueError("SPZ power arrays must be aligned finite non-negative matrices")
    epsilon = np.finfo(float).eps
    target_effective = config.minimum_effective_fraction * positive.shape[0]

    def weights_from_logits(logits: np.ndarray) -> np.ndarray:
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        weights = config.minimum_weight + (1.0 - config.minimum_weight) * sigmoid
        return weights / np.mean(weights)

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits)
        weighted_positive = weights @ positive
        weighted_negative = weights @ negative
        weighted_off = weights @ off
        stationary = weighted_positive + weighted_negative
        total = stationary + weighted_off
        off_fraction = weighted_off / np.maximum(total, epsilon)
        imbalance = (weighted_positive - weighted_negative) / np.maximum(
            stationary, epsilon
        )
        effective = effective_window_count(weights)
        effective_loss = max(0.0, (target_effective - effective) / target_effective) ** 2
        return float(
            config.off_spz_weight * np.mean(np.square(off_fraction))
            + config.bilateral_balance_weight * np.mean(np.square(imbalance))
            + config.effective_count_weight * effective_loss
        )

    result = minimize(
        objective,
        np.zeros(positive.shape[0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0)] * positive.shape[0],
        options={"maxiter": config.maximum_iterations, "ftol": 1e-12},
    )
    weights = enforce_effective_count_floor(
        weights_from_logits(result.x), config.minimum_effective_fraction
    )
    return weights, bool(result.success), str(result.message)


def optimize_one_sided_pair_spz_weights(
    target_power: np.ndarray,
    opposite_power: np.ndarray,
    off_power: np.ndarray,
    *,
    config: PairSPZConfig,
) -> tuple[np.ndarray, bool, str]:
    """Optimize one branch without requiring unavailable opposite illumination."""
    target = np.asarray(target_power, dtype=np.float64).reshape(-1)
    opposite = np.asarray(opposite_power, dtype=np.float64).reshape(-1)
    off = np.asarray(off_power, dtype=np.float64).reshape(-1)
    if (
        target.size < 2
        or target.shape != opposite.shape
        or target.shape != off.shape
        or np.any(~np.isfinite(target))
        or np.any(~np.isfinite(opposite))
        or np.any(~np.isfinite(off))
        or np.any(target < 0)
        or np.any(opposite < 0)
        or np.any(off < 0)
    ):
        raise ValueError("one-sided SPZ powers must be aligned and non-negative")
    epsilon = np.finfo(float).eps
    target_effective = config.minimum_effective_fraction * target.size

    def weights_from_logits(logits: np.ndarray) -> np.ndarray:
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        weights = config.minimum_weight + (1.0 - config.minimum_weight) * sigmoid
        return weights / np.mean(weights)

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits)
        weighted_target = float(weights @ target)
        weighted_contamination = float(weights @ (opposite + off))
        contamination_fraction = weighted_contamination / max(
            weighted_target + weighted_contamination, epsilon
        )
        effective = effective_window_count(weights)
        effective_loss = max(
            0.0, (target_effective - effective) / target_effective
        ) ** 2
        return float(
            config.off_spz_weight * contamination_fraction**2
            + config.effective_count_weight * effective_loss
        )

    result = minimize(
        objective,
        np.zeros(target.size, dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0)] * target.size,
        options={"maxiter": config.maximum_iterations, "ftol": 1e-12},
    )
    weights = enforce_effective_count_floor(
        weights_from_logits(result.x), config.minimum_effective_fraction
    )
    return weights, bool(result.success), str(result.message)


def build_oracle_pair_spz_selection(
    run_config: RunConfig,
    geometry: Geometry,
    frequency_hz: np.ndarray,
    *,
    config: PairSPZConfig,
) -> PairSPZSelectionPlan:
    """Replay the exact forward source field and optimize ensemble SPZ weights."""
    if (
        run_config.mode != "passive"
        or run_config.noise is None
        or run_config.ambient_sources is None
    ):
        raise ValueError("pair-SPZ selection requires a passive configuration")
    _validate_config(config, geometry.receiver_x_m.size)
    noise = run_config.noise
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    window_count = int(round(noise.duration_s / noise.window_s))
    theory = theoretical_dispersion(
        run_config.model,
        frequency,
        rayleigh_modes=1,
        love_modes=0,
        backend=run_config.processing.theory_solver,
    )
    phase_velocity = theory.rayleigh_mps[:, 0]
    (
        positive_pair_kernel,
        negative_pair_kernel,
        pair_i,
        pair_j,
    ) = pair_spz_kernels_resolved(
        geometry.source_x_m,
        geometry.source_y_m,
        geometry.receiver_x_m,
        geometry.receiver_y_m,
        frequency,
        phase_velocity,
        phase_tolerance_rad=config.phase_tolerance_rad,
        pair_scope=config.pair_scope,
        reference_receiver_index=config.reference_receiver_index,
    )
    positive_kernel = np.mean(positive_pair_kernel, axis=0)
    negative_kernel = np.mean(negative_pair_kernel, axis=0)
    pair_count = int(pair_i.size)
    source_activity = load_source_activity_matrix(
        run_config.ambient_sources, window_count
    )
    rng = np.random.default_rng(noise.seed)
    band_count = len(config.frequency_bands_hz)
    positive_power = np.zeros((window_count, band_count), dtype=np.float64)
    negative_power = np.zeros_like(positive_power)
    off_power = np.zeros_like(positive_power)
    positive_pair_power = np.zeros(
        (window_count, pair_count, band_count), dtype=np.float64
    )
    negative_pair_power = np.zeros_like(positive_pair_power)
    off_pair_power = np.zeros_like(positive_pair_power)
    for window_index in range(window_count):
        coefficients = stochastic_source_coefficients(
            frequency, geometry.source_x_m.size, noise, rng
        )
        coefficients *= source_activity[window_index, :, None]
        source_power = np.square(np.abs(coefficients))
        for band_index, (minimum, maximum) in enumerate(
            config.frequency_bands_hz
        ):
            selected = (frequency >= minimum) & (frequency <= maximum)
            if not np.any(selected):
                raise ValueError("SPZ frequency band has no transfer samples")
            total = float(np.sum(source_power[:, selected]))
            positive_power[window_index, band_index] = float(
                np.sum(source_power[:, selected] * positive_kernel[:, selected])
            )
            negative_power[window_index, band_index] = float(
                np.sum(source_power[:, selected] * negative_kernel[:, selected])
            )
            stationary = (
                positive_power[window_index, band_index]
                + negative_power[window_index, band_index]
            )
            off_power[window_index, band_index] = max(total - stationary, 0.0)
            positive_pair_power[window_index, :, band_index] = np.einsum(
                "sf,psf->p",
                source_power[:, selected],
                positive_pair_kernel[:, :, selected],
                optimize=True,
            )
            negative_pair_power[window_index, :, band_index] = np.einsum(
                "sf,psf->p",
                source_power[:, selected],
                negative_pair_kernel[:, :, selected],
                optimize=True,
            )
            off_pair_power[window_index, :, band_index] = np.maximum(
                total
                - positive_pair_power[window_index, :, band_index]
                - negative_pair_power[window_index, :, band_index],
                0.0,
            )
    weights, success, message = optimize_pair_spz_weights(
        positive_power,
        negative_power,
        off_power,
        config=config,
    )
    band_weights = np.column_stack(
        [
            optimize_pair_spz_weights(
                positive_power[:, band_index : band_index + 1],
                negative_power[:, band_index : band_index + 1],
                off_power[:, band_index : band_index + 1],
                config=config,
            )[0]
            for band_index in range(band_count)
        ]
    )
    positive_branch_results = [
        optimize_one_sided_pair_spz_weights(
            positive_power[:, band_index],
            negative_power[:, band_index],
            off_power[:, band_index],
            config=config,
        )
        for band_index in range(band_count)
    ]
    negative_branch_results = [
        optimize_one_sided_pair_spz_weights(
            negative_power[:, band_index],
            positive_power[:, band_index],
            off_power[:, band_index],
            config=config,
        )
        for band_index in range(band_count)
    ]
    positive_branch_band_weights = np.column_stack(
        [result[0] for result in positive_branch_results]
    )
    negative_branch_band_weights = np.column_stack(
        [result[0] for result in negative_branch_results]
    )
    pair_positive_results = [
        optimize_one_sided_pair_spz_weights(
            np.sum(positive_pair_power[:, pair_index, :], axis=1),
            np.sum(negative_pair_power[:, pair_index, :], axis=1),
            np.sum(off_pair_power[:, pair_index, :], axis=1),
            config=config,
        )
        for pair_index in range(pair_count)
    ]
    pair_negative_results = [
        optimize_one_sided_pair_spz_weights(
            np.sum(negative_pair_power[:, pair_index, :], axis=1),
            np.sum(positive_pair_power[:, pair_index, :], axis=1),
            np.sum(off_pair_power[:, pair_index, :], axis=1),
            config=config,
        )
        for pair_index in range(pair_count)
    ]
    pair_positive_branch_weights = np.column_stack(
        [result[0] for result in pair_positive_results]
    )
    pair_negative_branch_weights = np.column_stack(
        [result[0] for result in pair_negative_results]
    )
    pair_positive_branch_band_weights = np.empty(
        (window_count, band_count, pair_count), dtype=np.float64
    )
    pair_negative_branch_band_weights = np.empty_like(
        pair_positive_branch_band_weights
    )
    pair_band_success = []
    pair_band_messages = []
    for pair_index in range(pair_count):
        for band_index in range(band_count):
            positive_result = optimize_one_sided_pair_spz_weights(
                positive_pair_power[:, pair_index, band_index],
                negative_pair_power[:, pair_index, band_index],
                off_pair_power[:, pair_index, band_index],
                config=config,
            )
            negative_result = optimize_one_sided_pair_spz_weights(
                negative_pair_power[:, pair_index, band_index],
                positive_pair_power[:, pair_index, band_index],
                off_pair_power[:, pair_index, band_index],
                config=config,
            )
            pair_positive_branch_band_weights[:, band_index, pair_index] = (
                positive_result[0]
            )
            pair_negative_branch_band_weights[:, band_index, pair_index] = (
                negative_result[0]
            )
            pair_band_success.extend((positive_result[1], negative_result[1]))
            pair_band_messages.extend((positive_result[2], negative_result[2]))
    weighted_positive = weights @ positive_power
    weighted_negative = weights @ negative_power
    weighted_off = weights @ off_power
    stationary = weighted_positive + weighted_negative
    total = stationary + weighted_off
    epsilon = np.finfo(float).eps
    optimized_spz = stationary / np.maximum(total, epsilon)
    optimized_balance = 2.0 * np.minimum(
        weighted_positive, weighted_negative
    ) / np.maximum(stationary, epsilon)
    per_window_stationary = positive_power + negative_power
    per_window_total = per_window_stationary + off_power
    spz_fraction = per_window_stationary / np.maximum(per_window_total, epsilon)
    per_window_balance = 2.0 * np.minimum(
        positive_power, negative_power
    ) / np.maximum(per_window_stationary, epsilon)
    control_rng = np.random.default_rng(config.random_seed)
    random_weights = weights[control_rng.permutation(window_count)]
    band_random_weights = band_weights[control_rng.permutation(window_count)]
    branch_permutation = control_rng.permutation(window_count)
    positive_branch_band_random_weights = positive_branch_band_weights[
        branch_permutation
    ]
    negative_branch_band_random_weights = negative_branch_band_weights[
        branch_permutation
    ]
    pair_branch_permutation = control_rng.permutation(window_count)
    pair_positive_branch_random_weights = pair_positive_branch_weights[
        pair_branch_permutation
    ]
    pair_negative_branch_random_weights = pair_negative_branch_weights[
        pair_branch_permutation
    ]
    pair_band_permutation = control_rng.permutation(window_count)
    pair_positive_branch_band_random_weights = (
        pair_positive_branch_band_weights[pair_band_permutation]
    )
    pair_negative_branch_band_random_weights = (
        pair_negative_branch_band_weights[pair_band_permutation]
    )
    window_index = np.arange(window_count, dtype=np.int64)
    start_time = window_index * noise.window_s
    return PairSPZSelectionPlan(
        scope="pair_frequency_stationary_phase_known_source",
        method_version="oracle_pair_spz_v2_branch",
        frequency_bands_hz=np.asarray(config.frequency_bands_hz),
        window_index=window_index,
        start_time_s=start_time,
        end_time_s=start_time + noise.window_s,
        positive_spz_power=positive_power,
        negative_spz_power=negative_power,
        off_spz_power=off_power,
        spz_fraction=spz_fraction,
        per_window_balance=per_window_balance,
        weights=weights,
        random_neff_weights=random_weights,
        band_weights=band_weights,
        band_random_neff_weights=band_random_weights,
        band_effective_window_count=np.asarray(
            [
                effective_window_count(band_weights[:, band_index])
                for band_index in range(band_count)
            ]
        ),
        positive_branch_band_weights=positive_branch_band_weights,
        negative_branch_band_weights=negative_branch_band_weights,
        positive_branch_band_random_weights=(
            positive_branch_band_random_weights
        ),
        negative_branch_band_random_weights=(
            negative_branch_band_random_weights
        ),
        branch_band_effective_window_count=np.vstack(
            (
                [
                    effective_window_count(
                        positive_branch_band_weights[:, band_index]
                    )
                    for band_index in range(band_count)
                ],
                [
                    effective_window_count(
                        negative_branch_band_weights[:, band_index]
                    )
                    for band_index in range(band_count)
                ],
            )
        ),
        pair_positive_branch_weights=pair_positive_branch_weights,
        pair_negative_branch_weights=pair_negative_branch_weights,
        pair_positive_branch_random_weights=(
            pair_positive_branch_random_weights
        ),
        pair_negative_branch_random_weights=(
            pair_negative_branch_random_weights
        ),
        pair_branch_effective_window_count=np.vstack(
            (
                [
                    effective_window_count(pair_positive_branch_weights[:, index])
                    for index in range(pair_count)
                ],
                [
                    effective_window_count(pair_negative_branch_weights[:, index])
                    for index in range(pair_count)
                ],
            )
        ),
        pair_positive_branch_band_weights=pair_positive_branch_band_weights,
        pair_negative_branch_band_weights=pair_negative_branch_band_weights,
        pair_positive_branch_band_random_weights=(
            pair_positive_branch_band_random_weights
        ),
        pair_negative_branch_band_random_weights=(
            pair_negative_branch_band_random_weights
        ),
        pair_branch_band_effective_window_count=np.stack(
            (
                [
                    [
                        effective_window_count(
                            pair_positive_branch_band_weights[
                                :, band_index, pair_index
                            ]
                        )
                        for pair_index in range(pair_count)
                    ]
                    for band_index in range(band_count)
                ],
                [
                    [
                        effective_window_count(
                            pair_negative_branch_band_weights[
                                :, band_index, pair_index
                            ]
                        )
                        for pair_index in range(pair_count)
                    ]
                    for band_index in range(band_count)
                ],
            )
        ),
        pair_i=pair_i,
        pair_j=pair_j,
        effective_window_count=effective_window_count(weights),
        optimized_spz_fraction=optimized_spz,
        optimized_bilateral_balance=optimized_balance,
        pair_count=pair_count,
        optimization_success=bool(
            success
            and all(result[1] for result in positive_branch_results)
            and all(result[1] for result in negative_branch_results)
            and all(result[1] for result in pair_positive_results)
            and all(result[1] for result in pair_negative_results)
            and all(pair_band_success)
        ),
        optimization_message="; ".join(
            [
                message,
                *(result[2] for result in positive_branch_results),
                *(result[2] for result in negative_branch_results),
                *(result[2] for result in pair_positive_results),
                *(result[2] for result in pair_negative_results),
                *pair_band_messages,
            ]
        ),
    )
