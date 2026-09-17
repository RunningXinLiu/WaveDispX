"""Experimental cross-fitted, array-shared successor to upstream NNWS v1.

This module is intentionally separate from the vendored :mod:`nnws` package.
It addresses three failure modes found in the v1 audit:

* every saved weight is out-of-fold for a contiguous time block;
* one weight vector is shared by the full virtual-source gather;
* an explicit effective-sample-size penalty replaces the v1 negative weight-
  standard-deviation term that can encourage concentrated or unstable weights.

The implementation is a research ablation, not a claim of upstream parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .nnws import (
    PairwiseNCFStack,
    pair_nnws_eligibility,
    stack_pair_ncf_windows,
)
from .processing import NCFBranches, NCFPairBranches, bin_pair_ncf_branches


@dataclass(frozen=True)
class NNWSV2Config:
    folds: int = 5
    epochs: int = 200
    learning_rate: float = 1e-3
    arrival_weight: float = 0.55
    precursor_weight: float = 0.20
    symmetry_weight: float = 0.05
    effective_count_weight: float = 0.20
    temporal_smoothness_weight: float = 0.02
    minimum_weight: float = 0.05
    minimum_effective_fraction: float = 0.50
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class NNWSV2FitResult:
    weights: np.ndarray
    fold_index: np.ndarray
    effective_window_count: float
    proxy_stack: np.ndarray
    proxy_matrix: np.ndarray
    proxy_pair_mask: np.ndarray
    proxy_distance_m: float
    signal_window_s: tuple[float, float]
    fold_history: tuple[dict[str, list[float]], ...]
    sufficient: bool


@dataclass(frozen=True)
class NNWSV2Result:
    fit: NNWSV2FitResult
    pair_stack: PairwiseNCFStack
    binned_branches: NCFBranches
    random_control_stack: PairwiseNCFStack
    random_control_binned: NCFBranches
    scope: str = "array_shared_moveout_proxy_contiguous_oof"


def enforce_effective_window_floor(
    weights: np.ndarray, minimum_effective_fraction: float
) -> np.ndarray:
    """Apply the configured Neff constraint to final out-of-fold weights."""
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        values.size < 2
        or np.any(~np.isfinite(values))
        or np.any(values <= 0)
        or not np.isfinite(minimum_effective_fraction)
        or not 0 < minimum_effective_fraction <= 1
    ):
        raise ValueError("weights and minimum_effective_fraction are invalid")
    values = values / np.mean(values)
    target = minimum_effective_fraction * values.size

    def effective(candidate: np.ndarray) -> float:
        return float(np.sum(candidate) ** 2 / np.sum(np.square(candidate)))

    if effective(values) >= target:
        return values
    uniform = np.ones_like(values)
    lower = 0.0
    upper = 1.0
    for _ in range(60):
        fraction = 0.5 * (lower + upper)
        candidate = uniform + fraction * (values - uniform)
        if effective(candidate) >= target:
            lower = fraction
        else:
            upper = fraction
    result = uniform + lower * (values - uniform)
    return result / np.mean(result)


def blend_shared_weights_with_directional_guard(
    shared_weights: np.ndarray,
    directional_band_weights: np.ndarray,
    *,
    directional_fraction: float = 0.8,
    minimum_effective_fraction: float = 0.5,
) -> np.ndarray:
    """Conservatively blend shared NNWS-v2 weights with OOF tau-p evidence.

    The shared network supplies waveform-quality information but cannot observe
    source azimuth directly.  ``directional_band_weights`` supplies positive,
    cross-fitted tau-p diffuseness weights for each frequency band.  A log-space
    blend preserves both rankings, and the same effective-window floor used by
    NNWS-v2 prevents a small number of time windows from dominating the stack.
    """

    shared = np.asarray(shared_weights, dtype=np.float64).reshape(-1)
    directional = np.asarray(directional_band_weights, dtype=np.float64)
    if (
        directional.ndim < 2
        or directional.shape[0] != shared.size
        or shared.size < 2
        or np.any(~np.isfinite(shared))
        or np.any(~np.isfinite(directional))
        or np.any(shared <= 0.0)
        or np.any(directional <= 0.0)
    ):
        raise ValueError("shared and directional weights must be positive and aligned")
    if not np.isfinite(directional_fraction) or not 0.0 <= directional_fraction <= 1.0:
        raise ValueError("directional_fraction must lie in [0, 1]")
    shared_shape = (shared.size,) + (1,) * (directional.ndim - 1)
    combined = np.exp(
        (1.0 - directional_fraction) * np.log(shared.reshape(shared_shape))
        + directional_fraction * np.log(directional)
    )
    flattened = combined.reshape(shared.size, -1)
    constrained = np.column_stack(
        [
            enforce_effective_window_floor(
                flattened[:, column_index], minimum_effective_fraction
            )
            for column_index in range(flattened.shape[1])
        ]
    )
    return constrained.reshape(directional.shape)


def blend_frequency_and_pair_weights(
    frequency_weights: np.ndarray,
    pair_weights: np.ndarray,
    *,
    frequency_fraction: float = 0.7,
    minimum_effective_fraction: float = 0.5,
) -> np.ndarray:
    """Blend window-band azimuth evidence with window-pair arrival evidence."""

    frequency = np.asarray(frequency_weights, dtype=np.float64)
    pair = np.asarray(pair_weights, dtype=np.float64)
    if (
        frequency.ndim != 2
        or pair.ndim != 2
        or frequency.shape[0] != pair.shape[0]
        or frequency.shape[0] < 2
        or np.any(~np.isfinite(frequency))
        or np.any(~np.isfinite(pair))
        or np.any(frequency <= 0.0)
        or np.any(pair <= 0.0)
    ):
        raise ValueError("frequency and pair weights must be positive and aligned")
    if not np.isfinite(frequency_fraction) or not 0.0 <= frequency_fraction <= 1.0:
        raise ValueError("frequency_fraction must lie in [0, 1]")
    combined = np.exp(
        frequency_fraction * np.log(frequency[:, :, None])
        + (1.0 - frequency_fraction) * np.log(pair[:, None, :])
    )
    flattened = combined.reshape(combined.shape[0], -1)
    constrained = np.column_stack(
        [
            enforce_effective_window_floor(
                flattened[:, column_index], minimum_effective_fraction
            )
            for column_index in range(flattened.shape[1])
        ]
    )
    return constrained.reshape(combined.shape)


def apply_pair_weights_within_frequency_mask(
    frequency_mask: np.ndarray,
    pair_weights: np.ndarray,
    *,
    minimum_effective_support_fraction: float = 0.75,
) -> np.ndarray:
    """Reweight pairs only inside a fixed frequency-band window mask."""

    mask = np.asarray(frequency_mask, dtype=np.float64)
    pair = np.asarray(pair_weights, dtype=np.float64)
    if (
        mask.ndim != 2
        or pair.ndim != 2
        or mask.shape[0] != pair.shape[0]
        or mask.shape[0] < 2
        or np.any(~np.isfinite(mask))
        or np.any((mask != 0.0) & (mask != 1.0))
        or np.any(~np.isfinite(pair))
        or np.any(pair <= 0.0)
    ):
        raise ValueError("frequency mask and pair weights must be finite and aligned")
    if not 0.0 < minimum_effective_support_fraction <= 1.0:
        raise ValueError(
            "minimum_effective_support_fraction must lie in (0, 1]"
        )
    output = np.zeros((mask.shape[0], mask.shape[1], pair.shape[1]))
    for band_index in range(mask.shape[1]):
        support = mask[:, band_index] > 0.0
        support_count = int(np.count_nonzero(support))
        if support_count < 2:
            raise ValueError("each frequency mask band must retain at least two windows")
        target_fraction = (
            minimum_effective_support_fraction * support_count / mask.shape[0]
        )
        for pair_index in range(pair.shape[1]):
            candidate = np.zeros(mask.shape[0], dtype=np.float64)
            candidate[support] = pair[support, pair_index]
            selected = enforce_effective_window_floor(
                candidate[support], minimum_effective_support_fraction
            )
            candidate[support] = selected
            output[:, band_index, pair_index] = candidate
            effective = np.square(np.sum(candidate)) / np.sum(np.square(candidate))
            if effective < target_fraction * mask.shape[0] - 1e-10:
                raise RuntimeError("support-constrained Neff floor was not achieved")
    return output


def _validate_config(config: NNWSV2Config, window_count: int) -> None:
    if config.folds < 2 or config.folds > window_count // 2:
        raise ValueError("NNWS-v2 folds must leave at least two training windows")
    if config.epochs <= 0:
        raise ValueError("NNWS-v2 epochs must be positive")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("NNWS-v2 learning_rate must be finite and positive")
    loss_weights = np.asarray(
        (
            config.arrival_weight,
            config.precursor_weight,
            config.symmetry_weight,
            config.effective_count_weight,
            config.temporal_smoothness_weight,
        ),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(loss_weights)) or np.any(loss_weights < 0):
        raise ValueError("NNWS-v2 loss weights must be finite and non-negative")
    if np.sum(loss_weights[:4]) <= 0:
        raise ValueError("NNWS-v2 requires at least one physical or Neff loss")
    if not 0 <= config.minimum_weight < 1:
        raise ValueError("NNWS-v2 minimum_weight must lie in [0, 1)")
    if not 0 < config.minimum_effective_fraction <= 1:
        raise ValueError("NNWS-v2 minimum_effective_fraction must lie in (0, 1]")
    if config.seed < 0:
        raise ValueError("NNWS-v2 seed must be non-negative")
    if config.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("NNWS-v2 device must be auto, cpu, cuda, or mps")


def _full_lag_cube(windows: Sequence[NCFPairBranches]) -> tuple[np.ndarray, np.ndarray]:
    if not windows:
        raise ValueError("at least one pair-resolved NCF window is required")
    template = windows[0]
    positive_lag = np.asarray(template.positive.lag_time_s, dtype=np.float64)
    rows = []
    for window in windows:
        if (
            window.positive.data.shape != template.positive.data.shape
            or not np.array_equal(window.positive.pair_i, template.positive.pair_i)
            or not np.array_equal(window.positive.pair_j, template.positive.pair_j)
            or not np.array_equal(window.positive.distance_m, template.positive.distance_m)
        ):
            raise ValueError("pair-resolved NCF geometry changed between windows")
        full = np.concatenate(
            (window.negative.data[:0:-1], window.positive.data), axis=0
        ).T
        if np.any(~np.isfinite(full)):
            raise ValueError("NNWS-v2 input CCFs must be finite")
        rows.append(full)
    full_lag = np.concatenate((-positive_lag[:0:-1], positive_lag))
    return np.stack(rows, axis=0), full_lag


def build_moveout_aligned_proxy(
    windows: Sequence[NCFPairBranches],
    *,
    dt_s: float,
    minimum_velocity_mps: float,
    maximum_velocity_mps: float,
    maximum_period_s: float,
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    """Build one causal/acausal-preserving proxy CCF per time window."""
    cube, full_lag = _full_lag_cube(windows)
    template = windows[0]
    distance = np.asarray(template.positive.distance_m, dtype=np.float64)
    eligible, _ = pair_nnws_eligibility(
        distance,
        lag_time_s=template.positive.lag_time_s,
        dt_s=dt_s,
        minimum_velocity_mps=minimum_velocity_mps,
        maximum_velocity_mps=maximum_velocity_mps,
        maximum_period_s=maximum_period_s,
    )
    if np.count_nonzero(eligible) < 2:
        raise ValueError("NNWS-v2 moveout proxy requires at least two eligible pairs")
    zero = full_lag.size // 2
    branch_time = full_lag[zero:]
    positive = cube[:, :, zero:]
    negative = cube[:, :, : zero + 1][:, :, ::-1]
    tau_s = 0.5 * maximum_period_s
    center_velocity = 2.0 / (
        1.0 / minimum_velocity_mps + 1.0 / maximum_velocity_mps
    )
    reference_distance = float(np.median(distance[eligible]))
    reference_center = reference_distance / center_velocity
    reference_window = (
        reference_distance / maximum_velocity_mps,
        reference_distance / minimum_velocity_mps,
    )
    aligned_positive = []
    aligned_negative = []
    for pair_index in np.flatnonzero(eligible):
        pair_signal = (
            branch_time >= distance[pair_index] / maximum_velocity_mps - 2 * tau_s
        ) & (
            branch_time <= distance[pair_index] / minimum_velocity_mps + 2 * tau_s
        )
        pair_energy = np.sqrt(
            np.mean(
                0.5
                * (
                    np.square(positive[:, pair_index, pair_signal])
                    + np.square(negative[:, pair_index, pair_signal])
                ),
                axis=1,
            )
        )
        scale = float(np.median(pair_energy[pair_energy > 0]))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            continue
        pair_center = distance[pair_index] / center_velocity
        source_time = branch_time - reference_center + pair_center
        aligned_positive.append(
            np.stack(
                [
                    np.interp(
                        source_time,
                        branch_time,
                        row / scale,
                        left=0.0,
                        right=0.0,
                    )
                    for row in positive[:, pair_index]
                ]
            )
        )
        aligned_negative.append(
            np.stack(
                [
                    np.interp(
                        source_time,
                        branch_time,
                        row / scale,
                        left=0.0,
                        right=0.0,
                    )
                    for row in negative[:, pair_index]
                ]
            )
        )
    if len(aligned_positive) < 2:
        raise ValueError("NNWS-v2 proxy has fewer than two usable receiver pairs")
    proxy_positive = np.mean(aligned_positive, axis=0)
    proxy_negative = np.mean(aligned_negative, axis=0)
    proxy = np.concatenate(
        (proxy_negative[:, :0:-1], proxy_positive), axis=1
    )
    row_rms = np.sqrt(np.mean(np.square(proxy), axis=1, keepdims=True))
    proxy = proxy / np.maximum(row_rms, np.finfo(float).eps)
    return proxy, eligible, reference_distance, reference_window


def _torch_device(name: str):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("NNWS-v2 requires PyTorch") from exc
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train_crossfit_proxy_weights(
    proxy_matrix: np.ndarray,
    *,
    dt_s: float,
    signal_window_s: tuple[float, float],
    maximum_period_s: float,
    config: NNWSV2Config | None = None,
    proxy_pair_mask: np.ndarray | None = None,
    proxy_distance_m: float = float("nan"),
) -> NNWSV2FitResult:
    """Learn contiguous-block out-of-fold weights for one proxy CCF matrix."""
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("NNWS-v2 requires PyTorch") from exc

    matrix = np.asarray(proxy_matrix, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 4
        or matrix.shape[1] < 3
        or matrix.shape[1] % 2 == 0
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError("proxy_matrix must have shape [M >= 4, odd N >= 3]")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    tmin_s, tmax_s = map(float, signal_window_s)
    if not 0 <= tmin_s < tmax_s:
        raise ValueError("signal_window_s must satisfy 0 <= tmin < tmax")
    if not np.isfinite(maximum_period_s) or maximum_period_s <= 0:
        raise ValueError("maximum_period_s must be finite and positive")
    config = config or NNWSV2Config()
    _validate_config(config, matrix.shape[0])
    device = _torch_device(config.device)
    window_count, sample_count = matrix.shape
    fold_index = np.minimum(
        np.arange(window_count) * config.folds // window_count,
        config.folds - 1,
    )
    out_of_fold_weights = np.zeros(window_count, dtype=np.float64)
    histories: list[dict[str, list[float]]] = []
    zero = sample_count // 2
    tau_s = 0.5 * maximum_period_s
    start = int((tmin_s - 2 * tau_s) / dt_s)
    stop = int((tmax_s + 2 * tau_s) / dt_s)
    branch_count = zero + 1
    if start <= 0 or stop >= branch_count or start >= stop:
        raise ValueError("NNWS-v2 signal window leaves no precursor/noise samples")

    class WeightNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(sample_count, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                    nn.init.zeros_(layer.bias)

        def forward(self, values):
            raw = self.net(values).squeeze(-1)
            return config.minimum_weight + (1.0 - config.minimum_weight) * raw

    for fold in range(config.folds):
        validation = fold_index == fold
        training = ~validation
        training_indices = np.flatnonzero(training)
        torch.manual_seed(config.seed + fold)
        model = WeightNet().to(device)
        training_values = torch.as_tensor(
            matrix[training], dtype=torch.float32, device=device
        )
        mean = training_values.mean(dim=0, keepdim=True)
        std = training_values.std(dim=0, unbiased=False, keepdim=True).clamp_min(
            1e-5
        )
        standardized_training = (training_values - mean) / std
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        history = {
            name: []
            for name in (
                "total_loss",
                "arrival_loss",
                "precursor_loss",
                "symmetry_loss",
                "effective_count_loss",
                "temporal_smoothness_loss",
                "effective_window_count",
            )
        }
        for _ in range(config.epochs):
            weights = model(standardized_training)
            stack = torch.sum(weights[:, None] * training_values, dim=0) / weights.sum()
            positive = stack[zero:]
            negative = torch.flip(stack[: zero + 1], dims=(0,))
            symmetric = 0.5 * (positive + negative)
            signal_rms = torch.sqrt(torch.mean(symmetric[start:stop].square()) + 1e-8)
            noise = torch.cat((symmetric[:start], symmetric[stop:]))
            arrival_loss = torch.sqrt(torch.mean(noise.square()) + 1e-8) / signal_rms
            precursor_loss = torch.sqrt(
                torch.mean(
                    0.5
                    * (
                        positive[:start].square()
                        + negative[:start].square()
                    )
                )
                + 1e-8
            ) / signal_rms
            symmetry_loss = torch.sqrt(
                torch.mean(
                    (positive[start:stop] - negative[start:stop]).square()
                )
                + 1e-8
            ) / torch.sqrt(
                torch.mean(
                    (positive[start:stop] + negative[start:stop]).square()
                )
                + 1e-8
            )
            effective_count = weights.sum().square() / weights.square().sum()
            target_effective = config.minimum_effective_fraction * weights.numel()
            effective_count_loss = torch.relu(
                (target_effective - effective_count) / target_effective
            ).square()
            adjacent = np.diff(training_indices) == 1
            if np.any(adjacent):
                adjacent_tensor = torch.as_tensor(adjacent, device=device)
                temporal_loss = torch.mean(
                    (weights[1:][adjacent_tensor] - weights[:-1][adjacent_tensor]).square()
                )
            else:
                temporal_loss = torch.zeros((), device=device)
            total_loss = (
                config.arrival_weight * arrival_loss
                + config.precursor_weight * precursor_loss
                + config.symmetry_weight * symmetry_loss
                + config.effective_count_weight * effective_count_loss
                + config.temporal_smoothness_weight * temporal_loss
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            for name, value in (
                ("total_loss", total_loss),
                ("arrival_loss", arrival_loss),
                ("precursor_loss", precursor_loss),
                ("symmetry_loss", symmetry_loss),
                ("effective_count_loss", effective_count_loss),
                ("temporal_smoothness_loss", temporal_loss),
                ("effective_window_count", effective_count),
            ):
                history[name].append(float(value.detach().cpu()))
        validation_values = torch.as_tensor(
            matrix[validation], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            validation_weights = model((validation_values - mean) / std)
        validation_weights = validation_weights.cpu().numpy().astype(np.float64)
        validation_weights /= max(
            float(np.mean(validation_weights)), np.finfo(float).eps
        )
        out_of_fold_weights[validation] = validation_weights
        histories.append(history)

    out_of_fold_weights = enforce_effective_window_floor(
        out_of_fold_weights, config.minimum_effective_fraction
    )
    sum_weight = float(np.sum(out_of_fold_weights))
    sum_squared = float(np.sum(np.square(out_of_fold_weights)))
    effective_count = sum_weight * sum_weight / sum_squared
    proxy_stack = np.average(matrix, axis=0, weights=out_of_fold_weights)
    minimum_effective = config.minimum_effective_fraction * window_count
    return NNWSV2FitResult(
        weights=out_of_fold_weights,
        fold_index=fold_index,
        effective_window_count=effective_count,
        proxy_stack=proxy_stack,
        proxy_matrix=matrix,
        proxy_pair_mask=(
            np.ones(1, dtype=bool)
            if proxy_pair_mask is None
            else np.asarray(proxy_pair_mask, dtype=bool)
        ),
        proxy_distance_m=float(proxy_distance_m),
        signal_window_s=(tmin_s, tmax_s),
        fold_history=tuple(histories),
        sufficient=bool(effective_count >= minimum_effective),
    )


def train_shared_crossfit_nnws_v2(
    windows: Sequence[NCFPairBranches],
    *,
    dt_s: float,
    distance_bin_m: float,
    minimum_velocity_mps: float,
    maximum_velocity_mps: float,
    maximum_period_s: float,
    config: NNWSV2Config | None = None,
    random_control_seed: int = 271828,
) -> NNWSV2Result:
    """Train shared OOF weights and apply them to every receiver pair."""
    config = config or NNWSV2Config()
    proxy, pair_mask, proxy_distance, signal_window = build_moveout_aligned_proxy(
        windows,
        dt_s=dt_s,
        minimum_velocity_mps=minimum_velocity_mps,
        maximum_velocity_mps=maximum_velocity_mps,
        maximum_period_s=maximum_period_s,
    )
    fit = train_crossfit_proxy_weights(
        proxy,
        dt_s=dt_s,
        signal_window_s=signal_window,
        maximum_period_s=maximum_period_s,
        config=config,
        proxy_pair_mask=pair_mask,
        proxy_distance_m=proxy_distance,
    )
    pair_count = windows[0].positive.distance_m.size
    shared_weights = np.repeat(fit.weights[:, None], pair_count, axis=1)
    minimum_effective = config.minimum_effective_fraction * len(windows)
    pair_stack = stack_pair_ncf_windows(
        windows,
        shared_weights,
        minimum_effective_windows=minimum_effective,
    )
    rng = np.random.default_rng(int(random_control_seed))
    random_weights = rng.permutation(fit.weights)
    random_matrix = np.repeat(random_weights[:, None], pair_count, axis=1)
    random_stack = stack_pair_ncf_windows(
        windows,
        random_matrix,
        minimum_effective_windows=minimum_effective,
    )
    return NNWSV2Result(
        fit=fit,
        pair_stack=pair_stack,
        binned_branches=bin_pair_ncf_branches(
            pair_stack.branches, distance_bin_m=distance_bin_m
        ),
        random_control_stack=random_stack,
        random_control_binned=bin_pair_ncf_branches(
            random_stack.branches, distance_bin_m=distance_bin_m
        ),
    )
