"""Adapter for Fan et al. (2026) Neural Network Weighting Stacking.

The vendored :mod:`nnws` package is an unmodified core-source snapshot of the
authors' v1.0 release.  This module only translates WaveDispX short-window NCF
objects to the full-lag matrix expected by that package and applies the learned
scalar window weights through the existing auditable accumulator.

The upstream method is pairwise.  Applying weights learned at one calibration
distance to a whole distance-binned gather is an experimental array proxy and
is labelled as such by :func:`train_nnws_window_strategy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .processing import (
    NCFBranches,
    NCFPairBranches,
    NCFPairGather,
    bin_pair_ncf_branches,
)
from .stacking import MultiStrategyNCFAccumulator, StrategyNCFStack


class NNWSDependencyError(RuntimeError):
    """Raised when the optional PyTorch runtime is unavailable."""


@dataclass(frozen=True)
class NNWSConfig:
    """Training parameters matching the upstream v1.0 defaults."""

    epochs: int = 200
    learning_rate: float = 1e-3
    snr_weight: float = 0.8
    symmetry_weight: float = 0.2
    std_weight: float = 0.01
    std_mode: str = "upstream_sample"
    std_epsilon: float = 1e-12
    loss_epsilon: float = 1e-12
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class NNWSFitResult:
    """Learned weights and diagnostics for one full-lag CCF matrix."""

    stack: np.ndarray
    weights: np.ndarray
    effective_window_count: float
    training_snr: float
    reported_snr: float
    symmetry_ratio: float
    history: dict[str, list[float]]
    signal_window_s: tuple[float, float]
    maximum_period_s: float


@dataclass(frozen=True)
class NNWSWindowStrategyResult:
    """NNWS fit plus the same weights applied to all cached NCF branches."""

    fit: NNWSFitResult
    strategy_stack: StrategyNCFStack
    calibration_component: str
    calibration_distance_index: int
    calibration_distance_m: float
    scope: str = "array_proxy_from_one_distance_bin"


@dataclass(frozen=True)
class PairWeightStatistics:
    """Per-pair audit quantities for window-dependent weights."""

    weights: np.ndarray
    sum_weight: np.ndarray
    sum_weight_squared: np.ndarray
    effective_window_count: np.ndarray
    nonzero_window_count: np.ndarray
    sufficient: np.ndarray


@dataclass(frozen=True)
class PairwiseNCFStack:
    branches: NCFPairBranches
    statistics: PairWeightStatistics


@dataclass(frozen=True)
class PairwiseNNWSResult:
    pair_stack: PairwiseNCFStack
    binned_branches: NCFBranches
    common_linear_stack: PairwiseNCFStack
    common_linear_binned: NCFBranches
    random_control_stack: PairwiseNCFStack
    random_control_binned: NCFBranches
    hybrid_full_aperture_stack: PairwiseNCFStack
    hybrid_full_aperture_binned: NCFBranches
    hybrid_random_control_stack: PairwiseNCFStack
    hybrid_random_control_binned: NCFBranches
    eligible: np.ndarray
    eligibility_reason: tuple[str, ...]
    training_snr: np.ndarray
    reported_snr: np.ndarray
    symmetry_ratio: np.ndarray
    final_loss: np.ndarray
    failure_reason: tuple[str, ...]
    scope: str = "pair_resolved"


def _upstream_api():
    try:
        from nnws import (
            TrainingConfig as UpstreamTrainingConfig,
            apply_stacking_network,
            reported_snr,
            train,
            training_rms_ratio,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise NNWSDependencyError(
                "NNWS requires the optional PyTorch runtime; install the "
                "project's 'nnws' extra before enabling this method."
            ) from exc
        raise
    return (
        UpstreamTrainingConfig,
        apply_stacking_network,
        reported_snr,
        train,
        training_rms_ratio,
    )


def _validate_config(config: NNWSConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("NNWS epochs must be positive")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("NNWS learning_rate must be finite and positive")
    loss_weights = np.array(
        [config.snr_weight, config.symmetry_weight, config.std_weight],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(loss_weights)) or np.any(loss_weights < 0):
        raise ValueError("NNWS loss weights must be finite and non-negative")
    if config.snr_weight + config.symmetry_weight <= 0:
        raise ValueError("NNWS requires a positive SNR or symmetry loss weight")
    if config.std_mode not in {"upstream_sample", "population_eps", "disabled"}:
        raise ValueError(
            "NNWS std_mode must be upstream_sample, population_eps, or disabled"
        )
    if not np.isfinite(config.std_epsilon) or config.std_epsilon <= 0:
        raise ValueError("NNWS std_epsilon must be finite and positive")
    if not np.isfinite(config.loss_epsilon) or config.loss_epsilon <= 0:
        raise ValueError("NNWS loss_epsilon must be finite and positive")


def _train_stable_std_variant(
    matrix: np.ndarray,
    *,
    tau_s: float,
    tmin: float,
    tmax: float,
    dt_s: float,
    lag_s: float,
    config: NNWSConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Train an explicit numerical-stability ablation without editing upstream."""
    try:
        import torch
        from nnws import StackingNet, seed_everything
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise NNWSDependencyError(
                "NNWS requires the optional PyTorch runtime; install the "
                "project's 'nnws' extra before enabling this method."
            ) from exc
        raise

    seed_everything(config.seed)
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    model = StackingNet(
        lag=lag_s,
        delta=dt_s,
        sampling_rate=1.0 / dt_s,
    ).to(device)
    values = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    tau_index = int(tau_s / dt_s)
    start = int(tmin / dt_s) - 2 * tau_index
    stop = int(tmax / dt_s) + 2 * tau_index
    symmetric_length = (matrix.shape[1] + 1) // 2
    if start <= 0 or stop >= symmetric_length or start >= stop:
        raise ValueError("NNWS signal window must leave noise samples on both sides")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {
        name: [] for name in ("total_loss", "snr_loss", "sym_loss", "std_loss")
    }
    model.train()
    for _ in range(config.epochs):
        stack, weights = model(values)
        zero_lag = stack.shape[-1] // 2
        symmetric = (
            torch.flip(stack[:, : zero_lag + 1], dims=(-1,))
            + stack[:, zero_lag:]
        )
        signal_values = symmetric[:, start:stop]
        noise_values = torch.cat(
            (symmetric[:, :start], symmetric[:, stop:]), dim=-1
        )
        stable_rms = lambda values: torch.sqrt(
            torch.mean(values.square(), dim=-1) + config.loss_epsilon
        )
        snr_component = (
            stable_rms(noise_values) / stable_rms(signal_values)
        ).mean()
        anti_symmetric = stack - torch.flip(stack, dims=(-1,))
        symmetry_component = (
            stable_rms(anti_symmetric) / stable_rms(symmetric)
        ).mean()
        if config.std_mode == "disabled":
            std_component = weights.sum() * 0.0
        else:
            centered = weights - weights.mean()
            std_component = -torch.sqrt(
                torch.mean(centered.square()) + config.std_epsilon
            )
        total = (
            config.snr_weight * snr_component
            + config.symmetry_weight * symmetry_component
            + config.std_weight * std_component
        )
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        for key, value in (
            ("total_loss", total),
            ("snr_loss", snr_component),
            ("sym_loss", symmetry_component),
            ("std_loss", std_component),
        ):
            history[key].append(float(value.detach().cpu()))
    model.eval()
    with torch.no_grad():
        stack, weights = model(values)
    return (
        stack.squeeze(0).cpu().numpy(),
        weights.squeeze(-1).cpu().numpy(),
        history,
    )


def ccf_matrix_from_branches(
    windows: Sequence[Mapping[str, NCFBranches]],
    *,
    component: str = "zz",
    distance_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build an ``[window, full_lag]`` matrix for one distance bin.

    WaveDispX stores the negative-lag branch reversed onto a non-negative lag
    axis.  NNWS expects conventional ordering from negative to positive lag.
    Zero lag is included only once in the returned matrix.
    """
    if not windows:
        raise ValueError("at least one short-window NCF is required")
    if not isinstance(distance_index, (int, np.integer)):
        raise TypeError("distance_index must be an integer")

    rows: list[np.ndarray] = []
    reference_positive = None
    for window_index, component_branches in enumerate(windows):
        if component not in component_branches:
            raise ValueError(
                f"component {component!r} is absent from NCF window {window_index}"
            )
        branches = component_branches[component]
        positive = branches.positive
        negative = branches.negative
        if positive.data.shape != negative.data.shape:
            raise ValueError("positive and negative NCF branch shapes must match")
        if not 0 <= int(distance_index) < positive.data.shape[1]:
            raise IndexError("distance_index is outside the NCF gather")
        if not np.array_equal(positive.lag_time_s, negative.lag_time_s):
            raise ValueError("positive and negative lag axes must match")
        if not np.array_equal(positive.distance_m, negative.distance_m):
            raise ValueError("positive and negative distance axes must match")
        if reference_positive is None:
            reference_positive = positive
        else:
            if (
                positive.data.shape != reference_positive.data.shape
                or not np.array_equal(
                    positive.lag_time_s, reference_positive.lag_time_s
                )
                or not np.array_equal(
                    positive.distance_m, reference_positive.distance_m
                )
            ):
                raise ValueError("NCF gather geometry changed between windows")
        row = np.concatenate(
            (
                negative.data[:0:-1, int(distance_index)],
                positive.data[:, int(distance_index)],
            )
        )
        if np.any(~np.isfinite(row)):
            raise ValueError("NNWS input CCFs must be finite")
        rows.append(row.astype(np.float64, copy=False))

    assert reference_positive is not None
    positive_lag = np.asarray(reference_positive.lag_time_s, dtype=np.float64)
    full_lag = np.concatenate((-positive_lag[:0:-1], positive_lag))
    return (
        np.stack(rows, axis=0),
        full_lag,
        float(reference_positive.distance_m[int(distance_index)]),
    )


def train_nnws_ccfs(
    ccf_matrix: np.ndarray,
    *,
    dt_s: float,
    signal_window_s: tuple[float, float],
    maximum_period_s: float,
    config: NNWSConfig | None = None,
) -> NNWSFitResult:
    """Train upstream NNWS v1.0 on one set of short-window full-lag CCFs."""
    matrix = np.asarray(ccf_matrix, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[1] < 3
        or matrix.shape[1] % 2 == 0
    ):
        raise ValueError("ccf_matrix must have shape [M >= 2, odd N >= 3]")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("ccf_matrix must contain only finite values")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    tmin, tmax = map(float, signal_window_s)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or not 0 <= tmin < tmax:
        raise ValueError("signal_window_s must satisfy 0 <= tmin < tmax")
    if not np.isfinite(maximum_period_s) or maximum_period_s <= 0:
        raise ValueError("maximum_period_s must be finite and positive")
    config = config or NNWSConfig()
    _validate_config(config)

    (
        UpstreamTrainingConfig,
        apply_stacking_network,
        upstream_reported_snr,
        upstream_train,
        upstream_training_snr,
    ) = _upstream_api()
    tau_s = 0.5 * maximum_period_s
    lag_s = 0.5 * (matrix.shape[1] - 1) * dt_s
    if config.std_mode == "upstream_sample":
        upstream_config = UpstreamTrainingConfig(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            snr_weight=config.snr_weight,
            symmetry_weight=config.symmetry_weight,
            std_weight=config.std_weight,
            seed=config.seed,
            device=config.device,
        )
        training = upstream_train(
            matrix,
            tau=tau_s,
            tmin=tmin,
            tmax=tmax,
            delta=dt_s,
            sampling_rate=1.0 / dt_s,
            lag=lag_s,
            config=upstream_config,
        )
        stack, weights_column = apply_stacking_network(training.model, matrix)
        weights = np.asarray(weights_column, dtype=np.float64).reshape(-1)
        history = training.history
    else:
        stack, weights, history = _train_stable_std_variant(
            matrix,
            tau_s=tau_s,
            tmin=tmin,
            tmax=tmax,
            dt_s=dt_s,
            lag_s=lag_s,
            config=config,
        )
        stack = np.asarray(stack, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise RuntimeError(
            "upstream NNWS produced non-finite or negative weights; this can "
            "occur for low-diversity CCF batches because the weight-standard-"
            "deviation loss has an undefined gradient at zero variance"
        )
    sum_weight = float(weights.sum())
    sum_weight_squared = float(np.square(weights).sum())
    if sum_weight <= np.finfo(float).eps or sum_weight_squared <= 0:
        raise RuntimeError("upstream NNWS produced an unusable zero-weight stack")
    effective_count = (
        sum_weight * sum_weight / sum_weight_squared
        if sum_weight_squared > 0
        else 0.0
    )
    reversed_stack = stack[::-1]
    anti_rms = float(np.sqrt(np.mean(np.square(stack - reversed_stack))))
    symmetric_rms = float(
        np.sqrt(np.mean(np.square(stack + reversed_stack)))
    )
    try:
        reported = float(
            upstream_reported_snr(
                stack, tau_s, tmin, tmax, dt_s, sampling_rate=1.0 / dt_s
            )
        )
    except ValueError:
        reported = float("nan")
    return NNWSFitResult(
        stack=np.asarray(stack, dtype=np.float64),
        weights=weights,
        effective_window_count=effective_count,
        training_snr=float(
            upstream_training_snr(
                stack, tau_s, tmin, tmax, dt_s, sampling_rate=1.0 / dt_s
            )
        ),
        reported_snr=reported,
        symmetry_ratio=anti_rms / max(symmetric_rms, np.finfo(float).eps),
        history=history,
        signal_window_s=(tmin, tmax),
        maximum_period_s=float(maximum_period_s),
    )


def apply_nnws_weights_to_windows(
    windows: Sequence[Mapping[str, NCFBranches]],
    weights: np.ndarray,
    *,
    strategy_name: str = "nnws_array_proxy",
    minimum_effective_windows: float = 1.0,
) -> StrategyNCFStack:
    """Apply one scalar weight per cached window to all NCF components."""
    weight_vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(windows) != weight_vector.size:
        raise ValueError("one NNWS weight is required for every NCF window")
    accumulator = MultiStrategyNCFAccumulator(
        (strategy_name,), minimum_effective_windows=minimum_effective_windows
    )
    for component_branches, weight in zip(windows, weight_vector):
        accumulator.add_window(
            component_branches, {strategy_name: float(weight)}
        )
    return accumulator.finalize()[strategy_name]


def train_nnws_window_strategy(
    windows: Sequence[Mapping[str, NCFBranches]],
    *,
    component: str = "zz",
    distance_index: int,
    dt_s: float,
    signal_window_s: tuple[float, float],
    maximum_period_s: float,
    config: NNWSConfig | None = None,
    minimum_effective_windows: float = 1.0,
) -> NNWSWindowStrategyResult:
    """Learn at one distance bin and apply the weights to the full NCF gather.

    This function is the minimal integration needed for controlled WaveDispX
    experiments.  It is not the paper's exact pairwise application and must
    retain the returned ``scope`` label in saved artifacts.
    """
    matrix, lag_time_s, distance_m = ccf_matrix_from_branches(
        windows, component=component, distance_index=distance_index
    )
    if lag_time_s.size > 1 and not np.allclose(
        np.diff(lag_time_s), dt_s, rtol=0, atol=max(1e-12, dt_s * 1e-9)
    ):
        raise ValueError("NCF lag sampling does not match dt_s")
    fit = train_nnws_ccfs(
        matrix,
        dt_s=dt_s,
        signal_window_s=signal_window_s,
        maximum_period_s=maximum_period_s,
        config=config,
    )
    stack = apply_nnws_weights_to_windows(
        windows,
        fit.weights,
        minimum_effective_windows=minimum_effective_windows,
    )
    return NNWSWindowStrategyResult(
        fit=fit,
        strategy_stack=stack,
        calibration_component=component,
        calibration_distance_index=int(distance_index),
        calibration_distance_m=distance_m,
    )


def _validate_pair_windows(windows: Sequence[NCFPairBranches]) -> NCFPairBranches:
    if not windows:
        raise ValueError("at least one pair-resolved NCF window is required")
    template = windows[0]
    for window_index, window in enumerate(windows):
        for branch_name in ("positive", "negative", "symmetric"):
            actual = getattr(window, branch_name)
            expected = getattr(template, branch_name)
            if (
                actual.data.shape != expected.data.shape
                or not np.array_equal(actual.lag_time_s, expected.lag_time_s)
                or not np.array_equal(actual.pair_i, expected.pair_i)
                or not np.array_equal(actual.pair_j, expected.pair_j)
                or not np.array_equal(actual.distance_m, expected.distance_m)
            ):
                raise ValueError(
                    f"pair-resolved NCF geometry changed at window {window_index}"
                )
            if np.any(~np.isfinite(actual.data)):
                raise ValueError("pair-resolved NCF windows must be finite")
    return template


def pair_nnws_eligibility(
    distance_m: np.ndarray,
    *,
    lag_time_s: np.ndarray,
    dt_s: float,
    minimum_velocity_mps: float,
    maximum_velocity_mps: float,
    maximum_period_s: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Classify receiver pairs using only geometry and NNWS time windows."""
    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    lag = np.asarray(lag_time_s, dtype=np.float64).reshape(-1)
    if distance.size == 0 or np.any(~np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError("pair distances must be finite and positive")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if (
        lag.size < 2
        or not np.isclose(lag[0], 0.0, rtol=0, atol=1e-12)
        or not np.allclose(
            np.diff(lag), dt_s, rtol=0, atol=max(1e-12, dt_s * 1e-9)
        )
    ):
        raise ValueError("lag_time_s must be a uniform non-negative dt_s grid")
    if (
        not np.isfinite(minimum_velocity_mps)
        or not np.isfinite(maximum_velocity_mps)
        or minimum_velocity_mps <= 0
        or maximum_velocity_mps <= minimum_velocity_mps
    ):
        raise ValueError("velocity bounds must satisfy 0 < minimum < maximum")
    if not np.isfinite(maximum_period_s) or maximum_period_s <= 0:
        raise ValueError("maximum_period_s must be finite and positive")
    tau_index = int(0.5 * maximum_period_s / dt_s)
    if tau_index <= 0:
        raise ValueError("maximum_period_s must span at least two samples")

    eligible = np.ones(distance.size, dtype=bool)
    reasons = [""] * distance.size
    for pair_index, pair_distance in enumerate(distance):
        tmin_index = int((pair_distance / maximum_velocity_mps) / dt_s)
        tmax_index = int((pair_distance / minimum_velocity_mps) / dt_s)
        signal_start = tmin_index - 2 * tau_index
        signal_stop = tmax_index + 2 * tau_index
        remote_tail_start = tmax_index + 4 * tau_index
        remote_tail_stop = tmax_index + 8 * tau_index
        if signal_start <= 0:
            reasons[pair_index] = (
                "ineligible geometry: pre-signal noise window is empty"
            )
        elif signal_stop >= lag.size or signal_start >= signal_stop:
            reasons[pair_index] = (
                "ineligible geometry: post-signal noise window is outside the CCF"
            )
        elif remote_tail_start >= lag.size or remote_tail_stop > lag.size:
            reasons[pair_index] = (
                "ineligible geometry: full remote-tail noise window is outside the CCF"
            )
        if reasons[pair_index]:
            eligible[pair_index] = False
    return eligible, tuple(reasons)


def stack_pair_ncf_windows(
    windows: Sequence[NCFPairBranches],
    weights: np.ndarray,
    *,
    minimum_effective_windows: float = 1.0,
    negative_weights: np.ndarray | None = None,
) -> PairwiseNCFStack:
    """Stack each pair with optional separate positive/negative weights."""
    template = _validate_pair_windows(windows)
    positive_weight_matrix = np.asarray(weights, dtype=np.float64)
    negative_weight_matrix = (
        positive_weight_matrix
        if negative_weights is None
        else np.asarray(negative_weights, dtype=np.float64)
    )
    pair_count = template.distance_m.size
    if positive_weight_matrix.shape != (len(windows), pair_count):
        raise ValueError("weights must have shape [window, receiver_pair]")
    if negative_weight_matrix.shape != positive_weight_matrix.shape:
        raise ValueError("negative_weights must match weights")
    if (
        np.any(~np.isfinite(positive_weight_matrix))
        or np.any(positive_weight_matrix < 0)
        or np.any(~np.isfinite(negative_weight_matrix))
        or np.any(negative_weight_matrix < 0)
    ):
        raise ValueError("pair weights must be finite and non-negative")
    if (
        not np.isfinite(minimum_effective_windows)
        or minimum_effective_windows <= 0
    ):
        raise ValueError("minimum_effective_windows must be finite and positive")
    branch_weights = {
        "positive": positive_weight_matrix,
        "negative": negative_weight_matrix,
    }
    branch_sum = {
        name: values.sum(axis=0) for name, values in branch_weights.items()
    }
    branch_sum_squared = {
        name: np.square(values).sum(axis=0)
        for name, values in branch_weights.items()
    }
    branch_effective = {
        name: np.divide(
            np.square(branch_sum[name]),
            branch_sum_squared[name],
            out=np.zeros_like(branch_sum[name]),
            where=branch_sum_squared[name] > 0,
        )
        for name in branch_weights
    }
    effective = np.minimum(
        branch_effective["positive"], branch_effective["negative"]
    )
    nonzero = np.minimum(
        np.count_nonzero(positive_weight_matrix > 0, axis=0),
        np.count_nonzero(negative_weight_matrix > 0, axis=0),
    )
    threshold_tolerance = 1e-10 * max(1.0, minimum_effective_windows)
    sufficient = (branch_sum["positive"] > 0) & (
        branch_sum["negative"] > 0
    ) & (
        effective >= minimum_effective_windows - threshold_tolerance
    )
    output = {}
    branch_data = {}
    for branch_name in ("positive", "negative"):
        cube = np.stack(
            [getattr(window, branch_name).data for window in windows], axis=0
        )
        branch_weight = branch_weights[branch_name]
        weighted = np.einsum("wp,wlp->lp", branch_weight, cube, optimize=True)
        stacked = np.divide(
            weighted,
            branch_sum[branch_name][None, :],
            out=np.full_like(weighted, np.nan),
            where=branch_sum[branch_name][None, :] > 0,
        )
        stacked[:, ~sufficient] = np.nan
        branch_data[branch_name] = stacked
    branch_data["symmetric"] = 0.5 * (
        branch_data["positive"] + branch_data["negative"]
    )
    for branch_name in ("positive", "negative", "symmetric"):
        branch_template = getattr(template, branch_name)
        output[branch_name] = NCFPairGather(
            data=branch_data[branch_name],
            lag_time_s=branch_template.lag_time_s,
            pair_i=branch_template.pair_i,
            pair_j=branch_template.pair_j,
            distance_m=branch_template.distance_m,
            valid_windows=(
                int(nonzero[sufficient].min()) if np.any(sufficient) else 0
            ),
        )
    return PairwiseNCFStack(
        branches=NCFPairBranches(**output),
        statistics=PairWeightStatistics(
            weights=0.5 * (positive_weight_matrix + negative_weight_matrix),
            sum_weight=np.minimum(
                branch_sum["positive"], branch_sum["negative"]
            ),
            sum_weight_squared=np.maximum(
                branch_sum_squared["positive"],
                branch_sum_squared["negative"],
            ),
            effective_window_count=effective,
            nonzero_window_count=nonzero,
            sufficient=sufficient,
        ),
    )


def stack_pair_ncf_windows_frequency_weighted(
    windows: Sequence[NCFPairBranches],
    band_weights: np.ndarray,
    band_center_hz: np.ndarray,
    *,
    dt_s: float,
    minimum_effective_windows: float = 1.0,
    negative_band_weights: np.ndarray | None = None,
) -> PairwiseNCFStack:
    """Stack NCFs with smooth window-band weights, optionally per pair/branch."""
    template = _validate_pair_windows(windows)
    positive_weights = np.asarray(band_weights, dtype=np.float64)
    negative_weights = (
        positive_weights
        if negative_band_weights is None
        else np.asarray(negative_band_weights, dtype=np.float64)
    )
    centers = np.asarray(band_center_hz, dtype=np.float64).reshape(-1)
    pair_count = template.distance_m.size
    expected_shared = (len(windows), centers.size)
    expected_pairwise = (len(windows), centers.size, pair_count)
    if (
        positive_weights.shape not in {expected_shared, expected_pairwise}
        or negative_weights.shape != positive_weights.shape
        or centers.size < 1
        or np.any(~np.isfinite(positive_weights))
        or np.any(positive_weights < 0)
        or np.any(~np.isfinite(negative_weights))
        or np.any(negative_weights < 0)
        or np.any(~np.isfinite(centers))
        or np.any(centers <= 0)
        or np.any(np.diff(centers) <= 0)
        or not np.isfinite(dt_s)
        or dt_s <= 0
        or not np.isfinite(minimum_effective_windows)
        or minimum_effective_windows <= 0
    ):
        raise ValueError("frequency-dependent stack inputs are invalid")
    if positive_weights.ndim == 2:
        positive_weights = np.broadcast_to(
            positive_weights[:, :, None], expected_pairwise
        )
        negative_weights = np.broadcast_to(
            negative_weights[:, :, None], expected_pairwise
        )
    if (
        np.any(np.sum(positive_weights, axis=0) <= 0)
        or np.any(np.sum(negative_weights, axis=0) <= 0)
    ):
        raise ValueError(
            "every frequency band and pair must retain positive total weight"
        )
    sample_count = template.positive.data.shape[0]
    frequency = np.fft.rfftfreq(sample_count, dt_s)

    def interpolate_weights(weights: np.ndarray) -> np.ndarray:
        interpolated = np.empty(
            (len(windows), frequency.size, pair_count), dtype=np.float64
        )
        for window_index in range(len(windows)):
            for pair_index in range(pair_count):
                row = weights[window_index, :, pair_index]
                if np.any(row == 0):
                    # Zero is an intentional exclusion in hard selectors.  A
                    # linear interpolation preserves it when adjacent bands
                    # agree, whereas log interpolation cannot represent it.
                    interpolated[window_index, :, pair_index] = np.interp(
                        frequency,
                        centers,
                        row,
                        left=row[0],
                        right=row[-1],
                    )
                else:
                    log_row = np.log(row)
                    interpolated[window_index, :, pair_index] = np.exp(
                        np.interp(
                            frequency,
                            centers,
                            log_row,
                            left=log_row[0],
                            right=log_row[-1],
                        )
                    )
        mean_weight = np.mean(interpolated, axis=0, keepdims=True)
        if np.any(mean_weight <= 0):
            raise ValueError(
                "interpolated weights must retain positive mass at every frequency"
            )
        interpolated /= mean_weight
        for frequency_index in range(frequency.size):
            for pair_index in range(pair_count):
                values = interpolated[:, frequency_index, pair_index]
                effective = float(
                    np.sum(values) ** 2 / np.sum(np.square(values))
                )
                if effective >= minimum_effective_windows:
                    continue
                lower = 0.0
                upper = 1.0
                for _ in range(60):
                    fraction = 0.5 * (lower + upper)
                    candidate = 1.0 + fraction * (values - 1.0)
                    candidate_effective = float(
                        np.sum(candidate) ** 2
                        / np.sum(np.square(candidate))
                    )
                    if candidate_effective >= minimum_effective_windows:
                        lower = fraction
                    else:
                        upper = fraction
                interpolated[:, frequency_index, pair_index] = (
                    1.0 + lower * (values - 1.0)
                )
        return interpolated

    branch_frequency_weights = {
        "positive": interpolate_weights(positive_weights),
        "negative": interpolate_weights(negative_weights),
    }
    branch_sum_weights = {
        name: np.sum(weights, axis=0)
        for name, weights in branch_frequency_weights.items()
    }
    branch_frequency_effective = {
        name: np.square(branch_sum_weights[name])
        / np.sum(np.square(weights), axis=0)
        for name, weights in branch_frequency_weights.items()
    }
    conservative_effective = np.minimum(
        np.min(branch_frequency_effective["positive"], axis=0),
        np.min(branch_frequency_effective["negative"], axis=0),
    )
    branch_frequency_nonzero = {
        name: np.count_nonzero(weights > 0, axis=0)
        for name, weights in branch_frequency_weights.items()
    }
    conservative_nonzero = np.minimum(
        np.min(branch_frequency_nonzero["positive"], axis=0),
        np.min(branch_frequency_nonzero["negative"], axis=0),
    ).astype(np.int64)
    threshold_tolerance = 1e-10 * max(1.0, minimum_effective_windows)
    sufficient = conservative_effective >= (
        minimum_effective_windows - threshold_tolerance
    )

    branch_data: dict[str, np.ndarray] = {}
    for branch_name in ("positive", "negative"):
        frequency_weights = branch_frequency_weights[branch_name]
        sum_weight = branch_sum_weights[branch_name]
        cube = np.stack(
            [getattr(window, branch_name).data for window in windows], axis=0
        )
        spectrum = np.fft.rfft(cube, axis=1)
        weighted_spectrum = np.sum(
            frequency_weights * spectrum, axis=0
        ) / sum_weight
        data = np.fft.irfft(weighted_spectrum, n=sample_count, axis=0)
        data[:, ~sufficient] = np.nan
        branch_data[branch_name] = data
    branch_data["symmetric"] = 0.5 * (
        branch_data["positive"] + branch_data["negative"]
    )
    output = {}
    valid_windows = (
        int(np.min(conservative_nonzero[sufficient]))
        if np.any(sufficient)
        else 0
    )
    for branch_name in ("positive", "negative", "symmetric"):
        branch_template = getattr(template, branch_name)
        output[branch_name] = NCFPairGather(
            data=branch_data[branch_name],
            lag_time_s=branch_template.lag_time_s,
            pair_i=branch_template.pair_i,
            pair_j=branch_template.pair_j,
            distance_m=branch_template.distance_m,
            valid_windows=valid_windows,
        )
    audit_weight = 0.5 * (
        np.mean(branch_frequency_weights["positive"], axis=1)
        + np.mean(branch_frequency_weights["negative"], axis=1)
    )
    return PairwiseNCFStack(
        branches=NCFPairBranches(**output),
        statistics=PairWeightStatistics(
            weights=audit_weight,
            sum_weight=np.full(pair_count, float(len(windows))),
            sum_weight_squared=np.divide(
                float(len(windows) ** 2),
                conservative_effective,
                out=np.full(pair_count, np.inf),
                where=conservative_effective > 0,
            ),
            effective_window_count=conservative_effective,
            nonzero_window_count=conservative_nonzero,
            sufficient=sufficient,
        ),
    )


def train_pairwise_nnws_strategy(
    windows: Sequence[NCFPairBranches],
    *,
    dt_s: float,
    distance_bin_m: float,
    minimum_velocity_mps: float,
    maximum_velocity_mps: float,
    maximum_period_s: float,
    config: NNWSConfig | None = None,
    minimum_effective_windows: float = 1.0,
    random_control_seed: int = 271828,
) -> PairwiseNNWSResult:
    """Train NNWS independently for every identified receiver pair."""
    template = _validate_pair_windows(windows)
    if (
        isinstance(random_control_seed, bool)
        or int(random_control_seed) != random_control_seed
        or int(random_control_seed) < 0
    ):
        raise ValueError("random_control_seed must be a non-negative integer")
    config = config or NNWSConfig()
    pair_count = template.distance_m.size
    window_count = len(windows)
    eligible, eligibility_reason = pair_nnws_eligibility(
        template.distance_m,
        lag_time_s=template.lag_time_s,
        dt_s=dt_s,
        minimum_velocity_mps=minimum_velocity_mps,
        maximum_velocity_mps=maximum_velocity_mps,
        maximum_period_s=maximum_period_s,
    )
    weights = np.zeros((window_count, pair_count), dtype=np.float64)
    training_snr = np.full(pair_count, np.nan, dtype=np.float64)
    reported_snr = np.full(pair_count, np.nan, dtype=np.float64)
    symmetry_ratio = np.full(pair_count, np.nan, dtype=np.float64)
    final_loss = np.full(pair_count, np.nan, dtype=np.float64)
    failure_reason = [""] * pair_count
    for pair_index, distance_m in enumerate(template.distance_m):
        if not eligible[pair_index]:
            continue
        matrix = np.stack(
            [
                np.concatenate(
                    (
                        window.negative.data[:0:-1, pair_index],
                        window.positive.data[:, pair_index],
                    )
                )
                for window in windows
            ],
            axis=0,
        )
        signal_window = (
            float(distance_m / maximum_velocity_mps),
            float(distance_m / minimum_velocity_mps),
        )
        try:
            fit = train_nnws_ccfs(
                matrix,
                dt_s=dt_s,
                signal_window_s=signal_window,
                maximum_period_s=maximum_period_s,
                config=config,
            )
        except NNWSDependencyError:
            raise
        except (ValueError, RuntimeError) as exc:
            failure_reason[pair_index] = str(exc)
            continue
        weights[:, pair_index] = fit.weights
        training_snr[pair_index] = fit.training_snr
        reported_snr[pair_index] = fit.reported_snr
        symmetry_ratio[pair_index] = fit.symmetry_ratio
        final_loss[pair_index] = fit.history["total_loss"][-1]
    pair_stack = stack_pair_ncf_windows(
        windows,
        weights,
        minimum_effective_windows=minimum_effective_windows,
    )
    for pair_index, sufficient in enumerate(pair_stack.statistics.sufficient):
        if eligible[pair_index] and not sufficient and not failure_reason[pair_index]:
            failure_reason[pair_index] = (
                "effective_window_count "
                f"{pair_stack.statistics.effective_window_count[pair_index]:.6g} "
                "is below minimum_effective_windows "
                f"{minimum_effective_windows:.6g}"
            )
    successful = eligible & pair_stack.statistics.sufficient
    common_linear_weights = np.zeros_like(weights)
    common_linear_weights[:, successful] = 1.0
    common_linear_stack = stack_pair_ncf_windows(
        windows,
        common_linear_weights,
        minimum_effective_windows=minimum_effective_windows,
    )
    random_control_weights = np.zeros_like(weights)
    rng = np.random.default_rng(int(random_control_seed))
    for pair_index in np.flatnonzero(successful):
        random_control_weights[:, pair_index] = rng.permutation(
            weights[:, pair_index]
        )
    random_control_stack = stack_pair_ncf_windows(
        windows,
        random_control_weights,
        minimum_effective_windows=minimum_effective_windows,
    )
    # NNWS cannot define its pre-signal loss window for some short-offset
    # pairs.  A dispersion image made from only eligible far offsets can have
    # inadequate spatial aperture.  Preserve those ineligible pairs with an
    # explicit unit-weight linear fallback; do the same for the permuted-
    # weight control.  This does not alter any learned upstream NNWS weight.
    hybrid_weights = np.ones_like(weights)
    hybrid_weights[:, successful] = weights[:, successful]
    hybrid_full_aperture_stack = stack_pair_ncf_windows(
        windows,
        hybrid_weights,
        minimum_effective_windows=minimum_effective_windows,
    )
    hybrid_random_weights = np.ones_like(weights)
    hybrid_random_weights[:, successful] = random_control_weights[:, successful]
    hybrid_random_control_stack = stack_pair_ncf_windows(
        windows,
        hybrid_random_weights,
        minimum_effective_windows=minimum_effective_windows,
    )
    return PairwiseNNWSResult(
        pair_stack=pair_stack,
        binned_branches=bin_pair_ncf_branches(
            pair_stack.branches, distance_bin_m=distance_bin_m
        ),
        common_linear_stack=common_linear_stack,
        common_linear_binned=bin_pair_ncf_branches(
            common_linear_stack.branches, distance_bin_m=distance_bin_m
        ),
        random_control_stack=random_control_stack,
        random_control_binned=bin_pair_ncf_branches(
            random_control_stack.branches, distance_bin_m=distance_bin_m
        ),
        hybrid_full_aperture_stack=hybrid_full_aperture_stack,
        hybrid_full_aperture_binned=bin_pair_ncf_branches(
            hybrid_full_aperture_stack.branches, distance_bin_m=distance_bin_m
        ),
        hybrid_random_control_stack=hybrid_random_control_stack,
        hybrid_random_control_binned=bin_pair_ncf_branches(
            hybrid_random_control_stack.branches, distance_bin_m=distance_bin_m
        ),
        eligible=eligible,
        eligibility_reason=eligibility_reason,
        training_snr=training_snr,
        reported_snr=reported_snr,
        symmetry_ratio=symmetry_ratio,
        final_loss=final_loss,
        failure_reason=tuple(failure_reason),
    )
