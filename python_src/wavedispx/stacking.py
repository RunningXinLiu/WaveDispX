"""Auditable streaming accumulation of multiple weighted NCF strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .processing import NCFBranches, NCFGather


_BRANCH_NAMES = ("positive", "negative", "symmetric")


@dataclass(frozen=True)
class StackStatistics:
    windows_seen: int
    nonzero_window_count: int
    sum_weight: float
    sum_weight_squared: float
    effective_window_count: float
    sufficient: bool


@dataclass(frozen=True)
class StrategyNCFStack:
    component_branches: dict[str, NCFBranches]
    statistics: StackStatistics


def reference_amplitude_candidate_phase(
    amplitude_reference: NCFBranches, phase_candidate: NCFBranches
) -> NCFBranches:
    """Preserve the full-window amplitude spectrum and use candidate phase."""
    output: dict[str, NCFGather] = {}
    branch_data: dict[str, np.ndarray] = {}
    for branch_name in ("positive", "negative"):
        reference = getattr(amplitude_reference, branch_name)
        candidate = getattr(phase_candidate, branch_name)
        if reference.data.shape != candidate.data.shape:
            raise ValueError("reference and candidate NCF shapes must match")
        for field in ("lag_time_s", "distance_m", "pair_count"):
            if not np.array_equal(getattr(reference, field), getattr(candidate, field)):
                raise ValueError(f"reference and candidate NCF {field} must match")
        reference_spectrum = np.fft.rfft(reference.data, axis=0)
        candidate_spectrum = np.fft.rfft(candidate.data, axis=0)
        candidate_amplitude = np.abs(candidate_spectrum)
        phase_unit = np.divide(
            candidate_spectrum,
            candidate_amplitude,
            out=np.divide(
                reference_spectrum,
                np.maximum(np.abs(reference_spectrum), np.finfo(float).eps),
            ),
            where=candidate_amplitude > np.finfo(float).eps,
        )
        branch_data[branch_name] = np.fft.irfft(
            np.abs(reference_spectrum) * phase_unit,
            n=reference.data.shape[0],
            axis=0,
        )
    branch_data["symmetric"] = 0.5 * (
        branch_data["positive"] + branch_data["negative"]
    )
    for branch_name in ("positive", "negative", "symmetric"):
        reference = getattr(amplitude_reference, branch_name)
        output[branch_name] = NCFGather(
            data=branch_data[branch_name],
            lag_time_s=reference.lag_time_s,
            distance_m=reference.distance_m,
            pair_count=reference.pair_count,
            valid_windows=phase_candidate.positive.valid_windows,
        )
    return NCFBranches(**output)


class MultiStrategyNCFAccumulator:
    """Accumulate the same short-window NCF into several weighted strategies.

    Weights must be finite and non-negative.  Every strategy uses its actual
    weight sum as the denominator.  A zero-weight result is represented by NaN
    gathers and an explicit insufficient flag; it never falls back to another
    strategy or to the all-window stack.
    """

    def __init__(
        self,
        strategy_names: tuple[str, ...] | list[str],
        *,
        minimum_effective_windows: float = 1.0,
    ) -> None:
        names = tuple(strategy_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("strategy_names must be non-empty and unique")
        if (
            not np.isfinite(minimum_effective_windows)
            or minimum_effective_windows <= 0
        ):
            raise ValueError("minimum_effective_windows must be finite and positive")
        self.strategy_names = names
        self.minimum_effective_windows = float(minimum_effective_windows)
        self._templates: dict[str, NCFBranches] | None = None
        self._sums: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self._sum_weight = {name: 0.0 for name in names}
        self._sum_weight_squared = {name: 0.0 for name in names}
        self._nonzero_window_count = {name: 0 for name in names}
        self._windows_seen = 0

    @staticmethod
    def _assert_gather_compatible(actual: NCFGather, expected: NCFGather) -> None:
        if actual.data.shape != expected.data.shape:
            raise ValueError("NCF gather shapes changed during accumulation")
        for field in ("lag_time_s", "distance_m", "pair_count"):
            if not np.array_equal(getattr(actual, field), getattr(expected, field)):
                raise ValueError(f"NCF {field} changed during accumulation")

    def _initialize_or_validate(
        self, component_branches: Mapping[str, NCFBranches]
    ) -> None:
        if not component_branches:
            raise ValueError("component_branches may not be empty")
        if self._templates is None:
            self._templates = dict(component_branches)
            self._sums = {
                strategy: {
                    component: {
                        branch_name: np.zeros_like(
                            getattr(branches, branch_name).data,
                            dtype=np.float64,
                        )
                        for branch_name in _BRANCH_NAMES
                    }
                    for component, branches in component_branches.items()
                }
                for strategy in self.strategy_names
            }
            return
        if set(component_branches) != set(self._templates):
            raise ValueError("NCF components changed during accumulation")
        for component, branches in component_branches.items():
            template = self._templates[component]
            for branch_name in _BRANCH_NAMES:
                self._assert_gather_compatible(
                    getattr(branches, branch_name), getattr(template, branch_name)
                )

    def add_window(
        self,
        component_branches: Mapping[str, NCFBranches],
        strategy_weights: Mapping[str, float],
    ) -> None:
        """Add one receiver-only short-window correlation to every strategy."""
        if set(strategy_weights) != set(self.strategy_names):
            raise ValueError("strategy_weights must exactly match strategy_names")
        weights = {name: float(strategy_weights[name]) for name in self.strategy_names}
        if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("strategy weights must be finite and non-negative")
        self._initialize_or_validate(component_branches)
        for strategy, weight in weights.items():
            self._sum_weight[strategy] += weight
            self._sum_weight_squared[strategy] += weight * weight
            self._nonzero_window_count[strategy] += int(weight > 0)
            if weight == 0:
                continue
            for component, branches in component_branches.items():
                for branch_name in _BRANCH_NAMES:
                    self._sums[strategy][component][branch_name] += (
                        weight * getattr(branches, branch_name).data
                    )
        self._windows_seen += 1

    def snapshot(self, strategy_name: str) -> StrategyNCFStack:
        """Return the current normalized stack for one strategy."""
        if strategy_name not in self.strategy_names:
            raise KeyError(strategy_name)
        if self._templates is None:
            raise RuntimeError("no NCF windows have been added")
        sum_weight = self._sum_weight[strategy_name]
        sum_weight_squared = self._sum_weight_squared[strategy_name]
        effective_window_count = (
            sum_weight * sum_weight / sum_weight_squared
            if sum_weight_squared > 0
            else 0.0
        )
        nonzero_count = self._nonzero_window_count[strategy_name]
        statistics = StackStatistics(
            windows_seen=self._windows_seen,
            nonzero_window_count=nonzero_count,
            sum_weight=sum_weight,
            sum_weight_squared=sum_weight_squared,
            effective_window_count=effective_window_count,
            sufficient=(
                sum_weight > 0
                and effective_window_count >= self.minimum_effective_windows
            ),
        )
        components: dict[str, NCFBranches] = {}
        for component, template_branches in self._templates.items():
            output: dict[str, NCFGather] = {}
            for branch_name in _BRANCH_NAMES:
                template = getattr(template_branches, branch_name)
                weighted_sum = self._sums[strategy_name][component][branch_name]
                data = (
                    weighted_sum / sum_weight
                    if sum_weight > 0
                    else np.full_like(weighted_sum, np.nan)
                )
                output[branch_name] = NCFGather(
                    data=data,
                    lag_time_s=template.lag_time_s,
                    distance_m=template.distance_m,
                    pair_count=template.pair_count,
                    valid_windows=nonzero_count,
                )
            components[component] = NCFBranches(**output)
        return StrategyNCFStack(components, statistics)

    def finalize(self) -> dict[str, StrategyNCFStack]:
        """Return normalized stacks and auditable statistics for all strategies."""
        return {name: self.snapshot(name) for name in self.strategy_names}
