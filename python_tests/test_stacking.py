import numpy as np

from wavedispx.processing import NCFBranches, NCFGather
from wavedispx.stacking import (
    MultiStrategyNCFAccumulator,
    reference_amplitude_candidate_phase,
)


def _branches(value: float) -> dict[str, NCFBranches]:
    lag = np.array([0.0, 0.1])
    distance = np.array([5.0])
    pair_count = np.array([2])

    def gather(scale: float) -> NCFGather:
        return NCFGather(
            np.array([[value * scale], [2.0 * value * scale]]),
            lag,
            distance,
            pair_count,
            valid_windows=1,
        )

    return {
        "zz": NCFBranches(
            positive=gather(1.0),
            negative=gather(3.0),
            symmetric=gather(2.0),
        )
    }


def test_multi_strategy_accumulator_uses_each_actual_weight_sum():
    accumulator = MultiStrategyNCFAccumulator(
        ("baseline_linear", "oracle_soft_weight")
    )
    accumulator.add_window(
        _branches(1.0),
        {"baseline_linear": 1.0, "oracle_soft_weight": 1.0},
    )
    accumulator.add_window(
        _branches(3.0),
        {"baseline_linear": 1.0, "oracle_soft_weight": 0.5},
    )

    result = accumulator.finalize()
    baseline = result["baseline_linear"]
    weighted = result["oracle_soft_weight"]

    np.testing.assert_allclose(baseline.component_branches["zz"].positive.data, [[2.0], [4.0]])
    np.testing.assert_allclose(weighted.component_branches["zz"].positive.data, [[5.0 / 3.0], [10.0 / 3.0]])
    np.testing.assert_allclose(weighted.component_branches["zz"].negative.data, [[5.0], [10.0]])
    assert baseline.statistics.sum_weight == 2.0
    assert weighted.statistics.sum_weight == 1.5
    assert weighted.statistics.sum_weight_squared == 1.25
    assert weighted.statistics.effective_window_count == 1.8
    assert weighted.statistics.nonzero_window_count == 2
    assert weighted.statistics.sufficient


def test_zero_weight_strategy_is_insufficient_and_never_falls_back():
    accumulator = MultiStrategyNCFAccumulator(
        ("baseline_linear", "empty_mask"), minimum_effective_windows=2.0
    )
    accumulator.add_window(
        _branches(4.0),
        {"baseline_linear": 1.0, "empty_mask": 0.0},
    )

    result = accumulator.finalize()
    empty = result["empty_mask"]
    assert not empty.statistics.sufficient
    assert empty.statistics.sum_weight == 0.0
    assert empty.statistics.nonzero_window_count == 0
    assert np.isnan(empty.component_branches["zz"].symmetric.data).all()
    assert not result["baseline_linear"].statistics.sufficient
    np.testing.assert_allclose(
        result["baseline_linear"].component_branches["zz"].symmetric.data,
        [[8.0], [16.0]],
    )


def test_accumulator_rejects_negative_or_missing_strategy_weights():
    accumulator = MultiStrategyNCFAccumulator(("baseline_linear", "candidate"))
    with np.testing.assert_raises_regex(ValueError, "exactly match"):
        accumulator.add_window(_branches(1.0), {"baseline_linear": 1.0})
    with np.testing.assert_raises_regex(ValueError, "finite and non-negative"):
        accumulator.add_window(
            _branches(1.0),
            {"baseline_linear": 1.0, "candidate": -0.1},
        )


def test_phase_hybrid_preserves_reference_amplitude_and_candidate_phase():
    reference = _branches(1.0)["zz"]
    candidate = _branches(-2.0)["zz"]
    hybrid = reference_amplitude_candidate_phase(reference, candidate)
    for branch_name in ("positive", "negative"):
        reference_spectrum = np.fft.rfft(
            getattr(reference, branch_name).data, axis=0
        )
        candidate_spectrum = np.fft.rfft(
            getattr(candidate, branch_name).data, axis=0
        )
        hybrid_spectrum = np.fft.rfft(
            getattr(hybrid, branch_name).data, axis=0
        )
        np.testing.assert_allclose(
            np.abs(hybrid_spectrum), np.abs(reference_spectrum), atol=1e-12
        )
        np.testing.assert_allclose(
            np.angle(hybrid_spectrum), np.angle(candidate_spectrum), atol=1e-12
        )
    np.testing.assert_allclose(
        hybrid.symmetric.data,
        0.5 * (hybrid.positive.data + hybrid.negative.data),
    )
