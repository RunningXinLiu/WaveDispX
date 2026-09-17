import numpy as np

from wavedispx.pair_proxy_selection import fit_pair_crossfit_coherence_weights
from wavedispx.processing import NCFPairBranches, NCFPairGather


def _pair_window(data: np.ndarray) -> NCFPairBranches:
    lag = np.arange(data.size) * 0.1
    gather = NCFPairGather(
        data=data[:, None],
        lag_time_s=lag,
        pair_i=np.asarray((0,)),
        pair_j=np.asarray((1,)),
        distance_m=np.asarray((10.0,)),
        valid_windows=1,
    )
    return NCFPairBranches(gather, gather, gather)


def test_pair_crossfit_coherence_weights_are_pair_specific_and_auditable():
    lag = np.arange(101) * 0.1
    pulse = np.exp(-0.5 * ((lag - 1.5) / 0.15) ** 2) * np.cos(
        2.0 * np.pi * 2.0 * (lag - 1.5)
    )
    windows = [
        _pair_window(pulse),
        _pair_window(0.9 * pulse),
        _pair_window(1.1 * pulse),
        _pair_window(np.roll(pulse, 12)),
    ]
    result = fit_pair_crossfit_coherence_weights(
        windows,
        dt_s=0.1,
        frequency_range_hz=(1.0, 3.0),
        minimum_velocity_mps=5.0,
        maximum_velocity_mps=10.0,
        folds=2,
        minimum_effective_fraction=0.5,
    )
    assert result.positive_weights.shape == (4, 1)
    assert result.negative_weights.shape == (4, 1)
    assert result.fold_index.tolist() == [0, 0, 1, 1]
    assert result.positive_effective_window_count[0] >= 2.0 - 1e-10
    assert result.negative_effective_window_count[0] >= 2.0 - 1e-10
    assert np.all(np.isfinite(result.positive_score))
    assert np.all(np.isfinite(result.negative_score))
