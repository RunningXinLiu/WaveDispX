import numpy as np

from wavedispx.temporal_reweighting import fit_temporal_antisymmetry_weights


def test_temporal_antisymmetry_downweights_asymmetric_window():
    lag = np.linspace(0.0, 2.0, 201)
    signal = np.exp(-0.5 * ((lag - 1.0) / 0.08) ** 2)
    positive = np.vstack((signal, 0.9 * signal, signal))
    negative = positive.copy()
    negative[2] += 2.0 * np.exp(-0.5 * ((lag - 0.6) / 0.05) ** 2)

    result = fit_temporal_antisymmetry_weights(
        positive,
        negative,
        lag,
        window_s=(0.0, 1.5),
    )

    np.testing.assert_allclose(np.sum(result.weights), 1.0)
    assert np.all(result.weights >= 0.0)
    assert result.weights[2] < result.weights[0]
    assert result.objective <= result.uniform_objective
    assert 1.0 <= result.effective_window_count <= 3.0


def test_temporal_antisymmetry_keeps_uniform_start_when_all_are_symmetric():
    lag = np.linspace(0.0, 1.0, 101)
    signal = np.sin(2.0 * np.pi * 5.0 * lag)
    positive = np.vstack((signal, 2.0 * signal, 0.5 * signal))
    result = fit_temporal_antisymmetry_weights(
        positive,
        positive.copy(),
        lag,
        window_s=(0.0, 1.0),
    )

    np.testing.assert_allclose(result.weights, np.full(3, 1.0 / 3.0))
    assert result.objective == 0.0
    assert result.iterations == 0


def test_temporal_antisymmetry_is_invariant_to_global_ccf_scale():
    lag = np.linspace(0.0, 2.0, 201)
    signal = np.exp(-0.5 * ((lag - 1.0) / 0.08) ** 2)
    positive = np.vstack((signal, 0.9 * signal, signal, 1.1 * signal))
    negative = positive.copy()
    negative[2] += 2.0 * np.exp(-0.5 * ((lag - 0.6) / 0.05) ** 2)
    negative[3] -= 0.5 * np.exp(-0.5 * ((lag - 0.8) / 0.06) ** 2)

    baseline = fit_temporal_antisymmetry_weights(
        positive,
        negative,
        lag,
        window_s=(0.0, 1.5),
    )
    tiny = fit_temporal_antisymmetry_weights(
        positive * 1e-7,
        negative * 1e-7,
        lag,
        window_s=(0.0, 1.5),
    )

    np.testing.assert_allclose(tiny.weights, baseline.weights, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(tiny.scaled_objective, baseline.scaled_objective)
    np.testing.assert_allclose(
        tiny.uniform_objective / baseline.uniform_objective,
        1e-14,
    )
