from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest

from wavedispx.config import Layer, ModelConfig, load_config
from wavedispx.processing import (
    find_qedisp_forward,
    find_qedisp_group_library,
    run_qedisp_forward,
    theoretical_dispersion,
)

ROOT = Path(__file__).resolve().parents[1]


def _require_qedisp() -> None:
    try:
        find_qedisp_forward()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _require_qedisp_group() -> None:
    _require_qedisp()
    try:
        find_qedisp_group_library()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_qedisp_matches_reference_two_layer_fundamental_and_returns_kernels() -> None:
    _require_qedisp()
    config = load_config(ROOT / "examples/active_broadband_2_12hz_validation.toml")
    frequency = np.linspace(2.0, 12.0, 21)

    qedisp = theoretical_dispersion(
        config.model,
        frequency,
        rayleigh_modes=3,
        love_modes=1,
        backend="qedisp",
    )
    disba = theoretical_dispersion(
        config.model,
        frequency,
        rayleigh_modes=3,
        love_modes=1,
        backend="disba",
    )
    detailed = run_qedisp_forward(
        config.model,
        np.array([2.0, 6.0, 10.0]),
        rayleigh_modes=1,
        love_modes=1,
        compute_kernels=True,
    )

    np.testing.assert_allclose(
        qedisp.rayleigh_mps[:, 0], disba.rayleigh_mps[:, 0], atol=0.001, rtol=0
    )
    assert qedisp.solver == "QEDispInv/quadratic-extrema"
    assert detailed.rayleigh_kernels is not None
    assert detailed.love_kernels is not None
    assert detailed.rayleigh_kernels.vs.shape == (2, 3)
    assert detailed.love_kernels.vs.shape == (2, 3)


def test_qedisp_native_group_velocity_is_returned_without_phase_differentiation() -> None:
    _require_qedisp_group()
    config = load_config(ROOT / "examples/active_broadband_2_12hz_validation.toml")
    result = run_qedisp_forward(
        config.model,
        np.array([2.0, 6.0, 10.0]),
        rayleigh_modes=1,
        love_modes=1,
        compute_group=True,
    )

    assert result.rayleigh_group_mps is not None
    assert result.love_group_mps is not None
    np.testing.assert_allclose(
        result.rayleigh_group_mps[:, 0],
        np.array([323.0689, 159.4126, 194.3962]),
        atol=0.2,
        rtol=0.0,
    )
    assert np.all(np.isfinite(result.love_group_mps[:, 0]))
    assert np.all(result.love_group_mps[:, 0] > 0.0)


def test_qedisp_resolves_at_least_the_disba_roots_in_upstream_low_velocity_model() -> None:
    _require_qedisp()
    model = ModelConfig(
        (
            Layer(10.0, 1500.0, 180.0, 1780.0),
            Layer(10.0, 1700.0, 350.0, 1850.0),
            Layer(20.0, 1600.0, 250.0, 1800.0),
            Layer(float("inf"), 2000.0, 600.0, 1930.0),
        )
    )
    frequency = np.geomspace(0.1, 100.0, 500)

    qedisp = theoretical_dispersion(
        model, frequency, rayleigh_modes=8, love_modes=8, backend="qedisp"
    )
    disba = theoretical_dispersion(
        model, frequency, rayleigh_modes=8, love_modes=8, backend="disba"
    )

    qedisp_root_count = int(
        np.count_nonzero(np.isfinite(qedisp.rayleigh_mps))
        + np.count_nonzero(np.isfinite(qedisp.love_mps))
    )
    disba_root_count = int(
        np.count_nonzero(np.isfinite(disba.rayleigh_mps))
        + np.count_nonzero(np.isfinite(disba.love_mps))
    )
    assert qedisp_root_count >= disba_root_count
    assert qedisp_root_count > disba_root_count


def test_qedisp_rejects_frequency_collisions_at_output_precision() -> None:
    model = ModelConfig((Layer(float("inf"), 600.0, 300.0, 1900.0),))
    with pytest.raises(ValueError, match="5-decimal"):
        run_qedisp_forward(
            model,
            np.array([2.000001, 2.000002]),
            rayleigh_modes=1,
            love_modes=0,
        )
