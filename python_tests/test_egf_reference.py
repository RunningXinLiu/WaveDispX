from dataclasses import replace
from pathlib import Path

import numpy as np

from wavedispx.config import ReceiverLineConfig, load_config
from wavedispx.egf_reference import build_passive_egf_reference
from wavedispx.geometry import build_geometry


ROOT = Path(__file__).resolve().parents[1]


def test_passive_reference_exposes_unique_interreceiver_green_functions():
    base = load_config(ROOT / "examples/passive_quick.toml")
    config = replace(
        base,
        receivers=ReceiverLineConfig(0.0, 15.0, 5.0, 0.0),
    )
    geometry = build_geometry(config.receivers, config.ambient_sources)

    reference = build_passive_egf_reference(config, geometry)

    np.testing.assert_array_equal(reference.distance_m, [5.0, 10.0, 15.0])
    np.testing.assert_array_equal(reference.pair_i, [0, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(reference.pair_j, [1, 2, 3, 2, 3, 3])
    np.testing.assert_array_equal(reference.pair_distance_index, [0, 1, 2, 0, 1, 0])
    assert reference.component_names == ("zz",)
    assert reference.components.shape[1:] == (3, 1)
    assert reference.components.shape[0] == reference.time_s.size
    np.testing.assert_allclose(np.diff(reference.time_s), config.noise.dt_s)
    assert np.all(np.isfinite(reference.components))
    assert np.any(reference.components != 0.0)
    np.testing.assert_array_equal(reference.components[0], 0.0)
    assert reference.source_time_function == "unit_impulse"
    assert reference.calibration_operator == "negative_time_derivative_of_ncf"


def test_three_component_reference_preserves_reciprocal_cross_component_sign():
    base = load_config(ROOT / "examples/passive_quick.toml")
    config = replace(
        base,
        receivers=ReceiverLineConfig(0.0, 15.0, 5.0, 0.0),
        ambient_sources=replace(
            base.ambient_sources,
            component="three_component",
        ),
    )
    geometry = build_geometry(config.receivers, config.ambient_sources)

    reference = build_passive_egf_reference(config, geometry)

    assert reference.component_names == ("rr", "tt", "zr", "rz", "zz")
    assert reference.components.shape[1:] == (3, 5)
    zr = reference.components[:, :, reference.component_names.index("zr")]
    rz = reference.components[:, :, reference.component_names.index("rz")]
    np.testing.assert_allclose(zr, -rz, rtol=1e-12, atol=1e-24)
