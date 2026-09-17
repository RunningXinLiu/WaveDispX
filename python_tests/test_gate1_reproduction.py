from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_gate1_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_nnws_gate1_upstream_reproduction.py"
    )
    spec = importlib.util.spec_from_file_location("gate1_upstream_reproduction", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_segment_mapping_preserves_strict_upstream_boundaries():
    module = _load_gate1_script()
    activation = np.array((0.0, 0.1, 999.9, 1000.0, 1000.1, 1999.9, 2000.0))
    segment = module._source_segment_index(
        activation, segment_span_s=1000.0, segment_count=2
    )
    np.testing.assert_array_equal(segment, (-1, 0, 0, -1, 1, 1, -1))


def test_stackmaster_selective_reconstruction_rejects_anticorrelated_trace():
    module = _load_gate1_script()
    pulse = np.sin(np.linspace(0.0, 4.0 * np.pi, 1001))
    matrix = np.vstack((pulse, 1.2 * pulse, -pulse))
    stack, selected, steps = module._selective_stack_stackmaster(matrix)
    np.testing.assert_array_equal(selected, (True, True, False))
    np.testing.assert_allclose(stack, 1.1 * pulse)
    assert 1 <= steps <= 11


def test_station_axis_distance_treats_both_endfire_directions_as_stationary():
    module = _load_gate1_script()
    distance = module._axis_distance_deg(
        np.radians(np.array((0.0, 8.0, 90.0, 172.0, 180.0, 352.0)))
    )
    np.testing.assert_allclose(distance, (0.0, 8.0, 90.0, 8.0, 0.0, 8.0))
