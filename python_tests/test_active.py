from pathlib import Path

import h5py
import numpy as np

from wavedispx.active import simulate_active_shot
from wavedispx.config import ActiveSourceConfig, ReceiverLineConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_active_shot_matches_matlab_golden():
    model = load_config(ROOT / "examples/active_two_layer.toml").model
    with h5py.File(ROOT / "matlab_reference/golden_dwm_kernels.h5") as reference:
        group = reference["active_shot"]
        distance = group["distance_m"][...].reshape(-1)
        expected_components = np.transpose(group["components"][...], (2, 1, 0))
        dominant = float(group["dominant_frequency_hz"][...].reshape(-1)[0])
        dt = float(group["dt_s"][...].reshape(-1)[0])
        fmax = float(group["maximum_frequency_hz"][...].reshape(-1)[0])
    source = ActiveSourceConfig(0.0, 0.0, "both", "ricker", dominant, fmax, dt)
    regular_receivers = ReceiverLineConfig(10.0, 100.0, 30.0, 0.0)
    result = simulate_active_shot(model, source, regular_receivers, workers=2)
    np.testing.assert_allclose(result.distance_m, distance, rtol=0, atol=0)
    np.testing.assert_allclose(result.components, expected_components, rtol=2e-8, atol=2e-12)
