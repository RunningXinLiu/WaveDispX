"""Small Mac/Linux CPU consistency check; not independent physics validation."""
import json
from pathlib import Path
import numpy as np
from wavedispx.app_service import load_template_mapping
from wavedispx.workflow import WaveDispXWorkflow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/studio/acceptance_20261028"
mapping = load_template_mapping(ROOT / "examples/active_quick.toml")
mapping["processing"]["theory_solver"] = "qedisp"
w = WaveDispXWorkflow.from_mapping(mapping, base_directory=ROOT / "examples")
image = w.compute_dispersion().rayleigh_combined
with np.load(OUT / "active_linux_image.npz", allow_pickle=False) as linux:
    np.testing.assert_allclose(image.frequency_hz, linux["f"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(image.velocity_mps, linux["c"], rtol=0, atol=1e-12)
    error_image = float(np.linalg.norm(image.energy-linux["energy"])/np.linalg.norm(linux["energy"]))
with np.load(OUT / "active_linux_waveforms.npz", allow_pickle=False) as linux:
    np.testing.assert_allclose(w.shot.time_s, linux["time"], rtol=0, atol=1e-12)
    error_wave = float(np.linalg.norm(w.shot.vertical_vertical-linux["zz"])/np.linalg.norm(linux["zz"]))
report = {"case": "unchanged active_quick, CPU, QED overlay", "energy_relative_l2": error_image,
          "zz_waveform_relative_l2": error_wave, "thresholds": {"energy": 1e-6, "waveform": 1e-5},
          "note": "Linux waveform inspection reloaded from float32 HDF5; same algorithm, not independent solver evidence"}
(OUT / "platform_check.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert error_image < 1e-6 and error_wave < 1e-5
