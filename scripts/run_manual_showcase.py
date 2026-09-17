"""Reproducible manual examples; no smoothing or theory-guided masking.

Run from project root with PYTHONPATH=python_src conda run -n seisloc python
scripts/run_manual_showcase.py. All runs retain original two-layer physics.
The passive comparison changes duration only (fixed source field and seed).
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.app_service import load_template_mapping, apply_automatic_parameters
from wavedispx.workflow import WaveDispXWorkflow
from wavedispx.config import config_from_mapping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_python/manual_showcase_20260917"
ASSETS = ROOT / "docs/manual_assets"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for mode in ("active", "passive"):
        mapping, diagnostic = apply_automatic_parameters(
            load_template_mapping(ROOT / f"examples/{mode}_two_layer.toml"),
            base_directory=ROOT / "examples",
        )
        mapping["run"]["workers"] = 4
        # Use the computed, editable recommendations. Keep same imaging grid
        # throughout each duration comparison, not a narrower display crop.
        workflow = None
        images = []
        for duration in ((None,) if mode == "active" else (60, 600, 1800, 3600)):
            case = deepcopy(mapping)
            name = f"{mode}_{duration or 'auto'}"
            case["run"].update(name=name, output_directory=str(OUT / name))
            if duration is not None:
                case["noise"]["duration_s"] = duration
            if workflow is None:
                workflow = WaveDispXWorkflow.from_mapping(case, base_directory=ROOT)
            else:
                workflow.configure(config_from_mapping(case, base_directory=ROOT), source_mapping=case)
            print(f"RUN {name}", flush=True)
            workflow.run_all()
            path = OUT / name
            manifest = json.loads((path / "manifest.json").read_text())
            records[name] = {
                "processing": case["processing"],
                "timing": case.get("noise", case.get("active_source")),
                "quality": manifest["quality"],
                "results_sha256": hashlib.sha256((path / "results.h5").read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256((path / "input.toml").read_bytes()).hexdigest(),
            }
            if mode == "passive":
                images.append((duration, workflow.passive_analysis.dispersion))
            if duration is None or duration == 3600:
                shutil.copy2(path / "summary.png", ASSETS / f"showcase-{mode}-summary.png")
                shutil.copy2(path / "input.toml", ASSETS / f"showcase-{mode}-input.toml")
        if images:
            fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True, layout="constrained")
            final = images[-1][1].energy
            for ax, (duration, image) in zip(axes.flat, images):
                artist = ax.pcolormesh(image.frequency_hz, image.velocity_mps,
                                      image.energy, shading="nearest", cmap="turbo", vmin=0, vmax=1)
                ax.set(title=f"{duration} s | {duration // 20} independent windows",
                       xlabel="Frequency (Hz)", ylabel="Phase velocity (m/s)")
                records[f"passive_{duration}"]["image_relative_l2_to_3600"] = float(
                    np.linalg.norm(image.energy-final) / np.linalg.norm(final))
            fig.colorbar(artist, ax=axes, label="Per-frequency normalized image (not absolute power)")
            fig.suptitle("Same model / 384 sources / 600 m line / fixed seed / 20 s windows\n"
                         "Mixed finite records -> spectral whitening NCF -> phase shift; no smoothing")
            fig.savefig(ASSETS / "showcase-duration-comparison.png", dpi=160)
            plt.close(fig)
    (OUT / "report.json").write_text(json.dumps(records, indent=2, allow_nan=True))
    shutil.copy2(OUT / "report.json", ASSETS / "showcase-report.json")
    print(json.dumps({key: value.get("image_relative_l2_to_3600") for key, value in records.items()}), flush=True)


if __name__ == "__main__":
    main()
