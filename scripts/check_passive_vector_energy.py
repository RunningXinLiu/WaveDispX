"""Explicit CPU integration cases; bulk output belongs on the school server."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wavedispx.app_service import load_template_mapping
from wavedispx.workflow import WaveDispXWorkflow
from wavedispx.experiment import ExperimentRequest, run_experiment, save_experiment
from wavedispx.preflight import plot_preflight_review
from wavedispx.workflow_plotting import plot_workflow_observations


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--template", type=Path)
    source.add_argument("--custom-directory", type=Path, help="Run explicit TOML cases without changing their physical parameters")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "finite mixed records, R/L integration; NOT convergence certification",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": {}}
    cases = []
    if args.custom_directory:
        cases = [(path.stem, load_template_mapping(path), path.parent) for path in sorted(args.custom_directory.glob("*.toml"))]
        if not cases:
            raise ValueError("custom directory contains no TOML cases")
    else:
        baseline = load_template_mapping(args.template)
        for name, count, duration, polarization, azimuth in (
            ("independent_quick", 24, 60., "independent_hv", 0.),
            ("independent_extended", 192, 600., "independent_hv", 0.),
            ("directed_extended", 192, 600., "directed_hv", 45.),
        ):
            mapping = deepcopy(baseline)
            mapping["ambient_sources"].update(count=count, polarization=polarization, horizontal_azimuth_deg=azimuth)
            mapping["noise"]["duration_s"] = duration
            cases.append((name, mapping, args.template.parent))
    for name, mapping, base in cases:
        mapping["run"].update(name=name, output_directory=str((args.output/name).resolve()))
        input_files = {}
        for section in (mapping["geometry"]["receivers"], mapping["ambient_sources"], mapping["noise"]):
            for key in ("coordinates_file", "activity_file", "psd_file"):
                if section.get(key):
                    path = (base/section[key]).resolve()
                    section[key] = str(path)
                    input_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        w = WaveDispXWorkflow.from_mapping(mapping, base_directory=base)
        count, duration = w.config.ambient_sources.count, w.config.noise.duration_s
        polarization = w.config.ambient_sources.polarization
        review = w.review_inputs()
        if not review.ready:
            raise RuntimeError("preflight failed: "+name)
        result = run_experiment(w, ExperimentRequest.from_mapping(mapping))
        assert result.hvsr is None, "HVSR must not run for energy_theory"
        assert set(result.images) == {"rayleigh", "love"}, result.statuses
        assert all(np.isfinite(im.energy).all() and np.any(im.energy > 0) for im in result.images.values())
        directory = save_experiment(result, w, mapping)
        meta = json.loads((directory/"manifest.json").read_text())
        report["cases"][name] = {"output": str(directory), "statuses": result.statuses,
                                "elapsed_s": result.elapsed_s, "manifest": meta, "input_files_sha256": input_files}
        for label, fig in (("inputs", plot_preflight_review(review)),
                           ("ncf", plot_workflow_observations(w))):
            fig.savefig(args.output/(name+"_"+label+".png"), dpi=130)
            plt.close(fig)
        for overlay in (False, True):
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
            for ax, wave in zip(axes, ("rayleigh", "love")):
                im = result.images[wave]
                mesh = ax.pcolormesh(im.frequency_hz, im.velocity_mps, im.energy,
                                     shading="nearest", cmap="turbo", vmin=0, vmax=1)
                if overlay:
                    curves = getattr(result.theory, wave+"_mps")
                    for i in range(curves.shape[1]):
                        ax.plot(result.theory.frequency_hz, curves[:, i], "w--", lw=1, label=f"QED M{i}")
                    ax.legend(fontsize=8, facecolor="#555555", labelcolor="white")
                ax.set(title=wave.capitalize(), xlabel="Frequency (Hz)", ylabel="Phase velocity (m/s)")
                fig.colorbar(mesh, ax=ax, label="Per-frequency normalized image")
            fig.suptitle(f"{name} | {polarization} | {count} sources | {duration:g} s | phase_shift / whitened NCF\n"
                         "pH=0.6; native source smoothing/taper; finite-record example, NOT convergence-certified", fontsize=11)
            fig.savefig(args.output/(name+("_energy_theory.png" if overlay else "_energy.png")), dpi=140)
            plt.close(fig)
        (args.output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(name, result.statuses, result.elapsed_s, flush=True)


if __name__ == "__main__":
    main()
