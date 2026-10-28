"""CPU-only end-to-end checks; bulk artifacts belong in a server output path.

No source/model substitutions: eight routes use the existing quick templates.
Additional controlled diagnostics use active_das_line_quick (Lg=4/8/16 m)
and passive_quick with 192 sources and 600/1200 s, frozen before calculation.
The image-change threshold of 5% is a diagnostic target, not physical accuracy.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import time
import traceback
from types import SimpleNamespace as NS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from wavedispx.app_service import load_template_mapping, apply_automatic_parameters, toml_dumps
from wavedispx.workflow import WaveDispXWorkflow
from wavedispx.processing import find_qedisp_forward
from wavedispx.core import _dwm_core
from wavedispx.plotting import _layer_steps
from wavedispx.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replot-gauges", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.replot_gauges:
        replot_gauges(output)
        return
    report = {"scope": "CPU integration smoke and controlled diagnostics; not universal convergence",
              "cases": {}, "diagnostic_target_image_relative_l2": .05,
              "qed_binary_sha256": sha(find_qedisp_forward()),
              "core_binary_sha256": sha(_dwm_core.__file__),
              "cpp_sha256": sha(ROOT / "cpp/dwm_core.cpp"),
              "script_sha256": sha(__file__)}

    def write():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    def run(name, mapping):
        path = output / name
        path.mkdir(exist_ok=True)
        mapping = deepcopy(mapping)
        mapping["run"].update(name=name, workers=4, output_directory=str(path))
        mapping["processing"]["theory_solver"] = "qedisp"
        (path / "frozen_input.toml").write_text(toml_dumps(mapping), encoding="utf-8")
        print("START", name, flush=True)
        start = time.perf_counter()
        try:
            w = WaveDispXWorkflow.from_mapping(mapping, base_directory=ROOT / "examples")
            review = w.review_inputs()
            if not review.ready:
                raise RuntimeError("Required preflight checks failed")
            w.run_all()
            a = w.active_analysis if w.config.mode == "active" else w.passive_analysis
            im = a.rayleigh_combined if w.config.mode == "active" else a.dispersion
            assert np.isfinite(im.energy).all() and np.max(im.energy) > 0
            np.savez_compressed(path / "inspection.npz", f=im.frequency_hz, c=im.velocity_mps,
                                energy=im.energy, theory=a.theory.rayleigh_mps)
            entry = {"status": "smoke-pass", "elapsed_s": time.perf_counter()-start,
                     "shape": list(im.energy.shape), "input_sha256": sha(path / "frozen_input.toml"),
                     "results_sha256": sha(path / "results.h5"), "diagnostics": review.diagnostics}
            if w.config.mode == "active":
                # Small actual waveforms retained for platform and gauge checks.
                np.savez_compressed(path / "waveforms.npz", time=w.shot.time_s,
                                    distance=w.shot.distance_m, zz=w.shot.vertical_vertical,
                                    das=w.shot.das_record.data if w.shot.das_record is not None else np.empty(0))
            report["cases"][name] = entry
            write()
            print("DONE", name, round(entry["elapsed_s"], 2), flush=True)
            return w, im
        except Exception:
            report["cases"][name] = {"status": "failed", "traceback": traceback.format_exc()}
            write()
            traceback.print_exc()
            return None, None
        finally:
            plt.close("all")

    for name in ("active_quick", "manual_active_array", "passive_quick", "passive_grid_quick",
                 "active_das_line_quick", "active_das_grid_quick", "passive_das_line_quick", "passive_das_grid_quick"):
        mapping = load_template_mapping(ROOT / f"examples/{name}.toml")
        run(name, mapping)

    # Apply automatic settings without recomputing physics to audit proposed values.
    for mode in ("active", "passive"):
        mapping = load_template_mapping(ROOT / f"examples/{mode}_quick.toml")
        auto, diagnostics = apply_automatic_parameters(mapping, base_directory=ROOT / "examples")
        (output / f"{mode}_automatic.toml").write_text(toml_dumps(auto), encoding="utf-8")
        report[f"{mode}_automatic"] = diagnostics
    write()

    pairs = []
    passive = load_template_mapping(ROOT / "examples/passive_quick.toml")
    passive["ambient_sources"]["count"] = 192
    for duration in (600, 1200):
        passive["noise"]["duration_s"] = float(duration)
        w, im = run(f"passive_192_{duration}s", passive)
        if im is not None:
            pairs.append((str(duration) + " s", w, im))
    if len(pairs) == 2:
        first, last = pairs[0][2], pairs[1][2]
        np.testing.assert_array_equal(first.frequency_hz, last.frequency_hz)
        np.testing.assert_array_equal(first.velocity_mps, last.velocity_mps)
        delta = float(np.linalg.norm(first.energy-last.energy)/np.linalg.norm(last.energy))
        n0, n1 = pairs[0][1].correlation.ncf.data, pairs[1][1].correlation.ncf.data
        report["duration_comparison"] = {
            "normalized_image_relative_l2": delta,
            "ncf_relative_l2": float(np.linalg.norm(n0-n1)/np.linalg.norm(n1)),
            "image_target_met": delta <= .05,
            "qualification": "nested fixed-seed realizations; 1200 s is not ground truth; finite source geometry unchanged"}
        comparison_plot(pairs, output / "duration_qc.png", difference=True)
        write()

    gauges = []
    for gauge in (4., 8., 16.):
        mapping = load_template_mapping(ROOT / "examples/active_das_line_quick.toml")
        # Hold an explicit 4 s record for the gauge comparison; do not re-auto-design per gauge.
        mapping["active_source"]["record_duration_s"] = 4.0
        mapping["das"]["gauge_length_m"] = gauge
        w, im = run(f"gauge_{gauge:g}m", mapping)
        if im is not None:
            gauges.append((f"Lg={gauge:g} m", w, im))
    if len(gauges) == 3:
        comparison_plot(gauges, output / "gauge_qc.png")
        report["gauge_qualification"] = "finite-gauge endpoints; identical model, source, centers, axes and explicit period; per-frequency normalized energy does not show absolute amplitude suppression"
        write()
    if any(case["status"] == "failed" for case in report["cases"].values()):
        raise SystemExit(1)


def comparison_plot(cases, path, difference=False):
    """Common axes/colors; model, real geometry and waveforms remain visible."""
    cols = 3
    fig, axes = plt.subplots(3, cols, figsize=(16, 12), layout="constrained")
    config = cases[0][1].config
    for key, color, label in (("vp_mps", "#ba781c", "Vp"), ("vs_mps", "#16768c", "Vs")):
        val, z = _layer_steps(config, np.array([getattr(x, key) for x in config.model.layers]))
        axes[0, 0].plot(val, z, color=color, label=label)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(xlabel="Velocity (m/s)", ylabel="Depth (m)")
    axes[0, 0].legend()
    geometry = cases[0][1].geometry
    axes[0, 1].scatter(geometry.source_x_m, geometry.source_y_m, marker="*", label="Sources", s=12)
    axes[0, 1].scatter(geometry.receiver_x_m, geometry.receiver_y_m, marker="v", label="Receivers", s=12)
    axes[0, 1].set(xlabel="x (m)", ylabel="y (m)")
    axes[0, 1].set_aspect("equal", adjustable="datalim")
    axes[0, 1].legend()
    axes[0, 2].axis("off")
    axes[0, 2].text(0, 1, "Frozen inputs\n" + "\n".join(
        f"h={l.thickness_m:g} m; Vp/Vs={l.vp_mps:g}/{l.vs_mps:g}\nrho={l.density_kgm3:g} kg/m3; Qp/Qs={l.qp:g}/{l.qs:g}"
        for l in config.model.layers) + "\n\n" + (
        "192 independent vertical sources\nRicker PSD: f0=7 Hz, 3-10 Hz\n10 s windows, whitening NCF\nphase_shift; fixed seeds 42 / 314159"
        if difference else "Vertical Ricker source; finite DAS strain rate\n4 s record; phase_shift\nOnly gauge length varies"), va="top", fontsize=11)
    for i, (name, w, im) in enumerate(cases):
        mesh = axes[1, i].pcolormesh(im.frequency_hz, im.velocity_mps, im.energy,
                                    shading="nearest", cmap="viridis", vmin=0, vmax=1)
        a = w.active_analysis if w.config.mode == "active" else w.passive_analysis
        for m in range(a.theory.rayleigh_mps.shape[1]):
            axes[1, i].plot(im.frequency_hz, a.theory.rayleigh_mps[:, m], "w--", lw=.9)
        axes[1, i].set(title=name, xlabel="Frequency (Hz)", ylabel="Phase velocity (m/s)")
        if difference:
            from wavedispx.plotting import plot_ncf_branches
            plot_ncf_branches(axes[2, i], w.correlation.strategy_stacks["baseline_linear"].component_branches["zz"], title=name + " actual NCF")
        else:
            # Same global color limit across gauges preserves amplitude effects.
            scale = max(np.percentile(np.abs(case[1].shot.das_record.data), 99.5) for case in cases)
            axes[2, i].pcolormesh(w.shot.distance_m, w.shot.time_s, w.shot.das_record.data,
                                 shading="nearest", cmap="RdBu_r", vmin=-scale, vmax=scale)
            axes[2, i].invert_yaxis()
            axes[2, i].set(xlabel="Offset (m)", ylabel="Time (s)", title=name + " common amplitude scale")
    fig.colorbar(mesh, ax=axes[1, :len(cases)], label="Per-frequency normalized image")
    if difference:
        diff = axes[1, 2].pcolormesh(im.frequency_hz, im.velocity_mps,
                                   cases[0][2].energy-cases[1][2].energy,
                                   shading="nearest", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, 2].set(title="600 s minus 1200 s", xlabel="Frequency (Hz)", ylabel="Phase velocity (m/s)")
        fig.colorbar(diff, ax=axes[1, 2], label="Normalized-image difference")
        axes[2, 2].axis("off")
        axes[2, 2].text(0, 1, "No smoothing / no theory mask\n1200 s is a longer reference, not truth.\nSame source field and nested record prefix.\nNCF panels use per-panel 99% amplitude clips.\nThis comparison tests statistical change,\nnot spatial or Green-function convergence.", va="top")
    fig.suptitle("Controlled diagnostic — not a universal convergence certificate", fontsize=15)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def replot_gauges(output):
    """Recover a plotting-only failure from saved arrays; never rerun forward."""
    report = json.loads((output / "report.json").read_text())
    cases = []
    amplitudes = {}
    for gauge in (4, 8, 16):
        path = output / f"gauge_{gauge}m"
        config = load_config(path / "frozen_input.toml")
        with np.load(path / "inspection.npz", allow_pickle=False) as saved:
            im = NS(frequency_hz=saved["f"], velocity_mps=saved["c"], energy=saved["energy"])
            theory = NS(rayleigh_mps=saved["theory"])
        with h5py.File(path / "results.h5", "r") as h:
            shot = NS(time_s=h["shot/time_s"][:], distance_m=h["shot/distance_m"][:],
                      das_record=NS(data=h["das/data"][:]))
            geometry = NS(receiver_x_m=h["shot/receiver_x_m"][:], receiver_y_m=h["shot/receiver_y_m"][:],
                          source_x_m=np.array([config.active_source.x_m]), source_y_m=np.array([config.active_source.y_m]))
        w = NS(config=config, geometry=geometry, shot=shot, active_analysis=NS(theory=theory))
        cases.append((f"Lg={gauge} m", w, im))
        amplitudes[str(gauge)] = float(np.sqrt(np.mean(shot.das_record.data**2)))
    comparison_plot(cases, output / "gauge_qc.png")
    for path in output.glob("*/waveforms.npz"):
        with h5py.File(path.parent / "results.h5", "r") as h:
            # Numeric-only inspection archive, including originally successful active cases.
            np.savez_compressed(path, time=h["shot/time_s"][:], distance=h["shot/distance_m"][:],
                                zz=h["shot/components"][:, :, 4],
                                das=h["das/data"][:] if "das" in h else np.empty(0))
    report["gauge_waveform_rms"] = amplitudes
    report["gauge_qualification"] = "Only Lg changes; shared waveform amplitude scale; energy column normalization hides absolute suppression. Not an independent Green accuracy check."
    report["plot_recovery"] = {"forward_rerun": False, "plot_script_sha256": sha(__file__),
                               "cause": "DASRecord wrapper initially passed instead of its numeric data"}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
