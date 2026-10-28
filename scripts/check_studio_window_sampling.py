"""Targeted active period and imaging-grid sensitivity, CPU/server only."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wavedispx.app_service import load_template_mapping, toml_dumps
from wavedispx.workflow import WaveDispXWorkflow
from wavedispx.active import analyze_active_shot
from wavedispx.config import config_from_mapping

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = load_template_mapping(ROOT / "examples/active_das_line_quick.toml")
    mapping["das"]["gauge_length_m"] = 16.0
    mapping["processing"].update(theory_solver="qedisp", frequency_step_hz=.1, velocity_step_mps=2.)
    report = {"scope": "same active model/source/geometry, Lg16; period changes native DWM discretization and damping too",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": {}}
    results = []
    for period in (4., 8.):
        case = deepcopy(mapping)
        case["active_source"]["record_duration_s"] = period
        out = args.output / f"period_{period:g}s"
        out.mkdir(exist_ok=True)
        case["run"].update(workers=4, output_directory=str(out))
        (out / "frozen_input.toml").write_text(toml_dumps(case))
        w = WaveDispXWorkflow.from_mapping(case, base_directory=ROOT / "examples")
        t = time.perf_counter()
        w.run_all()
        x = w.shot.das_record.data
        tail = np.sum(x[int(.8*len(x)):]**2, axis=0) / np.maximum(np.sum(x*x, axis=0), 1e-300)
        report["cases"][str(period)] = {"elapsed_s":time.perf_counter()-t,
            "tail_20pct_energy_fraction_near_mid_far": tail[[0, len(tail)//2, -1]].tolist(),
            "input_sha256": hashlib.sha256((out / "frozen_input.toml").read_bytes()).hexdigest()}
        results.append(w)
    a, b = [w.active_analysis.rayleigh_combined.energy for w in results]
    report["period_image_relative_l2"] = float(np.linalg.norm(a-b)/np.linalg.norm(b))
    # Change only imaging sampling on the SAME 4 s shot. Actual new evaluation points.
    finer = deepcopy(mapping)
    finer["active_source"]["record_duration_s"] = 4.
    finer["processing"].update(frequency_step_hz=.05, velocity_step_mps=1.)
    fine = analyze_active_shot(config_from_mapping(finer, base_directory=ROOT / "examples"), results[0].shot)
    coarse = results[0].active_analysis.rayleigh_combined
    fine_image = fine.rayleigh_combined
    np.testing.assert_allclose(coarse.frequency_hz, fine_image.frequency_hz[::2])
    np.testing.assert_allclose(coarse.velocity_mps, fine_image.velocity_mps[::2])
    report["sampling_common_points_relative_l2"] = float(np.linalg.norm(coarse.energy-fine_image.energy[::2, ::2])/np.linalg.norm(fine_image.energy[::2, ::2]))
    report["sampling_note"] = "Common-point check only; finer interstitial peaks and weak modes are not certified by this scalar."
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    limit = max(np.percentile(np.abs(w.shot.das_record.data),99.5) for w in results)
    for col,w in enumerate(results):
        axes[0,col].pcolormesh(w.shot.distance_m, w.shot.time_s, w.shot.das_record.data,
                              shading="nearest", cmap="RdBu_r", vmin=-limit,vmax=limit)
        axes[0,col].invert_yaxis()
        axes[0,col].set(title=f"T={w.config.active_source.record_duration_s:g}s | common amplitude clip",
                        xlabel="Offset (m)",ylabel="Time (s)")
        im = w.active_analysis.rayleigh_combined
        mesh=axes[1,col].pcolormesh(im.frequency_hz,im.velocity_mps,im.energy,shading="nearest",cmap="viridis",vmin=0,vmax=1)
        axes[1,col].set(xlabel="Frequency (Hz)",ylabel="Phase velocity (m/s)")
    fig.colorbar(mesh,ax=axes[1,:],label="Per-frequency normalized image")
    fig.suptitle("DAS Lg=16 m | period-only input change; native damping/grid also change\nSame two-layer model, source and 16 receivers as gauge_qc.png; no tail clipping")
    fig.savefig(args.output / "period_qc.png",dpi=140)
    (args.output / "report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
