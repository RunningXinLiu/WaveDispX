"""Explicit numerical-cutoff sensitivity, not a production parameter override."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.app_service import load_template_mapping, toml_dumps
from wavedispx.workflow import WaveDispXWorkflow

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = load_template_mapping(ROOT / "examples/active_das_line_quick.toml")
    original["das"]["gauge_length_m"] = 16.0
    original["processing"].update(theory_solver="qedisp", frequency_step_hz=.1,
                                  velocity_step_mps=2.)
    report = {"scope": "Native cutoff sensitivity; unchanged model, source Ricker f0=7Hz, dt=.005, geometry, gauge16; not equivalent spectra",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": {}}
    results = {}
    for cutoff in (16., 24., 32.):
        for period in (4., 8.):
            key = f"cutoff{cutoff:g}_T{period:g}"
            case = deepcopy(original)
            case["active_source"].update(record_duration_s=period, maximum_frequency_hz=cutoff)
            case["run"].update(workers=4, output_directory=str(args.output / key))
            frozen = toml_dumps(case)
            (args.output / f"{key}.toml").write_text(frozen)
            w = WaveDispXWorkflow.from_mapping(case, base_directory=ROOT / "examples")
            analysis = w.compute_dispersion()
            x = w.shot.das_record.data
            tail = np.sum(x[int(.8*len(x)):]**2, axis=0) / np.maximum(np.sum(x*x, axis=0), 1e-300)
            im = analysis.rayleigh_combined
            np.savez_compressed(args.output / f"{key}.npz", time=w.shot.time_s,
                                offset=w.shot.distance_m, das=x, f=im.frequency_hz,
                                c=im.velocity_mps, energy=im.energy)
            report["cases"][key] = {"input_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
                "tail_20pct_energy_fraction_near_mid_far": tail[[0,len(tail)//2,-1]].tolist()}
            results[(cutoff,period)] = w
            print(key, report["cases"][key], flush=True)
        a,b = [results[(cutoff,p)].active_analysis.rayleigh_combined.energy for p in (4.,8.)]
        report.setdefault("period_image_relative_l2", {})[str(cutoff)] = float(np.linalg.norm(a-b)/np.linalg.norm(b))
    fig, axes = plt.subplots(2,3,figsize=(15,8),layout="constrained")
    limit = max(np.percentile(np.abs(w.shot.das_record.data),99.5) for w in results.values())
    for col,cutoff in enumerate((16.,24.,32.)):
        for row,period in enumerate((4.,8.)):
            w=results[(cutoff,period)]
            axes[row,col].pcolormesh(w.shot.distance_m,w.shot.time_s,w.shot.das_record.data,
                                     shading="nearest",cmap="RdBu_r",vmin=-limit,vmax=limit)
            axes[row,col].invert_yaxis()
            axes[row,col].set(title=f"Cutoff={cutoff:g} Hz | T={period:g} s",xlabel="Offset (m)",ylabel="Time (s)")
    fig.suptitle("Numerical cutoff sensitivity | same Ricker f0=7 Hz, Lg=16 m | common amplitude clip\nNative radius smoothing, Nyquist taper and exponential correction unchanged; no tail removal")
    fig.savefig(args.output / "cutoff_qc.png",dpi=130)
    (args.output / "report.json").write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
