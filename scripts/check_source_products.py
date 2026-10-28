"""CPU acceptance for source/products v1. Run bulk outputs on the server."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wavedispx.app_service import load_template_mapping, toml_dumps
from wavedispx.workflow import WaveDispXWorkflow
from wavedispx.experiment import ExperimentRequest, run_experiment, save_experiment, theory_context
from wavedispx.plotting import _layer_steps
from wavedispx.processing import find_qedisp_forward
from wavedispx.core import _dwm_core

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={"scope":"integration and controlled identities, NOT full physical convergence",
            "cases":{},"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "core_sha256":hashlib.sha256(Path(_dwm_core.__file__).read_bytes()).hexdigest(),
            "qed_sha256":hashlib.sha256(Path(find_qedisp_forward()).read_bytes()).hexdigest()}
    cases={}
    active=load_template_mapping(ROOT/"examples/active_quick.toml")
    active["active_source"].update(component="vector",force_amplitude=1.,force_azimuth_deg=90.,force_inclination_deg=0.,maximum_frequency_hz=32.,record_duration_s=4.)
    active["processing"].update(frequency_min_hz=4.,frequency_max_hz=18.,frequency_step_hz=.1,theory_solver="qedisp")
    active["experiment"]={"product":"energy_theory","waves":"both"}
    cases["active_transverse"]=active
    tilted=deepcopy(active)
    tilted["active_source"].update(force_azimuth_deg=45.,force_inclination_deg=35.)
    cases["active_tilted"]=tilted
    for name,pol in (("passive_isotropic_h","independent_hv"),("passive_directed_h","directed_hv")):
        passive=load_template_mapping(ROOT/"examples/passive_quick.toml")
        passive["ambient_sources"].update(component="three_component",polarization=pol,horizontal_power_fraction=.6,horizontal_azimuth_deg=45. if pol=="directed_hv" else 0.)
        passive["processing"]["theory_solver"]="qedisp"
        passive["experiment"]={"product":"hvsr","waves":"both","hvsr_smoothing_b":40.}
        cases[name]=passive
    for name,mapping in cases.items():
        mapping["run"].update(name=name,workers=4,output_directory=str(args.output/name))
        request=ExperimentRequest.from_mapping(mapping)
        w=WaveDispXWorkflow.from_mapping(mapping,base_directory=ROOT/"examples")
        if not w.review_inputs().ready:
            raise RuntimeError("preflight failed: "+name)
        result=run_experiment(w,request)
        directory=save_experiment(result,w,mapping)
        report["cases"][name]={"output":str(directory),"statuses":result.statuses,"elapsed_s":result.elapsed_s,
                               "config_sha256":hashlib.sha256(toml_dumps(mapping).encode()).hexdigest()}
        if name=="active_transverse":
            assert "rayleigh" not in result.images and "love" in result.images
            np.testing.assert_array_equal(w.shot.vector_xyz[...,2],0)
        if result.hvsr is not None:
            h=result.hvsr
            assert h.processed_windows==6 and np.isfinite(h.geometric_mean).any()
            report["cases"][name]["hvsr_valid_fraction"]=float(np.isfinite(h.geometric_mean).mean())
            # Generation is pre-correlation, same random stream. Compare the
            # first raw window to the legacy streaming NCF producer.
            corr=w.simulate_observations()
            n=len(w.hvsr_preview)
            diff=np.linalg.norm(corr.preview_record[:n]-w.hvsr_preview)/np.linalg.norm(w.hvsr_preview)
            assert diff<1e-6  # correlation preview is float32
            report["cases"][name]["raw_preview_relative_l2"]=float(diff)
        fig,axes=plt.subplots(2,3,figsize=(15,9),layout="constrained")
        for key,label in (("vp_mps","Vp"),("vs_mps","Vs")):
            values,depth=_layer_steps(w.config,np.array([getattr(l,key) for l in w.config.model.layers]))
            axes[0,0].plot(values,depth,label=label)
        axes[0,0].invert_yaxis()
        axes[0,0].set(xlabel="Velocity (m/s)",ylabel="Depth (m)",title="Model | rho=2000 kg/m³ both layers; Q=inf")
        axes[0,0].legend()
        g=w.geometry
        axes[0,1].scatter(g.source_x_m,g.source_y_m,marker="*",label="Sources")
        axes[0,1].scatter(g.receiver_x_m,g.receiver_y_m,s=12,marker="v",label="Receivers")
        axes[0,1].set(xlabel="x (m)",ylabel="y (m)",title="Actual acquisition")
        axes[0,1].legend()
        axes[0,2].axis("off")
        if w.config.mode=="active":
            source=w.config.active_source
            axes[0,2].text(0,1,f"Coherent vector force v1\nF XYZ={w.shot.force_xyz}\nAzimuth={source.force_azimuth_deg:g} deg; inclination={source.force_inclination_deg:g} deg\nRicker f0={source.dominant_frequency_hz:g} Hz; cutoff=32 Hz\ndt=.005 s; T=4 s\nNative source smoothing / taper retained\nImage: {w.config.processing.dispersion_method}",va="top")
            for ax,wave in zip(axes[1,:2],("rayleigh","love")):
                if wave in result.images:
                    im=result.images[wave]
                    mesh=ax.pcolormesh(im.frequency_hz,im.velocity_mps,im.energy,shading="nearest",vmin=0,vmax=1,cmap="turbo")
                    curves=getattr(result.theory,wave+"_mps")
                    for mode in range(curves.shape[1]):
                        ax.plot(im.frequency_hz,curves[:,mode],"w--",lw=1,label=f"QED M{mode}")
                    fig.colorbar(mesh,ax=ax,label="Per-frequency normalized image")
                    ax.set(xlabel="Frequency (Hz)",ylabel="Phase velocity (m/s)",title=wave)
                else:
                    ax.axis("off")
                    ax.text(.1,.5,"No radial/vertical forcing in this line geometry\nRayleigh energy not generated; tab kept with explanation.")
            for i,label in enumerate("XYZ"):
                axes[1,2].plot(w.shot.time_s,w.shot.vector_xyz[:,8,i],label=label)
            axes[1,2].set(xlabel="Time (s)",ylabel="Native displacement",title="Actual vector-force waveform | receiver 8")
            axes[1,2].legend()
        else:
            h=result.hvsr
            station=20
            axes[0,2].text(0,1,f"{w.config.ambient_sources.polarization}\npH=.6; pV=.4; NOT receiver H/V\n24 annulus sources; seed42 / noise314159\nRicker PSD peak7 Hz, band3–10 Hz\nTotal60 s, independent windows10 s, dt=.01 s\nKO b40; 10% Tukey; vertical floor1e-6\nStation {station}; >=3 valid windows; no whitening\nNative transfer interpolation/smoothing retained",va="top")
            mean,std=h.geometric_mean[station],h.log_std[station]
            axes[1,0].plot(h.frequency_hz,mean)
            axes[1,0].fill_between(h.frequency_hz,mean*np.exp(-std),mean*np.exp(std),alpha=.2)
            axes[1,0].set(xlabel="Frequency (Hz)",ylabel="H/V",title="Geometric mean ± window log scatter")
            for i,label in enumerate("XYZ"):
                axes[1,1].semilogy(h.frequency_hz,h.mean_amplitude_xyz[station,:,i],label=label)
            axes[1,1].legend()
            axes[1,1].set(xlabel="Frequency (Hz)",ylabel="Native Fourier amplitude",title="Unwhitened component spectra")
            mesh=axes[1,2].pcolormesh(h.frequency_hz,h.directions_deg,h.directional_geometric_mean[station].T,shading="nearest")
            axes[1,2].set(xlabel="Frequency (Hz)",ylabel="Azimuth from +X (deg)",title="Directional H/V")
            fig.colorbar(mesh,ax=axes[1,2])
        fig.suptitle(name+" | diagnostic only, not convergence-certified")
        fig.savefig(args.output/(name+".png"),dpi=130)
        plt.close(fig)
        (args.output/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
        print(name,report["cases"][name],flush=True)


if __name__=="__main__": main()
