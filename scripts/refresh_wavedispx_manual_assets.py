"""Refresh WaveDispX branding on manual figures without recomputing wavefields.

The scientific arrays are read from existing HDF5 results.  Only preflight and
summary figures plus application/version metadata are regenerated, which keeps
the published benchmark values unchanged during a branding-only release.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np

from wavedispx import WaveDispXWorkflow, __version__
from wavedispx.active import ActiveShotGather
from wavedispx.plotting import plot_active_summary, plot_passive_summary
from wavedispx.preflight import plot_preflight_review
from wavedispx.processing import DispersionImage, DispersionPicks, TheoreticalDispersion


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/manual_assets"
CASES = (
    "manual_active_line",
    "manual_active_array",
    "manual_passive_line",
    "manual_passive_array",
)


def _image(group: h5py.Group, method: str) -> DispersionImage:
    energy = group["energy"][:]
    return DispersionImage(
        energy=energy,
        raw_amplitude=energy,
        frequency_hz=group["frequency_hz"][:],
        velocity_mps=group["velocity_mps"][:],
        method=method,
    )


def _picks(path: Path) -> DispersionPicks:
    table = np.genfromtxt(path, delimiter=",", names=True)
    return DispersionPicks(
        frequency_hz=np.atleast_1d(table["frequency_hz"]),
        phase_velocity_mps=np.atleast_1d(table["phase_velocity_mps"]),
        normalized_energy=np.atleast_1d(table["normalized_energy"]),
        half_power_uncertainty_mps=np.atleast_1d(
            table["half_power_uncertainty_mps"]
        ),
    )


def _theory(handle: h5py.File) -> TheoreticalDispersion:
    group = handle["theory"]
    return TheoreticalDispersion(
        group["frequency_hz"][:],
        group["rayleigh_mps"][:],
        group["love_mps"][:],
    )


def _set_brand_metadata(handle: h5py.File) -> None:
    handle.attrs["application"] = "WaveDispX"
    handle.attrs["wavedispx_version"] = __version__
    handle.attrs["forward_solver"] = "discrete-wavenumber method (DWM)"
    if "dwmwave_version" in handle.attrs:
        del handle.attrs["dwmwave_version"]


def _refresh_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text())
    runtime = manifest.setdefault("runtime", {})
    runtime.pop("dwmwave_version", None)
    runtime.update(
        application="WaveDispX",
        wavedispx_version=__version__,
        forward_solver="discrete-wavenumber method (DWM)",
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def refresh(case: str) -> None:
    config_path = ROOT / f"examples/{case}.toml"
    result_directory = ROOT / "output_python" / case
    workflow = WaveDispXWorkflow.from_toml(config_path)
    review = workflow.review_inputs()
    figure = plot_preflight_review(review)
    figure.savefig(ASSETS / f"{case}-preflight.png", dpi=180)
    plt.close(figure)

    results_path = result_directory / "results.h5"
    with h5py.File(results_path, "r+") as handle:
        _set_brand_metadata(handle)
        theory = _theory(handle)
        if workflow.config.mode == "active":
            shot_group = handle["shot"]
            shot = ActiveShotGather(
                shot_group["time_s"][:],
                shot_group["receiver_x_m"][:],
                shot_group["receiver_y_m"][:],
                shot_group["distance_m"][:],
                shot_group["components"][:],
            )
            rayleigh = _image(
                handle["dispersion/rayleigh"],
                workflow.config.processing.dispersion_method,
            )
            love = _image(
                handle["dispersion/love"],
                workflow.config.processing.dispersion_method,
            )
            plot_active_summary(
                workflow.config,
                review.geometry,
                shot,
                rayleigh,
                love,
                _picks(result_directory / "picked_rayleigh.csv"),
                _picks(result_directory / "picked_love.csv"),
                theory,
                result_directory / "summary.png",
            )
        else:
            ncf_group = handle["ncf"]
            dispersion = _image(
                handle["dispersion/rayleigh"],
                workflow.config.processing.dispersion_method,
            )
            simulation = SimpleNamespace(
                geometry=review.geometry,
                ncf=SimpleNamespace(
                    data=ncf_group["gather"][:],
                    distance_m=ncf_group["distance_m"][:],
                    lag_time_s=ncf_group["lag_time_s"][:],
                ),
                dispersion=dispersion,
                theory=theory,
                picks=_picks(result_directory / "picked_rayleigh.csv"),
                transfer=SimpleNamespace(
                    frequency_hz=handle["transfer/frequency_hz"][:]
                ),
                independent_window_count=int(
                    handle.attrs.get(
                        "independent_window_count",
                        round(
                            workflow.config.noise.duration_s
                            / workflow.config.noise.window_s
                        ),
                    )
                ),
            )
            plot_passive_summary(
                workflow.config, simulation, result_directory / "summary.png"
            )

    source_summary = result_directory / "summary.png"
    target_summary = ASSETS / f"{case}-summary.png"
    target_summary.write_bytes(source_summary.read_bytes())
    _refresh_manifest(result_directory / "manifest.json")
    print(f"refreshed: {case}")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        refresh(case)


if __name__ == "__main__":
    main()
