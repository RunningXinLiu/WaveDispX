"""Replot saved passive data without forward/NCF/imaging recomputation."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import h5py
import matplotlib.pyplot as plt
import numpy as np

from wavedispx.config import load_config
from wavedispx.geometry import Geometry
from wavedispx.plotting import plot_ncf_branches, plot_passive_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.directory
    output = args.output or directory / "summary_two_sided.png"
    data_path = directory / "results.h5"
    before = hashlib.sha256(data_path.read_bytes()).hexdigest()
    config = load_config(directory / "input.toml")
    if config.das.enabled:
        raise ValueError("This saved-summary utility currently supports node cases only; use the live DAS workflow plots.")
    receivers = np.loadtxt(directory / "receivers.csv", delimiter=",", skiprows=1, ndmin=2)
    sources = np.loadtxt(directory / "sources.csv", delimiter=",", skiprows=1, ndmin=2)
    picks = np.loadtxt(directory / "picked_rayleigh.csv", delimiter=",", skiprows=1, ndmin=2)
    with h5py.File(data_path, "r") as handle:
        ncf = handle["ncf"]
        strategy = handle["ncf/strategies/baseline_linear"]
        branches = {}
        for name, group in strategy["components"].items():
            branches[name] = NS(**{
                side: NS(data=group[side][:], lag_time_s=strategy["lag_time_s"][:],
                         distance_m=strategy["distance_m"][:])
                for side in ("negative", "positive", "symmetric")})
        image = handle["dispersion/rayleigh"]
        simulation = NS(
            geometry=Geometry(receivers[:, 0], receivers[:, 1], sources[:, 0], sources[:, 1]),
            transfer=NS(das_geometry=None, frequency_hz=handle["transfer/frequency_hz"][:]),
            ncf=NS(data=ncf["gather"][:], lag_time_s=ncf["lag_time_s"][:], distance_m=ncf["distance_m"][:]),
            strategy_stacks={"baseline_linear": NS(component_branches=branches)},
            dispersion=NS(**{key: image[key][:] for key in ("energy", "frequency_hz", "velocity_mps")}),
            theory=NS(rayleigh_mps=handle["theory/rayleigh_mps"][:]),
            picks=NS(frequency_hz=picks[:, 0], phase_velocity_mps=picks[:, 1]),
            independent_window_count=int(strategy.attrs["windows_seen"]),
        )
    plot_passive_summary(config, simulation, output)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    plot_ncf_branches(ax, branches["zz"], title="Actual two-sided ZZ NCF | distance increases upward")
    fig.savefig(output.with_name(output.stem + "-ncf.png"), dpi=160)
    plt.close(fig)
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == before
    print(f"Unchanged results.h5 SHA256: {before}\nSaved: {output}")


if __name__ == "__main__":
    main()
