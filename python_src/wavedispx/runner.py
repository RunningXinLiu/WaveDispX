"""End-to-end active/passive command runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .active import analyze_active_shot, simulate_active_shot
from .artifacts import RunArtifacts, save_active_results, save_passive_results
from .config import RunConfig
from .geometry import Geometry, receiver_positions
from .plotting import plot_active_summary, plot_dispersion_comparison, plot_passive_summary
from .passive import simulate_passive


def run_active(config: RunConfig) -> RunArtifacts:
    if config.mode != "active" or config.active_source is None:
        raise ValueError("run_active requires an active configuration")
    start = time.perf_counter()
    shot = simulate_active_shot(
        config.model,
        config.active_source,
        config.receivers,
        das=config.das,
        processing=config.processing,
        workers=config.workers,
        distance_block_size=config.frequency_block_size * 8,
    )
    forward_seconds = time.perf_counter() - start
    analysis = analyze_active_shot(config, shot)
    receiver_x, receiver_y = receiver_positions(config.receivers)
    geometry = Geometry(
        receiver_x,
        receiver_y,
        np.array([config.active_source.x_m]),
        np.array([config.active_source.y_m]),
    )
    output = config.output_directory
    summary_path = output / "summary.png"
    plot_active_summary(
        config,
        geometry,
        shot,
        analysis.rayleigh,
        analysis.love,
        analysis.rayleigh_picks,
        analysis.love_picks,
        analysis.theory,
        summary_path,
        rayleigh_combined=analysis.rayleigh_combined,
        rayleigh_combined_picks=analysis.rayleigh_combined_picks,
    )
    plot_dispersion_comparison(
        {
            **{name: image for name, image in analysis.component_images.items() if name != "tt"},
            "combined Rayleigh": analysis.rayleigh_combined,
        },
        analysis.theory.rayleigh_mps,
        output / "active_components.png",
        title="Active component sensitivity",
    )
    if len(analysis.method_comparison) > 1:
        plot_dispersion_comparison(
            analysis.method_comparison,
            analysis.theory.rayleigh_mps,
            output / "active_method_comparison.png",
            title="Active Rayleigh imaging-method comparison",
        )
    timings = {
        "forward": forward_seconds,
        "imaging": analysis.imaging_seconds,
        "total": time.perf_counter() - start,
    }
    return save_active_results(
        config,
        geometry,
        shot,
        analysis.rayleigh,
        analysis.love,
        analysis.rayleigh_picks,
        analysis.love_picks,
        analysis.theory,
        analysis.quality,
        timings=timings,
        summary_png=summary_path,
        rayleigh_combined=analysis.rayleigh_combined,
        rayleigh_combined_picks=analysis.rayleigh_combined_picks,
        component_images=analysis.component_images,
        method_comparison=analysis.method_comparison,
    )


def run_passive(config: RunConfig) -> RunArtifacts:
    simulation = simulate_passive(config)
    summary_path = config.output_directory / "summary.png"
    plot_passive_summary(config, simulation, summary_path)
    if len(simulation.component_images) > 1:
        plot_dispersion_comparison(
            {
                **{
                    name: image
                    for name, image in simulation.component_images.items()
                    if name != "tt"
                },
                "combined Rayleigh": simulation.dispersion,
            },
            simulation.theory.rayleigh_mps,
            config.output_directory / "passive_components.png",
            title="Passive 3-C component sensitivity",
        )
    if len(simulation.method_comparison) > 1:
        plot_dispersion_comparison(
            simulation.method_comparison,
            simulation.theory.rayleigh_mps,
            config.output_directory / "passive_method_comparison.png",
            title="Passive Rayleigh imaging-method comparison",
        )
    return save_passive_results(config, simulation, summary_png=summary_path)


def run(config: RunConfig) -> RunArtifacts:
    if config.mode == "active":
        return run_active(config)
    return run_passive(config)
