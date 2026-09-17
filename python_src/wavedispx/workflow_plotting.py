"""Compact inline figures for staged notebook execution."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .workflow import WaveDispXWorkflow
from .plotting import plot_ncf_branches


def plot_receiver_layout_comparison(reviews: dict[str, object]):
    """Show full source-receiver and receiver-only views for layout options."""
    if not reviews:
        raise ValueError("at least one receiver layout review is required")
    layout_count = len(reviews)
    figure, axes = plt.subplots(
        2,
        layout_count,
        figsize=(7 * layout_count, 9),
        constrained_layout=True,
        squeeze=False,
    )
    for column, (name, review) in enumerate(reviews.items()):
        geometry = review.geometry
        full_axis = axes[0, column]
        receiver_axis = axes[1, column]
        full_axis.scatter(
            geometry.source_x_m,
            geometry.source_y_m,
            marker="*",
            s=55,
            c="tab:red",
            label="Sources",
        )
        full_axis.scatter(
            geometry.receiver_x_m,
            geometry.receiver_y_m,
            marker="v",
            s=22,
            c="tab:blue",
            label="Receivers",
        )
        full_axis.set(
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"{name}: full source-receiver geometry",
        )
        full_axis.set_aspect("equal", adjustable="datalim")
        full_axis.grid(alpha=0.25)
        full_axis.legend()
        receiver_axis.scatter(
            geometry.receiver_x_m,
            geometry.receiver_y_m,
            marker="v",
            s=28,
            c="tab:blue",
        )
        receiver_axis.set(
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"{name}: receiver-array zoom ({geometry.receiver_x_m.size})",
        )
        receiver_axis.set_aspect("equal", adjustable="datalim")
        receiver_axis.grid(alpha=0.25)
    figure.suptitle("Available receiver layouts: line and 2-D grid", fontsize=15)
    return figure


def _image(axis, image, theory, title: str) -> None:
    axis.imshow(
        image.energy,
        origin="lower",
        aspect="auto",
        extent=(
            image.frequency_hz[0],
            image.frequency_hz[-1],
            image.velocity_mps[0],
            image.velocity_mps[-1],
        ),
        cmap="turbo",
        vmin=0,
        vmax=1,
    )
    for mode in range(theory.shape[1]):
        axis.plot(image.frequency_hz, theory[:, mode], "w--", linewidth=1)
    axis.set(xlabel="Frequency (Hz)", ylabel="Phase velocity (m/s)", title=title)


def plot_workflow_forward(workflow: WaveDispXWorkflow):
    forward = workflow.compute_forward()
    if workflow.config.mode == "active":
        order = np.argsort(forward.distance_m)
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        panels = (
            (
                (forward.das_record.data[:, order], "DAS finite-gauge shot gather"),
                (forward.vertical_vertical[:, order], "Node ZZ comparison gather"),
            )
            if forward.das_record is not None
            else (
                (forward.vertical_vertical[:, order], "Vertical-load vertical gather"),
                (forward.horizontal_tangential[:, order], "Horizontal-load transverse gather"),
            )
        )
        for axis, (gather, title) in zip(axes, panels):
            scale = np.percentile(np.abs(gather), 99.5)
            axis.imshow(
                gather,
                origin="upper",
                aspect="auto",
                extent=(
                    forward.distance_m[order][0],
                    forward.distance_m[order][-1],
                    forward.time_s[-1],
                    forward.time_s[0],
                ),
                cmap="seismic",
                vmin=-scale,
                vmax=scale,
            )
            axis.set(xlabel="Source-receiver distance (m)", ylabel="Time (s)", title=title)
        figure.suptitle(
            f"Active waveform | saved duration {forward.time_s[-1]:.3f} s | "
            f"dt {workflow.config.active_source.dt_s:g} s"
        )
        return figure
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    mean_axes = tuple(range(1, forward.spectrum.ndim))
    mean_amplitude = np.mean(np.abs(forward.spectrum), axis=mean_axes)
    axes[0].plot(forward.frequency_hz, mean_amplitude, "k-")
    axes[0].set(
        xlabel="Frequency (Hz)",
        ylabel="Mean transfer amplitude",
        title="Band-limited source-receiver transfer",
    )
    axes[0].grid(alpha=0.25)
    target = workflow.config.noise.dominant_frequency_hz
    frequency_index = int(np.argmin(np.abs(forward.frequency_hz - target)))
    matrix = np.abs(
        forward.spectrum[frequency_index, ..., -1]
        if forward.spectrum.ndim == 4
        else forward.spectrum[frequency_index]
    )
    axes[1].imshow(matrix, origin="lower", aspect="auto", cmap="magma")
    axes[1].set(
        xlabel="Source index",
        ylabel="Receiver index",
        title=(
            f"ZZ transfer amplitude at {forward.frequency_hz[frequency_index]:.2f} Hz"
            if forward.spectrum.ndim == 4
            else (
                f"DAS transfer amplitude at {forward.frequency_hz[frequency_index]:.2f} Hz"
                if workflow.config.das.enabled
                else f"Transfer amplitude at {forward.frequency_hz[frequency_index]:.2f} Hz"
            )
        ),
    )
    figure.suptitle(
        f"Transfer {forward.bytes / 2**20:.1f} MiB | radial interpolation L2 "
        f"{forward.interpolation_relative_l2:.3e}"
    )
    return figure


def plot_workflow_observations(workflow: WaveDispXWorkflow):
    observations = workflow.simulate_observations()
    if workflow.config.mode == "active":
        return plot_workflow_forward(workflow)
    three_component = observations.preview_record.ndim == 3
    figure, axes = plt.subplots(
        1, 3 if three_component else 2,
        figsize=(19 if three_component else 14, 5),
        constrained_layout=True,
    )
    record = observations.preview_record
    if three_component:
        record = record[:, :, 2]
    scale = np.percentile(np.abs(record), 99.5)
    axes[0].imshow(
        record,
        origin="upper",
        aspect="auto",
        extent=(0, record.shape[1] - 1, observations.preview_time_s[-1], 0),
        cmap="seismic",
        vmin=-scale,
        vmax=scale,
    )
    axes[0].set(
        xlabel="Receiver index",
        ylabel="Time (s)",
        title=(
            "All-source mixed Z record preview"
            if three_component
            else (
                "All-source mixed DAS record preview"
                if workflow.config.das.enabled
                else "All-source mixed receiver record preview"
            )
        ),
    )
    branches = observations.strategy_stacks["baseline_linear"].component_branches
    plot_ncf_branches(
        axes[1], branches["das" if workflow.config.das.enabled else "zz"],
        title=(
            f"DAS–DAS NCF ({observations.independent_window_count} windows)"
            if workflow.config.das.enabled
            else f"ZZ receiver NCF ({observations.independent_window_count} windows)"
        ),
    )
    if three_component:
        plot_ncf_branches(
            axes[2], branches["rr"],
            title=f"Pair-rotated RR NCF ({observations.independent_window_count} windows)",
        )
    figure.suptitle(
        f"Passive finite record {workflow.config.noise.duration_s:g} s | "
        f"independent window {workflow.config.noise.window_s:g} s | "
        f"two-sided NCF lag ±{observations.ncf.lag_time_s[-1]:.3f} s"
    )
    return figure


def plot_workflow_dispersion(workflow: WaveDispXWorkflow):
    analysis = workflow.compute_dispersion()
    if workflow.config.mode == "active":
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        if workflow.config.das.enabled:
            _image(
                axes[0],
                analysis.component_images["das"],
                analysis.theory.rayleigh_mps,
                "DAS Rayleigh",
            )
            _image(
                axes[1],
                analysis.component_images["zz"],
                analysis.theory.rayleigh_mps,
                "Node ZZ comparison",
            )
            axes[0].plot(
                analysis.rayleigh_combined_picks.frequency_hz,
                analysis.rayleigh_combined_picks.phase_velocity_mps,
                "k-",
                linewidth=1.5,
            )
            return figure
        _image(
            axes[0],
            analysis.rayleigh_combined,
            analysis.theory.rayleigh_mps,
            "Combined Rayleigh (RR/ZR/RZ/ZZ)",
        )
        _image(axes[1], analysis.love, analysis.theory.love_mps, "Love")
        axes[0].plot(
            analysis.rayleigh_combined_picks.frequency_hz,
            analysis.rayleigh_combined_picks.phase_velocity_mps,
            "k-",
            linewidth=1.5,
        )
        axes[1].plot(
            analysis.love_picks.frequency_hz,
            analysis.love_picks.phase_velocity_mps,
            "k-",
            linewidth=1.5,
        )
        return figure
    if "rr" in analysis.component_images:
        figure, axes = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)
        _image(axes[0], analysis.component_images["zz"], analysis.theory.rayleigh_mps, "ZZ")
        _image(axes[1], analysis.component_images["rr"], analysis.theory.rayleigh_mps, "RR")
        _image(axes[2], analysis.dispersion, analysis.theory.rayleigh_mps, "Combined Rayleigh")
        axis = axes[2]
    else:
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        _image(
            axis,
            analysis.dispersion,
            analysis.theory.rayleigh_mps,
            "Passive DAS Rayleigh" if workflow.config.das.enabled else "Passive Rayleigh",
        )
    axis.plot(
        analysis.picks.frequency_hz,
        analysis.picks.phase_velocity_mps,
        "k-",
        linewidth=1.5,
    )
    return figure
