"""Stateful notebook workflow sharing the exact CLI computational stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from .active import ActiveAnalysis, ActiveShotGather, analyze_active_shot, simulate_active_shot
from .artifacts import RunArtifacts, save_active_results, save_passive_results
from .config import RunConfig, config_from_mapping, load_config
from .geometry import Geometry, build_geometry, receiver_positions
from .passive import (
    PassiveAnalysis,
    PassiveCorrelation,
    PassiveSimulation,
    PassiveTransfer,
    accumulate_passive_correlation,
    analyze_passive_correlation,
    build_passive_transfer,
)
from .plotting import plot_active_summary, plot_dispersion_comparison, plot_passive_summary
from .preflight import PreflightReview, build_preflight_review


def _hash_value(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        default=str,
        allow_nan=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _file_signature(path: Path | None) -> dict[str, object] | None:
    """Return a content-based signature for a user-supplied input file.

    A path alone is not a safe cache key: GUI users commonly upload a revised CSV
    under the same name.  Including the byte digest makes the staged workflow
    invalidate the affected Green functions/NCFs when the file content changes.
    """
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "name": resolved.name,
        "size": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _keys(config: RunConfig) -> tuple[str, str, str]:
    receiver_file = getattr(config.receivers, "coordinates_file", None)
    common_forward = {
        "mode": config.mode,
        "model": asdict(config.model),
        "receivers": asdict(config.receivers),
        "receiver_file": _file_signature(receiver_file),
        "workers": config.workers,
        "frequency_block_size": config.frequency_block_size,
    }
    if config.mode == "active":
        forward = _hash_value(
            {**common_forward, "active_source": asdict(config.active_source)}
        )
        observation = forward
    else:
        forward = _hash_value(
            {
                **common_forward,
                "ambient_sources": asdict(config.ambient_sources),
                "ambient_source_file": _file_signature(
                    config.ambient_sources.coordinates_file
                ),
                "noise_transfer": {
                    key: value
                    for key, value in asdict(config.noise).items()
                    if key
                    in {
                        "dt_s",
                        "window_s",
                        "psd_shape",
                        "dominant_frequency_hz",
                        "minimum_frequency_hz",
                        "maximum_frequency_hz",
                        "psd_file",
                    }
                },
                "psd_file": _file_signature(config.noise.psd_file),
                "radial_step": config.radial_interpolation_step_m,
            }
        )
        observation = _hash_value(
            {
                "forward": forward,
                "noise": asdict(config.noise),
                "source_block_size": config.source_block_size,
                "correlation_normalization": config.processing.correlation_normalization,
            }
        )
    analysis = _hash_value(
        {"observation": observation, "processing": asdict(config.processing)}
    )
    return forward, observation, analysis


class WaveDispXWorkflow:
    """WaveDispX staged workflow with visible memory cache and invalidation."""

    def __init__(self, config: RunConfig, *, source_mapping: dict[str, Any] | None = None):
        self.config = config
        self.source_mapping = source_mapping
        self.review: PreflightReview | None = None
        self.geometry: Geometry | None = None
        self.shot: ActiveShotGather | None = None
        self.transfer: PassiveTransfer | None = None
        self.correlation: PassiveCorrelation | None = None
        self.active_analysis: ActiveAnalysis | None = None
        self.passive_analysis: PassiveAnalysis | None = None
        self.artifacts: RunArtifacts | None = None
        self.timings: dict[str, float] = {}
        self._forward_key, self._observation_key, self._analysis_key = _keys(config)

    @classmethod
    def from_toml(cls, path: str | Path) -> "WaveDispXWorkflow":
        return cls(load_config(path))

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        base_directory: str | Path | None = None,
    ) -> "WaveDispXWorkflow":
        return cls(
            config_from_mapping(data, base_directory=base_directory),
            source_mapping=data,
        )

    def configure(
        self,
        config: RunConfig,
        *,
        source_mapping: dict[str, Any] | None = None,
    ) -> list[str]:
        """Apply a new case while retaining only dependency-safe memory cache."""
        new_forward, new_observation, new_analysis = _keys(config)
        invalidated: list[str] = []
        self.review = None
        self.geometry = None
        self.artifacts = None
        if new_forward != self._forward_key:
            self.shot = None
            self.transfer = None
            self.correlation = None
            self.active_analysis = None
            self.passive_analysis = None
            self.timings.clear()
            invalidated.extend(("forward", "observations", "analysis"))
        elif new_observation != self._observation_key:
            self.correlation = None
            self.active_analysis = None
            self.passive_analysis = None
            self.timings.pop("correlation", None)
            self.timings.pop("imaging", None)
            invalidated.extend(("observations", "analysis"))
        elif new_analysis != self._analysis_key:
            self.active_analysis = None
            self.passive_analysis = None
            self.timings.pop("imaging", None)
            invalidated.append("analysis")
        self.config = config
        if source_mapping is not None:
            self.source_mapping = source_mapping
        self._forward_key = new_forward
        self._observation_key = new_observation
        self._analysis_key = new_analysis
        return invalidated

    def review_inputs(self, *, force: bool = False) -> PreflightReview:
        if self.review is None or force:
            self.review = build_preflight_review(self.config)
            self.geometry = self.review.geometry
        return self.review

    def compute_forward(self, *, force: bool = False):
        """Active: complete shot forward. Passive: source-receiver transfer."""
        review = self.review_inputs()
        if not review.ready:
            raise RuntimeError("required preflight checks failed")
        if self.config.mode == "active":
            if self.shot is None or force:
                start = time.perf_counter()
                self.shot = simulate_active_shot(
                    self.config.model,
                    self.config.active_source,
                    self.config.receivers,
                    workers=self.config.workers,
                    distance_block_size=self.config.frequency_block_size * 8,
                )
                self.timings["forward"] = time.perf_counter() - start
                self.active_analysis = None
                self.artifacts = None
            return self.shot
        if self.transfer is None or force:
            self.transfer = build_passive_transfer(self.config, self.geometry)
            self.timings["forward"] = self.transfer.build_seconds
            self.correlation = None
            self.passive_analysis = None
            self.artifacts = None
        return self.transfer

    def simulate_observations(self, *, force: bool = False):
        """Active: return shot. Passive: stream mixed records into an NCF."""
        forward = self.compute_forward()
        if self.config.mode == "active":
            return forward
        if self.correlation is None or force:
            self.correlation = accumulate_passive_correlation(
                self.config, self.geometry, self.transfer
            )
            self.timings["correlation"] = self.correlation.correlation_seconds
            self.passive_analysis = None
            self.artifacts = None
        return self.correlation

    def compute_dispersion(self, *, force: bool = False):
        observations = self.simulate_observations()
        if self.config.mode == "active":
            if self.active_analysis is None or force:
                self.active_analysis = analyze_active_shot(self.config, self.shot)
                self.timings["imaging"] = self.active_analysis.imaging_seconds
                self.artifacts = None
            return self.active_analysis
        if self.passive_analysis is None or force:
            self.passive_analysis = analyze_passive_correlation(
                self.config, self.correlation
            )
            self.timings["imaging"] = self.passive_analysis.imaging_seconds
            self.artifacts = None
        return self.passive_analysis

    def save_results(self) -> RunArtifacts:
        analysis = self.compute_dispersion()
        output = self.config.output_directory
        summary_path = output / "summary.png"
        total = sum(self.timings.values())
        if self.config.mode == "active":
            plot_active_summary(
                self.config,
                self.geometry,
                self.shot,
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
                    **{
                        name: image
                        for name, image in analysis.component_images.items()
                        if name != "tt"
                    },
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
                "forward": self.timings.get("forward", 0.0),
                "imaging": self.timings.get("imaging", 0.0),
                "total": total,
            }
            self.artifacts = save_active_results(
                self.config,
                self.geometry,
                self.shot,
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
        else:
            timings = {
                "transfer": self.timings.get("forward", 0.0),
                "correlation": self.timings.get("correlation", 0.0),
                "imaging": self.timings.get("imaging", 0.0),
                "total": total,
            }
            simulation = PassiveSimulation(
                geometry=self.geometry,
                transfer=self.transfer,
                preview_record=self.correlation.preview_record,
                preview_time_s=self.correlation.preview_time_s,
                ncf=self.correlation.ncf,
                component_ncfs=self.correlation.component_ncfs,
                dispersion=analysis.dispersion,
                picks=analysis.picks,
                theory=analysis.theory,
                quality=analysis.quality,
                component_images=analysis.component_images,
                method_comparison=analysis.method_comparison,
                convergence=analysis.convergence,
                noise_event_catalog=self.correlation.noise_event_catalog,
                independent_window_count=self.correlation.independent_window_count,
                timings=timings,
            )
            plot_passive_summary(self.config, simulation, summary_path)
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
                    output / "passive_components.png",
                    title="Passive 3-C component sensitivity",
                )
            if len(simulation.method_comparison) > 1:
                plot_dispersion_comparison(
                    simulation.method_comparison,
                    simulation.theory.rayleigh_mps,
                    output / "passive_method_comparison.png",
                    title="Passive Rayleigh imaging-method comparison",
                )
            self.artifacts = save_passive_results(
                self.config, simulation, summary_png=summary_path
            )
        if self.source_mapping is not None:
            # Cell/UI cases have no original config_path to copy.  Persist the
            # user mapping with portable copies of every uploaded input.
            from .app_service import toml_dumps

            portable = deepcopy(self.source_mapping)
            receivers = portable.get("geometry", {}).get("receivers", {})
            if receivers.get("layout") == "file":
                receivers["coordinates_file"] = "receivers.csv"
            ambient = portable.get("ambient_sources", {})
            if ambient.get("layout") == "file":
                ambient["coordinates_file"] = "sources.csv"
            noise = portable.get("noise", {})
            if noise.get("psd_shape") == "file" and self.config.noise.psd_file:
                destination = self.artifacts.output_directory / "noise_psd.csv"
                source = self.config.noise.psd_file.resolve()
                if source != destination.resolve():
                    shutil.copy2(source, destination)
                noise["psd_file"] = "noise_psd.csv"
            (self.artifacts.output_directory / "input.toml").write_text(
                toml_dumps(portable),
                encoding="utf-8",
            )
        return self.artifacts

    def run_all(self, *, save: bool = True):
        self.review_inputs()
        self.compute_forward()
        self.simulate_observations()
        result = self.compute_dispersion()
        return self.save_results() if save else result

    def status(self) -> list[dict[str, object]]:
        if self.config.mode == "active":
            forward_ready = self.shot is not None
            observations_ready = forward_ready
            analysis_ready = self.active_analysis is not None
        else:
            forward_ready = self.transfer is not None
            observations_ready = self.correlation is not None
            analysis_ready = self.passive_analysis is not None
        return [
            {"stage": "inputs", "status": "ready" if self.review else "not run", "seconds": 0.0},
            {"stage": "forward", "status": "complete" if forward_ready else "not run", "seconds": self.timings.get("forward")},
            {"stage": "observations", "status": "complete" if observations_ready else "not run", "seconds": self.timings.get("correlation")},
            {"stage": "analysis", "status": "complete" if analysis_ready else "not run", "seconds": self.timings.get("imaging")},
            {"stage": "saved", "status": "complete" if self.artifacts else "not run", "seconds": None},
        ]

    @property
    def fingerprint(self) -> str:
        return self._analysis_key
