"""Active interreceiver Green-function references for passive simulations."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .config import RunConfig
from .forward import (
    compute_full_fk_green,
    compute_vertical_fk_green,
    synthesize_full_receivers,
    synthesize_vertical_receivers,
)
from .geometry import Geometry


@dataclass(frozen=True)
class PassiveEgfReference:
    """Band-limited unit-impulse Green functions on unique receiver offsets."""

    time_s: np.ndarray
    distance_m: np.ndarray
    components: np.ndarray
    component_names: tuple[str, ...]
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_distance_index: np.ndarray
    source_time_function: str
    calibration_operator: str
    delay_removed_s: float
    maximum_frequency_hz: float
    build_seconds: float

    @property
    def bytes(self) -> int:
        return self.components.nbytes


def build_passive_egf_reference(
    config: RunConfig,
    geometry: Geometry,
) -> PassiveEgfReference:
    """Compute the active-reference Green function between passive receivers.

    The returned wavefield is the raw DWM unit-impulse response.  The explicit
    ``calibration_operator`` records the interferometric ``-d/dt`` operation
    applied to an NCF before phase, arrival, and waveform comparison.  It does
    not establish an absolute-amplitude calibration constant.
    """
    if config.mode != "passive" or config.noise is None:
        raise ValueError("build_passive_egf_reference requires a passive configuration")
    receiver_x = np.asarray(geometry.receiver_x_m, dtype=np.float64).reshape(-1)
    receiver_y = np.asarray(geometry.receiver_y_m, dtype=np.float64).reshape(-1)
    if receiver_x.size < 2 or receiver_x.shape != receiver_y.shape:
        raise ValueError("at least two receiver coordinates are required")
    if np.any(~np.isfinite(receiver_x)) or np.any(~np.isfinite(receiver_y)):
        raise ValueError("receiver coordinates must be finite")

    pair_i, pair_j = np.triu_indices(receiver_x.size, k=1)
    pair_distance = np.hypot(
        receiver_x[pair_j] - receiver_x[pair_i],
        receiver_y[pair_j] - receiver_y[pair_i],
    )
    if np.any(pair_distance <= 0):
        raise ValueError("interreceiver Green functions require positive offsets")
    distance, pair_distance_index = np.unique(pair_distance, return_inverse=True)

    noise = config.noise
    start = time.perf_counter()
    reference_frequency = max(
        noise.minimum_frequency_hz,
        0.25 * noise.dominant_frequency_hz,
    )
    delay_s = 4.0 / reference_frequency
    response_duration_s = (
        2.0
        * float(distance[-1])
        / min(layer.vs_mps for layer in config.model.layers)
        / 0.862
        + 2.0 / reference_frequency
        + delay_s
    )
    response_sample_count = int(np.floor(response_duration_s / noise.dt_s)) + 1
    if response_sample_count % 2:
        response_sample_count += 1
    delay_samples = int(delay_s / noise.dt_s)
    green_options = dict(
        sample_count=response_sample_count,
        dt_s=noise.dt_s,
        maximum_distance_m=float(np.ceil(distance[-1])),
        delay_s=delay_s,
        maximum_frequency_hz=noise.maximum_frequency_hz,
        workers=config.workers,
    )
    three_component = config.ambient_sources.component == "three_component"
    if three_component:
        fk_green = compute_full_fk_green(config.model, **green_options)
        response = synthesize_full_receivers(
            fk_green,
            distance,
            sample_count=response_sample_count,
            dt_s=noise.dt_s,
            distance_block_size=max(32, config.frequency_block_size * 8),
        )
        components = np.stack(
            (
                response.horizontal_radial,
                response.horizontal_tangential,
                response.horizontal_vertical,
                response.vertical_radial,
                response.vertical_vertical,
            ),
            axis=2,
        )[delay_samples:]
        component_names = ("rr", "tt", "zr", "rz", "zz")
    else:
        fk_green = compute_vertical_fk_green(config.model, **green_options)
        response = synthesize_vertical_receivers(
            fk_green,
            distance,
            sample_count=response_sample_count,
            dt_s=noise.dt_s,
            distance_block_size=max(32, config.frequency_block_size * 8),
            include_radial=False,
        )
        components = response.vertical[delay_samples:, :, None]
        component_names = ("zz",)

    fastest_velocity_mps = 1.5 * max(
        layer.vp_mps for layer in config.model.layers
    )
    for column, offset_m in enumerate(distance):
        mute_samples = int(
            np.ceil(offset_m / fastest_velocity_mps / noise.dt_s)
        )
        components[:mute_samples, column] = 0.0
    output_time = np.arange(components.shape[0], dtype=np.float64) * noise.dt_s
    return PassiveEgfReference(
        time_s=output_time,
        distance_m=distance,
        components=components,
        component_names=component_names,
        pair_i=pair_i.astype(np.int64, copy=False),
        pair_j=pair_j.astype(np.int64, copy=False),
        pair_distance_index=pair_distance_index.astype(np.int64, copy=False),
        source_time_function="unit_impulse",
        calibration_operator="negative_time_derivative_of_ncf",
        delay_removed_s=delay_samples * noise.dt_s,
        maximum_frequency_hz=noise.maximum_frequency_hz,
        build_seconds=time.perf_counter() - start,
    )
