"""Staged streaming finite-record ambient-noise modelling and processing."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Mapping

import numpy as np

from .config import ReceiverLineConfig, RunConfig
from .das import (
    DASGeometry,
    build_configured_das_geometry,
    temporal_derivative_transfer,
)
from .egf_reference import PassiveEgfReference, build_passive_egf_reference
from .forward import (
    compute_full_fk_green,
    compute_vertical_fk_green,
    synthesize_full_receivers,
    synthesize_vertical_receivers,
)
from .geometry import Geometry, build_geometry
from .noise import (
    add_independent_receiver_noise,
    source_spectral_amplitude,
    synthesize_mixed_noise_window_3c_band,
    synthesize_mixed_noise_window_band,
)
from .pair_proxy_selection import (
    PairCrossfitCoherenceWeights,
    fit_pair_crossfit_coherence_weights,
)
from .pair_tau_p_selection import (
    PairTauPSelection,
    fit_pair_crossfit_tau_p_weights,
    joint_diffuse_prearrival_hard_weights,
)
from .processing import (
    DispersionImage,
    DispersionPicks,
    NCFGather,
    NCFPairBranches,
    NCFPairGather,
    TheoreticalDispersion,
    bin_pair_ncf_branches,
    combine_dispersion_images,
    linear_array_ncf_branches,
    pick_dispersion_ridge,
    receiver_array_ncf_branches,
    receiver_array_3c_ncf_branches,
    receiver_pair_ncf_branches,
    surface_wave_dispersion,
    theoretical_dispersion,
)
from .nnws import (
    NNWSConfig as NNWSAdapterConfig,
    PairwiseNNWSResult,
    stack_pair_ncf_windows,
    stack_pair_ncf_windows_frequency_weighted,
    train_pairwise_nnws_strategy,
)
from .nnws_v2 import (
    NNWSV2Config as NNWSV2AdapterConfig,
    NNWSV2Result,
    blend_frequency_and_pair_weights,
    blend_shared_weights_with_directional_guard,
    train_shared_crossfit_nnws_v2,
)
from .stacking import (
    MultiStrategyNCFAccumulator,
    StackStatistics,
    StrategyNCFStack,
    reference_amplitude_candidate_phase,
)
from .source_selection import OracleSelectionPlan, build_oracle_axis_selection
from .source_activity import load_source_activity_matrix
from .survey_design import model_group_velocity_bounds
from .spz_selection import (
    PairSPZConfig as PairSPZAdapterConfig,
    PairSPZSelectionPlan,
    blend_physical_and_data_weights,
    build_oracle_pair_spz_selection,
    combine_physical_and_data_weights,
)
from .supervised_oracle import (
    SupervisedEgfOracleWeights,
    SupervisedPairEgfOracleWeights,
    fit_supervised_egf_oracle_weights,
    fit_supervised_pair_egf_oracle_weights,
)
from .survey_design import model_group_velocity_bounds
from .quality import (
    dispersion_quality_metrics,
    egf_equivalence_metrics,
    pick_error_metrics,
    theoretical_mode_support_metrics,
)


@dataclass(frozen=True)
class PassiveTransfer:
    frequency_hz: np.ndarray
    spectrum: np.ndarray
    response_sample_count: int
    delay_samples: int
    radial_grid_count: int
    radial_step_m: float
    interpolation_relative_l2: float
    component_names: tuple[str, ...]
    das_geometry: DASGeometry | None
    build_seconds: float

    @property
    def bytes(self) -> int:
        return self.spectrum.nbytes


@dataclass(frozen=True)
class PairWindowNCF:
    """One configured receiver pair retained before distance-bin averaging."""

    lag_time_s: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    symmetric: np.ndarray
    pair_i: int
    pair_j: int
    distance_m: float


@dataclass(frozen=True)
class PassiveCorrelation:
    preview_record: np.ndarray
    preview_time_s: np.ndarray
    ncf: NCFGather
    component_ncfs: dict[str, NCFGather]
    strategy_stacks: dict[str, StrategyNCFStack]
    source_selection: OracleSelectionPlan | None
    pairwise_nnws: PairwiseNNWSResult | None
    nnws_v2: NNWSV2Result | None
    pair_spz_selection: PairSPZSelectionPlan | None
    supervised_egf_oracle: SupervisedEgfOracleWeights | None
    supervised_pair_egf_oracle: SupervisedPairEgfOracleWeights | None
    pair_crossfit_coherence: PairCrossfitCoherenceWeights | None
    pair_tau_p_selection: PairTauPSelection | None
    pair_window_ncf: PairWindowNCF | None
    checkpoint_gathers: dict[int, np.ndarray]
    checkpoint_component_gathers: dict[str, dict[int, np.ndarray]]
    checkpoint_durations_s: dict[int, float]
    noise_event_catalog: np.ndarray | None
    independent_window_count: int
    correlation_seconds: float


@dataclass(frozen=True)
class PassiveAnalysis:
    dispersion: DispersionImage
    picks: DispersionPicks
    theory: TheoreticalDispersion
    quality: dict[str, dict[str, float | bool]]
    component_images: dict[str, DispersionImage]
    method_comparison: dict[str, DispersionImage]
    convergence: list[dict[str, object]]
    strategy_images: dict[str, DispersionImage]
    strategy_component_images: dict[str, dict[str, DispersionImage]]
    strategy_dispersion_quality: dict[
        str, dict[str, dict[str, float | int | bool]]
    ]
    strategy_egf_quality: dict[
        str, dict[str, dict[str, float | int | bool | str]]
    ]
    imaging_seconds: float


def _extract_pair_window_ncf(
    windows: list[NCFPairBranches],
    reference_receiver_index: int,
    target_receiver_index: int | None,
) -> PairWindowNCF | None:
    """Extract one pair's unstacked window CCFs for strict two-station QC."""

    if target_receiver_index is None:
        return None
    if not windows:
        raise ValueError("pair FTAN output requires pair-resolved NCF windows")
    template = windows[0]
    selected = (
        (
            (template.pair_i == reference_receiver_index)
            & (template.pair_j == target_receiver_index)
        )
        | (
            (template.pair_i == target_receiver_index)
            & (template.pair_j == reference_receiver_index)
        )
    )
    if np.count_nonzero(selected) != 1:
        raise ValueError(
            "pair_ftan_receiver_index must identify exactly one cached reference pair"
        )
    pair_index = int(np.flatnonzero(selected)[0])
    pair_i = int(template.pair_i[pair_index])
    pair_j = int(template.pair_j[pair_index])
    lag = template.lag_time_s
    for window in windows:
        if (
            not np.array_equal(window.lag_time_s, lag)
            or int(window.pair_i[pair_index]) != pair_i
            or int(window.pair_j[pair_index]) != pair_j
        ):
            raise ValueError("pair-resolved NCF windows are not aligned")
    return PairWindowNCF(
        lag_time_s=lag.copy(),
        positive=np.stack(
            [window.positive.data[:, pair_index] for window in windows]
        ).astype(np.float32),
        negative=np.stack(
            [window.negative.data[:, pair_index] for window in windows]
        ).astype(np.float32),
        symmetric=np.stack(
            [window.symmetric.data[:, pair_index] for window in windows]
        ).astype(np.float32),
        pair_i=pair_i,
        pair_j=pair_j,
        distance_m=float(template.distance_m[pair_index]),
    )


@dataclass(frozen=True)
class PassiveSimulation:
    geometry: Geometry
    transfer: PassiveTransfer
    egf_reference: PassiveEgfReference | None
    preview_record: np.ndarray
    preview_time_s: np.ndarray
    ncf: NCFGather
    component_ncfs: dict[str, NCFGather]
    strategy_stacks: dict[str, StrategyNCFStack]
    source_selection: OracleSelectionPlan | None
    pairwise_nnws: PairwiseNNWSResult | None
    nnws_v2: NNWSV2Result | None
    pair_spz_selection: PairSPZSelectionPlan | None
    supervised_egf_oracle: SupervisedEgfOracleWeights | None
    supervised_pair_egf_oracle: SupervisedPairEgfOracleWeights | None
    pair_crossfit_coherence: PairCrossfitCoherenceWeights | None
    pair_tau_p_selection: PairTauPSelection | None
    pair_window_ncf: PairWindowNCF | None
    dispersion: DispersionImage
    picks: DispersionPicks
    theory: TheoreticalDispersion
    quality: dict[str, dict[str, float | bool]]
    component_images: dict[str, DispersionImage]
    method_comparison: dict[str, DispersionImage]
    convergence: list[dict[str, object]]
    strategy_images: dict[str, DispersionImage]
    strategy_component_images: dict[str, dict[str, DispersionImage]]
    strategy_dispersion_quality: dict[
        str, dict[str, dict[str, float | int | bool]]
    ]
    strategy_egf_quality: dict[
        str, dict[str, dict[str, float | int | bool | str]]
    ]
    noise_event_catalog: np.ndarray | None
    independent_window_count: int
    timings: dict[str, float]


def build_passive_transfer(config: RunConfig, geometry: Geometry) -> PassiveTransfer:
    """Compute the vertical or full 3-C source-receiver transfer tensor."""
    if config.mode != "passive" or config.noise is None:
        raise ValueError("build_passive_transfer requires a passive configuration")
    noise = config.noise
    start = time.perf_counter()
    das_geometry = None
    if config.das.enabled:
        das_geometry = build_configured_das_geometry(
            geometry.receiver_x_m,
            geometry.receiver_y_m,
            geometry_mode=config.das.geometry_mode,
            gauge_length_m=config.das.gauge_length_m,
            fiber_azimuth_deg=config.das.fiber_azimuth_deg,
            receiver_coordinates_file=getattr(
                config.receivers, "coordinates_file", None
            ),
        )
        observation_xy = das_geometry.endpoint_xy_m
    else:
        observation_xy = np.column_stack(
            (geometry.receiver_x_m, geometry.receiver_y_m)
        )
    distance = np.hypot(
        observation_xy[:, 0, None] - geometry.source_x_m[None, :],
        observation_xy[:, 1, None] - geometry.source_y_m[None, :],
    )
    if np.any(distance <= 0):
        raise ValueError("ambient sources may not coincide with receivers")
    maximum_distance = float(np.ceil(distance.max()))
    reference_frequency = max(
        noise.minimum_frequency_hz, 0.25 * noise.dominant_frequency_hz
    )
    delay_s = 4.0 / reference_frequency
    minimum_group_velocity, _ = model_group_velocity_bounds(
        config.model,
        config.processing.frequency_min_hz,
        config.processing.frequency_max_hz,
        rayleigh_modes=config.processing.rayleigh_modes,
        love_modes=(
            config.processing.love_modes
            if config.ambient_sources.component == "three_component"
            else 0
        ),
    )
    response_duration = (
        1.15 * maximum_distance / minimum_group_velocity
        + 2.0 / config.processing.frequency_min_hz
        + delay_s
    )
    response_sample_count = int(np.floor(response_duration / noise.dt_s)) + 1
    if response_sample_count % 2:
        response_sample_count += 1
    delay_samples = int(delay_s / noise.dt_s)
    three_component = config.ambient_sources.component == "three_component"
    full_green = three_component or config.das.enabled
    green_options = dict(
        sample_count=response_sample_count,
        dt_s=noise.dt_s,
        maximum_distance_m=maximum_distance,
        delay_s=delay_s,
        maximum_frequency_hz=noise.maximum_frequency_hz,
        workers=config.workers,
    )
    fk_green = (
        compute_full_fk_green(config.model, **green_options)
        if full_green
        else compute_vertical_fk_green(config.model, **green_options)
    )
    window_samples = int(round(noise.window_s / noise.dt_s))
    full_frequency = np.fft.rfftfreq(window_samples, noise.dt_s)
    amplitude = source_spectral_amplitude(full_frequency, noise)
    frequency_index = np.flatnonzero(amplitude > 0)
    transfer_frequency = full_frequency[frequency_index]
    receiver_count = observation_xy.shape[0]
    source_count = geometry.source_x_m.size
    transfer_shape = (transfer_frequency.size, receiver_count, source_count)
    if full_green:
        transfer_shape += (5,)
    transfer = np.empty(transfer_shape, dtype=np.complex64)
    radial_step = config.radial_interpolation_step_m
    radial_start = max(
        radial_step, np.floor(distance.min() / radial_step) * radial_step
    )
    radial_stop = np.ceil(distance.max() / radial_step) * radial_step
    radial_count = int(round((radial_stop - radial_start) / radial_step)) + 1
    radial_grid = radial_start + radial_step * np.arange(radial_count)
    fastest_velocity = 1.5 * max(layer.vp_mps for layer in config.model.layers)
    radial_shape = (transfer_frequency.size, radial_count)
    if full_green:
        radial_shape += (5,)
    radial_spectrum = np.empty(radial_shape, dtype=np.complex128)
    radial_block_size = max(32, config.frequency_block_size * 8)
    for first_radius in range(0, radial_count, radial_block_size):
        last_radius = min(radial_count, first_radius + radial_block_size)
        radial_block = radial_grid[first_radius:last_radius]
        if full_green:
            response = synthesize_full_receivers(
                fk_green,
                radial_block,
                sample_count=response_sample_count,
                dt_s=noise.dt_s,
                distance_block_size=radial_block_size,
            )
            impulse = np.stack(
                (
                    response.horizontal_radial,
                    response.horizontal_tangential,
                    response.horizontal_vertical,
                    response.vertical_radial,
                    response.vertical_vertical,
                ),
                axis=2,
            )[delay_samples:]
        else:
            response = synthesize_vertical_receivers(
                fk_green,
                radial_block,
                sample_count=response_sample_count,
                dt_s=noise.dt_s,
                distance_block_size=radial_block_size,
                include_radial=False,
            )
            impulse = response.vertical[delay_samples:]
        if impulse.shape[0] > window_samples:
            raise ValueError(
                "noise.window_s is shorter than the farthest-source Green response; "
                f"use at least {impulse.shape[0] * noise.dt_s:.3f} s"
            )
        for column, offset in enumerate(radial_block):
            mute_samples = int(offset / fastest_velocity / noise.dt_s)
            impulse[:mute_samples, column] = 0.0
        radial_spectrum[:, first_radius:last_radius] = np.fft.rfft(
            impulse, n=window_samples, axis=0
        )[frequency_index]
    flattened_distance = distance.reshape(-1)
    for frequency_position in range(transfer_frequency.size):
        if full_green:
            for component_index in range(5):
                values = radial_spectrum[frequency_position, :, component_index]
                interpolated = np.interp(
                    flattened_distance, radial_grid, values.real
                ) + 1j * np.interp(flattened_distance, radial_grid, values.imag)
                transfer[frequency_position, :, :, component_index] = interpolated.reshape(
                    receiver_count, source_count
                )
        else:
            values = radial_spectrum[frequency_position]
            interpolated = np.interp(
                flattened_distance, radial_grid, values.real
            ) + 1j * np.interp(flattened_distance, radial_grid, values.imag)
            transfer[frequency_position] = interpolated.reshape(
                receiver_count, source_count
            )
    check_count = min(16, flattened_distance.size)
    check_index = np.linspace(
        0, flattened_distance.size - 1, check_count, dtype=int
    )
    check_distance = flattened_distance[check_index]
    if full_green:
        check_response = synthesize_full_receivers(
            fk_green,
            check_distance,
            sample_count=response_sample_count,
            dt_s=noise.dt_s,
            distance_block_size=check_count,
        )
        check_impulse = np.stack(
            (
                check_response.horizontal_radial,
                check_response.horizontal_tangential,
                check_response.horizontal_vertical,
                check_response.vertical_radial,
                check_response.vertical_vertical,
            ),
            axis=2,
        )[delay_samples:]
    else:
        check_response = synthesize_vertical_receivers(
            fk_green,
            check_distance,
            sample_count=response_sample_count,
            dt_s=noise.dt_s,
            distance_block_size=check_count,
            include_radial=False,
        )
        check_impulse = check_response.vertical[delay_samples:]
    for column, offset in enumerate(check_distance):
        mute_samples = int(offset / fastest_velocity / noise.dt_s)
        check_impulse[:mute_samples, column] = 0.0
    exact_check = np.fft.rfft(check_impulse, n=window_samples, axis=0)[
        frequency_index
    ]
    interpolated_check = (
        transfer.reshape(transfer_frequency.size, -1, 5)[:, check_index]
        if full_green
        else transfer.reshape(transfer_frequency.size, -1)[:, check_index]
    )
    interpolation_relative_l2 = float(
        np.linalg.norm(interpolated_check - exact_check)
        / max(np.linalg.norm(exact_check), np.finfo(float).eps)
    )
    component_names = (
        ("rr", "tt", "zr", "rz", "zz") if three_component else ("zz",)
    )
    if config.das.enabled:
        assert das_geometry is not None
        source_receiver_azimuth = np.arctan2(
            observation_xy[:, 1, None] - geometry.source_y_m[None, :],
            observation_xy[:, 0, None] - geometry.source_x_m[None, :],
        )
        endpoint_tangent = np.concatenate(
            (das_geometry.tangent_xy, das_geometry.tangent_xy), axis=0
        )
        axial_projection = (
            np.cos(source_receiver_azimuth) * endpoint_tangent[:, 0, None]
            + np.sin(source_receiver_azimuth) * endpoint_tangent[:, 1, None]
        )
        # Component index 3 is horizontal radial receiver displacement from a
        # vertical point force (RZ).  Projection and endpoint differencing are
        # applied before random sources are mixed and before any correlation.
        endpoint_axial = transfer[..., 3] * axial_projection[None, ...]
        channel_count = das_geometry.channel_count
        transfer = (
            endpoint_axial[:, channel_count:] - endpoint_axial[:, :channel_count]
        ) / das_geometry.gauge_length_by_channel_m[None, :, None]
        if config.das.measurement == "strain_rate":
            transfer *= temporal_derivative_transfer(transfer_frequency)[
                :, None, None
            ]
        transfer = transfer.astype(np.complex64)
        component_names = ("das",)
    return PassiveTransfer(
        transfer_frequency,
        transfer,
        response_sample_count,
        delay_samples,
        radial_count,
        radial_step,
        interpolation_relative_l2,
        component_names,
        das_geometry,
        time.perf_counter() - start,
    )


def accumulate_passive_correlation(
    config: RunConfig,
    geometry: Geometry,
    transfer: PassiveTransfer,
    *,
    strategy_weights_by_window: Mapping[str, np.ndarray] | None = None,
    egf_reference: PassiveEgfReference | None = None,
) -> PassiveCorrelation:
    """Generate one mixed realization and stream all weighted NCF strategies."""
    if config.mode != "passive" or config.noise is None:
        raise ValueError("accumulate_passive_correlation requires a passive case")
    noise = config.noise
    duration_ratio = noise.duration_s / noise.window_s
    window_count = int(round(duration_ratio))
    if not np.isclose(duration_ratio, window_count, rtol=0, atol=1e-10):
        raise ValueError("noise.duration_s must be an integer multiple of noise.window_s")
    supplied_weights = strategy_weights_by_window or {}
    if "baseline_linear" in supplied_weights:
        raise ValueError("baseline_linear is reserved for the all-window unit weights")
    strategy_weights: dict[str, np.ndarray] = {}
    for name, values in supplied_weights.items():
        if not name:
            raise ValueError("strategy names may not be empty")
        weights = np.asarray(values, dtype=np.float64).reshape(-1)
        if weights.size != window_count:
            raise ValueError(
                f"strategy {name!r} must provide exactly {window_count} window weights"
            )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("strategy weights must be finite and non-negative")
        strategy_weights[name] = weights
    source_selection = None
    if config.source_selection.enabled:
        source_selection = build_oracle_axis_selection(
            config, geometry, transfer.frequency_hz
        )
        for name, weights in source_selection.strategy_weights.items():
            if name in strategy_weights:
                raise ValueError(f"strategy name {name!r} is reserved by source selection")
            strategy_weights[name] = weights
    pair_spz_selection = None
    if config.pair_spz.enabled:
        pair_spz_selection = build_oracle_pair_spz_selection(
            config,
            geometry,
            transfer.frequency_hz,
            config=PairSPZAdapterConfig(
                frequency_bands_hz=config.source_selection.frequency_bands_hz,
                phase_tolerance_rad=config.pair_spz.phase_tolerance_rad,
                off_spz_weight=config.pair_spz.off_spz_weight,
                bilateral_balance_weight=(
                    config.pair_spz.bilateral_balance_weight
                ),
                effective_count_weight=config.pair_spz.effective_count_weight,
                minimum_weight=config.pair_spz.minimum_weight,
                minimum_effective_fraction=(
                    config.pair_spz.minimum_effective_fraction
                ),
                maximum_iterations=config.pair_spz.maximum_iterations,
                random_seed=config.pair_spz.random_seed,
                pair_scope=config.pair_spz.pair_scope,
                reference_receiver_index=(
                    config.pair_spz.reference_receiver_index
                ),
            ),
        )
        for name, weights in pair_spz_selection.strategy_weights.items():
            if name in strategy_weights:
                raise ValueError(f"strategy name {name!r} is reserved by pair-SPZ")
            strategy_weights[name] = weights
    source_activity = load_source_activity_matrix(
        config.ambient_sources, window_count
    )
    aperture = np.hypot(
        geometry.receiver_x_m[:, None] - geometry.receiver_x_m[None, :],
        geometry.receiver_y_m[:, None] - geometry.receiver_y_m[None, :],
    ).max()
    minimum_group_velocity_mps, _ = model_group_velocity_bounds(
        config.model,
        config.processing.frequency_min_hz,
        config.processing.frequency_max_hz,
        rayleigh_modes=config.processing.rayleigh_modes,
        love_modes=(
            config.processing.love_modes
            if config.ambient_sources.component == "three_component"
            else 0
        ),
    )
    automatic_maximum_lag_s = 1.3 * aperture / minimum_group_velocity_mps
    pair_resolved_enabled = config.nnws.enabled or config.nnws_v2.enabled
    if pair_resolved_enabled:
        automatic_maximum_lag_s = max(
            automatic_maximum_lag_s,
            aperture / config.processing.velocity_min_mps
            + 4.0 / config.processing.frequency_min_hz,
        )
    maximum_lag_s = (
        automatic_maximum_lag_s
        if noise.maximum_lag_s is None
        else noise.maximum_lag_s
    )
    if maximum_lag_s >= noise.window_s:
        raise ValueError(
            f"noise.window_s must exceed the required {maximum_lag_s:.3f} s maximum lag"
        )
    receiver_count = geometry.receiver_x_m.size
    pair_scope = (
        config.nnws.pair_scope if config.nnws.enabled else config.nnws_v2.pair_scope
    )
    reference_receiver_index = (
        config.nnws.reference_receiver_index
        if config.nnws.enabled
        else config.nnws_v2.reference_receiver_index
    )
    pair_count = (
        receiver_count - 1
        if pair_scope == "reference_receiver"
        else receiver_count * (receiver_count - 1) // 2
    )
    lag_count = int(round(maximum_lag_s / noise.dt_s)) + 1
    estimated_pair_cache_bytes = (
        window_count
        * pair_count
        * lag_count
        * 3
        * np.dtype(np.float32).itemsize
    )
    cache_limits = []
    if config.nnws.enabled:
        cache_limits.append(("nnws", config.nnws.maximum_cache_mb))
    if config.nnws_v2.enabled:
        cache_limits.append(("nnws_v2", config.nnws_v2.maximum_cache_mb))
    for cache_name, maximum_cache_mb in cache_limits:
        if estimated_pair_cache_bytes <= maximum_cache_mb * 1024**2:
            continue
        raise ValueError(
            "pair-resolved NNWS cache estimate is "
            f"{estimated_pair_cache_bytes / 1024**2:.1f} MiB, exceeding "
            f"{cache_name}.maximum_cache_mb={maximum_cache_mb:.1f}"
        )
    rng = np.random.default_rng(noise.seed)
    receiver_noise_rng = np.random.default_rng(noise.receiver_noise_seed)
    three_component = transfer.spectrum.ndim == 4
    primary_component = "das" if config.das.enabled else "zz"
    source_receiver_azimuth = np.arctan2(
        geometry.receiver_y_m[:, None] - geometry.source_y_m[None, :],
        geometry.receiver_x_m[:, None] - geometry.source_x_m[None, :],
    )
    accumulator = MultiStrategyNCFAccumulator(
        ("baseline_linear", *strategy_weights.keys())
    )
    preview_windows = min(
        window_count, max(1, int(round(60.0 / noise.window_s)))
    )
    preview = []
    event_catalog_rows: list[np.ndarray] = []
    checkpoint_durations = sorted(
        {
            duration
            for duration in (60.0, 600.0, noise.duration_s)
            if duration <= noise.duration_s
            and np.isclose(
                duration / noise.window_s, round(duration / noise.window_s)
            )
        }
    )
    checkpoint_windows = {
        int(round(duration / noise.window_s)): duration
        for duration in checkpoint_durations
    }
    component_snapshots: dict[str, dict[int, np.ndarray]] = {}
    pair_windows: list[NCFPairBranches] = []
    correlation_start = time.perf_counter()
    for window_index in range(window_count):
        if three_component:
            realization = synthesize_mixed_noise_window_3c_band(
                transfer.spectrum,
                transfer.frequency_hz,
                source_receiver_azimuth,
                noise,
                rng,
                return_source_events=noise.temporal_model == "poisson_ricker",
                source_block_size=config.source_block_size,
                source_activity_weights=source_activity[window_index],
                source_polarization=config.ambient_sources,
            )
        else:
            realization = synthesize_mixed_noise_window_band(
                transfer.spectrum,
                transfer.frequency_hz,
                noise,
                rng,
                return_source_events=noise.temporal_model == "poisson_ricker",
                source_block_size=config.source_block_size,
                source_activity_weights=source_activity[window_index],
            )
        mixed = add_independent_receiver_noise(
            realization.record, noise, receiver_noise_rng
        )
        if realization.source_events is not None and realization.source_events.size:
            source_index = realization.source_events["source_index"]
            source_x = np.full(source_index.size, np.nan, dtype=np.float64)
            source_y = np.full(source_index.size, np.nan, dtype=np.float64)
            independent = source_index >= 0
            source_x[independent] = geometry.source_x_m[source_index[independent]]
            source_y[independent] = geometry.source_y_m[source_index[independent]]
            time_in_window = realization.source_events["time_in_window_s"]
            columns = [
                np.full(source_index.size, window_index, dtype=np.float64),
                source_index.astype(np.float64),
            ]
            if "component_index" in realization.source_events.dtype.names:
                columns.append(
                    realization.source_events["component_index"].astype(np.float64)
                )
            columns.extend(
                (
                    source_x,
                    source_y,
                    time_in_window,
                    window_index * noise.window_s + time_in_window,
                    realization.source_events["weight"],
                )
            )
            event_catalog_rows.append(np.column_stack(columns))
        if window_index < preview_windows:
            preview.append(mixed.astype(np.float32))
        correlation_options = dict(
            window_s=noise.window_s,
            overlap=0.0,
            maximum_lag_s=maximum_lag_s,
            normalization=config.processing.correlation_normalization,
        )
        distance_bin = (
            config.receivers.spacing_m
            if isinstance(config.receivers, ReceiverLineConfig)
            else config.receivers.distance_bin_m
        )
        assert distance_bin is not None
        if three_component:
            window_ncfs = receiver_array_3c_ncf_branches(
                mixed,
                geometry.receiver_x_m,
                geometry.receiver_y_m,
                noise.dt_s,
                distance_bin_m=distance_bin,
                **correlation_options,
            )
        elif pair_resolved_enabled:
            pair_branches = receiver_pair_ncf_branches(
                mixed,
                geometry.receiver_x_m,
                geometry.receiver_y_m,
                noise.dt_s,
                reference_receiver_index=(
                    reference_receiver_index
                    if pair_scope == "reference_receiver"
                    else None
                ),
                **correlation_options,
            )
            pair_branches = NCFPairBranches(
                **{
                    branch_name: NCFPairGather(
                        data=getattr(pair_branches, branch_name).data.astype(
                            np.float32
                        ),
                        lag_time_s=getattr(pair_branches, branch_name).lag_time_s,
                        pair_i=getattr(pair_branches, branch_name).pair_i,
                        pair_j=getattr(pair_branches, branch_name).pair_j,
                        distance_m=getattr(pair_branches, branch_name).distance_m,
                        valid_windows=getattr(
                            pair_branches, branch_name
                        ).valid_windows,
                    )
                    for branch_name in ("positive", "negative", "symmetric")
                }
            )
            pair_windows.append(pair_branches)
            window_ncfs = {
                primary_component: bin_pair_ncf_branches(
                    pair_branches, distance_bin_m=distance_bin
                )
            }
        elif not isinstance(config.receivers, ReceiverLineConfig):
            window_ncfs = {
                primary_component: receiver_array_ncf_branches(
                    mixed,
                    geometry.receiver_x_m,
                    geometry.receiver_y_m,
                    noise.dt_s,
                    distance_bin_m=config.receivers.distance_bin_m,
                    **correlation_options,
                )
            }
        else:
            window_ncfs = {
                primary_component: linear_array_ncf_branches(
                    mixed, geometry.receiver_x_m, noise.dt_s, **correlation_options
                )
            }
        accumulator.add_window(
            window_ncfs,
            {
                "baseline_linear": 1.0,
                **{
                    name: weights[window_index]
                    for name, weights in strategy_weights.items()
                },
            },
        )
        for name in window_ncfs:
            component_snapshots.setdefault(name, {})
        completed_windows = window_index + 1
        if completed_windows in checkpoint_windows:
            baseline_snapshot = accumulator.snapshot("baseline_linear")
            for name, branches in baseline_snapshot.component_branches.items():
                component_snapshots[name][completed_windows] = (
                    branches.symmetric.data.copy()
                )
    strategy_stacks = accumulator.finalize()
    pairwise_nnws = None
    if config.nnws.enabled:
        pairwise_nnws = train_pairwise_nnws_strategy(
            pair_windows,
            dt_s=noise.dt_s,
            distance_bin_m=distance_bin,
            minimum_velocity_mps=config.processing.velocity_min_mps,
            maximum_velocity_mps=config.processing.velocity_max_mps,
            maximum_period_s=1.0 / config.processing.frequency_min_hz,
            config=NNWSAdapterConfig(
                epochs=config.nnws.epochs,
                learning_rate=config.nnws.learning_rate,
                snr_weight=config.nnws.snr_weight,
                symmetry_weight=config.nnws.symmetry_weight,
                std_weight=config.nnws.std_weight,
                std_mode=config.nnws.std_mode,
                seed=config.nnws.seed,
                device=config.nnws.device,
            ),
            minimum_effective_windows=config.nnws.minimum_effective_windows,
            random_control_seed=config.nnws.random_control_seed,
        )
        eligible = pairwise_nnws.eligible
        successful = eligible & pairwise_nnws.pair_stack.statistics.sufficient
        eligible_count = int(np.count_nonzero(eligible))
        pair_coverage = (
            float(np.count_nonzero(successful) / eligible_count)
            if eligible_count
            else 0.0
        )
        pair_gate = (
            eligible_count > 0
            and pair_coverage >= config.nnws.minimum_pair_coverage
        )

        def add_pair_strategy(
            name: str,
            pair_stack,
            binned_branches,
            *,
            statistics_mask: np.ndarray | None = None,
            require_pair_gate: bool = True,
        ) -> None:
            statistics = pair_stack.statistics
            retained = (
                successful
                if statistics_mask is None
                else np.asarray(statistics_mask, dtype=bool)
            ) & statistics.sufficient
            if np.any(retained):
                sum_weight = float(np.median(statistics.sum_weight[retained]))
                sum_weight_squared = float(
                    np.median(statistics.sum_weight_squared[retained])
                )
                effective = float(
                    np.median(statistics.effective_window_count[retained])
                )
                nonzero = int(
                    np.min(statistics.nonzero_window_count[retained])
                )
            else:
                sum_weight = 0.0
                sum_weight_squared = 0.0
                effective = 0.0
                nonzero = 0
            binned_finite = (
                binned_branches.symmetric.distance_m.size >= 2
                and all(
                    np.all(np.isfinite(getattr(binned_branches, branch).data))
                    for branch in ("positive", "negative", "symmetric")
                )
            )
            strategy_stacks[name] = StrategyNCFStack(
                component_branches={"zz": binned_branches},
                statistics=StackStatistics(
                    windows_seen=window_count,
                    nonzero_window_count=nonzero,
                    sum_weight=sum_weight,
                    sum_weight_squared=sum_weight_squared,
                    effective_window_count=effective,
                    sufficient=(not require_pair_gate or pair_gate)
                    and binned_finite,
                ),
            )

        add_pair_strategy(
            "nnws_common_linear",
            pairwise_nnws.common_linear_stack,
            pairwise_nnws.common_linear_binned,
        )
        for source_strategy_name, window_weights in strategy_weights.items():
            common_strategy_name = f"nnws_common_{source_strategy_name}"
            if common_strategy_name in strategy_stacks:
                raise ValueError(
                    f"strategy name {source_strategy_name!r} conflicts with "
                    "an NNWS common-support control"
                )
            common_weights = np.zeros_like(
                pairwise_nnws.pair_stack.statistics.weights
            )
            common_weights[:, successful] = np.asarray(
                window_weights, dtype=np.float64
            )[:, None]
            common_stack = stack_pair_ncf_windows(
                pair_windows,
                common_weights,
                minimum_effective_windows=(
                    config.nnws.minimum_effective_windows
                ),
            )
            add_pair_strategy(
                common_strategy_name,
                common_stack,
                bin_pair_ncf_branches(
                    common_stack.branches, distance_bin_m=distance_bin
                ),
            )
        add_pair_strategy(
            "nnws_random_neff_control",
            pairwise_nnws.random_control_stack,
            pairwise_nnws.random_control_binned,
        )
        add_pair_strategy(
            "nnws_pairwise",
            pairwise_nnws.pair_stack,
            pairwise_nnws.binned_branches,
        )
        full_aperture_mask = np.ones_like(successful, dtype=bool)
        add_pair_strategy(
            "nnws_hybrid_random_neff_control",
            pairwise_nnws.hybrid_random_control_stack,
            pairwise_nnws.hybrid_random_control_binned,
            statistics_mask=full_aperture_mask,
            require_pair_gate=False,
        )
        add_pair_strategy(
            "nnws_hybrid_full_aperture",
            pairwise_nnws.hybrid_full_aperture_stack,
            pairwise_nnws.hybrid_full_aperture_binned,
            statistics_mask=full_aperture_mask,
            require_pair_gate=False,
        )
    nnws_v2 = None
    supervised_egf_oracle = None
    supervised_pair_egf_oracle = None
    pair_crossfit_coherence = None
    pair_tau_p_selection = None
    if config.nnws_v2.enabled:
        nnws_v2 = train_shared_crossfit_nnws_v2(
            pair_windows,
            dt_s=noise.dt_s,
            distance_bin_m=distance_bin,
            minimum_velocity_mps=config.processing.velocity_min_mps,
            maximum_velocity_mps=config.processing.velocity_max_mps,
            maximum_period_s=1.0 / config.processing.frequency_min_hz,
            config=NNWSV2AdapterConfig(
                folds=config.nnws_v2.folds,
                epochs=config.nnws_v2.epochs,
                learning_rate=config.nnws_v2.learning_rate,
                arrival_weight=config.nnws_v2.arrival_weight,
                precursor_weight=config.nnws_v2.precursor_weight,
                symmetry_weight=config.nnws_v2.symmetry_weight,
                effective_count_weight=config.nnws_v2.effective_count_weight,
                temporal_smoothness_weight=(
                    config.nnws_v2.temporal_smoothness_weight
                ),
                minimum_weight=config.nnws_v2.minimum_weight,
                minimum_effective_fraction=(
                    config.nnws_v2.minimum_effective_fraction
                ),
                seed=config.nnws_v2.seed,
                device=config.nnws_v2.device,
            ),
            random_control_seed=config.nnws_v2.random_control_seed,
        )

        def add_nnws_v2_strategy(name: str, pair_stack, binned_branches) -> None:
            statistics = pair_stack.statistics
            retained = statistics.sufficient
            binned_finite = (
                binned_branches.symmetric.distance_m.size >= 2
                and all(
                    np.all(np.isfinite(getattr(binned_branches, branch).data))
                    for branch in ("positive", "negative", "symmetric")
                )
            )
            strategy_stacks[name] = StrategyNCFStack(
                component_branches={"zz": binned_branches},
                statistics=StackStatistics(
                    windows_seen=window_count,
                    nonzero_window_count=(
                        int(np.min(statistics.nonzero_window_count[retained]))
                        if np.any(retained)
                        else 0
                    ),
                    sum_weight=(
                        float(np.median(statistics.sum_weight[retained]))
                        if np.any(retained)
                        else 0.0
                    ),
                    sum_weight_squared=(
                        float(np.median(statistics.sum_weight_squared[retained]))
                        if np.any(retained)
                        else 0.0
                    ),
                    effective_window_count=(
                        float(
                            np.median(
                                statistics.effective_window_count[retained]
                            )
                        )
                        if np.any(retained)
                        else 0.0
                    ),
                    sufficient=bool(
                        nnws_v2.fit.sufficient
                        and np.all(statistics.sufficient)
                        and binned_finite
                    ),
                ),
            )

        add_nnws_v2_strategy(
            "nnws_v2_random_neff_control",
            nnws_v2.random_control_stack,
            nnws_v2.random_control_binned,
        )
        add_nnws_v2_strategy(
            "nnws_v2_shared_oof",
            nnws_v2.pair_stack,
            nnws_v2.binned_branches,
        )
        pair_crossfit_coherence = fit_pair_crossfit_coherence_weights(
            pair_windows,
            dt_s=noise.dt_s,
            frequency_range_hz=(
                config.processing.frequency_min_hz,
                config.processing.frequency_max_hz,
            ),
            minimum_velocity_mps=config.processing.velocity_min_mps,
            maximum_velocity_mps=config.processing.velocity_max_mps,
            folds=config.nnws_v2.folds,
            minimum_effective_fraction=(
                config.nnws_v2.minimum_effective_fraction
            ),
        )
        pair_crossfit_stack = stack_pair_ncf_windows(
            pair_windows,
            pair_crossfit_coherence.positive_weights,
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
            negative_weights=pair_crossfit_coherence.negative_weights,
        )
        add_nnws_v2_strategy(
            "pair_crossfit_coherence",
            pair_crossfit_stack,
            bin_pair_ncf_branches(
                pair_crossfit_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        pair_crossfit_permutation = np.random.default_rng(
            config.nnws_v2.random_control_seed + 47
        ).permutation(window_count)
        pair_crossfit_control_stack = stack_pair_ncf_windows(
            pair_windows,
            pair_crossfit_coherence.positive_weights[
                pair_crossfit_permutation
            ],
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
            negative_weights=pair_crossfit_coherence.negative_weights[
                pair_crossfit_permutation
            ],
        )
        add_nnws_v2_strategy(
            "pair_crossfit_coherence_permuted_time_control",
            pair_crossfit_control_stack,
            bin_pair_ncf_branches(
                pair_crossfit_control_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        pair_prearrival_stack = stack_pair_ncf_windows(
            pair_windows,
            pair_crossfit_coherence.positive_prearrival_weights,
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
            negative_weights=(
                pair_crossfit_coherence.negative_prearrival_weights
            ),
        )
        add_nnws_v2_strategy(
            "pair_prearrival_reweighting",
            pair_prearrival_stack,
            bin_pair_ncf_branches(
                pair_prearrival_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        pair_prearrival_control_stack = stack_pair_ncf_windows(
            pair_windows,
            pair_crossfit_coherence.positive_prearrival_weights[
                pair_crossfit_permutation
            ],
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
            negative_weights=(
                pair_crossfit_coherence.negative_prearrival_weights[
                    pair_crossfit_permutation
                ]
            ),
        )
        add_nnws_v2_strategy(
            "pair_prearrival_reweighting_permuted_time_control",
            pair_prearrival_control_stack,
            bin_pair_ncf_branches(
                pair_prearrival_control_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        pair_tau_p_selection = fit_pair_crossfit_tau_p_weights(
            pair_windows,
            dt_s=noise.dt_s,
            frequency_bands_hz=(
                config.source_selection.frequency_bands_hz
            ),
            minimum_velocity_mps=config.processing.velocity_min_mps,
            maximum_velocity_mps=config.processing.velocity_max_mps,
            phase_tolerance_rad=config.pair_spz.phase_tolerance_rad,
            folds=config.nnws_v2.folds,
            minimum_effective_fraction=(
                config.nnws_v2.minimum_effective_fraction
            ),
        )
        joint_diffuse_prearrival_hard = joint_diffuse_prearrival_hard_weights(
            pair_tau_p_selection.directional_imbalance,
            pair_crossfit_coherence.positive_prearrival_fraction,
            pair_crossfit_coherence.negative_prearrival_fraction,
            retention_fraction=config.nnws_v2.minimum_effective_fraction,
            directional_fraction=0.75,
        )
        pair_tau_p_selection = replace(
            pair_tau_p_selection,
            diffuse_prearrival_hard_band_weights=joint_diffuse_prearrival_hard,
            diffuse_prearrival_directional_fraction=0.75,
        )
        joint_diffuse_prearrival_stack = (
            stack_pair_ncf_windows_frequency_weighted(
                pair_windows,
                joint_diffuse_prearrival_hard,
                pair_tau_p_selection.band_center_hz,
                dt_s=noise.dt_s,
                minimum_effective_windows=(
                    config.nnws_v2.minimum_effective_fraction * window_count
                ),
            )
        )
        add_nnws_v2_strategy(
            "pair_crossfit_tau_p_diffuse_prearrival_hard_frequency",
            joint_diffuse_prearrival_stack,
            bin_pair_ncf_branches(
                joint_diffuse_prearrival_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        joint_diffuse_prearrival_control = joint_diffuse_prearrival_hard[
            pair_crossfit_permutation
        ]
        joint_diffuse_prearrival_control_stack = (
            stack_pair_ncf_windows_frequency_weighted(
                pair_windows,
                joint_diffuse_prearrival_control,
                pair_tau_p_selection.band_center_hz,
                dt_s=noise.dt_s,
                minimum_effective_windows=(
                    config.nnws_v2.minimum_effective_fraction * window_count
                ),
            )
        )
        add_nnws_v2_strategy(
            "pair_crossfit_tau_p_diffuse_prearrival_hard_frequency_"
            "permuted_time_control",
            joint_diffuse_prearrival_control_stack,
            bin_pair_ncf_branches(
                joint_diffuse_prearrival_control_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        directional_guard_weights = blend_shared_weights_with_directional_guard(
            nnws_v2.fit.weights,
            pair_tau_p_selection.diffuse_soft_band_weights,
            directional_fraction=0.8,
            minimum_effective_fraction=(
                config.nnws_v2.minimum_effective_fraction
            ),
        )
        directional_guard_control_weights = directional_guard_weights[
            pair_crossfit_permutation
        ]
        pair_tau_p_selection = replace(
            pair_tau_p_selection,
            nnws_v2_directional_guard_band_weights=directional_guard_weights,
            nnws_v2_directional_guard_control_band_weights=(
                directional_guard_control_weights
            ),
            nnws_v2_directional_fraction=0.8,
        )
        directional_guard_stack = stack_pair_ncf_windows_frequency_weighted(
            pair_windows,
            directional_guard_weights,
            pair_tau_p_selection.band_center_hz,
            dt_s=noise.dt_s,
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
        )
        add_nnws_v2_strategy(
            "nnws_v2_tau_p_directional_guard_frequency",
            directional_guard_stack,
            bin_pair_ncf_branches(
                directional_guard_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        directional_guard_control_stack = (
            stack_pair_ncf_windows_frequency_weighted(
                pair_windows,
                directional_guard_control_weights,
                pair_tau_p_selection.band_center_hz,
                dt_s=noise.dt_s,
                minimum_effective_windows=(
                    config.nnws_v2.minimum_effective_fraction * window_count
                ),
            )
        )
        add_nnws_v2_strategy(
            "nnws_v2_tau_p_directional_guard_frequency_permuted_time_control",
            directional_guard_control_stack,
            bin_pair_ncf_branches(
                directional_guard_control_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        positive_diffuse_prearrival_weights = blend_frequency_and_pair_weights(
            directional_guard_weights,
            pair_crossfit_coherence.positive_prearrival_weights,
            frequency_fraction=0.7,
            minimum_effective_fraction=(
                config.nnws_v2.minimum_effective_fraction
            ),
        )
        negative_diffuse_prearrival_weights = blend_frequency_and_pair_weights(
            directional_guard_weights,
            pair_crossfit_coherence.negative_prearrival_weights,
            frequency_fraction=0.7,
            minimum_effective_fraction=(
                config.nnws_v2.minimum_effective_fraction
            ),
        )
        positive_diffuse_prearrival_control = positive_diffuse_prearrival_weights[
            pair_crossfit_permutation
        ]
        negative_diffuse_prearrival_control = negative_diffuse_prearrival_weights[
            pair_crossfit_permutation
        ]
        pair_tau_p_selection = replace(
            pair_tau_p_selection,
            nnws_v2_positive_diffuse_prearrival_band_weights=(
                positive_diffuse_prearrival_weights
            ),
            nnws_v2_negative_diffuse_prearrival_band_weights=(
                negative_diffuse_prearrival_weights
            ),
            nnws_v2_positive_diffuse_prearrival_control_band_weights=(
                positive_diffuse_prearrival_control
            ),
            nnws_v2_negative_diffuse_prearrival_control_band_weights=(
                negative_diffuse_prearrival_control
            ),
            nnws_v2_diffuse_prearrival_frequency_fraction=0.7,
        )
        diffuse_prearrival_stack = stack_pair_ncf_windows_frequency_weighted(
            pair_windows,
            positive_diffuse_prearrival_weights,
            pair_tau_p_selection.band_center_hz,
            dt_s=noise.dt_s,
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
            negative_band_weights=negative_diffuse_prearrival_weights,
        )
        add_nnws_v2_strategy(
            "nnws_v2_tau_p_diffuse_prearrival_frequency",
            diffuse_prearrival_stack,
            bin_pair_ncf_branches(
                diffuse_prearrival_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        diffuse_prearrival_control_stack = (
            stack_pair_ncf_windows_frequency_weighted(
                pair_windows,
                positive_diffuse_prearrival_control,
                pair_tau_p_selection.band_center_hz,
                dt_s=noise.dt_s,
                minimum_effective_windows=(
                    config.nnws_v2.minimum_effective_fraction * window_count
                ),
                negative_band_weights=negative_diffuse_prearrival_control,
            )
        )
        add_nnws_v2_strategy(
            "nnws_v2_tau_p_diffuse_prearrival_frequency_permuted_time_control",
            diffuse_prearrival_control_stack,
            bin_pair_ncf_branches(
                diffuse_prearrival_control_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        positive_branch_guard_weights = blend_shared_weights_with_directional_guard(
            nnws_v2.fit.weights,
            pair_tau_p_selection.positive_band_weights,
            directional_fraction=0.8,
            minimum_effective_fraction=(
                config.nnws_v2.minimum_effective_fraction
            ),
        )
        negative_branch_guard_weights = blend_shared_weights_with_directional_guard(
            nnws_v2.fit.weights,
            pair_tau_p_selection.negative_band_weights,
            directional_fraction=0.8,
            minimum_effective_fraction=(
                config.nnws_v2.minimum_effective_fraction
            ),
        )
        positive_branch_guard_control_weights = positive_branch_guard_weights[
            pair_crossfit_permutation
        ]
        negative_branch_guard_control_weights = negative_branch_guard_weights[
            pair_crossfit_permutation
        ]
        pair_tau_p_selection = replace(
            pair_tau_p_selection,
            nnws_v2_positive_branch_guard_band_weights=(
                positive_branch_guard_weights
            ),
            nnws_v2_negative_branch_guard_band_weights=(
                negative_branch_guard_weights
            ),
            nnws_v2_positive_branch_guard_control_band_weights=(
                positive_branch_guard_control_weights
            ),
            nnws_v2_negative_branch_guard_control_band_weights=(
                negative_branch_guard_control_weights
            ),
        )
        branch_guard_stack = stack_pair_ncf_windows_frequency_weighted(
            pair_windows,
            positive_branch_guard_weights,
            pair_tau_p_selection.band_center_hz,
            dt_s=noise.dt_s,
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
            negative_band_weights=negative_branch_guard_weights,
        )
        add_nnws_v2_strategy(
            "nnws_v2_tau_p_branch_guard_frequency",
            branch_guard_stack,
            bin_pair_ncf_branches(
                branch_guard_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        branch_guard_control_stack = stack_pair_ncf_windows_frequency_weighted(
            pair_windows,
            positive_branch_guard_control_weights,
            pair_tau_p_selection.band_center_hz,
            dt_s=noise.dt_s,
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
            negative_band_weights=negative_branch_guard_control_weights,
        )
        add_nnws_v2_strategy(
            "nnws_v2_tau_p_branch_guard_frequency_permuted_time_control",
            branch_guard_control_stack,
            bin_pair_ncf_branches(
                branch_guard_control_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        pair_tau_p_stack = stack_pair_ncf_windows_frequency_weighted(
            pair_windows,
            pair_tau_p_selection.positive_band_weights,
            pair_tau_p_selection.band_center_hz,
            dt_s=noise.dt_s,
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
            negative_band_weights=(
                pair_tau_p_selection.negative_band_weights
            ),
        )
        add_nnws_v2_strategy(
            "pair_crossfit_tau_p_spz_frequency",
            pair_tau_p_stack,
            bin_pair_ncf_branches(
                pair_tau_p_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        pair_tau_p_control_stack = stack_pair_ncf_windows_frequency_weighted(
            pair_windows,
            pair_tau_p_selection.positive_band_weights[
                pair_crossfit_permutation
            ],
            pair_tau_p_selection.band_center_hz,
            dt_s=noise.dt_s,
            minimum_effective_windows=(
                config.nnws_v2.minimum_effective_fraction * window_count
            ),
            negative_band_weights=(
                pair_tau_p_selection.negative_band_weights[
                    pair_crossfit_permutation
                ]
            ),
        )
        add_nnws_v2_strategy(
            "pair_crossfit_tau_p_spz_frequency_permuted_time_control",
            pair_tau_p_control_stack,
            bin_pair_ncf_branches(
                pair_tau_p_control_stack.branches,
                distance_bin_m=distance_bin,
            ),
        )
        for diffuse_name, diffuse_weights in (
            (
                "pair_crossfit_tau_p_diffuse_soft_frequency",
                pair_tau_p_selection.diffuse_soft_band_weights,
            ),
            (
                "pair_crossfit_tau_p_diffuse_hard_frequency",
                pair_tau_p_selection.diffuse_hard_band_weights,
            ),
        ):
            diffuse_stack = stack_pair_ncf_windows_frequency_weighted(
                pair_windows,
                diffuse_weights,
                pair_tau_p_selection.band_center_hz,
                dt_s=noise.dt_s,
                minimum_effective_windows=(
                    config.nnws_v2.minimum_effective_fraction * window_count
                ),
            )
            add_nnws_v2_strategy(
                diffuse_name,
                diffuse_stack,
                bin_pair_ncf_branches(
                    diffuse_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            diffuse_control_stack = (
                stack_pair_ncf_windows_frequency_weighted(
                    pair_windows,
                    diffuse_weights[pair_crossfit_permutation],
                    pair_tau_p_selection.band_center_hz,
                    dt_s=noise.dt_s,
                    minimum_effective_windows=(
                        config.nnws_v2.minimum_effective_fraction
                        * window_count
                    ),
                )
            )
            add_nnws_v2_strategy(
                f"{diffuse_name}_permuted_time_control",
                diffuse_control_stack,
                bin_pair_ncf_branches(
                    diffuse_control_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
        if pair_spz_selection is not None:
            if (
                not np.array_equal(
                    pair_spz_selection.pair_i,
                    pair_windows[0].positive.pair_i,
                )
                or not np.array_equal(
                    pair_spz_selection.pair_j,
                    pair_windows[0].positive.pair_j,
                )
            ):
                raise ValueError("pair-SPZ and NCF receiver-pair order differ")
            minimum_effective = max(
                config.nnws_v2.minimum_effective_fraction,
                config.pair_spz.minimum_effective_fraction,
            ) * window_count
            pair_branch_spz_stack = stack_pair_ncf_windows(
                pair_windows,
                pair_spz_selection.pair_positive_branch_weights,
                minimum_effective_windows=minimum_effective,
                negative_weights=(
                    pair_spz_selection.pair_negative_branch_weights
                ),
            )
            add_nnws_v2_strategy(
                "oracle_pair_branch_spz",
                pair_branch_spz_stack,
                bin_pair_ncf_branches(
                    pair_branch_spz_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            pair_branch_spz_control_stack = stack_pair_ncf_windows(
                pair_windows,
                pair_spz_selection.pair_positive_branch_random_weights,
                minimum_effective_windows=minimum_effective,
                negative_weights=(
                    pair_spz_selection.pair_negative_branch_random_weights
                ),
            )
            add_nnws_v2_strategy(
                "oracle_pair_branch_spz_random_neff_control",
                pair_branch_spz_control_stack,
                bin_pair_ncf_branches(
                    pair_branch_spz_control_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            pair_band_center_hz = np.mean(
                pair_spz_selection.frequency_bands_hz, axis=1
            )
            pair_branch_frequency_stack = (
                stack_pair_ncf_windows_frequency_weighted(
                    pair_windows,
                    pair_spz_selection.pair_positive_branch_band_weights,
                    pair_band_center_hz,
                    dt_s=noise.dt_s,
                    minimum_effective_windows=minimum_effective,
                    negative_band_weights=(
                        pair_spz_selection.pair_negative_branch_band_weights
                    ),
                )
            )
            add_nnws_v2_strategy(
                "oracle_pair_branch_spz_frequency",
                pair_branch_frequency_stack,
                bin_pair_ncf_branches(
                    pair_branch_frequency_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            pair_branch_frequency_control_stack = (
                stack_pair_ncf_windows_frequency_weighted(
                    pair_windows,
                    pair_spz_selection.
                    pair_positive_branch_band_random_weights,
                    pair_band_center_hz,
                    dt_s=noise.dt_s,
                    minimum_effective_windows=minimum_effective,
                    negative_band_weights=(
                        pair_spz_selection.
                        pair_negative_branch_band_random_weights
                    ),
                )
            )
            add_nnws_v2_strategy(
                "oracle_pair_branch_spz_frequency_random_neff_control",
                pair_branch_frequency_control_stack,
                bin_pair_ncf_branches(
                    pair_branch_frequency_control_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            combined_pair_positive_band = np.empty_like(
                pair_spz_selection.pair_positive_branch_band_weights
            )
            combined_pair_negative_band = np.empty_like(
                pair_spz_selection.pair_negative_branch_band_weights
            )
            for band_index in range(pair_band_center_hz.size):
                for pair_index in range(pair_spz_selection.pair_count):
                    combined_pair_positive_band[:, band_index, pair_index] = (
                        blend_physical_and_data_weights(
                            nnws_v2.fit.weights,
                            pair_spz_selection.
                            pair_positive_branch_band_weights[
                                :, band_index, pair_index
                            ],
                            physical_fraction=0.20,
                            minimum_effective_fraction=max(
                                config.nnws_v2.minimum_effective_fraction,
                                config.pair_spz.minimum_effective_fraction,
                            ),
                        )
                    )
                    combined_pair_negative_band[:, band_index, pair_index] = (
                        blend_physical_and_data_weights(
                            nnws_v2.fit.weights,
                            pair_spz_selection.
                            pair_negative_branch_band_weights[
                                :, band_index, pair_index
                            ],
                            physical_fraction=0.20,
                            minimum_effective_fraction=max(
                                config.nnws_v2.minimum_effective_fraction,
                                config.pair_spz.minimum_effective_fraction,
                            ),
                        )
                    )
            combined_pair_frequency_stack = (
                stack_pair_ncf_windows_frequency_weighted(
                    pair_windows,
                    combined_pair_positive_band,
                    pair_band_center_hz,
                    dt_s=noise.dt_s,
                    minimum_effective_windows=minimum_effective,
                    negative_band_weights=combined_pair_negative_band,
                )
            )
            add_nnws_v2_strategy(
                "nnws_v2_pair_branch_spz_frequency_blend_020",
                combined_pair_frequency_stack,
                bin_pair_ncf_branches(
                    combined_pair_frequency_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            combined_pair_frequency_permutation = np.random.default_rng(
                config.pair_spz.random_seed + 4
            ).permutation(window_count)
            combined_pair_frequency_control_stack = (
                stack_pair_ncf_windows_frequency_weighted(
                    pair_windows,
                    combined_pair_positive_band[
                        combined_pair_frequency_permutation
                    ],
                    pair_band_center_hz,
                    dt_s=noise.dt_s,
                    minimum_effective_windows=minimum_effective,
                    negative_band_weights=combined_pair_negative_band[
                        combined_pair_frequency_permutation
                    ],
                )
            )
            add_nnws_v2_strategy(
                "nnws_v2_pair_branch_spz_frequency_blend_"
                "permuted_time_control_020",
                combined_pair_frequency_control_stack,
                bin_pair_ncf_branches(
                    combined_pair_frequency_control_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            combined_pair_positive = np.column_stack(
                [
                    blend_physical_and_data_weights(
                        nnws_v2.fit.weights,
                        pair_spz_selection.pair_positive_branch_weights[:, index],
                        physical_fraction=0.20,
                        minimum_effective_fraction=max(
                            config.nnws_v2.minimum_effective_fraction,
                            config.pair_spz.minimum_effective_fraction,
                        ),
                    )
                    for index in range(pair_spz_selection.pair_count)
                ]
            )
            combined_pair_negative = np.column_stack(
                [
                    blend_physical_and_data_weights(
                        nnws_v2.fit.weights,
                        pair_spz_selection.pair_negative_branch_weights[:, index],
                        physical_fraction=0.20,
                        minimum_effective_fraction=max(
                            config.nnws_v2.minimum_effective_fraction,
                            config.pair_spz.minimum_effective_fraction,
                        ),
                    )
                    for index in range(pair_spz_selection.pair_count)
                ]
            )
            combined_pair_stack = stack_pair_ncf_windows(
                pair_windows,
                combined_pair_positive,
                minimum_effective_windows=minimum_effective,
                negative_weights=combined_pair_negative,
            )
            add_nnws_v2_strategy(
                "nnws_v2_pair_branch_spz_blend_020",
                combined_pair_stack,
                bin_pair_ncf_branches(
                    combined_pair_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            combined_pair_permutation = np.random.default_rng(
                config.pair_spz.random_seed + 3
            ).permutation(window_count)
            combined_pair_control_stack = stack_pair_ncf_windows(
                pair_windows,
                combined_pair_positive[combined_pair_permutation],
                minimum_effective_windows=minimum_effective,
                negative_weights=(
                    combined_pair_negative[combined_pair_permutation]
                ),
            )
            add_nnws_v2_strategy(
                "nnws_v2_pair_branch_spz_blend_permuted_time_control_020",
                combined_pair_control_stack,
                bin_pair_ncf_branches(
                    combined_pair_control_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            combined_weights = combine_physical_and_data_weights(
                nnws_v2.fit.weights,
                pair_spz_selection.weights,
                minimum_effective_fraction=max(
                    config.nnws_v2.minimum_effective_fraction,
                    config.pair_spz.minimum_effective_fraction,
                ),
            )
            pair_count = pair_windows[0].positive.distance_m.size
            combined_matrix = np.repeat(
                combined_weights[:, None], pair_count, axis=1
            )
            combined_stack = stack_pair_ncf_windows(
                pair_windows,
                combined_matrix,
                minimum_effective_windows=minimum_effective,
            )
            combined_binned = bin_pair_ncf_branches(
                combined_stack.branches, distance_bin_m=distance_bin
            )
            combined_random_weights = np.random.default_rng(
                config.pair_spz.random_seed + 1
            ).permutation(combined_weights)
            combined_random_stack = stack_pair_ncf_windows(
                pair_windows,
                np.repeat(combined_random_weights[:, None], pair_count, axis=1),
                minimum_effective_windows=minimum_effective,
            )
            add_nnws_v2_strategy(
                "nnws_v2_spz_random_neff_control",
                combined_random_stack,
                bin_pair_ncf_branches(
                    combined_random_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            add_nnws_v2_strategy(
                "nnws_v2_spz_constrained",
                combined_stack,
                combined_binned,
            )
            for physical_fraction in (0.10, 0.20, 0.30, 0.40):
                diagnostic_weights = blend_physical_and_data_weights(
                    nnws_v2.fit.weights,
                    pair_spz_selection.weights,
                    physical_fraction=physical_fraction,
                    minimum_effective_fraction=max(
                        config.nnws_v2.minimum_effective_fraction,
                        config.pair_spz.minimum_effective_fraction,
                    ),
                )
                diagnostic_stack = stack_pair_ncf_windows(
                    pair_windows,
                    np.repeat(diagnostic_weights[:, None], pair_count, axis=1),
                    minimum_effective_windows=minimum_effective,
                )
                add_nnws_v2_strategy(
                    f"nnws_v2_spz_blend_{int(round(100 * physical_fraction)):03d}",
                    diagnostic_stack,
                    bin_pair_ncf_branches(
                        diagnostic_stack.branches,
                        distance_bin_m=distance_bin,
                    ),
                )
            band_center_hz = np.mean(
                pair_spz_selection.frequency_bands_hz, axis=1
            )
            for physical_fraction in (0.10, 0.20, 0.30, 0.40, 0.50):
                band_combined_weights = np.column_stack(
                    [
                        blend_physical_and_data_weights(
                            nnws_v2.fit.weights,
                            pair_spz_selection.band_weights[:, band_index],
                            physical_fraction=physical_fraction,
                            minimum_effective_fraction=max(
                                config.nnws_v2.minimum_effective_fraction,
                                config.pair_spz.minimum_effective_fraction,
                            ),
                        )
                        for band_index in range(band_center_hz.size)
                    ]
                )
                frequency_stack = stack_pair_ncf_windows_frequency_weighted(
                    pair_windows,
                    band_combined_weights,
                    band_center_hz,
                    dt_s=noise.dt_s,
                    minimum_effective_windows=minimum_effective,
                )
                frequency_strategy_name = (
                    "nnws_v2_spz_frequency_blend_"
                    f"{int(round(100 * physical_fraction)):03d}"
                )
                frequency_binned = bin_pair_ncf_branches(
                    frequency_stack.branches,
                    distance_bin_m=distance_bin,
                )
                add_nnws_v2_strategy(
                    frequency_strategy_name,
                    frequency_stack,
                    frequency_binned,
                )
                frequency_strategy = strategy_stacks[frequency_strategy_name]
                phase_hybrid_name = (
                    "nnws_v2_spz_phase_hybrid_"
                    f"{int(round(100 * physical_fraction)):03d}"
                )
                strategy_stacks[phase_hybrid_name] = StrategyNCFStack(
                    component_branches={
                        "zz": reference_amplitude_candidate_phase(
                            strategy_stacks["baseline_linear"].component_branches[
                                "zz"
                            ],
                            frequency_binned,
                        )
                    },
                    statistics=frequency_strategy.statistics,
                )
                if np.isclose(physical_fraction, 0.10, rtol=0.0, atol=1e-12):
                    branch_permutation = np.random.default_rng(
                        config.pair_spz.random_seed + 2
                    ).permutation(window_count)
                    positive_branch_weights_010 = np.column_stack(
                        [
                            blend_physical_and_data_weights(
                                nnws_v2.fit.weights,
                                pair_spz_selection.positive_branch_band_weights[
                                    :, band_index
                                ],
                                physical_fraction=physical_fraction,
                                minimum_effective_fraction=max(
                                    config.nnws_v2.minimum_effective_fraction,
                                    config.pair_spz.minimum_effective_fraction,
                                ),
                            )
                            for band_index in range(band_center_hz.size)
                        ]
                    )
                    negative_branch_weights_010 = np.column_stack(
                        [
                            blend_physical_and_data_weights(
                                nnws_v2.fit.weights,
                                pair_spz_selection.negative_branch_band_weights[
                                    :, band_index
                                ],
                                physical_fraction=physical_fraction,
                                minimum_effective_fraction=max(
                                    config.nnws_v2.minimum_effective_fraction,
                                    config.pair_spz.minimum_effective_fraction,
                                ),
                            )
                            for band_index in range(band_center_hz.size)
                        ]
                    )
                    branch_stack_010 = (
                        stack_pair_ncf_windows_frequency_weighted(
                            pair_windows,
                            positive_branch_weights_010,
                            band_center_hz,
                            dt_s=noise.dt_s,
                            minimum_effective_windows=minimum_effective,
                            negative_band_weights=negative_branch_weights_010,
                        )
                    )
                    branch_binned_010 = bin_pair_ncf_branches(
                        branch_stack_010.branches,
                        distance_bin_m=distance_bin,
                    )
                    add_nnws_v2_strategy(
                        "nnws_v2_spz_branch_frequency_blend_010",
                        branch_stack_010,
                        branch_binned_010,
                    )
                    branch_statistics_010 = strategy_stacks[
                        "nnws_v2_spz_branch_frequency_blend_010"
                    ].statistics
                    strategy_stacks[
                        "nnws_v2_spz_branch_phase_hybrid_010"
                    ] = StrategyNCFStack(
                        component_branches={
                            "zz": reference_amplitude_candidate_phase(
                                strategy_stacks[
                                    "baseline_linear"
                                ].component_branches["zz"],
                                branch_binned_010,
                            )
                        },
                        statistics=branch_statistics_010,
                    )
                    branch_control_stack_010 = (
                        stack_pair_ncf_windows_frequency_weighted(
                            pair_windows,
                            positive_branch_weights_010[branch_permutation],
                            band_center_hz,
                            dt_s=noise.dt_s,
                            minimum_effective_windows=minimum_effective,
                            negative_band_weights=(
                                negative_branch_weights_010[branch_permutation]
                            ),
                        )
                    )
                    branch_control_binned_010 = bin_pair_ncf_branches(
                        branch_control_stack_010.branches,
                        distance_bin_m=distance_bin,
                    )
                    strategy_stacks[
                        "nnws_v2_spz_branch_phase_hybrid_"
                        "permuted_time_control_010"
                    ] = StrategyNCFStack(
                        component_branches={
                            "zz": reference_amplitude_candidate_phase(
                                strategy_stacks[
                                    "baseline_linear"
                                ].component_branches["zz"],
                                branch_control_binned_010,
                            )
                        },
                        statistics=branch_statistics_010,
                    )
                if np.isclose(physical_fraction, 0.20, rtol=0.0, atol=1e-12):
                    permutation = np.random.default_rng(
                        config.pair_spz.random_seed + 2
                    ).permutation(window_count)
                    positive_branch_band_weights = np.column_stack(
                        [
                            blend_physical_and_data_weights(
                                nnws_v2.fit.weights,
                                pair_spz_selection.positive_branch_band_weights[
                                    :, band_index
                                ],
                                physical_fraction=physical_fraction,
                                minimum_effective_fraction=max(
                                    config.nnws_v2.minimum_effective_fraction,
                                    config.pair_spz.minimum_effective_fraction,
                                ),
                            )
                            for band_index in range(band_center_hz.size)
                        ]
                    )
                    negative_branch_band_weights = np.column_stack(
                        [
                            blend_physical_and_data_weights(
                                nnws_v2.fit.weights,
                                pair_spz_selection.negative_branch_band_weights[
                                    :, band_index
                                ],
                                physical_fraction=physical_fraction,
                                minimum_effective_fraction=max(
                                    config.nnws_v2.minimum_effective_fraction,
                                    config.pair_spz.minimum_effective_fraction,
                                ),
                            )
                            for band_index in range(band_center_hz.size)
                        ]
                    )
                    branch_stack = stack_pair_ncf_windows_frequency_weighted(
                        pair_windows,
                        positive_branch_band_weights,
                        band_center_hz,
                        dt_s=noise.dt_s,
                        minimum_effective_windows=minimum_effective,
                        negative_band_weights=negative_branch_band_weights,
                    )
                    branch_binned = bin_pair_ncf_branches(
                        branch_stack.branches,
                        distance_bin_m=distance_bin,
                    )
                    branch_strategy_name = (
                        "nnws_v2_spz_branch_frequency_blend_020"
                    )
                    add_nnws_v2_strategy(
                        branch_strategy_name,
                        branch_stack,
                        branch_binned,
                    )
                    branch_statistics = strategy_stacks[
                        branch_strategy_name
                    ].statistics
                    strategy_stacks[
                        "nnws_v2_spz_branch_phase_hybrid_020"
                    ] = StrategyNCFStack(
                        component_branches={
                            "zz": reference_amplitude_candidate_phase(
                                strategy_stacks[
                                    "baseline_linear"
                                ].component_branches["zz"],
                                branch_binned,
                            )
                        },
                        statistics=branch_statistics,
                    )
                    strategy_stacks[
                        "nnws_v2_spz_branch_v2_amplitude_phase_hybrid_020"
                    ] = StrategyNCFStack(
                        component_branches={
                            "zz": reference_amplitude_candidate_phase(
                                strategy_stacks[
                                    "nnws_v2_shared_oof"
                                ].component_branches["zz"],
                                branch_binned,
                            )
                        },
                        statistics=branch_statistics,
                    )
                    branch_control_stack = (
                        stack_pair_ncf_windows_frequency_weighted(
                            pair_windows,
                            positive_branch_band_weights[permutation],
                            band_center_hz,
                            dt_s=noise.dt_s,
                            minimum_effective_windows=minimum_effective,
                            negative_band_weights=(
                                negative_branch_band_weights[permutation]
                            ),
                        )
                    )
                    branch_control_binned = bin_pair_ncf_branches(
                        branch_control_stack.branches,
                        distance_bin_m=distance_bin,
                    )
                    add_nnws_v2_strategy(
                        "nnws_v2_spz_branch_frequency_permuted_time_control_020",
                        branch_control_stack,
                        branch_control_binned,
                    )
                    strategy_stacks[
                        "nnws_v2_spz_branch_phase_hybrid_"
                        "permuted_time_control_020"
                    ] = StrategyNCFStack(
                        component_branches={
                            "zz": reference_amplitude_candidate_phase(
                                strategy_stacks[
                                    "baseline_linear"
                                ].component_branches["zz"],
                                branch_control_binned,
                            )
                        },
                        statistics=branch_statistics,
                    )
                    strategy_stacks[
                        "nnws_v2_spz_branch_v2_amplitude_phase_hybrid_"
                        "permuted_time_control_020"
                    ] = StrategyNCFStack(
                        component_branches={
                            "zz": reference_amplitude_candidate_phase(
                                strategy_stacks[
                                    "nnws_v2_shared_oof"
                                ].component_branches["zz"],
                                branch_control_binned,
                            )
                        },
                        statistics=branch_statistics,
                    )
                    strategy_stacks[
                        "nnws_v2_spz_v2_amplitude_phase_hybrid_020"
                    ] = StrategyNCFStack(
                        component_branches={
                            "zz": reference_amplitude_candidate_phase(
                                strategy_stacks[
                                    "nnws_v2_shared_oof"
                                ].component_branches["zz"],
                                frequency_binned,
                            )
                        },
                        statistics=frequency_strategy.statistics,
                    )
                    control_band_weights = band_combined_weights[permutation]
                    control_stack = stack_pair_ncf_windows_frequency_weighted(
                        pair_windows,
                        control_band_weights,
                        band_center_hz,
                        dt_s=noise.dt_s,
                        minimum_effective_windows=minimum_effective,
                    )
                    control_binned = bin_pair_ncf_branches(
                        control_stack.branches,
                        distance_bin_m=distance_bin,
                    )
                    strategy_stacks[
                        "nnws_v2_spz_phase_hybrid_permuted_time_control_020"
                    ] = StrategyNCFStack(
                        component_branches={
                            "zz": reference_amplitude_candidate_phase(
                                strategy_stacks[
                                    "baseline_linear"
                                ].component_branches["zz"],
                                control_binned,
                            )
                        },
                        statistics=frequency_strategy.statistics,
                    )
                    strategy_stacks[
                        "nnws_v2_spz_v2_amplitude_phase_hybrid_"
                        "permuted_time_control_020"
                    ] = StrategyNCFStack(
                        component_branches={
                            "zz": reference_amplitude_candidate_phase(
                                strategy_stacks[
                                    "nnws_v2_shared_oof"
                                ].component_branches["zz"],
                                control_binned,
                            )
                        },
                        statistics=frequency_strategy.statistics,
                    )
                linear_band_weights = np.column_stack(
                    [
                        blend_physical_and_data_weights(
                            np.ones(window_count, dtype=np.float64),
                            pair_spz_selection.band_weights[:, band_index],
                            physical_fraction=physical_fraction,
                            minimum_effective_fraction=(
                                config.pair_spz.minimum_effective_fraction
                            ),
                        )
                        for band_index in range(band_center_hz.size)
                    ]
                )
                linear_frequency_stack = (
                    stack_pair_ncf_windows_frequency_weighted(
                        pair_windows,
                        linear_band_weights,
                        band_center_hz,
                        dt_s=noise.dt_s,
                        minimum_effective_windows=(
                            config.pair_spz.minimum_effective_fraction
                            * window_count
                        ),
                    )
                )
                add_nnws_v2_strategy(
                    "linear_spz_frequency_blend_"
                    f"{int(round(100 * physical_fraction)):03d}",
                    linear_frequency_stack,
                    bin_pair_ncf_branches(
                        linear_frequency_stack.branches,
                        distance_bin_m=distance_bin,
                    ),
                )
        if egf_reference is not None:
            binned_windows = [
                bin_pair_ncf_branches(
                    window,
                    distance_bin_m=distance_bin,
                )
                for window in pair_windows
            ]
            supervised_egf_oracle = fit_supervised_egf_oracle_weights(
                binned_windows,
                egf_reference,
                component_name="zz",
                frequency_range_hz=(
                    config.processing.frequency_min_hz,
                    config.processing.frequency_max_hz,
                ),
                minimum_velocity_mps=config.processing.velocity_min_mps,
                maximum_velocity_mps=1.5
                * max(layer.vp_mps for layer in config.model.layers),
                minimum_effective_fraction=(
                    config.nnws_v2.minimum_effective_fraction
                ),
            )
            oracle_center_hz = np.asarray(
                [
                    0.5
                    * (
                        config.processing.frequency_min_hz
                        + config.processing.frequency_max_hz
                    )
                ]
            )
            oracle_stack = stack_pair_ncf_windows_frequency_weighted(
                pair_windows,
                supervised_egf_oracle.positive[:, None],
                oracle_center_hz,
                dt_s=noise.dt_s,
                minimum_effective_windows=(
                    config.nnws_v2.minimum_effective_fraction * window_count
                ),
                negative_band_weights=supervised_egf_oracle.negative[:, None],
            )
            add_nnws_v2_strategy(
                "supervised_egf_oracle_upper_bound",
                oracle_stack,
                bin_pair_ncf_branches(
                    oracle_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            oracle_permutation = np.random.default_rng(
                config.nnws_v2.random_control_seed + 31
            ).permutation(window_count)
            oracle_control_stack = stack_pair_ncf_windows_frequency_weighted(
                pair_windows,
                supervised_egf_oracle.positive[oracle_permutation, None],
                oracle_center_hz,
                dt_s=noise.dt_s,
                minimum_effective_windows=(
                    config.nnws_v2.minimum_effective_fraction * window_count
                ),
                negative_band_weights=(
                    supervised_egf_oracle.negative[oracle_permutation, None]
                ),
            )
            add_nnws_v2_strategy(
                "supervised_egf_oracle_permuted_time_control",
                oracle_control_stack,
                bin_pair_ncf_branches(
                    oracle_control_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            supervised_pair_egf_oracle = (
                fit_supervised_pair_egf_oracle_weights(
                    pair_windows,
                    egf_reference,
                    component_name="zz",
                    frequency_range_hz=(
                        config.processing.frequency_min_hz,
                        config.processing.frequency_max_hz,
                    ),
                    minimum_velocity_mps=config.processing.velocity_min_mps,
                    maximum_velocity_mps=1.5
                    * max(layer.vp_mps for layer in config.model.layers),
                    minimum_effective_fraction=(
                        config.nnws_v2.minimum_effective_fraction
                    ),
                )
            )
            pair_oracle_stack = stack_pair_ncf_windows(
                pair_windows,
                supervised_pair_egf_oracle.positive,
                minimum_effective_windows=(
                    config.nnws_v2.minimum_effective_fraction * window_count
                ),
                negative_weights=supervised_pair_egf_oracle.negative,
            )
            add_nnws_v2_strategy(
                "supervised_pair_egf_oracle_upper_bound",
                pair_oracle_stack,
                bin_pair_ncf_branches(
                    pair_oracle_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
            pair_oracle_control_stack = stack_pair_ncf_windows(
                pair_windows,
                supervised_pair_egf_oracle.positive[oracle_permutation],
                minimum_effective_windows=(
                    config.nnws_v2.minimum_effective_fraction * window_count
                ),
                negative_weights=(
                    supervised_pair_egf_oracle.negative[oracle_permutation]
                ),
            )
            add_nnws_v2_strategy(
                "supervised_pair_egf_oracle_permuted_time_control",
                pair_oracle_control_stack,
                bin_pair_ncf_branches(
                    pair_oracle_control_stack.branches,
                    distance_bin_m=distance_bin,
                ),
            )
    pair_window_ncf = _extract_pair_window_ncf(
        pair_windows,
        config.nnws_v2.reference_receiver_index,
        config.nnws_v2.pair_ftan_receiver_index,
    )
    component_ncfs = {
        name: branches.symmetric
        for name, branches in strategy_stacks[
            "baseline_linear"
        ].component_branches.items()
    }
    ncf = component_ncfs[primary_component]
    preview_record = np.concatenate(preview, axis=0)
    preview_time = np.arange(preview_record.shape[0]) * noise.dt_s
    event_catalog = (
        np.concatenate(event_catalog_rows, axis=0)
        if event_catalog_rows
        else (
            np.empty((0, 8 if three_component else 7), dtype=np.float64)
            if noise.temporal_model == "poisson_ricker"
            else None
        )
    )
    return PassiveCorrelation(
        preview_record,
        preview_time,
        ncf=ncf,
        component_ncfs=component_ncfs,
        strategy_stacks=strategy_stacks,
        source_selection=source_selection,
        pairwise_nnws=pairwise_nnws,
        nnws_v2=nnws_v2,
        pair_spz_selection=pair_spz_selection,
        supervised_egf_oracle=supervised_egf_oracle,
        supervised_pair_egf_oracle=supervised_pair_egf_oracle,
        pair_crossfit_coherence=pair_crossfit_coherence,
        pair_tau_p_selection=pair_tau_p_selection,
        pair_window_ncf=pair_window_ncf,
        checkpoint_gathers=component_snapshots[primary_component],
        checkpoint_component_gathers=component_snapshots,
        checkpoint_durations_s=checkpoint_windows,
        noise_event_catalog=event_catalog,
        independent_window_count=window_count,
        correlation_seconds=time.perf_counter() - correlation_start,
    )


def analyze_passive_correlation(
    config: RunConfig,
    correlation: PassiveCorrelation,
    egf_reference: PassiveEgfReference | None = None,
) -> PassiveAnalysis:
    """Image, pick, compare with theory, and quantify duration convergence."""
    if config.mode != "passive" or config.noise is None:
        raise ValueError("analyze_passive_correlation requires a passive case")
    start = time.perf_counter()
    processing = config.processing
    primary_component = "das" if config.das.enabled else "zz"
    velocity_count = int(
        round(
            (processing.velocity_max_mps - processing.velocity_min_mps)
            / processing.velocity_step_mps
        )
    )
    velocity = processing.velocity_min_mps + processing.velocity_step_mps * np.arange(
        velocity_count + 1
    )

    imaging_options = dict(
        frequency_min_hz=processing.frequency_min_hz,
        frequency_max_hz=processing.frequency_max_hz,
        frequency_step_hz=processing.frequency_step_hz,
        zero_padding=processing.zero_padding,
        distance_taper_fraction=processing.distance_taper_fraction,
        frequency_block_size=config.frequency_block_size,
        radial_kernel_lookup_step=processing.radial_kernel_lookup_step,
        workers=config.workers,
    )
    pick_options = dict(
        frequency_range_hz=(processing.frequency_min_hz, processing.frequency_max_hz),
        velocity_range_mps=(
            processing.velocity_min_mps,
            processing.velocity_max_mps,
        ),
        smoothness=processing.picker_smoothness,
        maximum_jump_mps=processing.picker_maximum_jump_mps,
        minimum_energy=processing.picker_minimum_energy,
    )

    def image_gather(
        gather: np.ndarray,
        distance_m: np.ndarray,
        component: str,
        method: str,
    ) -> DispersionImage:
        return surface_wave_dispersion(
            gather,
            distance_m,
            config.noise.dt_s,
            velocity,
            method=method,
            component=component,
            **imaging_options,
        )

    def analyze_gathers(
        gathers: dict[str, np.ndarray],
        method: str,
        *,
        distance_m: np.ndarray | None = None,
    ):
        gather_distance = (
            correlation.ncf.distance_m if distance_m is None else distance_m
        )
        component_images = {
            primary_component: image_gather(
                gathers[primary_component], gather_distance, "vertical", method
            )
        }
        if "rr" in gathers:
            component_images["rr"] = image_gather(
                gathers["rr"], gather_distance, "radial", method
            )
        if "tt" in gathers:
            component_images["tt"] = image_gather(
                gathers["tt"], gather_distance, "transverse", method
            )
        image = (
            combine_dispersion_images(
                [component_images[primary_component], component_images["rr"]]
            )
            if "rr" in component_images
            else component_images[primary_component]
        )
        image_picks = pick_dispersion_ridge(
            image.energy, image.frequency_hz, velocity, **pick_options
        )
        image_theory = theoretical_dispersion(
            config.model,
            image.frequency_hz,
            rayleigh_modes=processing.rayleigh_modes,
            love_modes=processing.love_modes,
            backend=processing.theory_solver,
        )
        image_quality = {
            "image": dispersion_quality_metrics(
                image,
                image_theory.frequency_hz,
                image_theory.rayleigh_mps,
                frequency_range_hz=(
                    processing.frequency_min_hz,
                    processing.frequency_max_hz,
                ),
            ),
            "picks": pick_error_metrics(
                image_picks,
                image_theory.frequency_hz,
                image_theory.rayleigh_mps[:, 0],
            ),
        }
        for name, component_image in component_images.items():
            branch = "love_mps" if name == "tt" else "rayleigh_mps"
            image_quality[f"{name}_image"] = dispersion_quality_metrics(
                component_image,
                image_theory.frequency_hz,
                getattr(image_theory, branch),
                frequency_range_hz=(
                    processing.frequency_min_hz,
                    processing.frequency_max_hz,
                ),
            )
        return image, image_picks, image_theory, image_quality, component_images

    analyses = {
        count: analyze_gathers(
            {
                name: snapshots[count]
                for name, snapshots in correlation.checkpoint_component_gathers.items()
            },
            processing.dispersion_method,
        )
        for count in sorted(correlation.checkpoint_gathers)
    }
    final_count = correlation.independent_window_count
    dispersion, picks, theory, quality, component_images = analyses[final_count]
    method_comparison = {processing.dispersion_method: dispersion}
    if processing.compare_methods:
        final_gathers = {
            name: gather.data for name, gather in correlation.component_ncfs.items()
        }
        for method in ("phase_shift", "bessel", "hankel"):
            if method == processing.dispersion_method:
                continue
            method_comparison[method] = analyze_gathers(final_gathers, method)[0]
    quality["rayleigh_mode_support"] = theoretical_mode_support_metrics(
        dispersion, theory.frequency_hz, theory.rayleigh_mps
    )
    for method, image in method_comparison.items():
        quality[f"rayleigh_{method}_mode_support"] = theoretical_mode_support_metrics(
            image, theory.frequency_hz, theory.rayleigh_mps
        )
    strategy_images: dict[str, DispersionImage] = {}
    strategy_component_images: dict[str, dict[str, DispersionImage]] = {}
    strategy_dispersion_quality: dict[
        str, dict[str, dict[str, float | int | bool]]
    ] = {}
    strategy_egf_quality: dict[
        str, dict[str, dict[str, float | int | bool | str]]
    ] = {}
    for strategy_name, stack in correlation.strategy_stacks.items():
        if not stack.statistics.sufficient:
            strategy_dispersion_quality[strategy_name] = {
                "stack": {
                    "sufficient": False,
                    "effective_window_count": stack.statistics.effective_window_count,
                }
            }
            strategy_egf_quality[strategy_name] = {}
            continue
        if strategy_name == "baseline_linear":
            strategy_image = dispersion
            strategy_components = component_images
            strategy_quality = quality
        else:
            strategy_distances = {
                tuple(branches.symmetric.distance_m)
                for branches in stack.component_branches.values()
            }
            if len(strategy_distances) != 1:
                raise ValueError(
                    f"strategy {strategy_name!r} components use different distance axes"
                )
            strategy_distance = next(
                iter(stack.component_branches.values())
            ).symmetric.distance_m
            strategy_image, _, strategy_theory, strategy_quality, strategy_components = (
                analyze_gathers(
                    {
                        component: branches.symmetric.data
                        for component, branches in stack.component_branches.items()
                    },
                    processing.dispersion_method,
                    distance_m=strategy_distance,
                )
            )
            strategy_quality["rayleigh_mode_support"] = (
                theoretical_mode_support_metrics(
                    strategy_image,
                    strategy_theory.frequency_hz,
                    strategy_theory.rayleigh_mps,
                )
            )
        strategy_images[strategy_name] = strategy_image
        strategy_component_images[strategy_name] = strategy_components
        strategy_dispersion_quality[strategy_name] = strategy_quality
        egf_metrics: dict[str, dict[str, float | int | bool | str]] = {}
        if egf_reference is not None:
            for component, branches in stack.component_branches.items():
                if component not in egf_reference.component_names:
                    continue
                for branch_name in ("positive", "negative", "symmetric"):
                    egf_metrics[f"{component}_{branch_name}"] = (
                        egf_equivalence_metrics(
                            getattr(branches, branch_name),
                            egf_reference,
                            component_name=component,
                            frequency_range_hz=(
                                processing.frequency_min_hz,
                                processing.frequency_max_hz,
                            ),
                            minimum_velocity_mps=processing.velocity_min_mps,
                            maximum_velocity_mps=1.5
                            * max(layer.vp_mps for layer in config.model.layers),
                            correlation_normalization=(
                                processing.correlation_normalization
                            ),
                        )
                    )
        strategy_egf_quality[strategy_name] = egf_metrics
    convergence: list[dict[str, object]] = []
    final_ncf_norm = max(
        np.sqrt(
            sum(
                np.linalg.norm(gather.data) ** 2
                for gather in correlation.component_ncfs.values()
            )
        ),
        np.finfo(float).eps,
    )
    final_image_norm = max(np.linalg.norm(dispersion.energy), np.finfo(float).eps)
    for count, (image, image_picks, _, image_quality, _) in analyses.items():
        ncf_difference = np.sqrt(
            sum(
                np.linalg.norm(
                    snapshots[count] - correlation.component_ncfs[name].data
                )
                ** 2
                for name, snapshots in correlation.checkpoint_component_gathers.items()
            )
        )
        convergence.append(
            {
                "duration_s": correlation.checkpoint_durations_s[count],
                "window_count": count,
                "ncf_relative_l2_to_final": float(
                    ncf_difference / final_ncf_norm
                ),
                "dispersion_relative_l2_to_final": float(
                    np.linalg.norm(image.energy - dispersion.energy) / final_image_norm
                ),
                "pick_mae_to_final_mps": float(
                    np.mean(
                        np.abs(
                            image_picks.phase_velocity_mps
                            - picks.phase_velocity_mps
                        )
                    )
                ),
                "quality": image_quality,
            }
        )
    return PassiveAnalysis(
        dispersion=dispersion,
        picks=picks,
        theory=theory,
        quality=quality,
        component_images=component_images,
        method_comparison=method_comparison,
        convergence=convergence,
        strategy_images=strategy_images,
        strategy_component_images=strategy_component_images,
        strategy_dispersion_quality=strategy_dispersion_quality,
        strategy_egf_quality=strategy_egf_quality,
        imaging_seconds=time.perf_counter() - start,
    )


def simulate_passive(config: RunConfig) -> PassiveSimulation:
    """Compatibility one-call API composed from the public notebook stages."""
    if config.mode != "passive" or config.ambient_sources is None:
        raise ValueError("simulate_passive requires a passive configuration")
    overall_start = time.perf_counter()
    geometry = build_geometry(config.receivers, config.ambient_sources)
    transfer = build_passive_transfer(config, geometry)
    egf_reference = (
        None
        if config.das.enabled
        else build_passive_egf_reference(config, geometry)
    )
    correlation = accumulate_passive_correlation(
        config, geometry, transfer, egf_reference=egf_reference
    )
    analysis = analyze_passive_correlation(config, correlation, egf_reference)
    timings = {
        "transfer": transfer.build_seconds,
        "correlation": correlation.correlation_seconds,
        "imaging": analysis.imaging_seconds,
        "total": time.perf_counter() - overall_start,
    }
    if egf_reference is not None:
        timings["egf_reference"] = egf_reference.build_seconds
    return PassiveSimulation(
        geometry=geometry,
        transfer=transfer,
        egf_reference=egf_reference,
        preview_record=correlation.preview_record,
        preview_time_s=correlation.preview_time_s,
        ncf=correlation.ncf,
        component_ncfs=correlation.component_ncfs,
        strategy_stacks=correlation.strategy_stacks,
        source_selection=correlation.source_selection,
        pairwise_nnws=correlation.pairwise_nnws,
        nnws_v2=correlation.nnws_v2,
        pair_spz_selection=correlation.pair_spz_selection,
        supervised_egf_oracle=correlation.supervised_egf_oracle,
        supervised_pair_egf_oracle=correlation.supervised_pair_egf_oracle,
        pair_crossfit_coherence=correlation.pair_crossfit_coherence,
        pair_tau_p_selection=correlation.pair_tau_p_selection,
        pair_window_ncf=correlation.pair_window_ncf,
        dispersion=analysis.dispersion,
        picks=analysis.picks,
        theory=analysis.theory,
        quality=analysis.quality,
        component_images=analysis.component_images,
        method_comparison=analysis.method_comparison,
        convergence=analysis.convergence,
        strategy_images=analysis.strategy_images,
        strategy_component_images=analysis.strategy_component_images,
        strategy_dispersion_quality=analysis.strategy_dispersion_quality,
        strategy_egf_quality=analysis.strategy_egf_quality,
        noise_event_catalog=correlation.noise_event_catalog,
        independent_window_count=correlation.independent_window_count,
        timings=timings,
    )
