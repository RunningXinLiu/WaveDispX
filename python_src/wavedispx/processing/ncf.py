"""Finite-record receiver correlation for a uniformly sampled line array."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.signal import detrend
from scipy.signal.windows import tukey


Normalization = Literal["spectral_whitening", "one_bit", "none"]
Branch = Literal["symmetric", "positive", "negative"]
BranchRequest = Literal["symmetric", "positive", "negative", "all"]


@dataclass(frozen=True)
class NCFGather:
    data: np.ndarray
    lag_time_s: np.ndarray
    distance_m: np.ndarray
    pair_count: np.ndarray
    valid_windows: int


@dataclass(frozen=True)
class NCFBranches:
    """Positive, negative, and symmetric views from one correlation pass."""

    positive: NCFGather
    negative: NCFGather
    symmetric: NCFGather


@dataclass(frozen=True)
class NCFPairGather:
    """One correlation trace per identified receiver pair before binning."""

    data: np.ndarray
    lag_time_s: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    distance_m: np.ndarray
    valid_windows: int


@dataclass(frozen=True)
class NCFPairBranches:
    """Positive, negative, and symmetric pair-resolved correlations."""

    positive: NCFPairGather
    negative: NCFPairGather
    symmetric: NCFPairGather

    @property
    def pair_i(self) -> np.ndarray:
        return self.positive.pair_i

    @property
    def pair_j(self) -> np.ndarray:
        return self.positive.pair_j

    @property
    def distance_m(self) -> np.ndarray:
        return self.positive.distance_m

    @property
    def lag_time_s(self) -> np.ndarray:
        return self.positive.lag_time_s

    @property
    def valid_windows(self) -> int:
        return self.positive.valid_windows


def _power_of_two_at_least(value: int) -> int:
    return 1 << max(0, (int(value) - 1).bit_length())


def _normalize_trace(trace: np.ndarray, method: Normalization) -> np.ndarray:
    if method == "none":
        return trace
    if method == "one_bit":
        return np.sign(trace)
    if method != "spectral_whitening":
        raise ValueError(f"unsupported normalization: {method}")
    nfft = _power_of_two_at_least(trace.size)
    spectrum = np.fft.fft(trace, nfft)
    amplitude = np.abs(spectrum)
    water_level = 0.01 * amplitude.max(initial=0.0)
    if water_level == 0:
        return trace.copy()
    whitened_amplitude = np.ones_like(amplitude)
    below = amplitude < water_level
    whitened_amplitude[below] = amplitude[below] / water_level
    result = np.fft.ifft(whitened_amplitude * np.exp(1j * np.angle(spectrum))).real
    return result[: trace.size]


def linear_array_ncf(
    data: np.ndarray,
    receiver_x_m: np.ndarray,
    dt_s: float,
    *,
    window_s: float | None = None,
    overlap: float = 0.5,
    maximum_lag_s: float | None = None,
    normalization: Normalization = "spectral_whitening",
    branch: BranchRequest = "symmetric",
) -> NCFGather | NCFBranches:
    """Correlate all pairs and stack directly by distance using spatial FFT.

    The input is a finite mixed receiver record, time by receiver. No source
    positions or source identities are accepted by this function.
    """
    traces = np.asarray(data, dtype=np.float64)
    receiver_x = np.asarray(receiver_x_m, dtype=np.float64).reshape(-1)
    if traces.ndim != 2 or traces.shape[1] != receiver_x.size:
        raise ValueError("data must be time-by-receiver and match receiver_x_m")
    if traces.shape[0] < 2 or receiver_x.size < 2 or not np.all(np.isfinite(traces)):
        raise ValueError("at least two finite samples and receivers are required")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    spacing = np.diff(receiver_x)
    tolerance = max(1.0, float(np.abs(receiver_x).max())) * 1e-10
    if np.any(spacing <= 0) or np.max(np.abs(spacing - spacing[0])) > tolerance:
        raise ValueError("linear_array_ncf requires increasing equally spaced receivers")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if branch not in {"symmetric", "positive", "negative", "all"}:
        raise ValueError("branch must be symmetric, positive, negative, or all")

    npts, receiver_count = traces.shape
    total_time = (npts - 1) * dt_s
    if window_s is None:
        window_s = total_time
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    window_samples = min(npts, max(2, int(round(window_s / dt_s))))
    step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    if maximum_lag_s is None:
        maximum_lag_s = 0.5 * window_s
    maximum_lag_samples = min(int(round(maximum_lag_s / dt_s)), window_samples - 1)
    if maximum_lag_samples < 0:
        raise ValueError("maximum_lag_s must be non-negative")
    window_count = max(1, math.floor((npts - window_samples) / step_samples) + 1)
    temporal_nfft = _power_of_two_at_least(2 * window_samples - 1)
    spatial_nfft = _power_of_two_at_least(2 * receiver_count - 1)
    pair_count = receiver_count - np.arange(1, receiver_count)
    distance = np.arange(1, receiver_count, dtype=np.float64) * spacing[0]
    gather_shape = (maximum_lag_samples + 1, receiver_count - 1)
    gather = np.zeros(gather_shape, dtype=np.float64)
    branch_gathers = (
        {name: np.zeros(gather_shape, dtype=np.float64) for name in (
            "positive", "negative", "symmetric"
        )}
        if branch == "all"
        else None
    )
    time_taper = tukey(window_samples, 0.1).reshape(-1, 1)
    valid_windows = 0

    for window_index in range(window_count):
        first = window_index * step_samples
        segment = traces[first : first + window_samples].copy()
        if np.any(np.all(segment == 0, axis=0)):
            continue
        for receiver_index in range(receiver_count):
            segment[:, receiver_index] = _normalize_trace(
                segment[:, receiver_index], normalization
            )
        segment = detrend(segment, axis=0, type="linear") * time_taper
        if np.any(np.std(segment, axis=0) == 0):
            continue
        temporal_spectrum = np.fft.fft(segment, temporal_nfft, axis=0)
        spatial_spectrum = np.fft.fft(temporal_spectrum, spatial_nfft, axis=1)
        spatial_correlation = np.fft.ifft(
            spatial_spectrum * np.conj(spatial_spectrum), axis=1
        )
        distance_cross_spectrum = np.conj(
            spatial_correlation[:, 1:receiver_count]
        ) / pair_count
        correlation = np.fft.ifft(distance_cross_spectrum, temporal_nfft, axis=0).real
        positive = correlation[: maximum_lag_samples + 1]
        if maximum_lag_samples:
            negative = np.vstack(
                (correlation[0], correlation[-1 : -maximum_lag_samples - 1 : -1])
            )
        else:
            negative = correlation[:1]
        unbiased = (window_samples - np.arange(maximum_lag_samples + 1)).reshape(-1, 1)
        positive = positive / unbiased
        negative = negative / unbiased
        if branch == "all":
            branch_gathers["positive"] += positive
            branch_gathers["negative"] += negative
            branch_gathers["symmetric"] += 0.5 * (positive + negative)
            window_gather = None
        elif branch == "positive":
            window_gather = positive
        elif branch == "negative":
            window_gather = negative
        else:
            window_gather = 0.5 * (positive + negative)
        if window_gather is not None:
            gather += window_gather
        valid_windows += 1

    if valid_windows and branch_gathers is not None:
        for branch_gather in branch_gathers.values():
            branch_gather /= valid_windows
    elif valid_windows:
        gather /= valid_windows
    lag_time = np.arange(maximum_lag_samples + 1, dtype=np.float64) * dt_s
    if branch_gathers is not None:
        return NCFBranches(
            **{
                name: NCFGather(
                    branch_gather,
                    lag_time,
                    distance,
                    pair_count,
                    valid_windows,
                )
                for name, branch_gather in branch_gathers.items()
            }
        )
    return NCFGather(gather, lag_time, distance, pair_count, valid_windows)


def linear_array_ncf_branches(
    data: np.ndarray,
    receiver_x_m: np.ndarray,
    dt_s: float,
    *,
    window_s: float | None = None,
    overlap: float = 0.5,
    maximum_lag_s: float | None = None,
    normalization: Normalization = "spectral_whitening",
) -> NCFBranches:
    """Return all lag branches while correlating the receiver record once."""
    result = linear_array_ncf(
        data,
        receiver_x_m,
        dt_s,
        window_s=window_s,
        overlap=overlap,
        maximum_lag_s=maximum_lag_s,
        normalization=normalization,
        branch="all",
    )
    assert isinstance(result, NCFBranches)
    return result


def receiver_array_ncf(
    data: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    dt_s: float,
    *,
    distance_bin_m: float,
    window_s: float | None = None,
    overlap: float = 0.5,
    maximum_lag_s: float | None = None,
    normalization: Normalization = "spectral_whitening",
    branch: BranchRequest = "symmetric",
) -> NCFGather | NCFBranches:
    """Correlate an arbitrary 2-D receiver array and stack by distance bins.

    Pair azimuth is discarded, so the result is a radial average appropriate
    for a 1-D isotropic model. Directional processing requires a separate
    azimuth-binned workflow.
    """
    traces = np.asarray(data, dtype=np.float64)
    receiver_x = np.asarray(receiver_x_m, dtype=np.float64).reshape(-1)
    receiver_y = np.asarray(receiver_y_m, dtype=np.float64).reshape(-1)
    if (
        traces.ndim != 2
        or traces.shape[1] != receiver_x.size
        or receiver_x.shape != receiver_y.shape
    ):
        raise ValueError("data must be time-by-receiver and match x/y coordinates")
    if receiver_x.size < 2 or np.any(~np.isfinite(receiver_x)) or np.any(~np.isfinite(receiver_y)):
        raise ValueError("at least two finite receiver coordinates are required")
    if not np.isfinite(distance_bin_m) or distance_bin_m <= 0:
        raise ValueError("distance_bin_m must be finite and positive")
    if branch not in {"symmetric", "positive", "negative", "all"}:
        raise ValueError("branch must be symmetric, positive, negative, or all")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    # Preserve the exact and faster spatial-FFT implementation whenever the
    # supplied coordinates are an equally spaced line with matching bins.
    order = np.lexsort((receiver_y, receiver_x))
    sorted_x = receiver_x[order]
    sorted_y = receiver_y[order]
    spacing = np.diff(sorted_x)
    tolerance = max(1.0, float(np.abs(sorted_x).max())) * 1e-10
    if (
        np.max(np.abs(sorted_y - sorted_y[0])) <= tolerance
        and spacing.size > 0
        and np.all(spacing > 0)
        and np.max(np.abs(spacing - spacing[0])) <= tolerance
        and np.isclose(distance_bin_m, spacing[0], rtol=0, atol=tolerance)
    ):
        return linear_array_ncf(
            traces[:, order],
            sorted_x,
            dt_s,
            window_s=window_s,
            overlap=overlap,
            maximum_lag_s=maximum_lag_s,
            normalization=normalization,
            branch=branch,
        )

    if traces.shape[0] < 2 or np.any(~np.isfinite(traces)):
        raise ValueError("at least two finite time samples are required")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    first_receiver, second_receiver = np.triu_indices(receiver_x.size, k=1)
    pair_distance = np.hypot(
        receiver_x[first_receiver] - receiver_x[second_receiver],
        receiver_y[first_receiver] - receiver_y[second_receiver],
    )
    raw_bin = np.rint(pair_distance / distance_bin_m).astype(np.int64)
    keep_pair = raw_bin > 0
    first_receiver = first_receiver[keep_pair]
    second_receiver = second_receiver[keep_pair]
    raw_bin = raw_bin[keep_pair]
    if raw_bin.size == 0:
        raise ValueError("distance_bin_m is too large for all receiver separations")
    unique_bin, compact_bin = np.unique(raw_bin, return_inverse=True)
    distance = unique_bin.astype(np.float64) * distance_bin_m
    pair_count = np.bincount(compact_bin, minlength=unique_bin.size)

    npts, receiver_count = traces.shape
    total_time = (npts - 1) * dt_s
    if window_s is None:
        window_s = total_time
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    window_samples = min(npts, max(2, int(round(window_s / dt_s))))
    step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    if maximum_lag_s is None:
        maximum_lag_s = 0.5 * window_s
    maximum_lag_samples = min(
        int(round(maximum_lag_s / dt_s)), window_samples - 1
    )
    if maximum_lag_samples < 0:
        raise ValueError("maximum_lag_s must be non-negative")
    window_count = max(1, math.floor((npts - window_samples) / step_samples) + 1)
    temporal_nfft = _power_of_two_at_least(2 * window_samples - 1)
    gather_shape = (maximum_lag_samples + 1, unique_bin.size)
    gather = np.zeros(gather_shape, dtype=np.float64)
    branch_gathers = (
        {
            name: np.zeros(gather_shape, dtype=np.float64)
            for name in ("positive", "negative", "symmetric")
        }
        if branch == "all"
        else None
    )
    time_taper = tukey(window_samples, 0.1).reshape(-1, 1)
    valid_windows = 0
    for window_index in range(window_count):
        first = window_index * step_samples
        segment = traces[first : first + window_samples].copy()
        if np.any(np.all(segment == 0, axis=0)):
            continue
        for receiver_index in range(receiver_count):
            segment[:, receiver_index] = _normalize_trace(
                segment[:, receiver_index], normalization
            )
        segment = detrend(segment, axis=0, type="linear") * time_taper
        if np.any(np.std(segment, axis=0) == 0):
            continue
        temporal_spectrum = np.fft.fft(segment, temporal_nfft, axis=0)
        distance_cross_spectrum = np.empty(
            (temporal_nfft, unique_bin.size), dtype=np.complex128
        )
        for bin_index in range(unique_bin.size):
            selected = compact_bin == bin_index
            distance_cross_spectrum[:, bin_index] = np.mean(
                np.conj(temporal_spectrum[:, first_receiver[selected]])
                * temporal_spectrum[:, second_receiver[selected]],
                axis=1,
            )
        correlation = np.fft.ifft(
            distance_cross_spectrum, temporal_nfft, axis=0
        ).real
        positive = correlation[: maximum_lag_samples + 1]
        if maximum_lag_samples:
            negative = np.vstack(
                (correlation[0], correlation[-1 : -maximum_lag_samples - 1 : -1])
            )
        else:
            negative = correlation[:1]
        unbiased = (
            window_samples - np.arange(maximum_lag_samples + 1)
        ).reshape(-1, 1)
        positive = positive / unbiased
        negative = negative / unbiased
        if branch_gathers is not None:
            branch_gathers["positive"] += positive
            branch_gathers["negative"] += negative
            branch_gathers["symmetric"] += 0.5 * (positive + negative)
            window_gather = None
        elif branch == "positive":
            window_gather = positive
        elif branch == "negative":
            window_gather = negative
        else:
            window_gather = 0.5 * (positive + negative)
        if window_gather is not None:
            gather += window_gather
        valid_windows += 1
    if valid_windows and branch_gathers is not None:
        for branch_gather in branch_gathers.values():
            branch_gather /= valid_windows
    elif valid_windows:
        gather /= valid_windows
    lag_time = np.arange(maximum_lag_samples + 1, dtype=np.float64) * dt_s
    if branch_gathers is not None:
        return NCFBranches(
            **{
                name: NCFGather(
                    branch_gather,
                    lag_time,
                    distance,
                    pair_count,
                    valid_windows,
                )
                for name, branch_gather in branch_gathers.items()
            }
        )
    return NCFGather(gather, lag_time, distance, pair_count, valid_windows)


def receiver_array_ncf_branches(
    data: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    dt_s: float,
    *,
    distance_bin_m: float,
    window_s: float | None = None,
    overlap: float = 0.5,
    maximum_lag_s: float | None = None,
    normalization: Normalization = "spectral_whitening",
) -> NCFBranches:
    """Return all lag branches for an arbitrary receiver array in one pass."""
    result = receiver_array_ncf(
        data,
        receiver_x_m,
        receiver_y_m,
        dt_s,
        distance_bin_m=distance_bin_m,
        window_s=window_s,
        overlap=overlap,
        maximum_lag_s=maximum_lag_s,
        normalization=normalization,
        branch="all",
    )
    assert isinstance(result, NCFBranches)
    return result


def receiver_pair_ncf_branches(
    data: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    dt_s: float,
    *,
    window_s: float | None = None,
    overlap: float = 0.5,
    maximum_lag_s: float | None = None,
    normalization: Normalization = "spectral_whitening",
    pair_block_size: int = 256,
    reference_receiver_index: int | None = None,
) -> NCFPairBranches:
    """Return scalar short-window NCFs before any distance-bin averaging.

    When ``reference_receiver_index`` is provided, only pairs containing that
    receiver are returned. This supports virtual-source gathers without the
    quadratic memory and training cost of all receiver pairs.
    """
    traces = np.asarray(data, dtype=np.float64)
    receiver_x = np.asarray(receiver_x_m, dtype=np.float64).reshape(-1)
    receiver_y = np.asarray(receiver_y_m, dtype=np.float64).reshape(-1)
    if (
        traces.ndim != 2
        or traces.shape[1] != receiver_x.size
        or receiver_x.shape != receiver_y.shape
    ):
        raise ValueError("data must be time-by-receiver and match x/y coordinates")
    if traces.shape[0] < 2 or receiver_x.size < 2 or np.any(~np.isfinite(traces)):
        raise ValueError("at least two finite time samples and receivers are required")
    if np.any(~np.isfinite(receiver_x)) or np.any(~np.isfinite(receiver_y)):
        raise ValueError("receiver coordinates must be finite")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if isinstance(pair_block_size, bool) or int(pair_block_size) != pair_block_size:
        raise ValueError("pair_block_size must be a positive integer")
    pair_block_size = int(pair_block_size)
    if pair_block_size <= 0:
        raise ValueError("pair_block_size must be a positive integer")
    if reference_receiver_index is not None:
        if (
            isinstance(reference_receiver_index, bool)
            or int(reference_receiver_index) != reference_receiver_index
            or not 0 <= int(reference_receiver_index) < receiver_x.size
        ):
            raise ValueError(
                "reference_receiver_index must identify an existing receiver"
            )
        reference_receiver_index = int(reference_receiver_index)

    order = np.lexsort((receiver_y, receiver_x))
    sorted_x = receiver_x[order]
    sorted_y = receiver_y[order]
    spacing = np.diff(sorted_x)
    tolerance = max(1.0, float(np.abs(sorted_x).max())) * 1e-10
    uses_linear_convention = (
        np.max(np.abs(sorted_y - sorted_y[0])) <= tolerance
        and spacing.size > 0
        and np.all(spacing > 0)
        and np.max(np.abs(spacing - spacing[0])) <= tolerance
    )
    local_i, local_j = np.triu_indices(receiver_x.size, k=1)
    if uses_linear_convention:
        pair_i = order[local_i]
        pair_j = order[local_j]
    else:
        pair_i, pair_j = local_i, local_j
    if reference_receiver_index is not None:
        includes_reference = (pair_i == reference_receiver_index) | (
            pair_j == reference_receiver_index
        )
        pair_i = pair_i[includes_reference]
        pair_j = pair_j[includes_reference]
    pair_distance = np.hypot(
        receiver_x[pair_j] - receiver_x[pair_i],
        receiver_y[pair_j] - receiver_y[pair_i],
    )
    keep = pair_distance > 0
    pair_i = pair_i[keep]
    pair_j = pair_j[keep]
    pair_distance = pair_distance[keep]
    if pair_distance.size == 0:
        raise ValueError("receiver pairs require positive separation")

    npts, receiver_count = traces.shape
    total_time = (npts - 1) * dt_s
    if window_s is None:
        window_s = total_time
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be finite and positive")
    window_samples = min(npts, max(2, int(round(window_s / dt_s))))
    step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    if maximum_lag_s is None:
        maximum_lag_s = 0.5 * window_s
    if not np.isfinite(maximum_lag_s) or maximum_lag_s < 0:
        raise ValueError("maximum_lag_s must be finite and non-negative")
    maximum_lag_samples = min(
        int(round(maximum_lag_s / dt_s)), window_samples - 1
    )
    window_count = max(1, math.floor((npts - window_samples) / step_samples) + 1)
    temporal_nfft = _power_of_two_at_least(2 * window_samples - 1)
    gather_shape = (maximum_lag_samples + 1, pair_distance.size)
    branch_data = {
        name: np.zeros(gather_shape, dtype=np.float64)
        for name in ("positive", "negative", "symmetric")
    }
    time_taper = tukey(window_samples, 0.1).reshape(-1, 1)
    unbiased = (
        window_samples - np.arange(maximum_lag_samples + 1)
    ).reshape(-1, 1)
    valid_windows = 0
    for window_index in range(window_count):
        first = window_index * step_samples
        segment = traces[first : first + window_samples].copy()
        if np.any(np.all(segment == 0, axis=0)):
            continue
        for receiver_index in range(receiver_count):
            segment[:, receiver_index] = _normalize_trace(
                segment[:, receiver_index], normalization
            )
        segment = detrend(segment, axis=0, type="linear") * time_taper
        if np.any(np.std(segment, axis=0) == 0):
            continue
        spectrum = np.fft.fft(segment, temporal_nfft, axis=0)
        for block_first in range(0, pair_distance.size, pair_block_size):
            block = slice(
                block_first, min(pair_distance.size, block_first + pair_block_size)
            )
            if uses_linear_convention:
                cross_spectrum = (
                    np.conj(spectrum[:, pair_j[block]]) * spectrum[:, pair_i[block]]
                )
            else:
                cross_spectrum = (
                    np.conj(spectrum[:, pair_i[block]]) * spectrum[:, pair_j[block]]
                )
            correlation = np.fft.ifft(cross_spectrum, temporal_nfft, axis=0).real
            positive = correlation[: maximum_lag_samples + 1] / unbiased
            if maximum_lag_samples:
                negative = np.vstack(
                    (
                        correlation[0],
                        correlation[-1 : -maximum_lag_samples - 1 : -1],
                    )
                ) / unbiased
            else:
                negative = correlation[:1] / unbiased
            branch_data["positive"][:, block] += positive
            branch_data["negative"][:, block] += negative
            branch_data["symmetric"][:, block] += 0.5 * (positive + negative)
        valid_windows += 1

    if valid_windows:
        for values in branch_data.values():
            values /= valid_windows
    lag_time = np.arange(maximum_lag_samples + 1, dtype=np.float64) * dt_s
    return NCFPairBranches(
        **{
            name: NCFPairGather(
                data=values,
                lag_time_s=lag_time,
                pair_i=pair_i,
                pair_j=pair_j,
                distance_m=pair_distance,
                valid_windows=valid_windows,
            )
            for name, values in branch_data.items()
        }
    )


def bin_pair_ncf_branches(
    branches: NCFPairBranches,
    *,
    distance_bin_m: float,
) -> NCFBranches:
    """Average identified pair correlations into the existing distance bins."""
    if not np.isfinite(distance_bin_m) or distance_bin_m <= 0:
        raise ValueError("distance_bin_m must be finite and positive")
    template = branches.positive
    for name in ("negative", "symmetric"):
        actual = getattr(branches, name)
        if (
            actual.data.shape != template.data.shape
            or not np.array_equal(actual.lag_time_s, template.lag_time_s)
            or not np.array_equal(actual.pair_i, template.pair_i)
            or not np.array_equal(actual.pair_j, template.pair_j)
            or not np.array_equal(actual.distance_m, template.distance_m)
            or actual.valid_windows != template.valid_windows
        ):
            raise ValueError("pair-resolved NCF branches are incompatible")
    raw_bin = np.rint(template.distance_m / distance_bin_m).astype(np.int64)
    keep = raw_bin > 0
    if not np.any(keep):
        raise ValueError("distance_bin_m is too large for all receiver separations")
    unique_bin, compact_bin = np.unique(raw_bin[keep], return_inverse=True)
    finite_pair = np.ones(np.count_nonzero(keep), dtype=bool)
    for name in ("positive", "negative", "symmetric"):
        finite_pair &= np.all(np.isfinite(getattr(branches, name).data[:, keep]), axis=0)
    pair_count = np.bincount(
        compact_bin[finite_pair], minlength=unique_bin.size
    )
    populated_bin = pair_count > 0
    unique_bin = unique_bin[populated_bin]
    pair_count = pair_count[populated_bin]
    distance = unique_bin.astype(np.float64) * distance_bin_m
    output = {}
    for name in ("positive", "negative", "symmetric"):
        values = getattr(branches, name).data[:, keep]
        binned = np.empty((values.shape[0], unique_bin.size), dtype=np.float64)
        retained_compact_bin = np.flatnonzero(populated_bin)
        for output_index, original_bin_index in enumerate(retained_compact_bin):
            selected = (compact_bin == original_bin_index) & finite_pair
            binned[:, output_index] = np.mean(values[:, selected], axis=1)
        output[name] = NCFGather(
            data=binned,
            lag_time_s=template.lag_time_s,
            distance_m=distance,
            pair_count=pair_count,
            valid_windows=template.valid_windows,
        )
    return NCFBranches(**output)


def receiver_array_3c_ncf(
    data: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    dt_s: float,
    *,
    distance_bin_m: float,
    window_s: float | None = None,
    overlap: float = 0.5,
    maximum_lag_s: float | None = None,
    normalization: Normalization = "spectral_whitening",
    branch: BranchRequest = "symmetric",
) -> dict[str, NCFGather | NCFBranches]:
    """Return pair-oriented ZZ, RR, and TT NCFs for an arbitrary 3-C array.

    Input component order is global X, Y, Z.  For every receiver pair the two
    horizontal records are rotated onto the interstation radial/transverse axes
    before cross-correlation.  The pair spectra are then stacked by distance,
    retaining the isotropic 1-D assumption of :func:`receiver_array_ncf`.
    """
    traces = np.asarray(data, dtype=np.float64)
    receiver_x = np.asarray(receiver_x_m, dtype=np.float64).reshape(-1)
    receiver_y = np.asarray(receiver_y_m, dtype=np.float64).reshape(-1)
    if (
        traces.ndim != 3
        or traces.shape[1] != receiver_x.size
        or traces.shape[2] != 3
        or receiver_x.shape != receiver_y.shape
    ):
        raise ValueError("data must be time-by-receiver-by-3 (X,Y,Z) and match coordinates")
    if traces.shape[0] < 2 or receiver_x.size < 2 or np.any(~np.isfinite(traces)):
        raise ValueError("at least two finite time samples and receivers are required")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not np.isfinite(distance_bin_m) or distance_bin_m <= 0:
        raise ValueError("distance_bin_m must be finite and positive")
    if branch not in {"symmetric", "positive", "negative", "all"}:
        raise ValueError("branch must be symmetric, positive, negative, or all")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    # The exact line-array projection is global X=radial, Y=transverse.
    order = np.lexsort((receiver_y, receiver_x))
    sorted_x = receiver_x[order]
    sorted_y = receiver_y[order]
    spacing = np.diff(sorted_x)
    tolerance = max(1.0, float(np.abs(sorted_x).max())) * 1e-10
    if (
        np.max(np.abs(sorted_y - sorted_y[0])) <= tolerance
        and spacing.size > 0
        and np.all(spacing > 0)
        and np.max(np.abs(spacing - spacing[0])) <= tolerance
        and np.isclose(distance_bin_m, spacing[0], rtol=0, atol=tolerance)
    ):
        options = dict(
            window_s=window_s,
            overlap=overlap,
            maximum_lag_s=maximum_lag_s,
            normalization=normalization,
            branch=branch,
        )
        return {
            "rr": linear_array_ncf(traces[:, order, 0], sorted_x, dt_s, **options),
            "tt": linear_array_ncf(traces[:, order, 1], sorted_x, dt_s, **options),
            "zz": linear_array_ncf(traces[:, order, 2], sorted_x, dt_s, **options),
        }

    first_receiver, second_receiver = np.triu_indices(receiver_x.size, k=1)
    delta_x = receiver_x[second_receiver] - receiver_x[first_receiver]
    delta_y = receiver_y[second_receiver] - receiver_y[first_receiver]
    pair_distance = np.hypot(delta_x, delta_y)
    raw_bin = np.rint(pair_distance / distance_bin_m).astype(np.int64)
    keep_pair = raw_bin > 0
    first_receiver = first_receiver[keep_pair]
    second_receiver = second_receiver[keep_pair]
    pair_distance = pair_distance[keep_pair]
    delta_x = delta_x[keep_pair]
    delta_y = delta_y[keep_pair]
    raw_bin = raw_bin[keep_pair]
    if raw_bin.size == 0:
        raise ValueError("distance_bin_m is too large for all receiver separations")
    cosine = delta_x / pair_distance
    sine = delta_y / pair_distance
    unique_bin, compact_bin = np.unique(raw_bin, return_inverse=True)
    distance = unique_bin.astype(np.float64) * distance_bin_m
    pair_count = np.bincount(compact_bin, minlength=unique_bin.size)

    npts = traces.shape[0]
    total_time = (npts - 1) * dt_s
    if window_s is None:
        window_s = total_time
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    window_samples = min(npts, max(2, int(round(window_s / dt_s))))
    step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    if maximum_lag_s is None:
        maximum_lag_s = 0.5 * window_s
    maximum_lag_samples = min(int(round(maximum_lag_s / dt_s)), window_samples - 1)
    if maximum_lag_samples < 0:
        raise ValueError("maximum_lag_s must be non-negative")
    window_count = max(1, math.floor((npts - window_samples) / step_samples) + 1)
    temporal_nfft = _power_of_two_at_least(2 * window_samples - 1)
    gathers = {
        name: np.zeros((maximum_lag_samples + 1, unique_bin.size), dtype=np.float64)
        for name in ("rr", "tt", "zz")
    }
    branch_gathers = (
        {
            name: {
                branch_name: np.zeros_like(gather)
                for branch_name in ("positive", "negative", "symmetric")
            }
            for name, gather in gathers.items()
        }
        if branch == "all"
        else None
    )
    time_taper = tukey(window_samples, 0.1).reshape(-1, 1, 1)
    valid_windows = 0
    pair_block_size = 128
    for window_index in range(window_count):
        first = window_index * step_samples
        segment = traces[first : first + window_samples].copy()
        if np.any(np.all(segment == 0, axis=(0, 2))):
            continue
        for receiver_index in range(receiver_x.size):
            for component_index in range(3):
                segment[:, receiver_index, component_index] = _normalize_trace(
                    segment[:, receiver_index, component_index], normalization
                )
        segment = detrend(segment, axis=0, type="linear") * time_taper
        if np.any(np.linalg.norm(np.std(segment, axis=0), axis=1) == 0):
            continue
        spectrum = np.fft.fft(segment, temporal_nfft, axis=0)
        distance_cross_spectrum = {
            name: np.zeros((temporal_nfft, unique_bin.size), dtype=np.complex128)
            for name in gathers
        }
        for bin_index in range(unique_bin.size):
            selected = np.flatnonzero(compact_bin == bin_index)
            accumulator = {
                name: np.zeros(temporal_nfft, dtype=np.complex128) for name in gathers
            }
            for block_first in range(0, selected.size, pair_block_size):
                pair = selected[block_first : block_first + pair_block_size]
                i = first_receiver[pair]
                j = second_receiver[pair]
                c = cosine[pair][None, :]
                s = sine[pair][None, :]
                radial_i = c * spectrum[:, i, 0] + s * spectrum[:, i, 1]
                radial_j = c * spectrum[:, j, 0] + s * spectrum[:, j, 1]
                transverse_i = -s * spectrum[:, i, 0] + c * spectrum[:, i, 1]
                transverse_j = -s * spectrum[:, j, 0] + c * spectrum[:, j, 1]
                accumulator["rr"] += np.sum(np.conj(radial_i) * radial_j, axis=1)
                accumulator["tt"] += np.sum(
                    np.conj(transverse_i) * transverse_j, axis=1
                )
                accumulator["zz"] += np.sum(
                    np.conj(spectrum[:, i, 2]) * spectrum[:, j, 2], axis=1
                )
            for name in gathers:
                distance_cross_spectrum[name][:, bin_index] = (
                    accumulator[name] / selected.size
                )
        unbiased = (window_samples - np.arange(maximum_lag_samples + 1)).reshape(-1, 1)
        for name, cross_spectrum in distance_cross_spectrum.items():
            correlation = np.fft.ifft(cross_spectrum, temporal_nfft, axis=0).real
            positive = correlation[: maximum_lag_samples + 1] / unbiased
            if maximum_lag_samples:
                negative = np.vstack(
                    (correlation[0], correlation[-1 : -maximum_lag_samples - 1 : -1])
                ) / unbiased
            else:
                negative = correlation[:1] / unbiased
            if branch_gathers is not None:
                branch_gathers[name]["positive"] += positive
                branch_gathers[name]["negative"] += negative
                branch_gathers[name]["symmetric"] += 0.5 * (positive + negative)
                window_gather = None
            elif branch == "positive":
                window_gather = positive
            elif branch == "negative":
                window_gather = negative
            else:
                window_gather = 0.5 * (positive + negative)
            if window_gather is not None:
                gathers[name] += window_gather
        valid_windows += 1
    lag_time = np.arange(maximum_lag_samples + 1, dtype=np.float64) * dt_s
    if branch_gathers is not None:
        return {
            name: NCFBranches(
                **{
                    branch_name: NCFGather(
                        branch_gather / valid_windows if valid_windows else branch_gather,
                        lag_time,
                        distance,
                        pair_count,
                        valid_windows,
                    )
                    for branch_name, branch_gather in component.items()
                }
            )
            for name, component in branch_gathers.items()
        }
    return {
        name: NCFGather(
            gather / valid_windows if valid_windows else gather,
            lag_time,
            distance,
            pair_count,
            valid_windows,
        )
        for name, gather in gathers.items()
    }


def receiver_array_3c_ncf_branches(
    data: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    dt_s: float,
    *,
    distance_bin_m: float,
    window_s: float | None = None,
    overlap: float = 0.5,
    maximum_lag_s: float | None = None,
    normalization: Normalization = "spectral_whitening",
) -> dict[str, NCFBranches]:
    """Return ZZ, RR, and TT lag branches from one 3-C correlation pass."""
    result = receiver_array_3c_ncf(
        data,
        receiver_x_m,
        receiver_y_m,
        dt_s,
        distance_bin_m=distance_bin_m,
        window_s=window_s,
        overlap=overlap,
        maximum_lag_s=maximum_lag_s,
        normalization=normalization,
        branch="all",
    )
    assert all(isinstance(branches, NCFBranches) for branches in result.values())
    return result  # type: ignore[return-value]
