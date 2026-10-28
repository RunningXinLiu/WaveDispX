"""Node HVSR v1 from unwhitened finite XYZ windows, never from an NCF.

Quadratic-mean H, optional Konno-Ohmachi amplitude smoothing, log-window
statistics. Standard deviation describes window scatter, not confidence.
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal.windows import tukey


@dataclass(frozen=True)
class HVSRResult:
    frequency_hz: np.ndarray
    geometric_mean: np.ndarray  # receiver, frequency
    log_std: np.ndarray
    valid_windows: np.ndarray
    mean_amplitude_xyz: np.ndarray  # receiver, frequency, XYZ; smoothed
    directions_deg: np.ndarray
    directional_geometric_mean: np.ndarray  # receiver, frequency, direction
    processed_windows: int
    smoothing_b: float
    vertical_floor_ratio: float


class HVSRAccumulator:
    def __init__(self, dt_s, samples, receiver_count, fmin, fmax, *, smoothing_b=40., floor_ratio=1e-6):
        if not np.isfinite(smoothing_b) or smoothing_b < 0 or not 0 < floor_ratio < 1:
            raise ValueError("HVSR smoothing_b must be finite >=0; vertical floor ratio must be in (0,1)")
        f = np.fft.rfftfreq(samples, dt_s)
        self.inside = (f > 0) & (f >= fmin) & (f <= fmax)
        self.frequency = f[self.inside]
        if len(self.frequency) < 2:
            raise ValueError("HVSR band contains fewer than two actual FFT frequencies")
        self.dt, self.samples, self.b, self.floor = dt_s, samples, smoothing_b, floor_ratio
        self.taper = tukey(samples, .1)
        self.kernel = None
        if smoothing_b:
            x = smoothing_b * np.log10(self.frequency[None, :] / self.frequency[:, None])
            self.kernel = np.sinc(x/np.pi)**4
            self.kernel /= self.kernel.sum(axis=1, keepdims=True)
        self.angles = np.arange(0., 180., 15.)
        shape = (receiver_count, len(self.frequency), 1+len(self.angles))
        self.count = np.zeros(shape, dtype=np.int64)
        self.sum_log = np.zeros(shape)
        self.sum_log2 = np.zeros(shape)
        self.spectra_sum = np.zeros(shape[:2]+(3,))
        self.windows = 0

    def smooth(self, amplitude):
        return amplitude if self.kernel is None else np.einsum("ij,rjk->rik", self.kernel, amplitude, optimize=True)

    def add(self, record):
        x = np.asarray(record, dtype=float)
        if x.shape != (self.samples, self.count.shape[0], 3) or not np.isfinite(x).all():
            raise ValueError("HVSR requires finite (time,receiver,XYZ) raw windows of the declared length")
        x = (x-x.mean(axis=0)) * self.taper[:, None, None]
        spectrum = np.fft.rfft(x, axis=0)[self.inside].transpose(1, 0, 2)
        spectrum *= 2.0/self.taper.sum()  # amplitude convention; same for XYZ
        amplitude = np.abs(spectrum)
        h = np.sqrt((amplitude[..., 0]**2+amplitude[..., 1]**2)/2)
        theta = np.deg2rad(self.angles)
        directional = np.abs(spectrum[..., 0, None]*np.cos(theta)+spectrum[..., 1, None]*np.sin(theta))
        numerator = self.smooth(np.concatenate((h[..., None], directional), axis=-1))
        smooth_xyz = self.smooth(amplitude)
        vertical = smooth_xyz[..., 2, None]
        floor = self.floor*np.max(vertical, axis=1, keepdims=True)
        supported = np.max(amplitude[..., 2], axis=1, keepdims=True)[..., None] > 0
        valid = (vertical > floor) & (vertical > 0) & supported & (numerator > 0)
        # A frequency below ten cycles per window is explicitly excluded.
        valid &= (self.frequency*self.samples*self.dt >= 10)[None, :, None]
        ratio = np.divide(numerator, vertical, out=np.ones_like(numerator), where=valid)
        logs = np.zeros_like(ratio)
        np.log(ratio, out=logs, where=valid)
        self.count += valid
        self.sum_log += logs
        self.sum_log2 += logs*logs
        self.spectra_sum += smooth_xyz
        self.windows += 1

    def finish(self):
        ok = self.count >= 3
        mean = np.divide(self.sum_log, self.count, out=np.full_like(self.sum_log, np.nan), where=ok)
        variance = np.divide(self.sum_log2-self.count*mean*mean, self.count-1,
                             out=np.full_like(mean, np.nan), where=ok)
        return HVSRResult(self.frequency, np.exp(mean[..., 0]), np.sqrt(np.maximum(variance[..., 0], 0)),
                          self.count[..., 0], self.spectra_sum/max(self.windows, 1), self.angles,
                          np.exp(mean[..., 1:]), self.windows, self.b, self.floor)


def compute_passive_hvsr(workflow, *, smoothing_b=40., floor_ratio=1e-6):
    """Stream the entire configured duration; do not use the 60s preview cache."""
    from .passive import build_passive_transfer
    from .geometry import build_geometry
    from .noise import synthesize_mixed_noise_window_3c_band, add_independent_receiver_noise
    from .source_activity import load_source_activity_matrix
    c = workflow.config
    if c.mode != "passive" or c.das.enabled or c.ambient_sources.component != "three_component":
        raise ValueError("节点HVSR需要被动XYZ记录：设置ambient_sources.component=three_component且关闭DAS；源水平占比可为0。")
    g = workflow.geometry or build_geometry(c.receivers, c.ambient_sources)
    workflow.geometry = g
    if workflow.transfer is None:
        workflow.transfer = build_passive_transfer(c, g)
    transfer, noise = workflow.transfer, c.noise
    count = int(round(noise.duration_s/noise.window_s))
    activity = load_source_activity_matrix(c.ambient_sources, window_count=count)
    az = np.arctan2(g.receiver_y_m[:, None]-g.source_y_m[None, :], g.receiver_x_m[:, None]-g.source_x_m[None, :])
    accumulator = HVSRAccumulator(noise.dt_s, int(round(noise.window_s/noise.dt_s)), len(g.receiver_x_m),
                                  c.processing.frequency_min_hz, c.processing.frequency_max_hz,
                                  smoothing_b=smoothing_b, floor_ratio=floor_ratio)
    rng, receiver_rng = np.random.default_rng(noise.seed), np.random.default_rng(noise.receiver_noise_seed)
    for window in range(count):
        record = synthesize_mixed_noise_window_3c_band(transfer.spectrum, transfer.frequency_hz, az, noise, rng,
                    source_block_size=c.source_block_size, source_activity_weights=activity[window],
                    source_polarization=c.ambient_sources).record
        mixed = add_independent_receiver_noise(record, noise, receiver_rng)
        if window == 0:
            workflow.hvsr_preview = mixed.copy()
        accumulator.add(mixed)
    return accumulator.finish()
