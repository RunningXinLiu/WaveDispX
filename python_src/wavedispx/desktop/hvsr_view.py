"""Read-only multi-station HVSR plots. No resampling or normalization."""
import numpy as np
import matplotlib.pyplot as plt

COLORS = ("#1768ac", "#c08016", "#8751a0", "#27857a", "#b64f70")
STYLES = ("-", "--", "-.")


def station_axis(x, y):
    """Collinear receivers: distance along line; area arrays: input row ID."""
    xy = np.column_stack((x, y)).astype(float)
    if len(xy) > 1:
        centered = xy-xy.mean(axis=0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        if s[0] > 0 and s[-1] <= 1e-8*s[0]:
            direction = vt[0]
            if direction[np.argmax(np.abs(direction))] < 0:
                direction = -direction
            distance = centered@direction
            distance -= distance.min()
            order = np.argsort(distance, kind="stable")
            return distance[order], order, "Distance along line (m)"
    ids = np.arange(len(xy))
    return ids, ids, "Station ID (input row, zero-based)"


def curve_figure(h, stations, *, scatter=True):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for index, station in enumerate(stations):
        mean, std = h.geometric_mean[station], h.log_std[station]
        color = COLORS[index % len(COLORS)]
        ax.plot(h.frequency_hz, mean, color=color, linestyle=STYLES[(index//len(COLORS)) % len(STYLES)],
                label=f"Station {station}", lw=1.8)
        if scatter:
            ax.fill_between(h.frequency_hz, mean*np.exp(-std), mean*np.exp(std), color=color, alpha=.12)
    ax.set(xlabel="Frequency (Hz)", ylabel="HVSR (H/V)", ylim=(0, None),
           title=f"Raw-waveform HVSR | {len(stations)} station(s) | {h.processed_windows} windows")
    fig.suptitle("Shading: ±1 log-standard-deviation window scatter, not confidence" if scatter else
                 "Individual station curves; no spatial averaging or per-station normalization",fontsize=10)
    ax.grid(alpha=.18)
    if stations:
        ax.legend(ncol=min(4, len(stations)), fontsize=9)
    if not stations or not any(np.isfinite(h.geometric_mean[s]).any() for s in stations):
        ax.text(.5,.5,"No valid HVSR bins for selected stations",ha="center",transform=ax.transAxes)
    return fig


def heatmap_figure(h, x, y):
    axis, order, label = station_axis(x, y)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d8dce1")
    values = np.ma.masked_invalid(h.geometric_mean[order])
    vmax = float(values.max()) if values.count() else 1.
    mesh = ax.pcolormesh(h.frequency_hz, axis, values, shading="nearest", cmap=cmap, vmin=0, vmax=max(vmax,1e-12))
    ax.set(xlabel="Frequency (Hz)", ylabel=label, title="All-station HVSR | click a row to inspect its curve")
    fig.colorbar(mesh, ax=ax, label="HVSR (H/V), common linear scale")
    fig.suptitle("Grey = invalid bins; no per-station normalization",fontsize=10)
    return fig, axis, order
