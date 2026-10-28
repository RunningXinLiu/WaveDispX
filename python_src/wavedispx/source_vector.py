"""Versioned global force projection. Local Z follows native Green's Z sign.

X/Y are the existing map coordinates, azimuth is +X toward +Y. No implicit
ENU/NED conversion; positive inclination points toward the native +Z axis.
"""
import numpy as np


def force_vector(amplitude, azimuth_deg, inclination_deg):
    az, inc = np.deg2rad([azimuth_deg, inclination_deg])
    v = np.array([np.cos(inc)*np.cos(az), np.cos(inc)*np.sin(az), np.sin(inc)])
    v[np.abs(v) < 1e-15] = 0.0
    return float(amplitude) * v


def project_green_to_xyz(fields, azimuth, force):
    """(time,receiver,RR/TT/ZR/RZ/ZZ) -> same physical force XYZ response."""
    c, s = np.cos(azimuth), np.sin(azimuth)
    ur, ut, uz = np.moveaxis(project_green_to_rtz(fields, azimuth, force), -1, 0)
    return np.stack((ur*c-ut*s, ur*s+ut*c, uz), axis=-1)


def project_green_to_rtz(fields, azimuth, force):
    """Keep exact zero transverse forcing; avoid round-trip XYZ cancellation."""
    c, s = np.cos(azimuth), np.sin(azimuth)
    fx, fy, fz = force
    fr, ft = c*fx+s*fy, -s*fx+c*fy
    tolerance = 1e-15*np.linalg.norm(force)
    fr = np.where(np.abs(fr) < tolerance, 0., fr)
    ft = np.where(np.abs(ft) < tolerance, 0., ft)
    ur = fields[..., 0]*fr + fields[..., 3]*fz
    ut = fields[..., 1]*ft
    uz = fields[..., 2]*fr + fields[..., 4]*fz
    return np.stack((ur, ut, uz), axis=-1)


def polarize_coefficients(coefficients, source):
    """Keep legacy bitwise unchanged; new modes fix total expected force PSD.

    independent_hv: diag(pH/2,pH/2,1-pH); directed_hv: one coherent horizontal
    direction, statistically independent vertical forcing. All sources share
    this input direction in v1; event-wise random orientation is unsupported.
    """
    if source is None or source.polarization == "legacy_xyz":
        return coefficients
    p = source.horizontal_power_fraction
    out = coefficients.copy()
    if source.polarization == "independent_hv":
        out *= np.sqrt([p/2, p/2, 1-p])[None, :, None]
    elif source.polarization == "directed_hv":
        theta = np.deg2rad(source.horizontal_azimuth_deg)
        horizontal = coefficients[:, 0] * np.sqrt(p)
        out[:, 0] = horizontal * np.cos(theta)
        out[:, 1] = horizontal * np.sin(theta)
        out[:, 2] = coefficients[:, 2] * np.sqrt(1-p)
    else:
        raise ValueError("unsupported source polarization")
    return out
