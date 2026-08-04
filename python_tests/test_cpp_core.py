from pathlib import Path

import h5py
import numpy as np

from wavedispx.core import full_fk_response, global_stiffness, vertical_fk_response
from wavedispx.config import load_config
from wavedispx.forward import (
    FullFkGreen,
    VerticalFkGreen,
    synthesize_full_receivers,
    synthesize_vertical_receivers,
)


ROOT = Path(__file__).resolve().parents[1]


def test_global_stiffness_matches_all_matlab_golden_cases():
    with h5py.File(ROOT / "matlab_reference/golden_dwm_kernels.h5") as reference:
        model = reference["model"]
        vp = model["vp_mps"][...].reshape(-1)
        vs = model["vs_mps"][...].reshape(-1)
        density = model["density_kgm3"][...].reshape(-1)
        thickness = model["thickness_m"][...].reshape(-1)
        group = reference["global_stiffness"]
        wavenumber = group["wavenumber_radpm"][...].reshape(-1)
        frequency = group["frequency_hz"][...].reshape(-1)
        expected_sh = group["kg_sh_real"][...] + 1j * group["kg_sh_imag"][...]
        expected_svp = group["kg_svp_real"][...] + 1j * group["kg_svp_imag"][...]
    damping = np.zeros_like(vp)
    for case_index, (case_k, case_frequency) in enumerate(zip(wavenumber, frequency)):
        omega = 2 * np.pi * case_frequency - 2j * np.pi * 0.25
        actual_sh, actual_svp = global_stiffness(
            vp, vs, damping, damping, density, thickness, case_k, omega
        )
        np.testing.assert_allclose(actual_sh, expected_sh[case_index], rtol=2e-12, atol=1e-7)
        np.testing.assert_allclose(actual_svp, expected_svp[case_index], rtol=2e-12, atol=1e-7)


def test_vertical_fk_response_matches_matlab_golden():
    with h5py.File(ROOT / "matlab_reference/golden_dwm_kernels.h5") as reference:
        model = reference["model"]
        vp = model["vp_mps"][...].reshape(-1)
        vs = model["vs_mps"][...].reshape(-1)
        density = model["density_kgm3"][...].reshape(-1)
        thickness = model["thickness_m"][...].reshape(-1)
        group = reference["vertical_fk"]
        expected_frequency = group["frequency_hz"][...].reshape(-1)
        expected_wavenumber = group["wavenumber_radpm"][...].reshape(-1)
        expected_vr = group["vr_real"][...] + 1j * group["vr_imag"][...]
        expected_vv = group["vv_real"][...] + 1j * group["vv_imag"][...]
        nt = int(group["nt"][...].reshape(-1)[0])
        dt = float(group["dt_s"][...].reshape(-1)[0])
        ymax = float(group["ymax_m"][...].reshape(-1)[0])
        delay = float(group["delay_s"][...].reshape(-1)[0])
        fmax = float(group["maximum_frequency_hz"][...].reshape(-1)[0])
    result = vertical_fk_response(
        vp,
        vs,
        np.zeros_like(vp),
        np.zeros_like(vp),
        density,
        thickness,
        sample_count=nt,
        dt_s=dt,
        maximum_distance_m=ymax,
        delay_s=delay,
        maximum_frequency_hz=fmax,
        workers=2,
    )
    np.testing.assert_allclose(result["frequency_hz"], expected_frequency, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["wavenumber_radpm"], expected_wavenumber, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["vr"], expected_vr, rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(result["vv"], expected_vv, rtol=2e-10, atol=1e-12)


def test_horizontal_fk_response_matches_matlab_golden():
    with h5py.File(ROOT / "matlab_reference/golden_dwm_kernels.h5") as reference:
        model = reference["model"]
        vp = model["vp_mps"][...].reshape(-1)
        vs = model["vs_mps"][...].reshape(-1)
        density = model["density_kgm3"][...].reshape(-1)
        thickness = model["thickness_m"][...].reshape(-1)
        vertical = reference["vertical_fk"]
        horizontal = reference["horizontal_fk"]
        expected = {
            name: horizontal[f"{name}_real"][...] + 1j * horizontal[f"{name}_imag"][...]
            for name in ("hr", "ht", "hv")
        }
        nt = int(vertical["nt"][...].reshape(-1)[0])
        dt = float(vertical["dt_s"][...].reshape(-1)[0])
        ymax = float(vertical["ymax_m"][...].reshape(-1)[0])
        delay = float(vertical["delay_s"][...].reshape(-1)[0])
        fmax = float(vertical["maximum_frequency_hz"][...].reshape(-1)[0])
    result = full_fk_response(
        vp,
        vs,
        np.zeros_like(vp),
        np.zeros_like(vp),
        density,
        thickness,
        sample_count=nt,
        dt_s=dt,
        maximum_distance_m=ymax,
        delay_s=delay,
        maximum_frequency_hz=fmax,
        workers=2,
        include_horizontal=True,
        include_vertical=False,
    )
    for name in ("hr", "ht", "hv"):
        np.testing.assert_allclose(result[name], expected[name], rtol=2e-10, atol=1e-12)


def test_inverse_hankel_receiver_response_matches_matlab_golden():
    with h5py.File(ROOT / "matlab_reference/golden_dwm_kernels.h5") as reference:
        fk_group = reference["vertical_fk"]
        fk = VerticalFkGreen(
            fk_group["frequency_hz"][...].reshape(-1),
            fk_group["wavenumber_radpm"][...].reshape(-1),
            fk_group["vr_real"][...] + 1j * fk_group["vr_imag"][...],
            fk_group["vv_real"][...] + 1j * fk_group["vv_imag"][...],
        )
        receiver_group = reference["vertical_receiver"]
        distance = receiver_group["distance_m"][...].reshape(-1)
        expected_radial = receiver_group["radial"][...].T
        expected_vertical = receiver_group["vertical"][...].T
        dt = float(fk_group["dt_s"][...].reshape(-1)[0])
        nt = int(fk_group["nt"][...].reshape(-1)[0])
    result = synthesize_vertical_receivers(
        fk, distance, sample_count=nt, dt_s=dt, distance_block_size=2
    )
    np.testing.assert_allclose(result.radial, expected_radial, rtol=2e-9, atol=1e-12)
    np.testing.assert_allclose(result.vertical, expected_vertical, rtol=2e-9, atol=1e-12)


def test_full_inverse_hankel_response_matches_matlab_golden():
    with h5py.File(ROOT / "matlab_reference/golden_dwm_kernels.h5") as reference:
        vertical = reference["vertical_fk"]
        horizontal = reference["horizontal_fk"]
        fk = FullFkGreen(
            vertical["frequency_hz"][...].reshape(-1),
            vertical["wavenumber_radpm"][...].reshape(-1),
            horizontal["hr_real"][...] + 1j * horizontal["hr_imag"][...],
            horizontal["ht_real"][...] + 1j * horizontal["ht_imag"][...],
            horizontal["hv_real"][...] + 1j * horizontal["hv_imag"][...],
            vertical["vr_real"][...] + 1j * vertical["vr_imag"][...],
            vertical["vv_real"][...] + 1j * vertical["vv_imag"][...],
        )
        distance = reference["vertical_receiver/distance_m"][...].reshape(-1)
        expected = {
            "horizontal_radial": reference["horizontal_receiver/radial"][...].T,
            "horizontal_tangential": reference["horizontal_receiver/tangential"][...].T,
            "horizontal_vertical": reference["horizontal_receiver/vertical"][...].T,
            "vertical_radial": reference["vertical_receiver/radial"][...].T,
            "vertical_vertical": reference["vertical_receiver/vertical"][...].T,
        }
        nt = int(vertical["nt"][...].reshape(-1)[0])
        dt = float(vertical["dt_s"][...].reshape(-1)[0])
    result = synthesize_full_receivers(
        fk, distance, sample_count=nt, dt_s=dt, distance_block_size=2
    )
    for name, expected_field in expected.items():
        np.testing.assert_allclose(
            getattr(result, name), expected_field, rtol=2e-9, atol=1e-12
        )
