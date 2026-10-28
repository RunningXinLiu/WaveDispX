from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from wavedispx.source_vector import force_vector, project_green_to_xyz, polarize_coefficients
from wavedispx.hvsr import HVSRAccumulator
from wavedispx.experiment import ExperimentRequest, run_experiment, theory_context, wave_unavailable
from wavedispx.config import config_from_mapping
from wavedispx.app_service import load_template_mapping

ROOT = Path(__file__).resolve().parents[1]


def test_vector_rotation_linearity_and_vertical_limit():
    fields = np.random.default_rng(3).normal(size=(40,8,5))
    az = np.linspace(-2,2,8)
    vertical=project_green_to_xyz(fields,az,[0,0,1])
    np.testing.assert_array_equal(vertical[...,2],fields[...,4])
    np.testing.assert_allclose(vertical[...,0],fields[...,3]*np.cos(az))
    force=force_vector(2,35,-20)
    np.testing.assert_allclose(np.linalg.norm(force),2)
    result=project_green_to_xyz(fields,az,force)
    np.testing.assert_allclose(result,sum(project_green_to_xyz(fields,az,np.eye(3)[i]*force[i]) for i in range(3)),atol=1e-14)
    angle=.7
    rotated=np.array([force[0]*np.cos(angle)-force[1]*np.sin(angle),force[0]*np.sin(angle)+force[1]*np.cos(angle),force[2]])
    actual=project_green_to_xyz(fields,az+angle,rotated)
    expected=result.copy()
    expected[...,0]=result[...,0]*np.cos(angle)-result[...,1]*np.sin(angle)
    expected[...,1]=result[...,0]*np.sin(angle)+result[...,1]*np.cos(angle)
    np.testing.assert_allclose(actual,expected,atol=1e-14)


def test_source_psd_weights_not_amplitude_weights():
    a=np.ones((5,3,40),complex)
    assert polarize_coefficients(a,None) is a
    source=SimpleNamespace(polarization="independent_hv",horizontal_power_fraction=.6,horizontal_azimuth_deg=0)
    out=polarize_coefficients(a,source)
    np.testing.assert_allclose(np.abs(out[0,:,0])**2,[.3,.3,.4])
    source.polarization="directed_hv"
    source.horizontal_azimuth_deg=90
    out=polarize_coefficients(a,source)
    np.testing.assert_allclose(np.abs(out[0,:,0])**2,[0,.6,.4],atol=1e-15)


@pytest.mark.parametrize("b",[0,40])
def test_hvsr_known_ratio_and_zero_vertical_mask(b):
    t=np.arange(2000)*.01
    signal=np.sin(2*np.pi*3*t)+.5*np.sin(2*np.pi*4*t)
    record=np.stack([2*signal,2*signal,signal],axis=-1)[:,None,:]
    acc=HVSRAccumulator(.01,len(t),1,2,5,smoothing_b=b)
    for _ in range(4): acc.add(record)
    result=acc.finish()
    good=result.valid_windows[0]>=3
    assert good.any()
    np.testing.assert_allclose(result.geometric_mean[0,good],2,rtol=1e-12)
    bad=HVSRAccumulator(.01,len(t),1,2,5,smoothing_b=b)
    record[...,2]=0
    for _ in range(4): bad.add(record)
    assert np.isnan(bad.finish().geometric_mean).all()
    assert not bad.finish().valid_windows.any()


def test_theory_only_does_not_require_geometry_or_call_forward(monkeypatch):
    mapping=load_template_mapping(ROOT/"examples/active_quick.toml")
    mapping.pop("geometry")
    mapping.pop("active_source")
    context=theory_context(mapping,ROOT)
    def theory(model,f,**kwargs):
        assert kwargs["love_modes"] == 0
        return SimpleNamespace(frequency_hz=f,rayleigh_mps=np.ones((len(f),1))*200,love_mps=np.empty((len(f),0)))
    monkeypatch.setattr("wavedispx.experiment.theoretical_dispersion",theory)
    r=run_experiment(context,ExperimentRequest(product="theory",waves="rayleigh"))
    assert r.theory is not None and not r.images
    assert "未选择" in r.statuses["love"]
    assert context.shot is None


def test_polarization_invalid_and_capability_reasons():
    mapping=load_template_mapping(ROOT/"examples/passive_quick.toml")
    c=config_from_mapping(mapping,base_directory=ROOT/"examples")
    assert "缺少" in wave_unavailable(c,"love")
    mapping["ambient_sources"].update(component="three_component",polarization="independent_hv",horizontal_power_fraction=0.)
    c=config_from_mapping(mapping,base_directory=ROOT/"examples")
    assert "为0" in wave_unavailable(c,"love")
    mapping["noise"]["temporal_model"]="poisson_ricker"
    with pytest.raises(ValueError,match="stationary_gaussian"):
        config_from_mapping(mapping,base_directory=ROOT/"examples")


def test_hvsr_rejected_before_energy_calculation():
    c=config_from_mapping(load_template_mapping(ROOT/"examples/active_quick.toml"),base_directory=ROOT/"examples")
    with pytest.raises(ValueError,match="HVSR"):
        run_experiment(SimpleNamespace(config=c),ExperimentRequest(product="energy_hvsr"))


def test_vector_config_rejects_das_and_legacy_ambiguity():
    mapping=load_template_mapping(ROOT/"examples/active_quick.toml")
    mapping["active_source"]["force_azimuth_deg"]=30
    with pytest.raises(ValueError,match="vector"):
        config_from_mapping(mapping,base_directory=ROOT/"examples")
    mapping["active_source"]["component"]="vector"
    config_from_mapping(mapping,base_directory=ROOT/"examples")


def test_compare_methods_is_not_silently_ignored():
    mapping=load_template_mapping(ROOT/"examples/active_quick.toml")
    mapping["processing"]["compare_methods"]=True
    c=config_from_mapping(mapping,base_directory=ROOT/"examples")
    with pytest.raises(ValueError,match="compare_methods"):
        run_experiment(SimpleNamespace(config=c),ExperimentRequest(product="energy"))


def test_default_legacy_coefficients_are_bitwise_unchanged():
    a=np.random.default_rng(17).normal(size=(5,3,20))+1j
    source=SimpleNamespace(polarization="legacy_xyz")
    assert polarize_coefficients(a,source) is a


def test_zero_projection_is_not_normalized_into_false_love():
    from wavedispx.source_vector import project_green_to_rtz
    fields=np.random.default_rng(5).normal(size=(12,6,5))
    rt=project_green_to_rtz(fields,np.ones(6)*np.pi/4,force_vector(1,45,90))
    np.testing.assert_array_equal(rt[...,1],0)


def test_old_cli_rejects_new_product_instead_of_silent_legacy_run(capsys):
    from wavedispx.cli import main
    assert main(["run",str(ROOT/"examples/passive_hvsr_studio.toml")])==2
    assert "run_experiment" in capsys.readouterr().err
