"""Explicit small packaging benchmarks, not new scientific default settings."""
from copy import deepcopy
import numpy as np

CASES = {
    'active_vector': 'active_vector_studio.toml',
    'passive_vector': 'passive_vector_energy_studio.toml',
    'active_das_line': 'active_das_line_quick.toml',
    'passive_das_line': 'passive_das_line_quick.toml',
    'active_das_array': 'active_das_grid_quick.toml',
    'passive_das_array': 'passive_das_grid_quick.toml',
    'nonuniform_array': 'custom_geometry/passive_switching_array.toml',
    'hvsr': 'passive_hvsr_studio.toml',
}


def case_mapping(root, name, output):
    from wavedispx.app_service import load_template_mapping
    path = root/'examples'/CASES[name]
    mapping = deepcopy(load_template_mapping(path))
    mapping['run']['output_directory'] = str(output)
    mapping['run']['workers'] = 2
    if name != 'hvsr':
        mapping['experiment'] = {'product':'energy_theory','waves':'both','theory_step_hz':.1}
    if name in ('active_das_array','passive_das_array'):
        # Explicit 3x3 packaging fixture; do not rewrite the shipped templates.
        mapping['geometry']['receivers'] = dict(layout='grid', start_x_m=10.,stop_x_m=30.,
            start_y_m=-10.,stop_y_m=10.,spacing_x_m=10.,spacing_y_m=10.,distance_bin_m=5.)
    return mapping, path.parent


def arrays(workflow, result):
    data = {}
    for wave, im in result.images.items():
        data[wave+'_energy'] = im.energy
        data[wave+'_f'] = im.frequency_hz
        data[wave+'_v'] = im.velocity_mps
    if result.theory is not None:
        data.update(theory_f=result.theory.frequency_hz, theory_r=result.theory.rayleigh_mps,
                    theory_l=result.theory.love_mps)
    if workflow.shot is not None:
        s = workflow.shot
        data.update(shot_time=s.time_s, shot_offset=s.distance_m, shot_basis=s.components)
        if s.vector_xyz is not None: data['shot_xyz'] = s.vector_xyz
        if s.das_record is not None: data['shot_das'] = s.das_record.data
    if workflow.correlation is not None:
        c=workflow.correlation
        data['noise_preview']=c.preview_record
        for key,ncf in c.component_ncfs.items():
            data[key+'_ncf']=ncf.data
            data[key+'_lag']=ncf.lag_time_s
    if result.hvsr is not None:
        h=result.hvsr
        data.update(hvsr_f=h.frequency_hz,hvsr_mean=h.geometric_mean,
                    hvsr_scatter=h.log_std,hvsr_windows=h.valid_windows)
    return {k:np.asarray(v) for k,v in data.items()}


def compare(actual, expected):
    """Frozen tolerance: same inputs, no gain/time/phase alignment or fitting."""
    metrics={}
    assert set(actual)==set(expected), 'Output fields differ'
    for key,a in actual.items():
        b=expected[key]
        assert a.shape==b.shape, key+' shape'
        np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b),err_msg=key+' mask')
        finite=np.isfinite(b)
        av,bv=a[finite],b[finite]
        if key in ('theory_r','theory_l'):
            np.testing.assert_allclose(av,bv,rtol=2e-6,atol=.002,err_msg=key)
        elif key.endswith(('_f','_v','_lag')) or key in ('shot_time','shot_offset','hvsr_windows'):
            np.testing.assert_allclose(av,bv,rtol=1e-12,atol=1e-12,err_msg=key)
        else:
            denom=float(np.linalg.norm(bv))
            error=float(np.linalg.norm(av-bv))/(denom or 1.)
            assert error <= 1e-5, f'{key}: relative L2 {error} >1e-5'
            metrics[key]=error
    return metrics
