"""Classroom contracts and real-solver/analytic regression checks."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from scipy.optimize import brentq
from wavedispx.classroom.engine import (PRESETS, model_from_rows, model_csv, read_model,
    calculate, ideal_response, cps_programs, find_qedisp_forward)
from wavedispx.classroom.plotting import export_result


def preset(index=0):
    return model_from_rows([(*r,np.inf,np.inf) for r in list(PRESETS.values())[index]])


def require_backends():
    try: cps_programs(); find_qedisp_forward()
    except FileNotFoundError as exc: pytest.skip(str(exc))


def test_csv_roundtrip_and_rejected_physics():
    m=preset(1)
    assert read_model(model_csv(m))==m
    text=model_csv(m)
    with pytest.raises(ValueError,match='Qp'):
        read_model(text.replace('inf,inf','100,100'))
    with pytest.raises(ValueError,match='半空间'):
        read_model(text.replace('inf,1700','20,1700'))
    with pytest.raises(ValueError,match='流体'):
        read_model(text.replace('320.0','0.0'))
    with pytest.raises(ValueError,match='体积模量'):
        model_from_rows([(np.inf,330,320,1800,np.inf,np.inf)])
    with pytest.raises(ValueError,match='表头'):
        read_model(text.replace('vp_mps','vp_kmps'))


def test_halfspace_analytic_sh_compliance():
    m=preset(2); f=np.array([5.,15.]); c=np.array([300.,600.,900.]); gamma=.2
    response=ideal_response(m,f,c,gamma)
    layer=m.layers[0]; omega=2*np.pi*f[:,None]-1j*gamma
    k=2*np.pi*f[:,None]/c
    expected=1/(layer.density_kgm3*layer.vs_mps**2*np.sqrt(k*k-omega*omega/layer.vs_mps**2))
    np.testing.assert_allclose(abs(response['love']),abs(expected),rtol=1e-11)


def test_ideal_rayleigh_peak_matches_halfspace_analytic_root():
    m=preset(2); vp,vs=m.layers[0].vp_mps,m.layers[0].vs_mps
    ratio=(vs/vp)**2
    root=vs*brentq(lambda x:x**6-8*x**4+(24-16*ratio)*x*x-16*(1-ratio),.5,.999)
    c=np.linspace(root-3,root+3,601)
    response=ideal_response(m,np.array([10.]),c,.01)
    peak=c[np.argmax(abs(response['rayleigh'][0])**2)]
    assert abs(peak-root) <= .02


def test_native_backends_and_analytic_rayleigh_halfspace():
    require_backends(); m=preset(2)
    vp,vs=m.layers[0].vp_mps,m.layers[0].vs_mps
    ratio=(vs/vp)**2
    x=brentq(lambda x:x**6-8*x**4+(24-16*ratio)*x*x-16*(1-ratio),.5,.999)
    for backend in ['cps','qedisp']:
        result=calculate(m,np.array([4.,10.,30.]),2,np.linspace(100,900,41),.2,backend=backend)
        np.testing.assert_allclose(result['curves']['rayleigh'][:,0],x*vs,atol=.002,rtol=0)
        assert np.isnan(result['curves']['love']).all()


@pytest.mark.parametrize('index',[0,1])
def test_layered_native_comparison_and_export(index,tmp_path):
    require_backends(); m=preset(index); f=np.linspace(4,40,37)
    r=calculate(m,f,3,np.linspace(100,900,321),.2,True,backend='qedisp')
    cps=calculate(m,f,3,np.linspace(100,900,321),.2,backend='cps')
    for wave in ['rayleigh','love']:
        q,c=r['curves'][wave],cps['curves'][wave]
        common=np.isfinite(q)&np.isfinite(c)
        assert np.all(common[:,0])
        np.testing.assert_allclose(q[common],c[common],atol=.02,rtol=0)
    if index==0:
        # Exact two-layer Love characteristic equation at native fundamental roots.
        c=cps['curves']['love'][:,0]; top,base=m.layers
        q=2*np.pi*f*np.sqrt(1/top.vs_mps**2-1/c**2)
        s=2*np.pi*f*np.sqrt(1/c**2-1/base.vs_mps**2)
        residual=np.sin(q*top.thickness_m)-(base.density_kgm3*base.vs_mps**2*s)/(top.density_kgm3*top.vs_mps**2*q)*np.cos(q*top.thickness_m)
        np.testing.assert_allclose(residual,0,atol=2e-5)
    export_result(tmp_path/'out.zip',r)
    with zipfile.ZipFile(tmp_path/'out.zip') as z:
        assert read_model(z.read('model.csv').decode())==m
        raw=np.load(io.BytesIO(z.read('arrays.npz')))
        np.testing.assert_array_equal(raw['rayleigh_raw_power'],abs(r['response']['rayleigh'])**2)
        manifest=json.loads(z.read('manifest.json'))
        import hashlib
        for name,digest in manifest['files_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest


@pytest.mark.parametrize('backend',['qedisp','cps'])
def test_selected_backend_isolated_and_matches_previous_arrays(backend,monkeypatch):
    import wavedispx.classroom.engine as engine
    require_backends()
    def forbidden(*args,**kwargs): raise AssertionError('Unselected backend called')
    monkeypatch.setattr(engine,'cps_curves' if backend=='qedisp' else 'find_qedisp_forward',forbidden)
    old=Path(__file__).resolve().parents[1]/'docs/classroom/qc/low_velocity_layer.zip'
    with zipfile.ZipFile(old) as z:
        a=np.load(io.BytesIO(z.read('arrays.npz')))
        r=calculate(preset(1),a['frequency_hz'],3,a['velocity_mps'],.2,backend=backend)
        assert r['response'] is None
        for wave in ('rayleigh','love'):
            np.testing.assert_array_equal(r['curves'][wave],a[wave+'_'+backend+'_mps'])


@pytest.fixture
def qapp():
    from wavedispx.classroom.app import ClassroomWindow
    from PySide6.QtWidgets import QApplication
    app=QApplication.instance() or QApplication([])
    yield app


def wait_job(qapp,w):
    import time
    deadline=time.monotonic()+15
    while w.busy and time.monotonic()<deadline:
        qapp.processEvents()
        time.sleep(.01)
    assert not w.busy
    qapp.processEvents()


def test_desktop_lazy_overlay_retains_curves(qapp,monkeypatch,tmp_path):
    require_backends()
    from wavedispx.classroom.app import ClassroomWindow
    import wavedispx.classroom.engine as engine
    w=ClassroomWindow()
    w.nf.setValue(7); w.nc.setValue(41)
    assert not w.energy.isChecked()
    w.start(); wait_job(qapp,w)
    curves=w.result['curves']
    assert w.result['response'] is None
    original=curves['rayleigh'].copy()
    def forbidden(*args,**kwargs): raise AssertionError('Curve solver rerun by ideal')
    monkeypatch.setattr(engine,'run_qedisp_forward',forbidden)
    limits=w.figures['rayleigh'].axes[0].get_ylim()
    w.energy.setChecked(True); wait_job(qapp,w)
    assert w.result['curves'] is curves and w.result['response'] is not None
    np.testing.assert_array_equal(curves['rayleigh'],original)
    assert w.figures['rayleigh'].axes[0].get_ylim()==limits
    response=w.result['response']
    w.energy.setChecked(False); w.energy.setChecked(True)
    assert not w.busy and w.result['response'] is response
    w.gamma.setValue(.1); w.energy_settings_changed(); wait_job(qapp,w)
    assert w.result['curves'] is curves and w.result['response'] is not response
    export_result(tmp_path/'selected.zip',w.result)
    with zipfile.ZipFile(tmp_path/'selected.zip') as z:
        assert json.loads(z.read('config.json'))['backend']=='qedisp'
        assert 'cps_mps' not in z.read('rayleigh_curves.csv').decode()
    w.table.item(0,2).setText('300')
    assert w.result is None and not w.export_button.isEnabled()
    w.close()


def test_overlay_failure_keeps_curves(qapp,monkeypatch):
    require_backends()
    import wavedispx.classroom.app as ui
    w=ui.ClassroomWindow(); w.nf.setValue(3)
    w.start(); wait_job(qapp,w)
    original=w.result
    def fail(**kwargs): raise RuntimeError('intentional ideal failure')
    monkeypatch.setattr(ui,'add_ideal',fail)
    monkeypatch.setattr(ui.QMessageBox,'warning',lambda *args:None)
    w.energy.setChecked(True); wait_job(qapp,w)
    assert w.result is original and w.export_button.isEnabled()
    assert not w.energy.isChecked()
    w.close()


def test_compact_layout_density_and_chinese_help(qapp):
    from PySide6.QtWidgets import QScrollArea
    from wavedispx.classroom.app import ClassroomWindow
    from wavedispx.classroom.help_text import TIPS
    w=ClassroomWindow(); w.resize(1200,780); w.show(); qapp.processEvents()
    assert not w.findChildren(QScrollArea)
    assert w.tabs.count()==2
    for key in TIPS:
        if key in ('thickness_m','vp_mps','vs_mps','density_kgm3','qp','qs'): continue
        widget=getattr(w,key)
        assert widget.toolTip() and any('\u4e00'<=c<='\u9fff' for c in widget.toolTip())
        assert widget.isVisible()
        from PySide6.QtCore import QPoint
        corner=widget.mapTo(w,QPoint(widget.width()-1,widget.height()-1))
        assert w.rect().contains(corner), key
    for j in range(6):
        assert w.table.horizontalHeaderItem(j).toolTip()
        assert w.table.item(0,j).toolTip()
    axes=w.figures['model'].axes
    assert len(axes)==2
    np.testing.assert_array_equal(axes[1].lines[0].get_xdata(),[1800,1800,2150,2150])
    assert axes[0].get_ylim()==axes[1].get_ylim()
    w.close()
