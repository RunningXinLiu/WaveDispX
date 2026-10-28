import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from PySide6.QtWidgets import QApplication
from wavedispx.desktop.hvsr_view import station_axis, curve_figure, heatmap_figure
from wavedispx.desktop.studio import StudioWindow
from wavedispx.experiment import ExperimentResult, ExperimentRequest
from wavedispx.app_service import load_template_mapping

ROOT=Path(__file__).resolve().parents[1]


def data(n=4):
    return SimpleNamespace(frequency_hz=np.array([2.,3.,4.]),geometric_mean=np.arange(n*3,dtype=float).reshape(n,3)+1,
        log_std=np.ones((n,3))*.1,processed_windows=6,valid_windows=np.ones((n,3),int)*6)


def test_line_distance_and_area_ids():
    axis,order,label=station_axis([10,0,5],[2,2,2])
    np.testing.assert_array_equal(order,[1,2,0])
    np.testing.assert_allclose(axis,[0,5,10])
    axis,order,label=station_axis([0,1,0,1],[0,0,1,1])
    np.testing.assert_array_equal(order,np.arange(4))
    assert "Station ID" in label


def test_plots_preserve_values_and_invalid_mask():
    h=data()
    h.geometric_mean[1,1]=np.nan
    before=h.geometric_mean.copy()
    fig=curve_figure(h,[0,2],scatter=False)
    np.testing.assert_array_equal(fig.axes[0].lines[1].get_ydata(),before[2])
    plt.close(fig)
    fig,axis,order=heatmap_figure(h,[3,2,1,0],[0,0,0,0])
    actual=fig.axes[0].collections[0].get_array()
    np.testing.assert_array_equal(actual.filled(np.nan).reshape(4,3),before[order])
    np.testing.assert_array_equal(h.geometric_mean,before)
    assert fig.axes[0].collections[0].get_clim()==(0,12)
    plt.close(fig)


def test_all_invalid_and_empty_selection_render():
    h=data()
    h.geometric_mean[:]=np.nan
    for fig in (curve_figure(h,[]),heatmap_figure(h,[0,1,2,3],[0]*4)[0]):
        fig.canvas.draw()
        plt.close(fig)


def test_geometry_pick_multiselect_heatmap_and_dirty(monkeypatch):
    app=QApplication.instance() or QApplication([])
    w=StudioWindow(ROOT/"examples/passive_hvsr_studio.toml")
    w.resize(1728,1084)
    w.show()
    app.processEvents()
    monkeypatch.setattr(w.workflow,"simulate_observations",lambda: pytest.fail("display must not forward"))
    h=data(w.config.receivers.count) if hasattr(w,"config") else data(len(w.workflow.geometry.receiver_x_m))
    before=h.geometric_mean.copy()
    w._experiment_workflow=w.workflow
    r=ExperimentResult(ExperimentRequest(product="hvsr"),statuses={"rayleigh":"not requested","love":"not requested"},hvsr=h)
    w._stage_succeeded("experiment",r)
    app.processEvents()
    g=w.workflow.geometry
    canvas=w.geometry_panel._canvas
    canvas.draw()
    px,py=w.geometry_panel._figure.axes[0].transData.transform((g.receiver_x_m[20],g.receiver_y_m[20]))
    event=MouseEvent("button_press_event",canvas,px,py,button=1)
    canvas.callbacks.process("button_press_event",event)
    assert w.hvsr_station.currentIndex()==20
    w._select_hvsr_station(3,multi=True)
    assert 3 in w._hvsr_selected and w.hvsr_view.currentData()=="overlay"
    w.hvsr_view.setCurrentIndex(w.hvsr_view.findData("heatmap"))
    app.processEvents()
    canvas=w.hvsr_panel._canvas
    canvas.draw()
    px,py=w.hvsr_panel._figure.axes[0].transData.transform((3.,25.))
    canvas.callbacks.process("button_press_event",MouseEvent("button_press_event",canvas,px,py,button=1))
    assert w.hvsr_station.currentIndex()==5 and w.hvsr_view.currentData()=="single"
    np.testing.assert_array_equal(h.geometric_mean,before)
    w._mark_dirty()
    assert w.hvsr_controls.isHidden() and w.experiment_result is None
    w.close()
