from types import SimpleNamespace as NS

import matplotlib.pyplot as plt
import numpy as np

from wavedispx.plotting import plot_ncf_branches
from wavedispx.workflow_plotting import plot_workflow_forward


def test_ncf_uses_true_negative_branch_single_zero_and_near_to_far_rows():
    lag = np.array([0., 1., 2.])
    distance = np.array([20., 5.])
    pos = np.array([[0., 0.], [1., 2.], [3., 4.]])
    neg = np.array([[0., 0.], [10., 20.], [30., 40.]])
    branches = NS(positive=NS(lag_time_s=lag, distance_m=distance, data=pos),
                  negative=NS(lag_time_s=lag, distance_m=distance, data=neg))
    fig, ax = plt.subplots()
    mesh = plot_ncf_branches(ax, branches, title="Test")
    np.testing.assert_array_equal(mesh.get_array(), [[40., 20., 0., 2., 4.], [30., 10., 0., 1., 3.]])
    assert ax.get_xlim() == (-2., 2.)
    assert ax.get_ylim() == (5., 20.)
    assert ax.get_xlabel() == "Lag time (s)"
    np.testing.assert_array_equal(branches.negative.data, neg)
    plt.close(fig)


def test_active_gather_keeps_time_zero_at_top():
    shot = NS(distance_m=np.array([20., 5., 10.]), time_s=np.arange(5.) / 10,
              vertical_vertical=np.ones((5, 3)), horizontal_tangential=np.ones((5, 3)),
              das_record=None)
    workflow = NS(compute_forward=lambda: shot,
                  config=NS(mode="active", active_source=NS(dt_s=0.1)))
    fig = plot_workflow_forward(workflow)
    for ax in fig.axes:
        assert ax.get_ylim() == (0.4, 0.)
        assert ax.get_xlim() == (5., 20.)
        assert ax.get_ylabel() == "Time (s)"
    plt.close(fig)
