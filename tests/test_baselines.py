# tests/test_baselines.py
import numpy as np
import pytest
from network import grid_network, RESISTOR
from tasks import regression_task
from baselines import adjoint_gradient, gd_baseline, gradient_alignment


def _fd_gradient(net, g, task):
    from coupled_learning import free_phase
    eps = 1e-6
    grad = np.zeros(len(net.edges))
    for e in range(len(net.edges)):
        gp = g.copy(); gp[e] += eps
        gm = g.copy(); gm[e] -= eps
        op = free_phase(net, gp, task.inputs)[:, net.outputs]
        om = free_phase(net, gm, task.inputs)[:, net.outputs]
        lp = 0.5 * np.mean(np.sum((op - task.targets) ** 2, axis=1))
        lm = 0.5 * np.mean(np.sum((om - task.targets) ** 2, axis=1))
        grad[e] = (lp - lm) / (2 * eps)
    return grad


def test_adjoint_matches_finite_difference():
    task = regression_task(seed=2)
    g = np.linspace(0.5, 1.5, len(task.net.edges))
    assert adjoint_gradient(task.net, g, task) == pytest.approx(_fd_gradient(task.net, g, task), abs=1e-5)


def test_gd_baseline_converges():
    task = regression_task(seed=0)
    params0 = np.full(len(task.net.edges), 1.0)
    res = gd_baseline(task.net, params0, RESISTOR, task, epochs=300, lr=0.5)
    assert res["final_error"] < 0.05


def test_gradient_alignment_extremes():
    v = np.array([1.0, -2.0, 3.0, -0.5])
    same = gradient_alignment(v, v)
    assert same["sign_match"] == pytest.approx(1.0)
    assert same["rho"] == pytest.approx(1.0)
    assert same["cosine"] == pytest.approx(1.0)
    opp = gradient_alignment(v, -v)
    assert opp["sign_match"] == pytest.approx(0.0)
    assert opp["cosine"] == pytest.approx(-1.0)
    assert opp["rho"] == pytest.approx(-1.0)
