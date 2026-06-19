# tests/test_mechanical_solver.py
import numpy as np
import pytest
from spring_network import SpringNetwork, default_rest_lengths, dof
from mechanical_solver import relax_mech, basin_jump


def _bridge():
    # anchors (0,0) and (2,0) pinned; free node near (1,1); rest length sqrt(2) -> equilibrium (1,1)
    x0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    edges = [(0, 1), (1, 2)]
    net = SpringNetwork(x0, edges, pinned_dofs=[dof(0, 0), dof(0, 1), dof(2, 0), dof(2, 1)],
                        input_dofs=[], output_dof=dof(1, 1))
    l = default_rest_lengths(net)   # both sqrt(2)
    return net, l


def test_relax_finds_known_equilibrium():
    net, l = _bridge()
    k = np.array([1.0, 1.0])
    clamp = {d: net.x0.reshape(-1)[d] for d in net.pinned_dofs}  # anchors at reference
    x_init = net.x0.copy(); x_init[1] = [1.0, 0.6]               # perturb free node, upper basin
    x, info = relax_mech(net, k, l, clamp, x_init=x_init)
    assert info["converged"]
    assert x[1] == pytest.approx([1.0, 1.0], abs=1e-6)


def test_warm_start_reproduces_equilibrium():
    net, l = _bridge()
    k = np.array([1.0, 1.0])
    clamp = {d: net.x0.reshape(-1)[d] for d in net.pinned_dofs}
    x1, _ = relax_mech(net, k, l, clamp, x_init=net.x0.copy())
    x2, _ = relax_mech(net, k, l, clamp, x_init=x1.copy())       # warm start from equilibrium
    assert x2 == pytest.approx(x1, abs=1e-8)


def test_basin_jump_detector():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    near = a + np.array([[0.0, 0.0], [0.0, 0.02]])
    far = np.array([[0.0, 0.0], [1.0, -1.0]])
    assert basin_jump(a, near, expected=0.05)[0] is False
    assert basin_jump(a, far, expected=0.05)[0] is True
