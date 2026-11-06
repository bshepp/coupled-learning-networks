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


def _snap_cell():
    # pre-compressed two-bar von-Mises cell: apex wells at y = +/- sqrt(1.1^2 - 1), saddle at y = 0
    x0 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    net = SpringNetwork(x0, [(0, 1), (1, 2)], pinned_dofs=[dof(0, 0), dof(0, 1), dof(2, 0), dof(2, 1)],
                        input_dofs=[], output_dof=dof(1, 1))
    k = np.array([1.0, 1.0]); l = np.array([1.1, 1.1])
    clamp = {d: x0.reshape(-1)[d] for d in net.pinned_dofs}
    return net, k, l, clamp


def test_settle_finds_a_well_where_relax_mech_raises_and_never_returns_the_saddle():
    from mechanical_solver import settle
    net, k, l, clamp = _snap_cell()
    well = np.sqrt(1.1 ** 2 - 1.0)
    near_saddle = net.x0.copy(); near_saddle[1, 1] = 0.02
    with pytest.raises(RuntimeError):
        relax_mech(net, k, l, clamp, x_init=near_saddle)              # B's solver refuses indefinite regions
    x, info = settle(net, k, l, clamp, x_init=near_saddle)
    assert info["converged"] and abs(abs(x[1, 1]) - well) < 1e-6      # lands in a well
    x_s, info_s = settle(net, k, l, clamp, x_init=net.x0.copy())      # exactly AT the saddle (zero force)
    assert info_s["escaped_saddle"] >= 1 and abs(abs(x_s[1, 1]) - well) < 1e-6
    x_in, _ = settle(net, k, l, clamp, x_init=np.array([[0.0, 0.0], [1.0, 0.3], [2.0, 0.0]]))
    x_ref, _ = relax_mech(net, k, l, clamp, x_init=np.array([[0.0, 0.0], [1.0, 0.3], [2.0, 0.0]]))
    assert x_in == pytest.approx(x_ref, abs=1e-7)                     # in-basin it agrees with Newton


def test_relax_reports_the_measured_stability_margin():
    net, l = _bridge()
    k = np.array([1.0, 1.0])
    clamp = {d: net.x0.reshape(-1)[d] for d in net.pinned_dofs}
    _, info = relax_mech(net, k, l, clamp, x_init=net.x0.copy())
    assert info["min_eig"] > 0                                        # a genuine minimum
    net_s, k_s, l_s, clamp_s = _snap_cell()
    _, info_s = relax_mech(net_s, k_s, l_s, clamp_s, x_init=net_s.x0.copy())   # started exactly at the saddle
    assert info_s["converged"] and info_s["min_eig"] < 0              # Newton "converges" to the saddle: measured
