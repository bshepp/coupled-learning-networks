# tests/test_mechanical_learning.py
import numpy as np
import pytest
from spring_network import SpringNetwork, triangulated_lattice, default_rest_lengths, dof
from mechanical_learning import readout, free_phase_mech, nudged_phase_mech, train_mech
from audit import AuditLog


def _small_task():
    x0, edges = triangulated_lattice(3, 3)
    in_nodes = [0, 1, 2]
    net = SpringNetwork(x0, edges, pinned_dofs=[dof(1, 0), dof(1, 1), dof(0, 0)],
                        input_dofs=[dof(n, 1) for n in in_nodes], output_dof=dof(8, 1))
    l = default_rest_lengths(net)
    k = np.ones(len(edges))
    return net, k, l


def test_nudge_moves_readout_toward_target():
    net, k, l = _small_task()
    pattern = np.array([1.0, -1.0, 1.0])
    x_free, _ = free_phase_mech(net, k, l, pattern, d=0.3)
    r_free = readout(net, x_free)
    target = r_free + 0.5         # push readout up
    x_nud, _, r0 = nudged_phase_mech(net, k, l, pattern, d=0.3, target=target, beta=0.2, x_free=x_free)
    assert r0 == pytest.approx(r_free, abs=1e-9)
    assert readout(net, x_nud) > r_free          # moved toward the (higher) target


def test_training_reduces_separation_gap_and_is_physics_local():
    # A friendly 2-pattern target: drive the readout of pattern A up, pattern B down (representable).
    net, k0, l0 = _small_task()
    patterns = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    labels = np.array([1.0, -1.0])
    task = {"net": net, "patterns": patterns, "labels": labels, "d": 0.3,
            "target_pos": 0.15, "target_neg": -0.15}
    audit = AuditLog()
    res = train_mech(task, k0, l0, beta=0.05, eta_k=2.0, eta_l=2.0, epochs=150, audit=audit)
    assert res["sep_history"][-1] > res["sep_history"][0]   # separation improved
    assert audit.assert_physics_local() is True
