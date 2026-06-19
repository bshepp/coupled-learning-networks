# tests/test_coupled_learning.py
import numpy as np
import pytest
from network import grid_network, RESISTOR
from tasks import regression_task
from coupled_learning import free_phase, nudged_phase, train
from audit import AuditLog


def test_nudge_moves_output_toward_target():
    net = grid_network(3, 3, n_in=1, n_out=1)
    g = np.full(len(net.edges), 1.0)
    inputs = np.array([[1.0]])
    targets = np.array([[0.9]])     # above the free output of a uniform net
    Vf = free_phase(net, g, inputs)
    free_out = Vf[:, net.outputs]
    Vn = nudged_phase(net, g, inputs, targets, beta=0.1, free_out=free_out)
    nud_out = Vn[:, net.outputs]
    # nudged output sits strictly between free output and target
    assert free_out[0, 0] < nud_out[0, 0] <= targets[0, 0] + 1e-9


def test_training_reduces_error():
    task = regression_task(seed=0)
    params0 = np.full(len(task.net.edges), 1.0)
    audit = AuditLog()
    res = train(task.net, params0, RESISTOR, task, beta=0.05, eta=0.5, epochs=200, audit=audit)
    assert res["history"][-1] < res["history"][0]
    assert res["final_error"] < 0.1
    assert audit.assert_physics_local() is True
