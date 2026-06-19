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


def test_local_update_aligns_with_true_gradient_as_beta_shrinks():
    import numpy as np
    from network import grid_network, RESISTOR
    from tasks import regression_task
    from coupled_learning import free_phase, nudged_phase, local_update
    from baselines import adjoint_gradient, gradient_alignment

    task = regression_task(seed=3)
    net = task.net
    g = np.linspace(0.6, 1.4, len(net.edges))
    neg_grad = -adjoint_gradient(net, g, task)

    rhos = []
    for beta in (0.2, 0.05, 0.01, 0.002):
        Vf = free_phase(net, g, task.inputs)
        out = Vf[:, net.outputs]
        Vn = nudged_phase(net, g, task.inputs, task.targets, beta, free_out=out)
        dp = local_update(net, g, RESISTOR, Vf, Vn, beta, eta=1.0)
        rhos.append(gradient_alignment(dp, neg_grad)["rho"])

    # alignment improves as beta shrinks, and is near-perfect in the small-beta limit
    assert rhos[-1] > rhos[0]
    align = gradient_alignment(
        local_update(
            net, g, RESISTOR,
            free_phase(net, g, task.inputs),
            nudged_phase(net, g, task.inputs, task.targets, 0.002,
                         free_out=free_phase(net, g, task.inputs)[:, net.outputs]),
            0.002, eta=1.0),
        neg_grad)
    assert align["sign_match"] >= 0.95
    assert align["rho"] >= 0.95
