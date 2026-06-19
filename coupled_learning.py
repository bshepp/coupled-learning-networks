# coupled_learning.py
"""The heart: the two-phase coupled-learning rule (Stern, Hexner, Rocks, Liu, PRX 2021).
Free phase clamps inputs; nudged phase additionally clamps outputs a small step toward
the target; each edge updates from the LOCAL power contrast between the two relaxed states.
No global error signal touches a weight -- that is the audit point made concrete."""
import numpy as np
from solver import relax
from network import conductance_edges
from tasks import task_error
from audit import PHYSICS_LOCAL


def _input_clamp(net, inputs):
    P = inputs.shape[0]
    clamp = {node: inputs[:, k] for k, node in enumerate(net.inputs)}
    if net.ground is not None:
        clamp[net.ground] = np.zeros(P)
    return clamp


def free_phase(net, g, inputs):
    edges = conductance_edges(net, g)
    return relax(net.n_nodes, edges, _input_clamp(net, inputs))


def nudged_phase(net, g, inputs, targets, beta, free_out=None):
    if free_out is None:
        free_out = free_phase(net, g, inputs)[:, net.outputs]
    clamp = _input_clamp(net, inputs)
    for j, onode in enumerate(net.outputs):
        clamp[onode] = free_out[:, j] + beta * (targets[:, j] - free_out[:, j])
    edges = conductance_edges(net, g)
    return relax(net.n_nodes, edges, clamp)


def edge_drops(net, V):
    """Voltage drop across each edge for each pattern: (P, E)."""
    e = np.array(net.edges, int)
    return V[:, e[:, 0]] - V[:, e[:, 1]]


def local_update(net, params, law, V_free, V_nudged, beta, eta, audit=None):
    """Δp = (eta/beta) * dg/dp * mean_patterns[(ΔV_free)^2 - (ΔV_nudged)^2].

    Sign: nudging the output toward the target shrinks the drop on edges feeding it, so
    the contrast is positive there and raises their conductance -- gradient descent on the
    task loss. This sign is enforced by the keystone alignment test in Task 7."""
    dvf = edge_drops(net, V_free)
    dvn = edge_drops(net, V_nudged)
    contrast = np.mean(dvf ** 2 - dvn ** 2, axis=0)          # (E,)
    dp = (eta / beta) * law.dg_dp(np.asarray(params, float)) * contrast
    if audit is not None:
        audit.record(PHYSICS_LOCAL, len(np.atleast_1d(params)))
    return dp


def train(net, params0, law, task, beta, eta, epochs, p_min=1e-3, p_max=10.0, audit=None):
    params = np.array(params0, float)
    history = []
    for _ in range(epochs):
        g = law.g_of_p(params)
        Vf = free_phase(net, g, task.inputs)
        out = Vf[:, net.outputs]
        history.append(task_error(out, task.targets))
        Vn = nudged_phase(net, g, task.inputs, task.targets, beta, free_out=out)
        dp = local_update(net, params, law, Vf, Vn, beta, eta, audit=audit)
        params = np.clip(params + dp, p_min, p_max)
    g = law.g_of_p(params)
    out = free_phase(net, g, task.inputs)[:, net.outputs]
    return {"params": params, "history": history, "final_error": task_error(out, task.targets)}
