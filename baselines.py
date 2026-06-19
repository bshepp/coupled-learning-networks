# baselines.py
"""The offline 'CPU computes the update' control: the exact loss gradient by the adjoint
method (validated against finite differences) and gradient descent on it. Plus the
gradient-alignment statistics that test whether the local rule tracks the true gradient.
Loss L = 0.5 * mean_patterns( sum_outputs (V_o - t_o)^2 )."""
import numpy as np
from network import conductance_edges
from coupled_learning import free_phase, edge_drops


def _full_laplacian(net, g):
    n = net.n_nodes
    L = np.zeros((n, n))
    for (a, b), ge in zip(net.edges, g):
        L[a, a] += ge; L[b, b] += ge
        L[a, b] -= ge; L[b, a] -= ge
    return L


def adjoint_gradient(net, g, task):
    """dL/dg_e by the adjoint method, averaged over patterns. For each pattern: solve the
    forward floating-node system, then the adjoint system (same symmetric L_ff), then
    dL/dg_e = (lambda_i - lambda_j)(V_i - V_j) with lambda=0 at clamped nodes."""
    g = np.asarray(g, float)
    n = net.n_nodes
    clamped = set(net.inputs) | ({net.ground} if net.ground is not None else set())
    floating = [i for i in range(n) if i not in clamped]
    fpos = {node: k for k, node in enumerate(floating)}
    L = _full_laplacian(net, g)
    Lff = L[np.ix_(floating, floating)]
    P = task.inputs.shape[0]
    grad = np.zeros(len(net.edges))

    # forward node voltages for all patterns (reuse the network solver)
    V = free_phase(net, g, task.inputs)                       # (P, n)
    out_idx = list(net.outputs)

    for p in range(P):
        r = np.zeros(len(floating))                           # dL/dV restricted to floating
        for j, onode in enumerate(out_idx):
            r[fpos[onode]] = V[p, onode] - task.targets[p, j]
        lam_f = np.linalg.solve(Lff, -r)                      # L_ff^T lam = -dL/dV_f
        lam = np.zeros(n)
        for node, k in fpos.items():
            lam[node] = lam_f[k]
        for e, (a, b) in enumerate(net.edges):
            grad[e] += (lam[a] - lam[b]) * (V[p, a] - V[p, b])
    return grad / P


def gd_baseline(net, params0, law, task, epochs, lr, p_min=1e-3, p_max=10.0, audit=None):
    from audit import OFFLINE_GRADIENT
    from tasks import task_error
    params = np.array(params0, float)
    history = []
    for _ in range(epochs):
        g = law.g_of_p(params)
        out = free_phase(net, g, task.inputs)[:, net.outputs]
        history.append(task_error(out, task.targets))
        grad_g = adjoint_gradient(net, g, task)
        grad_p = grad_g * law.dg_dp(params)
        params = np.clip(params - lr * grad_p, p_min, p_max)
        if audit is not None:
            audit.record(OFFLINE_GRADIENT, len(np.atleast_1d(params)))
    g = law.g_of_p(params)
    out = free_phase(net, g, task.inputs)[:, net.outputs]
    return {"params": params, "history": history, "final_error": task_error(out, task.targets)}


def gradient_alignment(delta_local, neg_grad):
    a = np.asarray(delta_local, float)
    b = np.asarray(neg_grad, float)
    sign_match = float(np.mean(np.sign(a) == np.sign(b)))
    rho = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {"sign_match": sign_match, "rho": rho, "cosine": cosine}
