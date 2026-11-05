# baselines.py
"""The offline 'CPU computes the update' control: the exact loss gradient by the adjoint
method (validated against finite differences) and gradient descent on it. Plus the
gradient-alignment statistics that test whether the local rule tracks the true gradient.
Loss L = 0.5 * mean_patterns( sum_outputs (V_o - t_o)^2 )."""
import numpy as np
from coupled_learning import free_phase


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


# --- Experiment B: implicit-adjoint gradient through the nonlinear equilibrium ---------------
from spring_network import tangent_stiffness, _bond_geometry, bond_tension, bond_tangent


def mech_readout_adjoint(net, k, l, patterns, d, dL_dr, xs=None):
    """Generic implicit-function-theorem adjoint: for a loss L(r_1..r_P) of the free-phase readouts,
    given dL/dr_p per pattern, return (dL/dk, dL/dl). At equilibrium grad_x E = 0, so
    dx/dtheta = -Hff^{-1} d(grad_free E)/dtheta; adjoint solve Hff lam = dL/dx_free (lam = 0 on
    clamped DOFs); dL/dtheta = -lam . d(grad_free E)/dtheta. Reuses the Newton tangent stiffness.
    xs: optional precomputed free equilibria per pattern (skips the re-solve)."""
    from mechanical_learning import free_phase_mech, _input_clamp
    n = net.x0.shape[0]
    gk = np.zeros(len(net.edges)); gl = np.zeros(len(net.edges))
    dL_dr = np.asarray(dL_dr, float)
    for p in range(len(patterns)):
        x = xs[p] if xs is not None else free_phase_mech(net, k, l, patterns[p], d)[0]
        clamp = _input_clamp(net, patterns[p], d)
        free = np.array([dd for dd in range(2 * n) if dd not in clamp], int)
        H = tangent_stiffness(net, x, k, l)
        Hff = H[np.ix_(free, free)]
        rhs = np.zeros(free.size)
        rhs[int(np.searchsorted(free, net.output_dof))] = dL_dr[p]   # dL/dx at the output free DOF
        lam = np.zeros(2 * n)
        lam[free] = np.linalg.solve(Hff, rhs)
        # d(grad E)/dtheta per bond, at node i: (T_b/k_b) dh for k_b ; -kt_b dh for l_b (node j: negated)
        E, L, dh, e = _bond_geometry(net, x, l)
        Tk = bond_tension(net, e, np.ones_like(e))                  # T / k
        kt = bond_tangent(net, e, k)
        dlam = lam.reshape(n, 2)[E[:, 0]] - lam.reshape(n, 2)[E[:, 1]]  # (B, 2)
        proj = np.einsum("bi,bi->b", dlam, dh)
        gk += -(Tk * proj)
        gl += -(-kt * proj)
    return gk, gl


def mech_adjoint_gradient(net, k, l, patterns, d, targets):
    """dL/dk, dL/dl for the per-class target loss L = 0.5*mean_p (r_p - t_p)^2 (the Experiment B
    baseline objective), via mech_readout_adjoint with dL/dr_p = (r_p - t_p)/P."""
    from mechanical_learning import free_phase_mech, readout
    P = len(patterns)
    r = np.array([readout(net, free_phase_mech(net, k, l, patterns[p], d)[0]) for p in range(P)])
    return mech_readout_adjoint(net, k, l, patterns, d, (r - np.asarray(targets, float)) / P)


def gd_baseline_mech(task, k0, l0, epochs, lr, k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0, audit=None):
    from mechanical_learning import free_phase_mech, readout
    from audit import OFFLINE_GRADIENT
    from tasks import two_sided_separation
    net, patterns, labels, d = task["net"], task["patterns"], task["labels"], task["d"]
    targets = np.where(labels > 0, task["target_pos"], task["target_neg"])
    k = np.array(k0, float); l = np.array(l0, float)
    readouts = None
    for _ in range(epochs):
        try:
            gk, gl = mech_adjoint_gradient(net, k, l, patterns, d, targets)
            k = np.clip(k - lr * gk, k_min, k_max)
            l = np.clip(l - lr * gl, l_min, l_max)
            if audit is not None:
                audit.record(OFFLINE_GRADIENT, 2 * len(net.edges))
        except RuntimeError:
            # Solver failure during gradient computation; skip this epoch's step
            continue
    try:
        readouts = np.array([readout(net, free_phase_mech(net, k, l, patterns[p], d)[0])
                             for p in range(len(patterns))])
        sep = two_sided_separation(readouts, labels)
    except RuntimeError:
        # Any pattern's final solve fails → baseline could not be evaluated (instability)
        readouts = np.zeros(len(patterns))
        sep = float("-inf")
    return {"k": k, "l": l, "sep": sep, "readouts": readouts}
