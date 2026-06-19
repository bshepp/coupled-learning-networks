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
from spring_network import tangent_stiffness, bond_extensions, m as _mB, dm as _dmB, d2m as _d2mB


def mech_adjoint_gradient(net, k, l, patterns, d, targets):
    """dL/dk, dL/dl for L = 0.5*mean_p (r_p - t_p)^2 via the implicit function theorem.
    At equilibrium grad_x E = 0, so dx/dtheta = -Hff^{-1} d(grad_free E)/dtheta. Adjoint:
    Hff lam = dL/dx_free; dL/dtheta = -lam . d(grad_free E)/dtheta. lam = 0 on clamped DOFs."""
    from mechanical_learning import free_phase_mech, readout, _input_clamp
    n = net.x0.shape[0]
    gk = np.zeros(len(net.edges)); gl = np.zeros(len(net.edges))
    onode, odim = divmod(net.output_dof, 2)
    P = len(patterns)
    for p in range(P):
        x, _ = free_phase_mech(net, k, l, patterns[p], d)
        clamp = _input_clamp(net, patterns[p], d)
        free = np.array([dd for dd in range(2 * n) if dd not in clamp], int)
        fpos = {dd: idx for idx, dd in enumerate(free)}
        H = tangent_stiffness(net, x, k, l)
        Hff = H[np.ix_(free, free)]
        rhs = np.zeros(free.size)
        r = readout(net, x)
        rhs[fpos[net.output_dof]] = (r - targets[p]) / P          # dL/dx at the output free DOF
        lam = np.zeros(2 * n)
        lam[free] = np.linalg.solve(Hff, rhs)
        # d(grad E)/dtheta per bond: dgrad/dk_b at node i = (T_b/k_b) dh ; dgrad/dl_b = -kt_b dh
        for b, (i, j) in enumerate(net.edges):
            dvec = x[i] - x[j]; L = np.linalg.norm(dvec); dh = dvec / L; e = L - l[b]
            me = float(_mB(e, net.alpha, net.w)); m1 = float(_dmB(e, net.alpha, net.w))
            Tk = 0.5 * m1 * e * e + me * e                        # T/k = d(E/k)/dL coefficient
            dlam_i = lam[2 * i:2 * i + 2]; dlam_j = lam[2 * j:2 * j + 2]
            dgk = Tk * dh                                         # dgrad_i/dk ; node j = -dgk
            gk[b] += -(dlam_i @ dgk - dlam_j @ dgk)
            # kt for dl term
            m2 = float(_d2mB(e, net.alpha, net.w))
            kt = 0.5 * k[b] * (m2 * e * e + 4.0 * m1 * e + 2.0 * me)
            dgl = -kt * dh                                        # dgrad_i/dl ; node j = -dgl
            gl[b] += -(dlam_i @ dgl - dlam_j @ dgl)
    return gk, gl


def gd_baseline_mech(task, k0, l0, epochs, lr, k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0, audit=None):
    from mechanical_learning import free_phase_mech, readout
    from audit import OFFLINE_GRADIENT
    from tasks import two_sided_separation
    net, patterns, labels, d = task["net"], task["patterns"], task["labels"], task["d"]
    targets = np.where(labels > 0, task["target_pos"], task["target_neg"])
    k = np.array(k0, float); l = np.array(l0, float)
    readouts = None
    for _ in range(epochs):
        gk, gl = mech_adjoint_gradient(net, k, l, patterns, d, targets)
        k = np.clip(k - lr * gk, k_min, k_max)
        l = np.clip(l - lr * gl, l_min, l_max)
        if audit is not None:
            audit.record(OFFLINE_GRADIENT, 2 * len(net.edges))
    readouts = np.array([readout(net, free_phase_mech(net, k, l, patterns[p], d)[0])
                         for p in range(len(patterns))])
    return {"k": k, "l": l, "sep": two_sided_separation(readouts, labels), "readouts": readouts}
