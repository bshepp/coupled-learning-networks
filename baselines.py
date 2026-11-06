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


def mech_ceiling(task, k0, l0, maxiter=2000, max_restarts=8, k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0,
                 barrier=10.0, audit=None):
    """REPRESENTABILITY ceiling for a mechanical task: L-BFGS-B on the implicit adjoint of the per-pattern
    target MSE, run to convergence (restarting from the best solvable point when a line search aborts at
    the stability wall). This is the budget-saturated "can THIS substrate express it?" number the
    commons' foreclosure-can-be-a-budget-artifact lesson requires beside any plain-GD baseline. Failed
    equilibrium solves return a quadratic barrier back to the last solvable point. OFFLINE_GRADIENT."""
    from scipy.optimize import minimize
    from mechanical_learning import free_phase_mech, readout, task_targets
    from audit import OFFLINE_GRADIENT
    from tasks import task_error
    net, patterns, d = task["net"], task["patterns"], task["d"]
    targets = task_targets(task); P = len(patterns); E = len(net.edges)
    state = {"last_ok": None, "best": None, "n_fail": 0, "n_eval": 0}

    def f(theta):
        k, l = theta[:E], theta[E:]
        state["n_eval"] += 1
        try:
            xs = [free_phase_mech(net, k, l, patterns[p], d)[0] for p in range(P)]
            r = np.array([readout(net, x) for x in xs])
            gk, gl = mech_readout_adjoint(net, k, l, patterns, d, (r - targets) / P, xs=xs)
        except RuntimeError:
            state["n_fail"] += 1
            if state["last_ok"] is None:
                return 1e3, np.zeros_like(theta)
            t_ok, L_ok = state["last_ok"]; dt = theta - t_ok
            return L_ok + barrier * float(dt @ dt) + 1e-3, 2.0 * barrier * dt
        if audit is not None:
            audit.record(OFFLINE_GRADIENT, 2 * E)
        L = 0.5 * float(np.mean((r - targets) ** 2))
        state["last_ok"] = (theta.copy(), L)
        if state["best"] is None or L < state["best"][0]:
            state["best"] = (L, theta.copy())
        return L, np.concatenate([gk, gl])

    x = np.concatenate([np.asarray(k0, float), np.asarray(l0, float)])
    bounds = [(k_min, k_max)] * E + [(l_min, l_max)] * E
    nit = 0; converged = False; restarts = 0
    while nit < maxiter and restarts <= max_restarts:
        res = minimize(f, x, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter - nit, "maxfun": 4 * (maxiter - nit)})
        nit += int(res.nit)
        if str(res.message).startswith("CONVERGENCE"):
            converged = True; break
        restarts += 1
        if state["best"] is None:
            break
        x = state["best"][1].copy()
    if state["best"] is None:
        return {"k": np.asarray(k0, float), "l": np.asarray(l0, float), "final_error": float("inf"),
                "converged": False, "nit": nit, "restarts": restarts, "eval_fail_rate": 1.0}
    theta = state["best"][1]; k, l = theta[:E], theta[E:]
    r = np.array([readout(net, free_phase_mech(net, k, l, patterns[p], d)[0]) for p in range(P)])
    return {"k": k, "l": l, "final_error": task_error(r, targets), "readouts": r,
            "converged": converged, "nit": nit, "restarts": restarts,
            "eval_fail_rate": state["n_fail"] / max(state["n_eval"], 1)}


def gd_baseline_mech(task, k0, l0, epochs, lr, k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0, audit=None):
    from mechanical_learning import free_phase_mech, readout, task_targets
    from audit import OFFLINE_GRADIENT
    from tasks import two_sided_separation, task_error
    net, patterns, d = task["net"], task["patterns"], task["d"]
    labels = task.get("labels")
    targets = task_targets(task)
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
        sep = two_sided_separation(readouts, labels) if labels is not None else float("nan")
        err = task_error(readouts, targets)
    except RuntimeError:
        # Any pattern's final solve fails → baseline could not be evaluated (instability)
        readouts = np.zeros(len(patterns))
        sep = float("-inf"); err = float("inf")
    return {"k": k, "l": l, "sep": sep, "final_error": err, "readouts": readouts}
