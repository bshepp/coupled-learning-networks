"""Batched Newton/Laplacian equilibrium solve + the relax seam.

Ported and adapted from self-assembly/stage2_solver.py (batched Newton Kirchhoff
forward solve). Edges carry a conductance weight g; the actual edge conductance is
g*phi(dV) (phi==1 for the linear resistor/flow case used in Experiment A). relax()
is the substrate-agnostic seam: Experiment B's vector spring solver plugs in here.
"""
import numpy as np


def _ones(u):
    return np.ones_like(u)


def _zeros(u):
    return np.zeros_like(u)


def assemble_laplacian(edges, V, weight_fn):
    """Batched conductance Laplacian (P, n, n). weight_fn(u, w) -> (P,) conductance for
    edge weight w and per-pattern bias u = V_a - V_b."""
    P, n = V.shape
    L = np.zeros((P, n, n))
    for (a, b, w) in edges:
        u = V[:, a] - V[:, b]
        g = np.asarray(weight_fn(u, w), float)
        L[:, a, a] += g; L[:, b, b] += g
        L[:, a, b] -= g; L[:, b, a] -= g
    return L


def solve_batched(edges, node_volts, floating, phi=_ones, dphi=_zeros, max_iter=60, tol=1e-11):
    """Node voltages by Newton on the Kirchhoff residual with the analytic dynamic-Laplacian
    Jacobian, batched over all patterns. node_volts: (P, n); floating positions may be np.nan —
    they are overwritten with 0.0 as the Newton warm-start guess. The warm start uses raw
    conductances (phi ignored): exact for the linear case, an approximate initial guess for
    nonlinear phi."""
    V = np.array(node_volts, float)
    P, n = V.shape
    fset = set(int(i) for i in floating)
    fidx = np.array(sorted(fset), int)
    fixed = np.array([i for i in range(n) if i not in fset], int)
    if fidx.size == 0:                      # nothing floats -> clamps are the answer
        return V
    V[:, fidx] = 0.0

    L0 = assemble_laplacian(edges, V, lambda u, w: np.full(u.shape, float(w)))
    Lff0 = L0[:, fidx[:, None], fidx[None, :]]
    Lfb0 = L0[:, fidx[:, None], fixed[None, :]]
    rhs0 = -np.einsum('pij,pj->pi', Lfb0, V[:, fixed])
    V[:, fidx] = np.linalg.solve(Lff0, rhs0[..., None])[..., 0]

    g_fn = lambda u, w: float(w) * np.asarray(phi(u), float)
    gdyn_fn = lambda u, w: float(w) * (np.asarray(phi(u), float) + u * np.asarray(dphi(u), float))

    last = np.inf
    for _ in range(max_iter):
        Lg = assemble_laplacian(edges, V, g_fn)
        R = np.einsum('pij,pj->pi', Lg, V)[:, fidx]
        last = float(np.max(np.abs(R)))
        if last < tol:
            return V
        Jd = assemble_laplacian(edges, V, gdyn_fn)
        Jff = Jd[:, fidx[:, None], fidx[None, :]]
        dVf = np.linalg.solve(Jff, (-R)[..., None])[..., 0]
        V[:, fidx] += dVf
    raise RuntimeError(f"solve_batched did not converge in {max_iter} iters (max|residual|={last:.2e})")


def relax(n_nodes, edges, clamp, phi=_ones, dphi=_zeros, **kw):
    """Solve node voltages. clamp: {node_index: scalar or (P,) array}. Floating nodes are
    those not present in clamp. Returns (P, n_nodes)."""
    P = 1
    for v in clamp.values():
        P = max(P, np.atleast_1d(v).shape[0])
    node_volts = np.full((P, n_nodes), np.nan)
    for node, val in clamp.items():
        node_volts[:, node] = val
    floating = [i for i in range(n_nodes) if i not in clamp]
    return solve_batched(edges, node_volts, floating, phi, dphi, **kw)
