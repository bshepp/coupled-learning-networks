# mechanical_solver.py
"""Quasi-static equilibrium of a spring network by damped Newton (energy minimization) with a
backtracking line search. clamp fixes input/pinned/output DOFs (Dirichlet); the rest are free.
warm-start (x_init) keeps the nudged phase in the free phase's basin; basin_jump flags when a
relaxation lands in a different basin than expected."""
import numpy as np
from spring_network import energy, forces, tangent_stiffness


def relax_mech(net, k, l, clamp, x_init=None, max_iter=100, tol=1e-9):
    n = net.x0.shape[0]
    x = (net.x0.copy() if x_init is None else np.asarray(x_init, float).copy())
    for d, v in clamp.items():
        node, dim = divmod(d, 2); x[node, dim] = v
    free = np.array([d for d in range(2 * n) if d not in clamp], int)
    if free.size == 0:
        return x, {"converged": True, "iters": 0, "energy": energy(net, x, k, l)}
    for it in range(max_iter):
        R = forces(net, x, k, l).reshape(-1)[free]          # net force on free DOFs (want 0)
        if np.max(np.abs(R)) < tol:
            return x, {"converged": True, "iters": it, "energy": energy(net, x, k, l)}
        H = tangent_stiffness(net, x, k, l)
        Hff = H[np.ix_(free, free)]
        try:
            step = np.linalg.solve(Hff, R)                  # Newton: minimize E (grad = -R, Hess = Hff)
        except np.linalg.LinAlgError:
            raise RuntimeError("singular tangent stiffness (floppy mode?) in relax_mech")
        E0 = energy(net, x, k, l)
        xf = x.reshape(-1).copy()
        s = 1.0
        for _ in range(40):
            xt = xf.copy(); xt[free] += s * step
            if energy(net, xt.reshape(n, 2), k, l) < E0:
                break
            s *= 0.5
        xf[free] += s * step
        x = xf.reshape(n, 2)
    raise RuntimeError(f"relax_mech did not converge in {max_iter} iters "
                       f"(max|force|={float(np.max(np.abs(R))):.2e})")


def basin_jump(x_free, x_nudged, expected, factor=10.0):
    """Flag a basin change: if the largest node displacement between the two equilibria exceeds
    `factor` times the expected nudge magnitude, the nudged solve fell into a different basin."""
    disp = float(np.max(np.linalg.norm(np.asarray(x_nudged) - np.asarray(x_free), axis=1)))
    return (disp > factor * expected), disp
