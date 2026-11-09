# spring_network.py
"""2D nonlinear spring network: positions, the smoothed-asymmetry constitutive law, and
energy/force/tangent-stiffness. Bond energy E = 0.5*k*m(e)*e^2 with e = L - l, L = |xi - xj|.
m(e) is a smooth tension-compression multiplier (m==1 when alpha==1). The geometric
nonlinearity (through L and the rotating bond direction) is what makes the response non-monotone."""
import numpy as np
from dataclasses import dataclass


def m(e, alpha, w):
    """Smooth asymmetry multiplier: ~1 in tension, ~alpha in compression. alpha=1 -> 1."""
    e = np.asarray(e, float)
    return alpha + (1.0 - alpha) * 0.5 * (1.0 + np.tanh(e / w))


def dm(e, alpha, w):
    e = np.asarray(e, float)
    return (1.0 - alpha) * 0.5 / w / np.cosh(e / w) ** 2


def d2m(e, alpha, w):
    e = np.asarray(e, float)
    t = np.tanh(e / w)
    return (1.0 - alpha) * 0.5 * (-2.0 * t) / (w * w) / np.cosh(e / w) ** 2


@dataclass
class SpringNetwork:
    x0: np.ndarray              # (N, 2) reference positions
    edges: list                  # list of (i, j)
    pinned_dofs: list            # flat DOF indices held at x0
    input_dofs: list             # flat DOF indices driven by inputs
    output_dof: int             # flat DOF index of the scalar readout
    alpha: float = 1.0
    w: float = 0.2


def dof(node, dim):
    return 2 * node + dim


def edge_array(net):
    """Edges as an (B, 2) int array (vectorized kernels index with it)."""
    return np.asarray(net.edges, int).reshape(-1, 2)


def _bond_geometry(net, x, l):
    """Per-bond vector, length, unit direction, and extension e = L - l. All (B,) or (B, 2)."""
    x = np.asarray(x, float)
    E = edge_array(net)
    dvec = x[E[:, 0]] - x[E[:, 1]]
    L = np.linalg.norm(dvec, axis=1)
    dh = dvec / L[:, None]
    e = L - np.asarray(l, float)
    return E, L, dh, e


def default_rest_lengths(net):
    E = edge_array(net)
    return np.linalg.norm(net.x0[E[:, 0]] - net.x0[E[:, 1]], axis=1)


def bond_extensions(net, x, l):
    return _bond_geometry(net, x, l)[3]


def bond_tension(net, e, k):
    """Scalar bond tension T = dE/dL = 0.5*k*m'(e)*e^2 + k*m(e)*e, vectorized over bonds."""
    k = np.asarray(k, float)
    return 0.5 * k * dm(e, net.alpha, net.w) * e * e + k * m(e, net.alpha, net.w) * e


def bond_tangent(net, e, k):
    """Scalar bond tangent stiffness kt = d^2 E/dL^2 = 0.5*k*(m''(e)*e^2 + 4*m'(e)*e + 2*m(e))."""
    k = np.asarray(k, float)
    return 0.5 * k * (d2m(e, net.alpha, net.w) * e * e
                      + 4.0 * dm(e, net.alpha, net.w) * e + 2.0 * m(e, net.alpha, net.w))


def energy(net, x, k, l):
    _, _, _, e = _bond_geometry(net, x, l)
    return float(np.sum(0.5 * np.asarray(k, float) * m(e, net.alpha, net.w) * e * e))


def forces(net, x, k, l):
    """F = -dE/dx, shape (N, 2). Per bond the scalar tension T = dE/dL acts along the unit bond vector."""
    x = np.asarray(x, float)
    E, L, dh, e = _bond_geometry(net, x, l)
    T = bond_tension(net, e, k)
    F = np.zeros_like(x)
    np.add.at(F, E[:, 0], -T[:, None] * dh)
    np.add.at(F, E[:, 1], T[:, None] * dh)
    return F


def tangent_stiffness(net, x, k, l):
    """Hessian d^2 E / dx^2, shape (2N, 2N). Per bond: material term kt*(dh dh^T) plus geometric
    term (T/L)*(I - dh dh^T), assembled as [[K,-K],[-K,K]] over the bond's two nodes."""
    x = np.asarray(x, float)
    n = x.shape[0]
    E, L, dh, e = _bond_geometry(net, x, l)
    T = bond_tension(net, e, k)
    kt = bond_tangent(net, e, k)
    P = dh[:, :, None] * dh[:, None, :]                              # (B, 2, 2)
    K = kt[:, None, None] * P + (T / L)[:, None, None] * (np.eye(2)[None] - P)
    # 4x4 block per bond over DOFs [2i, 2i+1, 2j, 2j+1]: [[K, -K], [-K, K]]
    blk = np.block([[K, -K], [-K, K]])                               # (B, 4, 4)
    dofs = np.stack([2 * E[:, 0], 2 * E[:, 0] + 1, 2 * E[:, 1], 2 * E[:, 1] + 1], axis=1)  # (B, 4)
    rows = np.repeat(dofs[:, :, None], 4, axis=2)
    cols = np.repeat(dofs[:, None, :], 4, axis=1)
    H = np.zeros((2 * n, 2 * n))
    np.add.at(H, (rows, cols), blk)
    return H


def triangulated_lattice(rows, cols, spacing=1.0):
    """Rectangular grid of nodes with horizontal, vertical, and one diagonal per cell -> rigid."""
    idx = lambda r, c: r * cols + c
    x0 = np.array([[c * spacing, r * spacing] for r in range(rows) for c in range(cols)], float)
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
            if r + 1 < rows and c + 1 < cols:
                edges.append((idx(r, c), idx(r + 1, c + 1)))
    return x0, edges
