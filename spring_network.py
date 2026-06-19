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


def default_rest_lengths(net):
    x = net.x0
    return np.array([np.linalg.norm(x[i] - x[j]) for (i, j) in net.edges], float)


def bond_extensions(net, x, l):
    x = np.asarray(x, float)
    L = np.array([np.linalg.norm(x[i] - x[j]) for (i, j) in net.edges], float)
    return L - np.asarray(l, float)


def energy(net, x, k, l):
    x = np.asarray(x, float)
    E = 0.0
    for b, (i, j) in enumerate(net.edges):
        L = np.linalg.norm(x[i] - x[j]); e = L - l[b]
        E += 0.5 * k[b] * float(m(e, net.alpha, net.w)) * e * e
    return float(E)


def forces(net, x, k, l):
    """F = -dE/dx, shape (N, 2). Per bond the scalar tension T = dE/dL = 0.5*k*m'(e)*e^2 + k*m(e)*e."""
    x = np.asarray(x, float)
    F = np.zeros_like(x)
    for b, (i, j) in enumerate(net.edges):
        d = x[i] - x[j]; L = np.linalg.norm(d); dh = d / L; e = L - l[b]
        T = 0.5 * k[b] * float(dm(e, net.alpha, net.w)) * e * e + k[b] * float(m(e, net.alpha, net.w)) * e
        F[i] -= T * dh; F[j] += T * dh
    return F


def tangent_stiffness(net, x, k, l):
    """Hessian d^2 E / dx^2, shape (2N, 2N). Per bond: material term kt*(dh dh^T) plus geometric
    term (T/L)*(I - dh dh^T), assembled as [[K,-K],[-K,K]] over the bond's two nodes.
    kt = d^2 E/dL^2 = 0.5*k*(m''(e)*e^2 + 4*m'(e)*e + 2*m(e));  T = dE/dL."""
    x = np.asarray(x, float)
    n = x.shape[0]
    H = np.zeros((2 * n, 2 * n))
    I2 = np.eye(2)
    for b, (i, j) in enumerate(net.edges):
        d = x[i] - x[j]; L = np.linalg.norm(d); dh = d / L; e = L - l[b]
        me = float(m(e, net.alpha, net.w)); m1 = float(dm(e, net.alpha, net.w)); m2 = float(d2m(e, net.alpha, net.w))
        T = 0.5 * k[b] * m1 * e * e + k[b] * me * e
        kt = 0.5 * k[b] * (m2 * e * e + 4.0 * m1 * e + 2.0 * me)
        P = np.outer(dh, dh)
        K = kt * P + (T / L) * (I2 - P)
        for (a, sa) in ((i, +1), (j, -1)):
            for (c, sc) in ((i, +1), (j, -1)):
                H[2 * a:2 * a + 2, 2 * c:2 * c + 2] += (sa * sc) * K
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
