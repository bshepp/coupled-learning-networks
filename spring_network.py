# spring_network.py
"""2D nonlinear spring network: positions, the smoothed-asymmetry constitutive law, and
energy/force/tangent-stiffness. Bond energy E = 0.5*k*m(e)*e^2 with e = L - l, L = |xi - xj|.
m(e) is a smooth tension-compression multiplier (m==1 when alpha==1). The geometric
nonlinearity (through L and the rotating bond direction) is what makes the response non-monotone."""
import numpy as np
from dataclasses import dataclass
from typing import List


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
    edges: List[tuple]          # list of (i, j)
    pinned_dofs: List[int]      # flat DOF indices held at x0
    input_dofs: List[int]       # flat DOF indices driven by inputs
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
