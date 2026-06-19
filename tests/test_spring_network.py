# tests/test_spring_network.py
import numpy as np
import pytest
from spring_network import (SpringNetwork, m, dm, d2m, dof, default_rest_lengths,
                            bond_extensions, energy, forces)


def _net(alpha=1.0, w=0.2):
    x0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    edges = [(0, 1), (1, 2), (2, 0)]
    return SpringNetwork(x0, edges, pinned_dofs=[0, 1], input_dofs=[2],
                         output_dof=5, alpha=alpha, w=w)


def test_m_symmetric_is_one():
    e = np.array([-1.0, 0.0, 1.0])
    assert m(e, 1.0, 0.2) == pytest.approx([1.0, 1.0, 1.0])
    assert dm(e, 1.0, 0.2) == pytest.approx([0.0, 0.0, 0.0])
    assert d2m(e, 1.0, 0.2) == pytest.approx([0.0, 0.0, 0.0])


def test_m_derivatives_match_fd():
    e = np.array([-0.3, 0.1, 0.5]); alpha, w = 0.5, 0.2
    eps = 1e-6
    fd1 = (m(e + eps, alpha, w) - m(e - eps, alpha, w)) / (2 * eps)
    fd2 = (dm(e + eps, alpha, w) - dm(e - eps, alpha, w)) / (2 * eps)
    assert dm(e, alpha, w) == pytest.approx(fd1, abs=1e-5)
    assert d2m(e, alpha, w) == pytest.approx(fd2, abs=1e-4)


def test_default_rest_lengths_and_extensions():
    net = _net()
    l = default_rest_lengths(net)
    assert l == pytest.approx([1.0, np.hypot(0.5, 0.8), np.hypot(0.5, 0.8)])
    # at reference, extensions are zero
    assert bond_extensions(net, net.x0, l) == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_forces_match_negative_energy_gradient(alpha):
    net = _net(alpha=alpha)
    rng = np.random.default_rng(0)
    x = net.x0 + 0.1 * rng.standard_normal(net.x0.shape)
    k = np.array([1.0, 1.3, 0.8]); l = np.array([0.9, 1.0, 1.0])
    F = forces(net, x, k, l)
    eps = 1e-6
    fd = np.zeros_like(x)
    for i in range(x.shape[0]):
        for d in range(2):
            xp = x.copy(); xp[i, d] += eps
            xm = x.copy(); xm[i, d] -= eps
            fd[i, d] = -(energy(net, xp, k, l) - energy(net, xm, k, l)) / (2 * eps)
    assert F == pytest.approx(fd, abs=1e-5)


def test_tangent_stiffness_matches_fd():
    from spring_network import tangent_stiffness
    net = _net(alpha=0.5)
    rng = np.random.default_rng(1)
    x = net.x0 + 0.1 * rng.standard_normal(net.x0.shape)
    k = np.array([1.0, 1.2, 0.7]); l = np.array([0.9, 1.0, 1.05])
    H = tangent_stiffness(net, x, k, l)
    n = x.shape[0]; eps = 1e-6
    fd = np.zeros((2 * n, 2 * n))
    g = lambda xx: -forces(net, xx, k, l).reshape(-1)   # gradient of E
    for b in range(2 * n):
        xp = x.reshape(-1).copy(); xp[b] += eps
        xm = x.reshape(-1).copy(); xm[b] -= eps
        fd[:, b] = (g(xp.reshape(n, 2)) - g(xm.reshape(n, 2))) / (2 * eps)
    assert H == pytest.approx(fd, abs=1e-4)
    assert H == pytest.approx(H.T, abs=1e-9)          # symmetric


def test_triangulated_lattice_is_rigid():
    from spring_network import triangulated_lattice, tangent_stiffness, default_rest_lengths
    x0, edges = triangulated_lattice(3, 3)
    assert x0.shape == (9, 2)
    net = SpringNetwork(x0, edges, pinned_dofs=[0, 1, dof(2, 1)],   # pin a corner fully + one DOF
                        input_dofs=[], output_dof=0)
    k = np.ones(len(edges)); l = default_rest_lengths(net)
    H = tangent_stiffness(net, x0, k, l)
    free = [d for d in range(2 * 9) if d not in {0, 1, dof(2, 1)}]
    Hff = H[np.ix_(free, free)]
    assert np.linalg.matrix_rank(Hff) == len(free)    # non-singular -> rigid
