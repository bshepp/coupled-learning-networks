import numpy as np
import pytest
from network import RESISTOR, poiseuille_law, Network, grid_network, conductance_edges


def test_resistor_law_identity():
    p = np.array([0.5, 2.0])
    assert RESISTOR.g_of_p(p) == pytest.approx(p)
    assert RESISTOR.dg_dp(p) == pytest.approx([1.0, 1.0])


def test_poiseuille_law_and_derivative():
    law = poiseuille_law(C=2.0)
    p = np.array([1.0, 1.5])
    assert law.g_of_p(p) == pytest.approx(2.0 * p ** 4)
    # finite-difference check of dg_dp
    eps = 1e-6
    fd = (law.g_of_p(p + eps) - law.g_of_p(p - eps)) / (2 * eps)
    assert law.dg_dp(p) == pytest.approx(fd, rel=1e-4)


def test_grid_network_topology():
    net = grid_network(3, 3, n_in=2, n_out=1)
    assert net.n_nodes == 9
    assert len(net.edges) == 12          # 3*2 horizontal + 2*3 vertical
    assert len(net.inputs) == 2 and len(net.outputs) == 1
    # inputs/outputs/ground all distinct
    special = net.inputs + net.outputs + [net.ground]
    assert len(set(special)) == len(special)


def test_conductance_edges_shapes():
    net = grid_network(2, 2)
    g = np.full(len(net.edges), 0.7)
    ce = conductance_edges(net, g)
    assert len(ce) == len(net.edges)
    a, b, w = ce[0]
    assert (a, b) == net.edges[0] and w == pytest.approx(0.7)
