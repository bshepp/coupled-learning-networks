import numpy as np
import pytest
from solver import relax, assemble_laplacian


def test_equal_series_divider_midpoint():
    # node0=1.0 --g=1-- node1(float) --g=1-- node2=0.0  => V1 = 0.5
    V = relax(3, [(0, 1, 1.0), (1, 2, 1.0)], {0: 1.0, 2: 0.0})
    assert V.shape == (1, 3)
    assert V[0, 1] == pytest.approx(0.5, abs=1e-9)


def test_unequal_divider():
    # g01=2, g12=1 => V1 = g01/(g01+g12) = 2/3
    V = relax(3, [(0, 1, 2.0), (1, 2, 1.0)], {0: 1.0, 2: 0.0})
    assert V[0, 1] == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_batched_patterns():
    V = relax(3, [(0, 1, 1.0), (1, 2, 1.0)], {0: np.array([1.0, 2.0]), 2: 0.0})
    assert V.shape == (2, 3)
    assert V[:, 1] == pytest.approx([0.5, 1.0], abs=1e-9)


def test_all_clamped_returns_clamps():
    # no floating nodes -> returns the clamp values unchanged
    V = relax(2, [(0, 1, 1.0)], {0: 1.0, 1: 0.25})
    assert V[0, 0] == pytest.approx(1.0)
    assert V[0, 1] == pytest.approx(0.25)


def test_laplacian_symmetric():
    V = np.zeros((1, 3))
    L = assemble_laplacian([(0, 1, 1.5), (1, 2, 0.7)], V, lambda u, w: np.full(u.shape, w))
    assert np.allclose(L[0], L[0].T)
