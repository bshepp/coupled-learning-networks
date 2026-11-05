import numpy as np
import pytest
from parity_analysis import (walsh_basis, walsh_coefficients, parity_purity, parity_purity_grad,
                             spectrum_report, PARITY)
from tasks import parity3_mech_task


def _patterns():
    return parity3_mech_task(3, 3, d=0.1)["patterns"]


def test_basis_is_orthogonal_on_the_full_cube():
    B = walsh_basis(_patterns())
    assert B.T @ B == pytest.approx(8.0 * np.eye(8))


def test_coefficients_recover_a_known_expansion():
    S = _patterns()
    r = 0.3 + 0.5 * S[:, 0] - 0.2 * S[:, 1] * S[:, 2] + 0.7 * S[:, 0] * S[:, 1] * S[:, 2]
    a = walsh_coefficients(r, S)
    assert a == pytest.approx([0.3, 0.5, 0.0, 0.0, 0.0, 0.0, -0.2, 0.7])


def test_purity_is_one_for_pure_parity_and_zero_without_it():
    S = _patterns()
    y = np.prod(S, axis=1)
    assert parity_purity(2.0 + 0.1 * y, S) == pytest.approx(1.0)      # offset + scale are irrelevant
    assert parity_purity(S[:, 0] + 0.5 * S[:, 1], S) == pytest.approx(0.0)


def test_purity_gradient_matches_finite_difference():
    S = _patterns()
    rng = np.random.default_rng(0)
    r = rng.standard_normal(8)
    g = parity_purity_grad(r, S)
    eps = 1e-6
    fd = np.array([(parity_purity(r + eps * np.eye(8)[p], S) - parity_purity(r - eps * np.eye(8)[p], S))
                   / (2 * eps) for p in range(8)])
    assert g == pytest.approx(fd, abs=1e-6)


def test_spectrum_report_fields():
    S = _patterns()
    rep = spectrum_report(S[:, 0] + 0.1 * np.prod(S, axis=1), S)
    assert rep["linear"] == pytest.approx(1.0)
    assert rep["parity"] == pytest.approx(0.1)
    assert rep["parity_to_linear"] == pytest.approx(0.1)
    assert 0.0 < rep["purity"] < 1.0
