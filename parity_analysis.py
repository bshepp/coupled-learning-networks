# parity_analysis.py
"""Walsh (Fourier-on-the-hypercube) decomposition of a 3-input readout. Any function of three
+/-1 bits is exactly r = a0 + sum_i a_i s_i + sum_{i<j} a_ij s_i s_j + a_123 s1 s2 s3. Parity-3 lives
entirely in the last coefficient, so the ratio of a_123 to the other non-constant coefficients is a
graded, scale-free measure of how much "parity" a substrate's response contains -- sharper than the
binary two-sided separation. Used as a diagnostic (how does the parity channel scale with
deformation?) and as the objective of the scale-free representability probe (Experiment B2)."""
import numpy as np

# Walsh basis order: [1, s1, s2, s3, s1s2, s1s3, s2s3, s1s2s3]
NAMES = ("1", "s1", "s2", "s3", "s1s2", "s1s3", "s2s3", "s1s2s3")
PARITY = 7            # index of the parity coefficient
LINEAR = slice(1, 4)
QUADRATIC = slice(4, 7)


def walsh_basis(patterns):
    """(P, 8) matrix of the Walsh basis evaluated at each +/-1 pattern row (P should be 8)."""
    S = np.asarray(patterns, float)
    return np.column_stack([np.ones(len(S)), S[:, 0], S[:, 1], S[:, 2],
                            S[:, 0] * S[:, 1], S[:, 0] * S[:, 2], S[:, 1] * S[:, 2],
                            S[:, 0] * S[:, 1] * S[:, 2]])


def walsh_coefficients(r, patterns):
    """Coefficients a (8,) with r = walsh_basis(patterns) @ a. Exact (the basis is orthogonal) when
    the 8 patterns are the full cube; least-squares otherwise."""
    B = walsh_basis(patterns)
    r = np.asarray(r, float)
    if B.shape[0] == 8 and np.allclose(B.T @ B, 8.0 * np.eye(8)):
        return B.T @ r / 8.0
    return np.linalg.lstsq(B, r, rcond=None)[0]


def parity_purity(r, patterns):
    """a_123^2 / sum_{j>=1} a_j^2 in [0, 1]: the fraction of the readout's non-constant 'energy' in the
    parity channel. 1 iff the response is exactly parity (up to offset and scale); 0 iff no parity."""
    a = walsh_coefficients(r, patterns)
    S = float(np.sum(a[1:] ** 2))
    return float(a[PARITY] ** 2 / S) if S > 0 else 0.0


def parity_purity_grad(r, patterns):
    """d(purity)/dr (P,) -- for driving an optimizer through an adjoint (dL/dr feeds the IFT)."""
    B = walsh_basis(patterns)
    a = B.T @ np.asarray(r, float) / B.shape[0]
    S = float(np.sum(a[1:] ** 2))
    if S <= 0:
        return np.zeros(len(r))
    ap = a[PARITY]
    da = np.zeros(8)
    da[1:] = -2.0 * ap ** 2 * a[1:] / S ** 2
    da[PARITY] += 2.0 * ap / S
    return B @ da / B.shape[0]


def spectrum_report(r, patterns):
    """Magnitudes of the three channel groups + purity; the numbers a findings doc tabulates."""
    a = walsh_coefficients(r, patterns)
    lin = float(np.max(np.abs(a[LINEAR]))); quad = float(np.max(np.abs(a[QUADRATIC])))
    par = float(abs(a[PARITY]))
    return {"linear": lin, "quadratic": quad, "parity": par,
            "parity_to_linear": par / lin if lin > 0 else float("inf"),
            "purity": parity_purity(r, patterns)}
