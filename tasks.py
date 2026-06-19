"""Friendly, representable tasks for Experiment A (allostery + regression), the
normalized error metric, and the pre-registered verdict logic. Targets are chosen
inside the passive network's achievable range (the maximum principle: output is a
sub-convex combination of clamped inputs and ground)."""
import numpy as np
from dataclasses import dataclass
from network import grid_network, Network


@dataclass
class Task:
    name: str
    net: Network
    inputs: np.ndarray     # (P, n_in)
    targets: np.ndarray    # (P, n_out)


def task_error(outputs, targets):
    """Normalized MSE: mean((out - tgt)^2) / (mean(tgt^2) + eps)."""
    outputs = np.asarray(outputs, float)
    targets = np.asarray(targets, float)
    return float(np.mean((outputs - targets) ** 2) / (np.mean(targets ** 2) + 1e-12))


def regression_task(rows=3, cols=3, n_in=2, n_out=1, n_samples=8, weights=(0.4, 0.3), seed=0):
    """Map input-node voltages to an output voltage via a positive linear map whose
    weights sum to < 1 (so targets stay inside the achievable range)."""
    net = grid_network(rows, cols, n_in=n_in, n_out=n_out)
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.2, 1.0, size=(n_samples, n_in))
    w = np.asarray(weights, float)[:n_in]
    Y = (X @ w).reshape(-1, n_out)
    return Task("regression", net, X, Y)


def allostery_task(rows=3, cols=3, ratio=0.5, n_samples=6, seed=1):
    """Train so a single source drive produces target = ratio * source at a distant node."""
    net = grid_network(rows, cols, n_in=1, n_out=1)
    rng = np.random.default_rng(seed)
    V_src = rng.uniform(0.3, 1.0, size=(n_samples, 1))
    Y = ratio * V_src
    return Task("allostery", net, V_src, Y)


def verdict(local_error, baseline_error, sign_match, rho, tol=0.15, sm_thresh=0.95, rho_thresh=0.9):
    """PASS: local rule reaches the offline baseline AND its update aligns with the true
    gradient. FORECLOSE: local rule cannot reach the baseline. AMBIGUOUS: otherwise."""
    learns = local_error <= baseline_error * (1.0 + tol)
    aligned = (sign_match >= sm_thresh) and (rho >= rho_thresh)
    if learns and aligned:
        status = "PASS"
    elif not learns:
        status = "FORECLOSE"
    else:
        status = "AMBIGUOUS"
    return {"status": status, "local_error": float(local_error),
            "baseline_error": float(baseline_error), "sign_match": float(sign_match),
            "rho": float(rho)}
