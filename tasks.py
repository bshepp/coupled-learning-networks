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


# --- Experiment B: mechanical parity-3 ---------------------------------------
from spring_network import SpringNetwork, triangulated_lattice, default_rest_lengths, dof


def parity3_mech_task(rows=4, cols=4, d=0.3, seed=0):
    """Triangulated lattice; 3 input nodes on the bottom row (driven in y by +/-d), output node
    top-right (read y-displacement), bottom-left corner pinned fully + one more DOF to remove
    rigid-body motion. Targets span the achievable readout range for the nudge."""
    x0, edges = triangulated_lattice(rows, cols)
    in_nodes = [0 * cols + 0, 0 * cols + (cols // 2), 0 * cols + (cols - 1)]  # bottom row
    out_node = (rows - 1) * cols + (cols - 1)                                 # top-right
    pin_node = 0 * cols + 1                                                   # a bottom node not an input
    input_dofs = [dof(nd, 1) for nd in in_nodes]                              # drive in y
    pinned_dofs = [dof(pin_node, 0), dof(pin_node, 1), dof(in_nodes[0], 0)]   # 2 + 1 DOFs
    net = SpringNetwork(x0, edges, pinned_dofs=pinned_dofs, input_dofs=input_dofs,
                        output_dof=dof(out_node, 1))
    patterns = np.array([[1 - 2 * ((i >> b) & 1) for b in range(3)] for i in range(8)], float)
    labels = np.prod(patterns, axis=1)
    return {"net": net, "patterns": patterns, "labels": labels, "d": d,
            "target_pos": +0.5 * d, "target_neg": -0.5 * d}


def allostery_mech_task(rows=4, cols=4, d_max=0.1, ratio=0.5, n_samples=6, seed=0):
    """Mechanical allostery (Rocks et al. 2017 idiom): ONE input node on the bottom row is displaced
    in y by amplitudes u in d_max*U[0.3, 1]; the task is a prescribed transfer ratio at the distant
    output node (top-right, y): target = ratio * u. Patterns are the amplitudes themselves (d = 1), so
    the same train_mech / baseline machinery applies with continuous targets. The bottom-left node is
    pinned fully and the bottom-right node in y (3 rigid-body DOFs removed; the bottom edge is held)."""
    x0, edges = triangulated_lattice(rows, cols)
    in_node = 0 * cols + (cols // 2)
    out_node = (rows - 1) * cols + (cols - 1)
    pinned_dofs = [dof(0, 0), dof(0, 1), dof(cols - 1, 1)]
    net = SpringNetwork(x0, edges, pinned_dofs=pinned_dofs, input_dofs=[dof(in_node, 1)],
                        output_dof=dof(out_node, 1))
    rng = np.random.default_rng(seed)
    u = d_max * rng.uniform(0.3, 1.0, size=(n_samples, 1))
    return {"net": net, "patterns": u, "labels": None, "targets": ratio * u[:, 0], "d": 1.0,
            "d_max": d_max, "ratio": ratio}


def verdict_b0(local_error, baseline_error, untrained_error, jump_rate, sign_match, rho,
               ceiling_error=None, tol=0.15, abs_floor=1e-3, rep_max=0.1, jump_max=0.2, sm_min=0.9,
               rho_min=0.8):
    """Pre-registered verdict for Experiment B0 (does the mechanical local rule LEARN a representable
    task on the elastic medium?). baseline_error = the matched-budget offline GD (same epochs, lr = eta);
    ceiling_error = the converged L-BFGS representability ceiling (defaults to baseline_error).
    jump_rate should be the instability rate over the LAST 20% of training (train_mech's
    jump_rate_tail), so a rule that drives the lattice into buckling late in training is labelled as
    such rather than as a learnability failure. Evaluated in order:
      MULTISTABILITY-FORECLOSE   jump_rate > jump_max (the two-phase rule is ill-defined here)
      FORECLOSE-REPRESENTABILITY the converged ceiling cannot bring normalized MSE below rep_max
      PASS                       local error <= max(baseline*(1+tol), abs_floor) AND init alignment holds
      FORECLOSE-LEARNABILITY     representable, but the local rule does not even halve the untrained error
      AMBIGUOUS                  otherwise (learns but misses the baseline, or weak alignment)."""
    if ceiling_error is None:
        ceiling_error = baseline_error
    out = {"local_error": float(local_error), "baseline_error": float(baseline_error),
           "ceiling_error": float(ceiling_error), "untrained_error": float(untrained_error),
           "jump_rate": float(jump_rate), "sign_match": float(sign_match), "rho": float(rho)}
    learns = local_error <= max(baseline_error * (1.0 + tol), abs_floor)
    aligned = sign_match >= sm_min and rho >= rho_min
    if jump_rate > jump_max:
        out["status"] = "MULTISTABILITY-FORECLOSE"
    elif not (ceiling_error <= rep_max):
        out["status"] = "FORECLOSE-REPRESENTABILITY"
    elif learns and aligned:
        out["status"] = "PASS"
    elif local_error > 0.5 * untrained_error:
        out["status"] = "FORECLOSE-LEARNABILITY"
    else:
        out["status"] = "AMBIGUOUS"
    return out


def two_sided_separation(r, y):
    """Separable in EITHER polarity: max(min(+)-max(-), min(-)-max(+)). >0 iff a threshold splits
    the classes; <=0 iff they overlap. (Ported idiom from self-assembly/stage1_network.py.)"""
    r = np.asarray(r, float); y = np.asarray(y)
    pos, neg = r[y > 0], r[y < 0]
    if pos.size == 0 or neg.size == 0:
        return float("-inf")   # only one class present: two-sided separation is undefined
    return float(max(np.min(pos) - np.max(neg), np.min(neg) - np.max(pos)))


def verdict_b(local_sep, baseline_sep, floor, jump_rate, sign_match, rho,
              k=3.0, jump_max=0.2, sm_min=0.9, rho_min=0.8):
    """Five-way verdict, evaluated in order: multistability gate, then representability,
    learnability, PASS (SNR>=k and aligned and low jumps), else AMBIGUOUS."""
    snr = local_sep / floor if floor > 0 else float("inf")
    base_snr = baseline_sep / floor if floor > 0 else float("inf")
    out = {"local_sep": float(local_sep), "baseline_sep": float(baseline_sep),
           "snr": float(snr), "jump_rate": float(jump_rate),
           "sign_match": float(sign_match), "rho": float(rho)}
    if jump_rate > jump_max:
        out["status"] = "MULTISTABILITY-FORECLOSE"
    elif baseline_sep <= 0 or base_snr < 1.0:
        out["status"] = "FORECLOSE-REPRESENTABILITY"
    elif local_sep <= 0 or snr < 1.0:
        out["status"] = "FORECLOSE-LEARNABILITY"
    elif snr >= k and sign_match >= sm_min and rho >= rho_min:
        out["status"] = "PASS"
    else:
        out["status"] = "AMBIGUOUS"
    return out


def verdict_b2(rows, k=3.0, unstable_max=0.2, sat_tol=0.02):
    """Pre-registered verdict for Experiment B2 (is B's representability FORECLOSE a budget/objective
    artifact?). rows: one dict per deformation d with sep_min / sep_max (two-sided separation at the
    offline optimum, min/max over seeds and optimizers), snr_min (sep_min / floor), unstable_frac
    (fraction of seeds whose lattice cannot even be solved at that d = the buckling regime),
    saturation_gap (purity gain over the last budget doubling, or 0 if the optimizer converged within
    budget; small = the budget is saturated).
      REPRESENTABLE: some STABLE d (unstable_frac <= unstable_max) separates robustly (sep_min > 0,
        snr_min >= k) -> B's representability foreclosure was an artifact; learnability reopens.
      FORECLOSE-REPRESENTABILITY-CONFIRMED: every stable d has sep_max <= 0 with a saturated budget, and
        no d (stable or not) separates.
      AMBIGUOUS: anything in between (marginal / single-seed / unstable-only positives, unsaturated budget,
        or no stable regime probed at all)."""
    stable = [r for r in rows if r["unstable_frac"] <= unstable_max]
    out = {"n_rows": len(rows), "n_stable": len(stable)}
    if any(r["sep_min"] > 0 and r["snr_min"] >= k for r in stable):
        out["status"] = "REPRESENTABLE"
    elif (stable and all(r["sep_max"] <= 0 and r["saturation_gap"] <= sat_tol for r in stable)
          and all(r["sep_max"] <= 0 for r in rows)):
        out["status"] = "FORECLOSE-REPRESENTABILITY-CONFIRMED"
    else:
        out["status"] = "AMBIGUOUS"
    return out
