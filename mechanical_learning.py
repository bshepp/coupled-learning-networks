# mechanical_learning.py
"""The two-phase mechanical coupled-learning rule (the heart of Experiment B). Free phase clamps
the input DOFs; the nudged phase additionally clamps the output readout DOF a small step toward the
target and is warm-started from the free equilibrium. Each bond updates its stiffness k and rest
length l from the LOCAL extension contrast between the two relaxed states. No global signal touches
a weight."""
import numpy as np
from spring_network import bond_extensions, m, dm
from mechanical_solver import relax_mech, basin_jump
from tasks import two_sided_separation
from audit import PHYSICS_LOCAL


def _input_clamp(net, pattern, d):
    clamp = {dd: net.x0.reshape(-1)[dd] for dd in net.pinned_dofs}
    for bit, dd in zip(pattern, net.input_dofs):
        node, dim = divmod(dd, 2)
        clamp[dd] = net.x0[node, dim] + float(bit) * d
    return clamp


def readout(net, x):
    node, dim = divmod(net.output_dof, 2)
    return float(x[node, dim] - net.x0[node, dim])


def free_phase_mech(net, k, l, pattern, d):
    return relax_mech(net, k, l, _input_clamp(net, pattern, d), x_init=net.x0)


def nudged_phase_mech(net, k, l, pattern, d, target, beta, x_free):
    r_free = readout(net, x_free)
    clamp = _input_clamp(net, pattern, d)
    node, dim = divmod(net.output_dof, 2)
    clamp[net.output_dof] = net.x0[node, dim] + (r_free + beta * (target - r_free))
    try:
        x, info = relax_mech(net, k, l, clamp, x_init=x_free)
    except RuntimeError:
        # Warm-start can fail when x_free is already near convergence for the nudged clamp
        # (line-search precision floor). Fall back to cold-start from reference geometry.
        x, info = relax_mech(net, k, l, clamp, x_init=net.x0)
    return x, info, r_free


def local_update_mech(net, k, l, x_free, x_nudged, beta, eta_k, eta_l, audit=None):
    ef = bond_extensions(net, x_free, l)
    en = bond_extensions(net, x_nudged, l)
    mf = m(ef, net.alpha, net.w); mn = m(en, net.alpha, net.w)
    dk = (eta_k / beta) * (0.5 * mf * ef ** 2 - 0.5 * mn * en ** 2)
    # dE/dl = -k*(0.5*m'(e)*e^2 + m(e)*e); contrast free - nudged
    dEdl_f = -k * (0.5 * dm(ef, net.alpha, net.w) * ef ** 2 + mf * ef)
    dEdl_n = -k * (0.5 * dm(en, net.alpha, net.w) * en ** 2 + mn * en)
    dl = (eta_l / beta) * (dEdl_f - dEdl_n)
    if audit is not None:
        audit.record(PHYSICS_LOCAL, 2 * len(net.edges))   # k and l per bond
    return dk, dl


def train_mech(task, k0, l0, beta, eta_k, eta_l, epochs,
               k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0, audit=None):
    net, patterns, labels, d = task["net"], task["patterns"], task["labels"], task["d"]
    targets = np.where(labels > 0, task["target_pos"], task["target_neg"])
    k = np.array(k0, float); l = np.array(l0, float)
    sep_history = []
    jumps = 0; trials = 0
    readouts = None
    for _ in range(epochs):
        dk = np.zeros_like(k); dl = np.zeros_like(l)
        readouts = np.zeros(len(patterns))
        for p in range(len(patterns)):
            xf, _ = free_phase_mech(net, k, l, patterns[p], d)
            readouts[p] = readout(net, xf)
            xn, _, r_free = nudged_phase_mech(net, k, l, patterns[p], d, targets[p], beta, xf)
            jumped, _ = basin_jump(xf, xn, expected=abs(beta * (targets[p] - r_free)) + 1e-12)
            jumps += int(jumped); trials += 1
            ddk, ddl = local_update_mech(net, k, l, xf, xn, beta, eta_k, eta_l, audit=audit)
            dk += ddk; dl += ddl
        k = np.clip(k + dk / len(patterns), k_min, k_max)
        l = np.clip(l + dl / len(patterns), l_min, l_max)
        sep_history.append(two_sided_separation(readouts, labels))
    return {"k": k, "l": l, "sep_history": sep_history,
            "jump_rate": jumps / max(trials, 1), "readouts": readouts}
