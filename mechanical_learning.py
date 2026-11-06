# mechanical_learning.py
"""The two-phase mechanical coupled-learning rule (the heart of Experiment B). Free phase clamps
the input DOFs; the nudged phase additionally clamps the output readout DOF a small step toward the
target and is warm-started from the free equilibrium. Each bond updates its stiffness k and rest
length l from the LOCAL extension contrast between the two relaxed states. No global signal touches
a weight."""
import numpy as np
from spring_network import bond_extensions, m, dm
from mechanical_solver import relax_mech, basin_jump
from tasks import two_sided_separation, task_error
from audit import PHYSICS_LOCAL


def task_targets(task):
    """Per-pattern targets: explicit task["targets"] (continuous tasks, e.g. allostery) or the two
    class targets selected by task["labels"] (classification tasks, e.g. parity)."""
    if task.get("targets") is not None:
        return np.asarray(task["targets"], float)
    return np.where(np.asarray(task["labels"]) > 0, task["target_pos"], task["target_neg"])


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
    x, info = relax_mech(net, k, l, clamp, x_init=x_free)
    return x, info, r_free


def read_bond_extensions(net, x_free, x_nudged, l, read_noise=None):
    """The two physical READS of each bond's extension that the local rule differences. read_noise =
    {"sigma": s, "mode": "independent" | "common", "rng": Generator} adds Gaussian read noise of std s
    (lattice-spacing units) to the reads: "independent" draws fresh noise for each read (the two-read
    fragility mechanism, commons H5), "common" applies the same realization to both (a per-bond gauge
    offset that the contrast cancels). The relaxations themselves stay noiseless."""
    ef = bond_extensions(net, x_free, l)
    en = bond_extensions(net, x_nudged, l)
    if read_noise is not None and read_noise.get("sigma", 0.0) > 0:
        s = float(read_noise["sigma"]); rng = read_noise["rng"]
        xi1 = rng.standard_normal(ef.shape)
        xi2 = rng.standard_normal(en.shape) if read_noise.get("mode", "independent") == "independent" else xi1
        ef = ef + s * xi1; en = en + s * xi2
    return ef, en


def local_update_mech(net, k, l, x_free, x_nudged, beta, eta_k, eta_l, audit=None, read_noise=None):
    ef, en = read_bond_extensions(net, x_free, x_nudged, l, read_noise)
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
               k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0, audit=None, read_noise=None):
    net, patterns, d = task["net"], task["patterns"], task["d"]
    labels = task.get("labels")
    targets = task_targets(task)
    k = np.array(k0, float); l = np.array(l0, float)
    sep_history = []; error_history = []; jump_history = []
    jumps = 0; trials = 0
    readouts = np.zeros(len(patterns))
    for _ in range(epochs):
        dk = np.zeros_like(k); dl = np.zeros_like(l)
        readouts = np.zeros(len(patterns))
        epoch_jumps = 0
        for p in range(len(patterns)):
            try:
                xf, _ = free_phase_mech(net, k, l, patterns[p], d)
                readouts[p] = readout(net, xf)
                xn, _, r_free = nudged_phase_mech(net, k, l, patterns[p], d, targets[p], beta, xf)
                jumped, _ = basin_jump(xf, xn, expected=abs(beta * (targets[p] - r_free)) + 1e-12)
                jumps += int(jumped); epoch_jumps += int(jumped); trials += 1
                ddk, ddl = local_update_mech(net, k, l, xf, xn, beta, eta_k, eta_l, audit=audit,
                                             read_noise=read_noise)
                dk += ddk; dl += ddl
            except RuntimeError:
                # Solver failure = instability event; count it and skip this pattern's update
                jumps += 1; epoch_jumps += 1; trials += 1
        jump_history.append(epoch_jumps)
        k = np.clip(k + dk / len(patterns), k_min, k_max)
        l = np.clip(l + dl / len(patterns), l_min, l_max)
        if labels is not None:
            sep_history.append(two_sided_separation(readouts, labels))
        error_history.append(task_error(readouts, targets))
    P = len(patterns)
    tail = max(1, len(jump_history) // 5)
    first_jump = next((i for i, j in enumerate(jump_history) if j > 0), None)
    return {"k": k, "l": l, "sep_history": sep_history, "error_history": error_history,
            "final_error": error_history[-1] if error_history else float("nan"),
            "best_error": min(error_history) if error_history else float("nan"),
            "best_epoch": int(np.argmin(error_history)) if error_history else None,
            "jump_rate": jumps / max(trials, 1),
            # instability in the LAST 20% of epochs: distinguishes a rule that drives the lattice into
            # buckling late in training (tail rate -> 1) from a transient early jump
            "jump_rate_tail": sum(jump_history[-tail:]) / max(tail * P, 1),
            "first_jump_epoch": first_jump, "readouts": readouts}
