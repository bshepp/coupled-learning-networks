# experiment_b0.py
"""Experiment B0: does the mechanical coupled-learning rule LEARN a representable task on the elastic
medium? (The positive control Experiment B never ran: B's keystone certifies the per-step update, and
B's parity task foreclosed, so the local rule's network-scale learning on this substrate was untested --
commons lesson `network-foreclose-needs-a-positive-control`.)

Task: mechanical allostery -- one input node displaced by amplitudes u, prescribed transfer ratio at a
distant output node (target = ratio * u), three ratios relative to the untrained lattice's own ratio r0:
attenuate (0.5 r0), amplify (2 r0), invert (-0.5 r0). Per task and seed: untrained error; init gradient
alignment (commons `alignment-certification`); the converged L-BFGS representability CEILING
(`mech_ceiling`, budget-saturated); the matched-budget GD baseline (same epochs, lr = eta -- the local
rule is GD at step eta in the beta->0 limit); the local two-phase rule (PHYSICS_LOCAL). Verdict per seed
by `tasks.verdict_b0`; per task, PASS only if every seed passes, else the most severe seed status.
A d-map then asks where on the deformation axis the rule works at all (jump rate, error vs d_max)."""
import json
import time
import numpy as np
from spring_network import default_rest_lengths
from tasks import allostery_mech_task, verdict_b0, task_error
from mechanical_learning import (free_phase_mech, nudged_phase_mech, local_update_mech, train_mech,
                                 readout, task_targets)
from baselines import mech_adjoint_gradient, gd_baseline_mech, mech_ceiling, gradient_alignment
from audit import AuditLog

DEFAULT_CONFIG = {"rows": 4, "cols": 4, "d_max": 0.1, "n_samples": 6,
                  "factors": {"attenuate": 0.5, "amplify": 2.0, "invert": -0.5},
                  "seeds": [0, 1, 2], "epochs": 6000, "beta": 0.05, "eta": 1.0,
                  "ceiling_maxiter": 2000,
                  "d_map": [0.05, 0.10, 0.20, 0.30], "d_map_task": "amplify", "d_map_epochs": 1000}
SEVERITY = ["MULTISTABILITY-FORECLOSE", "FORECLOSE-REPRESENTABILITY", "FORECLOSE-LEARNABILITY",
            "AMBIGUOUS", "PASS"]


def untrained_ratio(task, k, l):
    """Least-squares transfer ratio r/u of the lattice (k, l) on the task's amplitudes."""
    net, u = task["net"], task["patterns"]
    r = np.array([readout(net, free_phase_mech(net, k, l, u[p], task["d"])[0]) for p in range(len(u))])
    return float(np.sum(r * u[:, 0]) / np.sum(u[:, 0] ** 2)), r


def _init(net, seed):
    rng = np.random.default_rng(seed)
    k0 = rng.uniform(0.5, 2.0, len(net.edges))
    l0 = default_rest_lengths(net) * rng.uniform(0.9, 1.1, len(net.edges))
    return k0, l0


def _alignment(task, k, l, beta=0.005):
    """Full-batch local update vs -true gradient on the whole task (measured at init)."""
    net, u, d = task["net"], task["patterns"], task["d"]
    targets = task_targets(task)
    try:
        gk, gl = mech_adjoint_gradient(net, k, l, u, d, targets)
        dk = np.zeros_like(k); dl = np.zeros_like(l)
        for p in range(len(u)):
            xf, _ = free_phase_mech(net, k, l, u[p], d)
            xn, _, _ = nudged_phase_mech(net, k, l, u[p], d, targets[p], beta, xf)
            ddk, ddl = local_update_mech(net, k, l, xf, xn, beta, eta_k=1.0, eta_l=1.0)
            dk += ddk; dl += ddl
        return gradient_alignment(np.concatenate([dk, dl]), -np.concatenate([gk, gl]))
    except RuntimeError:
        return {"sign_match": 0.0, "rho": 0.0, "cosine": 0.0}


def _converged(history, tail=0.2, tol=0.02):
    """Relative decrease over the last `tail` fraction of epochs below `tol` -> plateaued."""
    n = len(history); i = int((1 - tail) * n)
    if n < 10 or history[i] <= 0:
        return False
    return (history[i] - history[-1]) / history[i] < tol


def _one_cell(task, seed, cfg, audit):
    net = task["net"]
    k0, l0 = _init(net, seed)
    try:
        _, r_unt = untrained_ratio(task, k0, l0)
    except RuntimeError as e:
        return {"seed": seed, "error": f"init unsolvable: {e}", "jump_rate": 1.0,
                "verdict": verdict_b0(float("inf"), float("inf"), float("inf"), 1.0, 0.0, 0.0,
                                      ceiling_error=float("inf"))}
    untrained_error = task_error(r_unt, task_targets(task))
    align = _alignment(task, k0, l0)
    ceil = mech_ceiling(task, k0, l0, maxiter=cfg["ceiling_maxiter"])
    base = gd_baseline_mech(task, k0, l0, cfg["epochs"], lr=cfg["eta"])
    local = train_mech(task, k0, l0, cfg["beta"], cfg["eta"], cfg["eta"], cfg["epochs"], audit=audit)
    v = verdict_b0(local["final_error"], base["final_error"], untrained_error, local["jump_rate_tail"],
                   align["sign_match"], align["rho"], ceiling_error=ceil["final_error"])
    return {"seed": seed, "untrained_error": untrained_error, "ceiling_error": ceil["final_error"],
            "ceiling_converged": ceil["converged"], "baseline_error": base["final_error"],
            "baseline_fail_epochs": base["fail_epochs"], "baseline_first_fail_epoch": base["first_fail_epoch"],
            "local_error": local["final_error"], "local_best_error": local["best_error"],
            "local_best_epoch": local["best_epoch"], "jump_rate": local["jump_rate"],
            "jump_rate_tail": local["jump_rate_tail"], "first_jump_epoch": local["first_jump_epoch"],
            "local_converged": _converged(local["error_history"]),
            "local_history_decimated": local["error_history"][::max(1, cfg["epochs"] // 30)],
            "sign_match": align["sign_match"], "rho": align["rho"], "verdict": v}


def _task_verdict(cells):
    statuses = [c["verdict"]["status"] for c in cells]
    if all(s == "PASS" for s in statuses):
        return "PASS"
    return min(statuses, key=SEVERITY.index)


def d_map(cfg, factor, r0_fn):
    """Where on the deformation axis does the local rule work? Seed 0, amplify task, per d_max."""
    rows = []
    for d_max in cfg["d_map"]:
        try:
            ref = allostery_mech_task(cfg["rows"], cfg["cols"], d_max=d_max, ratio=1.0, n_samples=cfg["n_samples"])
            r0 = r0_fn(ref)
            task = allostery_mech_task(cfg["rows"], cfg["cols"], d_max=d_max, ratio=factor * r0,
                                       n_samples=cfg["n_samples"])
            k0, l0 = _init(task["net"], 0)
            ceil = mech_ceiling(task, k0, l0, maxiter=cfg["ceiling_maxiter"])
            local = train_mech(task, k0, l0, cfg["beta"], cfg["eta"], cfg["eta"], cfg["d_map_epochs"])
            rows.append({"d_max": d_max, "untrained_ratio": r0, "ceiling_error": ceil["final_error"],
                         "local_error": local["final_error"], "local_best_error": local["best_error"],
                         "jump_rate": local["jump_rate"], "jump_rate_tail": local["jump_rate_tail"],
                         "first_jump_epoch": local["first_jump_epoch"], "stable": True})
        except RuntimeError as e:
            rows.append({"d_max": d_max, "stable": False, "error": str(e)})
    return rows


def run(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    t0 = time.time()
    audit = AuditLog()

    def r0_of(ref_task):
        net = ref_task["net"]
        return untrained_ratio(ref_task, np.ones(len(net.edges)), default_rest_lengths(net))[0]

    ref = allostery_mech_task(cfg["rows"], cfg["cols"], d_max=cfg["d_max"], ratio=1.0, n_samples=cfg["n_samples"])
    r0 = r0_of(ref)
    out = {"config": cfg, "untrained_ratio": r0, "effective_lr": cfg["eta"] / cfg["beta"], "tasks": {}}
    for name, factor in cfg["factors"].items():
        task = allostery_mech_task(cfg["rows"], cfg["cols"], d_max=cfg["d_max"], ratio=factor * r0,
                                   n_samples=cfg["n_samples"])
        cells = [_one_cell(task, seed, cfg, audit) for seed in cfg["seeds"]]
        out["tasks"][name] = {"ratio": factor * r0, "factor": factor, "seeds": cells,
                              "verdict": _task_verdict(cells)}
    if cfg.get("d_map"):
        out["d_map"] = d_map(cfg, cfg["factors"][cfg["d_map_task"]], r0_of)
    out["audit_fraction_physics_local"] = audit.summary()["fraction_physics_local"]
    out["audit"] = audit.summary()
    out["elapsed_s"] = time.time() - t0
    return out


def main():
    res = run()
    print(f"untrained transfer ratio r0 = {res['untrained_ratio']:.4f}   eta/beta = {res['effective_lr']:.0f}")
    print(f"{'task':<10}{'seed':>5}{'untrained':>11}{'ceiling':>11}{'baseline':>11}{'local':>11}{'best':>11}"
          f"{'jump':>6}{'tail':>6}{'1st':>6}{'sign':>6}{'rho':>7}{'conv':>5}  verdict")
    for name, t in res["tasks"].items():
        for c in t["seeds"]:
            if "error" in c:
                print(f"{name:<10}{c['seed']:>5}  {c['error']}"); continue
            fj = c["first_jump_epoch"]
            print(f"{name:<10}{c['seed']:>5}{c['untrained_error']:>11.3e}{c['ceiling_error']:>11.3e}"
                  f"{c['baseline_error']:>11.3e}{c['local_error']:>11.3e}{c['local_best_error']:>11.3e}"
                  f"{c['jump_rate']:>6.2f}{c['jump_rate_tail']:>6.2f}{(fj if fj is not None else '-'):>6}"
                  f"{c['sign_match']:>6.2f}{c['rho']:>7.3f}{'y' if c['local_converged'] else 'n':>5}  "
                  f"{c['verdict']['status']}" + (f"  [GD failed {c['baseline_fail_epochs']} epochs from {c['baseline_first_fail_epoch']}]" if c['baseline_fail_epochs'] else ""))
        print(f"{'':<10}{'':>5}  -> {name} (ratio {t['ratio']:+.4f}): {t['verdict']}")
    if "d_map" in res:
        print(f"\nd-map ({res['config']['d_map_task']}, seed 0, {res['config']['d_map_epochs']} epochs):")
        print(f"{'d_max':>6}{'r0':>8}{'ceiling':>11}{'local':>11}{'best':>11}{'jump':>6}{'tail':>6}{'1st':>6}")
        for r in res["d_map"]:
            if r["stable"]:
                fj = r["first_jump_epoch"]
                print(f"{r['d_max']:>6.2f}{r['untrained_ratio']:>8.4f}{r['ceiling_error']:>11.3e}"
                      f"{r['local_error']:>11.3e}{r['local_best_error']:>11.3e}{r['jump_rate']:>6.2f}"
                      f"{r['jump_rate_tail']:>6.2f}{(fj if fj is not None else '-'):>6}")
            else:
                print(f"{r['d_max']:>6.2f}  UNSTABLE: {r['error'][:60]}")
    print(f"\nphysics-local = {res['audit_fraction_physics_local']:.2f}   ({res['elapsed_s']:.0f}s)")
    with open("experiment_b0_results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("wrote experiment_b0_results.json")


if __name__ == "__main__":
    main()
