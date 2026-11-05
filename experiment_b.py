# experiment_b.py
"""Driver for Experiment B: train a 2D nonlinear spring network on parity-3 by the local mechanical
coupled-learning rule, compare to the offline adjoint-gradient baseline, measure separation/SNR,
gradient alignment, and the basin-jump rate, across seeds, and emit the five-way verdict. The
PASS/FORECLOSE outcome is genuinely unknown -- this driver computes it, it is not assumed."""
import json
import numpy as np
from spring_network import default_rest_lengths
from tasks import parity3_mech_task, two_sided_separation, verdict_b
from mechanical_learning import (free_phase_mech, nudged_phase_mech, local_update_mech,
                                 train_mech, readout)
from baselines import mech_adjoint_gradient, gd_baseline_mech, gradient_alignment
from audit import AuditLog

DEFAULT_CONFIG = {"rows": 4, "cols": 4, "epochs": 400, "baseline_epochs": 400,
                  "beta": 0.01, "eta_k": 1.0, "eta_l": 1.0, "lr": 1.0,
                  "seeds": [0, 1, 2], "d": 0.1, "floor": 1e-3}


def _alignment(net, k, l, task):
    """Compute gradient alignment between the local rule update and the true gradient.
    Returns a dict with sign_match and rho; on solver failure returns zeros."""
    try:
        p = task["patterns"][0]; tgt = task["target_pos"]
        gk, gl = mech_adjoint_gradient(net, k, l, np.array([p]), task["d"], np.array([tgt]))
        xf, _ = free_phase_mech(net, k, l, p, task["d"])
        xn, _, _ = nudged_phase_mech(net, k, l, p, task["d"], tgt, 0.005, xf)
        dk, dl = local_update_mech(net, k, l, xf, xn, 0.005, eta_k=1.0, eta_l=1.0)
        return gradient_alignment(np.concatenate([dk, dl]), -np.concatenate([gk, gl]))
    except RuntimeError:
        # Solver failure during alignment computation; return neutral zeros
        return {"sign_match": 0.0, "rho": 0.0}


def run(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    audit = AuditLog()
    seeds_out = []
    for seed in cfg["seeds"]:
        try:
            task = parity3_mech_task(cfg["rows"], cfg["cols"], d=cfg["d"], seed=seed)
            net = task["net"]
            rng = np.random.default_rng(seed)
            k0 = rng.uniform(0.5, 2.0, len(net.edges))
            l0 = default_rest_lengths(net) * rng.uniform(0.9, 1.1, len(net.edges))
            # alignment is certified at INIT (commons pattern `alignment-certification`: at a loss
            # minimum both the update and the gradient -> 0, so rho at convergence is meaningless);
            # the trained-weight alignment is still reported as a secondary diagnostic.
            align = _alignment(net, k0, l0, task)
            local = train_mech(task, k0, l0, cfg["beta"], cfg["eta_k"], cfg["eta_l"],
                               cfg["epochs"], audit=audit)
            base = gd_baseline_mech(task, k0, l0, cfg["baseline_epochs"], cfg["lr"])
            align_final = _alignment(net, local["k"], local["l"], task)
            seeds_out.append({"seed": seed,
                              "local_sep": local["sep_history"][-1],
                              "baseline_sep": base["sep"],
                              "jump_rate": local["jump_rate"],
                              "sign_match": align["sign_match"], "rho": align["rho"],
                              "sign_match_final": align_final["sign_match"],
                              "rho_final": align_final["rho"]})
        except RuntimeError as e:
            # Hard failure on one seed: record instability and continue with other seeds
            seeds_out.append({"seed": seed,
                              "local_sep": float("-inf"),
                              "baseline_sep": float("-inf"),
                              "jump_rate": 1.0,
                              "sign_match": 0.0, "rho": 0.0,
                              "error": str(e)})
    # aggregate across seeds: robust PASS requires the WORST seed to clear the bar
    local_sep = min(s["local_sep"] for s in seeds_out)
    baseline_sep = min(s["baseline_sep"] for s in seeds_out)
    jump_rate = max(s["jump_rate"] for s in seeds_out)
    sign_match = min(s["sign_match"] for s in seeds_out)
    rho = min(s["rho"] for s in seeds_out)
    verdict = verdict_b(local_sep, baseline_sep, cfg["floor"], jump_rate, sign_match, rho)
    return {"verdict": verdict, "seeds": seeds_out,
            "audit_fraction_physics_local": audit.summary()["fraction_physics_local"],
            "alignment": {"sign_match": sign_match, "rho": rho, "measured_at": "init"},
            # nudge-strength-is-task-scaled: report the effective learning rate eta/beta
            "effective_lr": {"k": cfg["eta_k"] / cfg["beta"], "l": cfg["eta_l"] / cfg["beta"]},
            "config": cfg}


def main():
    res = run()
    print(f"{'seed':>5}{'local_sep':>12}{'base_sep':>12}{'jump_rate':>10}{'sign@init':>10}{'rho@init':>9}"
          f"{'rho@end':>9}")
    for s in res["seeds"]:
        print(f"{s['seed']:>5}{s['local_sep']:>12.4e}{s['baseline_sep']:>12.4e}"
              f"{s['jump_rate']:>10.2f}{s['sign_match']:>10.2f}{s['rho']:>9.3f}{s.get('rho_final', float('nan')):>9.3f}")
    print(f"\nVERDICT: {res['verdict']['status']}  (SNR={res['verdict']['snr']:.2f}, "
          f"physics-local={res['audit_fraction_physics_local']:.2f}, "
          f"eta/beta={res['effective_lr']['k']:.0f})")
    with open("experiment_b_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote experiment_b_results.json")


if __name__ == "__main__":
    main()
