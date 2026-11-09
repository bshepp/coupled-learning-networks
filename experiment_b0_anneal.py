# experiment_b0_anneal.py
"""Annealed read noise on the elastic two-read rule (commons `noise-as-resource`, pre-registered follow-up
to the read-noise sweep). The sweep found that at the independent-read wall (sigma = 1e-3 at beta = 0.05,
1e-2 at beta = 0.2) the noisy trajectory's BEST error was ~100x below the clean run's, before drifting away.
Hypothesis: read noise acts as stochastic exploration on a slow gradient flow; annealing it to zero should
keep the gain. Three arms per (beta, seed) on B0's amplify task, same budget: clean (sigma = 0), constant
sigma at that beta's wall, and sigma annealed linearly from the wall to zero over the run. Verdict per beta
by `tasks.verdict_b0_anneal` (NOISE-HELPS / NOISE-NEUTRAL / NOISE-HURTS). Headline audit PHYSICS_LOCAL."""
import json
import time
import numpy as np
from spring_network import default_rest_lengths
from tasks import allostery_mech_task, verdict_b0_anneal
from mechanical_learning import train_mech
from audit import AuditLog
from experiment_b0 import untrained_ratio, _init

DEFAULT_CONFIG = {"rows": 4, "cols": 4, "d_max": 0.1, "n_samples": 6, "factor": 2.0, "eta": 1.0,
                  "epochs": 2000, "seeds": [0, 1, 2],
                  "sigma_by_beta": {"0.05": 1e-3, "0.2": 1e-2}}      # each beta's independent-read wall


def run(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    t0 = time.time()
    audit = AuditLog()
    ref = allostery_mech_task(cfg["rows"], cfg["cols"], d_max=cfg["d_max"], ratio=1.0, n_samples=cfg["n_samples"])
    net = ref["net"]
    r0 = untrained_ratio(ref, np.ones(len(net.edges)), default_rest_lengths(net))[0]
    task = allostery_mech_task(cfg["rows"], cfg["cols"], d_max=cfg["d_max"], ratio=cfg["factor"] * r0,
                               n_samples=cfg["n_samples"])
    cells = []; verdicts = {}
    for beta_s, sigma in cfg["sigma_by_beta"].items():
        beta = float(beta_s)
        errs = {"clean": [], "constant": [], "annealed": []}
        for seed in cfg["seeds"]:
            k0, l0 = _init(task["net"], seed)
            for arm in ("clean", "constant", "annealed"):
                rn = None
                if arm != "clean":
                    rn = {"sigma": sigma, "mode": "independent", "rng": np.random.default_rng(7 * seed + 1)}
                    if arm == "annealed":
                        rn["anneal_epochs"] = cfg["epochs"]
                res = train_mech(task, k0, l0, beta, cfg["eta"], cfg["eta"], cfg["epochs"], audit=audit, read_noise=rn)
                errs[arm].append(res["final_error"])
                cells.append({"beta": beta, "sigma": sigma, "seed": seed, "arm": arm,
                              "final_error": res["final_error"], "best_error": res["best_error"],
                              "best_epoch": res["best_epoch"], "jump_rate_tail": res["jump_rate_tail"],
                              "min_eig_final": res["min_eig_history"][-1]})
        verdicts[beta_s] = verdict_b0_anneal(errs["annealed"], errs["clean"], errs["constant"])
    return {"config": cfg, "ratio": cfg["factor"] * r0, "cells": cells, "verdict": verdicts,
            "audit_fraction_physics_local": audit.summary()["fraction_physics_local"],
            "elapsed_s": time.time() - t0}


def main():
    res = run()
    print(f"amplify task, ratio {res['ratio']:.4f}; epochs {res['config']['epochs']}")
    print(f"{'beta':>6}{'sigma':>8}{'seed':>5}{'arm':>10}{'final':>11}{'best':>11}{'best@':>7}{'tail':>6}{'minEig':>9}")
    for c in res["cells"]:
        print(f"{c['beta']:>6.2f}{c['sigma']:>8.0e}{c['seed']:>5}{c['arm']:>10}{c['final_error']:>11.3e}"
              f"{c['best_error']:>11.3e}{c['best_epoch']:>7}{c['jump_rate_tail']:>6.2f}{c['min_eig_final']:>+9.4f}")
    for beta_s, v in res["verdict"].items():
        print(f"\nbeta {beta_s}: {v['status']}  (median final: annealed {v['median_annealed']:.3e}, "
              f"clean {v['median_clean']:.3e}, constant {v['median_constant']:.3e})")
    print(f"\nphysics-local = {res['audit_fraction_physics_local']:.2f}  ({res['elapsed_s']:.0f}s)")
    with open("experiment_b0_anneal_results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("wrote experiment_b0_anneal_results.json")


if __name__ == "__main__":
    main()
