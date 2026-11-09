# experiment_b0_noise.py
"""Read noise in the elastic two-read rule (commons hypothesis H5, first data point on a mechanical
substrate). The local update differences two physical reads of each bond's extension (free, nudged)
and divides by beta; independent read noise therefore enters as ~ sigma/beta while a common-mode read
error cancels in the contrast (exactly, for the rest-length update of symmetric springs). 07 found the
flow/resistor rule read-noise-FRAGILE by this mechanism; 05's chemical rule was tolerant. This driver
sweeps sigma (lattice-spacing units) x {independent, common} x beta on Experiment B0's amplify task, with
the sigma = 0 run as the positive control, and records where the trained error crosses the wall
criterion. The verdict (`tasks.verdict_b0_noise`) is about the MECHANISM -- no hardware-realistic sigma
exists for this medium yet; the signal scale |e_nudged - e_free| is reported so a future device spec
can be placed on the axis. Every update is PHYSICS_LOCAL (noise is in the reads, not in who computes)."""
import json
import time
import numpy as np
from spring_network import default_rest_lengths
from tasks import allostery_mech_task, verdict_b0_noise, task_error
from mechanical_learning import train_mech, free_phase_mech, nudged_phase_mech, bond_extensions, readout, task_targets
from audit import AuditLog
from experiment_b0 import untrained_ratio, _init

DEFAULT_CONFIG = {"rows": 4, "cols": 4, "d_max": 0.1, "n_samples": 6, "factor": 2.0,
                  "sigmas": [0.0, 1e-5, 1e-4, 1e-3, 1e-2], "modes": ["independent", "common"],
                  "betas": [0.05, 0.2], "eta": 1.0, "epochs": 2000, "seeds": [0, 1],
                  "wall_factor": 2.0}        # wall: error > wall_factor * sigma=0 error (same beta, seed)


def contrast_scale(task, k, l, beta):
    """Typical |e_nudged - e_free| per bond: the signal the read noise competes with."""
    net, u, d = task["net"], task["patterns"], task["d"]
    targets = task_targets(task)
    vals = []
    for p in range(len(u)):
        xf, _ = free_phase_mech(net, k, l, u[p], d)
        xn, _, _ = nudged_phase_mech(net, k, l, u[p], d, targets[p], beta, xf)
        vals.append(np.abs(bond_extensions(net, xn, l) - bond_extensions(net, xf, l)))
    return float(np.median(np.concatenate(vals))), float(np.max(np.concatenate(vals)))


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
    cells = []
    scales = {}
    for seed in cfg["seeds"]:
        k0, l0 = _init(task["net"], seed)
        untrained = task_error(untrained_ratio(task, k0, l0)[1], task_targets(task))
        for beta in cfg["betas"]:
            scales[f"{seed}|{beta}"] = contrast_scale(task, k0, l0, beta)
            for sigma in cfg["sigmas"]:
                for mode in (cfg["modes"] if sigma > 0 else ["none"]):
                    rn = None if sigma == 0 else {"sigma": sigma, "mode": mode,
                                                  "rng": np.random.default_rng(1000 * seed + int(-np.log10(sigma)))}
                    res = train_mech(task, k0, l0, beta, cfg["eta"], cfg["eta"], cfg["epochs"],
                                     audit=audit, read_noise=rn)
                    cells.append({"seed": seed, "beta": beta, "sigma": sigma, "mode": mode,
                                  "untrained_error": untrained, "final_error": res["final_error"],
                                  "best_error": res["best_error"], "jump_rate": res["jump_rate"],
                                  "jump_rate_tail": res["jump_rate_tail"]})
    # positive control + walls
    controls = {(c["seed"], c["beta"]): c for c in cells if c["sigma"] == 0}
    control_ok = all(c["final_error"] < 0.5 * c["untrained_error"] and c["jump_rate_tail"] <= 0.2
                     for c in controls.values())
    walls = {}
    for mode in cfg["modes"]:
        for beta in cfg["betas"]:
            # wall = smallest sigma at which the MEDIAN over seeds of error/control-error exceeds wall_factor
            wall = float("inf")
            for sigma in sorted(s for s in cfg["sigmas"] if s > 0):
                ratios = [c["final_error"] / controls[(c["seed"], beta)]["final_error"]
                          for c in cells if c["sigma"] == sigma and c["mode"] == mode and c["beta"] == beta]
                if ratios and np.median(ratios) > cfg["wall_factor"]:
                    wall = sigma; break
            walls[(mode, beta)] = wall
    out = {"config": cfg, "untrained_ratio": r0, "ratio": cfg["factor"] * r0, "cells": cells,
           "contrast_scale_median_max": scales, "controls": {f"{s}|{b}": c["final_error"] for (s, b), c in controls.items()},
           "verdict": verdict_b0_noise(walls, control_ok, ),
           "audit_fraction_physics_local": audit.summary()["fraction_physics_local"],
           "elapsed_s": time.time() - t0}
    return out


def main():
    res = run()
    cfg = res["config"]
    print(f"amplify task, ratio {res['ratio']:.4f}; epochs {cfg['epochs']}, eta {cfg['eta']}")
    for key, (med, mx) in res["contrast_scale_median_max"].items():
        print(f"  signal |e_n - e_f| at init (seed|beta {key}): median {med:.2e}, max {mx:.2e}")
    print(f"\n{'seed':>5}{'beta':>6}{'sigma':>9}{'mode':>13}{'final':>11}{'best':>11}{'tail':>6}{'x ctrl':>8}")
    for c in res["cells"]:
        ctrl = res["controls"][f"{c['seed']}|{c['beta']}"]
        print(f"{c['seed']:>5}{c['beta']:>6.2f}{c['sigma']:>9.0e}{c['mode']:>13}{c['final_error']:>11.3e}"
              f"{c['best_error']:>11.3e}{c['jump_rate_tail']:>6.2f}{c['final_error']/ctrl:>8.2f}")
    v = res["verdict"]
    print("\nwalls (smallest sigma with median error > %.1fx control):" % cfg["wall_factor"])
    for key, w in v["walls"].items():
        print(f"  {key:>18}: {w:.0e}" if np.isfinite(w) else f"  {key:>18}: none within sweep")
    print(f"\nVERDICT: {v['status']}  (control_ok={v['control_ok']}, physics-local={res['audit_fraction_physics_local']:.2f}, "
          f"{res['elapsed_s']:.0f}s)")
    with open("experiment_b0_noise_results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("wrote experiment_b0_noise_results.json")


if __name__ == "__main__":
    main()
