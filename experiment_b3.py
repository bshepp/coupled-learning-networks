# experiment_b3.py
"""Experiment B3: parity-3 on the elastic lattice vs deformation, with multistability MEASURED.

Experiment B's June headline inferred multistability from solver crashes that turned out to be cold-start
artifacts (2026-11-06 addendum). B3 asks the question properly: train the local two-phase rule on parity-3
at increasing deformation d (where B2 shows the parity channel becoming large), then VALIDATE the learned
configuration by multi-start minimization (`settle()` from the reference geometry and perturbed starts) --
counting distinct energy minima per pattern, re-evaluating the separation at each pattern's lowest-energy
minimum, and measuring the smallest tangent-stiffness eigenvalue. A trained run's own readouts come from
whichever basin its warm-started trajectory sits in; a separation that exists only there is basin
selection (hysteresis), not a property of the lattice. Verdict per d: `tasks.verdict_b3`
(PASS-ROBUST / MULTISTABLE-HYSTERETIC / FORECLOSE-REPRESENTABILITY / AMBIGUOUS). Headline training is
PHYSICS_LOCAL; the validation is a measurement, not an update."""
import json
import time
import numpy as np
from spring_network import default_rest_lengths, tangent_stiffness
from mechanical_solver import settle
from mechanical_learning import train_mech, readout, _input_clamp
from tasks import parity3_mech_task, two_sided_separation, verdict_b3
from audit import AuditLog

DEFAULT_CONFIG = {"rows": 4, "cols": 4, "ds": [0.1, 0.2, 0.3, 0.4, 0.5], "seeds": [0, 1, 2],
                  "epochs": 400, "beta": 0.01, "eta_k": 1.0, "eta_l": 1.0, "floor": 1e-3,
                  "n_starts": 6, "start_scale": 0.1, "energy_tol": 1e-5}


def validate(net, k, l, patterns, d, n_starts=6, start_scale=0.1, energy_tol=1e-5, seed=0):
    """Multi-start minimization per pattern: distinct minima (by energy), lowest-energy readout and its
    smallest eigenvalue. Returns per-pattern dicts."""
    n = net.x0.shape[0]
    rng = np.random.default_rng(seed)
    out = []
    for p in range(len(patterns)):
        clamp = _input_clamp(net, patterns[p], d)
        free = [q for q in range(2 * n) if q not in clamp]
        starts = [net.x0] + [net.x0 + start_scale * rng.standard_normal(net.x0.shape) for _ in range(n_starts)]
        found = []
        for x0 in starts:
            try:
                x, info = settle(net, k, l, clamp, x_init=x0)
                found.append((info["energy"], x))
            except RuntimeError:
                continue
        if not found:
            out.append({"n_minima": 0, "readout_lowest": float("nan"), "min_eig": float("nan"), "n_converged": 0})
            continue
        energies = sorted(E for E, _ in found)
        distinct = 1 + sum(1 for a, b in zip(energies, energies[1:]) if b - a > energy_tol)
        E_best, x_best = min(found, key=lambda t: t[0])
        eig = float(np.linalg.eigvalsh(tangent_stiffness(net, x_best, k, l)[np.ix_(free, free)])[0])
        out.append({"n_minima": distinct, "readout_lowest": readout(net, x_best), "min_eig": eig,
                    "n_converged": len(found), "energy_lowest": float(E_best)})
    return out


def run(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    t0 = time.time()
    audit = AuditLog()
    rows = []
    for d in cfg["ds"]:
        for seed in cfg["seeds"]:
            task = parity3_mech_task(cfg["rows"], cfg["cols"], d=d, seed=seed)
            net, S, y = task["net"], task["patterns"], task["labels"]
            rng = np.random.default_rng(seed)
            k0 = rng.uniform(0.5, 2.0, len(net.edges))
            l0 = default_rest_lengths(net) * rng.uniform(0.9, 1.1, len(net.edges))
            local = train_mech(task, k0, l0, cfg["beta"], cfg["eta_k"], cfg["eta_l"], cfg["epochs"], audit=audit)
            val = validate(net, local["k"], local["l"], S, d, cfg["n_starts"], cfg["start_scale"],
                           cfg["energy_tol"], seed=seed)
            r_low = np.array([v["readout_lowest"] for v in val])
            sep_low = two_sided_separation(r_low, y) if np.all(np.isfinite(r_low)) else float("-inf")
            rows.append({"d": d, "seed": seed,
                         "sep_traj": local["sep_history"][-1], "sep_lowest": sep_low,
                         "snr_lowest": sep_low / cfg["floor"],
                         "frac_multi": float(np.mean([v["n_minima"] >= 2 for v in val])),
                         "n_minima": [v["n_minima"] for v in val],
                         "min_eig": float(np.nanmin([v["min_eig"] for v in val])) if val else float("nan"),
                         "min_eig_train": min(local["min_eig_history"]),
                         "rescues": local["rescues"], "jump_rate": local["jump_rate"],
                         "readout_abs_max": float(np.nanmax(np.abs(r_low))) if len(r_low) else float("nan"),
                         "readouts_traj": local["readouts"].tolist(), "readouts_lowest": r_low.tolist()})
    out = {"config": cfg, "rows": rows, "verdict": verdict_b3(rows),
           "audit_fraction_physics_local": audit.summary()["fraction_physics_local"],
           "elapsed_s": time.time() - t0}
    return out


def main():
    res = run()
    print(f"{'d':>5}{'seed':>5}{'sep_traj':>11}{'sep_lowest':>12}{'multi':>7}{'n_minima':>22}{'minEig':>9}{'|r|max':>8}{'resc':>6}")
    for r in res["rows"]:
        print(f"{r['d']:>5.2f}{r['seed']:>5}{r['sep_traj']:>+11.4f}{r['sep_lowest']:>+12.4f}{r['frac_multi']:>7.2f}"
              f"{str(r['n_minima']):>22}{r['min_eig']:>+9.4f}{r['readout_abs_max']:>8.2f}{r['rescues']:>6}")
    print("\nper-d verdict:")
    for d, v in res["verdict"].items():
        print(f"  d={d}: {v['status']}  (multi {v['frac_multi_mean']:.2f}, sep_traj min {v['sep_traj_min']:+.3f}, "
              f"sep_lowest max {v['sep_lowest_max']:+.3f})")
    print(f"\nphysics-local = {res['audit_fraction_physics_local']:.2f}  ({res['elapsed_s']:.0f}s)")
    with open("experiment_b3_results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("wrote experiment_b3_results.json")


if __name__ == "__main__":
    main()
