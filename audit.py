"""The audit hook: makes "who computes the update?" a logged, checkable fact.

Every weight change is tagged at its source. The portfolio commons (lessons-ledger#audit-is-two-axis)
records provenance on TWO axes -- (gradient / error source) x (weight-write mechanism) -- because a
physical gradient written by an electronic controller is fast inference, not a co-located learner.
This project's two labels project onto that grid:

    PHYSICS_LOCAL     -> (physical, physical)      the medium computes AND writes the update
    OFFLINE_GRADIENT  -> (cpu, electronic)         a CPU optimizer computes it (baselines only)

The one-axis API (record(provenance, n), summary()["fraction_physics_local"]) is unchanged -- the smoke
tests assert fraction_physics_local == 1.0 on every headline run -- and summary() now also reports the
two-axis breakdown and the both-axes PASS gate `is_physical`."""

PHYSICS_LOCAL = "physics-local"
OFFLINE_GRADIENT = "offline-gradient"

# canonical two-axis poles (vocabulary shared with commons/audit.py)
PHYSICAL = "physical"
CPU = "cpu"
ELECTRONIC = "electronic"

TWO_AXIS = {
    PHYSICS_LOCAL: (PHYSICAL, PHYSICAL),
    OFFLINE_GRADIENT: (CPU, ELECTRONIC),
}


class AuditLog:
    def __init__(self):
        self.records = []

    def record(self, provenance, n=1):
        if provenance not in TWO_AXIS:
            raise ValueError(f"bad provenance {provenance!r}; allowed {sorted(TWO_AXIS)}")
        self.records.append((provenance, int(n)))

    def summary(self):
        counts = {}
        for prov, n in self.records:
            counts[prov] = counts.get(prov, 0) + n
        total = sum(counts.values())
        frac = counts.get(PHYSICS_LOCAL, 0) / total if total else 0.0
        two_axis = {}
        for prov, n in counts.items():
            g, w = TWO_AXIS[prov]
            two_axis[f"{g}|{w}"] = two_axis.get(f"{g}|{w}", 0) + n
        return {"counts": counts, "total": total, "fraction_physics_local": frac,
                "two_axis": two_axis, "is_physical": self.is_physical()}

    def is_physical(self):
        """The both-axes PASS gate: every recorded update is physical in gradient source AND write."""
        if not self.records:
            return False
        return all(TWO_AXIS[prov] == (PHYSICAL, PHYSICAL) for prov, _ in self.records)

    def assert_physics_local(self):
        s = self.summary()
        if s["counts"].get(OFFLINE_GRADIENT, 0) > 0:
            raise AssertionError(f"headline run used offline-gradient updates: {s}")
        return True
