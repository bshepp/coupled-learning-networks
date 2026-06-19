"""The audit hook: makes "who computes the update?" a logged, checkable fact.
Every weight change is tagged at its source as physics-local or offline-gradient."""

PHYSICS_LOCAL = "physics-local"
OFFLINE_GRADIENT = "offline-gradient"


class AuditLog:
    def __init__(self):
        self.records = []

    def record(self, provenance, n=1):
        self.records.append((provenance, int(n)))

    def summary(self):
        counts = {}
        for prov, n in self.records:
            counts[prov] = counts.get(prov, 0) + n
        total = sum(counts.values())
        frac = counts.get(PHYSICS_LOCAL, 0) / total if total else 0.0
        return {"counts": counts, "total": total, "fraction_physics_local": frac}

    def assert_physics_local(self):
        s = self.summary()
        if s["counts"].get(OFFLINE_GRADIENT, 0) > 0:
            raise AssertionError(f"headline run used offline-gradient updates: {s}")
        return True
