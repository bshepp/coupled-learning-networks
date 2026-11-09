import pytest
from audit import AuditLog, PHYSICS_LOCAL, OFFLINE_GRADIENT


def test_all_physics_local():
    log = AuditLog()
    log.record(PHYSICS_LOCAL, 12)
    log.record(PHYSICS_LOCAL, 12)
    s = log.summary()
    assert s["fraction_physics_local"] == pytest.approx(1.0)
    assert log.assert_physics_local() is True


def test_offline_gradient_flagged():
    log = AuditLog()
    log.record(PHYSICS_LOCAL, 5)
    log.record(OFFLINE_GRADIENT, 5)
    s = log.summary()
    assert s["fraction_physics_local"] == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        log.assert_physics_local()


def test_summary_counts():
    log = AuditLog()
    log.record(PHYSICS_LOCAL, 3)
    log.record(OFFLINE_GRADIENT, 7)
    s = log.summary()
    assert s["counts"][PHYSICS_LOCAL] == 3
    assert s["counts"][OFFLINE_GRADIENT] == 7
    assert s["total"] == 10


def test_two_axis_projection_and_both_axes_gate():
    from audit import AuditLog, PHYSICS_LOCAL, OFFLINE_GRADIENT
    import pytest
    log = AuditLog()
    assert log.is_physical() is False                  # nothing recorded is not a PASS
    log.record(PHYSICS_LOCAL, 3)
    s = log.summary()
    assert s["two_axis"] == {"physical|physical": 3}
    assert s["is_physical"] is True
    log.record(OFFLINE_GRADIENT, 2)
    s = log.summary()
    assert s["two_axis"] == {"physical|physical": 3, "cpu|electronic": 2}
    assert s["is_physical"] is False                   # a CPU gradient + electronic write is not learning
    assert s["fraction_physics_local"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        log.record("mystery")
