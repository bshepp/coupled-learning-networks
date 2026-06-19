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
