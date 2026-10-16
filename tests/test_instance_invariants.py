"""Tests for the instance invariants and the independent feasibility re-check."""

from __future__ import annotations

import pytest

from bapbench.instances import Berth, Instance, Schedule, Vessel


def _vessels(n: int, **kw) -> list[Vessel]:
    return [Vessel(vid=i, name=f"V{i+1:02d}", length=100, arrive=0, handle=2, **kw)
            for i in range(n)]


def _berths(m: int) -> list[Berth]:
    return [Berth(bid=k, name=f"B{k+1}", length=200) for k in range(m)]


# --- the vid == index invariant ------------------------------------------------

def test_a_vessel_id_that_is_not_its_index_is_rejected():
    """Every accessor in the package uses vessels[vid]; a mismatch would surface
    as a KeyError deep inside a solve, or as the wrong vessel being scored."""
    vessels = _vessels(2)
    vessels[1].vid = 7
    with pytest.raises(ValueError, match="vid == index"):
        Instance(name="bad", vessels=vessels, berths=_berths(2))


def test_a_berth_id_that_is_not_its_index_is_rejected():
    berths = _berths(2)
    berths[1].bid = 5
    with pytest.raises(ValueError, match="bid == index"):
        Instance(name="bad", vessels=_vessels(2), berths=berths)


def test_well_numbered_instances_are_accepted():
    inst = Instance(name="ok", vessels=_vessels(3), berths=_berths(2))
    assert inst.n == 3 and inst.m == 2


# --- completeness of a schedule ------------------------------------------------

def test_an_empty_assignment_is_not_feasible():
    """Before the coverage check an empty assignment passed every other test
    vacuously, because each one iterates over the assignment."""
    inst = Instance(name="cov", vessels=_vessels(3), berths=_berths(2), horizon=20)
    sch = Schedule(assign={}).check(inst)
    assert not sch.feasible
    assert any("unassigned" in v for v in sch.violations)


def test_a_partially_assigned_schedule_is_not_feasible():
    inst = Instance(name="cov2", vessels=_vessels(3), berths=_berths(2), horizon=20)
    sch = Schedule(assign={0: (0, 0)}).check(inst)
    assert not sch.feasible
    assert any("unassigned" in v for v in sch.violations)


def test_completion_reports_missing_vessels_by_name():
    inst = Instance(name="cov3", vessels=_vessels(3), berths=_berths(2), horizon=20)
    with pytest.raises(ValueError, match="unassigned"):
        Schedule(assign={0: (0, 0)}).completion(inst)


def test_a_complete_valid_assignment_is_feasible():
    inst = Instance(name="cov4", vessels=_vessels(3), berths=_berths(2), horizon=20)
    sch = Schedule(assign={i: (0, i * 2) for i in range(3)}).check(inst)
    assert sch.feasible, sch.violations


def test_an_assignment_naming_an_unknown_vessel_is_rejected():
    inst = Instance(name="cov5", vessels=_vessels(2), berths=_berths(2), horizon=20)
    sch = Schedule(assign={0: (0, 0), 1: (0, 2), 9: (0, 4)}).check(inst)
    assert not sch.feasible
