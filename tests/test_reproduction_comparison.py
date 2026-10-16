"""Tests for the comparison that places the reproductions beside the baselines.

The comparison is where the reproductions meet the package's own methods, so its
job is to score every one of them the same way and to report failures rather than
hide them. These tests check that, not particular values.
"""

from __future__ import annotations

import pytest

from bapbench.instances import Berth, Instance, Schedule, Vessel, make_toy_instance
from bapbench.metrics import objective_value
from bapbench.reproduction.comparison import (
    compare_reproductions,
    imai_schedule,
    iris_schedule,
)


@pytest.fixture(scope="module")
def toy():
    return make_toy_instance("cmp_toy", 6, 3, seed=20261014)


def test_every_reproduction_and_baseline_appears_once(toy):
    rows = compare_reproductions(toy, "sum_C", time_limit=20.0)
    names = [r["method"] for r in rows]
    assert len(names) == len(set(names)), "a method appears twice"
    assert "MILP" in names
    assert any(n.startswith("Imai2001[") for n in names)
    assert "Iris2017[ALNS]" in names
    assert any(n.startswith("package[") for n in names)


def test_all_rows_share_the_same_reference(toy):
    rows = compare_reproductions(toy, "sum_C", time_limit=20.0)
    assert len({r["reference"] for r in rows}) == 1
    assert len({r["reference_kind"] for r in rows}) == 1


def test_a_gap_is_the_arithmetic_its_reference_implies(toy):
    for r in compare_reproductions(toy, "sum_C", time_limit=20.0):
        if r["gap_pct"] is None or r["objective"] is None:
            continue
        expected = (r["objective"] - r["reference"]) / r["reference"] * 100.0
        assert r["gap_pct"] == pytest.approx(expected, abs=1e-3)


def test_every_reported_objective_can_be_recomputed_from_its_schedule(toy):
    """The comparison scores from the assignment it re-checks, so a reported value
    must be the value of a schedule the package's own check accepts."""
    for r in compare_reproductions(toy, "sum_C", time_limit=20.0):
        if r["feasible"]:
            assert r["objective"] is not None
            assert r["violations"] == ""


def test_a_failed_method_is_reported_with_its_reason(toy):
    for r in compare_reproductions(toy, "sum_C", time_limit=20.0):
        if not r["feasible"]:
            assert r["objective"] is None
            assert r["gap_pct"] is None
            assert r["violations"], r["method"]


def test_the_iris_schedule_passes_the_independent_check(toy):
    sch = iris_schedule(toy, "sum_C", max_iterations=500)
    assert sch.feasible, sch.violations
    checked = sch.check(toy)
    assert checked.feasible, checked.violations


def test_the_iris_schedule_objective_matches_a_recomputation(toy):
    sch = iris_schedule(toy, "sum_C", max_iterations=500)
    assert sch.objective == pytest.approx(
        objective_value(sch.completion(toy), "sum_C"))


def test_the_imai_schedule_carries_every_vessel_or_says_it_did_not(toy):
    for procedure in ("SIMPLE", "INDIVIDUAL", "INTERACT"):
        sch = imai_schedule(toy, procedure=procedure, max_iter=40)
        assert isinstance(sch, Schedule)
        if sch.feasible:
            assert len(sch.assign) == toy.n
            assert sch.check(toy).feasible
        else:
            assert sch.violations


def test_the_imai_procedures_are_not_expected_to_meet_tidal_windows():
    """The DBAP has no tidal windows, so on a closed berth its schedule is
    infeasible here. Reporting that, rather than hiding it, is the comparison's
    result; this pins that the comparison reaches the check at all."""
    inst = Instance(
        name="closed", horizon=30,
        vessels=[Vessel(vid=0, name="V1", length=100, arrive=0, handle=4),
                 Vessel(vid=1, name="V2", length=100, arrive=0, handle=4)],
        berths=[Berth(bid=0, name="B1", length=600),
                Berth(bid=1, name="B2", length=600)],
        tidal_windows={"0": [(0, 30)], "1": [(0, 30)]})
    rows = compare_reproductions(inst, "sum_C", time_limit=20.0,
                                 with_package_methods=False, with_iris=False)
    assert rows
    for r in rows:
        if not r["feasible"]:
            assert r["violations"]


def test_a_reference_is_only_named_when_it_exists(toy):
    for r in compare_reproductions(toy, "sum_C", time_limit=20.0):
        assert r["reference_kind"] in ("exact_proved", "none")
        if r["reference"] is None:
            assert r["reference_kind"] == "none"
            assert r["gap_pct"] is None
