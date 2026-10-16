"""Tests for solver-status classification and the independent re-check.

Both exist because two classes of defect silently corrupt reported numbers:
a run cut off by a limit being called infeasible (losing a usable schedule and
misnaming the cause), and a feasibility re-check that is never actually invoked.
"""

from __future__ import annotations

import pytest

from bapbench.instances import make_hard_instance, make_toy_instance
from bapbench.metrics import (
    berth_overlaps,
    check_bound_sanity,
    sanity_check_schedule,
)
from bapbench.models.bap import LIMIT_STATUSES, BAPTimeIndexed


@pytest.fixture(scope="module")
def toy():
    return make_toy_instance("status_toy", 6, 3, seed=20261014)


def test_optimal_run_is_marked_proved(toy):
    sch = BAPTimeIndexed(toy, time_limit=60.0).solve()
    assert sch.feasible, sch.violations
    assert sch.extra["status"] == "Optimal"
    assert sch.extra["proved_optimal"] is True


def test_limit_statuses_are_recognised_as_limits():
    """A limit means "unproven", not "infeasible"; the codes must stay in sync
    with the solver's enum."""
    import pulp

    enum = pulp.LpSolveStatus
    for name in ("TimeLimit", "MemoryLimit", "NodeLimit", "GapLimit",
                 "IterationLimit", "SolutionLimit", "Interrupted", "Stopped"):
        assert enum[name].value in LIMIT_STATUSES, name
    assert enum["Optimal"].value not in LIMIT_STATUSES
    assert enum["Infeasible"].value not in LIMIT_STATUSES


def test_a_limited_run_records_how_it_was_classified():
    """A 1 ms limit cannot prove optimality. Whatever the solver managed, the
    result must say which status it was and whether optimality was proved, so a
    caller can tell a limit from a proof -- and it must never be labelled optimal
    when it was not."""
    inst = make_hard_instance("status_limit", 10, 3, seed=20261014, slack=1.8)
    sch = BAPTimeIndexed(inst, time_limit=0.001).solve()
    assert "status" in sch.extra
    assert sch.extra["proved_optimal"] is bool(sch.extra["status"] == "Optimal")
    if not sch.extra["proved_optimal"]:
        assert sch.extra["status"] == "TimeLimit" or sch.extra["has_solution"] is False
    if sch.feasible:
        assert sch.objective is not None
        assert sch.check(inst).feasible, sch.check(inst).violations


def test_the_classical_models_record_their_solver_status():
    """They read the same solver as the BAP models, so a caller must be able to
    tell a time limit from a proof of infeasibility for them too."""
    from bapbench.models.classical import (
        FlowShopMILP, JobShopMILP, ParallelMachineMILP, make_simple_problem,
        to_instance_view,
    )
    prob = make_simple_problem("status_cls", 6, 3, seed=11, horizon=120)
    view = to_instance_view(prob)
    for model, kw in ((ParallelMachineMILP, {}),
                      (FlowShopMILP, {}),
                      (JobShopMILP, {"routes": {j: list(range(3)) for j in range(6)}})):
        sch = model(prob, instance_view=view, **kw).solve()
        assert "status" in sch.extra, model.__name__
        assert "proved_optimal" in sch.extra, model.__name__
        assert "has_solution" in sch.extra, model.__name__
        assert sch.extra["proved_optimal"] is bool(sch.extra["status"] == "Optimal")


def test_horizon_starved_instance_reports_infeasible_not_optimal():
    """slack=1.0 leaves a horizon too short for the crane limit; the model must
    say infeasible and must not claim optimality."""
    inst = make_hard_instance("status_infeasible", 10, 3, seed=20261014, slack=1.0)
    sch = BAPTimeIndexed(inst, time_limit=30.0).solve()
    assert not sch.feasible
    assert sch.extra["proved_optimal"] is False
    assert "Infeasible" in sch.extra["status"]


def test_disjunctive_model_reports_status_too(toy):
    from bapbench.models.bap import BAPDisjunctive

    sch = BAPDisjunctive(toy, time_limit=60.0).solve()
    assert sch.feasible, sch.violations
    assert "proved_optimal" in sch.extra


# --- the independent re-check -------------------------------------------------

def test_sanity_check_accepts_a_valid_schedule(toy):
    sch = BAPTimeIndexed(toy, time_limit=60.0).solve()
    assert sanity_check_schedule(toy, sch) == []


def test_sanity_check_accepts_a_plain_assignment(toy):
    """The checker must take either a Schedule or a bare mapping."""
    sch = BAPTimeIndexed(toy, time_limit=60.0).solve()
    assert sanity_check_schedule(toy, sch.assign) == []


def test_sanity_check_detects_a_late_start_shifted_early(toy):
    sch = BAPTimeIndexed(toy, time_limit=60.0).solve()
    broken = dict(sch.assign)
    vid = next(iter(broken))
    bid, st = broken[vid]
    broken[vid] = (bid, max(0, toy.vessels[vid].arrive - 1))
    kinds = {i.kind for i in sanity_check_schedule(toy, broken)}
    assert "release" in kinds


def test_sanity_check_detects_missing_coverage(toy):
    sch = BAPTimeIndexed(toy, time_limit=60.0).solve()
    short = dict(list(sch.assign.items())[:-1])
    kinds = {i.kind for i in sanity_check_schedule(toy, short)}
    assert "coverage" in kinds


def test_sanity_check_detects_a_berth_overlap(toy):
    """Force two vessels onto the same berth at the same time."""
    vessels = sorted(toy.vessels, key=lambda v: v.vid)
    assign = {vessels[0].vid: (0, vessels[0].arrive),
              vessels[1].vid: (0, vessels[1].arrive)}
    for v in vessels[2:]:
        assign[v.vid] = (1, v.arrive)
    kinds = {i.kind for i in sanity_check_schedule(toy, assign)}
    assert "berth_overlap" in kinds


def test_berth_overlaps_reports_the_offending_berth(toy):
    vessels = sorted(toy.vessels, key=lambda v: v.vid)
    a, b = vessels[0], vessels[1]
    start = max(a.arrive, b.arrive)
    assign = {a.vid: (0, start), b.vid: (0, start)}
    problems = berth_overlaps(toy, assign)
    assert problems


def test_bound_sanity_flags_a_value_below_the_bound():
    assert check_bound_sanity(10.0, 20.0, "x") is not None
    assert check_bound_sanity(20.0, 20.0, "x") is None
    assert check_bound_sanity(30.0, 20.0, "x") is None
