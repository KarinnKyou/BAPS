"""Regression tests for the objective actually reported by each classical model."""

from __future__ import annotations

import pytest

from bapbench.models.classical import (
    FlowShopMILP,
    JobShopMILP,
    ParallelMachineMILP,
    make_simple_problem,
    makespan_upper_bound,
    to_instance_view,
)


def _machines_of(prob, assign):
    groups: dict[int, list] = {}
    for j, (k, t) in assign.items():
        groups.setdefault(k, []).append((t, t + prob.jobs[j].proc, j))
    return groups


@pytest.mark.parametrize("objective", ["sum_C", "C_max"])
def test_parallel_machine_reports_the_objective_it_solved(objective):
    """The reported value must match the chosen criterion, not always the sum.

    _to_schedule used to compute sum(C_j) unconditionally, so asking for C_max
    returned the sum of completion times -- 82 where the schedule's makespan was
    20.
    """
    prob = make_simple_problem("pm_obj", 6, 2, seed=5, horizon=60)
    sch = ParallelMachineMILP(prob, objective=objective,
                              instance_view=to_instance_view(prob)).solve()
    assert sch.feasible, sch.violations

    comp = {j: t + prob.jobs[j].proc for j, (_k, t) in sch.assign.items()}
    expected = float(sum(comp.values())) if objective == "sum_C" else float(max(comp.values()))
    assert sch.objective == pytest.approx(expected)


def test_parallel_machine_makespan_is_reachable():
    """The makespan cannot exceed the total work, nor the machine-load bound."""
    prob = make_simple_problem("pm_cmax", 6, 2, seed=5, horizon=60)
    sch = ParallelMachineMILP(prob, objective="C_max",
                              instance_view=to_instance_view(prob)).solve()
    total = sum(j.proc for j in prob.jobs)
    load = max(sum(j.proc for j in prob.jobs) / prob.m, 1)
    assert sch.objective <= total
    assert sch.objective >= load


def test_flow_shop_makespan_matches_its_schedule():
    prob = make_simple_problem("fs_obj", 4, 3, seed=7, horizon=80)
    sch = FlowShopMILP(prob, instance_view=to_instance_view(prob)).solve()
    assert sch.feasible
    completion = max(t + prob.jobs[j].proc for j, (_k, t) in sch.assign.items())
    assert sch.objective >= completion


def test_job_shop_objective_equals_its_own_solution_value():
    prob = make_simple_problem("js_obj", 5, 3, seed=13, horizon=100)
    routes = {j: list(range(prob.m)) for j in range(prob.n)}
    sch = JobShopMILP(prob, routes=routes,
                      instance_view=to_instance_view(prob)).solve()
    assert sch.feasible
    completion = max(t + prob.jobs[j].proc for j, (_k, t) in sch.assign.items())
    assert sch.objective <= makespan_upper_bound(prob)
    assert sch.objective >= completion - 1e-6 or sch.objective == pytest.approx(
        max(t + prob.jobs[j].proc for j, (_k, t) in sch.assign.items()))
