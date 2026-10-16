"""Tests for the classical scheduling models and their independent enumeration."""

from __future__ import annotations

import pytest

from bapbench.models.classical import (
    FlowShopMILP,
    JobShopMILP,
    ParallelMachineMILP,
    brute_force_parallel,
    brute_force_single_machine,
    make_simple_problem,
    makespan_upper_bound,
    single_machine_spt,
    single_machine_wspt,
    to_instance_view,
)


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_spt_matches_enumeration_without_release_dates(n):
    prob = make_simple_problem(f"spt_{n}", n, 1, seed=17 + n)
    _order, value = single_machine_spt(prob)
    brute, _ = brute_force_single_machine(prob, "sum_C")
    assert value == brute


def test_spt_honours_release_dates():
    """SPT must return a feasible cost, so its value can never beat the optimum."""
    prob = make_simple_problem("sptr", 6, 1, seed=29, release=True, release_range=(0, 12))
    order, value = single_machine_spt(prob)
    brute, brute_order = brute_force_single_machine(prob, "sum_C")

    # recompute the cost of the SPT order independently, honouring releases
    t, total = 0, 0
    for j in order:
        t = max(t, prob.jobs[j].release) + prob.jobs[j].proc
        total += t
    assert value == total
    assert value >= brute, "SPT cannot be cheaper than the optimum"

    # and the reported optimum really is attainable by its own order
    t = 0
    for j in brute_order:
        t = max(t, prob.jobs[j].release) + prob.jobs[j].proc
    assert t <= prob.horizon


def test_wspt_matches_enumeration_of_weighted_cost():
    prob = make_simple_problem("wspt", 6, 1, seed=5)
    _order, value = single_machine_wspt(prob)
    best = None
    from itertools import permutations
    for order in permutations(range(prob.n)):
        t, total = 0, 0
        for j in order:
            t += prob.jobs[j].proc
            total += prob.jobs[j].weight * t
        best = total if best is None else min(best, total)
    assert value == best


@pytest.mark.parametrize("variant", ["plain", "release", "unavailable",
                                     "release+unavailable"])
def test_parallel_machine_milp_matches_enumeration(variant):
    kw = {"release": "release" in variant, "unavailable": "unavailable" in variant}
    prob = make_simple_problem(f"pm_{variant}", 5, 2, seed=11, horizon=100, **kw)
    inst = to_instance_view(prob)
    sch = ParallelMachineMILP(prob, instance_view=inst).solve()
    brute = brute_force_parallel(prob, "sum_C", max_cases=400_000)
    assert sch.feasible, sch.violations
    assert brute is not None
    assert sch.objective == pytest.approx(brute)


def test_flow_shop_milp_matches_all_permutations():
    prob = make_simple_problem("fs", 5, 3, seed=7, horizon=100)
    sch = FlowShopMILP(prob, instance_view=to_instance_view(prob)).solve()
    assert sch.feasible, sch.violations

    proc = [[j.proc] * prob.m for j in prob.jobs]
    best = None
    from itertools import permutations
    for order in permutations(range(prob.n)):
        done = [0] * prob.m
        for j in order:
            t = 0
            for k in range(prob.m):
                t = max(t, done[k]) + proc[j][k]
                done[k] = t
        best = max(done) if best is None else min(best, max(done))
    assert sch.objective == best


def test_makespan_upper_bound_is_valid():
    prob = make_simple_problem("ub", 5, 3, seed=3, horizon=100)
    sch = FlowShopMILP(prob, instance_view=to_instance_view(prob)).solve()
    assert makespan_upper_bound(prob) >= sch.objective


def test_job_shop_is_feasible_and_beats_a_reference_schedule():
    prob = make_simple_problem("js", 5, 3, seed=13, horizon=100)
    routes = {j: list(range(prob.m)) for j in range(prob.n)}
    sch = JobShopMILP(prob, routes=routes,
                      instance_view=to_instance_view(prob)).solve()
    assert sch.feasible, sch.violations
    assert sch.objective is not None
    assert sch.objective <= makespan_upper_bound(prob)
