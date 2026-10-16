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
    parallel_search_work,
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


# --- the enumeration budget -----------------------------------------------------

def test_search_work_counts_the_per_machine_orderings():
    """A schedule is an assignment plus an ordering per machine. The multinomial
    factor and the orderings cancel, leaving n! * C(n+m-1, m-1); counting only the
    m**n assignments understates the work by a factor that grows with n."""
    from math import comb, factorial

    for n, m in ((4, 2), (6, 2), (5, 3), (8, 3)):
        assert parallel_search_work(n, m, 100) == factorial(n) * comb(n + m - 1, m - 1)
        assert parallel_search_work(n, m, 100) > m ** n


def test_search_work_matches_the_schedules_actually_enumerated():
    """The estimate must be the real total, not merely an upper bound."""
    from itertools import permutations, product as iproduct
    prob = make_simple_problem("work", 4, 2, seed=5, horizon=50)
    n, m = prob.n, prob.m
    count = 0
    for combo in iproduct(range(m), repeat=n):
        groups = {k: [] for k in range(m)}
        for j in range(n):
            groups[combo[j]].append(j)
        permsets = [list(permutations(g)) for g in groups.values()]
        for _ in iproduct(*permsets):
            count += 1
    # No machine is unavailable here, so the blocking factor is 1.
    assert count == parallel_search_work(n, m, prob.horizon, 0.0)


def test_the_budget_guard_actually_bounds_the_work():
    """The guard has to refuse an enumeration larger than max_cases rather than
    letting it run: that is the only thing the argument promises."""
    prob = make_simple_problem("budget", 5, 2, seed=9, horizon=100)
    real = parallel_search_work(prob.n, prob.m, prob.horizon)

    assert brute_force_parallel(prob, "sum_C", max_cases=real - 1) is None
    assert brute_force_parallel(prob, "sum_C", max_cases=real) is not None


def test_search_work_never_understates_a_blocked_instance():
    """Unavailable windows cost extra period-by-period steps, so the estimate for
    a blocked instance may not fall below the unblocked one."""
    plain = make_simple_problem("p", 5, 2, seed=3, horizon=100)
    blocked = make_simple_problem("b", 5, 2, seed=3, horizon=100, unavailable=True)
    assert parallel_search_work(blocked.n, blocked.m, blocked.horizon, 0.3) > \
        parallel_search_work(plain.n, plain.m, plain.horizon, 0.0)
