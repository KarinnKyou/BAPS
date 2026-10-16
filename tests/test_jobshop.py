"""Tests for the solver-free job-shop optimum."""

from __future__ import annotations

import pytest

from bapbench.models.classical import JobShopMILP, make_simple_problem, to_instance_view
from bapbench.verification.jobshop import job_shop_optimum


def _instance(n: int, m: int):
    prob = make_simple_problem(f"js_{n}x{m}", n, m, seed=51 + n, horizon=n * 12 + 40)
    routes = {j: list(range(m)) for j in range(n)}
    milp = JobShopMILP(prob, routes=routes,
                       instance_view=to_instance_view(prob)).solve()
    proc = [[job.proc] * m for job in prob.jobs]
    return prob, milp, proc


@pytest.mark.parametrize("n,m", [(4, 3), (5, 3), (4, 4)])
def test_milp_matches_the_independent_optimum(n, m):
    prob, milp, proc = _instance(n, m)
    result = job_shop_optimum([list(range(m)) for _ in range(n)], proc,
                              time_budget=60.0)
    assert result.objective is not None, result.note
    assert not result.note, "search should finish well inside the budget"
    assert milp.feasible, milp.violations
    assert milp.objective == pytest.approx(result.objective)


def test_optimum_is_at_least_the_machine_load_bound():
    n, m = 4, 3
    _prob, _milp, proc = _instance(n, m)
    result = job_shop_optimum([list(range(m)) for _ in range(n)], proc,
                              time_budget=60.0)
    # every job visits every machine exactly once, in order
    load = [sum(proc[j][mach] for j in range(n)) for mach in range(m)]
    assert result.objective >= max(load)


def test_a_two_job_two_machine_instance_is_solved_exactly():
    """Hand-checkable case: both jobs need 3 then 2 time units on machines 0 and 1."""
    routes = [[0, 1], [0, 1]]
    proc = [[3, 2], [3, 2]]
    result = job_shop_optimum(routes, proc, time_budget=10.0)
    # machine 0 is the bottleneck and cannot be shared, so C_max = 3 + 3 + 2 = 8
    assert result.objective == 8
