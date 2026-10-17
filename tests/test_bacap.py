"""Tests for the BACAP model layer and its ALNS.

The model, the crane pattern and the objective are checked against hand-computed
values and against the constraints they are supposed to satisfy. The search is
checked for feasibility with the independent checker, for an objective recomputed
from the plan, for determinism, and against a brute-force optimum on a tiny
instance.
"""

from __future__ import annotations

import itertools
import math

import pytest

from bapbench.reproduction import bacap


def _vessel(vid: int, eta: int, b0: int, *, length: int = 2, demand: int = 4,
            r_min: int = 1, r_max: int = 2, c1: float = 1.0, c2: float = 1.0,
            c3: float = 3.0, alpha: float = 0.9) -> bacap.BACAPVessel:
    d_min = math.ceil(demand / r_max ** alpha)
    return bacap.BACAPVessel(
        vid=vid, name=f"V{vid + 1:02d}", kind="small", length=length,
        eta=eta, est=math.floor(0.9 * eta), eft=eta + d_min,
        lft=eta + math.ceil(1.5 * d_min), demand=demand,
        r_min=r_min, r_max=r_max, b0=b0, c1=c1, c2=c2, c3=c3)


def _tiny() -> bacap.BACAPInstance:
    """Three short vessels on a small quay, with room for all of them."""
    return bacap.BACAPInstance(
        "tiny", [_vessel(0, 1, 1), _vessel(1, 3, 2), _vessel(2, 5, 3)],
        quay=6, cranes=4, horizon=10)


def _pair() -> bacap.BACAPInstance:
    """Two vessels, small enough to enumerate every candidate combination."""
    return bacap.BACAPInstance(
        "pair", [_vessel(0, 2, 1), _vessel(1, 4, 2)],
        quay=5, cranes=3, horizon=8)


def _overloaded() -> bacap.BACAPInstance:
    """Demand 21 QC-hours against 10 periods of two cranes: no plan exists."""
    return bacap.BACAPInstance(
        "overloaded",
        [_vessel(0, 1, 1, demand=5), _vessel(1, 2, 2, demand=6),
         _vessel(2, 3, 3, demand=6), _vessel(3, 4, 4, demand=4)],
        quay=8, cranes=2, horizon=10)


# ---------------------------------------------------------------------------
# Appendix A: the crane pattern
# ---------------------------------------------------------------------------

def test_qc_pattern_meets_the_workload_constraint():
    inst = _tiny()
    for v in inst.vessels:
        for berth in range(0, inst.quay - v.length + 1):
            for process in range(v.d_min(), inst.horizon):
                config = bacap.qc_pattern(inst, v, berth, process)
                if config is None:
                    continue
                assert config.f1 + config.f2 == process
                assert config.f1 >= 0 and config.f2 >= 0
                assert v.r_min <= config.q_hat <= v.r_max
                if config.f2:
                    assert config.q_hat + 1 <= v.r_max
                capacity = (config.f1 * config.q_hat ** inst.alpha
                            + config.f2 * (config.q_hat + 1) ** inst.alpha)
                assert capacity >= bacap.effective_demand(inst, v, berth) - 1e-9
                assert config.crane_hours == (config.f1 * config.q_hat
                                              + config.f2 * (config.q_hat + 1))


def test_f2_is_the_smallest_feasible_number_of_boosted_periods():
    """The printed loop keeps the last feasible ``p``; demoting one boosted period
    leaves the demand unmet, so no crane-hour is spent without need."""
    inst = _tiny()
    for v in inst.vessels:
        for berth in range(0, inst.quay - v.length + 1):
            for process in range(v.d_min(), inst.horizon):
                config = bacap.qc_pattern(inst, v, berth, process)
                if config is None or config.f2 == 0:
                    continue
                capacity = ((config.f1 + 1) * config.q_hat ** inst.alpha
                            + (config.f2 - 1) * (config.q_hat + 1) ** inst.alpha)
                assert capacity < bacap.effective_demand(inst, v, berth) - 1e-9


def test_qc_pattern_rejects_a_base_count_outside_the_range():
    """``q_hat`` is determined within the crane range, not clamped into it."""
    inst = bacap.BACAPInstance(
        "wide", [_vessel(0, 0, 0, length=1, demand=2, r_min=3, r_max=4)],
        quay=2, cranes=4, horizon=10)
    v = inst.vessels[0]
    # A long handling time needs fewer than r_min cranes, so there is no valid
    # assignment at this duration: it is rejected, not served with 3 cranes.
    assert bacap.qc_pattern(inst, v, 0, 9) is None


def test_min_process_is_the_shortest_for_its_base_count():
    inst = _tiny()
    for v in inst.vessels:
        for berth in range(0, inst.quay - v.length + 1):
            for q_hat in range(v.r_min, v.r_max + 1):
                process = bacap.min_process(inst, v, berth, q_hat)
                if process is None:
                    continue
                config = bacap.qc_pattern(inst, v, berth, process)
                assert config is not None and config.q_hat == q_hat
                shorter = bacap.qc_pattern(inst, v, berth, process - 1)
                assert shorter is None or shorter.q_hat != q_hat


# ---------------------------------------------------------------------------
# Objective (1)
# ---------------------------------------------------------------------------

def test_assignment_cost_matches_the_paper_objective():
    """``c1 dETA + c2 dEFT + c3 u + c4 (f1 q_hat + f2 (q_hat+1))``, by hand."""
    v = bacap.BACAPVessel(
        vid=0, name="V1", kind="small", length=2,
        eta=4, est=3, eft=4, lft=4, demand=6, r_min=2, r_max=2,
        b0=1, c1=2.0, c2=3.0, c3=5.0)
    inst = bacap.BACAPInstance("hand", [v], quay=4, cranes=4, horizon=10)
    a = bacap.Assignment(berth=1, start=2, config=bacap.QCConfig(q_hat=2, f1=2, f2=1))
    # dETA = 4 - 2 = 2; dEFT = 5 - 4 = 1; finish 5 > LFT 4 so c3 applies;
    # crane-hours = 2*2 + 1*3 = 7, at 0.1 per crane-hour.
    assert bacap.assignment_cost(inst, 0, a) == pytest.approx(
        2.0 * 2 + 3.0 * 1 + 5.0 + 0.1 * 7)


def test_solution_objective_is_the_sum_over_vessels():
    inst = _tiny()
    candidates = bacap.generate_candidates(inst, max_per_vessel=40)
    solution = bacap.BACAPSolution(inst)
    for vid in range(inst.n):
        for a in candidates[vid]:
            if solution.insert(vid, a):
                break
    assert len(solution.assign) == inst.n
    assert solution.objective() == pytest.approx(
        sum(solution.cost_of(vid) for vid in solution.assign))


# ---------------------------------------------------------------------------
# The plan and the independent checker
# ---------------------------------------------------------------------------

def _valid_solution() -> tuple[bacap.BACAPInstance, bacap.BACAPSolution]:
    inst = _tiny()
    candidates = bacap.generate_candidates(inst, max_per_vessel=40)
    solution = bacap.BACAPSolution(inst)
    for vid in range(inst.n):
        for a in candidates[vid]:
            if solution.insert(vid, a):
                break
    assert len(solution.assign) == inst.n
    return inst, solution


def test_checker_accepts_a_valid_plan():
    inst, solution = _valid_solution()
    assert bacap.check_solution(inst, solution) == []


def test_checker_reports_an_unassigned_vessel():
    inst, solution = _valid_solution()
    broken = solution.copy()
    del broken.assign[1]
    assert any("unassigned" in p for p in bacap.check_solution(inst, broken))


def test_checker_reports_a_quay_overlap():
    inst, solution = _valid_solution()
    broken = solution.copy()
    a = broken.assign[1]
    broken.assign[1] = bacap.Assignment(berth=0, start=a.start, config=a.config)
    assert any("overlap" in p for p in bacap.check_solution(inst, broken))


def test_checker_reports_a_crane_count_outside_the_range():
    inst, solution = _valid_solution()
    broken = solution.copy()
    period = next(iter(broken.cranes[0]))
    broken.cranes[0][period] = inst.vessels[0].r_max + 2
    assert any("outside" in p for p in bacap.check_solution(inst, broken))


def test_checker_reports_insufficient_effective_crane_hours():
    inst, solution = _valid_solution()
    broken = solution.copy()
    for t in list(broken.cranes[0]):
        broken.cranes[0][t] = inst.vessels[0].r_min
    assert any("effective crane-hours" in p for p in bacap.check_solution(inst, broken))


def test_checker_reports_a_start_before_est():
    inst, solution = _valid_solution()
    broken = solution.copy()
    a = broken.assign[0]
    broken.assign[0] = bacap.Assignment(berth=a.berth, start=inst.vessels[0].est - 1,
                                        config=a.config)
    assert any("before EST" in p for p in bacap.check_solution(inst, broken))


def test_checker_reports_a_vessel_off_the_quay():
    inst, solution = _valid_solution()
    broken = solution.copy()
    a = broken.assign[0]
    broken.assign[0] = bacap.Assignment(
        berth=inst.quay - inst.vessels[0].length + 1, start=a.start, config=a.config)
    assert any("leaves the quay" in p for p in bacap.check_solution(inst, broken))


def test_the_extra_crane_goes_where_there_is_room():
    inst = bacap.BACAPInstance(
        "room", [_vessel(0, 0, 0, length=1, demand=1, r_min=1, r_max=2),
                 _vessel(1, 0, 1, length=1, demand=1, r_min=1, r_max=2)],
        quay=3, cranes=3, horizon=6)
    solution = bacap.BACAPSolution(inst)
    first = bacap.Assignment(0, 0, bacap.QCConfig(q_hat=1, f1=1, f2=0))
    assert solution.insert(0, first)
    # One crane is taken in period 0; the second vessel's boosted period must be
    # the one with more room, i.e. period 1, not period 0.
    second = bacap.Assignment(1, 0, bacap.QCConfig(q_hat=1, f1=1, f2=1))
    assert solution.insert(1, second)
    assert solution.cranes[1] == {0: 1, 1: 2}


def test_a_candidate_is_refused_when_a_period_is_full():
    inst = bacap.BACAPInstance(
        "full", [_vessel(0, 0, 0, length=1, demand=1), _vessel(1, 0, 1, length=1,
                                                               demand=1)],
        quay=2, cranes=2, horizon=6)
    solution = bacap.BACAPSolution(inst)
    # Two cranes in both periods of the first vessel's stay.
    assert solution.insert(0, bacap.Assignment(0, 0, bacap.QCConfig(1, 0, 2)))
    assert not solution.insert(1, bacap.Assignment(1, 0, bacap.QCConfig(1, 1, 0)))


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

def test_generator_uses_the_published_ranges():
    inst = bacap.make_instance("gen", n_vessels=200, seed=7)
    seen = {"feeder": 0, "medium": 0, "jumbo": 0}
    for v in inst.vessels:
        spec = bacap.VESSEL_CLASSES[v.kind]
        seen[v.kind] += 1
        assert spec.length[0] <= v.length <= spec.length[1]
        assert spec.demand[0] <= v.demand <= spec.demand[1]
        assert (v.r_min, v.r_max) == spec.cranes
        assert (v.c1, v.c2, v.c3) == (spec.c1, spec.c2, spec.c3)
        assert 0 <= v.b0 <= inst.quay - v.length
        assert v.est == math.floor(0.9 * v.eta)
        assert v.eft == v.eta + v.d_min(inst.alpha)
        assert v.lft == v.eta + math.ceil(1.5 * v.d_min(inst.alpha))
        assert v.eta <= inst.horizon - v.d_min(inst.alpha)
    for kind, share in bacap.CLASS_MIX:
        assert seen[kind] / inst.n == pytest.approx(share, abs=0.08)


def test_generator_labels_itself_as_a_reconstruction():
    inst = bacap.make_instance("gen", n_vessels=10, seed=1)
    assert "reconstruction" in inst.meta["instance_kind"]
    assert "not a Meisel & Bierwirth" in inst.meta["instance_kind"]
    assert inst.meta["seed"] == 1


# ---------------------------------------------------------------------------
# The ALNS
# ---------------------------------------------------------------------------

def test_alns_returns_a_feasible_plan_and_recomputes_its_objective():
    inst = _tiny()
    result = bacap.run_alns(inst, seed=3, max_iterations=80)
    assert result.feasible and not result.unassigned
    assert bacap.check_solution(inst, result.solution) == []
    assert result.objective == pytest.approx(
        sum(result.solution.cost_of(vid) for vid in result.solution.assign))
    for vid, a in result.solution.assign.items():
        plan = result.solution.cranes[vid]
        assert set(plan) == set(range(a.start, a.finish))
        assert sum(1 for q in plan.values() if q == a.config.q_hat) == a.config.f1
        assert sum(1 for q in plan.values() if q == a.config.q_hat + 1) == a.config.f2


def test_alns_is_deterministic_for_one_seed():
    inst = _tiny()
    first = bacap.run_alns(inst, seed=11, max_iterations=40)
    second = bacap.run_alns(inst, seed=11, max_iterations=40)
    assert first.objective == second.objective
    assert first.solution.assign == second.solution.assign


def test_alns_matches_a_brute_force_optimum_on_a_tiny_instance():
    """Enumerate every candidate combination the search can use, in both insertion
    orders, and require the search to find it."""
    inst = _pair()
    candidates = bacap.generate_candidates(inst, max_per_vessel=200)
    best = math.inf
    for combo in itertools.product(*candidates):
        for order in itertools.permutations(range(inst.n)):
            solution = bacap.BACAPSolution(inst)
            if all(solution.insert(vid, combo[vid]) for vid in order):
                best = min(best, solution.objective())
    assert best < math.inf

    result = bacap.run_alns(inst, seed=2, max_iterations=150)
    assert result.feasible
    assert result.objective == pytest.approx(best)


def test_an_overloaded_instance_is_reported_as_infeasible():
    inst = _overloaded()
    result = bacap.run_alns(inst, seed=4, max_iterations=60)
    assert not result.feasible
    assert result.unassigned
    assert any("unassigned" in p for p in bacap.check_solution(inst, result.solution))


def test_a_generated_instance_is_solved_to_a_feasible_plan():
    """The generator's default classes must produce instances the search can plan:
    a generator whose output the model rejects would make every measurement on it
    meaningless."""
    inst = bacap.make_instance("generated_8", n_vessels=8, seed=20261014)
    result = bacap.run_alns(inst, seed=1, max_iterations=120)
    assert result.feasible and not result.unassigned
    assert bacap.check_solution(inst, result.solution) == []
