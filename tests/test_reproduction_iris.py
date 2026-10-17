"""Tests for the Iris, Pacino & Ropke (2017) ALNS reproduction.

The reproduction is of the paper's *method*: its operator set, adaptive weight
mechanism and acceptance criterion, run on this package's instances and
objective. The paper's own benchmark (Meisel & Bierwirth, 2009) is not publicly
available, so nothing here compares a value against the paper's tables; every
check is either an internal property of the mechanism or a comparison against an
optimum this package proves independently.
"""

from __future__ import annotations

import random

import pytest

from bapbench.heuristics.constructive import fcfs, lpt, spt
from bapbench.instances import make_hard_instance, make_toy_instance
from bapbench.metrics import berth_overlaps, sanity_check_schedule
from bapbench.models.bap import BAPTimeIndexed
from bapbench.reproduction.iris2017 import (
    ALNS_OPERATORS,
    ALNSConfig,
    ALNSSolution,
    DESTROY_OPERATORS,
    INSERTION_OPERATORS,
    INSERTION_P_MAX,
    INSERTION_P_MIN,
    _DestroyContext,
    alns as alns_run,
    destroy_cost_berth,
    destroy_cost_time,
    destroy_random,
    destroy_shaw,
    generate_candidates,
    make_meisel_bierwirth_instance,
    remove_vessels,
    repair_greedy,
    repair_smarter,
    run_alns,
)


def _short_config(iters: int = 400, **changes) -> ALNSConfig:
    """A config for tests: one cooling cycle, so no reheat lands inside the run."""
    changes.setdefault("reheat_period", max(1, iters))
    return ALNSConfig(max_iterations=iters, **changes)


@pytest.fixture(scope="module")
def toy():
    return make_toy_instance("iris_toy", 8, 3, seed=20261014)


@pytest.fixture(scope="module")
def constrained():
    """Tidal windows and a crane limit, so the operators meet real resources."""
    return make_hard_instance("iris_hard", 8, 3, seed=20261014, slack=2.0)


# --- operator set ------------------------------------------------------------

def test_the_four_named_destroy_operators_are_present():
    assert set(DESTROY_OPERATORS) == {"shaw", "cost_time", "cost_berth", "random"}
    assert DESTROY_OPERATORS["shaw"] is destroy_shaw
    assert DESTROY_OPERATORS["cost_time"] is destroy_cost_time
    assert DESTROY_OPERATORS["cost_berth"] is destroy_cost_berth
    assert DESTROY_OPERATORS["random"] is destroy_random


def test_both_insertion_operators_have_a_deterministic_and_stochastic_version():
    assert set(INSERTION_OPERATORS) == {"greedy", "greedy_stochastic",
                                        "smarter", "smarter_stochastic"}
    assert set(ALNS_OPERATORS["destroy"]) == set(DESTROY_OPERATORS)
    assert set(ALNS_OPERATORS["insertion"]) == set(INSERTION_OPERATORS)


@pytest.mark.parametrize("name", list(DESTROY_OPERATORS))
def test_each_destroy_operator_returns_a_usable_destroy_list(toy, name):
    """Every operator must select assignments that exist, within the budget."""
    candidates = generate_candidates(toy, "sum_C")
    baseline = ALNSSolution.empty(toy)
    order = list(range(toy.n))
    random.Random(0).shuffle(order)
    repair_greedy(baseline, candidates, order, random.Random(0), 1.0)
    assert baseline.complete

    upper = max(2, toy.n // 4)
    for seed in range(25):
        rng = random.Random(seed)
        ctx = _DestroyContext(toy, baseline, rng, _short_config())
        destroyed = DESTROY_OPERATORS[name](ctx)
        assert destroyed.operator == name
        taken = list(destroyed.removed)
        assert len(taken) == len(set(taken)), f"{name} repeated an assignment"
        assert 1 <= len(taken) <= upper
        assert all(vid in baseline.assign for vid in taken)
        # The rest of the plan must survive the removal untouched.
        partial = baseline.copy()
        remove_vessels(partial, taken)
        assert set(partial.assign) == set(baseline.assign) - set(taken)
        for vid in taken:
            k, st = baseline.assign[vid]
            assert not any(partial.occupancy[k][t]
                           for t in range(st, st + toy.vessels[vid].handle))
        assert partial.crane_load == [
            sum(1 for vid, (_k, st) in partial.assign.items()
                if st <= t < st + toy.vessels[vid].handle)
            for t in range(toy.horizon)]


@pytest.mark.parametrize("name", list(DESTROY_OPERATORS))
def test_removal_preserves_the_assignment_of_every_other_vessel(toy, name):
    baseline = ALNSSolution.from_assign(toy, dict(fcfs(toy, "sum_C").assign))
    rng = random.Random(7)
    destroyed = DESTROY_OPERATORS[name](_DestroyContext(toy, baseline, rng,
                                                        _short_config()))
    partial = baseline.copy()
    remove_vessels(partial, destroyed.removed)
    for vid, slot in partial.assign.items():
        assert slot == baseline.assign[vid], (
            f"{name} moved vessel {vid}, which it had not removed")


def test_destroy_budget_follows_the_paper(toy):
    """phi is drawn between phi_u/2 and phi_u, with phi_u = size / phi_r."""
    baseline = ALNSSolution.from_assign(toy, dict(fcfs(toy, "sum_C").assign))
    for phi_r in (2, 4, 6):
        cfg = _short_config(phi_r=phi_r)
        upper = max(2, toy.n // phi_r)
        seen = set()
        for seed in range(60):
            taken = destroy_random(_DestroyContext(toy, baseline,
                                                    random.Random(seed), cfg)).removed
            seen.add(len(taken))
            assert 2 <= len(taken) <= upper
        assert seen == set(range(max(2, upper // 2), upper + 1))


def test_shaw_removal_prefers_related_assignments(toy):
    """Shaw removal takes a tighter group of assignments than random removal.

    Relatedness is a distance, so a related group is a group whose berthing
    times are close together; the spread of the removed set is what separates
    the two operators.
    """
    baseline = ALNSSolution.from_assign(toy, dict(fcfs(toy, "sum_C").assign))
    cfg = _short_config(phi_r=4)

    def spreads(destroy, draws: int) -> list[int]:
        out = []
        for seed in range(draws):
            rng = random.Random(seed)
            taken = destroy(_DestroyContext(toy, baseline, rng, cfg)).removed
            starts = [baseline.assign[v][1] for v in taken]
            out.append((max(starts) - min(starts),
                        min((abs(a - b) for a in starts for b in starts if a != b),
                            default=0)))
        return out

    mean = lambda xs: sum(xs) / len(xs)          # noqa: E731
    for index, label in ((0, "diameter"), (1, "closest pair")):
        shaw = [s[index] for s in spreads(destroy_shaw, 200)]
        other = [s[index] for s in spreads(destroy_random, 200)]
        assert mean(shaw) * 2 < mean(other), (
            f"Shaw {label} {mean(shaw):.2f} is not much tighter than random "
            f"{mean(other):.2f}")


def test_shaw_relatedness_coefficients_drive_the_measure():
    """Each of a, b and c must move the relatedness ordering it is meant to."""
    from bapbench.instances import Berth, Instance, Vessel

    inst = Instance(
        name="iris_relatedness", horizon=40, dt=1.0,
        vessels=[Vessel(vid=0, name="V0", length=100, arrive=0, handle=4),
                 Vessel(vid=1, name="V1", length=100, arrive=0, handle=8),
                 Vessel(vid=2, name="V2", length=100, arrive=0, handle=4)],
        berths=[Berth(bid=0, name="B0", length=200),
                Berth(bid=1, name="B1", length=200),
                Berth(bid=2, name="B2", length=200)],
    )
    a, b, c = 0.01, 2.0, 0.01

    # Same berth, equal start, different handling: only |e_i - e_j| separates them.
    timings = ALNSSolution.from_assign(inst, {0: (0, 0), 1: (0, 0), 2: (0, 0)})
    ctx = _DestroyContext(inst, timings, random.Random(0), _short_config())
    assert ctx.relatedness(0, 1) == pytest.approx(c * abs(4 - 8))
    assert ctx.relatedness(0, 2) == pytest.approx(0.0)

    # Different berths and starts: all three terms contribute.
    apart = ALNSSolution.from_assign(inst, {0: (0, 0), 1: (1, 10), 2: (2, 10)})
    ctx = _DestroyContext(inst, apart, random.Random(0), _short_config())
    assert ctx.relatedness(0, 1) == pytest.approx(a * 1 + b * 10 + c * abs(4 - 18))
    assert ctx.relatedness(0, 2) == pytest.approx(a * 2 + b * 10 + c * abs(4 - 14))


def test_berth_relatedness_is_same_berth_at_different_times():
    """The relation is about the berth, not about the length of the interval."""
    inst = make_toy_instance("iris_berth_rel", 3, 2, seed=11)
    solution = ALNSSolution.from_assign(inst, {0: (0, 0), 1: (0, 20), 2: (1, 0)})
    ctx = _DestroyContext(inst, solution, random.Random(0), _short_config())
    assert ctx.berth_related(0, 1)
    assert not ctx.berth_related(0, 2)
    assert not ctx.berth_related(0, 0)


def test_time_relatedness_is_overlap_on_different_berths():
    inst = make_toy_instance("iris_time_rel", 3, 2, seed=11)
    solution = ALNSSolution.from_assign(inst, {0: (0, 0), 1: (1, 0), 2: (1, 30)})
    ctx = _DestroyContext(inst, solution, random.Random(0), _short_config())
    assert ctx.time_related(0, 1)
    assert not ctx.time_related(1, 2)


# --- insertion ---------------------------------------------------------------

def test_insertion_operators_produce_a_feasible_complete_solution(toy):
    """Destroy a subset with each operator and rebuild it with each insertion."""
    candidates = generate_candidates(toy, "sum_C")
    start = ALNSSolution.from_assign(toy, dict(fcfs(toy, "sum_C").assign))
    for destroy_name, destroy in DESTROY_OPERATORS.items():
        for insertion_name in INSERTION_OPERATORS:
            for seed in range(5):
                rng = random.Random(seed)
                destroyed = destroy(_DestroyContext(toy, start, rng, _short_config()))
                rebuilt = start.copy()
                remove_vessels(rebuilt, destroyed.removed)
                order = list(destroyed.removed)
                rng.shuffle(order)
                if insertion_name.startswith("smarter"):
                    repair_smarter(rebuilt, candidates, order, "sum_C")
                else:
                    p = 1.0 if insertion_name.endswith("greedy") else rng.uniform(0.25, 1.0)
                    repair_greedy(rebuilt, candidates, order, rng, p)
                assert rebuilt.complete, (destroy_name, insertion_name, seed)
                issues = sanity_check_schedule(toy, rebuilt.assign)
                assert issues == [], (destroy_name, insertion_name, issues)


def test_stochastic_insertion_uses_the_paper_parameter_range():
    assert INSERTION_P_MIN == 0.25 and INSERTION_P_MAX == 1.0


def test_run_uses_every_operator_at_least_once(toy):
    result = alns_run(toy, "sum_C", seed=5, config=_short_config(600))
    for name in DESTROY_OPERATORS:
        assert result.destroy_used[name] >= 1, f"destroy {name} never selected yet"
    for name in INSERTION_OPERATORS:
        assert result.insertion_used[name] >= 1, f"insertion {name} never selected"
    assert sum(result.destroy_used.values()) == 600
    assert sum(result.insertion_used.values()) == 600


# --- adaptive weights --------------------------------------------------------

def test_weights_start_equal_and_react_to_operator_performance(toy):
    """An 'adaptive' search whose weights never move is not adaptive."""
    result = alns_run(toy, "sum_C", seed=1, config=_short_config(1200))
    weights = {**result.destroy_weights, **result.insertion_weights}
    assert all(w >= 1.0 - 1e-12 for w in weights.values()), weights
    assert len(set(round(w, 9) for w in weights.values())) > 1, (
        "all operator weights are still equal after the run")
    assert result.destroy_scores and result.insertion_scores


def test_weights_are_updated_every_d_iterations_and_scores_reset(toy):
    """The paper resets the scores every d iterations; a run reports the count."""
    iters, interval = 250, 50
    result = alns_run(toy, "sum_C", seed=2, config=_short_config(
        iters, update_interval=interval))
    assert result.schedule.extra["weight_updates"] == iters // interval + 1
    assert sum(result.destroy_used.values()) == iters


def test_score_tiers_follow_the_paper(toy):
    default = ALNSConfig()
    assert (default.rho1, default.rho2, default.rho3) == (2.0, 5.0, 10.0)
    assert default.gamma == 0.8
    assert default.update_interval == 1000
    assert (default.shaw_a, default.shaw_b, default.shaw_c) == (0.01, 2.0, 0.01)
    assert default.phi_r == 4
    assert (default.start_pct, default.end_pct, default.reheat_period) == (
        0.05, 0.0005, 30000)


# --- acceptance and termination ----------------------------------------------

def test_temperature_cools_geometrically_from_start_to_end(toy):
    iters, reheat = 500, 500
    result = alns_run(toy, "sum_C", seed=4,
                      config=_short_config(iters, reheat_period=reheat))
    assert 0.0 < result.cooling_factor < 1.0
    # The cooling factor is derived from u and n, so one full cooling cycle
    # brings T_start exactly to T_end after e iterations.
    assert result.temperature_start * result.cooling_factor ** reheat == pytest.approx(
        result.temperature_end, rel=1e-9)
    assert result.temperature_start > result.temperature_end


def test_termination_is_the_iteration_count(toy):
    for iters in (0, 1, 37):
        result = alns_run(toy, "sum_C", seed=6, config=_short_config(iters))
        assert result.iterations == iters
        assert len(result.best_objective_trajectory) <= iters
        assert result.schedule.feasible


def test_best_objective_never_worsens_during_a_run(toy):
    """A monotone trajectory is what makes a longer run safe to trust."""
    result = alns_run(toy, "sum_C", seed=8, config=_short_config(3000))
    trajectory = result.best_objective_trajectory
    assert trajectory, "a run must report its best-objective trajectory"
    assert all(trajectory[i + 1] <= trajectory[i] + 1e-9
               for i in range(len(trajectory) - 1))
    assert result.objective == pytest.approx(trajectory[-1])


def test_reheat_resets_the_temperature_every_e_iterations(toy):
    """A wrapped temperature must keep the run exploring instead of freezing."""
    result = alns_run(toy, "sum_C", seed=9,
                      config=_short_config(600, reheat_period=100))
    assert result.cooling_factor < 1.0
    assert result.schedule.extra["params"]["reheat_period"] == 100


# --- determinism and monotonicity in the budget ------------------------------

def test_same_seed_reproduces_the_same_schedule(constrained):
    first = alns_run(constrained, "sum_C", seed=17, config=_short_config(800))
    second = alns_run(constrained, "sum_C", seed=17, config=_short_config(800))
    assert first.assign == second.assign
    assert first.objective == second.objective
    assert first.destroy_weights == second.destroy_weights


def test_different_seeds_are_allowed_to_differ(toy):
    values = {alns_run(toy, "sum_C", seed=s,
                       config=_short_config(300)).objective for s in range(6)}
    assert all(v >= 78.0 - 1e-9 for v in values)


def test_longer_run_is_never_worse_from_the_same_seed(toy):
    """The paper's mechanism is monotone, so extra iterations cannot hurt."""
    for seed in (1, 2, 3):
        short = alns_run(toy, "sum_C", seed=seed, config=_short_config(150))
        long = alns_run(toy, "sum_C", seed=seed, config=_short_config(4000))
        assert long.objective <= short.objective + 1e-9, (
            f"seed {seed}: {long.objective} after 4000 iterations is worse than "
            f"{short.objective} after 150")


# --- feasibility, objective and the proved optimum ---------------------------

def test_every_returned_schedule_passes_the_independent_sanity_check(constrained):
    result = alns_run(constrained, "sum_C", seed=23, config=_short_config(1500))
    assert result.schedule.feasible, result.schedule.violations
    assert sanity_check_schedule(constrained, result.schedule) == []
    assert berth_overlaps(constrained, result.schedule.assign) == []


def test_objective_equals_a_recomputation_from_the_assignment_alone(constrained):
    result = alns_run(constrained, "sum_C", seed=23, config=_short_config(1500))
    recomputed = float(sum(st + constrained.vessels[vid].handle
                           for vid, (_k, st) in result.assign.items()))
    assert recomputed == pytest.approx(result.schedule.objective)
    assert recomputed == pytest.approx(result.objective)


def test_objective_matches_the_model_convention_on_c_max(constrained):
    result = alns_run(constrained, "C_max", seed=3, config=_short_config(600))
    recomputed = float(max(st + constrained.vessels[vid].handle
                           for vid, (_k, st) in result.assign.items()))
    assert recomputed == pytest.approx(result.schedule.objective)
    assert sanity_check_schedule(constrained, result.schedule) == []


@pytest.mark.parametrize("n_vessels,n_berths", [(5, 3), (6, 3), (7, 3), (8, 3)])
def test_alns_is_never_worse_than_the_proved_optimum(n_vessels, n_berths):
    """A result below a proved optimum would mean an infeasible schedule."""
    inst = make_toy_instance(f"iris_opt_{n_vessels}x{n_berths}", n_vessels,
                             n_berths, seed=7)
    exact = BAPTimeIndexed(inst, objective="sum_C", time_limit=30.0).solve()
    assert exact.extra.get("proved_optimal"), "test needs a proved optimum"

    result = alns_run(inst, "sum_C", seed=3, config=_short_config(400))
    assert result.schedule.feasible, result.schedule.violations
    assert sanity_check_schedule(inst, result.schedule) == []
    assert result.objective >= exact.objective - 1e-9, (
        "the ALNS beat a proved optimum, so its schedule cannot be feasible")


@pytest.mark.parametrize("n_vessels,n_berths", [(5, 3), (6, 3), (7, 3), (8, 3)])
def test_alns_reaches_the_proved_optimum_on_small_instances(n_vessels, n_berths):
    """With a budget that lets it converge, the ALNS finds the exact optimum.

    The budget is part of the test on purpose: at a few hundred iterations the
    search can still be one schedule short, so reachability is stated at a budget
    where every seed on these instances has converged.
    """
    inst = make_toy_instance(f"iris_opt_{n_vessels}x{n_berths}", n_vessels,
                             n_berths, seed=7)
    exact = BAPTimeIndexed(inst, objective="sum_C", time_limit=30.0).solve()
    assert exact.extra.get("proved_optimal"), "test needs a proved optimum"

    for seed in (1, 3):
        result = alns_run(inst, "sum_C", seed=seed, config=_short_config(8000))
        assert result.schedule.feasible, result.schedule.violations
        assert sanity_check_schedule(inst, result.schedule) == []
        assert result.objective == pytest.approx(exact.objective), (
            f"seed {seed}: ALNS {result.objective} vs proved optimum "
            f"{exact.objective}")


@pytest.mark.parametrize("n_vessels,n_berths,slack,objective", [
    (8, 3, 2.0, "sum_C"),
    (9, 3, 1.8, "sum_C"),
    (8, 3, 2.0, "C_max"),
    (6, 2, 1.6, "sum_C"),
])
def test_alns_reaches_the_proved_optimum_with_crane_and_tidal_constraints(
        n_vessels, n_berths, slack, objective):
    """The binding constraints must not cost the search its optimal schedule."""
    inst = make_hard_instance(f"iris_opt_hard_{n_vessels}_{objective}", n_vessels,
                              n_berths, seed=20261014, slack=slack)
    assert inst.quay_crane_limit is not None and inst.tidal_windows
    exact = BAPTimeIndexed(inst, objective=objective, time_limit=60.0).solve()
    assert exact.extra.get("proved_optimal"), "test needs a proved optimum"

    for seed in (1, 3, 5):
        result = alns_run(inst, objective, seed=seed, config=_short_config(4000))
        assert result.schedule.feasible, result.schedule.violations
        assert sanity_check_schedule(inst, result.schedule) == []
        assert result.objective == pytest.approx(exact.objective), (
            f"seed {seed}: ALNS {result.objective} vs proved optimum "
            f"{exact.objective}")


def test_alns_is_at_least_as_good_as_the_constructive_rules(constrained):
    result = alns_run(constrained, "sum_C", seed=13, config=_short_config(1200))
    for rule in (fcfs, lpt, spt):
        sch = rule(constrained, "sum_C")
        assert result.objective <= sch.objective + 1e-9, rule.__name__


def test_run_alns_returns_the_same_schedule_as_the_full_api(toy):
    schedule = run_alns(toy, "sum_C", seed=21, config=_short_config(300))
    detailed = alns_run(toy, "sum_C", seed=21, config=_short_config(300))
    assert schedule.assign == detailed.assign
    assert schedule.objective == detailed.objective


# --- instance generation with the published characteristics ------------------

def test_generated_instance_has_the_published_characteristics():
    inst = make_meisel_bierwirth_instance("iris_gen20", n_vessels=20, seed=5)
    assert inst.n == 20
    assert sum(b.length for b in inst.berths) == 1000
    assert inst.horizon == 168
    assert inst.quay_crane_limit == 10
    assert inst.meta["quay_length_m"] == 1000
    assert inst.meta["section_m"] == 10
    assert inst.meta["interference_alpha"] == 0.9
    assert inst.meta["berthing_deviation_beta"] == 0.01
    assert len(inst.meta["desired_positions"]) == 20
    assert inst.validate() == []
    assert all(v.arrive + v.handle <= inst.horizon for v in inst.vessels)


@pytest.mark.parametrize("n_vessels", [20, 30, 40])
def test_generated_instances_are_feasible_and_every_vessel_fits(n_vessels):
    inst = make_meisel_bierwirth_instance(f"iris_gen{n_vessels}",
                                          n_vessels=n_vessels, seed=n_vessels)
    assert inst.n == n_vessels
    assert sanity_check_schedule(inst, {}) != []      # nothing is assigned yet
    for v in inst.vessels:
        assert any(v.length <= b.length for b in inst.berths), v.name


def test_generated_instances_can_be_searched(toy):
    """A real run on a generated instance, kept short for test runtime."""
    inst = make_meisel_bierwirth_instance("iris_gen_small", n_vessels=12,
                                          n_berths=4, seed=3)
    result = alns_run(inst, "sum_C", seed=1, config=_short_config(400))
    assert result.schedule.feasible, result.schedule.violations
    assert sanity_check_schedule(inst, result.schedule) == []
    assert result.objective == pytest.approx(
        float(sum(st + inst.vessels[vid].handle
                  for vid, (_k, st) in result.assign.items())))


def test_instance_generation_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        make_meisel_bierwirth_instance("bad", n_vessels=0)
    with pytest.raises(ValueError):
        make_meisel_bierwirth_instance("bad", n_vessels=5, quay_length=1005)
    with pytest.raises(ValueError):
        make_meisel_bierwirth_instance("bad", n_vessels=5, n_berths=0)
