"""Independent verification of the three published-method reproductions.

Each reproduction is checked against evidence it does not produce itself:

* Yalaoui 2006 -- the branch and bound must return the same value as a full
  enumeration over assignments and per-machine orders.
* Imai 2001 -- a lower bound may never exceed the true optimum, and a feasible
  heuristic value may never undercut it.  Ground truth is again enumeration.
* Iris 2017 -- the ALNS must reach an optimum the exact MILP has proved, and every
  schedule it returns must pass the independent feasibility check.

Run after any change to `bapbench/reproduction/`.  The enumeration bounds the
instance sizes, which is why the cases are small.
"""

from __future__ import annotations

import itertools
import random

from bapbench.instances import make_toy_instance
from bapbench.metrics import sanity_check_schedule
from bapbench.models.bap import BAPTimeIndexed
from bapbench.reproduction import imai2001 as I
from bapbench.reproduction import iris2017 as R
from bapbench.reproduction import yalaoui2006 as Y

YALAOU_I_CASES = ([(4, 2, s) for s in (1, 2, 3)]
                  + [(5, 2, s) for s in (1, 2, 3)]
                  + [(5, 3, s) for s in (1, 2, 3)]
                  + [(6, 2, s) for s in (1, 2, 3)]
                  + [(6, 3, s) for s in (1, 3)])

IMAI_CASES = ((4, 2), (5, 2))
IMAI_SEEDS = (1, 2)

IRIS_CASES = ((5, 3), (6, 3), (7, 3), (8, 3))
IRIS_SEEDS = (1, 3, 7)


def yalaoui_optimum(prob) -> float:
    """Every assignment of jobs to machines and every order on each machine."""
    n, m = prob.n, prob.m
    best = None
    for combo in itertools.product(range(m), repeat=n):
        groups = {k: [] for k in range(m)}
        for j in range(n):
            groups[combo[j]].append(j)
        for perms in itertools.product(*[list(itertools.permutations(g))
                                         for g in groups.values()]):
            total = 0
            for k, seq in zip(groups.keys(), perms):
                t = 0
                for j in seq:
                    job = prob.jobs[j]
                    t = max(t, job.release) + job.proc
                    total += t
            if best is None or total < best:
                best = total
    return best


def imai_optimum(inst: I.DBAPInstance) -> int:
    """Every partition of ships over berths and every service order within."""
    n, m = inst.n, inst.m
    best = None
    for combo in itertools.product(range(m), repeat=n):
        groups = [[] for _ in range(m)]
        for j in range(n):
            groups[combo[j]].append(j)
        for perms in itertools.product(*[list(itertools.permutations(g))
                                         for g in groups]):
            start = [[0] * len(p) for p in perms]
            for i, seq in enumerate(perms):
                t = inst.s[i]
                for k, j in enumerate(seq):
                    t = max(t, inst.a[j])
                    start[i][k] = t
                    t += inst.p[i][j]
            value = I.evaluate(inst, [list(reversed(p)) for p in perms], start)
            if best is None or value < best:
                best = value
    return best


def test_yalaoui_branch_and_bound_matches_enumeration():
    wrong = []
    for n, m, seed in YALAOU_I_CASES:
        prob = Y.generate_instance(n, m, lam=0.5, seed=seed)
        truth = yalaoui_optimum(prob)
        got = Y.solve_parallel_bnb(prob).objective
        if got != truth:
            wrong.append(f"n={n} m={m} seed={seed}: bnb={got} enumeration={truth}")
    assert not wrong, "branch and bound is not exact:\n" + "\n".join(wrong)


def test_yalaoui_bounds_never_exceed_the_optimum():
    for n, m, seed in YALAOU_I_CASES[:6]:
        prob = Y.generate_instance(n, m, lam=0.5, seed=seed)
        truth = yalaoui_optimum(prob)
        root = Y.empty_schedule(prob)
        for bound in (Y.lb_spt(root), Y.lb_srpt(root), Y.lb_srpt_spt(root)):
            assert bound <= truth + 1e-9, f"n={n} m={m} seed={seed}: {bound} > {truth}"


def test_imai_lower_bound_never_exceeds_the_optimum():
    wrong = []
    for n, m in IMAI_CASES:
        for seed in IMAI_SEEDS:
            rng = random.Random(seed)
            p = [[rng.randint(3, 12) for _ in range(n)] for _ in range(m)]
            a = [rng.randint(0, 12) for _ in range(n)]
            inst = I.DBAPInstance(p=p, a=a, s=[0] * m, name=f"v{n}x{m}s{seed}")
            truth = imai_optimum(inst)
            bound = min(r.best_lower_bound for r in I.solve_dbap(inst).values())
            if bound > truth + 1e-9:
                wrong.append(f"n={n} m={m} seed={seed}: bound={bound} optimum={truth}")
    assert not wrong, "a reported bound exceeds the optimum:\n" + "\n".join(wrong)


def test_imai_lower_bound_respects_a_hand_built_feasible_schedule():
    """A bound may not exceed any feasible value, and this check uses the module's
    own evaluator, so it does not depend on an enumeration being right."""
    inst = I.DBAPInstance(
        p=[[5, 12, 4, 7, 4], [10, 10, 10, 9, 6]],
        a=[1, 7, 0, 6, 6], s=[0, 0], name="hand_built")
    perms = [[2, 0, 4, 3], [1]]
    start = [[0] * len(seq) for seq in perms]
    for i, seq in enumerate(perms):
        t = inst.s[i]
        for k, j in enumerate(seq):
            t = max(t, inst.a[j])
            start[i][k] = t
            t += inst.p[i][j]
    value = I.evaluate(inst, [list(reversed(s)) for s in perms], start)

    bound = min(r.best_lower_bound for r in I.solve_dbap(inst).values())
    assert bound <= value, (
        f"bound {bound} exceeds a feasible value {value}, so it is not a bound")


def test_imai_heuristic_never_beats_the_optimum():
    for n, m in IMAI_CASES:
        for seed in IMAI_SEEDS:
            rng = random.Random(seed)
            p = [[rng.randint(3, 12) for _ in range(n)] for _ in range(m)]
            a = [rng.randint(0, 12) for _ in range(n)]
            inst = I.DBAPInstance(p=p, a=a, s=[0] * m, name=f"v{n}x{m}s{seed}")
            truth = imai_optimum(inst)
            for result in I.solve_dbap(inst).values():
                assert result.best_feasible.objective >= truth - 1e-9


def test_iris_alns_reaches_the_proved_optimum():
    wrong = []
    for n, m in IRIS_CASES:
        inst = make_toy_instance(f"repro_{n}x{m}", n, m, seed=7)
        exact = BAPTimeIndexed(inst, objective="sum_C", time_limit=30.0).solve()
        assert exact.extra.get("proved_optimal"), "the check needs a proved optimum"
        for seed in IRIS_SEEDS:
            sch = R.run_alns(inst, "sum_C", seed=seed,
                             config=R.ALNSConfig(max_iterations=4000))
            if sch.objective != exact.objective:
                wrong.append(f"n={n} m={m} seed={seed}: alns={sch.objective} "
                             f"optimum={exact.objective}")
    assert not wrong, "the ALNS did not reach the optimum:\n" + "\n".join(wrong)


def test_iris_every_schedule_passes_the_independent_check():
    for n, m in IRIS_CASES[:2]:
        inst = make_toy_instance(f"repro_sanity_{n}x{m}", n, m, seed=7)
        for seed in IRIS_SEEDS:
            sch = R.run_alns(inst, "sum_C", seed=seed,
                             config=R.ALNSConfig(max_iterations=600))
            assert sch.feasible, sch.violations
            assert sanity_check_schedule(inst, sch) == []
