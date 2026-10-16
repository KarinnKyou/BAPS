# -*- coding: utf-8 -*-
"""Audit experiment: classical models against independent enumeration."""
import itertools
import random

from bapbench.models.classical import (
    FlowShopMILP, JobShopMILP, ParallelMachineMILP,
    brute_force_parallel, brute_force_single_machine,
    make_simple_problem, makespan_upper_bound, single_machine_spt,
    single_machine_wspt, to_instance_view,
)
from bapbench.verification.benchmarks import solve_shop
from bapbench.verification.jobshop import job_shop_optimum


def pfsp_makespan(proc, order):
    n, m = len(order), len(proc[0])
    done = [0] * m
    for j in order:
        t = 0
        for k in range(m):
            start = max(t, done[k])
            t = start + proc[j][k]
            done[k] = t
    return max(done)


print("=== 1. flow shop MILP vs permutation enumeration ===")
bad = 0
for seed in range(60):
    for (n, m) in ((3, 3), (4, 3), (5, 3), (5, 4)):
        P = make_simple_problem(f"fs{seed}_{n}x{m}", n, m, seed=seed, horizon=n * 12 + 40)
        sch = FlowShopMILP(P, instance_view=to_instance_view(P)).solve()
        proc = [[j.proc for _ in range(m)] for j in P.jobs]
        best = min(pfsp_makespan(proc, o) for o in itertools.permutations(range(n)))
        if not sch.feasible or abs(sch.objective - best) > 1e-6:
            bad += 1
            print(f"  MISMATCH seed={seed} {n}x{m}: milp={sch.objective} "
                  f"(feas={sch.feasible}, {sch.violations[:1]}) enum={best}")
print(f"  mismatches={bad}")

print("=== 2. parallel machine MILP vs brute force ===")
bad = 0
for seed in range(40):
    for (n, m, kw) in ((5, 2, {}), (5, 2, dict(release=True)),
                       (5, 2, dict(unavailable=True)),
                       (4, 3, dict(release=True, unavailable=True))):
        P = make_simple_problem(f"pm{seed}_{n}x{m}", n, m, seed=seed,
                                horizon=n * 12 + 40, **kw)
        sch = ParallelMachineMILP(P, instance_view=to_instance_view(P)).solve()
        bf = brute_force_parallel(P, "sum_C", max_cases=400_000)
        if bf is None or not sch.feasible or abs(sch.objective - bf) > 1e-6:
            bad += 1
            print(f"  MISMATCH seed={seed} {n}x{m} {kw}: milp={sch.objective} "
                  f"(feas={sch.feasible}, {sch.violations[:1]}) brute={bf}")
print(f"  mismatches={bad}")

print("=== 3. single machine SPT / WSPT ===")
bad = 0
for seed in range(40):
    P = make_simple_problem(f"sm{seed}", 7, 1, seed=seed, horizon=200)
    _, spt = single_machine_spt(P)
    bf, _ = brute_force_single_machine(P, "sum_C")
    if abs(spt - bf) > 1e-9:
        bad += 1
        print(f"  SPT MISMATCH seed={seed}: spt={spt} brute={bf}")
    P2 = make_simple_problem(f"smr{seed}", 7, 1, seed=seed, release=True,
                             release_range=(0, 12), horizon=200)
    # weighted spt has no brute force in the library for weights on release;
    # check it against the plain one when weights are all equal
    P3 = make_simple_problem(f"smw{seed}", 6, 1, seed=seed, horizon=200)
    for j in P3.jobs:
        j.weight = 1
    _, w = single_machine_wspt(P3)
    b3, _ = brute_force_single_machine(P3, "sum_C")
    if abs(w - b3) > 1e-9:
        bad += 1
        print(f"  WSPT(=SPT) MISMATCH seed={seed}: wspt={w} brute={b3}")
print(f"  mismatches={bad}")

print("=== 4. makespan_upper_bound: is it achievable/valid? ===")
bad = 0
for seed in range(30):
    for (n, m) in ((4, 3), (5, 4)):
        P = make_simple_problem(f"ub{seed}_{n}x{m}", n, m, seed=seed, horizon=n * 12 + 40)
        ub = makespan_upper_bound(P)
        proc = [[j.proc for _ in range(m)] for j in P.jobs]
        best = min(pfsp_makespan(proc, o) for o in itertools.permutations(range(n)))
        if ub < best:
            bad += 1
            print(f"  UB BELOW OPTIMUM seed={seed} {n}x{m}: ub={ub} opt={best}")
print(f"  invalid upper bounds={bad}")

print("=== 5. job shop: MILP vs branch-and-bound (non-delay?) ===")
bad = 0
for seed in range(30):
    for (n, m) in ((4, 3), (5, 3), (4, 4)):
        P = make_simple_problem(f"js{seed}_{n}x{m}", n, m, seed=seed,
                                horizon=n * 12 + 40)
        routes = {j: list(range(m)) for j in range(n)}
        sch = JobShopMILP(P, routes=routes, instance_view=to_instance_view(P)).solve()
        rts = [list(range(m)) for _ in range(n)]
        prs = [[P.jobs[j].proc] * m for j in range(n)]
        bb = job_shop_optimum(rts, prs, time_budget=20.0)
        shop = solve_shop(rts, prs, "cmp", time_limit=30.0)["solved"]
        if bb.objective is None or shop is None:
            continue
        if abs(bb.objective - shop) > 1e-6:
            bad += 1
            print(f"  B&B MISMATCH seed={seed} {n}x{m}: bb={bb.objective} "
                  f"milp_shop={shop} jobshop_milp={sch.objective} note={bb.note}")
print(f"  B&B mismatches={bad}")
