# -*- coding: utf-8 -*-
"""Isolate which DJ constraint group rejects the TI-optimal schedule (seed 196).

Replicates BAPDisjunctive.build verbatim, but each constraint group can be
switched off, then fixes the variables to the TI-optimal schedule and tests
feasibility.
"""
import random

import pulp

from bapbench.instances import Instance, Vessel, Berth
from bapbench.models.bap import BAPTimeIndexed, safe_big_m


def build(seed, n, m, horizon, crane=None, tidal=False, hmin=2, hmax=None,
          maxarr=4, block=3, period=11):
    rng = random.Random(seed)
    hmax = hmax or max(2, horizon // 2)
    vessels = [Vessel(vid=i, name=f"V{i}", length=100,
                      arrive=rng.randint(0, maxarr),
                      handle=rng.randint(hmin, hmax)) for i in range(n)]
    berths = [Berth(bid=k, name=f"B{k}", length=200) for k in range(m)]
    tw = {}
    if tidal:
        for k in range(m):
            tw[str(k)] = [(a, min(a + block, horizon))
                          for a in range(k * 4, horizon, period)]
    return Instance(name=f"rnd{seed}", vessels=vessels, berths=berths,
                    horizon=horizon, tidal_windows=tw, quay_crane_limit=crane)


def build_dj(inst, disable=()):
    n, m, T = inst.n, inst.m, inst.horizon
    M = safe_big_m(inst)
    prob = pulp.LpProblem("DJ_copy", pulp.LpMinimize)
    x = {(i, k): prob.add_variable(f"x_{i}_{k}", cat="Binary")
         for i in range(n) for k in range(m)}
    b = {i: prob.add_variable(f"b_{i}", lowBound=inst.vessels[i].arrive,
                              upBound=inst.horizon, cat="Integer")
         for i in range(n)}
    z = {(i, j): prob.add_variable(f"z_{i}_{j}", cat="Binary")
         for i in range(n) for j in range(i + 1, n)}
    starts = {(i, k): [t for t in range(inst.vessels[i].arrive, T - inst.vessels[i].handle + 1)
                       if not any(inst.blocked(k, tt)
                                  for tt in range(t, t + inst.vessels[i].handle))]
              for i in range(n) for k in range(m)}
    s = {(i, t): prob.add_variable(f"s_{i}_{t}", cat="Binary")
         for i in range(n) for t in range(inst.vessels[i].arrive, T + 1)}
    prob.setObjective(pulp.lpSum(b[i] + inst.vessels[i].handle for i in range(n)))

    if "assign" not in disable:
        for i in range(n):
            prob.addConstraint(pulp.lpSum(x[(i, k)] for k in range(m)) == 1)
    if "closed" not in disable:
        for i in range(n):
            prob.addConstraint(pulp.lpSum(s[(i, t)] for t in range(inst.vessels[i].arrive,
                                                                   T + 1)) == 1)
            prob.addConstraint(b[i] == pulp.lpSum(t * s[(i, t)]
                                                  for t in range(inst.vessels[i].arrive, T + 1)))
            for k in range(m):
                for t in range(inst.vessels[i].arrive, T + 1):
                    if t not in starts[(i, k)]:
                        prob.addConstraint(s[(i, t)] + x[(i, k)] <= 1)
    if "seq" not in disable:
        for i in range(n):
            for j in range(i + 1, n):
                pi_, pj_ = inst.vessels[i].handle, inst.vessels[j].handle
                for k in range(m):
                    both = prob.add_variable(f"both_{i}_{j}_{k}", cat="Binary")
                    prob.addConstraint(both <= x[(i, k)])
                    prob.addConstraint(both <= x[(j, k)])
                    prob.addConstraint(both >= x[(i, k)] + x[(j, k)] - 1)
                    prob.addConstraint(b[i] + pi_ <= b[j] + M * (1 - z[(i, j)]) + M * (1 - both))
                    prob.addConstraint(b[j] + pj_ <= b[i] + M * z[(i, j)] + M * (1 - both))
    if "crane" not in disable and inst.quay_crane_limit is not None:
        for t in range(T):
            occupancy = []
            for i in range(n):
                span = range(max(inst.vessels[i].arrive, t - inst.vessels[i].handle + 1),
                             t + 1)
                for k in range(m):
                    if not any(tt in starts[(i, k)] for tt in span):
                        continue
                    for tp in span:
                        if tp not in starts[(i, k)]:
                            continue
                        occ = prob.add_variable(f"occ_{i}_{k}_{t}_{tp}", cat="Binary")
                        prob.addConstraint(occ <= x[(i, k)])
                        prob.addConstraint(occ <= s[(i, tp)])
                        prob.addConstraint(occ >= x[(i, k)] + s[(i, tp)] - 1)
                        occupancy.append(occ)
            cap = inst.crane_capacity(t)
            if occupancy and cap is not None:
                prob.addConstraint(pulp.lpSum(occupancy) <= cap)
    return prob, x, b, s


inst = build(196, n=4, m=3, horizon=20, crane=1, tidal=True, hmax=None)
ti = BAPTimeIndexed(inst, objective="sum_C").solve()
opt = ti.assign
print("TI optimum", ti.objective, opt)

for disable in [(), ("crane",), ("seq",), ("closed",), ("assign",),
                ("crane", "seq")]:
    prob, x, b, s = build_dj(inst, disable)
    for i, (k, st) in opt.items():
        prob.addConstraint(b[i] == st)
        for kk in range(inst.m):
            prob.addConstraint(x[(i, kk)] == (1 if kk == k else 0))
        for t in range(inst.vessels[i].arrive, inst.horizon + 1):
            prob.addConstraint(s[(i, t)] == (1 if t == st else 0))
    prob.setObjective(pulp.lpSum(0 * b[i] for i in range(inst.n)))
    stats = prob.solve(pulp.HiGHS(msg=False))
    print(f"disable={disable or '(none)'} -> status {getattr(stats,'status',None)} "
          f"({'FEASIBLE' if getattr(stats,'status',None)==1 else 'infeasible'})")
