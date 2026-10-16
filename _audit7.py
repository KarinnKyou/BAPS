# -*- coding: utf-8 -*-
"""Careful: real DJ, fix schedule, keep objective, inspect variable values."""
import random

import pulp

from bapbench.instances import Instance, Vessel, Berth
from bapbench.models.bap import BAPTimeIndexed, BAPDisjunctive


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


inst = build(196, n=4, m=3, horizon=20, crane=1, tidal=True, hmax=None)
ti = BAPTimeIndexed(inst, objective="sum_C").solve()
opt = ti.assign
print("TI optimum", ti.objective, opt)

dj = BAPDisjunctive(inst, objective="sum_C")
dj.build()
print("cons before:", dj.problem.numConstraints())
added = 0
for i, (k, s_) in opt.items():
    dj.problem.addConstraint(dj.b[i] == s_); added += 1
    for kk in range(inst.m):
        dj.problem.addConstraint(dj.x[(i, kk)] == (1 if kk == k else 0)); added += 1
    for t in range(inst.vessels[i].arrive, inst.horizon + 1):
        dj.problem.addConstraint(dj.s[(i, t)] == (1 if t == s_ else 0)); added += 1
print("added:", added, "cons after:", dj.problem.numConstraints())
stats = dj.problem.solve(pulp.HiGHS(msg=False))
print("status", getattr(stats, "status", None),
      "obj", pulp.value(dj.problem.objective))
print("b values:", {i: dj.b[i].value() for i in range(inst.n)})
print("sum b+h =", sum(dj.b[i].value() + inst.vessels[i].handle for i in range(inst.n)))
