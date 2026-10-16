# -*- coding: utf-8 -*-
"""Test the REAL BAPDisjunctive model with the TI-optimal schedule fixed."""
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
print("real DJ: vars", dj.problem.numVariables(), "cons", dj.problem.numConstraints())

st = dj.problem.solve(pulp.HiGHS(msg=False))
print("real DJ optimum:", pulp.value(dj.problem.objective),
      "status", getattr(dj.status if hasattr(dj, 'status') else st, 'status', None))

# fix variables to the TI optimum
for i, (k, s_) in opt.items():
    dj.problem.addConstraint(dj.b[i] == s_)
    for kk in range(inst.m):
        dj.problem.addConstraint(dj.x[(i, kk)] == (1 if kk == k else 0))
    for t in range(inst.vessels[i].arrive, inst.horizon + 1):
        dj.problem.addConstraint(dj.s[(i, t)] == (1 if t == s_ else 0))
dj.problem.setObjective(pulp.lpSum(0 * dj.b[i] for i in range(inst.n)))
stats2 = dj.problem.solve(pulp.HiGHS(msg=False))
print("fixed-schedule feasibility:", getattr(stats2, "status", None))

# print all constraints and their values at the fixed point
print("-- constraint residuals at the fixed point --")
bad = []
for v in dj.problem._model.list_constraints():
    c = pulp.LpConstraint(v)
    try:
        lhs = c.value()
    except Exception as e:
        continue
    name = getattr(c, "name", "?")
    sense = getattr(c, "sense", None)
    const = c.constant
    bad.append((name, sense, lhs, const))
print("total constraints enumerated:", len(bad))
