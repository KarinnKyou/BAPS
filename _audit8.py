# -*- coding: utf-8 -*-
"""Why does the base DJ model claim 44 when 42 is representable?"""
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
H = sum(v.handle for v in inst.vessels)

def fresh():
    d = BAPDisjunctive(inst, objective="sum_C")
    d.build()
    return d

d1 = fresh()
st = d1.problem.solve(pulp.HiGHS(msg=False))
print("base solve: status", getattr(st, "status", None),
      "obj", pulp.value(d1.problem.objective),
      "b", {i: d1.b[i].value() for i in range(inst.n)})

# add objective <= 43 and re-solve the same model
d1.problem.addConstraint(pulp.lpSum(d1.b[i] + inst.vessels[i].handle
                                    for i in range(inst.n)) <= 43)
st2 = d1.problem.solve(pulp.HiGHS(msg=False))
print("with obj<=43: status", getattr(st2, "status", None),
      "obj", pulp.value(d1.problem.objective),
      "b", {i: d1.b[i].value() for i in range(inst.n)})

# pure re-solve of a fresh model with tightened solver options
for opts in (dict(mip_rel_gap=0.0, mip_abs_gap=0.0),
             dict(presolve="off"),
             dict(mip_rel_gap=0.0, mip_abs_gap=0.0, presolve="off")):
    d = fresh()
    s = d.problem.solve(pulp.HiGHS(msg=False, **opts))
    print(f"opts={opts}: status {getattr(s,'status',None)} "
          f"obj {pulp.value(d.problem.objective)}")

# repeat solving the same fresh model several times
d = fresh()
for rep in range(3):
    s = d.problem.solve(pulp.HiGHS(msg=False))
    print(f"repeat {rep}: status {getattr(s,'status',None)} "
          f"obj {pulp.value(d.problem.objective)}")
