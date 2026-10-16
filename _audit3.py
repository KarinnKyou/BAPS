# -*- coding: utf-8 -*-
"""Dig into the DJ mismatch at crane1 seed=196."""
import random
import pulp

from bapbench.instances import Instance, Vessel, Berth, Schedule
from bapbench.models.bap import BAPTimeIndexed, BAPDisjunctive
from bapbench.verification.exhaustive import exhaustive_optimum


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


inst = build(196, n=4, m=3, horizon=20, crane=1, tidal=True, hmin=2, hmax=None)
print(inst.dumps())

ti = BAPTimeIndexed(inst, objective="sum_C")
ti.build()
ti.solve()
ti_sch = BAPTimeIndexed(inst, objective="sum_C")
ti_sch.build()
s = ti_sch.solve()
print("TI assign:", s.assign, "obj", s.objective, "feas", s.feasible)
ex = exhaustive_optimum(inst, "sum_C", max_cases=4_000_000, time_budget=60)
print("EXH", ex.objective, ex.assign)

dj = BAPDisjunctive(inst, objective="sum_C")
dj.build()
st = dj.problem.solve(pulp.HiGHS(msg=False))
print("DJ model status", getattr(st, "status", None),
      "model objective", pulp.value(dj.problem.objective))
djs = dj.solve()
print("DJ assign:", djs.assign, "obj", djs.objective, "feas", djs.feasible)

# Is the TI-optimal schedule representable in the DJ model?
print("\ncheck TI schedule feasibility:", Schedule(assign=dict(s.assign)).check(inst).violations)
print("TI assign:", s.assign)
