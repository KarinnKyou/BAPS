# -*- coding: utf-8 -*-
"""Capture HiGHS log for the DJ model to see the claimed dual bound."""
import random

import pulp

from bapbench.instances import Instance, Vessel, Berth
from bapbench.models.bap import BAPDisjunctive

import highspy
print("highspy version:", highspy.Highs().version())


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

d = BAPDisjunctive(inst, objective="sum_C")
d.build()
s = d.problem.solve(pulp.HiGHS(msg=True))
print("obj:", pulp.value(d.problem.objective))
