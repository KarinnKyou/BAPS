# -*- coding: utf-8 -*-
"""Compare the in-memory HiGHS model with the LP file pulp writes."""
import random
from pathlib import Path

import pulp

from bapbench.instances import Instance, Vessel, Berth
from bapbench.models.bap import BAPDisjunctive


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
d.problem.solve(pulp.HiGHS(msg=False))
print("pulp reported obj:", pulp.value(d.problem.objective))
mem = Path("D:/DSH_Workspace/01_Code/BAPS/_dj_mem.lp")
d.problem.solverModel.writeModel(str(mem))
print("wrote memory model to", mem)

# variable/constraint ids: are they unique?
vars_ = d.problem.exported_variables()
ids = [v.id for v in vars_]
print("exported vars:", len(vars_), "unique ids:", len(set(ids)))
cons = d.problem.constraints()
cids = [c.id for c in cons]
print("constraints:", len(cons), "unique ids:", len(set(cids)))
# does every constraint reference variables whose ids are in the exported set?
missing = 0
for c in cons:
    for v, _coef in c.items():
        if v.id not in set(ids):
            missing += 1
print("coefficient refs to vars not exported:", missing)
# duplicate variable names in the exported set
from collections import Counter
cnt = Counter(v.name for v in vars_)
print("duplicate exported names:", {k: n for k, n in cnt.items() if n > 1})
