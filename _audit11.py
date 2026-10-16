# -*- coding: utf-8 -*-
"""Is the wrong DJ optimum a model-build issue or a solver/presolve issue?

Write the model to MPS, reload it, and solve with highspy directly.
"""
import random
from pathlib import Path

import highspy
import pulp

from bapbench.instances import Instance, Vessel, Berth
from bapbench.models.bap import BAPDisjunctive

CONST = None


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
CONST = sum(v.handle for v in inst.vessels)
print("constant sum(handle) =", CONST)

d = BAPDisjunctive(inst, objective="sum_C")
d.build()
p = Path("D:/DSH_Workspace/01_Code/BAPS/_dj.lp")
d.problem.writeLP(str(p))
print("wrote", p)

# Fresh highspy read of the LP file
h = highspy.Highs()
h.readModel(str(p))
h.run()
info = h.getInfo()
print("highspy direct on LP file (presolve default): status", h.getModelStatus(),
      "obj", info.objective_function_value)
h2 = highspy.Highs()
h2.setOptionValue("presolve", "off")
h2.readModel(str(p))
h2.run()
print("highspy direct (presolve off):", h2.getModelStatus(),
      h2.getInfo().objective_function_value)

# pulp reload from MPS
mp = Path("D:/DSH_Workspace/01_Code/BAPS/_dj.mps")
d.problem.writeMPS(str(mp))
re = pulp.LpProblem.fromMPS(str(mp))[0] if isinstance(
    pulp.LpProblem.fromMPS(str(mp)), tuple) else pulp.LpProblem.fromMPS(str(mp))
st = re.solve(pulp.HiGHS(msg=False))
print("pulp reload from MPS: obj", pulp.value(re.objective))
