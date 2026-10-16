# -*- coding: utf-8 -*-
"""Check for variable-name collisions inside BAPDisjunctive (pulp 4 dedup?)."""
import random
from collections import Counter

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
dj = BAPDisjunctive(inst, objective="sum_C")
dj.build()

objs = []
for attr in ("x", "b", "z", "s"):
    d = getattr(dj, attr)
    objs.extend(list(d.values()))
print("python variable objects tracked:", len(objs))
print("model numVariables:", dj.problem.numVariables())

names = Counter(v.name for v in objs)
dups = {k: c for k, c in names.items() if c > 1}
print("duplicate names among tracked python vars:", dups)

allnames = Counter()
for v in dj.problem._model.list_variables():
    allnames[v.name] += 1
d2 = {k: c for k, c in allnames.items() if c > 1}
print("duplicate names in the model:", d2)
print("total model variables listed:", sum(allnames.values()))

# Are two distinct python variable objects the same column?
ids = Counter(id(v) for v in objs)
print("duplicate python objects:", {k: c for k, c in ids.items() if c > 1})

# key collision check for the x dict itself
print("len(x)", len(dj.x), "len(b)", len(dj.b), "len(z)", len(dj.z), "len(s)", len(dj.s))
print("expected s:", sum(inst.horizon + 1 - v.arrive for v in inst.vessels))
