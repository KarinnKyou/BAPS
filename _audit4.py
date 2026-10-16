# -*- coding: utf-8 -*-
"""Find which DJ constraint rejects the TI-optimal schedule (seed 196)."""
import random

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
print("vessels:", [(v.name, v.arrive, v.handle) for v in inst.vessels])
print("berths:", [(b.name, b.length) for b in inst.berths])
print("tw:", inst.tidal_windows)

ti = BAPTimeIndexed(inst, objective="sum_C").solve()
print("TI:", ti.assign, ti.objective, ti.feasible)

dj = BAPDisjunctive(inst, objective="sum_C")
dj.build()

# Set variable values to the TI-optimal schedule
xval = {}
bval = {}
sval = {}
for i, (k, st) in ti.assign.items():
    bval[i] = st
    sval[(i, st)] = 1
    xval[(i, k)] = 1
for (i, k), v in dj.x.items():
    v.varValue = xval.get((i, k), 0)
for i, v in dj.b.items():
    v.varValue = bval[i]
for (i, t), v in dj.s.items():
    v.varValue = sval.get((i, t), 0)
for (i, j), v in dj.z.items():
    ki, sti = ti.assign[i]
    kj, stj = ti.assign[j]
    if ki == kj:
        v.varValue = 1 if sti + inst.vessels[i].handle <= stj else 0
    else:
        v.varValue = 0

viol = []
for name, con in dj.problem.constraints.items():
    try:
        lhs = con.value()
    except Exception as e:
        viol.append((name, f"eval error {e}"))
        continue
    sense = con.sense
    rhs = -con.constant
    if sense == -1 and lhs > rhs + 1e-6:
        viol.append((name, f"LHS {lhs} > {rhs}"))
    elif sense == 1 and lhs < rhs - 1e-6:
        viol.append((name, f"LHS {lhs} < {rhs}"))
    elif sense == 0 and abs(lhs - rhs) > 1e-6:
        viol.append((name, f"LHS {lhs} != {rhs}"))

print(f"\nviolated constraints for the TI-optimal schedule: {len(viol)}")
for n_, d in viol:
    print("  ", n_, d)
