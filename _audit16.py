# -*- coding: utf-8 -*-
"""Audit experiment: crane failures, locks, sanity_check, vids, status flags."""
import random

from bapbench.instances import Instance, Vessel, Berth, Schedule, make_toy_instance, make_hard_instance
from bapbench.metrics import sanity_check_schedule
from bapbench.models.bap import BAPTimeIndexed
from bapbench.verification.exhaustive import exhaustive_optimum


def rnd(seed, n, m, horizon, crane=None, tidal=False, hmin=2, hmax=None,
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


print("=== 1. TI vs exhaustive with crane_failures ===")
bad = checked = 0
for seed in range(200):
    inst = rnd(seed, 4, 2, 20, crane=3, tidal=bool(seed % 2))
    inst.crane_failures = [(5 + seed % 3, 9 + seed % 3, 2)]
    try:
        ti = BAPTimeIndexed(inst, objective="sum_C").solve()
    except RuntimeError:
        continue
    ex = exhaustive_optimum(inst, "sum_C", max_cases=3_000_000, time_budget=20)
    if ex.objective is None:
        continue
    checked += 1
    if not ti.feasible or abs(ti.objective - ex.objective) > 1e-6:
        bad += 1
        print(f"  MISMATCH seed={seed} exh={ex.objective} ti={ti.objective} "
              f"feas={ti.feasible} {ti.violations[:1]}")
print(f"  checked={checked} mismatches={bad}")

print("=== 2. sanity_check_schedule vs Schedule.check under a crane outage ===")
inst = make_toy_instance("crane_out", 4, 2, seed=3, horizon=30)
inst.quay_crane_limit = 2
inst.crane_failures = [(4, 20, 1)]          # capacity 1 during [4,20)
ti = BAPTimeIndexed(inst, objective="sum_C").solve()
print("  TI assign:", ti.assign, "obj", ti.objective)
print("  TI work in the outage window:",
      {v: (st, st + inst.vessels[v].handle) for v, (_b, st) in ti.assign.items()
       if any(4 <= t < 20 for t in range(st, st + inst.vessels[v].handle))})
print("  Schedule.check feasible:", ti.feasible, ti.violations)
issues = sanity_check_schedule(inst, ti)
print("  sanity_check_schedule issues:", [(i.kind, i.detail) for i in issues])
print("  THESE DISAGREE" if (ti.feasible and issues) else "  agree")

print("=== 3. vid != index ===")
inst3 = Instance(name="vidgap", vessels=[Vessel(vid=10, name="A", length=100, arrive=0, handle=3),
                                         Vessel(vid=11, name="B", length=100, arrive=0, handle=3)],
                 berths=[Berth(bid=0, name="B0", length=200),
                         Berth(bid=1, name="B1", length=200)], horizon=20)
try:
    s3 = BAPTimeIndexed(inst3).solve()
    print("  solve assign:", s3.assign, "objective:", s3.objective)
    print("  violations:", s3.violations)
except Exception as e:
    print("  EXCEPTION:", type(e).__name__, e)
try:
    print("  sanity_check:", sanity_check_schedule(inst3, s3.assign))
except Exception as e:
    print("  sanity_check EXCEPTION:", type(e).__name__, e)

print("=== 4. full_berth=False on a tight instance ===")
tight = make_toy_instance("tightfb", 8, 2, seed=5, tight=True)
print("  berth length", tight.berths[0].length,
      "vessel lengths", sorted(v.length for v in tight.vessels))
sch_full = BAPTimeIndexed(tight, objective="sum_C", full_berth=True).solve()
sch_relax = BAPTimeIndexed(tight, objective="sum_C", full_berth=False).solve()
print("  full_berth=True :", sch_full.objective, "feasible", sch_full.feasible)
print("  full_berth=False:", sch_relax.objective, "feasible", sch_relax.feasible,
      sch_relax.violations[:1])
print("  relaxing changed the objective:",
      sch_full.objective != sch_relax.objective)

print("=== 5. time-limited status / proved_optimal ===")
hard = make_hard_instance("status", 10, 3, seed=20261014, slack=1.3)
long_run = BAPTimeIndexed(hard, objective="sum_C", time_limit=120.0).solve()
print("  unlimited:", long_run.objective, long_run.extra.get("status"),
      "proved", long_run.extra.get("proved_optimal"), "feasible", long_run.feasible)
short = BAPTimeIndexed(hard, objective="sum_C", time_limit=0.05).solve()
print("  0.05s    :", short.objective, short.extra.get("status"),
      "proved", short.extra.get("proved_optimal"), "feasible", short.feasible)
short2 = BAPTimeIndexed(hard, objective="sum_C", time_limit=0.2).solve()
print("  0.2s     :", short2.objective, short2.extra.get("status"),
      "proved", short2.extra.get("proved_optimal"), "feasible", short2.feasible)
