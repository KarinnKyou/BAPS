# -*- coding: utf-8 -*-
"""Audit experiment 2: bigger handles (big-M stress), C_max, locks, failures."""
import random

from bapbench.instances import Instance, Vessel, Berth
from bapbench.models.bap import BAPTimeIndexed, BAPDisjunctive, safe_big_m
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


def run(tag, objective, seeds, **kw):
    bad = 0
    checked = 0
    skipped = 0
    for seed in seeds:
        inst = build(seed, **kw)
        try:
            ti = BAPTimeIndexed(inst, objective=objective).solve()
            dj = BAPDisjunctive(inst, objective=objective).solve()
        except RuntimeError:
            skipped += 1
            continue
        ex = exhaustive_optimum(inst, objective, max_cases=4_000_000,
                                time_budget=30.0)
        if ex.objective is None:
            skipped += 1
            continue
        checked += 1
        ti_ok = ti.feasible and abs(ti.objective - ex.objective) < 1e-6
        dj_ok = dj.feasible and abs(dj.objective - ex.objective) < 1e-6
        if not (ti_ok and dj_ok):
            bad += 1
            print(f"[{tag}] MISMATCH seed={seed} exh={ex.objective} "
                  f"ti={ti.objective}({ti.feasible},{ti.violations[:1]}) "
                  f"dj={dj.objective}({dj.feasible},{dj.violations[:1]})")
            print("   vessels:", [(v.arrive, v.handle) for v in inst.vessels],
                  "tw:", inst.tidal_windows, "crane:", inst.quay_crane_limit)
            if bad >= 5:
                break
    print(f"[{tag}] checked={checked} skipped={skipped} mismatches={bad}")


# 1) large handles relative to horizon -> stresses big-M in DJ
run("long_handles", "sum_C", range(300), n=4, m=2, horizon=22, hmin=6, hmax=14,
    maxarr=6)
# 2) C_max objective
run("cmax", "C_max", range(300), n=4, m=2, horizon=20, crane=2, tidal=True)
# 3) tighter: crane limit 1
run("crane1", "sum_C", range(300), n=4, m=3, horizon=20, crane=1, tidal=True)
# 4) more vessels
run("n6", "sum_C", range(150), n=6, m=2, horizon=20, crane=3, tidal=True,
    hmin=2, hmax=5)
# 5) no tidal, no crane, long handles
run("plain_long", "sum_C", range(200), n=5, m=2, horizon=24, hmin=8, hmax=16,
    maxarr=8)
