# -*- coding: utf-8 -*-
"""Does a time-limited (or gap-limited) solve get classified as a limit?"""
from bapbench.instances import make_hard_instance
from bapbench.models.bap import BAPTimeIndexed, BAPDisjunctive

for slack in (1.8, 1.6):
    inst = make_hard_instance(f"t{slack}", 10, 3, seed=20261014, slack=slack)
    ref = BAPTimeIndexed(inst, objective="sum_C", time_limit=300.0).solve()
    print(f"--- slack={slack} horizon={inst.horizon} reference: "
          f"obj={ref.objective} status={ref.extra.get('status')} "
          f"proved={ref.extra.get('proved_optimal')} feasible={ref.feasible}")
    for tl in (0.01, 0.05, 0.3, 1.0):
        s = BAPTimeIndexed(inst, objective="sum_C", time_limit=tl).solve()
        flag = ""
        if s.extra.get("proved_optimal") and s.objective is not None and \
                ref.objective is not None and s.objective > ref.objective + 1e-6:
            flag = "  <<< CLAIMS PROVED OPTIMAL BUT IS WORSE THAN THE TRUE OPTIMUM"
        print(f"    tl={tl:<5} obj={s.objective} status={s.extra.get('status')} "
              f"proved={s.extra.get('proved_optimal')} feasible={s.feasible} "
              f"viol={s.violations[:1]}{flag}")
    for tl in (0.01, 0.3):
        s = BAPDisjunctive(inst, objective="sum_C", time_limit=tl).solve()
        print(f"    DJ tl={tl:<5} obj={s.objective} status={s.extra.get('status')} "
              f"proved={s.extra.get('proved_optimal')} feasible={s.feasible} "
              f"viol={s.violations[:1]}")
